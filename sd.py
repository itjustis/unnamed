from PIL import Image
import numpy as np
import torch , os, gc, torch
from diffusers.utils import load_image
from transformers import ( pipeline, CLIPTokenizer, CLIPTextModel )
from diffusers import (
    DiffusionPipeline,
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionControlNetPipeline,
    AutoencoderKL,
    UNet2DConditionModel,
    ControlNetModel,
    DDPMScheduler,
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    UniPCMultistepScheduler,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_controlnet import MultiControlNetModel
from stable_diffusion_controlnet_img2img import StableDiffusionControlNetImg2ImgPipeline
from huggingface_hub import snapshot_download
from clip_interrogator import Config, Interrogator
import tomesd

cnet_dict = {
    'content': 'lllyasviel/control_v11f1e_sd15_tile',
    'depth': 'lllyasviel/control_v11f1p_sd15_depth',
    'scribble': 'lllyasviel/control_v11p_sd15_scribble',
    'canny_edge': 'lllyasviel/control_v11p_sd15_canny',
    'soft_edge': 'lllyasviel/control_v11p_sd15_softedge',
    'shuffle': 'lllyasviel/control_v11e_sd15_shuffle',
    'openpose': 'lllyasviel/control_v11p_sd15_openpose'
}

samplers_dict={
  'unipcm':UniPCMultistepScheduler,
  'ddpm':DDPMScheduler, 
  'ddim':DDIMScheduler,
  'pndm':PNDMScheduler,
  'lms':LMSDiscreteScheduler,
  'euler_a':EulerAncestralDiscreteScheduler,
  'euler':EulerDiscreteScheduler,
  'dpm':DPMSolverMultistepScheduler
}


def resize_for_condition_image(input_image: Image, resolution: int):
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(round(H / 64.0)) * 64
    W = int(round(W / 64.0)) * 64
    img = input_image.resize((W, H), resample=Image.LANCZOS)
    return img

class SD:
    def __init__(self, models_path, model_id, controlnet_model_id=None, torch_dtype=torch.float16, mo=True):
        self.torch_dtype = torch_dtype
        self.mo = mo
        model_path = os.path.join(models_path,model_id)
        self.model_path=model_path
        
        self.init_models(model_path)
        self.clean()
    
    def interrogate(self, image,f=2,f2=4):
        print('Interrogating.')
      
        if self.ci.interrogate:
          return self.ci.interrogate(image,f,f2)
        else:
          self.ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))
          print('Interrogator loaded.')
          return self.ci.interrogate(image,f,f2)

    def init_models(self, model_path):
        print('initializing models.')
        self.txt2img = StableDiffusionPipeline.from_pretrained(
            model_path, torch_dtype=self.torch_dtype
        ).to('cuda')
        print('checkers')
        self.txt2img.safety_checker=None
        self.txt2img.feature_extractor=None
        self.txt2img.requires_safety_checker=False
        print('tome')
        
        tomesd.apply_patch(self.txt2img, ratio=0.5)
        tomesd.apply_patch(self.txt2img, ratio=0.5)
        
        print('cn load')
        
        self.controlnet = StableDiffusionControlNetPipeline(vae=self.txt2img.vae,
          text_encoder=self.txt2img.text_encoder,
          tokenizer=self.txt2img.tokenizer,
          unet=self.txt2img.unet,
          controlnet=None,
          scheduler=self.txt2img.scheduler,
          safety_checker=None,
          feature_extractor=None,
          requires_safety_checker=False)
          
        print('cnok')
        
        ######
        if self.mo:
          self.txt2img.enable_xformers_memory_efficient_attention()
          #self.txt2img.enable_model_cpu_offload()
          
        self.img2img = StableDiffusionImg2ImgPipeline(
            vae=self.txt2img.vae,
            text_encoder=self.txt2img.text_encoder,
            tokenizer=self.txt2img.tokenizer,
            unet=self.txt2img.unet,
            scheduler=self.txt2img.scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        ).to('cuda')
        
        self.img2imgcontrolnet = StableDiffusionControlNetImg2ImgPipeline(
            vae=self.txt2img.vae,
            text_encoder=self.txt2img.text_encoder,
            tokenizer=self.txt2img.tokenizer,
            unet=self.txt2img.unet,
            controlnet=None,
            scheduler=self.txt2img.scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        ).to('cuda')
        
        print('loading sampler')
        
        self.load_sampler('euler_a')
        
        #### .enable_xformers_memory_efficient_attention()

        self.clean()
        
    def load_cnets(self, cnets, torch_dtype=torch.float16):
      cnets_loaded = []
      for cnet in cnets:
        if cnet in cnet_dict:
            cnets_loaded.append(ControlNetModel.from_pretrained(cnet_dict[cnet], torch_dtype=torch_dtype).to('cuda'))
        else:
            print(cnet, 'not found')
            
      if len(cnets_loaded)==0:
        self.controlnet.controlnet = None
        self.img2imgcontrolnet.controlnet = self.controlnet.controlnet
            
      if len(cnets_loaded)==1:
        self.controlnet.controlnet = cnets_loaded[0]
        self.img2imgcontrolnet.controlnet = self.controlnet.controlnet
        print('ok')
        
      if len(cnets_loaded)>1:
        self.controlnet.controlnet = MultiControlNetModel(cnets_loaded)
        self.img2imgcontrolnet.controlnet = self.controlnet
        print('ok')
    
    def load_sampler(self,  sampler, torch_dtype=torch.float16):
      print('loading sampler',sampler)
      if sampler in samplers_dict:
        self.txt2img.scheduler = samplers_dict[sampler].from_pretrained(self.model_path, subfolder="scheduler")
        #from_config(self.txt2img.scheduler.config,torch_dtype=torch_dtype)
        self.img2img.scheduler = self.txt2img.scheduler
        self.controlnet.scheduler = self.txt2img.scheduler
        self.img2imgcontrolnet.scheduler = self.txt2img.scheduler
      else:
        print('sampler '+sampler+' not found')
      
    
    def load_model(self,model_id):
      self.model_path = os.path.join(models_path,model_id)
      self.txt2img.vae = AutoencoderKL.from_pretrained(self.model_path+'/vae').to('cuda')
      self.txt2img.unet = UNet2DConditionModel.from_pretrained(self.model_path+'/unet').to('cuda')
      self.txt2img.text_encoder = CLIPTextModel.from_pretrained(self.model_path+'/text_encoder').to('cuda')
      self.txt2img.tokenizer = CLIPTokenizer.from_pretrained(self.model_path+'/tokenizer')
      self.txt2img.scheduler = PNDMScheduler.from_pretrained(self.model_path+'/scheduler')
      print ('loaded.')
      self.clean()
    
    def clean(self):
      gc.collect()
      torch.cuda.empty_cache()
      torch.cuda.ipc_collect()
    
    def load_controlnet(self,model):
      return 'x'
       
    def annotate(self,image):
        image = load_image(image)
        image = depth_estimator(image)["depth"]
        image = np.array(image)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = Image.fromarray(image)
        return image

def download_models(models,models_path):
  for model in models:
    local_dir=os.path.join(models_path,model)    
    print ('downloading '+model+' ...')
    snapshot_download(repo_id=model, ignore_patterns=["*.msgpack", "*.safetensors", "*.ckpt"],local_dir=local_dir)
