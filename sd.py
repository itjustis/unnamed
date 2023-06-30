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


def load_cnet(cnet,torch_dtype=torch.float16):
  if cnet == 'content':
    return ControlNetModel.from_pretrained('lllyasviel/control_v11f1e_sd15_tile',
     torch_dtype=torch_dtype).to('cuda')
  if cnet == 'depth':
    return ControlNetModel.from_pretrained('lllyasviel/control_v11f1p_sd15_depth',
     torch_dtype=torch_dtype).to('cuda')
  if cnet == 'scribble':
    return ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_scribble',
     torch_dtype=torch_dtype).to('cuda')
  if cnet == 'canny_edge':
    return ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_canny',
     torch_dtype=torch_dtype).to('cuda')
  if cnet == 'soft_edge':
    return ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_softedge',
     torch_dtype=torch_dtype).to('cuda')
  if cnet == 'shuffle':
    return ControlNetModel.from_pretrained('lllyasviel/control_v11e_sd15_shuffle',
     torch_dtype=torch_dtype).to('cuda')
  
  
  
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
        if controlnet_model_id:
          controlnet_model_path  = os.path.join(models_path,controlnet_model_id)
        else:
          controlnet_model_path = None
        
        self.ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))

        #self.UniPCM = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        #self.ddpm = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
        #self.ddim = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler")
        #self.pndm = PNDMScheduler.from_pretrained(model_path, subfolder="scheduler")
        #self.lms = LMSDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
        #self.euler_a = EulerAncestralDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
        #self.euler = EulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
        #self.dpm = DPMSolverMultistepScheduler.from_pretrained(model_path, subfolder="scheduler")
        self.unipcm = UniPCMultistepScheduler.from_pretrained(model_path, subfolder="scheduler")
        self.init_models(model_path,controlnet_model_path)
        self.interrogate = self.ci.interrogate
        self.clean()

    def init_models(self, model_path, controlnet_model_path):
        self.txt2img = StableDiffusionPipeline.from_pretrained(
            model_path, torch_dtype=self.torch_dtype
        ).to('cuda')
        self.txt2img.scheduler = self.unipcm
        
        tomesd.apply_patch(self.txt2img, ratio=0.5)
        
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
        
        #### .enable_xformers_memory_efficient_attention()

        self.controlnet = None 
        
          
        self.clean()
    
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
