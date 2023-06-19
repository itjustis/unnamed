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
from huggingface_hub import snapshot_download
from clip_interrogator import Config, Interrogator



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
        self.ddpm = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
        self.ddim = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler")
        self.pndm = PNDMScheduler.from_pretrained(model_path, subfolder="scheduler")
        self.lms = LMSDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
        self.euler_a = EulerAncestralDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
        self.euler = EulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
        self.dpm = DPMSolverMultistepScheduler.from_pretrained(model_path, subfolder="scheduler")
        self.unipcm = UniPCMultistepScheduler.from_pretrained(model_path, subfolder="scheduler")
        self.init_models(model_path,controlnet_model_path)
        self.interrogate = self.ci.interrogate
        self.clean()

    def init_models(self, model_path, controlnet_model_path):
        self.txt2img = StableDiffusionPipeline.from_pretrained(
            model_path, torch_dtype=self.torch_dtype
        ).to('cuda')
        self.txt2img.scheduler = self.euler_a
        
        ######
        if self.mo:
          self.txt2img.enable_xformers_memory_efficient_attention()
          self.txt2img.enable_model_cpu_offload()
        
        self.img2img = DiffusionPipeline(
            vae=self.txt2img.vae,
            text_encoder=self.txt2img.text_encoder,
            tokenizer=self.txt2img.tokenizer,
            unet=self.txt2img.unet,
            custom_pipeline="stable_diffusion_controlnet_img2img",
            controlnet=None,
            torch_dtype=torch.float16
            scheduler=self.txt2img.scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        ).to('cuda')

        if controlnet_model_path:
            self.cn = ControlNetModel.from_pretrained(
                controlnet_model_path, torch_dtype=self.torch_dtype
            )

            self.controlnet = StableDiffusionControlNetPipeline.from_pretrained(
                controlnet_model_path, controlnet=self.cn, vae=self.txt2img.vae,
                text_encoder=self.txt2img.text_encoder,
                tokenizer=self.txt2img.tokenizer,
                unet=self.txt2img.unet,
                scheduler=self.txt2img.scheduler,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False, torch_dtype=torch.float16
            ).to('cuda')

            #annotators...
            self.depth_estimator = pipeline("depth-estimation")

            #x
            if self.mo:
              #self.controlnet.enable_model_cpu_offload()
              self.controlnet.enable_xformers_memory_efficient_attention()
        else:
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
