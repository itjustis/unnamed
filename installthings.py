import os
import subprocess

def install(package):
    subprocess.check_call(["python", "-m", "pip", "install", "-q", package])

packages = ["diffusers==0.19.3","safetensors","transformers","xformers","git+https://github.com/huggingface/accelerate.git","opencv-contrib-python","controlnet_aux","huggingface_hub","pyngrok","tomesd","clip-interrogator==0.6.0","scikit-image", "pytorch_lightning"]
    
def main():

    for p in packages:
      print('installing '+p+'...')
      install(p)
   

if __name__ == "__main__":
    main()
