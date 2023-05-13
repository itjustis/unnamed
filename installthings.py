import os
import subprocess

def install(package):
    subprocess.check_call(["python", "-m", "pip", "install", "-q", package])

packages = ["diffusers==0.14.0","transformers","xformers","git+https://github.com/huggingface/accelerate.git","opencv-contrib-python","controlnet_aux","huggingface_hub"]
    
def main():

    for p in packages:
      print('installing '+p+'...')
      install(p)
   

if __name__ == "__main__":
    main()
