import torch
import numpy as np
from skimage import exposure
from transformers import pipeline
from torchvision.transforms import ToPILImage, ToTensor
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageDraw, ImageOps
import cv2
from controlnet_aux import  HEDdetector, ContentShuffleDetector



def cnet_prepare(controlnets,cnets_p, images, sz):
	
	for controlnet, prepare,image_path in zip(controlnets,cnets_p,images):
		print ('prepare for', controlnet, 'is', prepare)
		image = Image.open(image_path).resize(sz)

		if prepare :
			if controlnet == 'depth':
				image = p_depth(image)
			elif controlnet == 'tile':
				image = p_tile(image, sz.size[0])
			elif controlnet == 'canny_edge':
				image = p_canny(image)
			elif controlnet == 'soft_edge':
				image = p_canny(image)
				
			print(image,'saving')
			
			image.resize(sz).convert('RGB').save(image_path)
	
def p_shuffle(image):
	shuffle_processor = ContentShuffleDetector()
	control_image = shuffle_processor(image)
	return control_image
	

def p_soft(image):
	processor = HEDdetector.from_pretrained('lllyasviel/Annotators')
	control_image = processor(image, safe=True)
	return control_image
	
def p_canny(image):
	image = np.array(image)	
	low_threshold = 100
	high_threshold = 200	
	image = cv2.Canny(image, low_threshold, high_threshold)
	image = image[:, :, None]
	image = np.concatenate([image, image, image], axis=2)
	control_image = Image.fromarray(image)
	return control_image


def p_depth(init_image):
	depth_estimator = pipeline('depth-estimation')

	image = depth_estimator(init_image)['depth']
	image = np.array(image)
	image = image[:, :, None]
	image = np.concatenate([image, image, image], axis=2)
	control_image = Image.fromarray(image)
	return control_image

def p_tile(input_image: Image, resolution: int):
	input_image = input_image.convert("RGB")
	W, H = input_image.size
	k = float(resolution) / min(H, W)
	H *= k
	W *= k
	H = int(round(H / 64.0)) * 64
	W = int(round(W / 64.0)) * 64
	img = input_image.resize((W, H), resample=Image.LANCZOS)
	return img


def tile_upscale(image_path,upr,prompt,negative,pipe,controlnets,cn_scales,tile_size=768, shift=0.333,steps=25,scale=7.5,strength=0.666,interrogate=False):
	img_upscaled , original_size , upscaled_size= upscale_image(image_path,upr)
	img_upscaled = img_upscaled.convert('RGB')
	result = process_tiles(pipe, controlnets, cn_scales, img_upscaled, original_size, prompt, negative, strength, tile_size, shift,steps,scale,interrogate)
	return result

def process_tiles(pipe, controlnets, cn_scales, img_upscaled, original_size, prompt, negative, strength, tile_size=768, shift=0.333,steps=25,scale=7.5, interrogate=False):
	zz = 0

	width, height = img_upscaled.size
	x_steps = int(width // (tile_size * shift))+2
	y_steps = int(height // (tile_size * shift))+2

	print (img_upscaled.size)

	for i in range(x_steps):
		for j in range(y_steps):
			#clear_output()

			left = int(i * tile_size * shift)
			upper = int(j * tile_size * shift)
			right = left + tile_size
			lower = upper + tile_size

			if right <= width and lower <= height:
				tile = img_upscaled.crop((left, upper, right, lower))

				if interrogate:
				  prompt=sd.interrogate(img_upscaled.resize((768,768)),2,4)
				
				#generator=torch.manual_seed(65),
				
				###need to fix
				
				condition_image = cnet_prepare(controlnets, tile)
			
				itile = tile

				tile = pipe.img2imgcontrolnet(prompt=prompt,
					  negative_prompt= negative,
					  image=tile,
					  controlnet_conditioning_image=condition_image,
					  width=tile.size[0],
					  height=tile.size[1],
					  strength=strength,
					  guidance_scale=scale,
					  controlnet_conditioning_scale=cn_scales,
					  num_inference_steps=steps,
					  ).images[0]

				tile = matchc(tile,itile)
				img_upscaled.paste(tile, (left, upper), mask=ImageOps.invert(Image.open('tmask.png')).convert('L'))


	img_processed = crop_image( img_upscaled, tile_size//6)

	return img_processed


def matchc(x,y):
  remapped_np = np.array(x)
  prev_np = np.array(y)
  matched = color_match(remapped_np, np.array(x))
  return Image.fromarray(matched)


def add_border(image, border):
	# Define the border color
	color = (0, 0, 0)  # black

	# Create a new image with a border
	new_image = ImageOps.expand(image, border=border, fill=color)

	return new_image


def crop_image(image, crop_pixels):
	width, height = image.size

	# Define the cropping box - left, upper, right, lower
	crop_box = (crop_pixels, crop_pixels, width - crop_pixels, height - crop_pixels)

	# Create a new image by cropping the original image
	new_image = image.crop(crop_box)

	return new_image


def upscale_image(image_path, upscale_factor=4, padding_size=768):
	img = Image.open(image_path)
	display(img)
	width, height = img.size
	new_size = (width * upscale_factor, height * upscale_factor)
	img_upscaled = img.resize(new_size, Image.ANTIALIAS)


	img_extended = add_border(img_upscaled, padding_size//3)

	print('img_upscaled.size',img_upscaled.size,'. img_extended.size',img_extended.size)

	return img_extended, (width, height), img_upscaled.size

def color_match(prev_img,color_match_sample):
  prev_img_lab = cv2.cvtColor(prev_img, cv2.COLOR_RGB2LAB)
  color_match_lab = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2LAB)
  matched_lab = exposure.match_histograms(prev_img_lab, color_match_lab, multichannel=True)
  return cv2.cvtColor(matched_lab, cv2.COLOR_LAB2RGB)

def add_noise(tensor, mean=0., std=1.):
	"""
	Add Gaussian noise to a tensor and clip values to valid range.

	Args:
	tensor (torch.Tensor): The input tensor.
	mean (float): Mean of the Gaussian distribution to generate noise.
	std (float): Standard deviation of the Gaussian distribution to generate noise.

	Returns:
	torch.Tensor: The tensor with added noise.
	"""
	noise = torch.randn_like(tensor) * std + mean
	noisy_tensor = tensor + noise
	noisy_tensor = torch.clamp(noisy_tensor, tensor.min(), tensor.max())
	return noisy_tensor

def create_gif(image_list, duration, output_path):
	"""
	Create a GIF from a list of PIL Images.

	Args:
	image_list (list): List of PIL Image objects.
	duration (int): Duration between frames in the GIF (in milliseconds).
	output_path (str): Path to save the output GIF.
	"""
	# Save the image list as a GIF
	image_list[0].save(
		output_path, save_all=True, append_images=image_list[1:], optimize=False, duration=duration, loop=0
	)

def tensor_to_pil(tensor):
	"""
	Convert a PyTorch tensor to a PIL Image.
	"""
	to_pil = ToPILImage()
	return to_pil(tensor)

def pil_to_tensor(image):
	"""
	Convert a PIL Image to a PyTorch tensor.
	"""
	to_tensor = ToTensor()
	return to_tensor(image)

def display_tensor_image(tensor):
	"""
	Display a PyTorch tensor as an image.
	"""
	plt.imshow(tensor_to_pil(tensor))
	plt.show()

def remap_displacement(image, displacement):
	"""
	Applies a displacement field (also known as remap) to an image.
	`image` should be a PyTorch tensor with shape (C, H, W).
	`displacement` should be a PyTorch tensor with shape (2, H, W) where
	the first channel is the x displacement and the second channel is the y displacement.
	"""
	if isinstance(image, Image.Image):
		image = pil_to_tensor(image)
	elif isinstance(image, torch.Tensor):
		pass
	else:
		raise ValueError('Image should be a PyTorch tensor or a PIL Image')
	
	image = image.to(displacement.device)

	# Create a grid to apply the displacement
	grid_y, grid_x = torch.meshgrid(
		torch.linspace(-1, 1, image.shape[1]),
		torch.linspace(-1, 1, image.shape[2]),
	)

	# Stack x and y grid components together
	grid = torch.stack((grid_x, grid_y)).unsqueeze(0).to(displacement.device)

	# Apply the displacement
	grid += displacement

	# Reshape the grid
	grid = grid.permute(0, 2, 3, 1)

	# Apply the grid to the image
	#return torch.nn.functional.grid_sample(image.unsqueeze(0), grid, mode='bilinear', padding_mode='border', align_corners=True).squeeze(0)
	return torch.nn.functional.grid_sample(image.unsqueeze(0), grid, mode='bilinear', padding_mode='reflection', align_corners=True).squeeze(0)


def create_uv_map(image):
	"""
	Create a UV map with the same size as `image`.
	`image` should be a PyTorch tensor with shape (C, H, W) or a PIL Image.
	Returns a tensor with shape (2, H, W) where the first channel is the U map and the second channel is the V map.
	"""
	if isinstance(image, Image.Image):
		# Convert PIL Image to tensor
		image = pil_to_tensor(image)
	elif isinstance(image, torch.Tensor):
		pass
	else:
		raise ValueError('Image should be a PyTorch tensor or a PIL Image')

	# Create a grid
	grid = torch.meshgrid(
		torch.linspace(0, 1, image.shape[1]),
		torch.linspace(0, 1, image.shape[2]),
	)
	return torch.stack(grid)

def process_displacement_image(displacement_image):
	"""
	Processes a displacement image to a displacement tensor.
	Red channel is used as x displacement and green channel is used as y displacement.
	"""
	tensor = pil_to_tensor(displacement_image)
	# Normalize displacement to [-1, 1] (assuming input is [0, 1])
	tensor = tensor * 2 - 1
	# Use only red and green channels for x and y displacement
	return tensor[:2, :, :]

def remap_displacement_depth(image, depth_map):
	"""
	Applies a displacement field (also known as remap) to an image using a depth map.
	`image` should be a PyTorch tensor with shape (C, H, W).
	`depth_map` should be a PyTorch tensor with shape (1, H, W).
	"""
	# Make sure the image and displacement are on the same device
	image = image.to(depth_map.device)

	# Create a grid to apply the displacement
	grid_y, grid_x = torch.meshgrid(
		torch.linspace(-1, 1, image.shape[1]), 
		torch.linspace(-1, 1, image.shape[2]),
	)

	# Stack x and y grid components together
	grid = torch.stack((grid_x, grid_y)).unsqueeze(0).to(depth_map.device)

	# Normalize depth_map to range [-1,1] for displacement
	depth_map = depth_map * 2 - 1

	# Scale the displacement by the depth_map (this is a simple model and might not be perfect for all use cases)
	grid += grid * depth_map

	# Reshape the grid
	grid = grid.permute(0, 2, 3, 1)

	# Apply the grid to the image
	return torch.nn.functional.grid_sample(image.unsqueeze(0), grid, mode='bilinear', padding_mode='reflection', align_corners=True).squeeze(0)


