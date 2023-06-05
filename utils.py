import torch
import numpy as np
from skimage import exposure
from torchvision.transforms import ToPILImage, ToTensor
import matplotlib.pyplot as plt
from PIL import Image

def add_noise(tensor, mean=0., std=1.):
    """
    Add Gaussian noise to a tensor.

    Args:
    tensor (torch.Tensor): The input tensor.
    mean (float): Mean of the Gaussian distribution to generate noise.
    std (float): Standard deviation of the Gaussian distribution to generate noise.

    Returns:
    torch.Tensor: The tensor with added noise.
    """
    noise = torch.randn_like(tensor) * std + mean
    return (tensor - std) + noise

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
	# Make sure the image and displacement are on the same device
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
	return torch.nn.functional.grid_sample(image.unsqueeze(0), grid, mode='bilinear', padding_mode='border', align_corners=True).squeeze(0)

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
	return torch.nn.functional.grid_sample(image.unsqueeze(0), grid, mode='bilinear', padding_mode='border', align_corners=True).squeeze(0)

