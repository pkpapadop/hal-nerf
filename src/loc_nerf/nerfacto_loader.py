#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 11:00:27 2023

@author: Asteris



"""
import matplotlib.pyplot as plt
import multiprocess as mp  # or multiprocessing, depending on your import
from pathlib import Path
import yaml
from nerfstudio.utils.eval_utils import eval_setup  # Adjust this import according to your project structure
import torch
import numpy as np
from nerfstudio.models.nerfacto import NerfactoModel ,NerfactoModelConfig
from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParserConfig, ColmapDataParser
from pathlib import Path
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.cameras import Cameras, CameraType
import torch.nn.functional as F
from nerfstudio.data.datasets.base_dataset import InputDataset
import os 
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg













def load_model():
  
    config_path = Path('/root/colmap_output/model/nerfacto/default/config.yml')
    config, pipeline, checkpoint_path, step = eval_setup(config_path=config_path)
    model = pipeline.model


    return model
 
  
 
def input_dataset(transform_path, index, factor):
  
  config = ColmapDataParserConfig(data=Path(transform_path), downscale_factor=factor, orientation_method="none", auto_scale_poses="true", center_method="none", load_3D_points=False, colmap_path='/root/colmap_output/colmap/sparse/0')
  dataparser = config.setup()
  dataparser_outputs = dataparser.get_dataparser_outputs(split="train")
  input_dataset = InputDataset(dataparser_outputs)
  ground_truth_pose = dataparser_outputs.cameras.camera_to_worlds[index,:,:]
  image = input_dataset.get_image_float32(index)
  image_filename = dataparser_outputs.image_filenames[index]
  width = dataparser_outputs.cameras.width[0][0]
  height = dataparser_outputs.cameras.height[0][0]

  
  return image, ground_truth_pose, width, height, image_filename
  
  
def get_params(transform_path, downscale_factor):
  config = ColmapDataParserConfig(data=Path(transform_path), downscale_factor=downscale_factor, orientation_method="none", auto_scale_poses="true", center_method="none", load_3D_points=False, colmap_path='/root/colmap_output/colmap/sparse/0')
  dataparser = config.setup()
  dataparser_outputs = dataparser.get_dataparser_outputs(split="train")
  fx = dataparser_outputs.cameras.fx[0][0]
  fy = dataparser_outputs.cameras.fy[0][0]
  cx = dataparser_outputs.cameras.cx[0][0]
  cy = dataparser_outputs.cameras.cy[0][0]
  width = dataparser_outputs.cameras.width[0][0]
  height = dataparser_outputs.cameras.height[0][0]
  distortion_params = dataparser_outputs.cameras.distortion_params[0]
  camera_type = CameraType.PERSPECTIVE
  result = {
    'fx': fx,
    'fy': fy,
    'cx': cx,
    'cy': cy,
    'width': width,
    'height': height,
    'distortion_params': distortion_params,
    'camera_type': camera_type
}
  return result 
  
  
  
def get_loss(pose, model, a, batch, batchsize):
  
  pose = pose[:3,:]
  camera_to_world = torch.tensor(pose)
  camera_to_worlds = camera_to_world[None, ...]
  camera = Cameras(camera_to_worlds, a['fx'], a['fy'], a['cx'], a['cy'], a['width'], a['height'],a['distortion_params'], a['camera_type'])
  batch = torch.tensor(batch)
  coords = torch.stack((batch[:,1], batch[:,0]), dim= -1 )
  camera_ray_bundle = camera.generate_rays(camera_indices = 0, coords=coords).to('cuda')
  nears_value = 0.05
  fars_value = 1000
  nears = torch.full((batchsize, 1), nears_value, dtype=torch.float32, device='cuda')
  fars = torch.full((batchsize, 1), fars_value, dtype=torch.float32, device='cuda')
  camera_ray_bundle.nears = nears
  camera_ray_bundle.fars = fars
  outputs = model.get_outputs(camera_ray_bundle)
  rgb = outputs["rgb"]
  depth = outputs["expected_depth"]


  
  return rgb, depth
  
def vizualize(pose, model, a, update_step, output_path):
  camera_to_world = torch.tensor(pose)
  camera_to_worlds = camera_to_world[None, ...]
  camera = Cameras(camera_to_worlds, a['fx'], a['fy'], a['cx'], a['cy'], a['width'], a['height'],a['distortion_params'], a['camera_type'])
  

#Create a grid of x and y coordinates using meshgrid

  x_coords, y_coords = torch.meshgrid(torch.arange(a['height']), torch.arange(a['width']))


# Reshape the coordinates to a single tensor of shape (num_pixels, 2)
  x_coords_flat = x_coords.reshape(-1, 1)
  y_coords_flat = y_coords.reshape(-1, 1)

# Concatenate x and y coordinates to form the final tensor
  coords = torch.cat((x_coords_flat, y_coords_flat), dim=-1)
  camera_ray_bundle = camera.generate_rays(camera_indices = 0, coords=coords).to('cuda')
  nears_value = 0.05
  fars_value = 1000
  batch_size = a['height']*a['width']  # Set your desired batch size
  nears = torch.full((batch_size, 1), nears_value, dtype=torch.float32, device='cuda')
  fars = torch.full((batch_size, 1), fars_value, dtype=torch.float32, device='cuda')
  camera_ray_bundle.nears = nears
  camera_ray_bundle.fars = fars
  outputs = model.get_outputs(camera_ray_bundle)
  rgb = outputs["rgb"]
  rgb = rgb.cpu().detach().numpy()
  rgb = rgb.reshape((a['height'], a['width'], 3))
  mpimg.imsave(output_path + '/images/image'+ str(update_step) + '.png', rgb)
  #return rgb





def upload_files(image_folder, pose_folder):
    # Lists to store loaded images and poses
    images = []
    poses = []

    # Load images
    for file_name in sorted(os.listdir(image_folder)):
        if file_name.endswith('.jpg'):
            image_path = os.path.join(image_folder, file_name)
            try:
                image = Image.open(image_path).convert("RGB")  # Ensure image is in RGB format
                images.append(image)
            except Exception as e:
                print(f"Error loading image {file_name}: {e}")

    # Load poses
    for file_name in sorted(os.listdir(pose_folder)):
        if file_name.endswith('.pt'):
            pose_path = os.path.join(pose_folder, file_name)
            try:
                pose = torch.load(pose_path)
                poses.append(pose)
            except Exception as e:
                print(f"Error loading pose {file_name}: {e}")

  
    return images, poses


    
  
  
  
  
  
  
  
  
 
















