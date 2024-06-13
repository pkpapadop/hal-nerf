#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 11:00:27 2023

@author: Asteris



"""
import matplotlib.pyplot as plt
import os
import json
from nerfstudio.pipelines.base_pipeline import VanillaPipeline 
from nerfstudio.pipelines import base_pipeline
from nerfstudio.engine import trainer
import torch
import numpy as np
from nerfstudio.models.nerfacto import NerfactoModel , NerfactoField , NerfactoModelConfig
from nerfstudio.data import scene_box
from nerfstudio.models.base_model import Model ,ModelConfig
from nerfstudio.configs.base_config import InstantiateConfig , PrintableConfig
import copy
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParserConfig

from pathlib import Path
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.cameras import Cameras, CameraType
from scipy.spatial.transform import Rotation
from torchmetrics.image import StructuralSimilarityIndexMeasure
import torch.nn.functional as F
import cv2
from nerfstudio.data.datasets.base_dataset import InputDataset
import matplotlib.image as mpimg
from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from pose_regressor.script.dfnet import DFNet, DFNet_s




### Nerfacto Config    
#config = NerfstudioDataParserConfig(data = Path('/home/asterisreppas/kareklaexo/transforms.json') , downscale_factor = 8 )
#dataparser = config.setup()
#dataparser_outputs = dataparser.get_dataparser_outputs(split="train")
#scene = dataparser_outputs.scene_box
#input_dataset = InputDataset(dataparser_outputs)
#fx = dataparser_outputs.cameras.fx[0][0]
#fy = dataparser_outputs.cameras.fy[0][0]
#cx = dataparser_outputs.cameras.cx[0][0]
#cy = dataparser_outputs.cameras.cy[0][0]
#width = dataparser_outputs.cameras.width[0][0]
#height = dataparser_outputs.cameras.height[0][0]
#distortion_params = dataparser_outputs.cameras.distortion_params[0]
#camera_type = CameraType.PERSPECTIVE

#result = {
#    'fx': fx,
#    'fy': fy,
#    'cx': cx,
#    'cy': cy,
#    'width': width,
#    'height': height,
#    'distortion_params': distortion_params,
#    'camera_type': camera_type
#}

#a = result


#index = 300
#image_height = height
#image_width = width







def load_model(transform_path, checkpoint_path, factor):
  global loaded_state

  config = ColmapDataParserConfig(data = Path(transform_path) , downscale_factor = factor)
  dataparser = config.setup()
  dataparser_outputs = dataparser.get_dataparser_outputs(split="train")
  scene = dataparser_outputs.scene_box
  print(scene)
#Model and model_state_dict
  #model = NerfactoModel(NerfactoModelConfig(ModelConfig(InstantiateConfig(PrintableConfig))),  scene, 70)
  model = NerfactoModel(NerfactoModelConfig(),  scene , 796)
# Load checkpoint
  root = checkpoint_path

  loaded_state = torch.load(root)
  loaded_state = loaded_state["pipeline"]
  common_substring = "_model."

# Create a new dictionary with updated keys
  updated_dict = {k.replace(common_substring, ""): v for k, v in loaded_state.items()}

  model.load_state_dict(updated_dict, strict=False)
  model.eval()
  
  return model
 
  
 
def input_dataset(transform_path, index, factor):
  
  config = ColmapDataParserConfig(data = Path(transform_path) , downscale_factor = factor )
  dataparser = config.setup()
  dataparser_outputs = dataparser.get_dataparser_outputs(split="train")
  input_dataset = InputDataset(dataparser_outputs)
  ground_truth_pose = dataparser_outputs.cameras.camera_to_worlds[index,:,:]
  image = input_dataset.get_image(index)
  width = dataparser_outputs.cameras.width[0][0]
  height = dataparser_outputs.cameras.height[0][0]

  
  return image, ground_truth_pose, width, height
  
  
def get_params(transform_path, downscale_factor):
  config = ColmapDataParserConfig(data = Path(transform_path) , downscale_factor = downscale_factor)
  dataparser = config.setup()
  dataparser_outputs = dataparser.get_dataparser_outputs(split="train")
  fx = dataparser_outputs.cameras.fx[0][0]
  fy = dataparser_outputs.cameras.fy[0][0]
  cx = dataparser_outputs.cameras.cx[0][0]
  cy = dataparser_outputs.cameras.cy[0][0]
  width = dataparser_outputs.cameras.width[0][0]
  height = dataparser_outputs.cameras.height[0][0]
  distortion_params = dataparser_outputs.cameras.distortion_params[0]
  scene_box = dataparser_outputs.scene_box
  aabb = scene_box.aabb 
  camera_type = CameraType.PERSPECTIVE
  result = {
    'fx': fx,
    'fy': fy,
    'cx': cx,
    'cy': cy,
    'width': width,
    'height': height,
    'distortion_params': distortion_params,
    'camera_type': camera_type,
    'scene_box' : scene_box,
    'aabb'  : aabb
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
  
  return rgb
  
def vizualize(pose, model, a):
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
  # coords1 = camera.get_image_coords(index=0)
  # print(coords1.shape)
  camera_ray_bundle = camera.generate_rays(camera_indices=0, coords = coords).to('cuda')
  nears_value = 0.05
  fars_value = 1000
  batch_size = a['height']*a['width']  # Set your desired batch size
  nears = torch.full((batch_size, 1), nears_value, dtype=torch.float32, device='cuda')
  fars = torch.full((batch_size, 1), fars_value, dtype=torch.float32, device='cuda')
  camera_ray_bundle.nears = nears
  camera_ray_bundle.fars = fars

  model.eval()
  with torch.no_grad():
   outputs = model.get_outputs(camera_ray_bundle)
  rgb = outputs["rgb"]
  rgb = rgb.cpu().detach().numpy()
  rgb = rgb.reshape((a['height'], a['width'], 3))
  #rgb = torch.tensor(rgb)
  #mpimg.imsave(output_path + '/images/image'+ str(update_step) + '.png', rgb)
  return rgb
  
  
def load_and_predict_pose(checkpoint_path_2, image, transform_path, downscale_factor):
  #config = NerfstudioDataParserConfig(data = Path(transform_path) , downscale_factor = downscale_factor )
  #dataparser = config.setup()
  #dataparser_outputs = dataparser.get_dataparser_outputs(split="train")
  #poses = dataparser_outputs.cameras.camera_to_worlds[:,:,:]
  feat_model = DFNet()
  feat_model.load_state_dict(torch.load(checkpoint_path_2))
  feat_model.to('cuda')
  feat_model.eval()
  with torch.no_grad():
    _,predict_pose = feat_model(image)
  predict_pose = predict_pose.view(1,3,4)
  predict_pose = predict_pose.squeeze(0)
  return predict_pose


# def render(pose, model, a):
#   camera_to_world = torch.tensor(pose)
#   camera_to_worlds = camera_to_world[None, ...]
#   camera = Cameras(camera_to_worlds, a['fx'], a['fy'], a['cx'], a['cy'], a['width'], a['height'],a['distortion_params'], a['camera_type'])
#   rays = RayGenerator(camera)
#   field_outputs = field(rays)
#   weights = ray_sampler.get_weights(field_outputs[FieldHeadNames.DENSITY])

#   rgb_renderer = RGBRenderer()
#   rgb = rgb_renderer(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
  


def main():
  factor = 8
  transform_path = '/home/asterisreppas/Nerfstudio_Problem/90FOV_320FRAMES_1920x1080/colmap'
  checkpoint_path = '/home/asterisreppas/Nerfstudio_Problem/90FOV_320FRAMES_1920x1080/modelgamo/nerfacto/2024-03-15_154328/nerfstudio_models/step-000029999.ckpt'
  nerfacto_model = load_model(transform_path, checkpoint_path, factor)
  nerfacto_params = get_params(transform_path, factor)
  _,pose,_,_ = input_dataset(transform_path, 20, factor)
  rgb = vizualize(pose, nerfacto_model, nerfacto_params)
  plt.figure()
  plt.imshow(rgb)
  plt.show() 




if __name__ == "__main__":
   main()



    
  
  
  
  
  
  
  
  
 










#camera_to_world = torch.tensor(final_pose)
#camera_to_worlds = camera_to_world[None, ...]


#camera = Cameras(camera_to_worlds, a['fx'], a['fy'], a['cx'], a['cy'], a['width'], a['height'],a['distortion_params'], a['camera_type'])


#Create a grid of x and y coordinates using meshgrid

#x_coords, y_coords = torch.meshgrid(torch.arange(image_height), torch.arange(image_width))
#x_coords = x_coords
#y_coords = y_coords


# Reshape the coordinates to a single tensor of shape (num_pixels, 2)
#x_coords_flat = x_coords.reshape(-1, 1)
#y_coords_flat = y_coords.reshape(-1, 1)

# Concatenate x and y coordinates to form the final tensor
#coords = torch.cat((x_coords_flat, y_coords_flat), dim=-1)
#camera_ray_bundle = camera.generate_rays(camera_indices = 0, coords=coords).to('cuda')
#nears_value = 0.1
#fars_value = 8
#batch_size = image_height*image_width  # Set your desired batch size
#nears = torch.full((batch_size, 1), nears_value, dtype=torch.float32, device='cuda')
#fars = torch.full((batch_size, 1), fars_value, dtype=torch.float32, device='cuda')
#camera_ray_bundle.nears = nears
#camera_ray_bundle.fars = fars
#outputs = model.get_outputs(camera_ray_bundle)
#rgb = outputs["rgb"]
#rgb = rgb.cpu().detach().numpy()
#rgb = cv2.normalize(rgb, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
#rgb = rgb.reshape((240, 135, 3))


#image = input_dataset.get_image(index)

#rgb = rgb[:,:256,:]
#plt.imshow(image)
#plt.show()
#plt.imshow(rgb)
#plt.show()

#rgb = torch.tensor(rgb)
#image = torch.tensor(image)
#target = image
#preds = rgb
#preds = preds.unsqueeze(0).permute(0, 3, 1, 2)
#target = target.unsqueeze(0).permute(0, 3, 1, 2)
#         #rgb = get_particle_pose_camera(pose, batch, self.model, params, self.batch_size)
#ssim = StructuralSimilarityIndexMeasure()
#loss7 = ssim(target,preds)
# print(loss)
#psnr = PeakSignalNoiseRatio()
#loss0 = psnr(target, preds)

#loss1 = F.mse_loss(target, preds)

#lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex') 
#loss2 = lpips(target, preds)
#lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg') 
#loss3= lpips(target, preds)
#rmse_sw = RootMeanSquaredErrorUsingSlidingWindow(window_size = 8)
#loss4 = rmse_sw(target, preds)
#rmse_sw = RootMeanSquaredErrorUsingSlidingWindow(window_size = 64)
#loss5 = rmse_sw(target, preds)
#rmse_sw = RootMeanSquaredErrorUsingSlidingWindow(window_size = 128)
#loss6 = rmse_sw(target, preds)
#rase = RelativeAverageSpectralError()
#loss8 = rase(preds, target)




#print('PSNR :',loss0)
#print('MSE:',loss1)
#print('PERCEPTUAL ALEX:', loss2)
#print('PERCEPTUAL VGG:', loss3)
#print('RMSE_W8', loss4)
#print('RMSE_W64', loss5)
#print('RMSE_W128', loss6)
#print('SSIM', loss7)
#print('RASE', loss8)


# # Create SIFT detector 
#     sift = cv2.SIFT_create()

# # Detect keypoints and compute descriptors
#     keypoints1, descriptors1 = sift.detectAndCompute(rgb1, None)
#     keypoints2, descriptors2 = sift.detectAndCompute(rgb, None)


# # Brute-force matcher3
#     bf = cv2.BFMatcher()
#     matches = bf.knnMatch(descriptors1, descriptors2, k=1)


  

#     orientation_threshold = 5  # Adjust as needed
#     filtered_matches_m1 = [m1 for m1 in matches if abs(keypoints1[m1[0].queryIdx].angle - keypoints2[m1[0].trainIdx].angle) < orientation_threshold]
   

#     score =  len(filtered_matches_m1)


#     sum_of_angles = [abs(keypoints1[m1[0].queryIdx].angle - keypoints2[m1[0].trainIdx].angle) for m1 in matches]

#     sum_of_angles = np.array(sum_of_angles)

#     mean = np.mean(sum_of_angles)
#     std = np.std(sum_of_angles)
#     median = np.median(sum_of_angles)
#     summ = np.sum(sum_of_angles)

#     score = score**2 * median

#     score1 = np.log(score)



# rgb = torch.tensor(rgb)
# rgb1 = torch.tensor(rgb1)
# preds = rgb
# target = rgb1
# preds = preds.unsqueeze(0).permute(0, 3, 1, 2)
# target = target.unsqueeze(0).permute(0, 3, 1, 2)
# rmse_sw = RootMeanSquaredErrorUsingSlidingWindow(window_size = 4)
# loss = rmse_sw(target, preds)
# print(loss)
        
    



  




   
# print(sum_of_angles)
# print(mean, std, median, summ)
# print(score, score1)
   
# Determine if the images are different based on mismatches
# if mismatch_count > some_threshold:
#     print("The images are different.")
# else:
#     print("The images are similar.")


# target = np.array(target)
# preds = np.array(preds)

# loss3 = fsim(target, preds)
# print(loss3)






























































