from matplotlib.markers import MarkerStyle
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from utils import show_img, find_POI, img2mse, load_llff_data, get_pose
from full_nerf_helpers import load_nerf
from render_helpers import render, to8b, get_rays
from particle_filter import ParticleFilter
# AR start
from nerfacto_loader import input_dataset
from nerfacto_loader import get_params
from nerfacto_loader import get_loss
from nerfacto_loader import load_model
# AR end

from scipy.spatial.transform import Rotation as R
from nerfacto_loader import vizualize
from nerfacto_loader_2 import load_and_predict_pose
# from depth_anything.dpt import DepthAnything
# from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
import torch.nn.functional as F
from torchvision.transforms import Compose

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# part of this script is adapted from iNeRF https://github.com/salykovaa/inerf
# and NeRF-Pytorch https://github.com/yenchenlin/nerf-pytorch/blob/master/load_llff.py

class NeRF:
    
    def __init__(self, nerf_params):
        # Parameters
        self.output_dir = './output/'
        self.data_dir = nerf_params['data_dir']
        self.model_name = nerf_params['model_name']
        self.obs_img_num = nerf_params['obs_img_num']
        self.batch_size = nerf_params['batch_size']
        self.factor = nerf_params['factor']
        self.near = nerf_params['near']
        self.far = nerf_params['far']
        self.spherify = False
        self.kernel_size = nerf_params['kernel_size']
        self.lrate = nerf_params['lrate']
        self.dataset_type = nerf_params['dataset_type']
        self.sampling_strategy = nerf_params['sampling_strategy']
        self.delta_phi, self.delta_theta, self.delta_psi, self.delta_x, self.delta_y, self.delta_z = [0,0,0,0,0,0]
        self.no_ndc = nerf_params['no_ndc']
        self.dil_iter = nerf_params['dil_iter']
        self.chunk = nerf_params['chunk'] # number of rays processed in parallel, decrease if running out of memory
        self.bd_factor = nerf_params['bd_factor']

        print("dataset type:", self.dataset_type)
        print("no ndc:", self.no_ndc)
        
        if self.dataset_type == 'custom':
            print("self.factor", self.factor)
            self.focal = nerf_params['focal'] / self.factor
            self.H =  nerf_params['H'] / self.factor
            self.W =  nerf_params['W'] / self.factor

            # we don't actually use obs_img_pose when we run live images. this prevents attribute errors later in the code
            self.obs_img_pose = None

            self.H, self.W = int(self.H), int(self.W)

        else:
            # AR start
            #self.obs_img, self.hwf, self.start_pose, self.obs_img_pose, self.bds = load_llff_data(self.data_dir, self.model_name, self.obs_img_num, self.delta_phi, self.delta_theta, self.delta_psi, self.delta_x, self.delta_y, self.delta_z, factor=self.factor, recenter=True, bd_factor=self.bd_factor, spherify=self.spherify)
            #self.H, self.W, self.focal = self.hwf
            
            checkpoint_path = '/root/weight.ckpt'
            checkpoint_path_2 = '/root/catkin_ws/src/Loc-NeRF/src/pose_regressor/logs/checkpoint.pt'
            transform_path = '/root/colmap_output'
            self.obs_img, self.obs_img_pose, self.W, self.H, self.image_filename = input_dataset(transform_path, self.obs_img_num, self.factor)
            print("The filename is :")
            print(self.image_filename)
            self.obs_img = torch.tensor(self.obs_img)
            img_dfnet = self.obs_img.permute(2, 0, 1)  # Reorder to [c, h, w]
            img_dfnet = img_dfnet.unsqueeze(0)
            img_dfnet = img_dfnet.to('cuda')

            # TODO : Depth Module

            # transform = Compose([
            #     Resize(
            #         width=518,
            #         height=518,
            #         resize_target=False,
            #         keep_aspect_ratio=True,
            #         ensure_multiple_of=14,
            #         resize_method='lower_bound',
            #         image_interpolation_method=cv2.INTER_CUBIC,
            #     ),
            #     # NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            #     # PrepareForNet(),
            # ])
            # print(self.obs_img.shape)
            # image_depth = transform({'image': self.obs_img.numpy()})['image']
            # image_depth = torch.from_numpy(image_depth).unsqueeze(0).to('cpu')
            


            #print(img_dfnet, img_dfnet.shape)
            predict_pose = load_and_predict_pose(checkpoint_path_2, img_dfnet, transform_path, self.factor)
            self.predicted_pose = predict_pose.cpu().numpy()
            self.predicted_pose = np.vstack([self.predicted_pose, [0, 0, 0, 1]])
            #print(self.predicted_pose[:3,3], self.obs_img_pose)


            self.obs_img_pose = self.obs_img_pose.cpu().numpy()
            t = self.obs_img_pose[:3, 3]
            #Extract the rotation matrix (R)    
            Rot = self.obs_img_pose[:3, :3]
            euler_angles = R.from_matrix(Rot).as_euler('xyz', degrees=True)
    # Combine the translation and rotation into a single vector
            self.pose_vector = np.concatenate((t, euler_angles))
            self.obs_img_pose = np.vstack([self.obs_img_pose, [0, 0, 0, 1]])



            
            
            # AR end
            
            if self.no_ndc:
                self.near = np.ndarray.min(self.bds) * .9
                self.far = np.ndarray.max(self.bds) * 1.
                print(self.near, self.far)
            else:
                self.near = 0.
                self.far = 1.
            
            self.H, self.W = int(self.H), int(self.W)
            
			#self.obs_img = (np.array(self.obs_img) / 255.).astype(np.float32) # AR
            self.obs_img_noised = self.obs_img

            # create meshgrid from the observed image
            self.coords = np.asarray(np.stack(np.meshgrid(np.linspace(0, self.W - 1, self.W), np.linspace(0, self.H - 1, self.H)), -1),
                                dtype=int)

            self.coords = self.coords.reshape(self.H * self.W, 2)
        
        # print("height, width, focal:", self.H, self.W, self.focal)

		# AR start
        # Load NeRF Model
        #self.render_kwargs = load_nerf(nerf_params, device)
        #bds_dict = {
        #    'near': self.near,
        #    'far': self.far,
        #}
            
        #self.render_kwargs.update(bds_dict)

        self.model= load_model(transform_path, checkpoint_path, self.factor)
        self.a = get_params(transform_path, self.factor)

        # TODO : Depth Module
        # self.depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format('vitl')).to('cpu').eval()
        # with torch.no_grad():
        #         depth = self.depth_anything(image_depth)
        # depth = F.interpolate(depth[None], (self.H, self.W), mode='bilinear', align_corners=False)[0, 0]
        # self.depth = (depth - depth.min()) / (depth.max() - depth.min())
        # print(depth)
        # # depth = depth.cpu().numpy().astype(np.uint8)
        # # depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

        

        # plt.imshow(depth, cmap='inferno')  # Use 'inferno' colormap for visualization
        # plt.colorbar()  # To see the color scale
        # plt.title('Normalized Depth Map')
        # plt.axis('off')  # Hide axes ticks
        # plt.show()

      

        if nerf_params['export_images']: # TODO: add self.export_images to NeRF object
            vizualize(self.obs_img_pose[:3,:], self.model, self.a,0, nerf_params['output_path'])
            #vizualize(self.predicted_pose[:3,:], self.model, self.a,1000, nerf_params['output_path'])
        # AR end
    
    def get_poi_interest_regions(self, show_img=False, sampling_type = None):
        # TODO see if other image normalization routines are better
        self.obs_img_noised = (np.array(self.obs_img) / 255.0).astype(np.float32)

        if show_img:
            plt.imshow(self.obs_img_noised)
            plt.show()

        self.coords = np.asarray(np.stack(np.meshgrid(np.linspace(0, self.W - 1, self.W), np.linspace(0, self.H - 1, self.H)), -1),
                            dtype=int)

        if sampling_type == 'random':
            self.coords = self.coords.reshape(self.H * self.W, 2)

    def get_loss(self, particles, batch, photometric_loss='rgb'):
        target_s = self.obs_img_noised[batch[:, 1], batch[:, 0]] # TODO check ordering here
        # depth_s = self.depth[batch[:, 1], batch[:, 0]]
        # depth_s = torch.Tensor(depth_s).to(device)
        #print(depth_s.shape)
        target_s = torch.Tensor(target_s).to(device)
        # print(target_s.shape)
        # _, target_depth = get_loss(self.obs_img_pose, self.model,self.a, batch, self.batch_size)
        # target_depth = torch.Tensor(target_depth).to(device)
        # print(target_depth)


        start_time = time.time()
        # AR start
        losses = []
        for i, particle in enumerate(particles):
            pose = torch.Tensor(particle).to(device)
		
            nerf_time = time.time() - start_time
            rgb, _ = get_loss(pose, self.model, self.a, batch, self.batch_size)
            rgb = torch.tensor(rgb).to(device)
            # depth = torch.tensor(depth).to(device)
            #print(depth)
    
            if photometric_loss == 'rgb':
                loss_rgb = img2mse(rgb, target_s)
                # loss_depth = img2mse(depth, depth_s)
                loss =loss_rgb 
                #print(loss_depth)
                
                
            else:
                # TODO throw an error
                print("DID NOT ENTER A VALID LOSS METRIC")
            losses.append(loss.item())
        return losses, nerf_time
		# AR end
    
    def visualize_nerf_image(self, nerf_pose):
        pose_dummy = torch.from_numpy(nerf_pose).cuda()
        with torch.no_grad():
            print(nerf_pose)
            rgb, disp, acc, _ = render(self.H, self.W, self.focal, chunk=self.chunk, c2w=pose_dummy[:3, :4], **self.render_kwargs)
            rgb = rgb.cpu().detach().numpy()
            rgb8 = to8b(rgb)
            ref = to8b(self.obs_img)
        plt.imshow(rgb8)
        plt.show()