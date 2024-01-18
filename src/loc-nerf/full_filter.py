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
            transform_path = '/root/colmap_output/transforms.json'
            self.obs_img, self.obs_img_pose, self.W, self.H = input_dataset(transform_path, self.obs_img_num, self.factor)
            self.obs_img_pose = self.obs_img_pose.cpu().numpy()
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

        if nerf_params['export_images']: # TODO: add self.export_images to NeRF object
            vizualize(self.obs_img_pose[:3,:], self.model, self.a,0, nerf_params['output_path'])
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
        target_s = torch.Tensor(target_s).to(device)

        start_time = time.time()
        # AR start
        losses = []
        for i, particle in enumerate(particles):
            pose = torch.Tensor(particle).to(device)
		
            nerf_time = time.time() - start_time
            rgb = get_loss(pose, self.model, self.a, batch, self.batch_size)
            rgb = torch.tensor(rgb)
    
            if photometric_loss == 'rgb':
                loss = img2mse(rgb, target_s)
                
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
