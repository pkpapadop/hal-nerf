import os
import os.path as osp
import numpy as np
import torch
import cv2
from torchvision import transforms
from PIL import Image
from torch.utils import data

def load_image(path):
    return Image.open(path).convert('RGB')

class mega_nerf_data(data.Dataset):
    def __init__(self, scene, data_path, mode, transform=None, target_transform=None, seed=7, df=2., trainskip=1, testskip=1, hwf=[480, 960, 480.], ret_idx=False, fix_idx=False, ret_hist=False, hist_bin=10):
        """
        :param scene: scene name ['chess', 'pumpkin', ...]
        :param data_path: root 7scenes data directory.
        Usually '../data/deepslam_data/7Scenes'
        :param mode: 'train', 'val', or 'test'
        :param transform: transform to apply to the images
        :param target_transform: transform to apply to the poses
        :param df: downscale factor
        :param trainskip: due to 7scenes are so big, now can use less training sets # of trainset = 1/trainskip
        :param testskip: skip part of testset, # of testset = 1/testskip
        :param hwf: H,W,Focal from COLMAP
        """

        self.transform = transform
        self.target_transform = target_transform
        self.df = df

        self.H, self.W, self.focal = hwf
        self.H = int(self.H)
        self.W = int(self.W)
        np.random.seed(seed)

        self.mode = mode
        self.ret_idx = ret_idx
        self.fix_idx = fix_idx
        self.ret_hist = ret_hist
        self.hist_bin = hist_bin  # histogram bin size

        if self.mode == 'train':
            root_dir = osp.join(data_path, scene) + '/train'
        elif self.mode == 'val':
            root_dir = osp.join(data_path, scene) + '/val'
        elif self.mode == 'test':
            root_dir = osp.join(data_path, scene) + '/test'
        else:
            raise ValueError("Mode must be 'train', 'val', or 'test'")

        rgb_dir = root_dir + '/rgbs/'
        pose_dir = root_dir + '/metadata/'

        # collect poses and image names
        self.rgb_files = os.listdir(rgb_dir)
        self.rgb_files = [rgb_dir + f for f in self.rgb_files]
        self.rgb_files.sort()

        self.pose_files = os.listdir(pose_dir)
        self.pose_files = [pose_dir + f for f in self.pose_files]
        self.pose_files.sort()

        if len(self.rgb_files) != len(self.pose_files):
            raise Exception('RGB file count does not match pose file count!')

        # trainskip and testskip
        frame_idx = np.arange(len(self.rgb_files))
        if self.mode == 'train' and trainskip > 1:
            frame_idx = frame_idx[::trainskip]
        elif self.mode in ['val', 'test'] and testskip > 1:
            frame_idx = frame_idx[::testskip]
        self.gt_idx = frame_idx

        self.rgb_files = [self.rgb_files[i] for i in frame_idx]
        self.pose_files = [self.pose_files[i] for i in frame_idx]

        if len(self.rgb_files) != len(self.pose_files):
            raise Exception('RGB file count does not match pose file count!')

        # read poses
        poses = []
        for i in range(len(self.pose_files)):
            pose = torch.load(self.pose_files[i])
            ### AR ###
            c2w_tensor = pose['c2w'].detach().cpu()
            c2w = c2w_tensor.numpy()
            ### AR ###
            zero = [[0, 0, 0, 1]]
            pose = np.r_[c2w, zero]
            poses.append(pose)
        poses = np.array(poses)  # [N, 4, 4]
        self.poses = poses[:, :3, :4].reshape(poses.shape[0], 12)
        # debug read one img and get the shape of the img
        img = load_image(self.rgb_files[0])
        img_np = (np.array(img) / 255.).astype(np.float32)

        self.H, self.W = img_np.shape[:2]
        if self.df != 1.:
            self.H = int(self.H // self.df)
            self.W = int(self.W // self.df)
            self.focal = self.focal / self.df

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, index):
        img = load_image(self.rgb_files[index])
        pose = self.poses[index]
        if self.df != 1.:
            img_np = (np.array(img) / 255.).astype(np.float32)
            dims = (self.W, self.H)
            img_half_res = cv2.resize(img_np, dims, interpolation=cv2.INTER_AREA)  # (H, W, 3)
            img = img_half_res

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            pose = self.target_transform(pose)

        if self.ret_idx:
            if self.mode == 'train' and not self.fix_idx:
                return img, pose, index
            else:
                return img, pose, 0

        return img, pose
