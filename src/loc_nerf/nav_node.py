#!/usr/bin/env python

from turtle import pos

# AR start
import time 
import sys
import signal
# AR end
import rospy
import numpy as np
import gtsam
import cv2
import torch
import time
import glob
import matplotlib.pyplot as plt

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray, Pose

from scipy.spatial.transform import Rotation as R
from copy import copy

import locnerf
from full_filter import NeRF
from particle_filter import ParticleFilter
from utils import get_pose
from navigator_base import NavigatorBase
from nerfacto_loader import vizualize # AR
from scipy.spatial.transform import Rotation as R
import gtsam

torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Navigator(NavigatorBase):
    def __init__(self, img_num=None, dataset_name=None):
        # Base class handles loading most params.
        NavigatorBase.__init__(self, img_num=img_num, dataset_name=dataset_name)

        # Set initial distribution of particles.
        self.get_initial_distribution()

        self.br = CvBridge()

        # Set up publishers.
        self.particle_pub = rospy.Publisher("/particles", PoseArray, queue_size = 10)
        self.pose_pub = rospy.Publisher("/estimated_pose", Odometry, queue_size = 10)
        self.gt_pub = rospy.Publisher("/gt_pose", PoseArray, queue_size = 10)
        self.pp_pub = rospy.Publisher("/pp_pose", PoseArray, queue_size = 10)

        # Set up subscribers.
        # We don't need callbacks to compare against inerf.
        if not self.run_inerf_compare:
            self.image_sub = rospy.Subscriber(self.rgb_topic,Image,self.rgb_callback, queue_size=1, buff_size=2**24)
            if self.run_predicts:
                self.vio_sub = rospy.Subscriber(self.pose_topic, Odometry, self.vio_callback, queue_size = 10)

        # Show initial distribution of particles
        if self.plot_particles:
            self.visualize()

        if self.log_results:
            # If using a provided start we already have ground truth, so don't log redundant gt.
            if not self.use_logged_start:
                with open(self.log_directory + "/" + "gt_" + self.model_name + "_" + str(self.obs_img_num) + "_" + "poses.npy", 'wb') as f:
                    np.save(f, self.gt_pose)

            # Add initial pose estimate before first update step is run.
            if self.use_weighted_avg:
                position_est = self.filter.compute_weighted_position_average()
            else:
                position_est = self.filter.compute_simple_position_average()
            rot_est = self.filter.compute_simple_rotation_average()
            pose_est = gtsam.Pose3(rot_est, position_est).matrix()
            self.all_pose_est.append(pose_est)

    def get_initial_distribution(self):
        # NOTE for now assuming everything stays in NeRF coordinates (x right, y up, z inward)
        if self.run_inerf_compare:
            # for non-global loc mode, get random pose based on iNeRF evaluation method from their paper
            # sample random axis from unit sphere and then rotate by a random amount between [-40, 40] degrees
            # translate along each axis by a random amount between [-10, 10] cm
            rot_rand = self.particles_random_initial_rotation # AR
            if self.global_loc_mode:
                trans_rand = self.particles_random_initial_position # AR  
            else:
                trans_rand = 0.1
            
            # get random axis and angle for rotation
            x = np.random.rand()
            y = np.random.rand()
            z = np.random.rand()
            axis = np.array([x,y,z])
            axis = axis / np.linalg.norm(axis)
            angle = np.pi * np.random.uniform(low=-rot_rand, high=rot_rand) / 180.0
            euler = (gtsam.Rot3.AxisAngle(axis, angle)).ypr()

            # get random translation offset
            t_x = np.random.uniform(low=-trans_rand, high=trans_rand)
            t_y = np.random.uniform(low=-trans_rand, high=trans_rand)
            t_z = np.random.uniform(low=-trans_rand, high=trans_rand)

            # use initial random pose from previously saved log
            if self.use_logged_start:
                log_file = self.log_directory + "/" + "initial_pose_" + self.model_name + "_" + str(self.obs_img_num) + "_" + "poses.npy"
                start = np.load(log_file)
                print(start)
                euler[0], euler[1], euler[2], t_x, t_y, t_z = start

            # log initial random pose
            elif self.log_results:
                with open(self.log_directory + "/" + "initial_pose_" + self.model_name + "_" + str(self.obs_img_num) + "_" + "poses.npy", 'wb') as f:
                    np.save(f, np.array([euler[0], euler[1], euler[2], t_x, t_y, t_z]))

            if self.global_loc_mode:
                # 360 degree rotation distribution about yaw
                self.initial_particles_noise = np.random.uniform(np.array([-trans_rand, -trans_rand, -trans_rand, 0, -179, 0]), np.array([trans_rand, trans_rand, trans_rand, 0, 179, 0]), size = (self.num_particles, 6))
            else:
                self.initial_particles_noise = np.random.uniform(np.array([-trans_rand, -trans_rand, -trans_rand, 0, 0, 0]), np.array([trans_rand, trans_rand, trans_rand, 0, 0, 0]), size = (self.num_particles, 6))

            # center translation at randomly sampled position
            self.initial_particles_noise[:, 0] += t_x
            self.initial_particles_noise[:, 1] += t_y
            self.initial_particles_noise[:, 2] += t_z

            if self.global_loc_mode: # AR 
                for i in range(self.initial_particles_noise.shape[0]):
                    # rotate random 3 DOF rotation about initial random rotation for each particle
                    n1 = np.pi * np.random.uniform(low=-rot_rand, high=rot_rand) / 180.0
                    n2 = np.pi * np.random.uniform(low=-rot_rand, high=rot_rand) / 180.0
                    n3 = np.pi * np.random.uniform(low=-rot_rand, high=rot_rand) / 180.0
                    euler_particle = gtsam.Rot3.AxisAngle(axis, angle).retract(np.array([n1, n2, n3])).ypr()

                    # add rotation noise for initial particle distribution
                    self.initial_particles_noise[i,3] = euler_particle[0] * 180.0 / np.pi
                    self.initial_particles_noise[i,4] = euler_particle[1] * 180.0 / np.pi 
                    self.initial_particles_noise[i,5] = euler_particle[2] * 180.0 / np.pi  
        
        else:
            # get distribution of particles from user
            self.initial_particles_noise = np.random.uniform(np.array(
                [self.min_bounds['px'], self.min_bounds['py'], self.min_bounds['pz'], self.min_bounds['rz'], self.min_bounds['ry'], self.min_bounds['rx']]),
                np.array([self.max_bounds['px'], self.max_bounds['py'], self.max_bounds['pz'], self.max_bounds['rz'], self.max_bounds['ry'], self.max_bounds['rx']]),
                size = (self.num_particles, 6))

        self.initial_particles = self.set_initial_particles()
         
        self.filter = ParticleFilter(self.initial_particles)

    def vio_callback(self, msg):
        # extract rotation and position from msg
        quat = msg.pose.pose.orientation
        position = msg.pose.pose.position

        # rotate vins to be in nerf frame
        rx = self.R_bodyVins_camNerf['rx']
        ry = self.R_bodyVins_camNerf['ry']
        rz = self.R_bodyVins_camNerf['rz']
        T_bodyVins_camNerf = gtsam.Pose3(gtsam.Rot3.Ypr(rz, ry, rx), gtsam.Point3(0,0,0))
        T_wVins_camVins = gtsam.Pose3(gtsam.Rot3(quat.w, quat.x, quat.y, quat.z), gtsam.Point3(position.x, position.y, position.z))
        T_wVins_camNeRF = gtsam.Pose3(T_wVins_camVins.matrix() @ T_bodyVins_camNerf.matrix())

        if self.previous_vio_pose is not None:
            T_camNerft_camNerftp1 = gtsam.Pose3(self.previous_vio_pose.inverse().matrix() @ T_wVins_camNeRF.matrix())
            self.run_predict(T_camNerft_camNerftp1)

        # log pose for next transform computation
        self.previous_vio_pose = T_wVins_camNeRF

        # publish particles for rviz
        if self.plot_particles:
            self.visualize()
    
    def rgb_callback(self, msg):
        self.img_msg = msg
        
    def rgb_run(self, msg, get_rays_fn=None, render_full_image=False):
        print("processing image")
        start_time = time.time()
        self.rgb_input_count += 1

        # make copies to prevent mutations
        particles_position_before_update = np.copy(self.filter.particles['position'])
        particles_rotation_before_update = [gtsam.Rot3(i.matrix()) for i in self.filter.particles['rotation']]

        if self.use_convergence_protection:
            for i in range(self.number_convergence_particles):
                t_x = np.random.uniform(low=-self.convergence_noise, high=self.convergence_noise)
                t_y = np.random.uniform(low=-self.convergence_noise, high=self.convergence_noise)
                t_z = np.random.uniform(low=-self.convergence_noise, high=self.convergence_noise)
                # TODO this is not thread safe. have two lines because we need to both update
                # particles to check the loss and the actual locations of the particles
                self.filter.particles["position"][i] = self.filter.particles["position"][i] + np.array([t_x, t_y, t_z])
                particles_position_before_update[i] = particles_position_before_update[i] + np.array([t_x, t_y, t_z])

        if self.use_received_image:
            img = self.br.imgmsg_to_cv2(msg)
            # resize input image so it matches the scale that NeRF expects
            img = cv2.resize(self.br.imgmsg_to_cv2(msg), (int(self.nerf.W), int(self.nerf.H)))
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            self.nerf.obs_img = img
            show_true = self.view_debug_image_iteration != 0 and self.num_updates == self.view_debug_image_iteration-1
            self.nerf.get_poi_interest_regions(show_true, self.sampling_strategy)
            # plt.imshow(self.nerf.obs_img)
            # plt.show()

        total_nerf_time = 0

        if self.sampling_strategy == 'random':
            rand_inds = np.random.choice(self.nerf.coords.shape[0], size=self.nerf.batch_size, replace=False)
            batch = self.nerf.coords[rand_inds]

        loss_poses = []

        #  ###AR#####
        
        # num_particles = 30
        # num_dimensions = 6  # Assuming 3 for position (x, y, z) and 3 for orientation (pitch, yaw, roll)
        # max_iterations = 100
        # cognitive_weight = 0.5
        # social_weight = 0.3
        # inertia_weight = 0.9
        # min_inertia = 0.4
        # max_inertia = 0.9
        # learning_rate = 0.01

    
        # velocities = np.zeros_like(particles_position_before_update)
        # personal_best_positions = np.copy(particles_position_before_update)
        # personal_best_scores = np.full(self.num_particles, np.inf)
        # global_best_position = None
        # global_best_score = np.inf


        # target_s = self.nerf.obs_img_noised[batch[:, 1], batch[:, 0]] # TODO check ordering here
        # target_s = torch.Tensor(target_s).to(device)


        # for iteration in range(max_iterations):
        #     for index, particle in enumerate(particles_position_before_update):
        #         pose = torch.Tensor(particle).to(device)
        #         rgb = get_loss(pose, self.nerf.model, self.nerf.a, batch, self.nerf.batch_size)
        #         rgb = torch.tensor(rgb)
        #         loss = img2mse(rgb, target_s)

        #         if loss < personal_best_scores[i]:
        #             personal_best_scores[i] = loss
        #             personal_best_positions[i] = particle

        #         if loss < global_best_score:
        #             global_best_score = loss
        #             global_best_position = particle

            
        #     for i, (particle, velocity) in enumerate(zip(particles, velocities)):
        # # Gradient-informed velocity update
        #         #gradient_influence = learning_rate * gradient
        #         inertia = inertia_weight * velocity
        #         cognitive = cognitive_weight * np.random.rand(num_dimensions) * (personal_best_positions[i] - particle)
        #         social = social_weight * np.random.rand(num_dimensions) * (global_best_position - particle)
        #         new_velocity = inertia + cognitive + social 
        
        # # Update the particle's velocity and position
        #         velocities[i] = new_velocity
        #         particles[i] += new_velocity






        for index, particle in enumerate(particles_position_before_update):
           loss_pose = np.zeros((4,4))
           rot = particles_rotation_before_update[index]
           loss_pose[0:3, 0:3] = rot.matrix()
           loss_pose[0:3,3] = particle[0:3]
           loss_pose[3,3] = 1.0
           loss_poses.append(loss_pose)
        losses, nerf_time = self.nerf.get_loss(loss_poses, batch, self.photometric_loss)
   
        for index, particle in enumerate(particles_position_before_update):
            
           self.filter.weights[index] = (1/losses[index])**2
        total_nerf_time += nerf_time


        self.filter.update()
        self.num_updates += 1
        print("UPDATE STEP NUMBER", self.num_updates, "RAN")
        print("number particles:", self.num_particles)
        # AR start
        with open(self.output_path + '/results.txt', 'a') as file:
          file.write(f"{self.num_updates},{self.num_particles}")
        # AR end
        
        if self.use_refining: # TODO make it where you can reduce number of particles without using refining
            self.check_refine_gate()

        if self.use_weighted_avg:
            avg_pose = self.filter.compute_weighted_position_average()
        else:
            avg_pose = self.filter.compute_simple_position_average()

        avg_rot = self.filter.compute_simple_rotation_average()
        self.nerf_pose = gtsam.Pose3(avg_rot, gtsam.Point3(avg_pose[0], avg_pose[1], avg_pose[2])).matrix()

        if self.plot_particles:
            self.visualize()
            
        # TODO add ability to render several frames
        if self.view_debug_image_iteration != 0 and (self.num_updates == self.view_debug_image_iteration):
            self.nerf.visualize_nerf_image(self.nerf_pose)

        if not self.use_received_image:
            if self.use_weighted_avg:
                position_error = np.linalg.norm(self.gt_pose[0:3,3] - self.filter.compute_weighted_position_average()) # AR
                print("average position of all particles: ", self.filter.compute_weighted_position_average())
                print("position error: ", np.linalg.norm(self.gt_pose[0:3,3] - self.filter.compute_weighted_position_average()))
                
            else:
                position_error = np.linalg.norm(self.gt_pose[0:3,3] - self.filter.compute_simple_position_average()) # AR
                print("average position of all particles: ", self.filter.compute_simple_position_average())
                print("position error: ", np.linalg.norm(self.gt_pose[0:3,3] - self.filter.compute_simple_position_average()))

        if self.use_weighted_avg:
            position_est = self.filter.compute_weighted_position_average()
        else:
            position_est = self.filter.compute_simple_position_average()
        rot_est = self.filter.compute_simple_rotation_average()
        pose_est = gtsam.Pose3(rot_est, position_est).matrix()
        # AR start
        if self.export_images:
            vizualize(pose_est[:3,:], self.nerf.model, self.nerf.a, self.num_updates, self.output_path)
        # AR end
        
        if self.log_results:
            self.all_pose_est.append(pose_est)
        
        if not self.run_inerf_compare:
            img_timestamp = msg.header.stamp
            self.publish_pose_est(pose_est, img_timestamp)
        else:
            self.publish_pose_est(pose_est)
    
        update_time = time.time() - start_time
        print("forward passes took:", total_nerf_time, "out of total", update_time, "for update step")

        if not self.run_predicts:
            self.filter.predict_no_motion(self.px_noise, self.py_noise, self.pz_noise, self.rot_x_noise, self.rot_y_noise, self.rot_z_noise) #  used if you want to localize a static image
        
        # return is just for logging
        return pose_est
    
    def check_if_position_error_good(self, return_error = False, threshold=0.05, write=False): # AR
        """
        check if position error is less than 5cm, or return the'/home/asterisreppas/catkin_ws/src/Loc-NeRF/results.txt' error if return_error is True
        """
        acceptable_error = threshold # AR
        if self.use_weighted_avg:
            error = np.linalg.norm(self.gt_pose[0:3,3] - self.filter.compute_weighted_position_average())
            # AR start
            if write:
                with open(self.output_path + '/results.txt', 'a') as file:
                    file.write(f",{error}")
            # AR end
            if return_error:
                return error
            return error < acceptable_error
        else:
            error = np.linalg.norm(self.gt_pose[0:3,3] - self.filter.compute_simple_position_average())
            # AR start
            with open(self.output_path + '/results.txt', 'a') as file:
                file.write(f",{error}")
            # AR end
            if return_error:
                return error
            return error < acceptable_error

    def check_if_rotation_error_good(self, return_error = False, threshold=2.0,  write=False): # AR
        """
        check if rotation error is less than 5 degrees, or return the error if return_error is True
        """
        acceptable_error = threshold  # Arbitrary threshold value for example purposes
        average_rot_t = (self.filter.compute_simple_rotation_average()).transpose()

        # Enforce normalization on the average rotation matrix
        U, _, Vt = np.linalg.svd(average_rot_t, full_matrices=False)
        average_rot_t_normalized = np.dot(U, Vt)

        # Now, ensure the ground truth rotation matrix is also normalized
        U_gt, _, Vt_gt = np.linalg.svd(self.gt_pose[0:3, 0:3], full_matrices=False)
        gt_pose_normalized = np.dot(U_gt, Vt_gt)

        # Compute the relative rotation matrix using normalized matrices
        r_ab = average_rot_t_normalized @ gt_pose_normalized

        # Continue with your existing calculation
        cos_angle = (np.trace(r_ab) - 1) / 2
        # Since we've normalized the input matrices, cos_angle should naturally be within [-1, 1],
        # but we'll still handle potential numerical precision issues without hard clamping.
        cos_angle_adjusted = min(1.0, max(-1.0, cos_angle))

        rot_error = np.rad2deg(np.arccos(cos_angle_adjusted))
        print("rotation error: ", rot_error)
        # AR start
        if write:
            with open(self.output_path + '/results.txt', 'a') as file:
                file.write(f",{rot_error}\n")
        # AR end
        if return_error:
            return rot_error
        return abs(rot_error) < acceptable_error

    def run_predict(self, delta_pose):
        self.filter.predict_with_delta_pose(delta_pose, self.px_noise, self.py_noise, self.pz_noise, self.rot_x_noise, self.rot_y_noise, self.rot_z_noise)

        if self.plot_particles:
            self.visualize()
    #AR start

    # def set_initial_particles(self):
    #     initial_positions = np.zeros((self.num_particles, 3))
    #     rots = []
    #     for index, particle in enumerate(self.initial_particles_noise):
    #         x = particle[0]
    #         y = particle[1]
    #         #z = particle[2] # AR
    #         z = 0
    #         #y = 0 # AR
    #         #phi = 0 # AR
    #         theta = 0
    #         psi = 0
    #         phi = particle[3]
    #         #theta = particle[4] # AR
    #         #psi = particle[5] # AR

    #         particle_pose = get_pose(phi, theta, psi, x, y, z, self.nerf.predicted_pose, self.center_about_true_pose)
            
    #         # set positions
    #         initial_positions[index,:] = [particle_pose[0,3], particle_pose[1,3], particle_pose[2,3]]
    #         # set orientations
    #         rots.append(gtsam.Rot3(particle_pose[0:3,0:3]))
    #         # print(initial_particles)
    #     return {'position':initial_positions, 'rotation':np.array(rots)}




    def set_initial_particles(self):
        np.random.seed(42)  # For reproducibility
        rots = []
        gt_translation = self.nerf.obs_img_pose[:3, 3]
        gt_rotation_matrix = self.nerf.obs_img_pose[:3, :3]
        print('aaaaaaaaaaaaaaaaaaaaaaaa')
        print(gt_rotation_matrix)

        # Parameters for uniform initialization within a disk
        disk_radius = self.particles_random_initial_position  # Radius of the disk
        num_particles = self.num_particles

        # Initialize positions uniformly within a disk
        angles = np.random.uniform(low=0, high=2*np.pi, size=num_particles)
        radii = np.sqrt(np.random.uniform(low=0, high=disk_radius**2, size=num_particles))

        positions_x = gt_translation[0] + radii * np.cos(angles)
        positions_y = gt_translation[1] + radii * np.sin(angles)
        positions = np.stack((positions_x, positions_y, np.full(num_particles, gt_translation[2])), axis=-1)

        # Get the Euler angles from the ground truth rotation matrix
        gt_euler_angles = R.from_matrix(gt_rotation_matrix).as_euler('xyz', degrees=True)

        # Define uniform distribution ranges for rotation angles. 
        # These ranges should be determined based on your application's needs.
        angle_range = self.particles_random_initial_rotation  # Example range of +/- 30 degrees for uniform distribution around the GT
        low_angles = gt_euler_angles - angle_range
        high_angles = gt_euler_angles + angle_range

        # Initialize rotation angles with uniform distribution
        random_rotation_angles = np.random.uniform(low=low_angles, high=high_angles, size=(num_particles, 3))

        # Set pitch to zero to keep rotation in the XY plane
        random_rotation_angles[:, 1] = 0

        # Convert Euler angles to rotation matrices and create gtsam.Rot3 objects
        for angles in random_rotation_angles:
            rotation_obj = R.from_euler('xyz', angles, degrees=True)
            rotation_matrix = rotation_obj.as_matrix()
            rot3_object = gtsam.Rot3(rotation_matrix)
            rots.append(rot3_object)  # Append the gtsam.Rot3 object to the list of rotations

        return {'position': positions, 'rotation': np.array(rots)}


        #AR end

    def set_noise(self, scale):
        self.px_noise = rospy.get_param('px_noise') / scale
        self.py_noise = rospy.get_param('py_noise') / scale
        self.pz_noise = rospy.get_param('pz_noise') / scale
        self.rot_x_noise = rospy.get_param('rot_x_noise') / scale
        self.rot_y_noise = rospy.get_param('rot_y_noise') / scale
        self.rot_z_noise = rospy.get_param('rot_z_noise') / scale


    def check_refine_gate(self):

        # get standard deviation of particle position
        sd_xyz = np.std(self.filter.particles['position'], axis=0)
        norm_std = np.linalg.norm(sd_xyz)
        refining_used = False
        print("sd_xyz:", sd_xyz)
        print("norm sd_xyz:", np.linalg.norm(sd_xyz))
        # AR start
        norm_sd = np.linalg.norm(sd_xyz)
        with open(self.output_path + '/results.txt', 'a') as file:
            file.write(f",{norm_std}")
        # AR end

        if norm_std < self.alpha_super_refine:
            print("SUPER REFINE MODE ON")
            # reduce original noise by a factor of 4
            self.set_noise(scale = 4.0)
            refining_used = True
        elif norm_std < self.alpha_refine:
            print("REFINE MODE ON")
            # reduce original noise by a factor of 2
            self.set_noise(scale = 2.0)
            refining_used = True
        else:
            # reset noise to original value
            self.set_noise(scale = 1.0)
        
        if refining_used and self.use_particle_reduction:
            self.filter.reduce_num_particles(self.min_number_particles)
            self.num_particles = self.min_number_particles

    def publish_pose_est(self, pose_est_gtsam, img_timestamp = None):
        pose_est = Odometry()
        pose_est.header.frame_id = "world"

        # if we don't run on rosbag data then we don't have timestamps
        if img_timestamp is not None:
            pose_est.header.stamp = img_timestamp

        pose_est_gtsam = gtsam.Pose3(pose_est_gtsam)
        position_est = pose_est_gtsam.translation()
        rot_est = pose_est_gtsam.rotation().toQuaternion() # AR TODO: why

        # populate msg with pose information
        pose_est.pose.pose.position.x = position_est[0]
        pose_est.pose.pose.position.y = position_est[1]
        pose_est.pose.pose.position.z = position_est[2]
        # AR start TODO: why
        pose_est.pose.pose.orientation.w = rot_est.w()
        pose_est.pose.pose.orientation.x = rot_est.x()
        pose_est.pose.pose.orientation.y = rot_est.y()
        pose_est.pose.pose.orientation.z = rot_est.z()
        # AR end
        # print(pose_est_gtsam.rotation().ypr())

        # publish pose
        self.pose_pub.publish(pose_est)

    def visualize(self):
        # publish pose array of particles' poses
        poses = []
        R_nerf_body = gtsam.Rot3.Rx(-np.pi/2)
        for index, particle in enumerate(self.filter.particles['position']): 
            p = Pose()
            p.position.x = particle[0]
            p.position.y = particle[1]
            p.position.z = particle[2]
            # print(particle[3],particle[4],particle[5])
            rot = self.filter.particles['rotation'][index]
            # AR start
            orient = rot.toQuaternion()
            p.orientation.w = orient.w()
            p.orientation.x = orient.x()
            p.orientation.y = orient.y()
            p.orientation.z = orient.z()
            # AR end
            poses.append(p)
            
        pa = PoseArray()
        pa.poses = poses
        pa.header.frame_id = "world"
        self.particle_pub.publish(pa)

        # if we have a ground truth pose then publish it
        if not self.use_received_image or self.gt_pose is not None:
            gt_array = PoseArray()
            gt = Pose()
            # AR start
            gt_rot = gtsam.Rot3(self.gt_pose[0:3,0:3]).toQuaternion()
            gt.orientation.w = gt_rot.w()
            gt.orientation.x = gt_rot.x()
            gt.orientation.y = gt_rot.y()
            gt.orientation.z = gt_rot.z()
            # AR end
            gt.position.x = self.gt_pose[0,3]
            gt.position.y = self.gt_pose[1,3]
            gt.position.z = self.gt_pose[2,3]
            gt_array.poses = [gt]
            gt_array.header.frame_id = "world"
            self.gt_pub.publish(gt_array)

        
        pp_array = PoseArray()
        pp = Pose()
            #AR start
        pp_rot = gtsam.Rot3(self.nerf.predicted_pose[0:3,0:3]).toQuaternion()
        pp.orientation.w = pp_rot.w()
        pp.orientation.x = pp_rot.x()
        pp.orientation.y = pp_rot.y()
        pp.orientation.z = pp_rot.z()
            
        pp.position.x = self.nerf.predicted_pose[0,3]
        pp.position.y = self.nerf.predicted_pose[1,3]
        pp.position.z = self.nerf.predicted_pose[2,3]
        pp_array.poses = [pp]
        pp_array.header.frame_id = "world"
        self.pp_pub.publish(pp_array)
    
    def average_arrays(axis_list):
        """
        average arrays of different size
        adapted from https://stackoverflow.com/questions/49037902/how-to-interpolate-a-line-between-two-other-lines-in-python/49041142#49041142

        axis_list = [forward_passes_all, accuracy]
        """
        min_max_xs = [(min(axis), max(axis)) for axis in axis_list[0]]

        new_axis_xs = [np.linspace(min_x, max_x, 100) for min_x, max_x in min_max_xs]
        new_axis_ys = []
        for i in range(len(axis_list[0])):
            new_axis_ys.append(np.interp(new_axis_xs[i], axis_list[0][i], axis_list[1][i]))

        midx = [np.mean([new_axis_xs[axis_idx][i] for axis_idx in range(len(axis_list[0]))])/1000.0 for i in range(100)]
        midy = [np.mean([new_axis_ys[axis_idx][i] for axis_idx in range(len(axis_list[0]))]) for i in range(100)]

        plt.plot(midx, midy)
        plt.xlabel("number of forward passes (in thousands)")
        plt.grid()
        plt.show()

# AR start
def next_it(mode, pos_flag, rot_flag, forward_passes_flag):
    if mode == 0: # Terminate based on position error 
        return pos_flag and forward_passes_flag
    elif mode == 1: # Terminate based on rotation error
        return rot_flag and forward_passes_flag
    elif mode == 2: # Terminate based on position and rotation error
        return (pos_flag or rot_flag) 

    
    return forward_passes_flag
# AR end

if __name__ == "__main__":
    rospy.init_node("nav_node")
    np.random.seed(int(time.time()))

    run_inerf_compare = rospy.get_param("run_inerf_compare")
    use_logged_start = rospy.get_param("use_logged_start")
    log_directory = rospy.get_param("log_directory")

    if run_inerf_compare:
        num_starts_per_dataset = 1 # TODO make this a param
        # datasets = ['fern', 'horns', 'fortress', 'room'] # TODO make this a param
        datasets = ['ergostasio'] # AR TODO: make this a param

        total_position_error_good = []
        total_rotation_error_good = []
        total_num_forward_passes = []
        for dataset_index, dataset_name in enumerate(datasets):
            print("Starting iNeRF Style Test on Dataset: ", dataset_name)
            if use_logged_start:
                start_pose_files = glob.glob(log_directory + "/initial_pose_" + dataset_name + '_' +'*')

            # only use an image number once per dataset
            used_img_nums = set()
            for i in range(num_starts_per_dataset):
                if not use_logged_start:
                    img_num = np.random.randint(low=0, high=300) # AR # TODO can increase the range of images
                    while img_num in used_img_nums:
                        img_num = np.random.randint(low=0, high=300) # AR
                    used_img_nums.add(img_num)
                
                else:
                    start_file = start_pose_files[i]
                    #img_num = int(start_file.split('_')[5])
                    img_num = rospy.get_param('image_idx')
                    
                    
                img_num = rospy.get_param('image_idx')
                mcl_local = Navigator(img_num, dataset_name)
                print()
                print("Using Image Number:", mcl_local.obs_img_num)
                print("Test", i+1, "out of", num_starts_per_dataset)

                num_forward_passes_per_iteration = [0]
                position_error_good = []
                rotation_error_good = []
                try:
                    start_exp_time= time.time() # AR
                    flag = True # AR
                    ii=0  
                    while flag: # AR
                        forward_passes_flag = num_forward_passes_per_iteration[-1] < mcl_local.forward_passes_limit # AR
                        print()
                        print("forward pass limit, current number forward passes:", mcl_local.forward_passes_limit, num_forward_passes_per_iteration[-1])

                        position_error_good.append(int(mcl_local.check_if_position_error_good(threshold=mcl_local.position_error_threshold))) 
                        rotation_error_good.append(int(mcl_local.check_if_rotation_error_good(threshold=mcl_local.rotation_error_threshold)))

                        if ii != 0:
                            mcl_local.rgb_run('temp')
                            num_forward_passes_per_iteration.append(num_forward_passes_per_iteration[ii-1] + mcl_local.num_particles * (mcl_local.course_samples + mcl_local.fine_samples) * mcl_local.batch_size)
                        
                        # AR start
                        if ii == 0:
                            flag = next_it(mode=mcl_local.termination_mode, 
                                       pos_flag=not(mcl_local.check_if_position_error_good(threshold=mcl_local.position_error_threshold)), 
                                       rot_flag=not(mcl_local.check_if_rotation_error_good(threshold=mcl_local.rotation_error_threshold)),
                                       forward_passes_flag=forward_passes_flag)
                        else:
                            flag = next_it(mode=mcl_local.termination_mode, 
                                       pos_flag=not(mcl_local.check_if_position_error_good(threshold=mcl_local.position_error_threshold, write=True)), 
                                       rot_flag=not(mcl_local.check_if_rotation_error_good(threshold=mcl_local.rotation_error_threshold, write=True)),
                                       forward_passes_flag=forward_passes_flag)
                        ii += 1
                except Exception as e:
                    print(f"An error occurred: {e}")
                finally:   
                    end_exp_time = time.time()
                    timez = end_exp_time - start_exp_time
                    with open(mcl_local.output_path + '/results.txt', 'a') as file:
                        file.write(f"\nTime,{timez}\n")
                        file.write(f"\nMode,{mcl_local.termination_mode}\n")
                        # AR end


                if mcl_local.log_results:
                    with open(mcl_local.log_directory + "/" + "mocnerf_" + mcl_local.log_prefix + "_" + mcl_local.model_name + "_" + str(mcl_local.obs_img_num) + "_" + "poses.npy", 'wb') as f:
                        np.save(f, np.array(mcl_local.all_pose_est))
                    with open(mcl_local.log_directory + "/" + "mocnerf_" + mcl_local.log_prefix + "_" + mcl_local.model_name + "_" + str(mcl_local.obs_img_num) + "_" + "forward_passes.npy", 'wb') as f:
                        np.save(f, np.array(num_forward_passes_per_iteration))

                total_num_forward_passes.append(num_forward_passes_per_iteration)
                total_position_error_good.append(position_error_good)
                total_rotation_error_good.append(rotation_error_good)
        
        #average_arrays([total_num_forward_passes, total_position_error_good])
        #average_arrays([total_num_forward_passes, total_rotation_error_good])

    # run normal live ROS mode
    else:
        mcl = Navigator()      
        while not rospy.is_shutdown():
            if mcl.img_msg is not None:
                mcl.rgb_run(mcl.img_msg)
                mcl.img_msg = None # TODO not thread safe