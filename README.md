# HAL-NeRF

Dockerized High Accuracy Localization application based on Loc-NeRF paper and using Nerfacto model as the base NeRF model. 

# Usage
1) Navigate to the folder directory:

       cd hal-nerf 

2) Compose the image with this command:

       docker build -t <your_image_name> .

3) Rename your colmap output file as "colmap_output" and move it inside the following directory:

       workspace

4) Rename your .ckpt file (pretrained nerfacto model) as "weight.ckpt" and move it inside the following directory:

       workspace

5) Change bash file permission with this command:

       chmod +x run.bash

6) Run bash script (5 arguments):

       ./run.bash --container-name <your_container_name> --cfg-dir $PWD/workspace/cfg_experiment --image-name <your_image
       _name> --poses-dir $PWD/workspace/colmap_output --ckpt $PWD/workspace/weight.ckpt

7) Now that you have created the container, you can run the experiment by running this command:

       roslaunch locnerf navigate.launch parameter_file:=<param_file.yaml>

- Replace <param_file.yaml> with "llff_global.yaml" inside the cfg_experiment folder. The configuration files are the same with the locnerf pipeline except the first eight args. Specifically, we added the following parameters:   
  1) position_error_threshold
  2) rotation_error_threshold 
  3) termination_mode    #  0: use position_error_threshold, 1: use rotation_error_threshold, 2: use position_error_threshold and rotation_error_threshold
  4) output_path    # the path in which the results will be saved inside the container.
  5) export_images    # if is true, the experinment results contain also visual information.
  6) particles_random_initial_position    # initalization of particles' position
  7) particles_random_initial_rotation    # initialization of particles' rotation
  8) image_idx    # the ground truth image. We trying to find the pose of the camera when this image was taken. It is used only for visualization purposes in order to compare it with the predicted result.

8) If you want to visualize the experiment, you can activate rviz vizualization. In other terminal, access the running container with this command:

       docker exec -it your_container_name /bin/bash

9) Once you are inside the container, run:

        rviz -d $(rospack find locnerf)/rviz/rviz.rviz 
