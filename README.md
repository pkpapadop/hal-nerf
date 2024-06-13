# HAL-NeRF v2.0

Dockerized High Accuracy Localization application based on Loc-NeRF (https://arxiv.org/abs/2209.09050) and using Nerfacto model (https://docs.nerf.studio/nerfology/methods/nerfacto.html) as the base NeRF model. 

This pipeline contains two parts:  
 * Training the DFNet pose regressor (https://arxiv.org/abs/2204.00559)
 * Optimizing the prediction of the pose regressor with Loc-NeRF 

## Usage 

1) Rename your colmap output file as "colmap_output" and move it inside the following directory:

```bash
../hal_nerf/workspace
```

2) Rename your .ckpt file (pre-trained nerfacto model) as "weight.ckpt" and move it inside the following directory:

```bash
../hal_nerf/workspace
```

3) Compose the image with this command:

```bash
docker build -t your_image_name .
```

4) Change bash file permission with this command:

```bash
chmod +x run.bash
```
5) Run bash script (5 arguments):

```bash
./run.bash --container-name <your_container_name> --cfg-dir $PWD/workspace/cfg_experiment --image-name <your_image_name> --poses-dir $PWD/workspace/colmap_output --ckpt $PWD/workspace/weight.ckpt
```

6) PART A. Now you are inside the container. First, prepare the dataset for DFNet training:

```bash
python colmap_to_mega_nerf.py --model_path /root/colmap_output/colmap --images_path /root/colmap_output/images --output_path /root/outputiw
```

7) Second, train DFNET: 

```bash
python run_posenet.py --config config_dfnet.txt
```

with config_dfnet.txt you can control some of the network training parameters


8) PART B. Now, you can run HAL-NeRF by running this command:

```bash
roslaunch locnerf navigate.launch parameter_file:=<param_file.yaml>
```

- Replace <param_file.yaml> with "hal_nerf.yaml" inside the cfg_experiment folder. The configuration files are the same with the locnerf pipeline except for the first eight args. Specifically, we added the following parameters:   
  1) position_error_threshold
  2) rotation_error_threshold 
  3) termination_mode    #  0: use position_error_threshold, 1: use rotation_error_threshold, 2: use position_error_threshold and rotation_error_threshold
  4) output_path    # the path in which the results will be saved inside the container.
  5) export_images    # If true, the experiment results contain also visual information.
  6) particles_random_initial_position around DFNet pose prediction    # initalization of particles' position
  7) particles_random_initial_rotation around DFNet psoe prediction    # initialization of particles' rotation
  8) image_idx    # the ground truth image. We tried to find the pose of the camera when this image was taken. It is used only for visualization purposes to compare it with the predicted result.

9) If you want to visualize the experiment, you can activate rviz visualization. In another terminal, access the running container with this command:

```bash
docker exec -it your_container_name /bin/bash
```

10) Once you are inside the container, run:

```bash
rviz -d $(rospack find locnerf)/rviz/rviz.rviz 
```

## Requirements
Tested on a system with:
- GPU: NVIDIA Geforce RTX 3060
- CPU: 12th Gen Intel® Core™ i7-12700F
- RAM: Memory 32 GB
- OS:  Ubuntu 22.04.3 LTS

# License
This project is licensed under the [MIT License]().
