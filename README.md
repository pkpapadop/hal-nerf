# HAL-NeRF v2.0

Dockerized High Accuracy Localization application based on Loc-NeRF (https://arxiv.org/abs/2209.09050) and using Nerfacto model (https://docs.nerf.studio/nerfology/methods/nerfacto.html) as the base NeRF model. 

This pipeline contains two parts:  
 * Training the DFNet pose regressor (https://arxiv.org/abs/2204.00559)
 * Optimizing the prediction of the pose regressor with Loc-NeRF

## Prerequisites

Install docker daemon. Install the Nvidia Docker toolkit.
 
When building your Docker image and installing the Tiny CUDA Neural Networks (tiny-cuda-nn) library, you need to set the TCNN_CUDA_ARCHITECTURES environment variable to match the CUDA architecture of your GPU. "RUN TCNN_CUDA_ARCHITECTURES=86 pip3 install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch " 

## Usage 
Colmap should be like this:

colmap
├── images
├── images_2
├── images_4
├── images_8
└── sparse
    └── 0
        ├── images.txt
        ├── cameras.txt
        └── points3D.txt


1) Go to the following directory: 
```bash
../hal_nerf/workspace
```
2) Create a folder named "colmap_output" with this structure:

colmap_output
├── colmap
│   ├── images
│   ├── images_2
│   ├── images_4
│   ├── images_8
│   └── sparse
│       └── 0
│           ├── images.txt
│           ├── cameras.txt
│           └── points3D.txt
├── images
├── images_2
├── images_4
└── images_8

3) Rename the folder that contains the pre-trained nerfacto model as "model" and move it inside the following directory:

```bash
../hal_nerf/workspace/colmap_output
```
In config.yml file of the model, you have to change the 'output_dir' as /root/colmap_output and the 'pipeline->dataparser->data' as /root/colmap_output 

4) Compose the image with this command:

```bash
docker build -t your_image_name .
```

5) Change bash file permission with this command:

```bash
chmod +x run.bash
```
6) Run bash script (5 arguments):

```bash
./run.bash --container-name <your_container_name> --cfg-dir $PWD/workspace/cfg_experiment --image-name <your_image_name> --poses-dir $PWD/workspace/colmap_output --ckpt $PWD/workspace/weight.ckpt
```

7) Now you are inside the container. First, prepare the dataset for DFNet training:

```bash
python colmap_to_mega_nerf.py --model_path /root/colmap_output/colmap --images_path /root/colmap_output/images --output_path /root/outputiw
```

In line 386 you can declare which images you want to use as testing samples

8) Second, train DFNET: 

```bash
python run_posenet.py --config config_dfnet.txt
```

with config_dfnet.txt you can control some of the pose regressor network training parameters. Especially, with 'random_view_synthesis=True' you can augment your training dataset using pretrained nerfacto model. 'rvs_refresh_rate', 'rvs_trans' and 'rvs_rotation' are the parameters that control how many epochs the dataset will be augmented, the uniform distribution for translation component perturbation and the uniform distribution for rotation component perturbation accordingly.



9) Now, you can run HAL-NeRF by running this command:

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

10) If you want to visualize the experiment, you can activate rviz visualization. In another terminal, access the running container with this command:

```bash
docker exec -it your_container_name /bin/bash
```

11) Once you are inside the container, run:

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
