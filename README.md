# HAL-NeRF v2.0

Dockerized High Accuracy Localization application based on Loc-NeRF (https://arxiv.org/abs/2209.09050) and using Nerfacto model (https://docs.nerf.studio/nerfology/methods/nerfacto.html) as the base NeRF model. 

This pipeline contains two parts:  
 * Training the DFNet pose regressor (https://arxiv.org/abs/2204.00559)
 * Optimizing the prediction of the pose regressor with Loc-NeRF

## Prerequisites

Install docker daemon. Install Nvidia Docker toolkit.
 
When building your Docker image and installing the Tiny CUDA Neural Networks (tiny-cuda-nn) library, you need to set the TCNN_CUDA_ARCHITECTURES environment variable to match the CUDA architecture of your GPU. "RUN TCNN_CUDA_ARCHITECTURES=86 pip3 install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch " 

## Usage 
Colmap should be like this:

```
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
```

1) Go to the following directory: 
```bash
../hal_nerf/workspace
```
2) Create a folder named "colmap_output" with this structure:

```
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
```

3) Rename the folder that contains the pre-trained nerfacto model as "model" and move it inside the following directory:

```bash
../hal_nerf/workspace/colmap_output
```
In config.yml file of the model, you have to change the 'output_dir' as /root/colmap_output and the 'pipeline->dataparser->data' as /root/colmap_output 

4) Compose the image with this command:

```bash
docker build -t <your_image_name> .
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

In line 386 of colmap_to_mega_nerf.py, you can declare which images you want to use as testing samples

Using the --images_path parameter, you can specify the scale of the images you want to use



8) Second, train DFNET: 

```bash
python run_posenet.py --config config_dfnet.txt
```

In the config_dfnet.txt configuration file, you can control various pose regressor network training parameters. Notably, setting random_view_synthesis=True enables augmentation of your training dataset using a pre-trained nerfacto model. The parameter rvs_refresh_rate determines the number of epochs after which the dataset will be augmented. Additionally, rvs_trans specifies the uniform distribution range for translation component perturbations, while rvs_rotation specifies the range for rotation component perturbations. These settings allow for dynamic augmentation, enhancing the training process by introducing controlled perturbations at specified intervals



9) Now, you can run HAL-NeRF by running this command:

```bash
roslaunch locnerf navigate.launch parameter_file:=<param_file.yaml>
```

- Replace <param_file.yaml> with "hal_nerf.yaml" inside the cfg_experiment folder. The configuration files are the same with the locnerf pipeline except for the first eight args. Specifically, we added the following parameters:   
  1) position_error_threshold
  2) rotation_error_threshold 
  3) termination_mode    #  0: use position_error_threshold, 1: use rotation_error_threshold, 2: use position_error_threshold and rotation_error_threshold
  4) output_path    # the path in which the results will be saved inside the container.
  5) export_images    # If true, the experiment results also contain visual information.
  6) particles_random_initial_position    # initalization of particles' position
  7) particles_random_initial_rotation    # initialization of particles' rotation
  8) image_idx    # the index of the testing sample
  9) factor      # The downscale factor used must match the one chosen during the training of the pose regressor

10) If you want to visualize the experiment, activate rviz visualization. In another terminal, access the running container with this command:

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
