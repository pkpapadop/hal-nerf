

FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04



# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

ENV DEBIAN_FRONTEND=noninteractive

ENV CUDA_HOME=/usr/local/cuda

RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    curl \
    git \
    lsb-release \
    gnupg2 \
    nano

RUN apt-get install -y python3 && \
    ln -s /usr/bin/python3 /usr/bin/python

RUN apt-get install -y python3-pip

# RUN add-apt-repository ppa:deadsnakes/ppa && \
#     apt-get update && \
#     apt-get install -y python3.8 python3.8-distutils python3.8-venv && \
#     update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8  && \
#     update-alternatives --set python3 /usr/bin/python3.98&& \
#     wget https://bootstrap.pypa.io/get-pip.py && \
#     python3.8 get-pip.py && \
#     rm get-pip.py


# RUN pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

#RUN TCNN_CUDA_ARCHITECTURES=86 pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

#ROS

RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -

RUN pip3 install catkin-tools

RUN apt-get update && apt-get install -y ros-noetic-desktop-full

RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc

RUN apt-get clean && rm -rf /var/lib/apt/lists/*

#LOC-NERF

RUN mkdir -p /root/catkin_ws/src && \
    cd /root/catkin_ws/ && \
    catkin init

RUN  cd ~/catkin_ws/src && \
    git clone https://github.com/MIT-SPARK/Loc-NeRF

RUN /bin/bash -c '. /opt/ros/noetic/setup.bash; cd /root/catkin_ws; catkin build'


RUN echo "source /root/catkin_ws/devel/setup.bash" >> /root/.bashrc

RUN cd ~/catkin_ws/src/Loc-NeRF && \
    apt-get update && apt-get install -y ros-noetic-cv-bridge && \
    sed '/cv_bridge/d' requirements.txt > requirements_filtered.txt && \
    sed -i '/rospy/d' requirements_filtered.txt && \
    sed -i '/rostopic/d' requirements_filtered.txt && \
    sed -i '/sensor_msgs/d' requirements_filtered.txt && \
    pip3 install -r requirements_filtered.txt 
    
#NERFSTUDIO

ARG CACHEBUSTER=none

RUN mkdir -p /root/catkin_ws/src/Loc-NeRF/src/nerfstudio && \
    git clone https://github.com/nerfstudio-project/nerfstudio.git /root/catkin_ws/src/Loc-NeRF/src/nerfstudio && \
    cd /root/catkin_ws/src/Loc-NeRF/src/nerfstudio && \
    git checkout tags/v0.3.4




RUN pip3 install --upgrade pip setuptools

WORKDIR /root/catkin_ws/src/Loc-NeRF

RUN apt-get update && apt-get install -y \
    build-essential \
    python3.9-dev

RUN cd /root/catkin_ws/src/Loc-NeRF/src/nerfstudio && pip3 install --ignore-installed -e .

RUN pip3 uninstall torch torchvision -y
RUN pip3 install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
#RUN pip3 install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

RUN TCNN_CUDA_ARCHITECTURES=86 pip3 install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

RUN pip3 install netifaces==0.11.0

RUN pip3 install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu117_pyt2010/download.html

WORKDIR /root/catkin_ws/src/Loc-NeRF/src

COPY src/loc_nerf/*.py ./

COPY src/nerfstudio /root/catkin_ws/src/Loc-NeRF/src/nerfstudio

COPY src/pose_regressor ./

COPY src/config_dfnet.txt ./

COPY src/*.py ./

# TODO : DEPTH MODULE

# RUN pip3 install huggingface_hub

# RUN pip3 install gradio_imageslider

# RUN pip3 install gradio==4.14.0

# RUN mkdir depth_anything

# RUN mkdir torchhub

# COPY src/DepthAnything/depth_anything /root/catkin_ws/src/Loc-NeRF/src/depth_anything

# COPY src/DepthAnything/torchhub /root/catkin_ws/src/Loc-NeRF/src/torchhub

# WORKDIR /root

CMD ["bash"]

