FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

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

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

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

RUN  mkdir -p /root/catkin_ws/src/Loc-NeRF/src/nerfstudio && \
     git clone https://github.com/nerfstudio-project/nerfstudio.git /root/catkin_ws/src/Loc-NeRF/src/nerfstudio


RUN pip3 install --upgrade pip setuptools

WORKDIR /root/catkin_ws/src/Loc-NeRF

RUN cd /root/catkin_ws/src/Loc-NeRF/src/nerfstudio && pip install --ignore-installed -e .


RUN pip3 install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

RUN TCNN_CUDA_ARCHITECTURES=86 pip3 install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

WORKDIR /root/catkin_ws/src/Loc-NeRF/src

COPY src/loc-nerf/*.py ./

COPY src/nerfstudio /root/catkin_ws/src/Loc-NeRF/src/nerfstudio

WORKDIR /root

CMD ["bash"]

