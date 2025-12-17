FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-lc"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 curl lsb-release ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" \
    > /etc/apt/sources.list.d/ros1-latest.list && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \
    apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-desktop-full \
    python3-rosdep python3-rosinstall python3-rosinstall-generator \
    python3-wstool python3-catkin-tools \
    && rm -rf /var/lib/apt/lists/*

RUN rosdep init && rosdep update
RUN echo "source /opt/ros/noetic/setup.bash" >> /etc/bash.bashrc

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git pkg-config \
    libboost-all-dev libflann-dev libusb-1.0-0-dev \
    libqhull-dev libvtk7-dev \
    python3 python3-pip python3-dev python3-distutils python3-tk \
    ffmpeg gdb \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch 3.4.0 https://gitlab.com/libeigen/eigen.git /tmp/eigen && \
    cmake -S /tmp/eigen -B /tmp/eigen/build && \
    cmake --build /tmp/eigen/build -j$(nproc) && \
    cmake --install /tmp/eigen/build && \
    rm -rf /tmp/eigen && ldconfig

RUN git clone --depth 1 --branch pcl-1.15.0 https://github.com/PointCloudLibrary/pcl.git /tmp/pcl && \
    cmake -S /tmp/pcl -B /tmp/pcl/build \
      -DWITH_CUDA=OFF \
      -DBUILD_gpu=OFF \
      -DBUILD_gpu_containers=OFF \
      -DBUILD_gpu_features=OFF \
      -DBUILD_gpu_octree=OFF \
      -DBUILD_gpu_segmentation=OFF \
      -DBUILD_gpu_surface=OFF \
      -DBUILD_gpu_tracking=OFF && \
    cmake --build /tmp/pcl/build -j$(nproc) && \
    cmake --install /tmp/pcl/build && \
    rm -rf /tmp/pcl && ldconfig
    
RUN python3 -m pip install --no-cache-dir \
    matplotlib numpy scikit-learn open3d

RUN mkdir -p /root/FLASH
WORKDIR /root

ENV CMAKE_PREFIX_PATH=/usr/local:$CMAKE_PREFIX_PATH
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

