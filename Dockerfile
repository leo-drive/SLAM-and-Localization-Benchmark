# Base Image: ROS Noetic Desktop Full (includes RViz, PCL, Eigen)
FROM osrf/ros:noetic-desktop-full

# Avoid user interaction during apt installations
ENV DEBIAN_FRONTEND=noninteractive

# 1. System Essentials & Python Tools for Evaluation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    unzip \
    nano \
    libmetis-dev \
    libomp-dev \
    python3-pip \
    python3-osrf-pycommon \
    python3-catkin-tools \
    terminator \
    && rm -rf /var/lib/apt/lists/*

# 2. ROS Dependencies for the 8 Contenders
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-geodesy \
    ros-noetic-pcl-ros \
    ros-noetic-nmea-msgs \
    ros-noetic-rviz \
    ros-noetic-tf-conversions \
    ros-noetic-tf2-geometry-msgs \
    ros-noetic-robot-localization \
    ros-noetic-rtabmap-ros \
    ros-noetic-diagnostic-updater \
    ros-noetic-velodyne-pcl \
    ros-noetic-velodyne-pointcloud \
    && rm -rf /var/lib/apt/lists/*

# Fix Velodyne header compatibility symlink for ROS Noetic
RUN mkdir -p /opt/ros/noetic/include/velodyne_pointcloud && \
    if [ -f /opt/ros/noetic/include/velodyne_pcl/point_types.h ]; then \
        ln -sf /opt/ros/noetic/include/velodyne_pcl/point_types.h /opt/ros/noetic/include/velodyne_pointcloud/point_types.h; \
    fi

# 3. Install GTSAM via Official PPA
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:borglab/gtsam-release-4.0 -y \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    libgtsam-dev \
    libgtsam-unstable-dev \
    && rm -rf /var/lib/apt/lists/*

# 4. Install Evaluation Metrics Toolkit (evo)
# Pinned to the versions the published results were produced with. numpy, matplotlib and psutil
# are deliberately NOT pinned here: the base image already provides them from apt
# (/usr/lib/python3/dist-packages), pip leaves those alone, and pinning would shadow the apt copy
# with a pip one. Their versions come from the ROS Noetic / Ubuntu 20.04 base and were
# numpy 1.17.4, matplotlib 3.1.2, psutil 5.5.1 -- the figures the papers quote.
RUN pip3 install --no-cache-dir \
    evo==1.31.1 \
    pandas==2.0.3 \
    scipy==1.10.1 \
    open3d==0.13.0 \
    numpy \
    matplotlib \
    psutil

# 5. Set up the Catkin Workspace
ENV WORKSPACE=/root/catkin_ws
RUN mkdir -p ${WORKSPACE}/src
WORKDIR ${WORKSPACE}

# Copy the fetched source codes and the build script into the container
COPY ./patch_and_build.sh /root/patch_and_build.sh
RUN chmod +x /root/patch_and_build.sh

# 6. Entrypoint configuration
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc
COPY ./entrypoint.sh /root/entrypoint.sh
RUN chmod +x /root/entrypoint.sh

ENTRYPOINT ["/root/entrypoint.sh"]
CMD ["bash"]
