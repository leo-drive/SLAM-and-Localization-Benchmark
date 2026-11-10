#!/bin/bash
set -e
#
# Clones the upstream algorithm sources this benchmark measures, each PINNED to the exact commit
# the published results were produced with -- the same revisions the papers' environment tables
# name (FAST-LIO2 7cc4175, LIO-SAM 0be1fbe, HDL-Graph-SLAM 95b8dce, DLO 11528c0).
#
# Pinning is the point. Tracking a default branch would mean a reader cloning this a year from now
# silently measures different code than the papers describe, and the configure/ scripts -- which
# assert each substitution matches exactly once -- would fail against an upstream that has moved.
SRC_DIR="./catkin_ws_src"

mkdir -p "$SRC_DIR"
cd "$SRC_DIR"

# 1. FAST-LIO & Livox ROS Driver
if [ ! -d "livox_ros_driver" ]; then
    git clone https://github.com/Livox-SDK/livox_ros_driver.git
    git -C livox_ros_driver checkout -q 3d240d5666129e1a3052e78ee8487a04b08fdda3
fi
if [ ! -d "FAST_LIO" ]; then
    git clone https://github.com/hku-mars/FAST_LIO.git
    git -C FAST_LIO checkout -q 7cc4175de6f8ba2edf34bab02a42195b141027e9
    cd FAST_LIO && git submodule update --init && cd ..
fi

# 2. LIO-SAM
if [ ! -d "LIO-SAM" ]; then
    git clone https://github.com/TixiaoShan/LIO-SAM.git
    git -C LIO-SAM checkout -q 0be1fbe6275fb8366d5b800af4fc8c76a885c869
fi

# 3. KISS-ICP & DLO
if [ ! -d "kiss-icp" ]; then
    git clone https://github.com/PRBonn/kiss-icp.git
    git -C kiss-icp checkout -q 1ffa7d7512f10bfc8b1185095011fa31184019e3
fi
if [ ! -d "direct_lidar_odometry" ]; then
    git clone https://github.com/vectr-ucla/direct_lidar_odometry.git
    git -C direct_lidar_odometry checkout -q 11528c02e89a1619ee3c42ba39ecaa992bd20bf4
    cd direct_lidar_odometry && git submodule update --init --recursive && cd ..
fi

# 4. Group 3: HDL-Graph-SLAM and its registration dependencies.
#    RTAB-Map is not here -- it is installed from apt by the Dockerfile.
if [ ! -d "fast_gicp" ]; then
    git clone https://github.com/SMRT-AIST/fast_gicp.git
    git -C fast_gicp checkout -q 0e7ec1441c99f7be453db2ea216d5de029387417
fi
if [ ! -d "ndt_omp" ]; then
    git clone https://github.com/koide3/ndt_omp.git
    git -C ndt_omp checkout -q 5495fd9214945afcb4b35d5a1da385e405c52bf9
fi
if [ ! -d "hdl_graph_slam" ]; then
    git clone https://github.com/koide3/hdl_graph_slam.git
    git -C hdl_graph_slam checkout -q 95b8dce41c10667dcf0e3e9801dc22754b3bd525
fi

# 5. Group 4: map-based localization.
if [ ! -d "hdl_localization" ]; then
    git clone https://github.com/koide3/hdl_localization.git
    git -C hdl_localization checkout -q de2dc7769aa2412876d05ae6161408d3295b6a3c
fi
if [ ! -d "hdl_global_localization" ]; then
    git clone https://github.com/koide3/hdl_global_localization.git
    git -C hdl_global_localization checkout -q a6a6b69929ac371d9531d0dcdef3c1593fd49245
fi
# FAST-LIO-Localization: the FAST-LIO front end plus a low-rate ICP correction against a prior map.
# Note this vendors its OWN older copy of FAST-LIO rather than depending on the one cloned above --
# hence the separate checkout, and hence the source patches patch_and_build.sh applies to it (the
# copy here predates the timestamp_unit option, among other things).
# The submodule init is not optional: ikd-Tree is a submodule and CMake fails at add_executable
# without it. patch_and_build.sh re-runs it defensively for existing checkouts.
if [ ! -d "FAST_LIO_LOCALIZATION" ]; then
    git clone https://github.com/HViktorTsoi/FAST_LIO_LOCALIZATION.git
    git -C FAST_LIO_LOCALIZATION checkout -q 2bc274ed0b36d14a5c34ada5e1473d52aa1db0d2
    cd FAST_LIO_LOCALIZATION && git submodule update --init --recursive && cd ..
fi

echo "All required dependencies successfully fetched!"
