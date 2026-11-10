#!/bin/bash
set -e

source /opt/ros/noetic/setup.bash

echo "[1/5] Configuring Git safe directory & installing system ROS msgs..."
git config --global --add safe.directory "*"
apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-jsk-recognition-msgs \
    liblz4-dev \
    && rm -rf /var/lib/apt/lists/*

WORKSPACE_DIR="/root/catkin_ws"
LIO_SAM_DIR="${WORKSPACE_DIR}/src/LIO-SAM"

echo "[2/5] Resetting LIO-SAM and applying C++14 & OpenCV 4 header order patches..."
if [ -d "${LIO_SAM_DIR}" ]; then
    cd "${LIO_SAM_DIR}"
    git checkout -f .
    git clean -fd .
    cd "${WORKSPACE_DIR}"

    # Force C++14 for ROS Noetic GTSAM & PCL
    sed -i 's/-std=c++11/-std=c++14/g' "${LIO_SAM_DIR}/CMakeLists.txt"

    # 1. Safely remove the legacy OpenCV header
    sed -i 's|#include <opencv/cv.h>||g' "${LIO_SAM_DIR}/include/utility.h"
    
    # 2. Inject OpenCV 4 cleanly after PCL headers to prevent FLANN conflicts
    sed -i '/class ParamServer/i #include <opencv2/opencv.hpp>' "${LIO_SAM_DIR}/include/utility.h"
fi

echo "[3/5] Installing KISS-ICP's Python core (its own ros/ package is ROS2-only; we run it via a"
echo "      custom ROS1 wrapper, see kiss_icp_ros1_node.py). Installed from PyPI's official wheel"
echo "      -- matches the v1.3.0 tag already cloned into catkin_ws_src/kiss-icp, avoids building"
echo "      the pybind11/C++ extension locally."
pip3 install --no-cache-dir kiss-icp==1.3.0

echo "[4/5] Patching FAST_LIO_LOCALIZATION (Group 4)."
echo "      Four upstream problems, none of them configurable:"
echo "        a) ikd-Tree is a git submodule and the repo does not build without it."
echo "        b) common_lib.h includes <fast_lio/Pose6D.h>, but the package generates that message"
echo "           under its own name -- the original path only resolves if a fast_lio package happens"
echo "           to be built alongside it."
echo "        c) the vendored FAST-LIO predates timestamp_unit and hard-codes the Velodyne 'time'"
echo "           field as MICROSECONDS. Boreas writes it in SECONDS, so every point in a scan would"
echo "           be treated as ~5e-5 ms apart and IMU de-skewing would silently do nothing."
echo "        d) this fork replaced the odometry data timestamp with ros::Time::now(). Replaying a"
echo "           bag without /use_sim_time then stamps every pose with the wall clock -- measured"
echo "           at 639 days adrift from the sequence's own time, so no pose associates with ground"
echo "           truth at all."
FLL_DIR="${WORKSPACE_DIR}/src/FAST_LIO_LOCALIZATION"
if [ -d "${FLL_DIR}" ]; then
    git -C "${FLL_DIR}" submodule update --init --recursive

    # global_localization.py needs Open3D, which touches numpy.random.BitGenerator at import and so
    # requires numpy >= 1.19. This image pins numpy 1.17.4 (Ubuntu 20.04's python3-numpy, what ROS
    # Noetic ships against), and raising it system-wide would put every tool that produced Groups
    # 1-3 on an untested numpy. A venv built with --system-site-packages isolates the change to
    # this one node: ROS, PCL bindings and everything else still come from the system, only numpy
    # is shadowed. 1.23.5 specifically -- new enough for Open3D 0.13, old enough that ros_numpy's
    # np.float alias still resolves (removed in numpy 1.24). The node's shebang points here.
    apt-get install -y python3-venv ros-noetic-ros-numpy
    python3 -m venv --system-site-packages /opt/fll_venv
    /opt/fll_venv/bin/pip install --no-cache-dir "numpy==1.23.5"

    # (b) generated-message include path
    sed -i 's|#include <fast_lio/Pose6D.h>|#include <fast_lio_localization/Pose6D.h>|' \
        "${FLL_DIR}/include/common_lib.h"
    sed -i 's|typedef fast_lio::Pose6D Pose6D;|typedef fast_lio_localization::Pose6D Pose6D;|' \
        "${FLL_DIR}/include/common_lib.h"

    # (c) restore the timestamp_unit option later FAST-LIO releases added, as
    #     preprocess/time_unit_scale. Default 1e-3 preserves upstream behaviour for stock configs;
    #     configure_group4.py sets 1000.0 in config/velodyne.yaml for Boreas.
    grep -q 'time_unit_scale' "${FLL_DIR}/src/preprocess.h" || sed -i \
        's|^  double blind;|  double blind;\n  double time_unit_scale;  // Velodyne "time" -> FAST-LIO milliseconds|' \
        "${FLL_DIR}/src/preprocess.h"
    grep -q 'time_unit_scale = ' "${FLL_DIR}/src/preprocess.cpp" || sed -i \
        '0,/^Preprocess::Preprocess()/{s|^\(Preprocess::Preprocess().*{\)|\1\n  time_unit_scale = 1e-3;  // upstream default: microsecond "time" field|}' \
        "${FLL_DIR}/src/preprocess.cpp"
    sed -i 's|pl_orig.points\[i\].time / 1000.0|pl_orig.points[i].time * time_unit_scale|g' \
        "${FLL_DIR}/src/preprocess.cpp"
    grep -q 'preprocess/time_unit_scale' "${FLL_DIR}/src/laserMapping.cpp" || sed -i \
        's|\(    nh.param<double>("preprocess/blind", p_pre->blind, 0.01);\)|\1\n    nh.param<double>("preprocess/time_unit_scale", p_pre->time_unit_scale, 1e-3);|' \
        "${FLL_DIR}/src/laserMapping.cpp"

    # (d) publish the LiDAR frame's end time, as upstream FAST-LIO does and as every other
    #     algorithm in this benchmark reports.
    sed -i 's|    odomAftMapped.header.stamp = ros::Time::now();.*|    odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time);|' \
        "${FLL_DIR}/src/laserMapping.cpp"
fi

echo "[5/5] Building workspace cleanly..."
cd "${WORKSPACE_DIR}"
catkin config --init --workspace "${WORKSPACE_DIR}" || true
catkin clean -y --workspace "${WORKSPACE_DIR}" || true
catkin build -DCMAKE_BUILD_TYPE=Release

echo "Workspace build completed successfully!"
