#!/usr/bin/env python3
"""
Applies the Boreas-RT-specific parameter changes to a freshly
cloned Direct LiDAR Odometry (DLO) checkout. KISS-ICP needs no equivalent step: it's not a catkin
package on this rig and is run entirely
via nodes/group2/kiss_icp_ros1_node.py + nodes/group2/kiss_icp_boreas.yaml, both original files
written for this project, not part of any upstream clone.

Usage: ./configure_group2.py   (run after fetch_algorithms.sh, before patch_and_build.sh)
"""

import os
import re
import xml.dom.minidom

# Repo root is one level up: this script lives in configure/, while catkin_ws_src/ (the
# cloned upstream algorithm checkouts this patches) sits at the repo root.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DLO_DIR = os.path.join(REPO_ROOT, "catkin_ws_src", "direct_lidar_odometry")
DLO_YAML = os.path.join(DLO_DIR, "cfg", "dlo.yaml")
DLO_PARAMS_YAML = os.path.join(DLO_DIR, "cfg", "params.yaml")
DLO_HEADLESS_LAUNCH = os.path.join(DLO_DIR, "launch", "dlo_headless.launch")
DLO_HEADLESS_NO_IMU_LAUNCH = os.path.join(DLO_DIR, "launch", "dlo_headless_no_imu.launch")


def substitute(text, pattern, replacement, label):
    new_text, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(
            f"Expected exactly 1 match for '{label}' but found {count}. "
            "The upstream file may have changed -- inspect it and update the pattern."
        )
    return new_text


def configure_dlo():
    with open(DLO_YAML) as f:
        text = f.read()
    text = substitute(text, r"leafSize:\s*0\.25", "leafSize: 0.4", "DLO mapNode.leafSize")
    with open(DLO_YAML, "w") as f:
        f.write(text)
    print(f"[OK] Configured {DLO_YAML}")

    with open(DLO_PARAMS_YAML) as f:
        text = f.read()
    text = substitute(text, r"res:\s*0\.25", "res: 0.4", "DLO odomNode.preprocessing.voxelFilter.scan.res")
    with open(DLO_PARAMS_YAML, "w") as f:
        f.write(text)
    print(f"[OK] Configured {DLO_PARAMS_YAML}")


def write_dlo_headless_launch(out_path, imu_enabled):
    # Ablation pair: dlo_headless.launch (imu_enabled=True) vs dlo_headless_no_imu.launch
    # (imu_enabled=False) -- lets the Group 2 writeup compare DLO with a correctly-extrinsic-
    # rotated IMU prior against true pure-LiDAR DLO, and both against KISS-ICP.
    imu_topic_default = "/dmu/imu_lidar_frame" if imu_enabled else "/dmu/imu"
    imu_overrides = "" if imu_enabled else (
        '        <param name="dlo/imu" type="bool" value="false"/>\n'
        '        <param name="dlo/gravityAlign" type="bool" value="false"/>\n'
    )

    content = f"""<launch>

    <arg name="robot_namespace" default="robot"/>

    <!-- Boreas-RT topics (boreas2rosbag.py) baked in as defaults, no rviz (headless container,
         and rviz would bias CPU/RAM profiling). -->
    <arg name="pointcloud_topic" default="/velodyne/points"/>
    <arg name="imu_topic" default="{imu_topic_default}"/>

    <node ns="$(arg robot_namespace)" name="dlo_odom" pkg="direct_lidar_odometry" type="dlo_odom_node" output="screen" clear_params="true">
        <rosparam file="$(find direct_lidar_odometry)/cfg/dlo.yaml" command="load"/>
        <rosparam file="$(find direct_lidar_odometry)/cfg/params.yaml" command="load"/>
{imu_overrides}        <remap from="~pointcloud" to="$(arg pointcloud_topic)"/>
        <remap from="~imu" to="$(arg imu_topic)"/>
        <remap from="~odom" to="dlo/odom_node/odom"/>
        <remap from="~pose" to="dlo/odom_node/pose"/>
        <remap from="~kfs" to="dlo/odom_node/odom/keyframe"/>
        <remap from="~keyframe" to="dlo/odom_node/pointcloud/keyframe"/>
    </node>

    <node ns="$(arg robot_namespace)" name="dlo_map" pkg="direct_lidar_odometry" type="dlo_map_node" output="screen" clear_params="true">
        <rosparam file="$(find direct_lidar_odometry)/cfg/dlo.yaml" command="load"/>
        <remap from="~keyframes" to="dlo/odom_node/pointcloud/keyframe"/>
        <remap from="~map" to="dlo/map_node/map"/>
    </node>

</launch>
"""
    xml.dom.minidom.parseString(content)  # fail loudly on malformed XML rather than a silent roslaunch crash

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(content)
    print(f"[OK] Wrote {out_path}")


if __name__ == "__main__":
    configure_dlo()
    write_dlo_headless_launch(DLO_HEADLESS_LAUNCH, imu_enabled=True)
    write_dlo_headless_launch(DLO_HEADLESS_NO_IMU_LAUNCH, imu_enabled=False)
    print("[DONE] Group 2 configs applied")
