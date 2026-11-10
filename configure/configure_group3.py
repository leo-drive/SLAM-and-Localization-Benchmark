#!/usr/bin/env python3
"""
Applies the Boreas-RT-specific parameter changes to a freshly
cloned HDL-Graph-SLAM checkout.

Usage: ./configure_group3.py   (run after fetch_algorithms.sh, before patch_and_build.sh)
"""

import os
import re

# Repo root is one level up: this script lives in configure/, while catkin_ws_src/ (the
# cloned upstream algorithm checkouts this patches) sits at the repo root.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HDL_LAUNCH = os.path.join(REPO_ROOT, "catkin_ws_src", "hdl_graph_slam", "launch", "hdl_graph_slam.launch")


def substitute(text, pattern, replacement, label):
    new_text, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(
            f"Expected exactly 1 match for '{label}' but found {count}. "
            "The upstream file may have changed -- inspect it and update the pattern."
        )
    return new_text


def configure_hdl_graph_slam():
    with open(HDL_LAUNCH) as f:
        text = f.read()

    # boreas2rosbag.py publishes PointCloud2 with frame_id="lidar_link", but the stock static
    # transform here only defines a "velodyne" child frame. HDL-Graph-SLAM's prefiltering nodelet
    # looks up base_link->lidar_link to transform incoming clouds; the lookup fails, throws an
    # uncaught tf2::LookupException, and that terminate()s the ENTIRE nodelet manager process --
    # taking every other nodelet down with it (crash reproduced directly: "failed to find
    # transform between base_link and lidar_link" / SIGABRT).
    text = substitute(
        text,
        r'(args="0 0 0 0 0 0 \$\(arg base_link_frame_id\) )velodyne( 10" />)',
        r"\g<1>lidar_link\g<2>",
        "HDL-Graph-SLAM lidar2base_publisher child frame",
    )

    # Floor/ground-plane detection is OFF by default upstream. This project exists specifically to
    # observe how ground-plane-dependent algorithms fare on winter/snow sequences -- leaving it off
    # would mean never exercising the failure mode Group 3 is built around.
    text = substitute(
        text, r'<arg name="enable_floor_detection" default="false" />',
        '<arg name="enable_floor_detection" default="true" />',
        "HDL-Graph-SLAM enable_floor_detection",
    )
    text = substitute(
        text, r'<arg name="points_topic" default="/velodyne_points" />',
        '<arg name="points_topic" default="/velodyne/points" />',
        "HDL-Graph-SLAM points_topic",
    )
    # The *other* downsample_resolution (scan_matching_odometry_nodelet) is dead value: its
    # downsample_method is "NONE", so it's matched here via the VOXELGRID method line right above
    # it to target only the one that's actually active (prefiltering_nodelet).
    text = substitute(
        text,
        r'(<param name="downsample_method" value="VOXELGRID" />\s*\n\s*<param name="downsample_resolution" value=")0\.1(" />)',
        r"\g<1>0.4\g<2>",
        "HDL-Graph-SLAM prefiltering downsample_resolution",
    )

    # Removed entirely rather than patched in place: the stock map2odom_publisher.py broadcasts
    # map->odom using rospy.Time.now() -- the CONTAINER'S wall clock -- while
    # scan_matching_odometry_nodelet broadcasts odom->base_link using the bag's real (2024)
    # timestamps. tf2 can't bridge a permanent ~1.7-year gap between the two frames, so every
    # map->base_link lookup fails with an extrapolation error (confirmed directly: "Lookup would
    # require extrapolation 54403377s into the past"). map2odom_bridge.py (repo root, run as an
    # aux_proc in collect_metrics.py) replaces it with the same logic using the correction
    # message's own real timestamp -- and as a standalone script, sidesteps an intermittent
    # FileNotFoundError seen resolving the upstream script through catkin's devel/.private path.
    text = substitute(
        text,
        r'\s*<node pkg="hdl_graph_slam" type="map2odom_publisher\.py" name="map2odom_publisher" >\s*\n'
        r'\s*<param name="map_frame_id" value="\$\(arg map_frame_id\)" />\s*\n'
        r'\s*<param name="odom_frame_id" value="\$\(arg lidar_odom_frame_id\)" />\s*\n'
        r"\s*</node>\n",
        "\n",
        "HDL-Graph-SLAM remove stock map2odom_publisher node",
    )

    with open(HDL_LAUNCH, "w") as f:
        f.write(text)
    print(f"[OK] Configured {HDL_LAUNCH}")


if __name__ == "__main__":
    configure_hdl_graph_slam()
    print("[DONE] Group 3 (HDL-Graph-SLAM) configs applied")
