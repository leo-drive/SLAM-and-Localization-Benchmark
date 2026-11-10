#!/usr/bin/env python3
"""
Applies the Boreas-RT-specific parameter changes to a freshly
cloned FAST-LIO2 / LIO-SAM checkout.

Usage: ./configure_group1.py   (run after fetch_algorithms.sh, before patch_and_build.sh)
"""

import os
import re
import xml.dom.minidom

# Repo root is one level up: this script lives in configure/, while catkin_ws_src/ (the
# cloned upstream algorithm checkouts this patches) sits at the repo root.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FASTLIO_YAML = os.path.join(REPO_ROOT, "catkin_ws_src", "FAST_LIO", "config", "velodyne.yaml")
LIOSAM_YAML = os.path.join(REPO_ROOT, "catkin_ws_src", "LIO-SAM", "config", "params.yaml")
LIOSAM_HEADLESS_LAUNCH = os.path.join(REPO_ROOT, "catkin_ws_src", "LIO-SAM", "launch", "run_headless.launch")

# T_lidar_dmu, computed from calib/T_applanix_lidar.txt & calib/T_applanix_dmu.txt via
# boreas2rosbag.py::extract_extrinsics(). Re-derive and update here if the calib files ever change.
EXTRINSIC_ROT_ROWS = [
    (0.728265, -0.685295, -0.001208),
    (-0.685295, -0.728266, 0.000785),
    (-0.001418, 0.000256, -0.999999),
]


def matrix_literal(rows, continuation_indent):
    """Formats a 3x3 matrix as a YAML flow-sequence literal, one row per line."""
    row_strs = [", ".join(f"{v:g}" for v in row) for row in rows]
    return "[" + (",\n" + continuation_indent).join(row_strs) + "]"


def substitute(text, pattern, replacement, label):
    new_text, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(
            f"Expected exactly 1 match for '{label}' but found {count}. "
            "The upstream file may have changed -- inspect it and update the pattern."
        )
    return new_text


def configure_fastlio():
    with open(FASTLIO_YAML) as f:
        text = f.read()

    text = substitute(text, r'lid_topic:\s*"/velodyne_points"', 'lid_topic:  "/velodyne/points"',
                       "FAST-LIO2 lid_topic")
    text = substitute(text, r'imu_topic:\s*"/imu/data"', 'imu_topic:  "/dmu/imu"', "FAST-LIO2 imu_topic")
    text = substitute(text, r"scan_line:\s*32", "scan_line: 128", "FAST-LIO2 scan_line")
    text = substitute(text, r"timestamp_unit:\s*2\b", "timestamp_unit: 0", "FAST-LIO2 timestamp_unit")
    text = substitute(text, r"blind:\s*2\b", "blind: 0.5", "FAST-LIO2 blind")
    text = substitute(
        text, r"extrinsic_T:\s*\[\s*0,\s*0,\s*0\.28\s*\]",
        "extrinsic_T: [ 0.0, 0.0, -0.28 ]", "FAST-LIO2 extrinsic_T",
    )
    text = substitute(
        text,
        r"extrinsic_R:\s*\[\s*1,\s*0,\s*0,\s*\n\s*0,\s*1,\s*0,\s*\n\s*0,\s*0,\s*1\]",
        "extrinsic_R: " + matrix_literal(EXTRINSIC_ROT_ROWS, "               "),
        "FAST-LIO2 extrinsic_R",
    )

    with open(FASTLIO_YAML, "w") as f:
        f.write(text)
    print(f"[OK] Configured {FASTLIO_YAML}")


def configure_liosam():
    with open(LIOSAM_YAML) as f:
        text = f.read()

    text = substitute(text, r'pointCloudTopic:\s*"points_raw"', 'pointCloudTopic: "/velodyne/points"',
                       "LIO-SAM pointCloudTopic")
    text = substitute(text, r'imuTopic:\s*"imu_raw"', 'imuTopic: "/dmu/imu"', "LIO-SAM imuTopic")
    text = substitute(text, r"N_SCAN:\s*16\b", "N_SCAN: 128", "LIO-SAM N_SCAN")
    text = substitute(text, r"imuAccNoise:\s*[\d.eE+-]+", "imuAccNoise: 8.33e-04", "LIO-SAM imuAccNoise")
    text = substitute(text, r"imuGyrNoise:\s*[\d.eE+-]+", "imuGyrNoise: 5.82e-06", "LIO-SAM imuGyrNoise")
    text = substitute(text, r"imuAccBiasN:\s*[\d.eE+-]+", "imuAccBiasN: 4.09e-08", "LIO-SAM imuAccBiasN")
    text = substitute(text, r"imuGyrBiasN:\s*[\d.eE+-]+", "imuGyrBiasN: 4.85e-07", "LIO-SAM imuGyrBiasN")
    text = substitute(
        text, r"extrinsicTrans:\s*\[0\.0,\s*0\.0,\s*0\.0\]",
        "extrinsicTrans: [0.0, 0.0, -0.28]", "LIO-SAM extrinsicTrans",
    )
    text = substitute(
        text,
        r"extrinsicRot:\s*\[-1,\s*0,\s*0,\s*\n\s*0,\s*1,\s*0,\s*\n\s*0,\s*0,\s*-1\]",
        "extrinsicRot: " + matrix_literal(EXTRINSIC_ROT_ROWS, "                 "),
        "LIO-SAM extrinsicRot",
    )
    text = substitute(
        text,
        r"extrinsicRPY:\s*\[0,\s*-1,\s*0,\s*\n\s*1,\s*0,\s*0,\s*\n\s*0,\s*0,\s*1\]",
        "extrinsicRPY: " + matrix_literal(EXTRINSIC_ROT_ROWS, "                 "),
        "LIO-SAM extrinsicRPY",
    )
    text = substitute(text, r"savePCD:\s*false", "savePCD: true", "LIO-SAM savePCD")
    text = substitute(
        text, r'savePCDDirectory:\s*"/Downloads/LOAM/"',
        'savePCDDirectory: "/lio_sam_map_export/"', "LIO-SAM savePCDDirectory",
    )

    with open(LIOSAM_YAML, "w") as f:
        f.write(text)
    print(f"[OK] Configured {LIOSAM_YAML}")


def write_liosam_headless_launch():
    content = """<launch>

    <arg name="project" default="lio_sam"/>

    <!--- Parameters -->
    <rosparam file="$(find lio_sam)/config/params.yaml" command="load" />

    <!--- LOAM -->
    <include file="$(find lio_sam)/launch/include/module_loam.launch" />

    <!--- Robot State TF -->
    <include file="$(find lio_sam)/launch/include/module_robot_state_publisher.launch" />

    <!-- No navsat (no GPS topic published by boreas2rosbag.py) and no rviz: this variant is for
         automated benchmark runs where CPU/RAM profiling must reflect only the algorithm itself,
         and where a headless container has no display for rviz anyway. See run.launch for the
         interactive/visualization variant. -->

</launch>
"""
    xml.dom.minidom.parseString(content)  # fails loudly on malformed XML instead of a silent roslaunch crash later

    os.makedirs(os.path.dirname(LIOSAM_HEADLESS_LAUNCH), exist_ok=True)
    with open(LIOSAM_HEADLESS_LAUNCH, "w") as f:
        f.write(content)
    print(f"[OK] Wrote {LIOSAM_HEADLESS_LAUNCH}")


if __name__ == "__main__":
    configure_fastlio()
    configure_liosam()
    write_liosam_headless_launch()
    print("[DONE] Group 1 configs applied")
