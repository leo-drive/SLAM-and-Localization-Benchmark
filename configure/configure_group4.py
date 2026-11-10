#!/usr/bin/env python3
"""
Applies the Boreas-RT-specific parameter changes to freshly
cloned hdl_localization and FAST_LIO_LOCALIZATION checkouts.

catkin_ws_src/ is gitignored on purpose: upstream sources are cloned, not redistributed.

Usage: ./configure_group4.py   (run after fetch_algorithms.sh, before patch_and_build.sh)
"""

import os
import re
import sys
import xml.dom.minidom

# Repo root is one level up: this script lives in configure/, while catkin_ws_src/ (the
# cloned upstream algorithm checkouts this patches) sits at the repo root.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HDL_LOC_LAUNCH = os.path.join(REPO_ROOT, "catkin_ws_src", "hdl_localization", "launch", "hdl_localization.launch")
FLL_DIR = os.path.join(REPO_ROOT, "catkin_ws_src", "FAST_LIO_LOCALIZATION")
FLL_YAML = os.path.join(FLL_DIR, "config", "velodyne.yaml")
FLL_LAUNCH = os.path.join(FLL_DIR, "launch", "localization_boreas.launch")

# Imported rather than restated so this rig's LiDAR-to-IMU extrinsic has exactly one definition in
# the repo. FAST_LIO_LOCALIZATION embeds the same FAST-LIO front end Group 1 benchmarks, so it takes
# the same matrix; a second copy here could drift out of sync with calib/ silently.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configure_group1 import EXTRINSIC_ROT_ROWS, matrix_literal  # noqa: E402


def substitute(text, pattern, replacement, label):
    new_text, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(
            f"Expected exactly 1 match for '{label}' but found {count}. "
            "The upstream file may have changed -- inspect it and update the pattern."
        )
    return new_text


def configure_hdl_localization():
    with open(HDL_LOC_LAUNCH) as f:
        text = f.read()

    # globalmap_pcd is a hardcoded <param>, not an <arg> -- can't be overridden via roslaunch CLI,
    # but the reference map differs per sequence (collect_metrics.py fills in "{sequence}").
    # Adding an <arg> right after nodelet_manager's, matching that arg's declaration style.
    text = substitute(
        text,
        r'(<arg name="nodelet_manager" default="velodyne_nodelet_manager" />\n)',
        r'\g<1>  <arg name="globalmap_pcd" default="$(find hdl_localization)/data/map.pcd" />\n',
        "hdl_localization globalmap_pcd arg declaration",
    )
    text = substitute(
        text,
        r'<param name="globalmap_pcd" value="\$\(find hdl_localization\)/data/map\.pcd" />',
        '<param name="globalmap_pcd" value="$(arg globalmap_pcd)" />',
        "hdl_localization globalmap_pcd param value",
    )

    # enable_robot_odometry_prediction ships as <arg value="false">, which roslaunch treats as a
    # CONSTANT -- passing it on the command line fails with "cannot override arg". Turned into a
    # default so it can be switched from the command line without editing the launch file, for
    # anyone who wants to try odometry prediction. The default stays false, which is what every
    # reported run used.
    text = substitute(
        text,
        r'<arg name="enable_robot_odometry_prediction" value="false" />',
        '<arg name="enable_robot_odometry_prediction" default="false" />',
        "hdl_localization enable_robot_odometry_prediction overridable",
    )

    # NDT settings, all now backed by measurement against the CORRECTED reference map (see
    # resolved for hdl_localization):
    #   ndt_resolution 2.0 / neighbor radius 2.0 -- DIRECT7 inspects only adjacent voxels, so
    #     capture range is ~one voxel per frame. At 1.0 that is ~1m against 1.65m of vehicle motion
    #     per frame at this sequence's peak speed, so NDT fell behind during acceleration. These
    #     values were rejected earlier, but that test ran against the smeared map where coarse
    #     voxels merely aliased; against the fixed map they give 51 points per voxel.
    #   downsample_resolution 0.8 on the scan -- coarser than the project's usual 0.4 because
#     NDT here must keep up with the sensor: at 0.4 it ran ~7.7Hz, giving 1.02m of vehicle
#     motion per frame against DIRECT7's ~one-voxel capture range. 0.8 restores ~9Hz.
#     0.75 on the GLOBAL map.
    #     globalmap_server re-filters the map with pcl::VoxelGrid, which indexes its grid with
    #     int32: too small a leaf over a km-scale map overflows, and PCL then returns an EMPTY
    #     cloud that is assigned straight over the map, so NDT silently gets nothing to match
    #     against. 0.75m is above the overflow bound for every sequence this group runs
    #     (highway needs >=0.71m, the tightest) while still leaving ~7 points per 2.0m NDT voxel.
    #     boreas-forest is NOT run by this group: its map spans 3.5 x 5.3 km and needs >=1.07m,
    #     which re-sparsifies the map below what NDT can use.
    text = substitute(
        text,
        r'<param name="ndt_resolution" value="1\.0" />',
        '<param name="ndt_resolution" value="2.0" />',
        "hdl_localization ndt_resolution",
    )
    text = substitute(
        text,
        r'<param name="downsample_resolution" value="0\.1" />\n(\s*)</node>',
        r'<param name="downsample_resolution" value="1.2" />\n\g<1></node>',
        "hdl_localization globalmap_server downsample_resolution",
    )
    text = substitute(
        text,
        r'<param name="downsample_resolution" value="0\.1" />',
        '<param name="downsample_resolution" value="0.8" />',
        "hdl_localization scan downsample_resolution",
    )

    # Superseded note, kept for history -- see the Group 4 tracking-accuracy
    # section for the full story. Position stays frozen at the stock 1.0m resolution/2.0m search
    # radius (only 3 unique x / 2 unique y values across 289 poses over 30s) while orientation
    # updates normally every frame, the signature of the UKF's IMU-based prediction correctly
    # updating rotation while its NDT-based correction never meaningfully updates position.
    # Doubling both (1.0->2.0, 2.0->4.0) looked like a real fix at 30s (every pose got a unique
    # position, ATE 28.5m) but was confirmed at 200s to be the SAME underlying failure wearing a
    # different mask: position drifted 1128m/698m off in a rough constant-velocity line by 200s --
    # not genuine NDT correction, just the UKF's own prediction running open-loop because NDT
    # still isn't correcting it, the same failure mode an open-loop predictor shows. A
    # frozen position is at least obviously non-functional; a plausible-looking but silently wrong
    # drifting trajectory is more dangerous to ship, so this project keeps the stock (frozen)
    # values rather than the drifting ones pending an actual fix to NDT-OMP's correction step.

    xml.dom.minidom.parseString(text)  # fail loudly on malformed XML rather than a silent roslaunch crash

    with open(HDL_LOC_LAUNCH, "w") as f:
        f.write(text)
    print(f"[OK] Configured {HDL_LOC_LAUNCH}")


def configure_fast_lio_localization():
    """FAST_LIO_LOCALIZATION = the FAST-LIO front end plus a low-rate ICP correction against a prior
    map. Its sensor config therefore takes the same values Group 1 gives FAST-LIO2 (configure_group1
    .py), and only the map-matching layer is tuned here.

    Note this repo vendors an OLDER FAST-LIO than Group 1's: it predates the timestamp_unit option,
    and its Velodyne handler hard-codes `time / 1000.0`, i.e. it assumes the per-point time field is
    in MICROSECONDS. Boreas writes it in seconds, so left alone every point in a scan would be
    treated as ~5e-5 ms apart and IMU de-skewing would silently do nothing. patch_and_build.sh
    restores the option as `preprocess/time_unit_scale`; this sets it to 1000 (seconds -> the
    milliseconds FAST-LIO works in), the equivalent of Group 1's `timestamp_unit: 0`.
    """
    with open(FLL_YAML) as f:
        text = f.read()

    text = substitute(text, r'lid_topic:\s*"/velodyne_points"', 'lid_topic:  "/velodyne/points"',
                      "FLL lid_topic")
    text = substitute(text, r'imu_topic:\s*"/imu/data"', 'imu_topic:  "/dmu/imu"', "FLL imu_topic")
    text = substitute(text, r"scan_line:\s*32", "scan_line: 128               # Velodyne Alpha-Prime, 128-beam",
                      "FLL scan_line")
    text = substitute(
        text, r"blind:\s*4\b",
        "blind: 0.5                   # roof blind spot already stripped upstream in boreas2rosbag.py\n"
        "    time_unit_scale: 1000.0      # Boreas 'time' is in SECONDS; FAST-LIO wants milliseconds",
        "FLL blind + time_unit_scale",
    )
    text = substitute(
        text, r"extrinsic_T:\s*\[\s*0,\s*0,\s*0\.28\]",
        "# T_lidar_dmu -- from calib/, via boreas2rosbag.py::extract_extrinsics(). Same values\n"
        "    # configure_group1.py gives FAST-LIO2; do not hand-edit.\n"
        "    extrinsic_T: [ 0.0, 0.0, -0.28 ]",
        "FLL extrinsic_T",
    )
    text = substitute(
        text,
        r"extrinsic_R:\s*\[\s*1,\s*0,\s*0,\s*\n\s*0,\s*1,\s*0,\s*\n\s*0,\s*0,\s*1\]",
        "extrinsic_R: " + matrix_literal(EXTRINSIC_ROT_ROWS, "                   "),
        "FLL extrinsic_R",
    )

    with open(FLL_YAML, "w") as f:
        f.write(text)
    print(f"[OK] Configured {FLL_YAML}")


def write_fast_lio_localization_launch():
    """A headless Boreas variant of localization_velodyne.launch.

    Differences from the stock launch, all forced by the rig rather than by preference:
      * no rviz -- runs are headless and CPU/RAM profiling must measure the algorithm only,
        matching Group 1's run_headless.launch.
      * fov 6.3 (> pi) selects crop_global_map_in_FOV's ring-LiDAR branch, which filters the map by
        range alone. The stock 1.6 rad is a forward-facing Livox FOV and would discard roughly
        three quarters of an Alpha-Prime's 360-degree scan.
      * fov_far 100 matches the config's det_range, so the map crop and the odometry front end see
        the same horizon.
      * scan/map voxel sizes coarser than stock: registration here runs against a 13-42M point
        reference map on 128-beam scans, not the ~1M point maps and 6-beam scans upstream targets.
      * freq_localization 1.0 rather than the stock 0.5. Swept on boreas-suburb over 120s
        (unaligned map-frame ATE RMSE): 0.5Hz 0.836m, 1.0Hz 0.729m, 2.0Hz 0.655m. The gain is real
        but flattening, and 2.0 is misleading as a setting -- one ICP cycle costs ~0.85s here, so
        at 2.0 the thread never sleeps and the actual correction rate is bounded by compute, not by
        this parameter. 1.0 takes most of the improvement while leaving the thread duty-cycled,
        which matters because this benchmark also reports CPU.
      * localization_th 0.8. Fitness is the fraction of scan points finding a map point within 1m;
        below the threshold the map->odom correction is DISCARDED and the output silently degrades
        into pure FAST-LIO odometry with a ground-truth-seeded start. That failure looks like
        success in the trajectory file, so the threshold is set where it can actually be met and
        the correction rate is checked explicitly per run.
      * point_filter_num 4 and max_iteration 3 are the stock Velodyne values, left alone.
    """
    content = """<launch>
  <!-- FAST_LIO_LOCALIZATION on Boreas-RT. Generated by configure/configure_group4.py; edit that,
       not this file. Each value's rationale is in the comments above it. -->

  <arg name="map" default="" />
  <!-- Exposed as args so a run can be swept without editing this generated file. -->
  <arg name="freq_localization" default="1.0" />
  <arg name="scan_voxel_size" default="0.4" />
  <arg name="localization_th" default="0.8" />

  <rosparam command="load" file="$(find fast_lio_localization)/config/velodyne.yaml" />

  <param name="feature_extract_enable" type="bool" value="0"/>
  <param name="point_filter_num" type="int" value="4"/>
  <param name="max_iteration" type="int" value="3" />
  <param name="filter_size_surf" type="double" value="0.5" />
  <param name="filter_size_map" type="double" value="0.5" />
  <param name="cube_side_length" type="double" value="1000" />
  <param name="runtime_pos_log_enable" type="bool" value="0" />
  <param name="pcd_save_enable" type="bool" value="0" />

  <!-- FAST-LIO front end: provides high-rate odometry on /Odometry and the registered scan the
       ICP layer matches against on /cloud_registered. -->
  <node pkg="fast_lio_localization" type="fastlio_mapping" name="laserMapping" output="screen" />

  <!-- Low-rate scan-to-map ICP; publishes the map->odom correction on /map_to_odom. -->
  <node pkg="fast_lio_localization" type="global_localization.py" name="global_localization" output="screen">
    <param name="map_voxel_size" value="0.4" />
    <param name="scan_voxel_size" value="$(arg scan_voxel_size)" />
    <param name="freq_localization" value="$(arg freq_localization)" />
    <param name="localization_th" value="$(arg localization_th)" />
    <param name="fov" value="6.3" />
    <param name="fov_far" value="100" />
  </node>

  <!-- Composes map->odom with the odometry to publish the final pose on /localization. -->
  <node pkg="fast_lio_localization" type="transform_fusion.py" name="transform_fusion" output="screen" />

  <!-- Prior map. Published once, latched, on /map. -->
  <node pkg="pcl_ros" type="pcd_to_pointcloud" name="map_publisher" output="screen"
        args="$(arg map) 5 _frame_id:=/map cloud_pcd:=/map" />

</launch>
"""
    xml.dom.minidom.parseString(content)  # fail loudly rather than a silent roslaunch crash later

    os.makedirs(os.path.dirname(FLL_LAUNCH), exist_ok=True)
    with open(FLL_LAUNCH, "w") as f:
        f.write(content)
    print(f"[OK] Wrote {FLL_LAUNCH}")


if __name__ == "__main__":
    configure_hdl_localization()
    configure_fast_lio_localization()
    write_fast_lio_localization_launch()
    print("[DONE] Group 4 configs applied")
