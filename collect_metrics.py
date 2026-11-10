#!/usr/bin/env python3
"""
Runs a SLAM/odometry algorithm N times against Boreas-RT sequence bags, recording:
  - the estimated trajectory (TUM format) from its odometry topic
  - CPU/RAM usage of the algorithm's full process tree, sampled at a fixed interval
  - run metadata (wall-clock duration, bag duration, real-time factor, exit codes)

Meant to run INSIDE the benchmark Docker container (docker-compose.yml), where the dataset is
mounted read-only at /data/boreas and output goes to /root/benchmark_results (-> ./benchmark_results
on the host). The extrinsic, noise and topic fixes it depends on are applied by configure/.

Usage:
  ./collect_metrics.py --algorithm fastlio2 --sequences boreas-highway boreas-snowy --runs 3
  ./collect_metrics.py --algorithm liosam --runs 3   # all 5 sequences by default
  ./collect_metrics.py --algorithm liosam --sequences boreas-suburb --runs 1  # just run 1, collect
                                                                              # the rest later
  ./collect_metrics.py --algorithm liosam --sequences boreas-suburb --run-numbers 2 3  # top up to
                                                                              # N=3 without redoing run 1
"""

import argparse
import csv
import json
import os
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import rosbag
import rosgraph

DATASET_ROOT = Path("/data/boreas")
RESULTS_ROOT = Path("/root/benchmark_results")

# Helper ROS nodes this project wrote itself, organized by which group needs them (nodes/README.md
# explains what each one does). Paths are relative to this file rather than absolute so the repo
# works at whatever path it's mounted at, inside the container or out.
NODES = Path(__file__).parent / "nodes"

SEQUENCES = ["boreas-forest", "boreas-highway", "boreas-snowy", "boreas-suburb", "boreas-urban"]

# One entry per algorithm. Add Group 2-4 entries here as their configs come online.
ALGORITHMS = {
    "fastlio2": {
        "group": "group1_tight_coupled_lio",
        "gt_frame": "imu",  # publishes its IMU state directly (laserMapping.cpp state_point)
        "roslaunch": ["fast_lio", "mapping_velodyne.launch", "rviz:=false"],
        "odom_topic": "/Odometry",
        # FAST-LIO2 registers every scan against its own accumulated map (ikd-Tree), not just
        # frame-to-frame -- a real persistent map exists here despite this being "Group 1", so it
        # gets the same map-quality treatment as Group 3. `pcd_save_en: true` (velodyne.yaml,
        # already set) makes it write that map itself, as part of its own clean-shutdown handler,
        # to a FIXED path under its own package source dir (`ROOT_DIR`, a compile-time constant --
        # confirmed via CMakeLists.txt) -- not per-run-unique, so copy-and-rename happens every run
        # before the next one's shutdown would overwrite it. Point type is PointXYZINormal, not
        # plain XYZI (confirmed via common_lib.h) -- metrics_lib.py's PCD reader parses the actual
        # header rather than assuming a fixed layout, for exactly this reason.
        "map_export_shutdown_file": "/root/catkin_ws/src/FAST_LIO/PCD/scans.pcd",
    },
    "liosam": {
        "group": "group1_tight_coupled_lio",
        "gt_frame": "lidar",  # imuConverter rotates IMU into the LiDAR frame at input
        "roslaunch": ["lio_sam", "run_headless.launch"],
        # NOT /lio_sam/mapping/odometry: that topic only updates once per mapOptimization cycle
        # (~2.6Hz average, observed gaps up to 1.55s and single hops up to 21.5m on a real
        # 986s run) -- badly undersamples sharp/slow turns, which inflates RPE-rotation via
        # SO(3)'s geodesic-distance sensitivity to axis disagreement (grows with sin(theta/2),
        # so even a small, physically-plausible axis discrepancy between GT and LIO-SAM balloons
        # into 100+ degree apparent error on a segment containing a large accumulated turn).
        # /odometry/imu is LIO-SAM's IMU-preintegration-fused output (TransformFusion, publishes
        # at full IMU rate -- confirmed 200Hz, std dev 0.24ms on this dataset's DMU41) which still
        # incorporates every /lio_sam/mapping/odometry correction (including loop closure) via
        # imuOdomAffineFront/Back fusion, so nothing about global consistency is lost.
                "odom_topic": "/odometry/imu",
        # LIO-SAM also registers against its own accumulated corner/surface maps, not just
        # frame-to-frame -- same reasoning as FAST-LIO2 above. `savePCD:=true` and a dedicated
        # `savePCDDirectory` are set by configure_group1.py (config/params.yaml) -- when true,
        # LIO-SAM's own shutdown handler (visualizeGlobalMapThread) calls its internal save-map
        # service automatically, no live call needed from us. That service does `rm -r` on its
        # destination directory with NO safety check (confirmed via source,
        # mapOptmization.cpp::saveMapService) -- savePCDDirectory must always be this small,
        # dedicated, disposable scratch path, never anything this project actually values.
        "map_export_shutdown_file": "/root/lio_sam_map_export/GlobalMap.pcd",
    },
    "kissicp": {
        "group": "group2_pure_lidar_odometry",
        "gt_frame": "lidar",
        # Not a catkin package -- KISS-ICP's own ros/ wrapper is ROS2-only.
        "launch_cmd": ["python3", str(NODES / "group2/kiss_icp_ros1_node.py")],
        "odom_topic": "/kiss_icp/odometry",
        # kiss_icp_ros1_node.py's subscriber queue_size is 200 (see its own comment) to absorb
        # bursty contention (rosbag recording flush, sysstats sampling) that briefly pushes
        # register_frame() past the ~104ms scan interval even once its average cost is back under
        # budget (see kiss_icp_boreas.yaml). At worst case (queue full, ~150-300ms/frame) draining
        # it after rosbag play stops can take up to ~60s -- far past the generic TAIL_DRAIN_GRACE_S
        # (3.0s, sized for "a few last scans," not a deep backlog). Without this override the
        # recorder was torn down before the backlog it still had left to process ever got
        # published, silently truncating the trajectory regardless of queue_size.
        "drain_grace_s": 60.0,
    },
    "dlo": {
        "group": "group2_pure_lidar_odometry",
        "gt_frame": "lidar",
        "roslaunch": ["direct_lidar_odometry", "dlo_headless.launch"],
        "odom_topic": "/robot/dlo/odom_node/odom",
        # DLO has no extrinsic calibration mechanism -- rotate the IMU
        # into the LiDAR frame ourselves before DLO ever sees it.
        "aux_procs": [["python3", str(NODES / "shared/imu_frame_rotator.py")]],
        # DLO keeps its own accumulated map (dlo::MapNode) for scan-to-map registration -- a real
        # persistent map, unlike KISS-ICP's distance-bounded rolling window (see metrics_lib.py's
        # module docstring for why that one's excluded). Exported via its own `save_pcd` service
        # (dlo::MapNode::savePcd, confirmed via source -- safe, writes directly, no rm-r unlike
        # LIO-SAM's) at node name "dlo_map" under the launch file's "robot" namespace, giving
        # /robot/dlo_map/save_pcd. Unlike HDL-Graph-SLAM's save_map, this service takes a
        # DIRECTORY and always writes a fixed "dlo_map.pcd" inside it, so the directory itself
        # must be made unique per run (see map_export_service_dir handling in run_once()).
        # leaf_size:=0.05 (not 0.0): DLO's savePcd() has no zero-guard the way LIO-SAM's
        # equivalent service does (confirmed via source, dlo::MapNode::savePcd) -- it
        # unconditionally calls pcl::VoxelGrid::filter() with whatever leaf_size is given, and a
        # literal 0.0 leaf size is degenerate (confirmed directly: collapsed a real ~1.9M-point
        # map down to a single point). 0.05m is finer than DLO's own live per-keyframe voxel
        # filter (0.4m), so it doesn't remove anything further -- the
        # intent ("keep native resolution") still holds, just via a valid small value instead of
        # a literal no-op zero.
        "map_export_service_dir": {
            "name": "/robot/dlo_map/save_pcd",
            "request": {"leaf_size": 0.05},
            "dest_field": "save_path",
            "output_filename": "dlo_map.pcd",
        },
    },
    "dlo_no_imu": {
        "group": "group2_pure_lidar_odometry",
        "gt_frame": "lidar",
        # Ablation pair with "dlo": imu/gravityAlign forced off (dlo_headless_no_imu.launch), no
        # rotator needed. Lets the Group 2 writeup compare DLO with vs without the IMU prior, and
        # both against KISS-ICP (true pure-LiDAR either way).
        "roslaunch": ["direct_lidar_odometry", "dlo_headless_no_imu.launch"],
        "odom_topic": "/robot/dlo/odom_node/odom",
    },
    "hdlgraphslam": {
        "group": "group3_full_slam",
        "gt_frame": "lidar",
        "roslaunch": ["hdl_graph_slam", "hdl_graph_slam.launch"],
        # HDL-Graph-SLAM never publishes the corrected trajectory directly -- only raw
        # pre-loop-closure odometry (/odom) plus a map->odom TF correction. map2odom_bridge.py
        # broadcasts that correction (replacing the stock node), and
        # tf_corrected_pose_republisher.py composes it with odom->base_link into one topic.
        "odom_topic": "/hdl_graph_slam/corrected_odom",
        "aux_procs": [
            ["python3", str(NODES / "group3/map2odom_bridge.py")],
            ["python3", str(NODES / "group3/tf_corrected_pose_republisher.py"),
             "_trigger_topic:=/odom", "_output_topic:=/hdl_graph_slam/corrected_odom",
             "_map_frame:=map", "_base_link_frame:=base_link"],
        ],
        # Native map export for map-quality metrics (spatial entropy, plane-fitting RMSE,
        # ghosting) -- HDL-Graph-SLAM's own /hdl_graph_slam/save_map service, called while the
        # node is still alive (see run_once()). 0.05m matches a typical fine export resolution for
        # this service; deliberately finer than the project's 0.3-0.5m live-processing voxel
        # target, since a downsampled-for-evaluation map would understate real wall thickness.
        # utm:false is a no-op either way -- zero_utm is only ever set from GPS/NMEA input, which
        # this project never feeds it.
        "map_export_service": {"name": "/hdl_graph_slam/save_map", "request": {"resolution": 0.05, "utm": False}},
    },
    "rtabmap": {
        "group": "group3_full_slam",
        "gt_frame": "lidar",
        # Apt-installed (ros-noetic-rtabmap-ros), not in catkin_ws_src -- everything's passed via
        # roslaunch args directly, no source patching needed. Pure ICP/LiDAR mode: no camera data
        # exists in this dataset, so visual_odometry/depth/stereo are all off -- which also means
        # RTAB-Map's optional SURF feature detector (OpenCV nonfree, patent-encumbered for
        # commercial use) is never reachable in this configuration.
        "roslaunch": [
            "rtabmap_ros", "rtabmap.launch",
            "rtabmap_viz:=false", "rviz:=false",
            "stereo:=false", "depth:=false",
            "visual_odometry:=false", "icp_odometry:=true",
            "subscribe_scan_cloud:=true", "scan_cloud_topic:=/velodyne/points",
            # icp_odometry exact-syncs on BOTH scan and scan_cloud by default; since nothing ever
            # publishes /scan, that sync would never reliably fire (confirmed directly: "Did not
            # receive data... subscribed to /scan and /velodyne/points (make sure only one of
            # this topic is published, otherwise remap one to a dummy topic name)" -- its own
            # diagnostic states the fix).
            "scan_topic:=/unused_scan_topic",
            "frame_id:=lidar_link", "odom_topic:=odom",
            # RTAB-Map's own docs recommend IMU for icp_odometry ("deskewing will work best if
            # you use an IMU"). Wired to the same extrinsic-rotated IMU built for DLO (icp_odometry
            # has no calibration mechanism of its own either -- reuses imu_frame_rotator.py as an
            # aux_proc) and scan_deskewing:=true (uses the real per-point time field, same
            # capability KISS-ICP's deskew:=True already relies on). Tested directly: this does
            # NOT eliminate the tracking-loss episode documented below -- it recurs at the same
            # point in the sequence even with IMU assist and deskewing enabled, which is stronger
            # evidence that it's real algorithmic behavior rather than a missing config knob. Kept
            # anyway since it's the more complete, standards-supported configuration regardless.
            "imu_topic:=/dmu/imu_lidar_frame", "scan_deskewing:=true",
            # use_sim_time: icp_odometry's own diagnostic "delay" was computed against the
            # CONTAINER's wall clock (2026) instead of the bag's real (2024) time -- a benign
            # diagnostic number, not the cause of the registration failures below, but a real bug
            # worth fixing anyway (needs_clock -> rosbag play --clock, see run_once()).
            "use_sim_time:=true",
            # Odom/ResetCountdown=0 (default) never auto-recovers from lost tracking -- confirmed
            # directly: odometry runs cleanly for ~12 frames, hits one genuine low-correspondence-
            # ratio ICP failure (a real registration difficulty, not a bug), then get permanently
            # stuck failing "cannot do registration with a null guess" forever after (matches
            # RTAB-Map's own documented "lidar odometry cannot recover from being lost" design).
            # Icp/VoxelSize matches the project's 0.3-0.5m target and roughly halves per-frame
            # processing time. Neither eliminates tracking loss on hard segments -- with both set,
            # it now resets and keeps running instead of freezing, so tracking-loss/recovery
            # events show up as real robustness data in the results instead of a dead node.
            # Icp/MaxTranslation defaults to 0.2m -- a per-registration sanity limit sized for an
            # indoor robot, and far below what a road vehicle covers between two scans here
            # (0.8m at suburb's 8m/s up to 3.2m at highway's 30.3m/s, at ~9.6Hz). Left at the
            # default, libpointmatcher rejected nearly every registration as implausible
            # ("limit out of bounds: tr: 0.2036/0.2", confirmed directly in icp_odometry's own
            # log), odometry reset on almost every frame, and the trajectory never accumulated:
            # 0.46m of estimated travel against 144m actually driven. 5.0m covers the fastest
            # sequence's per-scan motion with ~58% margin while still rejecting the physically
            # impossible jumps the check exists to catch. Icp/MaxRotation's 0.78rad default is
            # fine and untouched -- observed rotations were ~0.011rad, nowhere near it.
            # Icp/MaxCorrespondenceDistance must scale WITH Icp/VoxelSize -- upstream pairs a 0.05m
            # voxel with a 0.1m search radius (a 2:1 ratio). This project raised the voxel to 0.4m
            # (the 0.3-0.5m target applied to every algorithm) but left the radius at its 0.1m
            # default, i.e. ICP was searching a radius 4x SMALLER than the spacing of the points it
            # was matching -- so almost nothing could correspond. Confirmed in icp_odometry's own
            # log: "corrRatio=0.0994/0.100000", failing Icp/CorrespondenceRatio by a hair on
            # essentially every frame, which reset odometry continuously. 0.8m restores upstream's
            # 2:1 ratio at our voxel size rather than being tuned to taste.
            "args:=--delete_db_on_start --Odom/ResetCountdown 1 --Icp/VoxelSize 0.4 "
            "--Icp/MaxCorrespondenceDistance 0.8 --Icp/MaxTranslation 5.0",
            # Per-run-unique database path (templated with {sequence}/{run}, same substitution as
            # every other {sequence}-templated arg -- see run_once()) so N runs of the same
            # sequence never collide, and so the finished .db survives long enough after the node
            # exits to be read by rtabmap-export below. --delete_db_on_start still applies per
            # path, harmless now that each run has its own.
            "database_path:=/root/benchmark_results/group3_full_slam/raw/{sequence}_run{run}_rtabmap.db",
        ],
        "needs_clock": True,
        # Same "raw odom + map->odom TF correction only" pattern as HDL-Graph-SLAM (publish_tf_map
        # default true, no corrected topic published directly) -- reuse the same republisher.
        "odom_topic": "/rtabmap/corrected_odom",
        "aux_procs": [
            ["python3", str(NODES / "shared/imu_frame_rotator.py")],
            ["python3", str(NODES / "group3/tf_corrected_pose_republisher.py"),
             "_trigger_topic:=/rtabmap/odom", "_output_topic:=/rtabmap/corrected_odom",
             "_map_frame:=map", "_base_link_frame:=lidar_link"],
        ],
        # Native map export for map-quality metrics -- RTAB-Map's own persistent map lives in this
        # SQLite database, not a live service call like HDL-Graph-SLAM's; rtabmap-export (ships
        # with ros-noetic-rtabmap-ros) pulls an assembled point cloud out of it after the node has
        # exited and closed the file (see run_once()). Must match database_path above exactly.
        "map_export_db": "/root/benchmark_results/group3_full_slam/raw/{sequence}_run{run}_rtabmap.db",
    },
    "hdl_localization": {
        "group": "group4_map_localization",
        "gt_frame": "lidar",
        # use_global_localization:=false -- global (no-prior) place-recognition relocalization is
        # unneeded and adds a dependency on hdl_global_localization's own service being up: we
        # already have a good prior via gt_initial_pose_publisher.py.
        # globalmap_pcd is per-sequence -- configure_group4.py exposes it as an overridable arg
        # (hardcoded upstream). Publishes nav_msgs/Odometry on /odom directly -- no bridge needed.
        "roslaunch": [
            "hdl_localization", "hdl_localization.launch",
            "points_topic:=/velodyne/points", "use_global_localization:=false",
            "globalmap_pcd:=/data/boreas/{sequence}/ground_truth_map.pcd",
            # odom_child_frame_id defaults to "velodyne"; our PointCloud2 frame_id is "lidar_link"
            # (boreas2rosbag.py) -- left mismatched, hdl_localization looks up a TF between the two
            # that doesn't exist and never processes anything (confirmed directly: "canTransform:
            # target_frame velodyne does not exist. source_frame lidar_link does not exist").
            "odom_child_frame_id:=lidar_link",
            # Stock leaves BOTH prediction inputs off (use_imu false,
            # enable_robot_odometry_prediction false), making the UKF's prediction step a bare
            # constant-velocity model with no motion input at all. Survivable for a slow indoor
            # robot, not for a vehicle at 8-30 m/s: through any turn the prediction lags, NDT gets
            # a poor initial guess, and in repetitive suburban geometry it converges confidently
            # onto the WRONG street -- measured directly as 100% convergence at 0.062m matching
            # error and 0.966 inlier fraction while sitting 149m from truth. Feeding the IMU gives
            # the prediction real motion, and because NDT genuinely converges now, that prediction
            # is corrected every frame instead of running open-loop -- the thing that made the
            # an open-loop IMU prediction diverge without bound.
            # IMU prediction ON. Without any motion input the UKF's constant-velocity prediction
            # lags badly (43m error by t=60s); with it, tracking holds sub-metre while the vehicle
            # is moving steadily. See ndt_resolution below for the piece that makes it hold through
            # acceleration too.
            "use_imu:=true", "imu_topic:=/dmu/imu_lidar_frame",
        ],
        "odom_topic": "/odom",
        # boreas-forest is EXCLUDED, and this is a reported finding rather than a defect to patch
        # around. hdl_localization's globalmap_server re-filters the reference map with
        # pcl::VoxelGrid, which indexes its grid with int32; forest's map spans 3.5 x 5.3 km, so any
        # leaf below ~1.07m overflows and PCL silently returns an EMPTY cloud. A leaf above that
        # bound re-sparsifies the map below the >=6 points per voxel NDT needs to form a
        # distribution. There is no configuration that satisfies both on this sequence, and the
        # project does not modify algorithm source to work around it.
        # boreas-forest was excluded here until its exclusion was actually measured. The stated
        # reason -- PCL VoxelGrid overflows int32 below ~1.07m leaf, and above that NDT starves --
        # holds only in its first half: at 1.2m there is no overflow AND 59% of map cells still
        # carry the >=6 points NDT needs (suburb runs at 61.5%). Measured at 1.2m: forest 0.43m
        # median, suburb 0.34m RMSE, i.e. unchanged from its 0.75m baseline. One resolution serves
        # all five sequences, so there is no sequence to skip.
        "skip_sequences": [],
        "aux_procs": [
            # A wheel-odometry publisher used to sit here and was removed: nothing consumed it.
            # hdl_localization takes robot odometry through TF (lookup_transform_full via
            # robot_odom_frame_id), never a topic, and enable_robot_odometry_prediction is false
            # anyway, so that path is not even active. This is not merely untidy: ResourceSampler
            # profiles [launch_proc] + aux_procs, so a process doing nothing for this algorithm
            # was being charged to its reported CPU and RAM. Trajectory results were unaffected --
            # nothing read its output.
            # hdl_localization applies IMU in the sensor frame and has no extrinsic mechanism of
            # its own -- same situation as DLO and RTAB-Map, so it reuses the same rotator.
            ["python3", str(NODES / "shared/imu_frame_rotator.py")],
            ["python3", str(NODES / "group4/gt_initial_pose_publisher.py"),
             "_lidar_poses_csv:=/data/boreas/{sequence}/applanix/lidar_poses.csv", "_map_frame:=map"],
        ],
    },
    "fastlio_localization": {
        "group": "group4_map_localization",
        # /localization is T_map_to_odom composed with FAST-LIO's own odometry, and FAST-LIO's
        # odometry is the IMU body-frame state (laserMapping.cpp publishes the filter state
        # directly, no extrinsic applied on output). Same frame as Group 1's fastlio2, hence "imu"
        # rather than the "lidar" the rest of Group 4 uses.
        "gt_frame": "imu",
        # A different shape of localizer from the other two in this group: instead of matching each
        # scan to the map, it runs FAST-LIO as a full LIO front end and corrects its accumulated
        # drift against the map by ICP at 0.5Hz. That makes it the group's only entry whose
        # short-term motion comes from inertial odometry rather than from the map, which is the
        # comparison worth having here -- and it also means a failed map correction degrades
        # quietly into odometry rather than visibly into a frozen or diverging pose, so the
        # correction rate is reported alongside the trajectory error rather than assumed.
        "roslaunch": [
            "fast_lio_localization", "localization_boreas.launch",
            "map:=/data/boreas/{sequence}/ground_truth_map.pcd",
        ],
        "odom_topic": "/localization",
        "capture_launch_log": True,
        "launch_log_counters": {"icp_cycles": "Time: ", "icp_rejected": "Not match"},
        "skip_sequences": [],
        "aux_procs": [
            # imu_frame:=true -- this is the one Group 4 algorithm tracking the IMU body frame, and
            # this rig's LiDAR-to-IMU extrinsic is a 179.98-degree rotation. Seeding the ICP with
            # the LiDAR pose instead would put the initial guess essentially backwards, which no
            # convergence basin recovers from. See the node's docstring.
            ["python3", str(NODES / "group4/gt_initial_pose_publisher.py"),
             "_lidar_poses_csv:=/data/boreas/{sequence}/applanix/lidar_poses.csv",
             "_map_frame:=map", "_imu_frame:=true"],
        ],
    },
}

AUX_PROC_STARTUP_GRACE_S = 2.0
NODE_STARTUP_GRACE_S = 3.0
RECORD_SUBSCRIBE_GRACE_S = 1.0
TAIL_DRAIN_GRACE_S = 3.0
SAMPLE_INTERVAL_S = 0.5


def bag_path_for_sequence(sequence):
    return DATASET_ROOT / f"{sequence.replace('-', '_')}.bag"


def ground_truth_path_for(sequence, algo_cfg):
    """Ground truth in the body frame THIS algorithm reports in.

    `format_gt.py` writes the same rigid-body motion twice, once per body frame. A trajectory may
    only be scored against the ground truth in its own frame: the two are a full 180 degrees apart
    (T_lidar_dmu), and scoring across them leaves that rotation sitting between the trajectories.
    ATE barely notices (it compares positions, and the 0.28m lever arm is nearly pure Z, which
    align() absorbs -- measured impact under 2mm), but body-frame RPE does: up to 47x on
    translation, and 16x on rotation when the offset is off the yaw axis.

    Almost everything here reports in the LiDAR frame; FAST-LIO2 is the exception, publishing its
    IMU state directly. Scoring one against the other's ground truth leaves a 180-degree
    rotation between the trajectories, which body-frame RPE picks up even where ATE barely does.
    """
    return DATASET_ROOT / sequence / f"ground_truth_{algo_cfg['gt_frame']}.tum"


def ensure_roscore():
    """Starts a roscore if none is running. Returns its Popen handle, or None if one was already up."""
    try:
        rosgraph.Master("/collect_metrics").getPid()
        return None
    except Exception:
        pass
    proc = subprocess.Popen(["roscore"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True)
    for _ in range(50):
        try:
            rosgraph.Master("/collect_metrics").getPid()
            return proc
        except Exception:
            time.sleep(0.2)
    raise RuntimeError("roscore did not come up within 10s")


def terminate_process_group(proc, timeout=10):
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait()


class ResourceSampler:
    """Samples CPU% and RSS of a process tree at a fixed interval, on a background thread.

    Re-enumerates children every interval rather than caching the list once, since LIO-SAM's
    nodes are launched with respawn="true" and can restart mid-run.
    """

    def __init__(self, root_pids, interval_s):
        self.root_pids = root_pids if isinstance(root_pids, (list, tuple)) else [root_pids]
        self.interval_s = interval_s
        self.samples = []
        self._tracked = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=self.interval_s * 3)

    def _live_processes(self):
        procs = []
        for root_pid in self.root_pids:
            try:
                root = psutil.Process(root_pid)
            except psutil.NoSuchProcess:
                continue
            procs.append(root)
            procs.extend(root.children(recursive=True))
        for p in procs:
            if p.pid not in self._tracked:
                try:
                    p.cpu_percent(None)  # prime the baseline; first read is meaningless, discarded
                except psutil.NoSuchProcess:
                    continue
                self._tracked[p.pid] = p
        live_pids = {p.pid for p in procs}
        for pid in list(self._tracked):
            if pid not in live_pids:
                del self._tracked[pid]
        return list(self._tracked.values())

    def _run(self):
        time.sleep(self.interval_s)  # let the primed cpu_percent baselines establish
        while not self._stop.is_set():
            cpu_total, rss_total, n = 0.0, 0, 0
            for p in self._live_processes():
                try:
                    cpu_total += p.cpu_percent(None)
                    rss_total += p.memory_info().rss
                    n += 1
                except psutil.NoSuchProcess:
                    continue
            self.samples.append({
                "t": time.time(),
                "cpu_percent_total": cpu_total,
                "rss_bytes_total": rss_total,
                "num_processes": n,
            })
            self._stop.wait(self.interval_s)


def bag_duration_s(bag_path):
    bag = rosbag.Bag(str(bag_path))
    duration = bag.get_end_time() - bag.get_start_time()
    bag.close()
    return duration


def bag_to_tum(bag_path, topic, out_path):
    """Converts a recorded nav_msgs/Odometry bag to a TUM-format trajectory file."""
    bag = rosbag.Bag(str(bag_path))
    n = 0
    with open(out_path, "w") as f:
        for _, msg, _ in bag.read_messages(topics=[topic]):
            stamp = msg.header.stamp.to_sec()
            p, q = msg.pose.pose.position, msg.pose.pose.orientation
            f.write(f"{stamp:.6f} {p.x:.6f} {p.y:.6f} {p.z:.6f} {q.x:.6f} {q.y:.6f} {q.z:.6f} {q.w:.6f}\n")
            n += 1
    bag.close()
    return n


_PCD_TYPE_MAP = {
    ("F", 4): "f4", ("F", 8): "f8",
    ("U", 1): "u1", ("U", 2): "u2", ("U", 4): "u4", ("U", 8): "u8",
    ("I", 1): "i1", ("I", 2): "i2", ("I", 4): "i4", ("I", 8): "i8",
}


def _read_pcd_header(path):
    """Parses FIELDS/SIZE/TYPE/COUNT rather than assuming a fixed point type -- this project's map
    exporters use several different ones for the same binary PCD container (plain XYZI for
    LIO-SAM/DLO/HDL-Graph-SLAM, PointXYZINormal for FAST-LIO2). Same approach as
    metrics_lib.py's read_pcd_xyz(); duplicated in miniature here rather than imported, so this
    live collection harness doesn't pull in metrics_lib.py's heavier analysis dependencies
    (evo, matplotlib, scipy) just to downsample a file. Returns (dtype, n_points, data_offset)
    rather than the array itself -- callers memmap/chunk-read from data_offset, since a full-length
    FAST-LIO2 export can be 600M+ points and reading it into one array is exactly what caused a
    real OOM (see _voxel_downsample_pcd_inplace)."""
    with open(path, "rb") as f:
        fields = sizes = types = counts = n = None
        while True:
            line = f.readline().decode("ascii")
            if line.startswith("FIELDS"):
                fields = line.split()[1:]
            elif line.startswith("SIZE"):
                sizes = [int(x) for x in line.split()[1:]]
            elif line.startswith("TYPE"):
                types = line.split()[1:]
            elif line.startswith("COUNT"):
                counts = [int(x) for x in line.split()[1:]]
            elif line.startswith("POINTS"):
                n = int(line.split()[1])
            elif line.startswith("DATA"):
                break
        dtype_fields = [
            (name, _PCD_TYPE_MAP[(type_, size)]) if count == 1 else (name, _PCD_TYPE_MAP[(type_, size)], count)
            for name, size, type_, count in zip(fields, sizes, types, counts)
        ]
        return np.dtype(dtype_fields), n, f.tell()


def _voxel_downsample_pcd_inplace(path, voxel_size, chunk_points=20_000_000):
    """Rewrites a binary PCD in place, keeping one point per occupied voxel (first-seen), as plain
    XYZI float32 regardless of the source point type -- drops any extra fields (normals,
    curvature) this project's own metrics never use anyway (metrics_lib.py recomputes normals
    itself from XYZ). Disk space, not accuracy, is the concern: FAST-LIO2 in particular
    accumulates every raw point via its ikd-Tree with no aggressive downsampling of its own --
    confirmed to reach 605M points (19.4GB) on a real boreas-forest run (dense vegetation returns
    far more points per second than open-road sequences).

    Memory-bounded by construction, not just "usually small enough" -- a real 605M-point run
    OOM-killed the *entire machine* under the previous version, which read the whole file into one
    array and derived several more full-length arrays from it (xyz copy, int64 voxel indices,
    packed keys, plus np.unique's own internal copy) -- 50-60GB simultaneously for that file, a
    genuine reproduction, not a worst-case estimate. Also found and fixed along the way:
    `intensity = data["intensity"]` is a *view* into the structured array, not a copy, so the old
    code's `del data` never actually freed that buffer as its own comment claimed.

    Two passes over an np.memmap (never loads the raw file into process RAM at once -- the OS
    pages it in on demand):
      1. Chunk through the file computing each point's packed voxel key into one preallocated
         int64 array (the only full-length array this function ever holds -- 8 bytes/point, ~4.8GB
         even at 605M points, vs. the ~32-57GB the previous full-array approach needed for the
         same file).
      2. np.unique() on that key array alone (still just one array, not competing with xyz/idx/data
         copies) gives first-seen indices, sorted for sequential access, then re-read *only* those
         specific rows from the memmap for the final (much smaller, one-row-per-voxel) output --
         never materializes xyz/intensity for all N raw points, only for the deduplicated result.
    """
    dtype, n, offset = _read_pcd_header(path)
    mm = np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(n,))
    has_intensity = "intensity" in dtype.names

    keys = np.empty(n, dtype=np.int64)
    voxel_size_f32 = np.float32(voxel_size)
    for start in range(0, n, chunk_points):
        end = min(start + chunk_points, n)
        chunk = mm[start:end]
        # float32, not float64: this is a floor-division voxel-hash dedup, not the numerically
        # sensitive SVD/eigendecomposition work metrics_lib.py's analysis does -- float32's
        # ~1.2mm ULP at even a 10km coordinate magnitude is far finer than any voxel_size this
        # project uses, so there's no real precision to gain, only memory to spend.
        xyz_chunk = np.column_stack([chunk["x"], chunk["y"], chunk["z"]])
        idx_chunk = np.floor(xyz_chunk / voxel_size_f32).astype(np.int64) + 1_000_000
        keys[start:end] = (idx_chunk[:, 0] << 42) | (idx_chunk[:, 1] << 21) | idx_chunk[:, 2]

    _, first_idx = np.unique(keys, return_index=True)
    del keys
    first_idx.sort()  # sequential memmap access instead of scattered random reads

    kept = mm[first_idx]
    xyz_out = np.column_stack([kept["x"], kept["y"], kept["z"]]).astype(np.float32)
    intensity_out = (kept["intensity"] if has_intensity
                      else np.zeros(len(kept), dtype=np.float32)).astype(np.float32)
    del mm, kept

    n_out = len(xyz_out)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n"
        f"WIDTH {n_out}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n_out}\nDATA binary\n"
    )
    out = np.column_stack([xyz_out, intensity_out])
    # Writing to a temp file first: `path` is memmap'd open (mm/kept only just went out of scope
    # above, but the underlying mmap object's finalization isn't guaranteed synchronous) and is
    # also the same file this function was handed to overwrite in place -- truncating it via a
    # bare "wb" open while anything still references the old mapping is asking for corruption.
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(np.ascontiguousarray(out).tobytes())
    tmp_path.replace(path)


def run_once(algo_name, algo_cfg, sequence, run_idx, raw_dir, max_duration_s=None):
    bag_in = bag_path_for_sequence(sequence)
    if not bag_in.exists():
        raise FileNotFoundError(f"Sequence bag not found: {bag_in}")

    run_tag = f"{algo_name}_{sequence}_run{run_idx}"
    record_bag = raw_dir / f"{run_tag}_record.bag"
    est_tum = raw_dir / f"{run_tag}_est.tum"
    sysstats_csv = raw_dir / f"{run_tag}_sysstats.csv"
    meta_json = raw_dir / f"{run_tag}_meta.json"

    if "roslaunch" in algo_cfg:
        pkg, launch_file, *extra_args = algo_cfg["roslaunch"]
        launch_argv = ["roslaunch", pkg, launch_file, *extra_args]
    else:
        launch_argv = algo_cfg["launch_cmd"]
    launch_argv = [arg.format(sequence=sequence, run=run_idx) for arg in launch_argv]

    # Ensure the shutdown-file export's target directory exists before launch: FAST-LIO2's
    # pcl::PCDWriter::writeBinary does NOT create missing parent directories -- it throws
    # pcl::IOException ("Error during open!"), uncaught, which SIGABRTs the whole process.
    # Confirmed directly via gdb: the save code path runs correctly (pcd_save_en true,
    # pcl_wait_save non-null and populated) and still fails, purely because its own package's
    # "PCD/" subdirectory didn't exist. That directory isn't guaranteed to survive a fresh
    # reproduction (fetch_algorithms.sh doesn't create it, and it doesn't come back on its own
    # once removed) -- worth being self-sufficient about rather than depending on it happening to
    # exist. Since stdout/stderr are suppressed for CPU/RAM measurement cleanliness, this failure
    # is otherwise completely invisible: the process exits via SIGABRT, which
    # terminate_process_group doesn't distinguish from a clean exit, and run_once() simply
    # observes no map file and moves on. LIO-SAM's own save-map service already does its own
    # `mkdir -p`, so this is FAST-LIO2-specific, but harmless to run unconditionally.
    if algo_cfg.get("map_export_shutdown_file"):
        shutdown_file = algo_cfg["map_export_shutdown_file"].format(sequence=sequence, run=run_idx)
        Path(shutdown_file).parent.mkdir(parents=True, exist_ok=True)

    # /use_sim_time is a GLOBAL rosparam, and this harness deliberately reuses one roscore across
    # every run (ensure_roscore() only starts one if none is up). RTAB-Map's launch sets it true
    # because it genuinely needs /clock (needs_clock), and roslaunch never unsets it afterwards --
    # so it persists on the master for every subsequent run.
    #
    # The consequence is severe and completely silent: an algorithm that does NOT publish /clock
    # inherits use_sim_time=true, every node's rospy.Time.now() stays pinned at 0.0 forever, TF
    # lookups never resolve, and nothing is ever published. Confirmed directly -- an hdlgraphslam
    # run launched after an rtabmap run recorded ZERO poses on its odometry topic, and since
    # rosbag record never receives a message it never creates its output file at all, so the run
    # dies later with a confusing FileNotFoundError on a bag that was never written rather than
    # anything pointing at the actual cause.
    #
    # Setting it explicitly per run (not just when true) makes each run independent of whatever
    # ran before it, which is the property the rest of this harness already assumes.
    subprocess.run(
        ["rosparam", "set", "/use_sim_time", "true" if algo_cfg.get("needs_clock") else "false"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    # aux_procs start BEFORE the main algorithm, not alongside it: several algorithms do one-shot
    # startup reads (a topic subscription that needs its first message, a rosparam that must
    # already be set) with no retry after failure, so anything an aux_proc provides has to be live
    # *before* that exact instant (confirmed directly for an earlier TF-lookup approach:
    # starting both together, even from the same Python call, was enough to lose that race
    # consistently -- part of why it now uses the tf_baselink2primarylidar rosparam fallback
    # instead of a live TF publisher at all).
    # e.g. DLO's IMU-frame rotator (imu_frame_rotator.py) is also here -- genuinely part of running
    # the algorithm correctly on this rig, so its CPU/RAM cost counts toward the algorithm's total.
    # "{sequence}" in any arg is substituted with the actual sequence name -- Group 4's map/
    # initial-pose publishers need per-sequence file paths, unlike every other aux_proc so far.
    # ROS parameters outlive the node that set them, and ensure_roscore() reuses a master that is
    # already up -- so one algorithm's private params leak into the next algorithm's identically
    # named node. This is not hypothetical. Group 4's algorithms all run
    # gt_initial_pose_publisher.py, and when one of them set a seed-position override that the
    # next one did not, the override survived on the shared master: the second algorithm's seed
    # silently became (0,0,0) instead of the sequence's real start -- harmless on sequences that
    # begin at the origin, and a 51.7km error on boreas-forest, which begins 51.7km away.
    # Each helper node's name matches its script's basename, so clear those namespaces first.
    _master = rosgraph.Master("/collect_metrics")
    for cmd in algo_cfg.get("aux_procs", []):
        for arg in cmd:
            if arg.endswith(".py"):
                try:
                    _master.deleteParam("/" + Path(arg).stem)
                except Exception:
                    pass  # never set on this master -- nothing to clear
                break

    # Helper-node output goes to a per-run log rather than DEVNULL. Discarding it has cost this
    # project real time three times over: a CATKIN_IGNORE'd package meant roslaunch never started
    # and the run silently produced 0 poses; a frame-name mismatch terminate()d a whole nodelet
    # manager with no visible error; and a helper that exits on a bad argument looks exactly like
    # one that ran fine. A run that produces nothing should say why.
    aux_log_path = raw_dir / f"{run_tag}_auxprocs.log"
    aux_log = open(aux_log_path, "w")
    aux_procs = [
        subprocess.Popen(
            [arg.format(sequence=sequence) for arg in cmd],
            stdout=aux_log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        for cmd in algo_cfg.get("aux_procs", [])
    ]
    if aux_procs:
        time.sleep(AUX_PROC_STARTUP_GRACE_S)

    # roslaunch output is discarded by default: over a 1000s run most of these nodes are verbose
    # and none of it is needed. An algorithm opts in with "capture_launch_log" when something it
    # prints is actually evidence -- see fastlio_localization, whose ICP correction either fires or
    # silently does not, with no difference visible in the trajectory file.
    launch_log = None
    launch_log_path = None
    if algo_cfg.get("capture_launch_log"):
        launch_log_path = raw_dir / f"{run_tag}_launch.log"
        launch_log = open(launch_log_path, "w")
    launch_proc = subprocess.Popen(
        launch_argv,
        stdout=launch_log if launch_log else subprocess.DEVNULL,
        stderr=subprocess.STDOUT if launch_log else subprocess.DEVNULL,
        start_new_session=True,
    )
    time.sleep(NODE_STARTUP_GRACE_S)

    sampler = ResourceSampler([launch_proc.pid] + [p.pid for p in aux_procs], SAMPLE_INTERVAL_S)
    sampler.start()

    record_proc = subprocess.Popen(
        ["rosbag", "record", "-O", str(record_bag), algo_cfg["odom_topic"]],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
    )
    time.sleep(RECORD_SUBSCRIBE_GRACE_S)

    play_cmd = ["rosbag", "play", "-r", "1.0", str(bag_in)]
    if max_duration_s is not None:
        play_cmd += ["-u", str(max_duration_s)]  # smoke-test mode: play only the first N seconds
    if algo_cfg.get("needs_clock"):
        play_cmd += ["--clock"]  # only algorithms with use_sim_time:=true care; harmless to others

    play_start = time.time()
    play_result = subprocess.run(play_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    wall_clock_s = time.time() - play_start

    # Per-algorithm override for algorithms carrying a deep subscriber queue that can need much
    # longer than "a few last scans" to drain (see kissicp's registry entry).
    time.sleep(algo_cfg.get("drain_grace_s", TAIL_DRAIN_GRACE_S))

    # Map export, live-service variant (HDL-Graph-SLAM): must happen here, before the node is
    # killed -- /hdl_graph_slam/save_map is a ROS service on the still-running node, not a
    # post-hoc file operation. The DB-based variant (RTAB-Map) happens after teardown instead,
    # since it reads a database file that's only safely readable once the node has cleanly closed
    # it (see below).
    map_path = None
    if algo_cfg.get("map_export_service"):
        map_path = raw_dir / f"{run_tag}_map.pcd"
        svc = algo_cfg["map_export_service"]
        req_fields = {**svc.get("request", {}), "destination": str(map_path)}
        req_str = "{" + ", ".join(
            f"{k}: '{v}'" if isinstance(v, str) else f"{k}: {v}" for k, v in req_fields.items()
        ) + "}"
        subprocess.run(["rosservice", "call", svc["name"], req_str],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not map_path.exists():
            map_path = None  # service call failed silently (e.g. node already gone) -- don't claim a map exists

    # Map export, live-service-into-a-directory variant (DLO): DLO's own save_pcd service takes a
    # DIRECTORY and always writes a fixed filename ("dlo_map.pcd") inside it, rather than accepting
    # a full destination path the way HDL-Graph-SLAM's does -- so the directory itself must be
    # unique per run (a shared scratch dir would let a later run's rename step silently pick up an
    # earlier run's file if the service call ever failed silently). Safe to create/reuse freely:
    # unlike LIO-SAM's save-map mechanism below, DLO's service never deletes anything.
    if algo_cfg.get("map_export_service_dir"):
        svc = algo_cfg["map_export_service_dir"]
        scratch_dir = Path(f"/root/dlo_map_export_{run_tag}")
        scratch_dir.mkdir(parents=True, exist_ok=True)
        req_fields = {**svc.get("request", {}), svc["dest_field"]: str(scratch_dir)}
        req_str = "{" + ", ".join(
            f"{k}: '{v}'" if isinstance(v, str) else f"{k}: {v}" for k, v in req_fields.items()
        ) + "}"
        subprocess.run(["rosservice", "call", svc["name"], req_str],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        written = scratch_dir / svc["output_filename"]
        if written.exists():
            map_path = raw_dir / f"{run_tag}_map.pcd"
            # shutil.move, not Path.rename(): the scratch dir is on the container's own
            # filesystem layer while raw_dir is a bind-mounted host directory -- a different
            # device, so a plain rename (an atomic same-device operation) fails with "Invalid
            # cross-device link" (confirmed directly). shutil.move falls back to copy+delete.
            shutil.move(str(written), str(map_path))
        shutil.rmtree(scratch_dir, ignore_errors=True)

    terminate_process_group(record_proc)
    sampler.stop()
    # Longer grace period for any algorithm that builds a persistent map -- originally added only
    # for FAST-LIO2/LIO-SAM's map_export_shutdown_file (SIGINT triggers the write itself, and a
    # full-length run's map can be tens of millions of points), but a real 600s+ HDL-Graph-SLAM run
    # showed the same class of problem for a different reason: its map_export_service call
    # finishes fine *before* this point (the export already happened live, earlier in run_once()),
    # but the nodelet manager itself -- now holding a full pose graph plus every keyframe's point
    # cloud after 1000+ seconds, not just a 30s smoke test's worth -- simply takes longer than 10s
    # to tear down cleanly regardless of whether a map write is still pending. Confirmed directly:
    # the exported map was complete and uncorrupted (header POINTS count matched actual file size
    # exactly) despite the SIGKILL, so this was purely a too-short teardown window, not data loss --
    # but a real one, and not unique to the shutdown-file export mechanism. Generalized to any
    # map-building algorithm rather than re-special-casing per mechanism: strictly safe to widen
    # even for algorithms that already tear down fast (DLO's own real-duration test completed
    # cleanly at the default 10s) since terminate_process_group returns as soon as the process
    # actually exits, never waiting out the full timeout unnecessarily.
    builds_a_map = any(algo_cfg.get(k) for k in (
        "map_export_shutdown_file", "map_export_service", "map_export_service_dir", "map_export_db",
    ))
    launch_term_timeout = 60 if builds_a_map else 10
    terminate_process_group(launch_proc, timeout=launch_term_timeout)
    for p in aux_procs:
        terminate_process_group(p)

    # We only ever send SIGINT (2) to launch_proc; a returncode reflecting any OTHER signal means
    # the algorithm's own shutdown handling crashed on the way out (uncaught exception -> SIGABRT,
    # a segfault, etc.) rather than exiting the way we asked it to. Worth a loud, visible warning
    # -- this exact blind spot cost real debugging time once already (FAST-LIO2's shutdown-triggered
    # PCD write threw an uncaught pcl::IOException and SIGABRT'd, completely invisible otherwise
    # since stdout/stderr are suppressed and terminate_process_group's proc.wait() doesn't
    # distinguish a crash from a clean exit). print(), not suppressed: this is collect_metrics.py's
    # own stdout, not the launched subprocess's.
    if launch_proc.returncode is not None and launch_proc.returncode < 0 and -launch_proc.returncode != signal.SIGINT:
        print(f"[WARN] {run_tag}: {algo_name} exited via signal {-launch_proc.returncode}, not the "
              f"SIGINT we sent -- likely crashed during its own shutdown handling. Check for a "
              f"silently-failed map export or other cleanup step.")

    # Map export, shutdown-file variant (FAST-LIO2, LIO-SAM): both write their own map to a fixed
    # path as part of their normal clean-shutdown handler when configured to (FAST-LIO2's
    # `pcd_save_en`, set in its own YAML config; LIO-SAM's `savePCD`, set by configure_group1.py) --
    # no live service call needed from us, just a copy-and-rename after the node has exited and
    # finished writing. LIO-SAM's own save-map service (used internally by its shutdown handler)
    # does `rm -r` on its destination directory with no safety check, so that directory
    # (savePCDDirectory) is always a small, dedicated, disposable scratch path -- never anything
    # this project actually values -- and the file we want is copied out before the *next* run
    # would recreate and reuse that same directory.
    if algo_cfg.get("map_export_shutdown_file"):
        source = Path(algo_cfg["map_export_shutdown_file"].format(sequence=sequence, run=run_idx))
        if source.exists():
            map_path = raw_dir / f"{run_tag}_map.pcd"
            shutil.copy2(source, map_path)
            # See _voxel_downsample_pcd_inplace(): both of these accumulate every raw point with
            # no aggressive downsampling of their own -- 0.05m keeps native wall-thickness detail
            # (finer than the project's 0.3-0.5m live-processing target) while keeping file size
            # sane at full run length.
            _voxel_downsample_pcd_inplace(map_path, voxel_size=0.05)

    # Map export, database variant (RTAB-Map): the node's `database_path` (see its roslaunch args,
    # templated with the same {sequence}/{run} used above) is only safe to read once the node has
    # actually exited and closed its SQLite connection -- hence after terminate_process_group,
    # not before. rtabmap-export is the same CLI tool RTAB-Map's own documentation uses to pull an
    # assembled point cloud out of a .db file after the fact.
    if algo_cfg.get("map_export_db"):
        db_path = algo_cfg["map_export_db"].format(sequence=sequence, run=run_idx)
        if os.path.exists(db_path):
            map_stem = f"{run_tag}_map"
            # --scan: this dataset has no depth/stereo, only LiDAR scans -- without it,
            # rtabmap-export tries to build the cloud from camera data and silently produces an
            # empty (0-point) export instead (confirmed directly). Default optimization mode (full
            # global optimization, not --opt 3/raw odometry) is deliberate: RTAB-Map's whole
            # purpose in this benchmark is loop-closure graph optimization, so the exported map
            # must reflect the optimized graph, not throw that away.
            subprocess.run(
                ["rtabmap-export", "--cloud", "--scan", "--output_dir", str(raw_dir), "--output", map_stem, db_path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            candidate = raw_dir / f"{map_stem}_cloud.ply"  # rtabmap-export appends "_cloud" itself
            map_path = candidate if candidate.exists() else None

    n_poses = bag_to_tum(record_bag, algo_cfg["odom_topic"], est_tum)
    full_bag_dur = bag_duration_s(bag_in)
    bag_dur = min(full_bag_dur, max_duration_s) if max_duration_s is not None else full_bag_dur
    rtf = bag_dur / wall_clock_s if wall_clock_s > 0 else None

    with open(sysstats_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["t", "cpu_percent_total", "rss_bytes_total", "num_processes"])
        writer.writeheader()
        writer.writerows(sampler.samples)

    meta = {
        "algorithm": algo_name,
        "sequence": sequence,
        "run": run_idx,
        "odom_topic": algo_cfg["odom_topic"],
        "n_poses_recorded": n_poses,
        "full_bag_duration_s": full_bag_dur,
        "max_duration_s": max_duration_s,
        "bag_duration_s": bag_dur,
        "wall_clock_s": wall_clock_s,
        "real_time_factor": rtf,
        "rosbag_play_exit_code": play_result.returncode,
        "cpu_count": psutil.cpu_count(),
        "ground_truth_path": str(ground_truth_path_for(sequence, algo_cfg)),
        "estimated_trajectory_path": str(est_tum),
        "sysstats_path": str(sysstats_csv),
        "map_path": str(map_path) if map_path else None,
    }

    # Counters pulled out of the captured roslaunch log. For fastlio_localization these are the
    # ICP cycle count and how many of those were rejected for low fitness: a run whose correction
    # never fires degrades into pure odometry and still writes a perfectly plausible trajectory, so
    # the rate is recorded rather than inferred from the error.
    if launch_log is not None:
        launch_log.close()
    if launch_log_path is not None and launch_log_path.exists():
        log_text = launch_log_path.read_text(errors="replace")
        counters = {name: log_text.count(marker)
                    for name, marker in algo_cfg.get("launch_log_counters", {}).items()}
        meta.update(counters)
        if counters.get("icp_cycles"):
            meta["icp_correction_rate"] = round(
                1.0 - counters.get("icp_rejected", 0) / counters["icp_cycles"], 4)
    with open(meta_json, "w") as f:
        json.dump(meta, f, indent=2)

    record_bag.unlink(missing_ok=True)  # superseded by est_tum; keeps raw/ lean

    rtf_str = f"{rtf:.3f}" if rtf else "undefined"
    print(f"[{'OK' if n_poses > 0 else 'WARN'}] {run_tag}: {n_poses} poses, RTF={rtf_str}")
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algorithm", required=True, choices=sorted(ALGORITHMS))
    parser.add_argument("--sequences", nargs="+", default=SEQUENCES, choices=SEQUENCES)
    parser.add_argument("--runs", type=int, default=3,
                         help="Collect runs 1..N. Ignored if --run-numbers is given.")
    parser.add_argument("--run-numbers", type=int, nargs="+", default=None,
                         help="Explicit run indices to collect (e.g. --run-numbers 2 3), applied "
                              "to every sequence given. Lets you top up to N runs incrementally "
                              "-- e.g. collect run 1 today, runs 2 and 3 next week -- without ever "
                              "redoing a run you already have. Overrides --runs.")
    parser.add_argument("--max-duration", type=float, default=None,
                         help="Play only the first N seconds of each bag (smoke-test mode)")
    args = parser.parse_args()

    algo_cfg = ALGORITHMS[args.algorithm]
    raw_dir = RESULTS_ROOT / algo_cfg["group"] / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    run_indices = args.run_numbers if args.run_numbers is not None else list(range(1, args.runs + 1))

    # Some algorithms genuinely cannot run a given sequence (see each entry's "skip_sequences" note).
    # Skipping loudly rather than silently: a sequence quietly missing from results is exactly the
    # kind of gap this project has been bitten by before.
    skip = set(algo_cfg.get("skip_sequences", []))
    sequences = [s for s in args.sequences if s not in skip]
    for s in args.sequences:
        if s in skip:
            print(f"[SKIP] {args.algorithm} does not run {s} -- see its skip_sequences note "
                  f"in collect_metrics.py")
    if not sequences:
        print("[SKIP] nothing left to run after exclusions")
        return

    started_roscore = ensure_roscore()
    try:
        for sequence in sequences:
            for run_idx in run_indices:
                run_once(args.algorithm, algo_cfg, sequence, run_idx, raw_dir,
                         max_duration_s=args.max_duration)
    finally:
        terminate_process_group(started_roscore)


if __name__ == "__main__":
    main()
