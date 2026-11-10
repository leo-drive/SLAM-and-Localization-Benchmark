#!/usr/bin/env python3
"""
Generates IMU-frame ground truth (TUM format) from Boreas-RT lidar_poses.csv.

Composes the full T_world_lidar (from the CSV) with T_lidar_dmu (from calib/,
via boreas2rosbag.extract_extrinsics — the single source of truth for this
extrinsic, so it can't drift out of sync with the rosbag conversion) to get
T_world_dmu, rather than applying a hardcoded translation-only lever arm.
"""

import os
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R

from boreas2rosbag import extract_extrinsics

SEQUENCES = ["boreas-forest", "boreas-highway", "boreas-snowy", "boreas-suburb", "boreas-urban"]


def resolve_data_root(explicit=None):
    """Where the per-sequence directories live.

    These scripts originally sat inside the dataset directory, so sequences resolved against
    __file__. They now ship in the repository instead, which would silently look in the wrong
    place -- hence the explicit precedence: --data-root, then $BOREAS_DATASET (the same variable
    docker-compose.yml reads), then the script's own directory for the original layout.
    """
    root = explicit or os.environ.get("BOREAS_DATASET") or os.path.dirname(os.path.abspath(__file__))
    if not os.path.isdir(root):
        raise SystemExit("data root %s does not exist -- pass --data-root or set BOREAS_DATASET" % root)
    return root


def convert_sequence(seq_dir):
    poses_csv = os.path.join(seq_dir, "applanix", "lidar_poses.csv")
    if not os.path.exists(poses_csv):
        print(f"[WARN] {poses_csv} not found, skipping.")
        return

    T_lidar_dmu = extract_extrinsics(seq_dir)
    R_lidar_dmu = R.from_matrix(T_lidar_dmu[:3, :3])
    t_lidar_dmu = T_lidar_dmu[:3, 3]

    df = np.genfromtxt(poses_csv, delimiter=",", names=True)
    t = df["GPSTime"] / 1e6
    t_world_lidar = np.column_stack([df["easting"], df["northing"], df["altitude"]])
    # Applanix "heading" is a compass bearing (0=North, clockwise-positive), not a standard
    # math yaw -- the CSV's (roll, pitch, heading) becomes a rotation matrix as ZYX[-heading,
    # pitch, roll]. Note from_euler("xyz",[r,p,q]) == from_euler("ZYX",[q,p,r]), so the ONLY
    # thing that changed from the previous version here is dropping a pi/2 from the yaw.
    #
    # That pi/2 was a real 90deg error, and it is worth recording how it survived review. It was
    # justified by checking that the resulting DMU frame's +X lines up with vel_east/vel_north,
    # which it does -- but "+X points forward" was an ASSUMPTION about how the DMU is mounted,
    # never a measured fact, and it is wrong: the DMU sits rotated 90deg from vehicle-forward.
    # The authority on what the DMU and LiDAR frames actually are is the rosbag the algorithms
    # consume, not a mounting guess. Re-derived against exactly that, with no pi/2:
    #
    #   GT(DMU)   vs FAST-LIO2 (publishes its IMU state directly, laserMapping.cpp:579)
    #   GT(LiDAR) vs LIO-SAM   (converts IMU into the LiDAR frame at input, imuConverter)
    #
    # residual body-frame offset over 5 sequences x 2 algorithms: 0.55-1.86 deg, where ~1.13 deg
    # is these algorithms' own independently-measured attitude error. With the pi/2 present,
    # every one of those ten combinations sat at 89.5-91.5 deg instead.
    R_world_lidar = R.from_euler(
        "ZYX", np.column_stack([-df["heading"], df["pitch"], df["roll"]]))

    # Two ground truths, not one: the same rigid-body motion expressed in two body frames. An
    # algorithm may only be scored against a ground truth in the frame it actually reports in,
    # and these two differ by a full 180 deg (T_lidar_dmu above). Pick the wrong one and a
    # constant rotation sits between the trajectories -- ATE absorbs it (it compares positions),
    # but body-frame RPE does not: measured on this data, that inflates RPE translation by up to
    # 47x, and when the offset is not about the yaw axis, RPE rotation by ~16x as well.
    #
    #   ground_truth_imu.tum    DMU/IMU frame -- FAST-LIO2, and anything estimating IMU state
    #   ground_truth_lidar.tum  LiDAR frame   -- LIO-SAM, KISS-ICP, DLO, Group 3 and Group 4
    #
    # T_world_dmu = T_world_lidar @ T_lidar_dmu, composed in full (rotation + translation).
    outputs = {
        "ground_truth_imu.tum": (R_world_lidar * R_lidar_dmu,
                                 t_world_lidar + R_world_lidar.apply(t_lidar_dmu)),
        "ground_truth_lidar.tum": (R_world_lidar, t_world_lidar),
    }
    for filename, (rot, trans) in outputs.items():
        tum_data = np.column_stack([t, trans, rot.as_quat()])
        out_path = os.path.join(seq_dir, filename)
        np.savetxt(out_path, tum_data, fmt="%.6f")
        print(f"[OK] Wrote {len(tum_data)} poses to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate IMU-frame ground truth (TUM format) from Boreas-RT lidar_poses.csv."
    )
    parser.add_argument(
        "sequence_dir",
        nargs="?",
        default=None,
        help="Path to a single sequence directory. Omit to process all known sequences.",
    )
    parser.add_argument("--data-root", default=None, help="directory holding the per-sequence folders; defaults to $BOREAS_DATASET, then to this script's own directory (where these scripts originally lived, beside the data)")
    args = parser.parse_args()

    root = resolve_data_root(args.data_root)
    targets = [args.sequence_dir] if args.sequence_dir else [os.path.join(root, s) for s in SEQUENCES]

    for seq_dir in targets:
        convert_sequence(seq_dir)
