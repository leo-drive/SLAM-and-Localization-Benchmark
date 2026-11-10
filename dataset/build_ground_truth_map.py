#!/usr/bin/env python3
"""
Builds a ground-truth-derived reference map for Group 4 (map-based localization:
hdl_localization, FAST-LIO-Localization) by accumulating raw LiDAR scans transformed by the real
ground-truth poses --
deliberately NOT reusing a Group 3 algorithm's own (imperfect) map, so Group 4's evaluation isn't
confounded by Group 3's own mapping errors (drift, ghosting, winter ground-plane failures). See
confounded by Group 3's own mapping errors.

Reads applanix/lidar_poses.csv directly rather than ground_truth_imu.tum: lidar_poses.csv already
gives T_world_lidar (no need to compose/invert T_lidar_dmu extrinsics -- raw scans are already in
the lidar frame), and its GPSTime column matches scan filenames as *exact* integers (verified:
"1739657987565613" row <-> "1739657987565613.bin"), avoiding any float round-trip risk from
ground_truth_imu.tum's %.6f-formatted seconds.

Does NOT assume row N <-> scan file N: boreas-snowy was found to have 11092 GT rows but only
11085 scan files (7 missing, presumably dropped during collection) -- matches by exact GPSTime
instead, skipping any GT row without a corresponding scan file.

Processes in batches and voxel-compacts as it goes rather than accumulating every raw scan in
memory at once -- a full sequence can be ~19k scans x ~190k points, far too much to hold raw.

Usage: ./build_ground_truth_map.py boreas-suburb [--keyframe-dist 3.0] [--voxel-size 1.0]
"""

import argparse
import glob
import os

import numpy as np
from scipy.spatial.transform import Rotation as R

from boreas2rosbag import MIN_ROOF_DISTANCE_SQ

BATCH_SIZE = 50  # keyframes accumulated raw before voxel-compacting, bounds peak memory


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


def load_scan_xyzi(bin_path, max_range):
    """Returns x, y, z, intensity, t -- where t is the raw file's per-point acquisition time
    offset (column 5, seconds, spanning one +/-0.052s sweep). That column used to be discarded,
    which is what made the accumulated map unusable for localization: pasting a whole 0.104s sweep
    into the map at one single pose smears it by however far the vehicle travelled during the
    sweep (0.8m at 8m/s). Measured on the resulting map, only 0.04% of points were locally planar
    against 22.6% for an algorithm's own map of the same street, and zero wall planes were
    detectable at all -- NDT had nothing crisp to lock onto. See deskew() below."""
    raw = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 6)
    dist_sq = raw[:, 0] ** 2 + raw[:, 1] ** 2 + raw[:, 2] ** 2
    # Long-range returns are the dominant driver of map size: at range, angular resolution spreads
    # a few points across a wide swath of world-frame voxels per keyframe, and consecutive
    # keyframes see the same far surface from different angles, so it barely overlaps/dedups.
    # Capping range keeps the map to what's actually useful for localization matching.
    valid = raw[(dist_sq > MIN_ROOF_DISTANCE_SQ) & (dist_sq < max_range ** 2)]
    return valid[:, [0, 1, 2, 3, 5]]  # x, y, z, intensity, per-point time offset


def deskew(points_xyz, point_dt, T_this, T_next, dt_scan):
    """Motion-compensate one sweep into the frame of T_this.

    Each point was measured at a slightly different instant while the vehicle kept moving, so the
    sensor pose differs across the sweep. Rather than trusting the CSV's velocity columns (whose
    frame convention would have to be assumed), the correction is interpolated from the pose
    trajectory itself: the body-frame motion from this scan to the next is scaled by each point's
    own fraction of the scan interval. First-order in rotation, which is ample here -- the vehicle
    turns at most a couple of degrees within a single 0.104s sweep.
    """
    if T_next is None or dt_scan <= 0:
        return points_xyz
    delta = np.linalg.inv(T_this) @ T_next          # body-frame motion over one scan interval
    rotvec = R.from_matrix(delta[:3, :3]).as_rotvec()
    trans = delta[:3, 3]
    frac = (point_dt / dt_scan)[:, None]            # where in the interval each point falls
    # p' = p + (frac*rotvec) x p + frac*trans
    return points_xyz + np.cross(frac * rotvec, points_xyz) + frac * trans


VOXEL_KEY_OFFSET = 1_000_000  # keeps packed indices non-negative; safe well beyond map extent


def voxel_keys(points_xyz, voxel_size):
    """Packs (x,y,z) voxel indices into one int64 (21 bits/axis) for fast 1D set ops -- axis=0
    np.unique on the raw float array is dramatically slower at this scale (tested: got stuck for
    15+ minutes on a single ~9.5M-point batch) since it row-wise-compares instead of sorting a
    flat array of primitives."""
    idx = np.floor(points_xyz / voxel_size).astype(np.int64) + VOXEL_KEY_OFFSET
    return (idx[:, 0] << 42) | (idx[:, 1] << 21) | idx[:, 2]


class VoxelMap:
    """Incrementally accumulates unique voxel-downsampled points across many batches without
    ever re-comparing the full growing map row-by-row -- keeps a sorted int64 key array and uses
    np.isin/argsort (fast 1D primitive ops) to fold in only genuinely new voxels each batch."""

    def __init__(self, voxel_size):
        self.voxel_size = voxel_size
        self.keys = np.empty(0, dtype=np.int64)
        self.points = np.empty((0, 4), dtype=np.float32)

    def add_batch(self, points_xyzi):
        keys = voxel_keys(points_xyzi[:, :3], self.voxel_size)
        _, first_idx = np.unique(keys, return_index=True)  # dedup within the batch itself
        keys, points = keys[first_idx], points_xyzi[first_idx]

        is_new = ~np.isin(keys, self.keys, assume_unique=True) if self.keys.size else np.ones(len(keys), dtype=bool)
        if not is_new.any():
            return

        merged_keys = np.concatenate([self.keys, keys[is_new]])
        merged_points = np.vstack([self.points, points[is_new]])
        order = np.argsort(merged_keys)
        self.keys, self.points = merged_keys[order], merged_points[order]

    def __len__(self):
        return len(self.keys)


def write_pcd(path, points_xyzi):
    n = points_xyzi.shape[0]
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z intensity\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        "DATA binary\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(np.ascontiguousarray(points_xyzi, dtype=np.float32).tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sequence", help="e.g. boreas-suburb")
    parser.add_argument("--keyframe-dist", type=float, default=3.0,
                         help="minimum travel distance (m) between accumulated scans")
    parser.add_argument("--voxel-size", type=float, default=1.0,
                         help="map voxel downsample size (m) -- coarser than the project's live "
                              "per-scan 0.3-0.5m target on purpose: a static reference map covering "
                              "a whole sequence doesn't need per-scan density (NDT-based localizers "
                              "re-voxelize the map internally at ~1-2m cells anyway), and finer "
                              "settings produced impractically huge maps (246MB/15.4M points at "
                              "0.5m on just the shortest sequence; 1.0m gives 53MB/3.3M points)")
    parser.add_argument("--max-range", type=float, default=50.0,
                         help="drop points beyond this range (m) -- long-range returns barely "
                              "overlap between keyframes and dominate final map size for little "
                              "localization-matching value; also tested and reduced, see above")
    parser.add_argument("--data-root", default=None, help="directory holding the per-sequence folders; defaults to $BOREAS_DATASET, then to this script's own directory (where these scripts originally lived, beside the data)")
    args = parser.parse_args()

    seq_dir = os.path.join(resolve_data_root(args.data_root), args.sequence)
    poses_csv = os.path.join(seq_dir, "applanix", "lidar_poses.csv")
    scan_by_gpstime = {
        int(os.path.basename(p).replace(".bin", "")): p
        for p in glob.glob(os.path.join(seq_dir, "lidar", "*.bin"))
    }

    poses = np.genfromtxt(poses_csv, delimiter=",", names=True)
    n_missing = 0

    voxel_map = VoxelMap(args.voxel_size)
    batch = []
    last_kept_pos = None
    n_keyframes = 0

    for i, row in enumerate(poses):
        gpstime = int(row["GPSTime"])
        scan_path = scan_by_gpstime.get(gpstime)
        if scan_path is None:
            n_missing += 1
            continue

        pos = np.array([row["easting"], row["northing"], row["altitude"]])
        if last_kept_pos is not None and np.linalg.norm(pos - last_kept_pos) < args.keyframe_dist:
            continue
        last_kept_pos = pos
        n_keyframes += 1

        # Applanix "heading" is a compass bearing (0=North, clockwise-positive), not a standard
        # math yaw (0=+X/East, counter-clockwise-positive) -- same bug found and fixed in
        # format_gt.py, independently present here since this script
        # builds its own rotation directly from the raw CSV rather than reusing that fix. Left
        # uncorrected, every scan gets rotated into the world frame by the wrong yaw before being
        # merged into the accumulated map, misaligning scan-to-scan structure (walls, poles,
        # curbs) even though each individual scan's own points are internally consistent.
        # Rotation convention determined EMPIRICALLY, by transforming pairs of real scans into the
        # world with each candidate and measuring how well they agree (median nearest-neighbour
        # distance between two scans ~2s apart). The previous "xyz [roll, pitch, pi/2 - heading]"
        # gave 4.117m; "ZYX [-heading, pitch, roll]" gives 0.240m -- a 17x difference, and the
        # reason the accumulated map was too blurred for NDT to localise against at all.
        #
        # Why a constant yaw offset was not harmless: it is applied on the LEFT (world side) while
        # the translation is taken from the CSV unchanged, so each scan gets rotated about its own
        # position rather than the whole map being rigidly rotated. The induced misalignment grows
        # with the distance between scan positions, which is exactly a smear.
        # Candidates measured: ZYX[-h,p,r] 0.240 | ZYX[-h,-p,r] 0.426 | ZYX[-h,p,-r] 0.430 |
        #                     ZYX[-h,-p,-r] 0.602 | zyx[-h,p,r] 0.644 | xyz[r,p,pi/2-h] 4.117
        T_world_lidar = np.eye(4)
        T_world_lidar[:3, :3] = R.from_euler(
            "ZYX", [-row["heading"], row["pitch"], row["roll"]]).as_matrix()
        T_world_lidar[:3, 3] = pos

        # Pose of the following row, used only to interpolate within this sweep.
        next_T, next_dt = None, 0.0
        if i + 1 < len(poses):
            nrow = poses[i + 1]
            nT = np.eye(4)
            nT[:3, :3] = R.from_euler(
                "ZYX", [-nrow["heading"], nrow["pitch"], nrow["roll"]]).as_matrix()
            nT[:3, 3] = [nrow["easting"], nrow["northing"], nrow["altitude"]]
            next_T = nT
            next_dt = (int(nrow["GPSTime"]) - gpstime) * 1e-6  # GPSTime is in microseconds

        scan = load_scan_xyzi(scan_path, args.max_range)
        # Deskew before transforming: without this the whole sweep lands at one pose (see
        # load_scan_xyzi's note) and the map is blurred by the distance travelled during it.
        xyz_lidar = deskew(scan[:, :3], scan[:, 4], T_world_lidar, next_T, next_dt)
        xyz_world = (T_world_lidar[:3, :3] @ xyz_lidar.T).T + T_world_lidar[:3, 3]
        batch.append(np.column_stack([xyz_world, scan[:, 3]]).astype(np.float32))

        if len(batch) >= BATCH_SIZE:
            voxel_map.add_batch(np.vstack(batch))
            batch = []
            print(f"[INFO] {i + 1}/{len(poses)} poses processed, {n_keyframes} keyframes, "
                  f"map at {len(voxel_map)} points")

    if batch:
        voxel_map.add_batch(np.vstack(batch))

    if n_missing:
        print(f"[WARN] {n_missing}/{len(poses)} GT poses had no matching scan file, skipped")

    print(f"[INFO] Done: {n_keyframes} keyframes kept, final map has {len(voxel_map)} points")

    out_path = os.path.join(seq_dir, "ground_truth_map.pcd")
    write_pcd(out_path, voxel_map.points)
    print(f"[OK] Wrote {out_path}")



if __name__ == "__main__":
    main()
