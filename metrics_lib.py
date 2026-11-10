"""
Shared metrics-computation library for every group's results_notebook.ipynb. Not a standalone
script -- imported by each cloned notebook (benchmark_results/<group>/results_notebook.ipynb), via
sys.path.insert(0, "/root/boreas_slam_benchmark") in the notebook's first cell.

Deliberately avoids pandas: this container's numpy is pinned to 1.17.4 (ROS Noetic's apt-provided
version -- likely load-bearing for other ROS Python C-extension bindings built against it, so not
something to casually upgrade), too old for any pandas release actually installed here (confirmed:
`import pandas` raises `AttributeError: module 'numpy.random' has no attribute 'BitGenerator'`).
Plain dict/list structures + evo + matplotlib + IPython.display cover everything a results
notebook needs without touching the container's global package set.

Pose accuracy is computed directly from each run's `_est.tum` against the ground truth for the
frame that algorithm actually reports in -- `ground_truth_imu.tum` or `ground_truth_lidar.tum`,
selected by each algorithm's `gt_frame` in collect_metrics.py. Those two files differ by this rig's
LiDAR-IMU extrinsic, a 179.98-degree rotation, so scoring against the wrong one leaves translation
almost unchanged while making every rotation-dependent metric meaningless. Via evo (already
installed): ATE (Absolute Trajectory Error, translation)
and RPE (Relative Pose Error, translation + rotation) over a fixed real-world distance rather than
a fixed frame count -- algorithms publish odometry at very different rates (confirmed across this
project's smoke tests: e.g. ~1.5Hz vs FAST-LIO2's ~6Hz), so a frame-count delta
would compare a different physical distance per algorithm. Alignment is SE(3) only (no scale
correction): LiDAR odometry is already metric-scale, unlike monocular VO where Sim(3) alignment is
the norm -- correcting scale here would hide a real drift-in-scale failure mode instead of
measuring it.

Compute/systems metrics (RTF, CPU%, RAM) are read directly from each run's `_meta.json`/
`_sysstats.csv`, written by `collect_metrics.py`.

Map-quality metrics (spatial entropy, wall plane-fitting RMSE, ghosting/double-wall detection) are
computed from each run's own exported map -- `_meta.json["map_path"]`, written by
collect_metrics.py's map-export step. Every algorithm that maintains a persistent point-cloud map
internally gets one exported, regardless of group: FAST-LIO2 and LIO-SAM (Group 1, tightly-coupled
LIO -- both register scans against an accumulated map, not just against each other) and DLO (Group
2) via their own native save mechanisms, same as HDL-Graph-SLAM and RTAB-Map (Group 3). KISS-ICP
(Group 2) is the one deliberate exclusion: its internal map is a distance-bounded rolling window
(max_distance=100m, confirmed directly in the `kiss_icp` package's `VoxelHashMap`), not a
persistent global map -- exporting "the map" at the end of a run would only capture the last ~100m
near wherever the run happened to end, not a fair map-quality comparison against the other five
algorithms' full accumulated maps. Group 4 (hdl_localization, FAST-LIO-Localization) consumes a pre-built
map for localization rather than building a new one, so there's nothing of its own to export.
FAST-LIO-Localization (Group 4) likewise consumes a map rather than building one.
Algorithms without an exported map simply have no map_path, which aggregate()'s union-of-keys
handling already treats as "no data" gracefully -- no special-casing needed per group.
"""

import copy
import csv
import functools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless container, no display -- Jupyter's inline backend overrides this
                        # on import when running inside a notebook (%matplotlib inline)
import matplotlib.pyplot as plt
import re

import numpy as np
from scipy.spatial.transform import Rotation
from evo.core import filters, metrics, sync
from evo.tools import file_interface
from scipy.spatial import cKDTree

RESULTS_ROOT = Path("/root/benchmark_results")

# Fixed-distance RPE segment length -- makes RPE comparable across algorithms publishing at very
# different frame rates (see module docstring). 10m is short enough to still be measured within a
# 30s smoke-test slice (typically 50-150m of travel) while being long enough to average out
# per-frame registration noise.
RPE_DELTA_M = 10.0
# Max timestamp gap allowed when associating an estimated pose to its nearest GT pose. GT is at
# 200Hz (Applanix POS LV); any real estimated pose has a GT sample within 100ms of it.
SYNC_MAX_DIFF_S = 0.1


def raw_dir(group):
    return RESULTS_ROOT / group / "raw"


def meta_path(group, algo, sequence, run):
    return raw_dir(group) / f"{algo}_{sequence}_run{run}_meta.json"


def load_meta(group, algo, sequence, run):
    path = meta_path(group, algo, sequence, run)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_sysstats(group, algo, sequence, run):
    path = raw_dir(group) / f"{algo}_{sequence}_run{run}_sysstats.csv"
    if not path.exists():
        return None
    with open(path) as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


# A step implying more than this multiple of the sequence's own top ground-truth speed is a
# pose-graph correction, not motion. Used only by the drift metric below -- see its docstring.
JUMP_SPEED_FACTOR = 1.5


def _drift_per_segment(traj_ref, traj_est_aligned, segment_m, exclude_jumps=True):
    """World-frame displacement error over ~segment_m of travel, RMS over all such segments.

    This is the project's primary LOCAL-accuracy figure, and it is deliberately NOT evo's
    RPE(translation_part), for two reasons that both still hold now that ground truth is emitted
    per body frame and each algorithm is scored against its own:

    1. RPE expresses each displacement in the ESTIMATE's body frame, so attitude error is applied
       to the displacement before the two are differenced -- 1 degree over a 10m segment
       contributes ~0.17m on its own. That term is not a fixed penalty: measured on Group 1 it
       inflates FAST-LIO2 by 68% and DEFLATES LIO-SAM by 35% (forest: 0.435 -> 0.731 and
       1.362 -> 0.891). Differencing the two displacement VECTORS in the world frame has no such
       term. Substituting the reference's orientations into either estimate collapses RPE onto this
       metric exactly, which is what identifies the attitude term as the whole of the difference.
    2. evo cannot express the jump exclusion below, which matters for every pose-graph system.

    Segments are cut by distance along the REFERENCE, non-overlapping, so each covers ~segment_m of
    real travel and a drifting estimate cannot shorten its own segments.

    Steps implying more than JUMP_SPEED_FACTOR times the sequence's own top ground-truth speed are
    pose-graph corrections rather than motion; segments containing one are dropped. Measured effect:
    <=0.3% on four of Group 1's five sequences, and 2.4x on urban (16 of 2466 segments).

    Returns None if the trajectory is too short to contain even one segment.
    """
    ref_xyz, est_xyz = traj_ref.positions_xyz, traj_est_aligned.positions_xyz
    seg = np.linalg.norm(np.diff(ref_xyz, axis=0), axis=1)
    if len(seg) == 0:
        return None
    v_max = (seg / np.maximum(np.diff(traj_ref.timestamps), 1e-9)).max() * JUMP_SPEED_FACTOR
    est_v = (np.linalg.norm(np.diff(est_xyz, axis=0), axis=1)
             / np.maximum(np.diff(traj_est_aligned.timestamps, ), 1e-9))
    is_jump = est_v > v_max

    cumulative = np.concatenate([[0.0], np.cumsum(seg)])
    pairs, start = [], 0
    for i in range(len(cumulative)):
        if cumulative[i] - cumulative[start] >= segment_m:
            pairs.append((start, i))
            start = i

    errors = [
        np.linalg.norm((est_xyz[b] - est_xyz[a]) - (ref_xyz[b] - ref_xyz[a]))
        for a, b in pairs
        if not (exclude_jumps and is_jump[a:b].any())
    ]
    return float(np.sqrt(np.mean(np.square(errors)))) if errors else None


def _rpe_rmse(traj_ref, traj_est_aligned, pose_relation, delta_m):
    """None (rather than raising) when the trajectory is too short to contain even one delta_m
    segment -- evo's own failure mode here (FilterException), expected for short slices (e.g. a
    30s smoke-test run covering ~20m) and not something the caller should have to guard against
    separately. A genuinely too-short *real* run (an algorithm crashing after a few poses) simply
    shows up as a missing RPE value in the summary table instead of crashing the whole notebook."""
    rpe = metrics.RPE(pose_relation, delta=delta_m, delta_unit=metrics.Unit.meters)
    try:
        rpe.process_data((traj_ref, traj_est_aligned))
    except filters.FilterException:
        return None
    return rpe.get_statistic(metrics.StatisticsType.rmse)


def compute_pose_metrics(gt_path, est_path, rpe_delta_m=RPE_DELTA_M):
    traj_ref = file_interface.read_tum_trajectory_file(str(gt_path))
    traj_est = file_interface.read_tum_trajectory_file(str(est_path))
    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est, max_diff=SYNC_MAX_DIFF_S)

    traj_est_aligned = copy.deepcopy(traj_est)
    traj_est_aligned.align(traj_ref, correct_scale=False)

    # evo calls this APE (Absolute Pose Error) because the same implementation can measure the
    # error in translation, rotation or full SE(3), selected by PoseRelation. With
    # translation_part it IS the classic ATE of Sturm et al. 2012, which is the name used for the
    # result keys because that is what a reader of this benchmark expects.
    ape = metrics.APE(metrics.PoseRelation.translation_part)
    ape.process_data((traj_ref, traj_est_aligned))

    # The same error WITHOUT alignment. Meaningless for odometry and SLAM, which have no external
    # reference and cannot be blamed for an arbitrary rigid offset -- but it is the operationally
    # meaningful figure for Group 4, whose algorithms are given the world frame by a prior map and
    # a seed pose. Aligning those hands them a correction they do not have at run time. Computed
    # here rather than in an analysis script so the number the Group 4 report leads with comes from
    # the same reviewed path as every other metric in this benchmark.
    ape_unaligned = metrics.APE(metrics.PoseRelation.translation_part)
    ape_unaligned.process_data((traj_ref, traj_est))

    result = {
        "n_matched_poses": len(traj_ref.timestamps),
        "ate_trans_rmse_m": ape.get_statistic(metrics.StatisticsType.rmse),
        "ate_trans_rmse_unaligned_m": ape_unaligned.get_statistic(metrics.StatisticsType.rmse),
        "ate_trans_median_unaligned_m": ape_unaligned.get_statistic(metrics.StatisticsType.median),
        "ate_trans_mean_m": ape.get_statistic(metrics.StatisticsType.mean),
        "ate_trans_median_m": ape.get_statistic(metrics.StatisticsType.median),
        "ate_trans_std_m": ape.get_statistic(metrics.StatisticsType.std),
        "ate_trans_max_m": ape.get_statistic(metrics.StatisticsType.max),
        "rpe_trans_rmse_m": _rpe_rmse(traj_ref, traj_est_aligned, metrics.PoseRelation.translation_part, rpe_delta_m),
        "rpe_rot_rmse_deg": _rpe_rmse(traj_ref, traj_est_aligned, metrics.PoseRelation.rotation_angle_deg, rpe_delta_m),
        # Computed here rather than in a separate tool so one read and one alignment serve every
        # metric, and so drift reaches aggregate()/summary_table_markdown() like all the others.
        "drift_trans_rmse_m": _drift_per_segment(traj_ref, traj_est_aligned, rpe_delta_m),
        # Same figure without the jump exclusion -- the pair quantifies how much pose-graph
        # discontinuities contribute, which is a reported number for every Group 3 system.
        "drift_trans_rmse_with_jumps_m": _drift_per_segment(traj_ref, traj_est_aligned, rpe_delta_m,
                                                            exclude_jumps=False),
    }
    # None values (RPE unavailable) are dropped rather than kept as None: aggregate()/summary
    # tables treat a missing key as "no data for this metric" and render "--", instead of every
    # downstream mean/std computation needing an extra None-check.
    return {k: v for k, v in result.items() if v is not None}


_PCD_TYPE_MAP = {
    ("F", 4): "f4", ("F", 8): "f8",
    ("U", 1): "u1", ("U", 2): "u2", ("U", 4): "u4", ("U", 8): "u8",
    ("I", 1): "i1", ("I", 2): "i2", ("I", 4): "i4", ("I", 8): "i8",
}


def read_pcd_xyz(path):
    """Generic binary PCD reader -- parses FIELDS/SIZE/TYPE/COUNT so it works regardless of point
    type, extracting only x/y/z. This project's map-export producers use several different point
    types for the *same* binary PCD container (confirmed directly, not assumed): plain XYZI
    float32 for LIO-SAM, DLO, and HDL-Graph-SLAM's /hdl_graph_slam/save_map, but PointXYZINormal
    (x,y,z,intensity,normal_x,normal_y,normal_z,curvature) for FAST-LIO2 -- a single XYZI-only
    reader would have silently misread FAST-LIO2's file (wrong stride, corrupted data). ASCII-mode
    PCD is not supported; none of this project's producers write it."""
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
                if "binary" not in line:
                    raise ValueError(f"Unsupported PCD DATA mode {line!r}: {path}")
                break
        dtype_fields = []
        for name, size, type_, count in zip(fields, sizes, types, counts):
            np_type = _PCD_TYPE_MAP[(type_, size)]
            dtype_fields.append((name, np_type) if count == 1 else (name, np_type, count))
        dtype = np.dtype(dtype_fields)
        data = np.frombuffer(f.read(n * dtype.itemsize), dtype=dtype, count=n)
    return np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float64)


_PLY_TYPE_MAP = {
    "float": ("f4", 4), "float32": ("f4", 4), "double": ("f8", 8), "float64": ("f8", 8),
    "char": ("i1", 1), "int8": ("i1", 1), "uchar": ("u1", 1), "uint8": ("u1", 1),
    "short": ("i2", 2), "int16": ("i2", 2), "ushort": ("u2", 2), "uint16": ("u2", 2),
    "int": ("i4", 4), "int32": ("i4", 4), "uint": ("u4", 4), "uint32": ("u4", 4),
}


def read_ply_xyz(path):
    """rtabmap-export's binary_little_endian PLY output (confirmed directly: x,y,z,intensity,
    nx,ny,nz,curvature per vertex, plus a trailing "element camera" block we don't need). Reads
    the property list generically so field order/extra properties don't matter, but only ever
    extracts x/y/z; everything after the vertex block (the camera element) is simply never read."""
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"Not a PLY file: {path}")
        fmt, vertex_count, props, in_vertex = None, None, [], False
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("format"):
                fmt = line.split()[1]
            elif line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
                in_vertex = True
            elif line.startswith("element"):
                in_vertex = False
            elif line.startswith("property") and in_vertex:
                _, ptype, pname = line.split()
                props.append((pname, ptype))
            elif line == "end_header":
                break
        if fmt != "binary_little_endian":
            raise ValueError(f"Unsupported PLY format {fmt!r}: {path}")
        dtype = np.dtype([(name, _PLY_TYPE_MAP[ptype][0]) for name, ptype in props])
        stride = sum(_PLY_TYPE_MAP[ptype][1] for _, ptype in props)
        data = np.frombuffer(f.read(vertex_count * stride), dtype=dtype, count=vertex_count)
    return np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float64)


def load_map_points(map_path):
    map_path = Path(map_path)
    if map_path.suffix == ".pcd":
        return read_pcd_xyz(map_path)
    if map_path.suffix == ".ply":
        return read_ply_xyz(map_path)
    raise ValueError(f"Unsupported map file type: {map_path}")


# Map-quality tuning constants. All somewhat arbitrary choices, same as this project's other
# voxel/threshold constants (e.g. the 0.3-0.5m live-processing voxel target) -- chosen to be
# reasonable defaults, not derived from a formula, and worth revisiting if the numbers they
# produce look wrong on a real map.
MAP_QUALITY_MAX_POINTS = 300_000   # subsample larger maps -- keeps KDTree/eigendecomposition cost bounded
MAP_QUALITY_KNN = 20               # neighbors used for local normal/planarity estimation
MAP_ENTROPY_VOXEL_M = 0.5          # voxel size for the spatial-entropy histogram
MAP_QUALITY_PLANARITY_THRESH = 0.97          # 1 - (smallest eigenvalue / sum) -- how "flat" a local neighborhood must be
MAP_QUALITY_VERTICAL_NORMAL_Z_MAX = 0.3      # |normal.z| below this => locally-vertical surface (a candidate wall)
MAP_QUALITY_MAX_PLANES = 6                   # stop after finding this many wall planes
# Calibrated directly against a real 300s HDL-Graph-SLAM map (boreas-suburb): at the tighter 0.05m/
# 300-inlier combination originally guessed, the best of 300 RANSAC trials only ever found ~208
# inliers -- real walls in a 300K-point subsample of a suburb-scale map are sparser than that
# tighter tolerance allows for. 0.08m/150 inliers reliably finds real walls on that same map
# (confirmed: best-of-300 trials reached 337 inliers) while still being tight enough to be a
# meaningful "wall thickness" signal, not just noise.
MAP_QUALITY_PLANE_DIST_THRESH_M = 0.08       # RANSAC inlier distance to a candidate plane
MAP_QUALITY_MIN_PLANE_INLIERS = 150          # minimum support to accept a detected plane as real
MAP_QUALITY_RANSAC_ITERS = 60                # random seed-point trials per plane
MAP_QUALITY_GHOST_PARALLEL_COS_MIN = 0.97    # |cos(angle between normals)| above this => "parallel"
MAP_QUALITY_GHOST_OFFSET_MIN_M = 0.03        # below this, two planes are the same wall, not a ghost
MAP_QUALITY_GHOST_OFFSET_MAX_M = 0.6         # above this, they're plausibly two different walls


def map_spatial_entropy(points, voxel_size=MAP_ENTROPY_VOXEL_M):
    """Shannon entropy (bits) of point counts across occupied voxels. Lower means points are
    concentrated into fewer, sharper voxels -- a "clean" map; higher means the same points are
    smeared more evenly across many voxels, e.g. drift spreading one real surface's returns over a
    wider region. Same int64 voxel-hash packing as build_ground_truth_map.py (collision-free,
    unlike a hashed key) -- see that script for why."""
    idx = np.floor(points / voxel_size).astype(np.int64) + 1_000_000
    keys = (idx[:, 0] << 42) | (idx[:, 1] << 21) | idx[:, 2]
    _, counts = np.unique(keys, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum()), int(len(counts))


def _estimate_normals_and_planarity(points, k=MAP_QUALITY_KNN):
    """Fully vectorized local PCA over each point's k nearest neighbors -- no per-point Python
    loop, so this scales to hundreds of thousands of points. normal = eigenvector of the smallest
    eigenvalue of the local covariance; planarity = 1 - (smallest eigenvalue / sum of eigenvalues),
    i.e. how flat vs. volumetric the local neighborhood is (1 = perfectly flat)."""
    tree = cKDTree(points)
    _, idx = tree.query(points, k=k, workers=-1)
    neighbor_pts = points[idx]  # (N, k, 3)
    centered = neighbor_pts - neighbor_pts.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centered, centered) / k
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending order
    normals = eigvecs[:, :, 0]
    denom = eigvals.sum(axis=1)
    denom[denom == 0] = 1e-12
    planarity = 1.0 - eigvals[:, 0] / denom
    return normals, planarity


def _extract_wall_planes(points, normals, planarity, rng):
    """Greedy RANSAC: repeatedly picks a random locally-planar, locally-vertical seed point,
    grows the largest inlier set among the remaining wall candidates, refines the plane via PCA on
    those inliers, and reports that plane's RMSE (point-to-plane distance) -- a direct proxy for
    "wall thickness"/point-cloud sharpness on that surface. Stops after MAX_PLANES or once too few
    candidate points remain to plausibly contain another real wall."""
    is_candidate = (planarity > MAP_QUALITY_PLANARITY_THRESH) & (np.abs(normals[:, 2]) < MAP_QUALITY_VERTICAL_NORMAL_Z_MAX)
    remaining = np.where(is_candidate)[0]
    planes = []
    for _ in range(MAP_QUALITY_MAX_PLANES):
        if len(remaining) < MAP_QUALITY_MIN_PLANE_INLIERS:
            break
        best_inliers = None
        for _ in range(MAP_QUALITY_RANSAC_ITERS):
            seed = rng.choice(remaining)
            p0, n0 = points[seed], normals[seed]
            dist = np.abs((points[remaining] - p0) @ n0)
            inliers = remaining[dist < MAP_QUALITY_PLANE_DIST_THRESH_M]
            if best_inliers is None or len(inliers) > len(best_inliers):
                best_inliers = inliers
        if best_inliers is None or len(best_inliers) < MAP_QUALITY_MIN_PLANE_INLIERS:
            break
        pts = points[best_inliers]
        centroid = pts.mean(axis=0)
        centered = pts - centroid
        cov = centered.T @ centered / len(pts)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, 0]
        dist = (pts - centroid) @ normal
        planes.append({
            "n_inliers": int(len(best_inliers)),
            "rmse_m": float(np.sqrt(np.mean(dist ** 2))),
            "centroid": centroid.tolist(),
            "normal": normal.tolist(),
        })
        remaining = np.setdiff1d(remaining, best_inliers, assume_unique=True)
    return planes


def _detect_ghost_pairs(planes):
    """Two detected wall planes with near-parallel normals and a small (but non-zero) perpendicular
    offset are the automated proxy for the "visual ghosting/double-wall" check: exactly the
    signature drift leaves behind when the same real wall gets registered twice, slightly apart,
    instead of once. Not a substitute for actually looking at the map -- see plot_map_quality."""
    pairs = []
    for i in range(len(planes)):
        for j in range(i + 1, len(planes)):
            n_i, n_j = np.array(planes[i]["normal"]), np.array(planes[j]["normal"])
            cos_angle = abs(float(np.dot(n_i, n_j)))
            if cos_angle < MAP_QUALITY_GHOST_PARALLEL_COS_MIN:
                continue
            offset = abs(float(np.dot(np.array(planes[j]["centroid"]) - np.array(planes[i]["centroid"]), n_i)))
            if MAP_QUALITY_GHOST_OFFSET_MIN_M < offset < MAP_QUALITY_GHOST_OFFSET_MAX_M:
                pairs.append({"plane_a": i, "plane_b": j, "offset_m": offset, "parallelism": cos_angle})
    return pairs


def _map_quality_analysis(map_path, max_points=MAP_QUALITY_MAX_POINTS, seed=0):
    """Shared implementation behind compute_map_quality_metrics() (flat scalars, for aggregation)
    and map_quality_detail() (full geometry, for plotting) -- loads once, computes once. A fixed
    seed makes the RANSAC plane search reproducible: this project's whole standard is that a
    metric should give the same answer on the same data, not a different one each notebook run."""
    if map_path is None:
        return None
    points = load_map_points(map_path)
    if len(points) == 0:
        return None
    rng = np.random.default_rng(seed)
    if len(points) > max_points:
        points = points[rng.choice(len(points), size=max_points, replace=False)]
    entropy_bits, n_occupied_voxels = map_spatial_entropy(points)
    normals, planarity = _estimate_normals_and_planarity(points)
    planes = _extract_wall_planes(points, normals, planarity, rng)
    ghost_pairs = _detect_ghost_pairs(planes)
    return {
        "points": points, "entropy_bits": entropy_bits, "n_occupied_voxels": n_occupied_voxels,
        "planes": planes, "ghost_pairs": ghost_pairs,
    }


def compute_map_quality_metrics(map_path):
    """Flat scalar dict, safe to merge into load_run()'s per-run metric dict (aggregate() calls
    np.mean/np.std on every value, so this deliberately excludes the plane/ghost-pair detail --
    see map_quality_detail() for that). None if map_path is None or the map has zero points."""
    analysis = _map_quality_analysis(map_path)
    if analysis is None:
        return None
    result = {
        "map_n_points": int(len(analysis["points"])),
        "map_spatial_entropy_bits": analysis["entropy_bits"],
        "map_n_wall_planes": len(analysis["planes"]),
        "n_ghost_pairs": len(analysis["ghost_pairs"]),
    }
    if analysis["planes"]:
        rmses = [p["rmse_m"] for p in analysis["planes"]]
        result["plane_rmse_mean_m"] = float(np.mean(rmses))
        result["plane_rmse_median_m"] = float(np.median(rmses))
    return result


def map_quality_detail(map_path):
    """Full geometric detail (the actual points analyzed, detected planes, ghost pairs) for
    plot_map_quality() -- recomputed on demand rather than cached alongside the scalar metrics."""
    return _map_quality_analysis(map_path)


def compute_compute_metrics(meta, sysstats):
    result = {
        "rtf": meta["real_time_factor"],
        "n_poses_recorded": meta["n_poses_recorded"],
        "bag_duration_s": meta["bag_duration_s"],
    }
    if sysstats:
        cpu_samples = [row["cpu_percent_total"] for row in sysstats]
        rss_samples = [row["rss_bytes_total"] for row in sysstats]
        result["avg_cpu_pct_of_machine"] = float(np.mean(cpu_samples)) / meta["cpu_count"]
        # psutil reports per-process CPU where 100 = one fully occupied core, and ResourceSampler
        # sums that over the process tree -- so dividing by 100 gives cores directly, with no
        # dependence on this machine's core count. Reports have been converting by hand.
        result["avg_cpu_cores"] = float(np.mean(cpu_samples)) / 100.0
        result["peak_rss_mb"] = float(np.max(rss_samples)) / 1e6

    # Counters collect_metrics.py extracted from the run's captured roslaunch output
    # (its launch_log_counters). Passed through as ordinary metrics so aggregate() averages them
    # like any other, rather than each report re-reading meta.json for itself. For
    # FAST-LIO-Localization icp_correction_rate is the difference between "localized" and
    # "silently running on odometry with a plausible-looking trajectory".
    for key in ("icp_cycles", "icp_rejected", "icp_correction_rate", "odom_lookup_failures"):
        if key in meta:
            result[key] = meta[key]
    return result


def body_frame_residual_deg(group, algo, sequence, run):
    """Median body-frame rotation between an estimate and ground truth, against the frame the run
    was actually scored in and against the other one.

    Not a Group 4 concern: this rig's LiDAR-IMU extrinsic is a 179.98 degree rotation, so an
    algorithm scored against the wrong frame's ground truth shows almost no change in translation
    while every rotation-dependent metric silently becomes meaningless. A working assignment gives
    a couple of degrees here and ~179 for the alternative -- collect_metrics.py tabulates
    this for all of Parts 1-3.

    Returns (own_frame_deg, other_frame_deg), or None if the run has no data.
    """
    meta = load_meta(group, algo, sequence, run)
    if meta is None:
        return None
    own_gt = Path(meta["ground_truth_path"])
    other = "lidar" if "_imu." in own_gt.name else "imu"
    other_gt = own_gt.with_name(re.sub(r"_(imu|lidar)\.tum$", f"_{other}.tum", own_gt.name))

    est = np.loadtxt(meta["estimated_trajectory_path"])
    out = []
    for path in (own_gt, other_gt):
        gt = np.loadtxt(path)
        idx = np.clip(np.searchsorted(gt[:, 0], est[:, 0]), 0, len(gt) - 1)
        matched = np.abs(gt[idx, 0] - est[:, 0]) < SYNC_MAX_DIFF_S
        rel = (Rotation.from_quat(gt[idx][matched, 4:8]).inv()
               * Rotation.from_quat(est[matched, 4:8])).magnitude()
        out.append(float(np.degrees(np.median(rel))))
    return tuple(out)


@functools.lru_cache(maxsize=None)
def load_run(group, algo, sequence, run):
    """Full per-run metric dict, or None if this run hasn't been collected yet (or produced 0
    poses -- a real failed run, excluded from aggregates rather than crashing evo on empty data).

    Cached per (group, algo, sequence, run): a notebook session calls this repeatedly across
    multiple tables/plots for the same run. Restart the kernel (or clear_load_run_cache()) to pick
    up newly-collected runs written after the notebook started."""
    meta = load_meta(group, algo, sequence, run)
    if meta is None or meta["n_poses_recorded"] == 0:
        return None
    pose_metrics = compute_pose_metrics(meta["ground_truth_path"], meta["estimated_trajectory_path"])
    sysstats = load_sysstats(group, algo, sequence, run)
    result = {**pose_metrics, **compute_compute_metrics(meta, sysstats)}
    map_metrics = compute_map_quality_metrics(meta.get("map_path"))
    if map_metrics:
        result.update(map_metrics)
    return result


def clear_load_run_cache():
    load_run.cache_clear()


def aggregate(group, algo, sequence, n_runs):
    """Mean +/- std across whatever runs 1..n_runs actually exist and have data -- gracefully
    partial (e.g. 1 of 3 runs done during smoke-testing) rather than requiring the full N. Returns
    None if no runs exist yet."""
    runs = [load_run(group, algo, sequence, r) for r in range(1, n_runs + 1)]
    runs = [r for r in runs if r is not None]
    if not runs:
        return None
    agg = {"n_runs": len(runs)}
    # Union of keys, not runs[0].keys(): a metric can be missing from just one run (e.g. RPE on a
    # run too short to contain a full delta_m segment) without dropping that metric everywhere.
    all_keys = set().union(*(r.keys() for r in runs))
    for key in all_keys:
        values = [r[key] for r in runs if key in r]
        if not values:
            continue
        agg[f"{key}_mean"] = float(np.mean(values))
        agg[f"{key}_std"] = float(np.std(values)) if len(values) > 1 else 0.0
    return agg


# (metric key, display label, format string) -- the "core 4" comparison metrics: one pose-accuracy
# summary (ATE), one relative-drift summary each for translation and rotation (RPE), one
# systems-performance summary (RTF). Everything else is "extended" detail below.
CORE_METRICS = [
    (f"drift_trans_rmse_m", f"Drift (m/{RPE_DELTA_M:.0f}m)", "{:.3f}"),
    ("ate_trans_rmse_m", "ATE trans RMSE (m)", "{:.3f}"),
    ("rpe_trans_rmse_m", f"RPE trans RMSE (m/{RPE_DELTA_M:.0f}m)", "{:.3f}"),
    ("rpe_rot_rmse_deg", f"RPE rot RMSE (deg/{RPE_DELTA_M:.0f}m)", "{:.3f}"),
    ("rtf", "RTF", "{:.3f}"),
]

EXTENDED_METRICS = [
    ("ate_trans_rmse_unaligned_m", "ATE trans RMSE, unaligned (m)", "{:.3f}"),
    ("drift_trans_rmse_with_jumps_m", "Drift, jumps included (m/10m)", "{:.3f}"),
    ("avg_cpu_cores", "Avg CPU (cores)", "{:.2f}"),
    ("icp_correction_rate", "Map corrections accepted", "{:.1%}"),
    ("avg_cpu_pct_of_machine", "Avg CPU (% of machine)", "{:.1%}"),
    ("peak_rss_mb", "Peak RAM (MB)", "{:.0f}"),
    ("ate_trans_mean_m", "ATE trans mean (m)", "{:.3f}"),
    ("ate_trans_median_m", "ATE trans median (m)", "{:.3f}"),
    ("ate_trans_std_m", "ATE trans std (m)", "{:.3f}"),
    ("ate_trans_max_m", "ATE trans max (m)", "{:.3f}"),
    ("n_poses_recorded", "Poses recorded", "{:.0f}"),
]

# Map-quality metrics -- every algorithm that builds a persistent map (FAST-LIO2, LIO-SAM, DLO,
# HDL-Graph-SLAM, RTAB-Map), which cuts across groups rather than aligning with them. Absent for
# KISS-ICP (rolling local map only, see module docstring) and Group 4 (consumes a map, builds
# none); summary_table_markdown()/bar_chart() already render "--" for a missing key, so these can
# be requested from any group's table without special-casing.
MAP_QUALITY_METRICS = [
    ("map_spatial_entropy_bits", "Map spatial entropy (bits)", "{:.2f}"),
    ("plane_rmse_mean_m", "Wall plane-fit RMSE, mean (m)", "{:.4f}"),
    ("n_ghost_pairs", "Suspected ghost/double-wall pairs", "{:.0f}"),
    ("map_n_wall_planes", "Wall planes detected", "{:.0f}"),
    ("map_n_points", "Map points analyzed", "{:.0f}"),
]


def summary_table_markdown(group, algorithms, sequences, n_runs, metrics_list, title=None):
    lines = [f"### {title}", ""] if title else []
    header = "| Algorithm | Sequence | Runs | " + " | ".join(label for _, label, _ in metrics_list) + " |"
    sep = "|---|---|---|" + "---|" * len(metrics_list)
    lines += [header, sep]
    for algo in algorithms:
        for seq in sequences:
            agg = aggregate(group, algo, seq, n_runs)
            if agg is None:
                lines.append(f"| {algo} | {seq} | 0/{n_runs} | " + " | ".join(["--"] * len(metrics_list)) + " |")
                continue
            cells = []
            for key, _, fmt in metrics_list:
                mean_key, std_key = f"{key}_mean", f"{key}_std"
                if mean_key not in agg:
                    cells.append("--")
                    continue
                mean_str = fmt.format(agg[mean_key])
                cells.append(f"{mean_str} +/- {fmt.format(agg[std_key])}" if agg["n_runs"] > 1 else mean_str)
            lines.append(f"| {algo} | {seq} | {agg['n_runs']}/{n_runs} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def bar_chart(group, algorithms, sequences, n_runs, metric_key, ylabel, title=None, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, 1.6 * len(sequences) * len(algorithms)), 5))
    x = np.arange(len(sequences))
    width = 0.8 / max(len(algorithms), 1)
    for i, algo in enumerate(algorithms):
        means, stds = [], []
        for seq in sequences:
            agg = aggregate(group, algo, seq, n_runs)
            if agg is None or f"{metric_key}_mean" not in agg:
                means.append(np.nan)
                stds.append(0.0)
            else:
                means.append(agg[f"{metric_key}_mean"])
                stds.append(agg[f"{metric_key}_std"])
        ax.bar(x + i * width, means, width, yerr=stds, label=algo, capsize=3)
    ax.set_xticks(x + width * (len(algorithms) - 1) / 2)
    ax.set_xticklabels(sequences, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    return ax


def plot_trajectories(group, algorithms, sequence, run, ax=None):
    """Top-down XY overlay: ground truth vs each algorithm's SE(3)-aligned estimate, for one
    sequence/run -- the project's spotlight-figure convention (visual accuracy/drift/robustness at
    a glance, not just an RMSE number)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    gt_path = None
    for algo in algorithms:
        meta = load_meta(group, algo, sequence, run)
        if meta is not None and meta["n_poses_recorded"] > 0:
            gt_path = meta["ground_truth_path"]
            break
    if gt_path is None:
        ax.set_title(f"{sequence} (run {run}): no data yet")
        return ax

    traj_ref_full = file_interface.read_tum_trajectory_file(gt_path)
    ax.plot(traj_ref_full.positions_xyz[:, 0], traj_ref_full.positions_xyz[:, 1],
            "k--", label="ground truth", linewidth=2)

    for algo in algorithms:
        meta = load_meta(group, algo, sequence, run)
        if meta is None or meta["n_poses_recorded"] == 0:
            continue
        traj_ref = file_interface.read_tum_trajectory_file(meta["ground_truth_path"])
        traj_est = file_interface.read_tum_trajectory_file(meta["estimated_trajectory_path"])
        traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est, max_diff=SYNC_MAX_DIFF_S)
        traj_est_aligned = copy.deepcopy(traj_est)
        traj_est_aligned.align(traj_ref, correct_scale=False)
        ax.plot(traj_est_aligned.positions_xyz[:, 0], traj_est_aligned.positions_xyz[:, 1],
                label=algo, alpha=0.85)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.set_title(f"{sequence} (run {run})")
    ax.legend()
    return ax


def plot_map_quality(group, algo, sequence, run, ax=None):
    """Top-down scatter of the algorithm's own built map (colored by height), with any suspected
    ghost/double-wall plane pairs marked -- the visual counterpart to n_ghost_pairs. A count is not
    a substitute for actually looking at the map; this is what an a rejected-experiment-style validation
    pass should check by eye before trusting the automated number."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 9))
    meta = load_meta(group, algo, sequence, run)
    if meta is None or not meta.get("map_path"):
        ax.set_title(f"{algo} / {sequence} (run {run}): no map")
        return ax
    detail = map_quality_detail(meta["map_path"])
    if detail is None:
        ax.set_title(f"{algo} / {sequence} (run {run}): map empty")
        return ax

    pts = detail["points"]
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2], s=0.3, cmap="viridis", alpha=0.5)
    plt.colorbar(sc, ax=ax, label="z (m)", shrink=0.7)
    for pair in detail["ghost_pairs"]:
        for plane_idx in (pair["plane_a"], pair["plane_b"]):
            cx, cy, _ = detail["planes"][plane_idx]["centroid"]
            ax.plot(cx, cy, "rx", markersize=14, markeredgewidth=2)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.set_title(f"{algo} / {sequence} (run {run}) -- {len(detail['planes'])} wall plane(s), "
                 f"{len(detail['ghost_pairs'])} suspected ghost pair(s)")
    return ax
