#!/usr/bin/env python3
"""
Validates that collected runs are *structurally sound* before anyone trusts their accuracy numbers.

Why this exists: `collect_metrics.py` marks a run successful if it recorded more than zero poses.
That check is too weak, and twice it let genuinely broken data through looking perfectly healthy:

  - KISS-ICP (Group 2, all 12 runs): its ROS subscriber silently dropped most incoming scans once
    per-frame cost exceeded the ~104ms scan interval. Every run still exited [OK] with a plausible
    pose count and RTF ~1.0, but the trajectory covered only 6-20% of the distance actually driven.
  - FAST-LIO2 (Group 1, boreas-urban, all 3 runs): the process died ~1458s into a 2001s bag
    (unbounded ikd-Tree map exhausting memory). `rosbag play` ran to completion, so the run looked
    normal -- but 27% of the route simply has no trajectory, and the ATE was silently scored on
    only the 73% that survived.

Both share a signature the accuracy metrics can't express: the *estimate* is fine, there's just far
less of it than there should be. ATE/RPE are computed only over poses that exist, so a truncated or
decimated run can post a perfectly respectable RMSE. These checks compare what a run produced
against what the bag and ground truth say it *should* have produced.

Usage:
  ./tools/check_run_health.py                        # every group, every algorithm
  ./tools/check_run_health.py --group group1_tight_coupled_lio
  ./tools/check_run_health.py --algorithm fastlio2 --verbose

Run it inside the container (results live at /root/benchmark_results). Exits non-zero if any run
FAILs, so it can gate a collection script or CI step.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from metrics_lib import RESULTS_ROOT, load_meta  # noqa: E402

# --- Thresholds -------------------------------------------------------------------------------
# Deliberately split into "the harness failed" vs "the algorithm did poorly". Only the former is a
# FAIL: a genuinely bad trajectory is a legitimate result worth publishing (Group 4's localization
# failures are real findings), but a truncated or decimated one is corrupt input to any conclusion.

# Fraction of the bag's duration the estimated trajectory must span. A run that stops early means
# the algorithm process died mid-bag -- the FAST-LIO2/urban case. 0.98 leaves room for the ordinary
# second or two of startup/shutdown trim without flagging it.
MIN_DURATION_COVERAGE = 0.98

# Estimated poses relative to ground-truth samples over the same window. GT here is sampled per
# LiDAR scan (~9.6Hz), so an algorithm publishing one pose per scan lands near 1.0. Well below that
# means scans were dropped rather than processed -- the KISS-ICP case. Algorithms that publish
# faster than scan rate (LIO-SAM's /odometry/imu runs at 200Hz) land far above 1.0, which is fine.
MIN_POSE_RATE_RATIO = 0.90

# Below this, the trajectory is not tracking at all -- position frozen or near-frozen while the
# vehicle genuinely moved. Distinct from "drifted badly", which can still have a sane path length.
FROZEN_PATH_RATIO = 0.20

# Path ratio above which a below-sensor-rate pose stream is treated as a deliberate design choice
# rather than data loss (see the pose-rate check). Well above the frozen threshold, well below the
# ~1.0 a well-tracking run reaches, so it separates "tracking fine, just publishing less often" from
# "lost the trajectory".
PATH_OK_FOR_LOW_RATE = 0.50

# A step implying more than this multiple of the sequence's own maximum ground-truth speed cannot be
# vehicle motion; it is a pose discontinuity (loop-closure correction, or an odometry reset).
JUMP_SPEED_FACTOR = 1.5
# Fraction of total estimated path length allowed to come from such steps before it is worth saying
# so. A pure-odometry run sits near zero; Group 3's pose-graph systems reach 23-47% on routes with
# many revisits.
MAX_JUMP_PATH_FRACTION = 0.05

MIN_RTF = 0.95  # bag played slower than real time; resource numbers stop being comparable

# <algo>_<sequence>_run<N>_meta.json, anchored on the "boreas-" sequence prefix so that algorithm
# names containing underscores (dlo_no_imu, fastlio_localization) parse correctly.
RUN_FILENAME_RE = re.compile(r"^(?P<algo>.+)_(?P<seq>boreas-[a-z0-9]+)_run(?P<run>\d+)_meta\.json$")


def read_tum(path):
    """(timestamps, xyz) from a TUM trajectory file. Plain numpy rather than evo: this runs over
    every collected run, and only timestamps/positions are needed."""
    data = np.loadtxt(path)
    if data.ndim == 1:  # single-pose file
        data = data.reshape(1, -1)
    return data[:, 0], data[:, 1:4]


def path_length(xyz):
    return float(np.sum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))) if len(xyz) > 1 else 0.0


def check_run(group, algo, sequence, run):
    """One run -> (status, metrics dict, list of human-readable problems).

    status is "FAIL" (data is structurally unsound, don't draw conclusions from it), "WARN"
    (worth a look, but usable), or "OK". Returns None if the run wasn't collected."""
    meta = load_meta(group, algo, sequence, run)
    if meta is None:
        return None

    problems = []
    m = {}

    n_poses = meta.get("n_poses_recorded", 0)
    m["n_poses"] = n_poses
    if n_poses == 0:
        return "FAIL", m, ["recorded 0 poses"]

    est_path = Path(meta["estimated_trajectory_path"])
    gt_path = Path(meta["ground_truth_path"])
    if not est_path.exists():
        return "FAIL", m, [f"estimated trajectory file missing: {est_path}"]

    est_t, est_xyz = read_tum(est_path)
    gt_t, gt_xyz = read_tum(gt_path)

    # --- 1. Duration coverage: did the algorithm survive the whole bag? ---
    bag_dur = meta["bag_duration_s"]
    est_dur = float(est_t[-1] - est_t[0])
    coverage = est_dur / bag_dur if bag_dur else 0.0
    m["bag_dur_s"], m["est_dur_s"], m["coverage"] = bag_dur, est_dur, coverage
    if coverage < MIN_DURATION_COVERAGE:
        problems.append(
            f"trajectory covers only {coverage:.1%} of the bag "
            f"({est_dur:.0f}s of {bag_dur:.0f}s) -- process likely died mid-run"
        )

    # --- 2. Path length: is it tracking motion at all? ---
    # Computed before the pose-rate check because it decides how seriously to take that one.
    gt_window_xyz = gt_xyz[(gt_t >= est_t[0]) & (gt_t <= est_t[-1])]
    gt_len, est_len = path_length(gt_window_xyz), path_length(est_xyz)
    path_ratio = est_len / gt_len if gt_len else 0.0
    m["gt_path_m"], m["est_path_m"], m["path_ratio"] = gt_len, est_len, path_ratio
    if path_ratio < FROZEN_PATH_RATIO:
        problems.append(
            f"estimated path is {est_len:.0f}m against {gt_len:.0f}m driven ({path_ratio:.1%}) "
            f"-- position frozen or barely tracking"
        )

    # --- 2b. Discontinuities: does the trajectory "travel" further than the vehicle did? ---
    # Originally this check only had a lower bound, which let a trajectory covering 182% of the
    # driven distance pass as healthy. That excess is not jitter -- it is discrete pose jumps, and
    # on Group 3 they are real: a pose-graph SLAM system relocates its whole trajectory when it
    # closes a loop, and RTAB-Map additionally jumps whenever its odometry resets. Measured on
    # rtabmap/forest, 10% of steps implied speeds up to 4271 m/s (against a 24.3 m/s ground-truth
    # maximum) and contributed 47% of total path length.
    #
    # This is not a data-validity failure -- ATE stays meaningful, because after a correction the
    # pose is closer to truth, which is the whole point. But it does invalidate any INCREMENTAL
    # metric on that run: a fixed-distance segment containing a correction records the jump as if
    # it were drift. Reported as a warning so the run is usable but the limitation is visible.
    jump_frac = 0.0
    if len(est_xyz) > 1 and gt_len:
        steps = np.linalg.norm(np.diff(est_xyz, axis=0), axis=1)
        dt = np.maximum(np.diff(est_t), 1e-9)
        gt_dt = np.maximum(np.diff(gt_t), 1e-9)
        gt_speed_max = float(np.max(np.linalg.norm(np.diff(gt_xyz, axis=0), axis=1) / gt_dt))
        implausible = (steps / dt) > (gt_speed_max * JUMP_SPEED_FACTOR)
        jump_frac = float(steps[implausible].sum() / steps.sum()) if steps.sum() else 0.0
    m["jump_path_frac"] = jump_frac
    if jump_frac > MAX_JUMP_PATH_FRACTION:
        warn_only_early = (
            f"{jump_frac:.0%} of estimated path length comes from steps faster than physically "
            f"possible -- pose-graph corrections/resets, so incremental drift metrics are not "
            f"valid on this run (ATE is)"
        )
    else:
        warn_only_early = None

    # --- 3. Pose rate: was every scan actually processed? ---
    # Compared against GT samples inside the estimate's own time window, so a run that's already
    # flagged as truncated isn't double-penalized for the poses it never got the chance to publish.
    #
    # Severity depends on whether the missing poses actually cost anything. Two very different
    # things produce a low rate: an algorithm that deliberately runs its pipeline below sensor rate
    # and still tracks correctly (RTAB-Map's ICP odometry publishes ~2.5Hz against 9.6Hz scans on
    # this data while recovering 101% of the driven path -- real behaviour, not data loss), versus
    # one silently dropping scans it needed and losing the trajectory as a result (KISS-ICP at stock
    # settings, which paired a low rate with 6-20% of the real distance). Path ratio separates them,
    # so a low rate alongside healthy tracking is reported but not treated as disqualifying.
    gt_in_window = int(np.count_nonzero((gt_t >= est_t[0]) & (gt_t <= est_t[-1])))
    rate_ratio = n_poses / gt_in_window if gt_in_window else 0.0
    m["gt_in_window"], m["pose_rate_ratio"] = gt_in_window, rate_ratio

    warn_only = []
    if warn_only_early:
        warn_only.append(warn_only_early)
    if rate_ratio < MIN_POSE_RATE_RATIO:
        msg = (f"only {n_poses} poses for {gt_in_window} ground-truth samples in the same window "
               f"({rate_ratio:.1%})")
        if path_ratio >= PATH_OK_FOR_LOW_RATE:
            warn_only.append(msg + " -- runs below sensor rate, but tracking is unaffected")
        else:
            problems.append(msg + " -- scans dropped rather than processed")

    # --- 4. Real-time factor (advisory) ---
    rtf = meta.get("real_time_factor")
    m["rtf"] = rtf
    if rtf is not None and rtf < MIN_RTF:
        warn_only.append(f"RTF {rtf:.3f} -- bag played slower than real time")

    if problems:
        return "FAIL", m, problems + warn_only
    if warn_only:
        return "WARN", m, warn_only
    return "OK", m, []


def main():
    ap = argparse.ArgumentParser(
        description="Validate collected runs for truncation, dropped scans, and frozen tracking.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--group", help="only this group directory (default: all found)")
    ap.add_argument("--algorithm", help="only this algorithm")
    ap.add_argument("--verbose", action="store_true", help="show per-run detail for healthy runs too")
    args = ap.parse_args()

    groups = [args.group] if args.group else sorted(
        p.name for p in RESULTS_ROOT.iterdir() if (p / "raw").is_dir()
    )

    n_ok = n_warn = n_fail = 0
    for group in groups:
        raw = RESULTS_ROOT / group / "raw"
        if not raw.is_dir():
            print(f"!! no such group: {group}")
            continue

        # Discover what's actually been collected from the meta files themselves, so this needs no
        # hardcoded algorithm/sequence lists and keeps working as new ones are added.
        #
        # Anchored on the sequence name rather than splitting on "_": algorithm names contain
        # underscores too (dlo_no_imu, fastlio_localization), and a naive split silently
        # mis-parsed those as algo="dlo"/sequence="no" -- which then resolved to no meta file and
        # was skipped without a word. That is precisely the class of silent omission this script
        # exists to catch, so it fails loudly on an unparseable name instead.
        # Run numbers come from the filenames too, rather than iterating a fixed 1..N range: run
        # indices are arbitrary (--run-numbers lets you collect run 7 without runs 1-6, and scratch
        # validation runs use high numbers like 90+), so a range would silently skip anything above
        # its bound -- the same silent omission this script exists to prevent.
        found, unparsed = set(), []
        for f in raw.glob("*_run*_meta.json"):
            m = RUN_FILENAME_RE.match(f.name)
            if m:
                found.add((m.group("algo"), m.group("seq"), int(m.group("run"))))
            else:
                unparsed.append(f.name)
        if unparsed:
            print(f"\n!! {len(unparsed)} meta file(s) in {group} could not be parsed and were NOT "
                  f"checked -- filename must match <algo>_<sequence>_run<N>_meta.json:")
            for name in sorted(unparsed):
                print(f"     {name}")
        found = sorted(found)
        if args.algorithm:
            found = [t for t in found if t[0] == args.algorithm]
        if not found:
            continue

        print(f"\n{'=' * 78}\n{group}\n{'=' * 78}")
        for algo, sequence, run in found:
            result = check_run(group, algo, sequence, run)
            if result is None:
                continue
            status, m, problems = result
            n_ok += status == "OK"
            n_warn += status == "WARN"
            n_fail += status == "FAIL"

            if status == "OK" and not args.verbose:
                continue
            label = f"{algo}/{sequence}/run{run}"
            print(f"\n  [{status}] {label}")
            print(f"        poses={m.get('n_poses')}  coverage={m.get('coverage', 0):.1%}  "
                  f"pose_rate={m.get('pose_rate_ratio', 0):.1%}  "
                  f"path={m.get('path_ratio', 0):.1%}  rtf={m.get('rtf') or 0:.3f}")
            for p in problems:
                print(f"        - {p}")

    print(f"\n{'=' * 78}")
    print(f"{n_ok} OK   {n_warn} WARN   {n_fail} FAIL"
          + ("" if args.verbose or not n_ok else "   (healthy runs hidden; --verbose to show)"))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
