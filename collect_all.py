#!/usr/bin/env python3
"""
Sequentially runs collect_metrics.py across every algorithm in its ALGORITHMS registry, one
algorithm at a time. Never parallel, by explicit project decision: running anything else on this
machine during collection -- including another algorithm's collection -- corrupts the CPU/RAM
measurements ResourceSampler records for whichever run happens to share the CPU.

Each algorithm runs as its own subprocess (not an in-process call into collect_metrics.py) for full
isolation: a crash, leaked thread, or stuck subprocess in one algorithm's run must never contaminate
the next algorithm's measurements or abort the rest of an unattended multi-hour sweep.

Resumable at the individual-run granularity: for each (algorithm, sequence), only the specific run
numbers that don't already have a valid _meta.json (n_poses_recorded > 0) are collected, via
collect_metrics.py's --run-numbers. Topping up from N=1 to N=3 later never redoes the run you
already have. An interrupted sweep can just be re-invoked with the same arguments. Use --force to
redo everything regardless.

Usage:
  ./collect_all.py --dry-run                       # print the per-algorithm ETA table, run nothing
  ./collect_all.py                                  # full sweep: all algorithms, all sequences, N=3
  ./collect_all.py --algorithms fastlio2 liosam      # explicit subset
  ./collect_all.py --runs 1                          # fewer runs per combo
  ./collect_all.py --force                           # redo everything, ignore existing results
  ./collect_all.py --budget-hours 3                  # auto-fit as many whole algorithms as fit in
                                                      # 3 hours (order-preserving greedy over
                                                      # --algorithms -- an algorithm that doesn't
                                                      # fit is skipped, not split, and later
                                                      # cheaper ones are still tried); --dry-run
                                                      # first to see the selection before running it

Must be invoked the same way as collect_metrics.py -- inside the container, with the ROS
environment already sourced (this script's own subprocess calls inherit that environment; it does
not re-source per algorithm):
  docker exec <container> bash -lc 'source /opt/ros/noetic/setup.bash && \\
    source /root/catkin_ws/devel/setup.bash && python3 /root/boreas_slam_benchmark/collect_all.py'
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from collect_metrics import ALGORITHMS, SEQUENCES, RESULTS_ROOT, bag_duration_s, bag_path_for_sequence

SCRIPT_DIR = Path(__file__).parent
COLLECT_METRICS = SCRIPT_DIR / "collect_metrics.py"

# Conservative: every algorithm smoke-tested so far in this project has measured RTF 0.93-0.98
# (real-time or slightly under, never faster) -- used only for the upfront ETA estimate, not to
# control anything at run time.
ASSUMED_RTF = 0.95
# AUX_PROC_STARTUP_GRACE_S(2) + NODE_STARTUP_GRACE_S(3) + RECORD_SUBSCRIBE_GRACE_S(1) +
# TAIL_DRAIN_GRACE_S(3) from collect_metrics.py -- fixed per-run overhead independent of bag length.
PER_RUN_OVERHEAD_S = 9.0
# Between every collect_metrics.py invocation (now one per (algorithm, sequence), not one per
# algorithm) -- gives a crashed process's resources a moment to actually release before the next
# measurement window starts.
INTER_INVOCATION_SETTLE_S = 2.0


def bag_duration_for_sequence(sequence):
    """Real full bag duration, read from any existing run's _meta.json to avoid re-opening the
    (multi-GB) bag file just for an ETA estimate; falls back to opening it directly via rosbag
    if this sequence has never been run before."""
    for meta_path in RESULTS_ROOT.glob(f"*/raw/*_{sequence}_run1_meta.json"):
        try:
            with open(meta_path) as f:
                return json.load(f)["full_bag_duration_s"]
        except (json.JSONDecodeError, KeyError, OSError):
            continue
    return bag_duration_s(bag_path_for_sequence(sequence))


def meta_path_for(algo_cfg, algo_name, sequence, run_idx):
    raw_dir = RESULTS_ROOT / algo_cfg["group"] / "raw"
    return raw_dir / f"{algo_name}_{sequence}_run{run_idx}_meta.json"


def run_is_valid(meta_path):
    if not meta_path.exists():
        return False
    try:
        with open(meta_path) as f:
            return json.load(f).get("n_poses_recorded", 0) > 0
    except (json.JSONDecodeError, OSError):
        return False


def missing_runs_for(algo_cfg, algo_name, sequence, n_runs):
    """Which of runs 1..n_runs don't yet have a valid result -- empty list means fully done."""
    return [
        run_idx for run_idx in range(1, n_runs + 1)
        if not run_is_valid(meta_path_for(algo_cfg, algo_name, sequence, run_idx))
    ]


def format_duration(seconds):
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


# Executables that belong to a ROS runtime and are safe to kill between runs. Matched against the
# BASENAME of each command-line token, never as a bare substring of the whole line -- the previous
# implementation grepped for 'ros|nodelet' anywhere in `ps aux` output, which also matches the
# perfectly innocent `bash -lc 'source /opt/ros/noetic/setup.bash && ...'` wrapper that every
# documented invocation of this project uses. It was killing the caller's own shell: the collection
# kept running (collect_all.py is a child, and its own cmdline contains no "ros") but the shell died
# with SIGKILL, so the exit status reported to the user was 137 regardless of how the run actually
# went. Any path containing "/opt/ros/" was a candidate too.
ROS_RUNTIME_BINARIES = {
    "roscore", "rosmaster", "rosout", "roslaunch", "rosbag", "rosrun",
    "nodelet", "static_transform_publisher",
}
# rosbag's own worker binaries are named just "record"/"play" (/opt/ros/.../lib/rosbag/record), which
# are far too generic to match on basename alone -- matched on their containing directory instead.
ROS_BAG_WORKER_DIR = "/rosbag/"
# This project's own helper nodes (nodes/), which roslaunch does not own and so does not reap.
# DERIVED from the directory rather than listed by hand: the hand-written list silently fell behind
# as Group 4 grew, leaving two relays off it while both were in active use as aux_procs --
# exactly the orphan this function exists to prevent, since a helper left running keeps
# publishing stale data into the next run's filter. Any new node under nodes/ is now covered
# the moment it is added.
OUR_NODE_SCRIPTS = {
    path.name for path in (SCRIPT_DIR / "nodes").rglob("*.py")
}


def kill_stray_ros_processes():
    """Defensive cleanup: collect_metrics.py's own `finally` block tears down its roscore/aux_procs
    on a clean exit, but a hard crash (segfault, OOM-kill), an interrupted run or a power cut can
    leave orphans behind -- and they survive killing their parent, because collect_metrics.py starts
    them with start_new_session=True. A leftover `rosbag play` silently feeds the next run's master;
    a leftover `rosbag record` silently competes for its topic. Both were observed producing runs
    that recorded 0 poses with no error reported anywhere.

    Never kills this process or any of its ancestors -- that is the bug the old implementation had.
    """
    import psutil  # available via collect_metrics.py's dependencies

    # Walk parents one at a time rather than psutil.Process.parents(): that convenience method
    # doesn't exist in the psutil this container ships, and calling it would raise AttributeError
    # here -- at the very top of a multi-hour sweep.
    protected = {os.getpid()}
    try:
        current = psutil.Process(os.getpid())
        while current is not None:
            protected.add(current.pid)
            current = current.parent()
    except psutil.Error:
        pass

    for proc in psutil.process_iter(["pid", "cmdline"]):
        if proc.info["pid"] in protected:
            continue
        cmdline = proc.info["cmdline"] or []
        names = {os.path.basename(tok) for tok in cmdline}
        is_bag_worker = any(ROS_BAG_WORKER_DIR in tok for tok in cmdline)
        if names & ROS_RUNTIME_BINARIES or names & OUR_NODE_SCRIPTS or is_bag_worker:
            try:
                proc.kill()
            except psutil.Error:
                pass


def algo_eta_s(missing, bag_durations):
    """missing: {sequence: [run indices]}. Costed by actual missing run counts, not a blanket
    n_runs -- a sequence needing only 1 more run costs a third of one needing all 3."""
    total_s = 0.0
    for seq, run_list in missing.items():
        n = len(run_list)
        total_s += bag_durations[seq] * n / ASSUMED_RTF + PER_RUN_OVERHEAD_S * n
    return total_s


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algorithms", nargs="+", default=sorted(ALGORITHMS), choices=sorted(ALGORITHMS),
                         help="Also sets priority order for --budget-hours -- list your preferred "
                              "ones first")
    parser.add_argument("--sequences", nargs="+", default=SEQUENCES, choices=SEQUENCES)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true", help="Print the ETA table and exit without running anything")
    parser.add_argument("--force", action="store_true", help="Redo (algorithm, sequence) combos that already have valid results")
    parser.add_argument("--budget-hours", type=float, default=None,
                         help="Only run whole algorithms whose remaining (not-already-done) work "
                              "fits in this many hours total. Order-preserving greedy over "
                              "--algorithms: an algorithm that doesn't fit is skipped (not split "
                              "across sequences), and later, cheaper algorithms are still tried.")
    parser.add_argument("--max-duration", type=float, default=None,
                         help="Passed through to collect_metrics.py -- play only the first N seconds "
                              "of each bag. For validating this script's own orchestration logic "
                              "quickly; never use for a real collection run.")
    args = parser.parse_args()

    bag_durations = {seq: bag_duration_for_sequence(seq) for seq in args.sequences}
    print("[PLAN] Per-sequence bag durations: " +
          ", ".join(f"{seq}={format_duration(d)}" for seq, d in bag_durations.items()))

    # Per-algorithm breakdown: pending sequences, ETA for just this algorithm's remaining work,
    # and (if --budget-hours given) whether it's selected -- this is the piece that lets you
    # choose algorithms by how many hours you actually have, rather than guessing from the
    # all-or-nothing aggregate total.
    budget_s = args.budget_hours * 3600 if args.budget_hours is not None else None
    running_total_s = 0.0
    selected = {}  # algo -> {sequence: [missing run indices]}
    print(f"[PLAN] Per-algorithm breakdown ({args.runs} run(s)/sequence"
          f"{f', budget {args.budget_hours}h' if budget_s is not None else ''}):")
    for algo in args.algorithms:
        algo_cfg = ALGORITHMS[algo]
        if args.force:
            missing = {seq: list(range(1, args.runs + 1)) for seq in args.sequences
                       if seq not in algo_cfg.get("skip_sequences", [])}
        else:
            missing = {}
            for seq in args.sequences:
                if seq in algo_cfg.get("skip_sequences", []):
                    continue
                run_list = missing_runs_for(algo_cfg, algo, seq, args.runs)
                if run_list:
                    missing[seq] = run_list
        if not missing:
            print(f"  {algo:<16} already complete, nothing to do")
            continue

        eta_s = algo_eta_s(missing, bag_durations)
        n_missing_runs = sum(len(v) for v in missing.values())
        detail = f"({len(missing)}/{len(args.sequences)} sequence(s), {n_missing_runs} run(s) pending)"
        if budget_s is None:
            selected[algo] = missing
            print(f"  {algo:<16} {format_duration(eta_s):>10}  {detail}")
        elif running_total_s + eta_s <= budget_s:
            selected[algo] = missing
            running_total_s += eta_s
            print(f"  {algo:<16} {format_duration(eta_s):>10}  {detail} -- selected, running total {format_duration(running_total_s)}")
        else:
            print(f"  {algo:<16} {format_duration(eta_s):>10}  {detail} -- SKIPPED, would exceed budget")

    total_selected_s = sum(algo_eta_s(missing, bag_durations) for missing in selected.values())
    print(f"[PLAN] {len(selected)}/{len(args.algorithms)} algorithm(s) selected, "
          f"estimated {format_duration(total_selected_s)} total at RTF~{ASSUMED_RTF}")

    if args.dry_run or not selected:
        return

    print("[PLAN] Starting sequential collection. Do not run anything else on this machine while "
          "this is in progress -- CPU/RAM measurements assume a quiet system.")

    # Also clear strays BEFORE the first algorithm, not just between them. Orphans don't only come
    # from a crash mid-collection: an interrupted run, a power cut, or a manually killed debugging
    # session can leave a `rosbag play`/`rosbag record` alive, and those survive killing their
    # parent because collect_metrics.py starts them with start_new_session=True. A leftover
    # `rosbag play` silently feeds the next run's master -- confirmed directly during Group 3
    # verification, where one produced runs that recorded 0 poses with no error anywhere.
    kill_stray_ros_processes()

    run_start = time.time()
    n_algo_done = n_algo_failed = 0
    for algo_idx, (algo, missing) in enumerate(selected.items()):
        algo_cfg = ALGORITHMS[algo]
        elapsed = time.time() - run_start
        n_missing_runs = sum(len(v) for v in missing.values())
        print(f"[{algo}] ({algo_idx + 1}/{len(selected)}) {format_duration(elapsed)} "
              f"elapsed so far -- {len(missing)} sequence(s), {n_missing_runs} run(s) total")

        algo_failed = False
        for seq, run_list in missing.items():
            cmd = ["python3", str(COLLECT_METRICS), "--algorithm", algo,
                   "--sequences", seq, "--run-numbers", *map(str, run_list)]
            if args.max_duration is not None:
                cmd += ["--max-duration", str(args.max_duration)]
            log_path = RESULTS_ROOT / algo_cfg["group"] / "raw" / f"{algo}_{seq}_collect_all.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w") as log_f:
                result = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT)

            kill_stray_ros_processes()
            time.sleep(INTER_INVOCATION_SETTLE_S)

            if result.returncode == 0:
                print(f"[{algo}] {seq} run(s) {run_list} done")
            else:
                algo_failed = True
                print(f"[{algo}] {seq} run(s) {run_list} FAILED (exit {result.returncode}) "
                      f"-- see {log_path}, continuing")

        if algo_failed:
            n_algo_failed += 1
        else:
            n_algo_done += 1
            print(f"[{algo}] done")

    elapsed = time.time() - run_start
    print(f"[DONE] {n_algo_done} algorithm(s) completed cleanly, {n_algo_failed} with at least "
          f"one failed run, {format_duration(elapsed)} elapsed")


if __name__ == "__main__":
    main()
