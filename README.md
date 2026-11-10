# Boreas-RT SLAM Benchmark

Eight open-source LiDAR SLAM, odometry and map-localization systems, measured on five sequences
of the [Boreas-RT](https://www.boreas.utias.utoronto.ca/) dataset (51.9 km) on identical hardware,
with identical preprocessing — three runs each, 135 runs in total.

Four short papers report the results; this repository is everything needed to regenerate the
numbers in them from the dataset up.

| Part | Question | Systems |
|---|---|---|
| 1 | Does fusing IMU tightly buy accuracy worth its cost? | FAST-LIO2, LIO-SAM |
| 2 | How far does LiDAR alone get you? | KISS-ICP, DLO, DLO without IMU |
| 3 | What does loop closure actually recover? | HDL-Graph-SLAM, RTAB-Map |
| 4 | Can you localize against a prior map? | hdl_localization, FAST-LIO-Localization |

Eight distinct systems over five sequences, three runs each; "DLO without IMU" is an ablation of
DLO rather than a ninth system. Run count: 30 + 45 + 30 + 30 = **135**.

The four write-ups are published separately: **<add the write-up URL here>**.

## Checking the papers' numbers

Once you have collected runs, one command prints every number the papers report:

```bash
python3 print_tables.py                 # all four groups
python3 print_tables.py --group 3       # one group
python3 print_tables.py --runs          # per-run values, not aggregates
python3 print_tables.py --csv           # machine-readable
python3 print_tables.py --help          # conventions, and what each column means
```

Read `--help` before comparing anything. Three conventions decide whether a number matches:

- **Standard deviations are population** (NumPy's default, dividing by *n*), matching the paper
  captions. At *n*=3 the sample convention is 1.22× larger, which looks like a disagreement and
  is not one.
- **`ATE unal.` is only meaningful for Part 4**, where both systems start from a ground-truth
  seed pose in a fixed map frame. Parts 1–3 start at the origin in their own frame, so their
  unaligned numbers measure that offset rather than tracking quality. Ignore that column there.
- **Partial runs are included by default** and flagged beneath the table with their coverage.
  The papers do not treat them uniformly: Part 1 reports FAST-LIO2's truncated `boreas-urban`
  runs with a footnote, Part 3 excludes RTAB-Map's truncated `boreas-forest` run and reports
  that cell as *n*=2. `--drop-partial` reproduces Part 3's choice.

Everything else is downstream of whether the runs are sound at all:

```bash
python3 tools/check_run_health.py --verbose
```

A run that died partway through is still scored over the poses it produced, so it can post a
respectable error while covering half the route. That has happened in this project. **Check
health before drawing any conclusion from an error number.**

## Reproducing from scratch

Four stages. The first two are one-time; the third is where the time goes.

### 1. Prepare the dataset (on the host)

Budget **~510 GB**, in two halves: about 265 GB of downloaded sequence data, and about 245 GB of
bags built from it. Per sequence, largest first: urban 94 + 85 GB, forest 56 + 50, snowy 55 + 50,
suburb 45 + 41, highway 19 + 17. Part 4 adds roughly 1 GB of prior map per sequence. Conversion
is single-threaded and I/O bound; allow hours, not minutes.

Boreas-RT publishes 60 sequences named by collection time. These five are the ones measured
here; the short names on the left are this project's own labels, used everywhere in the harness:

| name used here | Boreas-RT sequence |
|---|---|
| `boreas-suburb` | `boreas-2024-12-03-12-54` |
| `boreas-highway` | `boreas-2024-12-04-12-34` |
| `boreas-snowy` | `boreas-2025-02-15-17-19` |
| `boreas-forest` | `boreas-2025-07-18-10-33` |
| `boreas-urban` | `boreas-2025-08-06-07-05` |

Fetch each one into the directory name this project uses. The `--include` filters matter: this
benchmark reads only the Velodyne scans, the IMU, the Applanix ground truth and the calibration,
and a full sequence carries camera, radar and Aeva data besides:

```bash
export BOREAS_DATASET=/absolute/path/to/your/boreas_data

aws s3 sync s3://boreas/boreas-2024-12-03-12-54 "$BOREAS_DATASET/boreas-suburb" \
  --exclude "*" --include "lidar/*" --include "imu/*" --include "applanix/*" --include "calib/*" \
  --no-sign-request
```

...and the same for the other four, substituting both names from the table. No AWS account is
needed; `--no-sign-request` makes the bucket public-read. Then:

```bash
pip3 install numpy scipy rosbags tqdm                    # host-side only
cd dataset

SEQS="forest highway snowy suburb urban"

# One bag per sequence. The harness looks for $BOREAS_DATASET/boreas_<name>.bag --
# note the underscore, where the sequence directory uses a hyphen.
for s in $SEQS; do
  ./boreas2rosbag.py "$BOREAS_DATASET/boreas-$s" --output "$BOREAS_DATASET/boreas_$s.bag"
done

# Ground truth in both body frames. This one walks all five sequences itself.
./format_gt.py --data-root "$BOREAS_DATASET"

# Prior maps -- Part 4 only, and slow. Writes ground_truth_map.pcd, which both
# Part 4 systems load.
for s in $SEQS; do
  ./build_ground_truth_map.py "boreas-$s" --data-root "$BOREAS_DATASET"
done
```

`format_gt.py` and `build_ground_truth_map.py` take `--data-root` because they originally lived
inside the dataset directory and resolved sequences relative to themselves.
`boreas2rosbag.py` derives the LiDAR↔IMU extrinsic from the dataset's own
calibration files and is the single source of truth for it — `format_gt.py` imports it rather
than hardcoding a lever arm, so the two cannot drift apart.

Each system is scored against the ground truth **in the frame it publishes in**: the `_lidar`
file for most, the `_imu` file for FAST-LIO-Localization. Every run's `_meta.json` records which
file it used, so this is checkable after the fact rather than assumed.

Point `BOREAS_DATASET` at the result:

```bash
cp .env.example .env && $EDITOR .env
```

### 2. Build the environment

```bash
./fetch_algorithms.sh                      # clone the upstream repos into catkin_ws_src/
./configure/configure_group1.py            # then group2, group3, group4 -- all four
./configure/configure_group2.py
./configure/configure_group3.py
./configure/configure_group4.py

docker compose up -d --build               # build the ROS Noetic image and start the container
docker exec boreas_slam_eval /root/patch_and_build.sh    # build fixes, then catkin build
```

The build is a separate step on purpose: `catkin_ws_src/` is a bind mount, so it does not exist
while the image is being built. `patch_and_build.sh` is baked into the image but has to run once
the container is up and the sources are visible. Expect it to take a while.

Order matters — `configure/` must run after the clone and before the build, because it writes
launch files the harness later invokes by name (`run_headless.launch`, `dlo_headless.launch`,
`dlo_headless_no_imu.launch`, `localization_boreas.launch`); none of those ship upstream.

`catkin_ws_src/` is gitignored deliberately: several upstream repos are copyleft, so this
repository documents *what to change* rather than redistributing modified copies. The
`configure/` scripts assert each substitution matches exactly once and fail loudly if an
upstream default has moved, instead of silently doing nothing. Each value they set is commented
in place with the reasoning for it, so the scripts are the record of what differs from upstream
and why — read them before trusting a comparison between any two systems here.

### 3. Collect (the long part)

All commands run inside the container:

```bash
docker exec boreas_slam_eval bash -lc '
  source /opt/ros/noetic/setup.bash && source /root/catkin_ws/devel/setup.bash
  cd /root/boreas_slam_benchmark && <command>'
```

`--algorithm` takes one of:

| Part | names |
|---|---|
| 1 | `fastlio2`, `liosam` |
| 2 | `kissicp`, `dlo`, `dlo_no_imu` |
| 3 | `hdlgraphslam`, `rtabmap` |
| 4 | `hdl_localization`, `fastlio_localization` |

```bash
python3 collect_metrics.py --algorithm liosam --sequences boreas-suburb --runs 3
python3 collect_metrics.py --algorithm liosam --sequences boreas-suburb --run-numbers 2 3  # top up
python3 collect_all.py --algorithms fastlio2 liosam                   # whatever is missing
```

Collection is sequential and resumable: nothing runs in parallel, and a run is recorded complete
only once it has fully finished, so interrupting loses at most the run in flight. Runs play in
real time, so a full sweep is on the order of a week of wall clock.

### 4. Read the numbers

`print_tables.py`, above.

## What is held constant

Identical bags, one container, 0.4 m voxel leaf size everywhere, and one LiDAR-to-IMU extrinsic
derived once from the dataset calibration and applied to every system that needs it. Where a
system could not be given the same treatment as the others — a few could not — the deviation and
the reason for it are commented at the point in `configure/` or `nodes/` where it is applied.

Hardware: Intel Core i9-14900 (32 logical cores), 62 GB RAM, Ubuntu 20.04, ROS Noetic.
Scoring: `evo` 1.31.1, NumPy 1.17.4, SciPy 1.10.1. CPU figures are cores on that 32-core machine,
so they are not portable to other hardware; accuracy figures are.

**Versions are pinned.** `fetch_algorithms.sh` checks out the exact upstream commit each result
was produced with, and the Dockerfile pins the pip packages, so a clone made later measures the
same code rather than whatever the default branches have become. One thing is not pinned and can
drift: `ros-noetic-rtabmap-ros` comes from apt, and was 0.21.13 when these runs were collected.

## Repository map

```
collect_metrics.py     Runs ONE system over sequences x runs. The core harness.
collect_all.py         Runs collect_metrics.py across many systems, sequentially, resumably.
metrics_lib.py         Scoring: ATE, RPE, drift per 10 m, CPU/RAM, real-time factor.
print_tables.py        Prints every number the papers report, from the collected runs.

dataset/               Dataset preparation, run on the host before anything else.
configure/             Applies our parameter changes onto freshly cloned upstream repos.
nodes/                 ROS nodes written for this project to make systems run on this rig;
                       grouped by which systems need them -- see nodes/README.md.
tools/check_run_health.py  Flags truncation, dropped scans and frozen tracking.

Dockerfile             Build/runtime environment.
docker-compose.yml     Mounts: dataset (ro), benchmark_results (rw), this repo (rw).
.env.example           Copy to .env and point BOREAS_DATASET at your prepared dataset.
LICENSE, NOTICE        Apache-2.0.
CITATION.cff           How to cite this benchmark, and the dataset it uses.
fetch_algorithms.sh    Clones the upstream repos into catkin_ws_src/.
patch_and_build.sh     Build fixes, then catkin build.
```

`catkin_ws_src/` and `benchmark_results/` are gitignored — the first for licensing, the second
because it is ~144 GB of generated data.

## Citing the dataset

The sequences come from **Boreas-RT ("Boreas Road Trip")**, which is CC BY 4.0: commercial use is
permitted and **attribution is required**. Anything published from these numbers must cite:

> D. Lisus, K. M. Papais, C. Le Gentil, E. Preston-Krebs, A. Lambert, K. Y. K. Leung and
> T. D. Barfoot, "Boreas Road Trip: A Multi-Sensor Autonomous Driving Dataset on Challenging
> Roads," arXiv:2602.16870 [cs.RO], 2026.

## Licensing

This repository's own code is **Apache-2.0** (`LICENSE`, `NOTICE`).
