# Helper ROS nodes

Original nodes written for this project — not part of any upstream algorithm. Each one exists
because an algorithm needed something this rig doesn't provide out of the box (a calibration it has
no mechanism for, a topic it never publishes, a map it can't load from a launch arg).

They're launched automatically by `collect_metrics.py` as `aux_procs` (or as the algorithm's own
`launch_cmd`), resolved via its `NODES` constant. Folders group them by which algorithms need them.
Their CPU/RAM is counted toward the algorithm's own totals wherever the node is genuinely required
to run that algorithm correctly, rather than being a measurement-harness artifact.

Each node's own docstring explains why it exists and what it replaces.

## `shared/`

| Node | Why it exists |
|---|---|
| `imu_frame_rotator.py` | DLO and RTAB-Map have no extrinsic calibration mechanism at all. Both integrate raw IMU every frame, so feeding them uncompensated data (the IMU is mounted ~43° plus an axis flip from the LiDAR) corrupts every registration — not something a post-hoc trajectory alignment can undo. Rotates angular velocity and linear acceleration into the LiDAR frame and republishes. Used by **group 2** (DLO) and **group 3** (RTAB-Map). |

## `group2/` — pure LiDAR odometry

| File | Why it exists |
|---|---|
| `kiss_icp_ros1_node.py` | KISS-ICP's own ROS wrapper is ROS2-only as of v0.4.0+; the last ROS1 tag is a year of improvements behind. Rather than benchmark a stale version, this wraps the current `kiss_icp` Python core directly. Also strides the raw scan before registration to stay inside the real-time budget on this rig's dense 128-beam scans. |
| `kiss_icp_boreas.yaml` | KISS-ICP tuning for this dataset. Its `max_range` also bounds the internal local map, which matters more than it looks — see the comments in the file. |

## `group3/` — full SLAM

| Node | Why it exists |
|---|---|
| `map2odom_bridge.py` | HDL-Graph-SLAM never publishes its loop-closure-corrected trajectory as a topic — only raw pre-correction odometry plus a separate `map->odom` TF. This broadcasts that correction, replacing the stock publisher, which used wall-clock stamps against the bag's real timestamps and so could never resolve. |
| `tf_corrected_pose_republisher.py` | Composes `map->odom` with `odom->base_link` into the single corrected odometry topic `collect_metrics.py` records. Used by HDL-Graph-SLAM and RTAB-Map. |

## `group4/` — map-based localization

| Node | Why it exists |
|---|---|
| `gt_initial_pose_publisher.py` | Every group 4 configuration needs a starting pose prior. Publishes one derived from ground truth, latched — a localizer given no prior is being tested on a different problem than the one this benchmark asks about. |
