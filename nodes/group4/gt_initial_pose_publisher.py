#!/usr/bin/env python3
"""
Publishes the sequence's first ground-truth pose as the initial pose for Group 4's map-based
localizers. hdl_localization
(hdl_localization_nodelet.cpp) subscribe to the same /initialpose
(geometry_msgs/PoseWithCovarianceStamped) convention -- normally fed by RViz's "2D Pose Estimate"
tool -- so one script serves both. Without this, both default to (0,0,0)/identity, which is wrong
given this dataset's large (UTM-like) world coordinates.

Published latched and exactly once (not repeatedly for several seconds, as an earlier version of
this script did) -- latching handles the subscriber-connection race correctly on its own: a
subscriber that connects after this single publish() call still receives that one message
immediately upon connecting, for as long as this process keeps running. Repeated publishing was
a REAL BUG, not just a redundant safety margin: an initialpose_callback and
hdl_localization_nodelet.cpp's equivalent (`pose_estimator.reset(new PoseEstimator(...))`) treat
*every* incoming /initialpose message as a fresh manual relocalization event, unconditionally
discarding whatever tracking progress had accumulated and resetting back to this message's (static)
pose. With the previous 5-25s repeat-publish window overlapping most of a 30s smoke-test run,
genuine tracking got stomped back to the start pose roughly every 0.5s for nearly the entire run --
confirmed directly as the root cause of both algorithms' apparent "frozen near the start" behavior,
via live inspection showing near-zero velocity and 0-1-iteration "convergence" throughout.

~imu_frame (default false): compose the LiDAR pose with T_lidar_dmu so the seed is T_world_imu
rather than T_world_lidar. Needed by FAST_LIO_LOCALIZATION, whose odometry origin is FAST-LIO's
gravity-aligned IMU body frame, not the LiDAR frame -- and this rig's extrinsic is a 180-degree
rotation (see below), so seeding it with the LiDAR pose puts the ICP initial guess exactly
backwards. hdl_localization tracks the LiDAR frame and leaves this false.

THE POSE COMES FROM applanix/lidar_poses.csv, NOT ground_truth_imu.tum -- and that distinction is
load-bearing, not a preference. `ground_truth_imu.tum` is T_world_imu; the reference map these
localizers match against is accumulated by build_ground_truth_map.py from `lidar_poses.csv`, which
is T_world_lidar. Those two frames differ by this rig's LiDAR-to-IMU extrinsic, whose rotation is
*exactly 180 degrees* (T_lidar_dmu's trace is -1.0)
plus the 0.28m the IMU sits below the LiDAR. Measured directly on three sequences: the two first
poses differ by 179.98 degrees and 0.280m, identically every time.

Seeding NDT 180 degrees away from the map it is matching against is hopeless -- NDT's convergence
basin is on the order of one voxel and a few degrees, so it fails on the very first frame and never
recovers. That was the real cause of BOTH Group 4 algorithms' long-standing "frozen position"
behaviour, and it is why every attempt to fix it by widening NDT resolution failed: the resolution
was never the problem. Reading the same file the map was built from makes the frames consistent by
construction, rather than by composing an extrinsic here that could be applied in the wrong
direction.
"""

import numpy as np
import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped
from scipy.spatial.transform import Rotation as R


# T_lidar_dmu -- the same matrix configure_group1.py writes into FAST-LIO2's extrinsic_R/_T, from
# calib/T_applanix_lidar.txt & calib/T_applanix_dmu.txt via boreas2rosbag.py::extract_extrinsics().
# Its trace is -1.0, i.e. a rotation of exactly 180 degrees; that is the whole reason ~imu_frame
# exists rather than the two frames being treated as interchangeable.
T_LIDAR_DMU_ROT = np.array([
    [0.728265, -0.685295, -0.001208],
    [-0.685295, -0.728266, 0.000785],
    [-0.001418, 0.000256, -0.999999],
])
T_LIDAR_DMU_TRANS = np.array([0.0, 0.0, -0.28])


def first_lidar_pose(csv_path):
    """First T_world_lidar from applanix/lidar_poses.csv, derived exactly as
    build_ground_truth_map.py derives it -- same columns, same compass-to-math heading conversion.
    Any divergence between the two would put the seed pose back out of the map's frame."""
    row = np.atleast_1d(np.genfromtxt(csv_path, delimiter=",", names=True, max_rows=1))[0]
    # Applanix "heading" is a compass bearing (0=North, clockwise-positive), not a math yaw
    # (0=+X/East, counter-clockwise-positive). Same conversion as build_ground_truth_map.py.
    # Rotation convention MUST match build_ground_truth_map.py exactly -- it defines the frame the
    # reference map lives in. Determined empirically there by scan-to-scan agreement:
    # ZYX[-heading, pitch, roll] gives 0.240m, the previous xyz[roll, pitch, pi/2-heading] 4.117m.
    quat = R.from_euler("ZYX", [-row["heading"], row["pitch"], row["roll"]]).as_quat()  # x,y,z,w
    pos = np.array([row["easting"], row["northing"], row["altitude"]])
    return pos, quat


if __name__ == "__main__":
    rospy.init_node("gt_initial_pose_publisher")
    lidar_poses_csv = rospy.get_param("~lidar_poses_csv")
    map_frame = rospy.get_param("~map_frame", "map")
    imu_frame = rospy.get_param("~imu_frame", False)

    pos, quat = first_lidar_pose(lidar_poses_csv)
    if imu_frame:
        R_world_lidar = R.from_quat(quat).as_matrix()
        pos = pos + R_world_lidar @ T_LIDAR_DMU_TRANS
        quat = R.from_matrix(R_world_lidar @ T_LIDAR_DMU_ROT).as_quat()
    x, y, z = pos
    qx, qy, qz, qw = quat
    msg = PoseWithCovarianceStamped()
    msg.header.frame_id = map_frame
    msg.pose.pose.position.x = x
    msg.pose.pose.position.y = y
    msg.pose.pose.position.z = z
    msg.pose.pose.orientation.x = qx
    msg.pose.pose.orientation.y = qy
    msg.pose.pose.orientation.z = qz
    msg.pose.pose.orientation.w = qw

    pub = rospy.Publisher("/initialpose", PoseWithCovarianceStamped, queue_size=1, latch=True)
    rospy.sleep(0.5)  # let this publisher's own connection register before the first publish
    msg.header.stamp = rospy.Time.now()
    pub.publish(msg)
    rospy.spin()
