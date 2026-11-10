#!/usr/bin/env python3
"""
Minimal ROS1 (rospy) wrapper around KISS-ICP's current Python pipeline API.

KISS-ICP's own `ros/` package is ROS2-only as of v0.4.0+ (ament_cmake, rclcpp) -- the last
ROS1/catkin-compatible tag is v0.3.0, over a year of algorithm improvements behind current. Rather
than benchmark an outdated version against 7 current algorithms, this wraps the current
kiss_icp.kiss_icp.KissICP core directly. Requires `pip3 install <repo>/kiss-icp/python` first (see
patch_and_build.sh).

Per-point timestamp handling matches boreas2rosbag.py's PointCloud2 layout exactly (x,y,z,
intensity: float32; ring: uint16; time: float32; point_step=22), using the
real per-point time field normalized to KISS-ICP's expected [0,1] fraction-of-scan range -- higher
fidelity than upstream's own BoreasDataset loader, which only has azimuth to synthesize timestamps
from since the offline .bin reader it's written against doesn't carry one.
"""

from pathlib import Path

import numpy as np
import rospy
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import PointCloud2

from kiss_icp.config import load_config
from kiss_icp.kiss_icp import KissICP

POINT_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("intensity", "<f4"), ("ring", "<u2"), ("time", "<f4"),
])


# Stride applied to each raw scan before it ever reaches KissICP.register_frame(). The Velodyne
# Alpha-Prime here delivers ~200-225k pts/scan -- roughly 2x the ~100k KISS-ICP is normally
# benchmarked against on 64-channel KITTI-style data -- and KISS-ICP already voxelizes internally
# (0.2m/0.6m derived from mapping.voxel_size) before registration, so most of that extra raw
# density is redundant within a single voxel rather than distinct signal. Cutting it at the source
# reduces preprocessing/registration cost roughly in proportion, buying real margin under the
# ~104ms scan interval instead of running right at the edge of it (see kiss_icp_boreas.yaml and
# see this file's notes below for the map-growth/degeneracy fixes this compounds with).
INPUT_STRIDE = 3


class KissIcpNode:
    def __init__(self, input_topic, output_topic, config_path):
        config = load_config(config_path)
        self.odometry = KissICP(config=config)
        # queue_size=10 (~1s of buffer at this LiDAR's ~9.6Hz) was too thin to survive short bursts
        # of contention (rosbag recording flush, sysstats sampling, etc.) even once register_frame()
        # itself was brought under the ~104ms real-time budget on average (see kiss_icp_boreas.yaml) --
        # confirmed directly: a live production run still showed a bursty gap pattern (median
        # inter-pose gap 0.104s matching real-time, but a few gaps up to 5.6s), not a sustained
        # overload. Each Odometry message keeps the source scan's original header.stamp, so a
        # deeper queue only delays *when* a pose is computed, never which ground-truth timestamp
        # it's evaluated against -- purely a buffer against transient stalls, not a correctness risk.
        self.pub = rospy.Publisher(output_topic, Odometry, queue_size=200)
        rospy.Subscriber(input_topic, PointCloud2, self._callback, queue_size=200)

    def _callback(self, msg):
        cloud = np.frombuffer(msg.data, dtype=POINT_DTYPE)[::INPUT_STRIDE]
        points = np.column_stack([cloud["x"], cloud["y"], cloud["z"]]).astype(np.float64)

        t = cloud["time"].astype(np.float64)
        t_range = t.max() - t.min()
        timestamps = (t - t.min()) / t_range if t_range > 0 else np.zeros_like(t)

        self.odometry.register_frame(points, timestamps)
        self._publish_pose(msg.header, self.odometry.last_pose)

    def _publish_pose(self, header, pose_4x4):
        odom = Odometry()
        odom.header = header
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"

        tx, ty, tz = pose_4x4[:3, 3]
        qx, qy, qz, qw = R.from_matrix(pose_4x4[:3, :3]).as_quat()

        odom.pose.pose.position.x = tx
        odom.pose.pose.position.y = ty
        odom.pose.pose.position.z = tz
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw

        self.pub.publish(odom)


if __name__ == "__main__":
    rospy.init_node("kiss_icp_node")
    input_topic = rospy.get_param("~input_topic", "/velodyne/points")
    output_topic = rospy.get_param("~output_topic", "/kiss_icp/odometry")
    # Default sits next to this file, so the repo works at whatever path it's mounted at.
    default_config = str(Path(__file__).parent / "kiss_icp_boreas.yaml")
    config_path = rospy.get_param("~config_path", default_config)
    KissIcpNode(input_topic, output_topic, config_path)
    rospy.spin()
