#!/usr/bin/env python3
"""
Broadcasts HDL-Graph-SLAM's map->odom loop-closure correction as TF, using the correction
message's own (real, bag-time) timestamp.

Replaces hdl_graph_slam's stock map2odom_publisher.py node entirely (removed from the launch file
by configure_group3.py) rather than patching it in place: that upstream script broadcasts using
rospy.Time.now() -- the CONTAINER's wall clock -- while scan_matching_odometry_nodelet broadcasts
odom->base_link using the bag's real (2024) timestamps. tf2 can't bridge a permanent ~1.7-year gap
between the two frames, so every map->base_link lookup fails with an extrapolation error (confirmed
directly: "Lookup would require extrapolation 54403377s into the past"). The correction message is
already stamped with the keyframe's real timestamp
(hdl_graph_slam_nodelet.cpp's `matrix2transform(keyframe->stamp, ...)`) -- this just uses it.

Running as a standalone script (not a catkin package node) also sidesteps an intermittent
FileNotFoundError seen when roslaunch resolved the upstream script through
devel/.private/hdl_graph_slam/ -- a catkin/rospkg path-resolution quirk in this container's devel
volume, not a logic bug, but not worth fighting either.
"""

import rospy
import tf
from geometry_msgs.msg import TransformStamped


class Map2OdomBridge:
    def __init__(self, correction_topic):
        self.odom_msg = None
        self.broadcaster = tf.TransformBroadcaster()
        rospy.Subscriber(correction_topic, TransformStamped, self._callback)

    def _callback(self, msg):
        self.odom_msg = msg

    def spin(self):
        if self.odom_msg is None:
            # Don't broadcast a wall-clock-stamped placeholder while waiting for the first real
            # correction: tf2 rejects same-edge broadcasts that are chronologically older than
            # what it's already seen, so a single rospy.Time.now() (2026) broadcast here would
            # permanently poison this edge -- every later real bag-time (2024) correction would
            # look "old" by comparison and get silently dropped forever (confirmed directly: this
            # was happening to 100% of corrections before this fix). Skip instead; map->odom
            # simply won't exist in TF until the first real correction arrives, which lookups
            # handle as an ordinary (transient) LookupException.
            return

        t = self.odom_msg.transform
        pos = (t.translation.x, t.translation.y, t.translation.z)
        quat = (t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w)
        self.broadcaster.sendTransform(
            pos, quat, self.odom_msg.header.stamp, self.odom_msg.child_frame_id, self.odom_msg.header.frame_id
        )


if __name__ == "__main__":
    rospy.init_node("map2odom_bridge")
    correction_topic = rospy.get_param("~correction_topic", "/hdl_graph_slam/odom2pub")

    bridge = Map2OdomBridge(correction_topic)
    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        bridge.spin()
        rate.sleep()
