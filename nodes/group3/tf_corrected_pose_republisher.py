#!/usr/bin/env python3
"""
Republishes a loop-closure-corrected pose as a single nav_msgs/Odometry topic, for any algorithm
that follows the common "raw odom topic + separate map->odom TF correction" pattern instead of
publishing its corrected trajectory directly. Used by both Group 3 algorithms:
- HDL-Graph-SLAM: `scan_matching_odometry_nodelet` publishes only raw, pre-loop-closure odometry;
  the correction is map2odom_bridge.py's TF broadcast.
- RTAB-Map: `icp_odometry` publishes only raw odometry (namespaced `/rtabmap/odom`); the main
  `rtabmap` node publishes only a map->odom TF correction (`publish_tf_map`), no corrected topic.

Recording just the raw odom topic would silently give the *uncorrected* trajectory -- pointless
for a "full SLAM/loop closure" group. This subscribes to the raw odometry purely as a "new pose
available" trigger (same rate the algorithm actually updates), looks up the composed
map->base_link transform via tf2 for each one, and republishes on a single topic so
collect_metrics.py's existing record-and-convert pipeline works unchanged.
"""

import rospy
import tf2_ros
from nav_msgs.msg import Odometry


class TfCorrectedPoseRepublisher:
    def __init__(self, trigger_topic, output_topic, map_frame, base_link_frame, odom_frame):
        self.map_frame = map_frame
        self.base_link_frame = base_link_frame
        self.odom_frame = odom_frame
        self.pub = rospy.Publisher(output_topic, Odometry, queue_size=10)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        rospy.Subscriber(trigger_topic, Odometry, self._callback, queue_size=10)

    def _callback(self, msg):
        # A plain lookup_transform(map, base_link, Time(0)) is WRONG here, and silently so.
        # map->base_link is a chain of two edges updated at very different rates: map->odom is the
        # loop-closure correction (keyframe rate -- as slow as ~0.2Hz) while odom->base_link is
        # per-scan (~10Hz). Time(0) means "the latest time available on the whole chain", which is
        # capped by the SLOWEST edge -- so every lookup between two keyframes returned the pose as
        # of the last keyframe, while line below stamps it with the current scan's timestamp. The
        # recorded trajectory became a staircase of stale poses carrying fresh timestamps
        # (confirmed directly: 7 unique positions across 249 poses on a 30s HDL-Graph-SLAM run, and
        # RTAB-Map frozen at 0.40m of travel against 144m actually driven).
        #
        # lookup_transform_full is the API for exactly this: take base_link at THIS message's time,
        # carry it through the fixed odom frame, and apply the latest available map->odom
        # correction. That yields a genuinely per-scan pose that still includes loop closure.
        try:
            t = self.tf_buffer.lookup_transform_full(
                target_frame=self.map_frame, target_time=rospy.Time(0),
                source_frame=self.base_link_frame, source_time=msg.header.stamp,
                fixed_frame=self.odom_frame,
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return  # no correction published yet (e.g. first keyframe) -- skip this one

        odom = Odometry()
        odom.header = msg.header
        odom.header.frame_id = self.map_frame
        odom.child_frame_id = self.base_link_frame
        odom.pose.pose.position.x = t.transform.translation.x
        odom.pose.pose.position.y = t.transform.translation.y
        odom.pose.pose.position.z = t.transform.translation.z
        odom.pose.pose.orientation = t.transform.rotation
        self.pub.publish(odom)


if __name__ == "__main__":
    rospy.init_node("tf_corrected_pose_republisher")
    trigger_topic = rospy.get_param("~trigger_topic", "/odom")
    output_topic = rospy.get_param("~output_topic", "/corrected_odom")
    map_frame = rospy.get_param("~map_frame", "map")
    base_link_frame = rospy.get_param("~base_link_frame", "base_link")
    # The intermediate frame the correction is applied above: map->odom is the corrected edge,
    # odom->base_link the per-scan one. Same name for both Group 3 algorithms, but exposed rather
    # than hardcoded since it's load-bearing for the composition above.
    odom_frame = rospy.get_param("~odom_frame", "odom")
    TfCorrectedPoseRepublisher(trigger_topic, output_topic, map_frame, base_link_frame, odom_frame)
    rospy.spin()
