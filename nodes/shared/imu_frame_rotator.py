#!/usr/bin/env python3
"""
Republishes an Imu topic with angular_velocity/linear_acceleration rotated into the LiDAR frame
via R_lidar_dmu -- for algorithms with no extrinsic calibration mechanism of their own (e.g. DLO)
that would otherwise consume raw IMU vectors as if the IMU were co-located
with the LiDAR, when on this rig it's offset and rotated ~43 degrees plus an axis flip.

Only angular_velocity and linear_acceleration are rotated. `orientation` is passed through
unrotated -- it's the low-quality accel-derived placeholder from boreas2rosbag.py (yaw=0, not a
real estimate), and none of this project's algorithms read it directly (they integrate raw gyro
themselves), so there is nothing meaningful to rotate there.
"""

import numpy as np
import rospy
from sensor_msgs.msg import Imu

# Rotation part of T_lidar_dmu, computed from calib/T_applanix_lidar.txt & calib/T_applanix_dmu.txt
# via boreas2rosbag.py::extract_extrinsics(), the single source of truth for this extrinsic.
R_LIDAR_DMU = np.array([
    [0.728265, -0.685295, -0.001208],
    [-0.685295, -0.728266, 0.000785],
    [-0.001418, 0.000256, -0.999999],
])


def rotated(vec3, R):
    x, y, z = R @ np.array([vec3.x, vec3.y, vec3.z])
    return x, y, z


class ImuFrameRotator:
    def __init__(self, input_topic, output_topic):
        self.pub = rospy.Publisher(output_topic, Imu, queue_size=50)
        rospy.Subscriber(input_topic, Imu, self._callback, queue_size=50)

    def _callback(self, msg):
        out = Imu()
        out.header = msg.header
        out.header.frame_id = "lidar_link"
        out.orientation = msg.orientation
        out.orientation_covariance = msg.orientation_covariance

        out.angular_velocity.x, out.angular_velocity.y, out.angular_velocity.z = rotated(
            msg.angular_velocity, R_LIDAR_DMU
        )
        out.angular_velocity_covariance = msg.angular_velocity_covariance

        out.linear_acceleration.x, out.linear_acceleration.y, out.linear_acceleration.z = rotated(
            msg.linear_acceleration, R_LIDAR_DMU
        )
        out.linear_acceleration_covariance = msg.linear_acceleration_covariance

        self.pub.publish(out)


if __name__ == "__main__":
    rospy.init_node("imu_frame_rotator")
    input_topic = rospy.get_param("~input_topic", "/dmu/imu")
    output_topic = rospy.get_param("~output_topic", "/dmu/imu_lidar_frame")
    ImuFrameRotator(input_topic, output_topic)
    rospy.spin()
