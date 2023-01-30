#!/usr/bin/python2

import os
import sys

import rospy
import tf
import numpy as np

from std_msgs.msg import Float64MultiArray, Float32
from geometry_msgs.msg import Vector3
from hpe_ros_package.msg import TorsoJointPositions


# TODO:
# - Camera transformation https://www.cs.toronto.edu/~jepson/csc420/notes/imageProjection.pdf
# - Read camera_info
# - add painting of a z measurements

class hpe2uavcmd():

    def __init__(self, freq):

        rospy.init_node("hpe2cmd", log_level=rospy.DEBUG)

        self.rate = rospy.Rate(int(float(freq)))

        # Recv var
        self.hpe3d_recv = False
        # Calibration vars
        self.calib_first = True
        self.p_list = []

        # Initialize publishers and subscribers
        self._init_subscribers()
        self._init_publishers()

        self.camera_frame_name = "camera_color_frame"
        # Initialize transform listener
        self.tf_br = tf.TransformListener()
        self.ntf_br = tf.TransformBroadcaster()

        rospy.loginfo("[Hpe3D] started!")

    def _init_subscribers(self):

        # self.hpe_3d_sub         = rospy.Subscriber("camera/color/image_raw", Image, self.hpe3d_cb, queue_size=1)
        self.hpe_3d_sub = rospy.Subscriber("upper_body_3d", TorsoJointPositions, self.hpe3d_cb, queue_size=1)

    def _init_publishers(self):

        # self.q_pos_cmd_pub = rospy.Publisher("")
        # TODO: Add publisher for publishing joint angles
        # CMD publishers
        # Publish commands :)
        # self.roll_pub = rospy.Publisher("roll")
        # self.pitch_pub = rospy.Publisher("pitch")
        # self.yaw_pub = rospy.Publisher("yaw")
        # self.height_pub = rospy.Publisher("height")
        pass

    def hpe3d_cb(self, msg):

        # Msg has header, use it for timestamping
        self.hpe3d_recv_t = msg.header.stamp  # -> to system time
        # Extract data --> Fix pBaseNeck
        self.p_base_lwrist = self.createPvect(msg.left_wrist)

        # recieved HPE 3D
        self.hpe3d_recv = True

    def createPvect(self, msg):
        # Create position vector from Vector3
        return np.array([msg.x, msg.y, msg.z])

    def calibrate(self, timeout):

        if self.calib_first:
            self.start_time = rospy.Time.now().to_sec()
            elapsed = 0
        else:
            elapsed = rospy.Time.now().to_sec() - self.start_time

        if self.hpe3d_recv and elapsed < timeout:
            rospy.loginfo("Calibration procedure running {}".format(elapsed))
            self.calib_first = False
            self.p_list.append(self.p_base_lwrist)
            return False

        else:
            rospy.loginfo("Calibration procedure finished!")
            x = [p[0] for p in self.p_list]
            y = [p[1] for p in self.p_list]
            z = [p[2] for p in self.p_list]
            self.calib_point = Vector3()
            self.calib_point.x = sum(x)/len(x)
            self.calib_point.y = sum(y)/len(y)
            self.calib_point.z = sum(z)/len(z)
            rospy.loginfo("Calibration point is: {}".format(self.calib_point))
            return True

    def run_ctl(self):

        dist_x = (self.calib_point.x - self.p_base_lwrist[0])
        dist_y = (self.calib_point.y - self.p_base_lwrist[1]) * (-1)
        dist_z = (self.calib_point.z - self.p_base_lwrist[2]) * (-1)

        self.body_ctl = Vector3()
        self.body_ctl.x = dist_x; self.body_ctl.y = dist_y; self.body_ctl.z = dist_z

        # TODO: How to convert these measurements into commands
        # Z - height
        # X - pitch
        # Y - roll||yaw

        debug = False
        if debug:

            rospy.loginfo("dx: {}\t dy: {}\t dz: {}\t".format(dist_x, dist_y, dist_z))

    def run(self):

        calibrated = False
        while not rospy.is_shutdown():
            # Multiple conditions neccessary to run program!
            run_ready = self.hpe3d_recv
            calibration_timeout = 15

            # First run condition
            if run_ready and not calibrated:

                calibrated = self.calibrate(calibration_timeout)

            # We can start control if we have calibrated point
            if run_ready and calibrated:
                self.run_ctl()


            self.rate.sleep()



if __name__ == "__main__":
    hpe2cmd_ = hpe2uavcmd(sys.argv[1])
    hpe2cmd_.run()