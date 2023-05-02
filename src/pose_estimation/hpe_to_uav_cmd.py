#!/usr/bin/python2

import os
import sys

import rospy
import tf
import numpy as np

from std_msgs.msg import Float64MultiArray, Float32
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Vector3
from hpe_ros_package.msg import TorsoJointPositions
from geometry_msgs.msg import PoseStamped, Pose
from visualization_msgs.msg import Marker


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
        self.uav_pose_recv = False
        # Calibration vars
        self.calib_first = True
        self.p_list = []

        self.pose_sub_name = "/hawk1/vrpn_client/estimated_odometry"
        self.pose_pub_name = "/red/tracker/input_pose"
        
        # Initialize publishers and subscribers
        self._init_subscribers()
        self._init_publishers()

        self.camera_frame_name = "camera_color_frame"
        # Initialize transform listener
        self.tf_br = tf.TransformListener()
        self.ntf_br = tf.TransformBroadcaster()

        rospy.loginfo("[Hpe3D] started!")

    def _init_subscribers(self):

        self.hpe_3d_sub = rospy.Subscriber("upper_body_3d", TorsoJointPositions, self.hpe3d_cb, queue_size=1)
        self.pos_sub = rospy.Subscriber(self.pose_sub_name, Odometry, self.pos_cb, queue_size=1)

    def _init_publishers(self):

        # self.q_pos_cmd_pub = rospy.Publisher("")
        # TODO: Add publisher for publishing joint angles
        self.gen_r_pub = rospy.Publisher("/uav/r", Vector3, queue_size=10)
        self.pos_pub = rospy.Publisher(self.pose_pub_name, PoseStamped, queue_size=10)
        self.marker_pub = rospy.Publisher("ctl/viz", Marker, queue_size=10)
        self.cb_point_marker_pub = rospy.Publisher("ctl/cb_point", Marker, queue_size=10)    

    def hpe3d_cb(self, msg):
        
        rospy.loginfo_once("Recieved HPE!")
        # Msg has header, use it for timestamping
        self.hpe3d_recv_t = msg.header.stamp  # -> to system time
        # Extract data --> Fix pBaseNeck
        self.p_base_lwrist = self.createPvect(msg.left_wrist)

        # recieved HPE 3D
        self.hpe3d_recv = True

    def pos_cb(self, msg):
        
        rospy.loginfo_once("Recieved UAV pose!")
        self.uav_pose_recv = True
        self.currentPose = PoseStamped()
        self.currentPose.header = msg.header
        self.currentPose.pose = msg.pose.pose

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

    def create_marker(self, shape, px, py, pz, dist_x, dist_y, dist_z): 
        marker = Marker()
        marker.header.frame_id = "n_thorax"
        marker.header.stamp = rospy.Time().now()
        marker.ns = "arrow"
        marker.id = 0
        marker.type = shape
        marker.action = Marker.ADD
        marker.pose.position.x = self.calib_point.x
        marker.pose.position.y = self.calib_point.y
        marker.pose.position.z = self.calib_point.z
        # How to transform x,y,z values to the orientation 
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.25
        marker.scale.y = 0.25
        marker.scale.z = 0.25
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        return marker

    def run_ctl(self, r, R):

        dist_x = (self.p_base_lwrist[0] - self.calib_point.x )
        dist_y = (self.p_base_lwrist[1] - self.calib_point.y ) 
        dist_z = (self.p_base_lwrist[2] - self.calib_point.z ) 

        self.body_ctl = Vector3()

        # X,Y are swapped because CFs of UAV and World are rotated for 90 degs
        if R > abs(dist_y) > r:
            rospy.logdebug("Y: {}".format(dist_x))
            self.body_ctl.x = -dist_x
        else:
            self.body_ctl.x = 0

        if R > abs(dist_x) > r:
            rospy.logdebug("X: {}".format(dist_y))
            self.body_ctl.y = dist_y
        else:
            self.body_ctl.y = 0

        if R > abs(dist_z) > r:
            rospy.logdebug("Z: {}".format(dist_z))
            self.body_ctl.z = dist_z
        else:
            self.body_ctl.z = 0

        scaling_x = 3.0; scaling_y = 3.0; scaling_z = 1.0;
        pos_ref = PoseStamped()
        pos_ref.header.stamp = rospy.Time.now()
        pos_ref.pose.position.x = self.currentPose.pose.position.x + self.body_ctl.x * scaling_x
        pos_ref.pose.position.y = self.currentPose.pose.position.y + self.body_ctl.y * scaling_y
        pos_ref.pose.position.z = self.currentPose.pose.position.z + self.body_ctl.z * scaling_z
        pos_ref.pose.orientation = self.currentPose.pose.orientation
        
        self.pos_pub.publish(pos_ref)
        self.gen_r_pub.publish(self.body_ctl)

        # ARROW to visualize direction of a command
        arrowMsg = self.create_marker(Marker.ARROW, self.calib_point.x, self.calib_point.y, self.calib_point.z, 
                                      dist_x, dist_y, dist_z)
        self.marker_pub.publish(arrowMsg)

        debug = False
        if debug:
            rospy.loginfo("Dist z: {}".format(dist_z))
            rospy.loginfo("Publishing z: {}".format(pos_ref.z))
            rospy.loginfo("dx: {}\t dy: {}\t dz: {}\t".format(dist_x, dist_y, dist_z))

    def run(self):

        calibrated = False
        while not rospy.is_shutdown():
            # Multiple conditions neccessary to run program!
            run_ready = self.hpe3d_recv
            calibration_timeout = 10

            # First run condition
            if run_ready and not calibrated:

                calibrated = self.calibrate(calibration_timeout)

            # We can start control if we have calibrated point
            if run_ready and calibrated and self.uav_pose_recv:
                rospy.loginfo_throttle(1, "Started H2AMI control!")
                r_ = 0.05
                R_ = 0.5
                # Deadzone is 
                self.run_ctl(r_, R_)

            self.rate.sleep()



if __name__ == "__main__":
    hpe2uavcmd_ = hpe2uavcmd(sys.argv[1])
    hpe2uavcmd_.run()
