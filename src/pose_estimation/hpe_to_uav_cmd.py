#!/usr/bin/python3

import os
import sys

import rospy
import tf
import numpy as np
import copy

from std_msgs.msg import Float64MultiArray, Float32
from geometry_msgs.msg import Vector3
from hpe_ros_msgs.msg import TorsoJointPositions
from geometry_msgs.msg import PoseStamped, Pose
from visualization_msgs.msg import Marker
from hpe_ros_msgs.msg import HumanPose3D

from utils import pointToArray, create_homogenous_vector, create_homogenous_matrix, get_RotX, get_RotY, get_RotZ


# TODO:
# - Camera transformation https://www.cs.toronto.edu/~jepson/csc420/notes/imageProjection.pdf
# - Read camera_info
# - add painting of a z measurements

UAV_CMD_TOPIC_NAME = "/red/tracker/input_pose"
UAV_POS_TOPIC_NAME = "/red/pose"
HPE3D_PRED_TOPIC_NAME = "/hpe3d/openpose_hpe3d"

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

        self.prev_pose_ref = PoseStamped()
        self.first = True
        rospy.loginfo("[Hpe3D] started!")   

    def _init_subscribers(self):

        # self.hpe_3d_sub  = rospy.Subscriber("camera/color/image_raw", Image, self.hpe3d_cb, queue_size=1)
        self.hpe_3d_sub = rospy.Subscriber(HPE3D_PRED_TOPIC_NAME, HumanPose3D, self.hpe3d_cb, queue_size=1)
        self.pos_sub = rospy.Subscriber(UAV_POS_TOPIC_NAME, PoseStamped, self.pos_cb, queue_size=1)

    def _init_publishers(self):

        # self.q_pos_cmd_pub = rospy.Publisher("")
        # TODO: Add publisher for publishing joint angles
        # CMD publishers
        # Publish commands :)
        self.gen_r_pub = rospy.Publisher("/uav/pose_ref", Pose, queue_size=1)
        self.test_r_pub = rospy.Publisher("/uav/test_ref", Vector3, queue_size=1)
        self.pos_pub = rospy.Publisher(UAV_CMD_TOPIC_NAME, PoseStamped, queue_size=1)
        self.marker_pub = rospy.Publisher("ctl/viz", Marker, queue_size=1)
        self.cb_point_marker_pub = rospy.Publisher("ctl/cb_point", Marker, queue_size=1)    

    def hpe3d_cb(self, msg):
        self.hpe3d_recv = True
        rospy.loginfo_once("Recieved HPE3D message")
        self.hpe3d_msg = HumanPose3D()
        self.hpe3d_msg = msg

    def pos_cb(self, msg):

        self.pos_recv = True
        self.currentPose = PoseStamped()
        self.currentPose.header = msg.header
        self.currentPose.pose.position.x = msg.pose.position.x
        self.currentPose.pose.position.y = msg.pose.position.y
        self.currentPose.pose.position.z = msg.pose.position.z
        self.currentPose.pose.orientation = msg.pose.orientation

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
            # Get the position of the right wrist
            self.p_list.append(self.b_d_rw)
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
    
    def proc_hpe_est(self):
        # TODO: Everything in this method should be moved to the separate ctl script
        try:

            # TODO: Get torso coordinate frame [move this to a method]
            # TODO: Compare this to the online estimation of the HPE by openpose
            c_d_ls = pointToArray(self.hpe3d_msg.l_shoulder)
            c_d_rs = pointToArray(self.hpe3d_msg.r_shoulder)
            c_d_t  = pointToArray(self.hpe3d_msg.neck)
            c_d_n  = pointToArray(self.hpe3d_msg.nose)
            c_d_le = pointToArray(self.hpe3d_msg.l_elbow)
            c_d_re = pointToArray(self.hpe3d_msg.r_elbow)
            c_d_rw = pointToArray(self.hpe3d_msg.r_wrist)
            c_d_lw = pointToArray(self.hpe3d_msg.l_wrist)

            cD = np.array([create_homogenous_vector(c_d_t),
                           create_homogenous_vector(c_d_ls), 
                           create_homogenous_vector(c_d_rs), 
                           create_homogenous_vector(c_d_le), 
                           create_homogenous_vector(c_d_re), 
                           create_homogenous_vector(c_d_rw),
                           create_homogenous_vector(c_d_lw)])

            # body in the camera coordinate frame 
            bRc = np.matmul(get_RotX(np.pi/2), get_RotY(np.pi/2))
            # thorax in the camera frame --> TODO: Fix transformations
            T = create_homogenous_matrix(bRc, np.zeros(3))
            # T_inv = np.linalg.inv(T)
            # This seems like ok transformation for beginning :) 
            bD = np.matmul(T, cD.T).T
            # This is in the coordinate frame of the camera
            self.bD = bD

            self.b_d_rw = np.matmul(bRc, c_d_rw - c_d_t) 
            #rospy.loginfo("b_d_rw: {}".format(self.b_d_rw))
            self.b_d_lw = np.matmul(bRc, c_d_lw - c_d_t)
            #self.publishMarkerArray(bD)             

            #torso_msg = self.packSimpleTorso3DMsg(bD)
            #self.upper_body_3d_pub.publish(torso_msg)
        except Exception as e:
            rospy.logwarn("Failed to generate or publish HPE3d message: {}".format(e))

    # TODO: Write it as a matrix because this is horrendous
    def run_ctl(self, r, R):

        # Calc r
        dist_x = -(self.b_d_rw[0] - self.calib_point.x )
        dist_y = (self.b_d_rw[1] - self.calib_point.y ) 
        dist_z = -(self.b_d_rw[2] - self.calib_point.z ) 

        self.body_ctl = Pose()
        test_ref = Vector3()
        test_ref.x = dist_x
        test_ref.y = dist_y
        test_ref.z = dist_z
        self.test_r_pub.publish(test_ref)

        if R > abs(dist_x) > r:
            #rospy.logdebug("X: {}".format(dist_x))
            self.body_ctl.position.x = dist_x
        else:
            self.body_ctl.position.x = 0

        if R > abs(dist_y) > r:
            #rospy.logdebug("Y: {}".format(dist_y))
            self.body_ctl.position.y = dist_y
        else:
            self.body_ctl.position.y = 0

        if R > abs(dist_z) > r:
            #rospy.logdebug("Z: {}".format(dist_z))
            self.body_ctl.position.z = dist_z
        else:
            self.body_ctl.position.z = 0

        scaling_x = 0.25; scaling_y = 0.25; scaling_z = 0.25;
        pos_ref = PoseStamped()

        # Generate pose_ref
        if self.first:
            pos_ref.pose.position = self.currentPose.pose.position
            pos_ref.pose.orientation = self.currentPose.pose.orientation
            self.prev_pose_ref = pos_ref
        else: 
            pos_ref.pose.position.x = self.prev_pose_ref.pose.position.x + self.body_ctl.position.x * scaling_x
            pos_ref.pose.position.y = self.prev_pose_ref.pose.position.y + self.body_ctl.position.y * scaling_y
            pos_ref.pose.position.z = self.prev_pose_ref.pose.position.z + self.body_ctl.position.z * scaling_z
            pos_ref.pose.orientation = self.prev_pose_ref.pose.orientation
        
        self.prev_pose_ref = pos_ref
        
        self.pos_pub.publish(pos_ref)
        self.gen_r_pub.publish(self.body_ctl)

        # ARROW to visualize direction of a command
        arrowMsg = self.create_marker(Marker.ARROW, self.calib_point.x, self.calib_point.y, self.calib_point.z, 
                                      dist_x, dist_y, dist_z)
        self.marker_pub.publish(arrowMsg)
        self.first = False

        debug = False
        if debug:
            rospy.loginfo("Dist z: {}".format(dist_z))
            rospy.loginfo("Publishing z: {}".format(pos_ref.z))
            rospy.loginfo("dx: {}\t dy: {}\t dz: {}\t".format(dist_x, dist_y, dist_z))

    def run(self):

        calibrated = False
        rospy.sleep(5.0)
        while not rospy.is_shutdown():
            # Multiple conditions neccessary to run program!
            run_ready = self.hpe3d_recv
            calibration_timeout = 10
            self.proc_hpe_est()
        
            # First run condition
            if run_ready and not calibrated:

                calibrated = self.calibrate(calibration_timeout)

            # We can start control if we have calibrated point
            if run_ready and calibrated:
                r_ = 0.10
                R_ = 0.40
                # Deadzone is 
                self.run_ctl(r_, R_)
                
                # Publish markers
                cbMarker = self.create_marker(Marker.SPHERE, self.calib_point.x, 
                                              self.calib_point.y, self.calib_point.z, 
                                              0.1, 0.1, 0.1)
                self.cb_point_marker_pub.publish(cbMarker)


            self.rate.sleep()



if __name__ == "__main__":
    hpe2uavcmd_ = hpe2uavcmd(sys.argv[1])
    hpe2uavcmd_.run()