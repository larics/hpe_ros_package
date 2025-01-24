#!/usr/bin/python3

import os
import sys

import rospy
import tf
import numpy as np
import copy

from geometry_msgs.msg import Vector3
from geometry_msgs.msg import PoseStamped, Pose, Transform, Twist
from visualization_msgs.msg import Marker
from trajectory_msgs.msg import MultiDOFJointTrajectoryPoint

from linalg_utils import pointToArray, create_homogenous_vector, create_homogenous_matrix, get_RotX, get_RotY, get_RotZ, getZeroTwist, getZeroTransform


# TODO:
# - Camera transformation https://www.cs.toronto.edu/~jepson/csc420/notes/imageProjection.pdf
# - Read camera_info
# - add painting of a z measurements

UAV_CMD_TOPIC_NAME = "/red/tracker/input_pose"
UAV_POS_TOPIC_NAME = "/red/pose"
TRAJ_CMD_TOPIC_NAME = "/red/position_hold/trajectory"

CTL_TYPE = "POSITION" # RATE 
CTL_TYPE = "RATE"

VR_POS_LW_TOPIC_NAME = "/vr/pos/lw"
VR_POS_RW_TOPIC_NAME = "/vr/pos/rw"
VR_TWIST_LW_TOPIC_NAME = "/vr/twist/lw"
VR_TWIST_RW_TOPIC_NAME = "/vr/twist/rw"
VR_POS_HEAD_TOPIC_NAME = "/vr/pos/head"
VR_TWIST_HEAD_TOPIC_NAME = "/vr/twist/head"


class vr2uavcmd():

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

        # body in the camera coordinate frame (T to move to the coordiante frame)
        self.bRvc = np.matmul(get_RotX(np.pi/2), get_RotY(np.pi/2))


    def _init_subscribers(self):

        # Pose msgs
        self.vr_pos_lw_sub = rospy.Subscriber(VR_POS_LW_TOPIC_NAME, Pose, self.hpe3d_cb, queue_size=1)
        self.vr_pos_rw_sub = rospy.Subscriber(VR_POS_RW_TOPIC_NAME, Pose, self.hpe3d_cb, queue_size=1)ž
        self.vr_pos_head_sub = rospy.Subscriber(VR_POS_HEAD_TOPIC_NAME, Pose, self.hpe3d_cb, queue_size=1)
        # Twist msgs
        self.vr_twist_rw_cb = rospy.Subscriber(VR_TWIST_RW_TOPIC_NAME, Twist, self.hpe3d_cb, queue_size=1)
        self.vr_twist_lw_sub = rospy.Subscriber(VR_TWIST_LW_TOPIC_NAME, Twist, self.hpe3d_cb, queue_size=1)
        self.vr_twist_head_sub = rospy.Subscriber(VR_TWIST_HEAD_TOPIC_NAME, Twist, self.hpe3d_cb, queue_size=1)
        # Uav msgs
        self.pos_sub = rospy.Subscriber(UAV_POS_TOPIC_NAME, PoseStamped, self.pos_cb, queue_size=1)

    def _init_publishers(self):

        # TODO: Add publisher for publishing joint angles
        # CMD publishers
        # Publish commands :)
        self.gen_r_pub = rospy.Publisher("/uav/pose_ref", Pose, queue_size=1)
        self.test_r_pub = rospy.Publisher("/uav/test_ref", Vector3, queue_size=1)
        self.pos_pub = rospy.Publisher(UAV_CMD_TOPIC_NAME, PoseStamped, queue_size=1)
        self.traj_pub = rospy.Publisher(TRAJ_CMD_TOPIC_NAME, MultiDOFJointTrajectoryPoint, queue_size=1)
        self.marker_pub = rospy.Publisher("ctl/viz", Marker, queue_size=1)
        self.cb_point_marker_pub = rospy.Publisher("ctl/cb_point", Marker, queue_size=1)    
        self.r_hand_normal = rospy.Publisher("ctl/r_hand_normal", Vector3, queue_size=1)

    def vr_pos_lw_cb(self, msg):
        self.p_lw = pointToArray(msg)
        #self.R_lw = quatToRot(msg.orientation)

    def vr_pos_rw_cb(self, msg): 
        self.p_rw = pointToArray(msg)
        #self.R_rw = quatToRot(msg.orientation)

    def vr_twist_lw_cb(self, msg):
        self.v_lw = pointToArray(msg)
        #self.Rdot_lw = quatToRot(msg.orientation)

    def vr_twist_rw_cb(self, msg):
        self.v_rw = pointToArray(msg)
        #self.Rdot_rw = quatToRot(msg.orientation)

    def vr_pos_head_cb(self, msg):
        self.p_h = pointToArray(msg)
        self.R_h = quatToRot(msg.orientation)


    def vr_twist_head_cb(self):
        pass

    def pos_cb(self, msg):

        self.pos_recv = True
        self.currentPose = PoseStamped()
        self.currentPose.header = msg.header
        self.currentPose.pose.position.x = msg.pose.position.x
        self.currentPose.pose.position.y = msg.pose.position.y
        self.currentPose.pose.position.z = msg.pose.position.z
        self.currentPose.pose.orientation = msg.pose.orientation

    def calibrate(self, timeout):

        if self.calib_first:
            self.start_time = rospy.Time.now().to_sec()
            elapsed = 0
        else:
            elapsed = rospy.Time.now().to_sec() - self.start_time

        if self.hpe3d_recv and elapsed < timeout:
            rospy.loginfo_throttle(1, "Calibration procedure running {}".format(elapsed))
            self.calib_first = False
            # Get the position of the right wrist
            self.p_list.append(self.b_d_rw)
            return False

        else:
            rospy.loginfo("Calibration procedure finished!")
            n = 25 # Remove first 50 measurements, movement at first is not stable
            x = [p[0] for p in self.p_list][n:]
            y = [p[1] for p in self.p_list][n:]
            z = [p[2] for p in self.p_list][n:]
            self.calib_point = Vector3()
            self.calib_point.x = sum(x)/len(x)
            self.calib_point.y = sum(y)/len(y)
            self.calib_point.z = sum(z)/len(z)
            rospy.loginfo("Calibration point is: {}".format(self.calib_point))
            return True
    
    def proc_vr(self):
        try:

            # TODO: Get torso coordinate frame [move this to a method]
            c_d_lw = pointToArray(self.hpe3d_msg.l_wrist)
            c_d_rw = pointToArray(self.hpe3d_msg.r_wrist)

            # Comented out OpenPose part
            cD = np.array([create_homogenous_vector(c_d_rw), 
                           create_homogenous_vector(c_d_lw)])

            # thorax in the camera frame --> TODO: Fix transformations
            T = create_homogenous_matrix(self.bRvc, np.zeros(3))
            # T_inv = np.linalg.inv(T)
            # This seems like ok transformation for beginning :) 
            bD = np.matmul(T, cD.T).T
            # This is in the coordinate frame of the camera
            self.bD = bD
            # Right wrist in the body frame
            self.b_d_lw = np.matmul(self.bRc, c_d_lw) #- c_d_n) 
            # Left wrist in the body frame
            self.b_d_rw = np.matmul(self.bRc, c_d_rw) #- c_d_n)
            #self.publishMarkerArray(bD)             

            #torso_msg = self.packSimpleTorso3DMsg(bD)
            #self.upper_body_3d_pub.publish(torso_msg)
        except Exception as e:
            rospy.logwarn("Failed to generate or publish HPE3d message: {}".format(e))

    # TODO: Write it as a matrix because this is horrendous
    def run_ctl(self, r, R):
        
        # Calc pos r 
        dist_x = (self.calib_point.x - self.b_d_rw[0])
        dist_y = (self.calib_point.y - self.b_d_rw[1]) 
        dist_z = (self.calib_point.z - self.b_d_rw[2]) 

        self.body_ctl = Pose()
        self.b_cmd = Vector3()
        self.test_r_pub.publish(self.b_cmd)

        # TODO: Check this as vect
        if R > abs(dist_x) > r:
            rospy.logdebug("X: {}".format(dist_x))
            self.b_cmd.x = dist_x
        else:
            self.b_cmd.x = 0

        if R > abs(dist_y) > r:
            rospy.logdebug("Y: {}".format(dist_y))
            self.b_cmd.y = dist_y
        else:
            self.b_cmd.y = 0

        if R > abs(dist_z) > r:
            rospy.logdebug("Z: {}".format(dist_z))
            self.b_cmd.z = dist_z
        else:
            self.b_cmd.z = 0
            
        # TODO: Move to roslaunch params
        scaling_x = 0.05; scaling_y = 0.05; scaling_z = 0.05;

        # Generate pose_ref --> if DRY run do not generate cmd
        gen_r = True
        if gen_r:
            pos_ref = self.generate_cmd(scaling_x, scaling_y, scaling_z)
            #print(type(pos_ref))
            self.prev_pose_ref.pose.position.x = pos_ref.transforms[0].translation.x
            self.prev_pose_ref.pose.position.y = pos_ref.transforms[0].translation.y
            self.prev_pose_ref.pose.position.z = pos_ref.transforms[0].translation.z
            self.prev_pose_ref.pose.orientation = pos_ref.transforms[0].rotation
            self.traj_pub.publish(pos_ref)
        debug = False
        if debug:
            rospy.loginfo("dx: {}\t dy: {}\t dz: {}\t".format(dist_x, dist_y, dist_z))

        # ARROW to visualize direction of a command
        arrowMsg = self.create_marker(Marker.ARROW, self.calib_point.x, self.calib_point.y, self.calib_point.z, 
                                      dist_x, dist_y, dist_z)
        self.marker_pub.publish(arrowMsg)
        self.first = False

    # TODO: Move this to the separate methods
    def generate_cmd(self, sx, sy, sz):
        pos_ref = PoseStamped()
        if self.first:
            pos_ref.pose.position = self.currentPose.pose.position
            pos_ref.pose.orientation = self.currentPose.pose.orientation
        else: 
            pos_ref.pose.position.x = self.prev_pose_ref.pose.position.x + sx * self.b_cmd.x
            pos_ref.pose.position.y = self.prev_pose_ref.pose.position.y + sy * self.b_cmd.y
            pos_ref.pose.position.z = self.prev_pose_ref.pose.position.z + sz * self.b_cmd.z
            pos_ref.pose.orientation = self.prev_pose_ref.pose.orientation

        trajPt = MultiDOFJointTrajectoryPoint()
        if CTL_TYPE == "POSITION":
            # Generate MultiDOFJointTrajectory
            trajPt.transforms.append(Transform())
            trajPt.transforms[0].translation.x = pos_ref.pose.position.x
            trajPt.transforms[0].translation.y = pos_ref.pose.position.y
            trajPt.transforms[0].translation.z = pos_ref.pose.position.z
            trajPt.transforms[0].rotation = pos_ref.pose.orientation
            trajPt.velocities.append(getZeroTwist())
            trajPt.accelerations.append(getZeroTwist())

        if CTL_TYPE == "RATE": 
            trajPt.transforms.append(getZeroTransform())
            trajPt.transforms[0].translation.x = pos_ref.pose.position.x
            trajPt.transforms[0].translation.y = pos_ref.pose.position.y
            trajPt.transforms[0].translation.z = pos_ref.pose.position.z
            trajPt.transforms[0].rotation = pos_ref.pose.orientation
            trajPt.velocities.append(getZeroTwist())
            trajPt.velocities[0].linear.x = self.b_cmd.x
            trajPt.velocities[0].linear.y = self.b_cmd.y
            trajPt.velocities[0].linear.z = self.b_cmd.z
            trajPt.velocities[0].angular.x = 0
            trajPt.velocities[0].angular.y = 0
            trajPt.velocities[0].angular.z = 0
            trajPt.accelerations.append(getZeroTwist())
        return trajPt

    def run(self):

        calibrated = False
        rospy.sleep(5.0)
        while not rospy.is_shutdown():
            # Multiple conditions neccessary to run program!
            run_ready = self.hpe3d_recv
            calib_duration = 10
            self.proc_vr()
        
            # First run condition
            if run_ready and not calibrated:
                calibrated = self.calibrate(calib_duration)

            # We can start control if we have calibrated point
            if run_ready and calibrated:
                r_ = 0.05
                R_ = 0.15 
                # Deadzone is 
                self.run_ctl(r_, R_)
                
                # Publish markers
                cbMarker = self.create_marker(Marker.SPHERE, self.calib_point.x, 
                                              self.calib_point.y, self.calib_point.z, 
                                              0.1, 0.1, 0.1)
                self.cb_point_marker_pub.publish(cbMarker)


            self.rate.sleep()



if __name__ == "__main__":
    vr2uavcmd_ = vr2uavcmd(sys.argv[1])
    vr2uavcmd_.run()