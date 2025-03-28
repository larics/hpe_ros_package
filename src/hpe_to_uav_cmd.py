#!/usr/bin/python3

import os
import sys

import rospy
import tf
import numpy as np
import copy
from scipy.spatial.transform import Rotation

from std_msgs.msg import Float64MultiArray, Float32
from geometry_msgs.msg import Vector3
from hpe_ros_msgs.msg import TorsoJointPositions
from geometry_msgs.msg import PoseStamped, Pose, Transform
from visualization_msgs.msg import Marker
from hpe_ros_msgs.msg import HumanPose3D, HandPose3D, MpHumanPose3D
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint

from utils import getZeroTwist, getZeroTransform
from linalg_utils import pointToArray, get_RotX, get_RotY, get_RotZ, create_homogenous_vector, create_homogenous_matrix


# TODO:
# - Camera transformation https://www.cs.toronto.edu/~jepson/csc420/notes/imageProjection.pdf
# - Read camera_infoWWWWWWW
# - add painting of a z measurements

# Constants
# TODO: Set as launch file argument
HPE = "OPENPOSE"
UAV_CMD_TOPIC_NAME = "/red/tracker/input_pose"
UAV_POS_TOPIC_NAME = "/red/pose"
TRAJ_CMD_TOPIC_NAME = "/red/position_hold/trajectory"

if HPE == "OPENPOSE":
    HPE3D_PRED_TOPIC_NAME = "/hpe3d/openpose_hpe3d"
    hpe_msg_type=HumanPose3D

if HPE == "MPI":
    HPE3D_PRED_TOPIC_NAME = "/mp_ros/loc/hpe3d"
    hpe_msg_type=MpHumanPose3D

CTL_TYPE = "POSITION" # RATE 
#CTL_TYPE = "RATE"

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

        # TODO: How to move to a body frame (any movement of my hand should be in the body frame)
        if HPE == "OPENPOSE":
            # body in the camera coordinate frame 
            self.bRc = np.matmul(get_RotZ(np.pi), np.matmul(get_RotX(np.pi/2), get_RotY(np.pi/2)))
        if HPE == "MPI": 
            # body in the camera coordinate frame 
            self.bRc = get_RotY(np.pi)

    def _init_subscribers(self):

        # self.hpe_3d_sub  = rospy.Subscriber("camera/color/image_raw", Image, self.hpe3d_cb, queue_size=1)
        self.hpe_3d_sub = rospy.Subscriber(HPE3D_PRED_TOPIC_NAME, hpe_msg_type, self.hpe3d_cb, queue_size=1)
        self.pos_sub = rospy.Subscriber(UAV_POS_TOPIC_NAME, PoseStamped, self.pos_cb, queue_size=1)

    def _init_publishers(self):

        # self.q_pos_cmd_pub = rospy.Publisher("")
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

    def hpe3d_cb(self, msg):
        rospy.loginfo_once("Recieved HPE3D message")
        self.hpe3d_msg = HumanPose3D()
        self.hpe3d_msg = msg
        self.hpe3d_recv = True

    def hand3d_cb(self, msg): 
        rospy.loginfo_once("Recieved Hand3D message")
        self.hand3d_msg = HandPose3D()
        self.hand3d_msg = msg
        self.hand3d_recv = True
        # Orientation of the hand # Check STACKOVERFLOW

    def pos_cb(self, msg):

        self.pos_recv = True
        self.currentPose = PoseStamped()
        self.currentPose.header = msg.header
        self.currentPose.pose.position.x = msg.pose.position.x
        self.currentPose.pose.position.y = msg.pose.position.y
        self.currentPose.pose.position.z = msg.pose.position.z
        self.currentPose.pose.orientation = msg.pose.orientation
        self.currRot = Rotation.from_quat([self.currentPose.pose.orientation.x, self.currentPose.pose.orientation.y, self.currentPose.pose.orientation.z, self.currentPose.pose.orientation.w]).as_matrix()

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
            self.calib_point.x = np.median(x)
            self.calib_point.y = np.median(y)
            self.calib_point.z = np.median(z)
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
        try:

            # TODO: Get torso coordinate frame [move this to a method]
            # TODO: Compare this to the online estimation of the HPE by openpose
            c_d_ls = pointToArray(self.hpe3d_msg.l_shoulder)
            c_d_rs = pointToArray(self.hpe3d_msg.r_shoulder)
            c_d_torso = (c_d_ls + c_d_rs)/2
            #c_d_t  = pointToArray(self.hpe3d_msg.neck)
            c_d_n  = pointToArray(self.hpe3d_msg.nose)
            c_d_le = pointToArray(self.hpe3d_msg.l_elbow)
            c_d_re = pointToArray(self.hpe3d_msg.r_elbow)
            c_d_rw = pointToArray(self.hpe3d_msg.r_wrist)
            c_d_lw = pointToArray(self.hpe3d_msg.l_wrist)

            # Comented out OpenPose part ==> wrong nomenclature
            Tc = np.array([#create_homogenous_vector(c_d_t),
                           create_homogenous_vector(c_d_ls), 
                           create_homogenous_vector(c_d_rs), 
                           create_homogenous_vector(c_d_le), 
                           create_homogenous_vector(c_d_re), 
                           create_homogenous_vector(c_d_rw),
                           create_homogenous_vector(c_d_lw)])

         
            # thorax in the camera frame --> TODO: Fix transformations
            T = create_homogenous_matrix(self.bRc, -np.matmul(self.bRc, c_d_torso))
            # T_inv = np.linalg.inv(T)
            # This seems like ok transformation for beginning :) 
            bTc = np.matmul(T, Tc.T).T
            # This is in the coordinate frame of the camera
            self.bD = bTc
            # Right wrist in the body frame
            # self.b_d_rw = np.matmul(self.bRc, c_d_rw) #- c_d_n) 
            # Left wrist in the body frame
            # self.b_d_lw = np.matmul(self.bRc, c_d_lw) #- c_d_n)
            #self.publishMarkerArray(bD)
            # rospy.loginfo(f"bTC is {bTc}")
            self.b_d_rw = bTc[-2] # added substraction by c_d_torso because it is already included in the bD
            self.b_d_lw = bTc[-1] 
            #rospy.loginfo(f"b_d_rw: {self.b_d_rw}")
            #rospy.loginfo(f"b_d_lw: {self.b_d_lw}")


            #torso_msg = self.packSimpleTorso3DMsg(bD)
            #self.upper_body_3d_pub.publish(torso_msg)
        except Exception as e:
            rospy.logwarn("Failed to generate or publish HPE3d message: {}".format(e))

    # TODO: Write it as a matrix because this is horrendous
    def run_ctl(self, r, R):
        
        # Calc pos r 
        dist_x = self.b_d_rw[0] - self.calib_point.x 
        dist_y = self.b_d_rw[1] - self.calib_point.y  
        dist_z = -(self.b_d_rw[2] - self.calib_point.z)

        self.body_ctl = Pose()
        self.b_cmd = Vector3()
        self.test_r_pub.publish(self.b_cmd)

        # TODO: Check this as vect
        if R > abs(dist_x) > r:
            self.b_cmd.x = dist_x
        else:
            self.b_cmd.x = 0

        if R > abs(dist_y) > r:
            self.b_cmd.y = dist_y
        else:
            self.b_cmd.y = 0

        if R > abs(dist_z) > r:
            self.b_cmd.z = dist_z
        else:
            self.b_cmd.z = 0
        
        rospy.logdebug("X: {}\t Y: {}\t Z: {}\t".format(self.b_cmd.x, self.b_cmd.y, self.b_cmd.z))
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

    def generate_cmd(self, sx, sy, sz):
        pos_ref = PoseStamped()
        if self.first:
            pos_ref.pose.position = self.currentPose.pose.position
            pos_ref.pose.orientation = self.currentPose.pose.orientation
        else: 
            type_ = "LOCAL"
            if type_ == "LOCAL":
                b_cmd = np.matmul(self.currRot, np.array([self.b_cmd.x, 
                                                                         self.b_cmd.y,
                                                                         self.b_cmd.z]))
                self.b_cmd = Vector3(b_cmd[0], b_cmd[1], b_cmd[2])
            
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
            self.proc_hpe_est()
        
            # First run condition
            if run_ready and not calibrated:
                calibrated = self.calibrate(calib_duration)
                rospy.loginfo(f"b_d_rw: f{self.b_d_rw}")

            # We can start control if we have calibrated point
            if run_ready and calibrated:
                r_ = 0.05
                R_ = 0.25 
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