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
from geometry_msgs.msg import PoseStamped, Pose, Transform, TwistStamped
from visualization_msgs.msg import Marker
from hpe_ros_msgs.msg import HumanPose3D, HandPose3D, MpHumanPose3D
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint

from utils import getZeroTwist, getZeroTransform, createMarkerArrow
from linalg_utils import pointToArray, get_RotX, get_RotY, get_RotZ, create_homogenous_vector, create_homogenous_matrix


# TODO:
# - Camera transformation https://www.cs.toronto.edu/~jepson/csc420/notes/imageProjection.pdf
# - Read camera_info
# - add painting of a z measurements
# - Add stuff for recording data 

# Constants
# TODO: Set as launch file argument
HPE = "OPENPOSE"
UAV_CMD_TOPIC_NAME = "/red/tracker/input_pose"
UAV_POS_TOPIC_NAME = "/red/pose"
ARM_CMD_TOPIC_NAME = "/control_arm/delta_twist_cmds"
TRAJ_CMD_TOPIC_NAME = "/red/position_hold/trajectory"

if HPE == "OPENPOSE":
    HPE3D_PRED_TOPIC_NAME = "/hpe3d/openpose_hpe3d"
    LHAND3D_PRED_TOPIC_NAME = "/hpe3d/lhand3d"
    RHAND3D_PRED_TOPIC_NAME = "/hpe3d/rhand3d"
    hpe_msg_type = HumanPose3D
    hand_msg_type = HandPose3D

if HPE == "MPI":
    HPE3D_PRED_TOPIC_NAME = "/mp_ros/loc/hpe3d"
    hpe_msg_type = MpHumanPose3D

CTL_TYPE = "POSITION" # RATE 
#CTL_TYPE = "RATE"

class hpe2amcmd():

    def __init__(self, freq):

        rospy.init_node("hpe2cmd", log_level=rospy.INFO)

        self.rate = rospy.Rate(int(float(freq)))

        # Recv var
        self.hpe3d_recv = False
        # Calibration vars
        self.calib_first = True
        self.pr_list = []
        self.pl_list = []

        # Initialize publishers and subscribers
        self._init_subscribers()
        self._init_publishers()

        self.camera_frame_name = "camera_color_frame"
        # Initialize transform listener
        self.tf_br = tf.TransformListener()
        self.ntf_br = tf.TransformBroadcaster()

        self.prev_pose_ref = PoseStamped()
        self.currentPose = PoseStamped()
        self.currRot = np.eye(3)
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
        self.l_hand_3d_sub = rospy.Subscriber(LHAND3D_PRED_TOPIC_NAME, hand_msg_type, self.lhand3d_cb, queue_size=1)
        self.r_hand_3d_sub = rospy.Subscriber(RHAND3D_PRED_TOPIC_NAME, hand_msg_type, self.rhand3d_cb, queue_size=1)

    def _init_publishers(self):

        # self.q_pos_cmd_pub = rospy.Publisher("")
        # TODO: Add publisher for publishing joint angles
        # CMD publishers
        # Publish commands :)
        self.gen_r_pub = rospy.Publisher("/uav/pose_ref", PoseStamped, queue_size=1)
        self.test_r_pub = rospy.Publisher("/uav/test_ref", Vector3, queue_size=1)
        self.arm_pub = rospy.Publisher(ARM_CMD_TOPIC_NAME, TwistStamped, queue_size=10)        
        self.pos_pub = rospy.Publisher(UAV_CMD_TOPIC_NAME, PoseStamped, queue_size=1)
        self.traj_pub = rospy.Publisher(TRAJ_CMD_TOPIC_NAME, MultiDOFJointTrajectoryPoint, queue_size=1)
        self.cb_point_marker_pub = rospy.Publisher("ctl/cb_point", Marker, queue_size=1)    
        self.r_hand_normal = rospy.Publisher("r_h_normal", Marker, queue_size=1)
        self.l_hand_normal = rospy.Publisher("l_h_normal", Marker, queue_size=1)

        # Check which marker to pub
        # self.marker_pub = rospy.Publisher("ctl/viz", Marker, queue_size=1) 

    def hpe3d_cb(self, msg):
        rospy.loginfo_once("Recieved HPE3D message")
        self.hpe3d_msg = HumanPose3D()
        self.hpe3d_msg = msg
        self.hpe3d_recv = True

    def lhand3d_cb(self, msg): 
        rospy.loginfo_once("Recieved L Hand3D message")
        self.lhand3d_msg = HandPose3D()
        self.lhand3d_msg = msg
        self.lhand3d_recv = True
        self.l_n = self.gen_hand_normal(self.lhand3d_msg.thumb0, self.lhand3d_msg.wrist,
                                        self.lhand3d_msg.index0, self.lhand3d_msg.pinky0)
        m = createMarkerArrow(rospy.Time.now(), [self.lhand3d_msg.wrist.x, self.lhand3d_msg.wrist.y, self.lhand3d_msg.wrist.z],
                              [self.lhand3d_msg.wrist.x + self.l_n[0], self.lhand3d_msg.wrist.y + self.l_n[1], self.lhand3d_msg.wrist.z + self.l_n[2]], 0)
        self.l_hand_normal.publish(m)

    def rhand3d_cb(self, msg): 
        rospy.loginfo_once("Recieved R Hand3D message")
        self.rhand3d_msg = HandPose3D()
        self.rhand3d_msg = msg
        self.rhand3d_recv = True
        self.r_n = self.gen_hand_normal(self.rhand3d_msg.thumb0, self.rhand3d_msg.wrist,
                                        self.rhand3d_msg.index0, self.rhand3d_msg.pinky0)
        m = createMarkerArrow(rospy.Time.now(), [self.rhand3d_msg.wrist.x, self.rhand3d_msg.wrist.y, self.rhand3d_msg.wrist.z],
                                [self.rhand3d_msg.wrist.x + self.r_n[0], self.rhand3d_msg.wrist.y + self.r_n[1], self.rhand3d_msg.wrist.z + self.r_n[2]], 1)
        self.r_hand_normal.publish(m)


    def pos_cb(self, msg):

        self.pos_recv = True
        self.currentPose = PoseStamped()
        self.currentPose.header = msg.header
        self.currentPose.pose.position.x = msg.pose.position.x
        self.currentPose.pose.position.y = msg.pose.position.y
        self.currentPose.pose.position.z = msg.pose.position.z
        self.currentPose.pose.orientation = msg.pose.orientation
        self.currRot = Rotation.from_quat([self.currentPose.pose.orientation.x,
                                           self.currentPose.pose.orientation.y,
                                           self.currentPose.pose.orientation.z,
                                           self.currentPose.pose.orientation.w]).as_matrix()

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
            self.pr_list.append(self.b_d_rw)
            self.pl_list.append(self.b_d_lw)
            return False

        else:
            rospy.loginfo("Calibration procedure finished!")
            n = 25 # Remove first 50 measurements, movement at first is not stable
            rx = [p[0] for p in self.pr_list][n:]; lx = [p[0] for p in self.pl_list][n:]
            ry = [p[1] for p in self.pr_list][n:]; ly = [p[1] for p in self.pl_list][n:]
            rz = [p[2] for p in self.pr_list][n:]; lz = [p[2] for p in self.pl_list][n:]
            self.r_cp = Vector3(); self.l_cp = Vector3()
            self.r_cp.x = np.median(rx); self.l_cp.x = np.median(lx)
            self.r_cp.y = np.median(ry); self.l_cp.y = np.median(ly)
            self.r_cp.z = np.median(rz); self.l_cp.z = np.median(lz)
            rospy.loginfo("l_cp: {}".format(self.l_cp))
            rospy.loginfo("r_cp: {}".format(self.r_cp))
            return True

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
            self.b_d_rw = bTc[-2] # [Right wrist]
            # added substraction by c_d_torso because it is already included in the bD
            self.b_d_lw = bTc[-1] # [Left wrist]
            #rospy.loginfo(f"b_d_rw: {self.b_d_rw}")
            #rospy.loginfo(f"b_d_lw: {self.b_d_lw}")
            #torso_msg = self.packSimpleTorso3DMsg(bD)
            #self.upper_body_3d_pub.publish(torso_msg)
        except Exception as e:
            rospy.logwarn("Failed to generate or publish HPE3d message: {}".format(e))

    # TODO: Write it as a matrix because this is horrendous
    def run_ctl(self, r, R):
        
        # Calc pos r 
        drx = self.b_d_rw[0] - self.r_cp.x; dlx = self.b_d_lw[0] - self.l_cp.x 
        dry = self.b_d_rw[1] - self.r_cp.y; dly = self.b_d_lw[1] - self.l_cp.y  
        drz = -(self.b_d_rw[2] - self.r_cp.z); dlz = -(self.b_d_lw[2] - self.l_cp.z)
        dr = np.array([drx, dry, drz]); dl = np.array([dlx, dly, dlz])

        self.body_ctl = Pose()
        self.b_cmd = Vector3()
        self.test_r_pub.publish(self.b_cmd)

        # b_cmd --> body UAV cmd; a_cmd --> ee ARM cmd 
        self.b_cmd = self.calc_body_cmd(r, R, dr)
        self.a_cmd = self.calc_body_cmd(r, R, dl)
        
        rospy.logdebug("X: {}\t Y: {}\t Z: {}\t".format(self.b_cmd.x, self.b_cmd.y, self.b_cmd.z))
        # TODO: Move to roslaunch params

        # Generate pose_ref --> if DRY run do not generate cmd
        gen_uav = True
        if gen_uav:
            usX = 0.01; usY = 0.01; usZ = 0.01;
            pos_ref = self.gen_uav_cmd(usX, usY, usZ)
            #print(type(pos_ref))
            self.prev_pose_ref.pose.position.x = pos_ref.transforms[0].translation.x
            self.prev_pose_ref.pose.position.y = pos_ref.transforms[0].translation.y
            self.prev_pose_ref.pose.position.z = pos_ref.transforms[0].translation.z
            self.prev_pose_ref.pose.orientation = pos_ref.transforms[0].rotation
            self.traj_pub.publish(pos_ref)

        gen_arm = True 
        if gen_arm: 
            asX = 3.0; asY = 3.0; asZ = 3.0
            current_time = rospy.Time.now()
            vel_ref = TwistStamped()
            vel_ref.header.stamp = current_time
            vel_ref.header.frame_id = "end_effector_base"
            vel_ref.twist.linear.x = self.a_cmd.x * asX
            vel_ref.twist.linear.y = self.a_cmd.y * asY  
            vel_ref.twist.linear.z = self.a_cmd.z * asZ
            self.arm_pub.publish(vel_ref)
        
        # ARROW to visualize direction of a command
        # TODO: Visualize ARM command
        dr_ = np.linalg.norm(dr)
        
        #rAMsg = self.create_marker(Marker.ARROW, self.r_cp.x, self.r_cp.y, self.r_cp.z, 
        #                           drx/dr_, dry/dr_, drz/dr_)
        #self.marker_pub.publish(rAMsg)
        self.first = False

    def calc_body_cmd(self, r, R, d): 
        if r < np.linalg.norm(d) < R:
            return Vector3(d[0], d[1], d[2])
        else:
            return Vector3(0, 0, 0)
        
    def gen_hand_normal(self, thumb, wrist, index, pinky): 
        # For left hand
        t = np.array([thumb.x, thumb.y, thumb.z])
        w = np.array([wrist.x, wrist.y, wrist.z])
        i = np.array([index.x, index.y, index.z])
        p = np.array([pinky.x, pinky.y, pinky.z])

        vt = t - w;         
        norm_vt = np.linalg.norm(vt); 
        if norm_vt != 0:
            vt_ = vt / norm_vt
        
        vi = i - w; 
        norm_vi = np.linalg.norm(vi)
        if norm_vi != 0: 
            vi_ = vi / norm_vi
        
        vp = p - w; 
        norm_vp = np.linalg.norm(vp)
        if norm_vp != 0:
            vp_ = vp / norm_vp

        if norm_vt != 0 and norm_vi != 0: 
            return np.cross(vt_, vi_)
        
        else: 
            return np.array([0, 0, 0])

    def gen_uav_cmd(self, sx, sy, sz):
        pos_ref = PoseStamped()
        if self.first:
            pos_ref.pose.position = self.currentPose.pose.position
            pos_ref.pose.orientation = self.currentPose.pose.orientation
        else: 
            type_ = "LOCAL"
            if type_ == "LOCAL":
                b_cmd = np.matmul(self.currRot.T, np.array([self.b_cmd.x, 
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

        # Publish debug pos_ref for plotting
        self.gen_r_pub.publish(pos_ref)
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
                rospy.loginfo_throttle(1, f"b_d_rw: {self.b_d_rw}")
                rospy.loginfo_throttle(1, f"b_d_lw: {self.b_d_lw}")

            # We can start control if we have calibrated point
            if run_ready and calibrated:
                r_ = 0.05
                R_ = 0.25 
                # Deadzone is 
                self.run_ctl(r_, R_)
                
                # Publish markers of calib pts
                #cbMarker = self.create_marker(Marker.SPHERE, self.l_cp.x, 
                #                              self.lp.y, self.l_cp.z)
                rospy.sleep(rospy.Duration(0.01))

if __name__ == "__main__":
    try:
        hpe2amcmd_ = hpe2amcmd(sys.argv[1])
        hpe2amcmd_.run()
    except KeyboardInterrupt:
        print("Shutting down gracefully...")