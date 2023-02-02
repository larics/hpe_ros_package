#!/usr/bin/python2

import os
import sys

import rospy
import tf
import copy
import numpy as np

from PIL import ImageDraw, ImageFont
from PIL import Image as PILImage
from img_utils import convert_pil_to_ros_img

from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from std_msgs.msg import Float64MultiArray, Float32
from geometry_msgs.msg import Vector3
from hpe_ros_msgs.msg import TorsoJointPositions, JointArmCmd, CartesianArmCmd

import sensor_msgs.point_cloud2 as pc2

# TODO:
# - Camera transformation https://www.cs.toronto.edu/~jepson/csc420/notes/imageProjection.pdf
# - Read camera_info 
# - add painting of a z measurements  
# - This is something that could be useful!

class hpe2armcmd():

    def __init__(self, freq):

        rospy.init_node("hpe2cmd", log_level=rospy.DEBUG)

        self.rate = rospy.Rate(int(float(freq)))

        self.hpe3d_recv = False

        # Initialize publishers and subscribers
        self._init_subscribers()
        self._init_publishers()

        # Obtain 4 DoF angles 
        self.unit_x = np.array([1, 0, 0])
        self.unit_y = np.array([0, 1, 0])
        self.unit_z = np.array([0, 0, 1])

        self.m_dict = {"shoulder_rpitch": [], "shoulder_rroll": [], "shoulder_ryaw": [], 
                       "shoulder_lpitch": [], "shoulder_lroll": [], "shoulder_lyaw": [], 
                       "relbow": [], "lelbow": []}

        self.camera_frame_name = "camera_color_frame"
        # Initialize transform listener                  
        self.tf_br = tf.TransformListener()
        self.ntf_br = tf.TransformBroadcaster()

        rospy.loginfo("[Hpe3D] started!")


    def _init_subscribers(self):

        #self.hpe_3d_sub         = rospy.Subscriber("camera/color/image_raw", Image, self.hpe3d_cb, queue_size=1)
        self.hpe_3d_sub = rospy.Subscriber("upper_body_3d", TorsoJointPositions, self.hpe3d_cb, queue_size=1)


    def _init_publishers(self): 

        # Publish array of messages
        # self.q_pos_cmd_pub = rospy.Publisher("")
        # TODO: Add publisher for publishing joint angles
        self.left_arm_pub = rospy.Publisher("left_arm", JointArmCmd)
        self.right_arm_pub = rospy.Publisher("right_arm", JointArmCmd)
        self.cleft_arm_pub = rospy.Publisher("cart_left_arm", CartesianArmCmd)
        self.cright_arm_pub = rospy.Publisher("cart_right_arm", CartesianArmCmd)

    def arrayToVect(self, array): 

        v = Vector3()
        v.x = array[0]
        v.y = array[1]
        v.z = array[2]
        return v


    def hpe3d_cb(self, msg):

        # Msg has header, use it for timestamping 
        self.hpe3d_recv_t = msg.header.stamp # -> to system time 
        self.p_base_thorax  = self.createPvect(msg.thorax)        
        # Left arm
        self.p_base_lshoulder = self.createPvect(msg.left_shoulder) 
        self.p_base_lelbow = self.createPvect(msg.left_elbow)
        self.p_base_lwrist = self.createPvect(msg.left_wrist)
        # Right arm
        self.p_base_rshoulder = self.createPvect(msg.right_shoulder)
        self.p_base_relbow = self.createPvect(msg.right_elbow)
        self.p_base_rwrist = self.createPvect(msg.right_wrist)
        # Broadcast this vects as new TFs
        # p_base_shoulder - p_base_neck
        # Left arm --> these are the vectors from which we should extract angles :) 
        self.p_thorax_lshoulder = (self.p_base_lshoulder - self.p_base_thorax) * (-1)
        self.p_shoulder_lelbow = (self.p_base_lelbow - self.p_base_lshoulder) * (-1)
        self.p_elbow_lwrist = (self.p_base_lwrist - self.p_base_lelbow) * (-1)
        # Right arm 
        self.p_thorax_rshoulder = (self.p_base_rshoulder - self.p_base_thorax) * (-1)
        self.p_shoulder_relbow = (self.p_base_relbow - self.p_base_rshoulder) * (-1)
        self.p_elbow_rwrist = (self.p_base_rwrist - self.p_base_relbow) * (-1)
        # recieved HPE 3D
        self.hpe3d_recv = True


    def publish_left_arm(self): 

        try:
            # Joint space
            armCmdMsg = JointArmCmd()
            armCmdMsg.header.stamp = rospy.Time().now()
            armCmdMsg.shoulder_pitch.data = float(self.lpitch)
            armCmdMsg.shoulder_roll.data = float(self.lroll)
            armCmdMsg.shoulder_yaw.data = float(self.lyaw)
            armCmdMsg.elbow.data = float(self.lelbow)
            self.left_arm_pub.publish(armCmdMsg)
            # Cartesian space
            cartArmCmdMsg = CartesianArmCmd()
            cartArmCmdMsg.header.stamp = rospy.Time().now()
            cartArmCmdMsg.positionEE = self.arrayToVect(-1 * self.p_base_lwrist)
            cartArmCmdMsg.velocityEE = self.arrayToVect(self.v_base_lwrist)
            self.cleft_arm_pub.publish(cartArmCmdMsg)

            
        except Exception as e: 
            rospy.logwarn("[LeftArmMsg] Exception encoutered {}".format(str(e)))



    def publish_right_arm(self): 

        try: 
            # Joint space
            armCmdMsg = JointArmCmd()
            armCmdMsg.header.stamp = rospy.Time().now()
            armCmdMsg.shoulder_pitch.data = float(self.rpitch)
            armCmdMsg.shoulder_roll.data = float(self.rroll)
            armCmdMsg.shoulder_yaw.data = float(self.ryaw)
            armCmdMsg.elbow.data = float(self.relbow)
            self.right_arm_pub.publish(armCmdMsg)
            cartArmCmdMsg = CartesianArmCmd()
            cartArmCmdMsg.header.stamp = rospy.Time().now()
            cartArmCmdMsg.positionEE = self.arrayToVect(-1 * self.p_base_rwrist)
            cartArmCmdMsg.velocityEE = self.arrayToVect(self.v_base_rwrist)
            self.cright_arm_pub.publish(cartArmCmdMsg)

        except Exception as e: 
            rospy.logwarn("[RightArmMsg] Exception encountered: {}".format(str(e)))


    def send_transform(self, p_vect, parent_frame, child_frame):

        debug = False
        if debug: 
            rospy.loginfo("P^{0}_{1}: {2}".format(parent_frame, child_frame, p_vect))

        self.ntf_br.sendTransform((p_vect[0], p_vect[1], p_vect[2]), 
                                  (0, 0, 0, 1), 
                                  rospy.Time.now(), 
                                  child_frame, 
                                  parent_frame)


    def send_arm_transforms(self): 

        try:
            self.send_transform(self.p_thorax_lshoulder, 
                                "n_thorax", 
                                "left_shoulder")
            self.send_transform(self.p_shoulder_lelbow, 
                                "left_shoulder",
                                "left_elbow")
            self.send_transform(self.p_elbow_lwrist, 
                                "left_elbow", 
                                "left_wrist")
            self.send_transform(self.p_thorax_rshoulder, 
                                "n_thorax", 
                                "right_shoulder")
            self.send_transform(self.p_shoulder_relbow, 
                                "right_shoulder", 
                                "right_elbow")
            self.send_transform(self.p_elbow_rwrist, 
                                "right_elbow", 
                                "right_wrist")
        except Exception as e: 
            rospy.logwarn("Sending arm transforms failed: {}".format(str(e)))


    def get_larm_angles(self, p_shoulder_lelbow, p_elbow_lwrist): 
        # No delay filtering 
        # https://www.planetanalog.com/five-things-to-know-about-prediction-and-negative-delay-filters/
        

        self.lpitch = self.get_angle(p_shoulder_lelbow, 'xz', 'z')    # mediolateral axis (yz) 
        self.lroll = self.get_angle(p_shoulder_lelbow, 'yz', 'z')     # anterior axis shoulder (xz)
        self.lyaw = self.get_angle(p_elbow_lwrist, 'xy', 'x')         # longitudinal axis (xy)
        self.lelbow = self.get_angle(p_elbow_lwrist, 'yz', 'z')       # elbow rotational axis

        if p_shoulder_lelbow[0] < 0:
            self.lpitch *= -1
        if p_elbow_lwrist[0] < 0: 
            self.lelbow *= -1
        if p_shoulder_lelbow[1] < 0:
            self.lyaw *= -1
        if p_shoulder_lelbow[1] > 0: 
            self.lroll *= -1

    
    def get_rarm_angles(self, p_shoulder_relbow, p_elbow_rwrist): 
        
        self.rpitch = self.get_angle(p_shoulder_relbow, 'xz', 'z')
        self.rroll = self.get_angle(p_shoulder_relbow, 'yz', 'z')
        self.ryaw = self.get_angle(p_elbow_rwrist, 'xy', 'x')
        self.relbow = self.get_angle(p_elbow_rwrist, 'yz', 'z')

        if p_shoulder_relbow[0] < 0: 
            self.rpitch *= -1
        if p_elbow_rwrist[0] < 0: 
            self.relbow *= -1
        if p_shoulder_relbow[1] < 0: 
            self.ryaw *= -1
        if p_shoulder_relbow[1] > 0: 
            self.rroll *= -1

    def get_arm_velocities(self):

        # Velocity calculation
        try:
            meas_t = rospy.Time.now().to_sec()
            self.v_base_lwrist = (self.p_base_rwrist - self.prev_p_base_lwrist) * (meas_t - self.last_pass_t)
            self.v_base_rwrist = (self.p_base_lwrist - self.prev_p_base_rwrist) * (meas_t - self.last_pass_t)
            self.prev_p_base_lwrist = copy.deepcopy(self.p_base_lwrist)
            self.prev_p_base_rwrist = copy.deepcopy(self.p_base_rwrist)
            self.last_pass_t = meas_t
        except Exception as e: 
            rospy.logwarn("Calculating EE velocity: {}".format(str(e)))
            self.last_pass_t = meas_t
            self.prev_p_base_lwrist = copy.deepcopy(self.p_base_lwrist)
            self.prev_p_base_rwrist = copy.deepcopy(self.p_base_rwrist)
            self.v_base_rwrist = Vector3(0, 0, 0)
            self.v_base_rwrist = Vector3(0, 0, 0)


    def filter_arm(self, roll, pitch, yaw, elbow, arm): 
        # Overusage of copy.deepcopy (bad usage of the class and variable definitions -> HACKING!)
        roll_ = self.m_avg_filter(copy.deepcopy(roll), window_size, "shoulder_{}roll".format(arm))
        pitch_ = self.m_avg_filter(copy.deepcopy(pitch), window_size, "shoulder_{}pitch".format(arm))
        yaw_ = self.m_avg_filter(copy.deepcopy(yaw), window_size, "shoulder_{}yaw".format(arm))
        elbow_ = self.m_avg_filter(copy.deepcopy(elbow), window_size, "{}elbow".format(arm))
        return roll_, pitch_, yaw_, elbow


    def filter_avg(self, measurement, window_size, var_name):

        self.m_dict["{}".format(var_name)].append(measurement)

        rospy.logdebug("{}: {}".format(var_name, self.m_dict["{}".format(var_name)]))
        
        if len(self.m_dict["{}".format(var_name)]) < window_size: 
            return measurement
        else: 
            self.m_dict["{}".format(var_name)] = self.m_dict["{}".format(var_name)][-window_size:]
            avg  = sum(self.m_dict["{}".format(var_name)])/len(self.m_dict["{}".format(var_name)])

            return avg 
    

    def filter_lowpass(self, prev_meas, meas, coeff):

        s = 1 / (1 + coeff)
        f = 1 - coeff 

        filtered_meas = s * (prev_meas + meas - f * prev_meas)

        return filtered_meas



    def get_angle(self, p, plane="xy", rAxis = "x", format="degrees"): 

        # theta = cos-1 [ (a * b) / (abs(a) abs(b)) ]
        # Orthogonal projection of the vector to the wanted plane
        proj_p = self.getOrthogonalProjection(p, plane)

        if rAxis == "x": 
            vectBase = self.unit_x
        if rAxis == "y": 
            vectBase = self.unit_y
        if rAxis == "z": 
            vectBase = self.unit_z

        # All angles have 270deg offset -> radians      
        # TODO: Check if this is norm or absolute value
        value = np.dot(proj_p, vectBase)/(np.linalg.norm(proj_p))
        angle = np.arccos(value)

        if format == "degrees": 
            angle = np.degrees(angle)
        
        num_decimals = 4

        return np.round(angle, num_decimals)


    def createPvect(self, msg): 
        # Create position vector from Vector3
        return np.array([msg.x, msg.y, msg.z])


    def getOrthogonalProjection(self, x, plane): 

        if plane == "xy": 
            # Matrix is defined as a span of two vectors
            M = np.array([[1, 0], [0, 1], [0, 0]])
        if plane == "xz": 
            M = np.array([[1, 0], [0, 0], [0, 1]])
        if plane == "yz": 
            M = np.array([[0, 0], [-1, 0], [0, 1]])    

        Minv = np.linalg.inv(np.matmul(M.T, M))
        MTx = np.matmul(M.T, x)
        proj_x = np.matmul(np.matmul(M, Minv), MTx)

        return proj_x


    def getOrthogonalVect(self, vect, plane="xy"): 

        if plane=="xy": 
            vect_ = np.array([vect[0], vect[1], 0])
        
        if plane=="xz": 
            vect_ = np.array([vect[0], 0, vect[2]])

        if plane=="yz":
            vect_ = np.array([0, vect[1], vect[2]])
        
        return vect_


    def debug_print(self): 

        if not self.hpe3d_recv:
            rospy.logwarn_throttle(1, "Human pose estimation from camera stream has not been recieved.")


    def run(self): 

        while not rospy.is_shutdown(): 
            # Multiple conditions neccessary to run program!
            run_ready = self.hpe3d_recv 

            if run_ready: 

                # Maybe save indices for easier debugging
                start_time = rospy.Time.now().to_sec()
                # Get angles arm joint have
                self.get_larm_angles(copy.deepcopy(self.p_shoulder_lelbow),
                                     copy.deepcopy(self.p_elbow_lwrist))
                                    
                self.get_rarm_angles(copy.deepcopy(self.p_shoulder_relbow), 
                                     copy.deepcopy(self.p_elbow_rwrist))

                # Send transforms
                self.send_arm_transforms()

                filtering = True; window_size = 5; 
                
                if filtering == "avg":
                    
                    self.lpitch, self.lroll, self.lyaw, self.lelbow = self.filter_avg(self.lpitch, self.lroll, self.lyaw, self.lelbow, "l")
                    self.rpitch, self.rroll, self.ryaw, self.relbow = self.filter_avg(self.rpitch, self.rroll, self.ryaw, self.relbow, "r")

                if filtering == "lowpass": 
                    
                    coeff = 0.8
                    self.lpitch = self.filter_lowpass(self.prev_lpitch, copy.deepcopy(self.lpitch), 0.8); self.prev_lpitch = self.lpitch
                    self.lroll = self.filter_lowpass(self.prev_lroll, copy.deepcopy(self.lroll), 0.8); self.prev_lroll = self.lroll
                    self.lyaw = self.filter_lowpass(self.prev_lyaw, copy.deepcopy(self.lroll), 0.8); self.prev_lyaw = self.lyaw
                    self.lelbow = self.filter_lowpass(self.prev_lelbow, copy.deepcopy(self.lelbow), 0.8); self.prev_lelbow = self.lelbow

                    self.rpitch = self.filter_lowpass(self.prev_rpitch, copy.deepcopy(self.rpitch), 0.8); self.prev_rpitch = self.rpitch
                    self.rroll = self.filter_lowpass(self.prev_rroll, copy.deepcopy(self.rroll), 0.8); self.prev_rroll = self.rroll
                    self.ryaw = self.filter_lowpass(self.prev_ryaw, copy.deepcopy(self.rroll), 0.8); self.prev_ryaw = self.ryaw
                    self.relbow = self.filter_lowpass(self.prev_relbow, copy.deepcopy(self.relbow), 0.8); self.prev_relbow = self.relbow

                # get EE velocity
                self.get_arm_velocities()
                # publish vals for following
                self.publish_left_arm()
                self.publish_right_arm()
                
                measure_runtime = False; 
                if measure_runtime:
                    duration = rospy.Time.now().to_sec() - start_time
                    rospy.logdebug("Run t: {}".format(duration)) # --> very fast!

            else: 
                self.debug_print()
                
            self.rate.sleep()


if __name__ == "__main__": 

    hpe2armcmd_ = hpe2armcmd(sys.argv[1])
    hpe2armcmd_.run()