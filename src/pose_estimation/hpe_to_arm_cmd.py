#!/usr/bin/python2

import os
import sys

import rospy
import tf
import copy
import numpy as np

from utils import *

from std_msgs.msg import Float64, Float32
from geometry_msgs.msg import Vector3
from hpe_ros_msgs.msg import TorsoJointPositions, JointArmCmd, CartesianArmCmd, EstLinkLengths



# TODO:
# - Camera transformation https://www.cs.toronto.edu/~jepson/csc420/notes/imageProjection.pdf
# - Read camera_info 
# - add painting of a z measurements  


class hpe2armcmd():

    def __init__(self, freq):

        rospy.init_node("hpe2cmd", log_level=rospy.DEBUG)

        self.rate = rospy.Rate(int(float(freq)))

        self.hpe3d_recv = False
        self.calibrated = False

        self.calib_lpitch = 0; self.calib_lroll = 0;  
        self.calib_rpitch = 0; self.calib_rroll = 0; 

        # Initialize publishers and subscribers
        self._init_subscribers()
        self._init_publishers()

        # Obtain 4 DoF angles 
        self.unit_x = np.array([1, 0, 0])
        self.unit_y = np.array([0, 1, 0])
        self.unit_z = np.array([0, 0, 1])

        self.m_dict = {"right_pitch": [], "right_roll": [], "right_yaw": [], "right_elbow": [], 
                       "left_pitch": [], "left_roll": [], "left_yaw": [], "left_elbow": [],
                        "right_x": [], "right_y": [], "right_z": [], 
                        "left_x": [], "left_y": [], "left_z": []}
        
        self.calib_dict = {"right_pitch": [], "right_roll": [], 
                           "left_pitch": [], "left_roll": []}

        self.prev = {}

        self.camera_frame_name = "camera_color_frame"
        # Initialize transform listener                  
        self.tf_br = tf.TransformListener()
        self.ntf_br = tf.TransformBroadcaster()

        rospy.loginfo("[Hpe3D] started!")

        self.depth_detection_threshold = 0.002

        #f = open("{}-{}".format("jointPositions", datetime.now().strftime("%H:%M:%S"))

    def _init_subscribers(self):

        #self.hpe_3d_sub         = rospy.Subscriber("camera/color/image_raw", Image, self.hpe3d_cb, queue_size=1)
        self.hpe_3d_sub = rospy.Subscriber("upper_body_3d", TorsoJointPositions, self.hpe3d_cb, queue_size=1)

    def _init_publishers(self): 

        # Publish array of messages
        # self.q_pos_cmd_pub = rospy.Publisher("")
        # TODO: Add publisher for publishing joint angles
        self.left_arm_pub = rospy.Publisher("left_arm", JointArmCmd, queue_size=1)
        self.right_arm_pub = rospy.Publisher("right_arm", JointArmCmd, queue_size=1)
        self.cleft_arm_pub = rospy.Publisher("cart_left_arm", CartesianArmCmd, queue_size=1)
        self.cright_arm_pub = rospy.Publisher("cart_right_arm", CartesianArmCmd, queue_size=1)
        self.norm_left_arm_pub = rospy.Publisher("norm_left_arm", EstLinkLengths, queue_size=1)
        self.norm_right_arm_pub = rospy.Publisher("norm_right_arm", EstLinkLengths, queue_size=1)


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
        self.p_elbow_lwrist = (self.p_base_lwrist - self.p_base_lelbow) * (-1)
        self.p_shoulder_lelbow = (self.p_base_lelbow - self.p_base_lshoulder) * (-1)
        self.p_shoulder_lwrist = (self.p_base_lwrist - self.p_base_lshoulder) * (-1)
        
        # Right arm --> This should prevent suddent jumps in reference!
        self.p_thorax_rshoulder = (self.p_base_rshoulder - self.p_base_thorax) * (-1)
        self.p_elbow_rwrist = (self.p_base_rwrist - self.p_base_relbow) * (-1)
        self.p_shoulder_relbow = (self.p_base_relbow - self.p_base_rshoulder) * (-1)
        self.p_shoulder_rwrist = (self.p_base_rwrist - self.p_base_rshoulder) * (-1 )
        
        self.norm_p_s_lelbow = np.linalg.norm(self.p_shoulder_lelbow)
        self.norm_p_s_relbow = np.linalg.norm(self.p_shoulder_relbow)
        self.norm_p_e_lwrist = np.linalg.norm(self.p_elbow_lwrist)
        self.norm_p_e_rwrist = np.linalg.norm(self.p_elbow_rwrist)
        self.norm_p_s_lwrist = np.linalg.norm(self.p_shoulder_lwrist)
        self.norm_p_s_rwrist = np.linalg.norm(self.p_shoulder_rwrist)

        # recieved HPE 3D
        self.hpe3d_recv = True

    def publish_norms(self):
        # Publish norms of the vectors
        msg = Float32()
        # Left arm
        norm_links = 0.3
        normMsg = EstLinkLengths()
        normMsg.header.stamp = rospy.Time().now()
        msg.data = self.norm_p_s_lelbow/norm_links
        normMsg.shoulderElbow = copy.deepcopy(msg)
        msg.data = self.norm_p_e_lwrist/norm_links
        normMsg.elbowWrist = copy.deepcopy(msg)
        msg.data = self.norm_p_s_lwrist/(2*norm_links)
        normMsg.shoulderWrist = copy.deepcopy(msg)
        self.norm_left_arm_pub.publish(normMsg)

        # Right arm
        normMsg = EstLinkLengths()
        normMsg.header.stamp = rospy.Time().now()
        msg.data = self.norm_p_s_relbow/norm_links
        normMsg.shoulderElbow = copy.deepcopy(msg)
        msg.data = self.norm_p_e_rwrist/norm_links
        normMsg.elbowWrist = copy.deepcopy(msg)
        msg.data = self.norm_p_s_rwrist/(2*norm_links)
        normMsg.shoulderWrist = copy.deepcopy(msg)
        self.norm_right_arm_pub.publish(normMsg) 


    def publish_arm(self, pitch, roll, yaw, elbow, p_shoulder_wrist, v_shoulder_wrist, arm): 

        try: 
            # Joint space
            armCmdMsg = JointArmCmd()
            armCmdMsg.header.stamp          = rospy.Time().now()
            armCmdMsg.shoulder_pitch.data   = float(pitch)
            armCmdMsg.shoulder_roll.data    = float(roll)
            armCmdMsg.shoulder_yaw.data     = float(yaw)
            armCmdMsg.elbow.data            = float(elbow)
            cartArmCmdMsg = CartesianArmCmd()
            cartArmCmdMsg.header.stamp = rospy.Time().now()
            # Change z value for the cartesian command sending
            p_shoulder_wrist[2] *= -1
            cartArmCmdMsg.positionEE = arrayToVect(p_shoulder_wrist, Vector3()) # Wrong pos, not neccessary now!
            cartArmCmdMsg.velocityEE = arrayToVect(v_shoulder_wrist, Vector3())

            if arm == "right": 
                self.right_arm_pub.publish(armCmdMsg)
                self.cright_arm_pub.publish(cartArmCmdMsg)
            if arm == "left": 
                self.left_arm_pub.publish(armCmdMsg)
                self.cleft_arm_pub.publish(cartArmCmdMsg)


        except Exception as e: 
            rospy.logwarn("[{}ArmMsg] Exception encountered: {}".format(arm, str(e)))


    def publish_robot_arm(self, spitch, sroll, syaw, elbow): 

        spitchMsg = Float64()
        spitchMsg.data = np.radians(spitch)
        srollMsg = Float64()
        srollMsg.data = np.radians(sroll)
        syawMsg = Float64()
        syawMsg.data = np.radians(syaw)
        elbowMsg = Float64()
        elbowMsg.data = np.radians(elbow)


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
            self.send_transform(self.p_thorax_lshoulder, "n_thorax", "left_shoulder")
            self.send_transform(self.p_shoulder_lelbow, "left_shoulder", "left_elbow")
            self.send_transform(self.p_elbow_lwrist, "left_elbow", "left_wrist")
            self.send_transform(self.p_thorax_rshoulder, "n_thorax", "right_shoulder")
            self.send_transform(self.p_shoulder_relbow, "right_shoulder", "right_elbow")
            self.send_transform(self.p_elbow_rwrist, "right_elbow", "right_wrist")
        except Exception as e: 
            rospy.logwarn("Sending arm transforms failed: {}".format(str(e)))


    def get_arm_angles(self, p_shoulder_elbow, p_elbow_wrist, arm): 
        # No delay filtering 
        # https://www.planetanalog.com/five-things-to-know-about-prediction-and-negative-delay-filters/
        
        pitch = self.get_angle(p_shoulder_elbow, 'xz', 'z')    # mediolateral axis (yz) 
        roll = self.get_angle(p_shoulder_elbow, 'yz', 'z')     # anterior axis shoulder (xz)
        yaw = self.get_angle(p_elbow_wrist, 'xy', 'x')         # longitudinal axis (xy)

        #roll, pitch, yaw = self.get_RPY(p_shoulder_elbow)
        elbow = self.get_vect_angle(p_shoulder_elbow, p_elbow_wrist)       # elbow rotational axis

        if p_shoulder_elbow[0] < 0: 
            pitch *= -1
        if p_shoulder_elbow[1] > 0:
            roll *= -1
        if p_elbow_wrist[1] < 0: 
            yaw *= -1

        return pitch, roll, yaw, elbow    


    def get_RPY(self, p_shoulder_elbow):

        unit_p_shoulder_elbow = p_shoulder_elbow/np.linalg.norm(p_shoulder_elbow)
        rospy.loginfo("unit_p_shoulder_elbow: {}".format(unit_p_shoulder_elbow))
        x, y, z = unit_p_shoulder_elbow[0], unit_p_shoulder_elbow[1], unit_p_shoulder_elbow[2]

        # RPY angles --> pilot
        Y_ = np.arctan(x/y)
        P_ = np.arccos(-z)
        
        n = x*np.sin(Y_)
        d = x*np.cos(Y_)*np.sin(P_) + y*np.sin(P_)+x*np.cos(P_)
        R_ = np.arctan(n/d)
        
        #if np.isnan(Y_): 
        #    Y.append(Y[-1])
        #else: 
        #    Y.append(Y_)
            
        #if np.isnan(P_):
        #    P.append(P[-1])
        #else: 
        #    P.append(P_)
            
        #if np.isnan(R_):
        #    R.append(R[-1])
        #else: 
        #    R.append(R_)
    
        return R_, P_, Y_


    def normalize(v): 
    
        norml = []
        for v_ in v: 
            normv = v_/(np.linalg.norm(v_))
        
            if np.isnan(normv).any():
                norml.append(norml[-1])
            else: 
                norml.append(normv)
        
        return np.array(norml)    


    def get_arm_velocities(self):

        # Velocity calculation -> not sure that it's correct
        try:
            meas_t = rospy.Time.now().to_sec()
            self.v_base_lwrist = (self.p_base_lwrist - self.prev_p_base_lwrist) * (meas_t - self.last_pass_t)
            self.v_base_rwrist = (self.p_base_rwrist - self.prev_p_base_rwrist) * (meas_t - self.last_pass_t)
            self.prev_p_base_lwrist = copy.deepcopy(self.p_base_lwrist)
            self.prev_p_base_rwrist = copy.deepcopy(self.p_base_rwrist)
            self.last_pass_t = meas_t
        except Exception as e: 
            self.prev_p_base_lwrist = copy.deepcopy(self.p_base_lwrist)
            self.prev_p_base_rwrist = copy.deepcopy(self.p_base_rwrist)
            self.v_base_lwrist = Vector3(0, 0, 0)
            self.v_base_rwrist = Vector3(0, 0, 0)
            self.last_pass_t = meas_t
            rospy.logwarn("Calculating EE velocity: {}".format(str(e)))


    def filter_avg(self, measurement, window_size, var_name):

        try:
            if np.isnan(measurement):
                self.m_dict["{}".format(var_name)].append(self.m_dict["{}".format(var_name)][-1])
            else: 
                self.m_dict["{}".format(var_name)].append(measurement)
        except: 
            rospy.logwarn("Not enough measurements to calculate average.")

        #rospy.logdebug("{}: {}".format(var_name, self.m_dict["{}".format(var_name)]))
        
        if len(self.m_dict["{}".format(var_name)]) < window_size: 
            return measurement
        else:
            prev = self.m_dict["{}".format(var_name)][-1] 
            
            self.m_dict["{}".format(var_name)] = self.m_dict["{}".format(var_name)][-window_size:]
            avg  = sum(self.m_dict["{}".format(var_name)])/len(self.m_dict["{}".format(var_name)])

            return avg 

    def filter_median(self, measurement, window_size, var_name):

        try:
            if np.isnan(measurement):
                self.m_dict["{}".format(var_name)].append(self.m_dict["{}".format(var_name)][-1])
            else: 
                self.m_dict["{}".format(var_name)].append(measurement)
        except:
            rospy.logwarn("Not enough measurements to calculate median.")
        
        if (len(self.m_dict["{}".format(var_name)]) < window_size):
            return measurement
        else: 
            prev = self.m_dict["{}".format(var_name)][-1]
            self.m_dict["{}".format(var_name)] = self.m_dict["{}".format(var_name)][-window_size:]
            median = np.median(np.sort(self.m_dict["{}".format(var_name)]))


            return median
    
    def filter_lowpass(self, prev_meas, meas, coeff):

        s = 1 / (1 + coeff)
        f = 1 - coeff 

        filtered_meas = s * (prev_meas + meas - f * prev_meas)

        return filtered_meas

    def filter_arm(self, pitch, roll, yaw, elbow, arm, filter_type, first=False): 

        if filter_type == "avg":
            window_size = 7
            # Overusage of copy.deepcopy (bad usage of the class and variable definitions -> HACKING!)
            pitch_  = self.filter_avg(pitch, window_size, "{}_pitch".format(arm))
            roll_   = self.filter_avg(roll, window_size, "{}_roll".format(arm))
            yaw_    = self.filter_avg(yaw, window_size, "{}_yaw".format(arm))
            elbow_  = self.filter_avg(elbow, window_size, "{}_elbow".format(arm))

        if filter_type == "lowpass":
                coeff = 0.8
                # Init first vars
                if first:
                    self.prev["{}_pitch".format(arm)]   = pitch; 
                    self.prev["{}_roll".format(arm)]    = roll
                    self.prev["{}_yaw".format(arm)]     = yaw
                    self.prev["{}_elbow".format(arm)]  = elbow
                
                pitch_  = self.filter_lowpass(self.prev["{}_pitch".format(arm)], pitch, 0.8);   self.prev["{}_pitch"] = pitch_
                roll_   = self.filter_lowpass(self.prev["{}_roll".format(arm)], roll, 0.8);     self.prev["{}_roll"] = roll_
                yaw_    = self.filter_lowpass(self.prev["{}_yaw".format(arm)], yaw, 0.8);       self.prev["{}_yaw".format(arm)] = yaw_
                elbow_  = self.filter_lowpass(self.prev["{}_elbow".format(arm)], elbow, 0.8);   self.prev["{}_elbow".format(arm)] = elbow_ 


        return pitch_, roll_, yaw_, elbow_

    def filter_cartesian(self, x, y, z, arm, filter_type):

        if filter_type == "avg": 
            window_size = 7
            x = self.filter_avg(x, window_size, "{}_x".format(arm))
            y = self.filter_avg(y, window_size, "{}_y".format(arm))
            z = self.filter_avg(z, window_size, "{}_z".format(arm))

        if filter_type == "median":
            window_size = 5
            x = self.filter_median(x, window_size, "{}_x".format(arm))
            y = self.filter_median(y, window_size, "{}_y".format(arm))
            z = self.filter_median(z, window_size, "{}_z".format(arm))

        return np.array([x, y, z])

    def get_angle(self, p, plane="xy", rAxis = "x", format="radians"): 

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


    def get_vect_angle(self, v1, v2, format="radians"): 

        d_ = np.dot(v1, v2)
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        ang = np.arccos(d_/(n1 * n2))
        if format=="degrees": 
            ang = np.degrees(ang)

        return ang


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
        
        first_filt = True
        while not rospy.is_shutdown(): 

            run_ready = self.hpe3d_recv 

            if run_ready: 

                # Maybe save indices for easier debugging
                start_time = rospy.Time.now().to_sec()
                # Get angles arm joints
                lpitch, lroll, lyaw, lelbow = self.get_arm_angles(copy.deepcopy(self.p_shoulder_lelbow),
                                                                  copy.deepcopy(self.p_elbow_lwrist), "left")
                                    
                rpitch, rroll, ryaw, relbow = self.get_arm_angles(copy.deepcopy(self.p_shoulder_relbow), 
                                                                  copy.deepcopy(self.p_elbow_rwrist), "right")

                # Send transforms
                self.send_arm_transforms()
                self.publish_norms()

                #if self.norm_p_s_lwrist > 1.5 or self.norm_p_s_rwrist > 1.5: 
                #    continue 

                # Filtering!
                filtering = "nothing"; 
                if filtering == "avg":                    
                    lpitch, lroll, lyaw, lelbow = self.filter_arm(lpitch, lroll, lyaw, lelbow, "left", "avg")
                    rpitch, rroll, ryaw, relbow = self.filter_arm(rpitch, rroll, ryaw, relbow, "right", "avg")
                    p_shoulder_lwrist = self.filter_cartesian(self.p_shoulder_lwrist[0], self.p_shoulder_lwrist[1], self.p_shoulder_lwrist[2], "left", "median")
                    p_shoulder_rwrist = self.filter_cartesian(self.p_shoulder_rwrist[0], self.p_shoulder_rwrist[1], self.p_shoulder_rwrist[2], "right", "median")

                if filtering == "lowpass": 
                    lpitch, lroll, lyaw, lelbow = self.filter_arm(lpitch, lroll, lyaw, lelbow, "left", "lowpass", first_filt)
                    rpitch, rroll, ryaw, relbow = self.filter_arm(rpitch, rroll, ryaw, relbow, "right", "lowpass", first_filt)
                    first_filt = False             

                else: 
                    p_shoulder_lwrist = self.p_shoulder_lwrist
                    p_shoulder_rwrist = self.p_shoulder_rwrist               

                # get EE velocity --> Decouple
                self.get_arm_velocities()


                # publish vals for following
                self.publish_arm(lpitch, lroll, lyaw, lelbow, p_shoulder_lwrist, self.p_shoulder_lwrist, "left")
                self.publish_arm(rpitch, rroll, ryaw, relbow, p_shoulder_rwrist, self.p_shoulder_rwrist, "right")

                self.publish_robot_arm(lpitch, lroll, lyaw, lelbow)
                
                measure_runtime = False; 
                if measure_runtime:
                    duration = rospy.Time.now().to_sec() - start_time
                    rospy.logdebug("Run t: {}".format(duration)) # --> very fast!

            else: 
                self.debug_print()
                
            self.rate.sleep()


if __name__ == "__main__": 

    hpe2armcmd_ = hpe2armcmd(sys.argv[1])
    try:
        hpe2armcmd_.run()
    except Exception as e: 
        rospy.logerr("Failed run hpe2armcmd: {}".format(e))