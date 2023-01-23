#!/usr/bin/python2

import os
import sys

import rospy
import tf
import numpy as np

from PIL import ImageDraw, ImageFont
from PIL import Image as PILImage
from img_utils import convert_pil_to_ros_img

from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from std_msgs.msg import Float64MultiArray, Float32
from geometry_msgs.msg import Vector3
from hpe_ros_package.msg import TorsoJointPositions

import sensor_msgs.point_cloud2 as pc2

# TODO:
# - Camera transformation https://www.cs.toronto.edu/~jepson/csc420/notes/imageProjection.pdf
# - Read camera_info 
# - add painting of a z measurements  

class hpe2cmd(): 

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
        self.shoulder_roll_angle_pub = rospy.Publisher("shoulder_roll", Float32, queue_size=1)
        self.shoulder_pitch_angle_pub = rospy.Publisher("shoulder_pitch", Float32, queue_size=1)
        self.shoulder_yaw_angle_pub = rospy.Publisher("shoulder_yaw", Float32, queue_size=1)
        self.elbow_angle_pub = rospy.Publisher("elbow", Float32, queue_size=1)

    def hpe3d_cb(self, msg):

        # Msg has header, use it for timestamping 
        self.hpe3d_recv_t = msg.header.stamp # -> to system time 
        # Extract data --> Fix pBaseNeck
        self.p_base_thorax  = self.createPvect(msg.thorax)         # could be camera_link_frame - thorax
        self.p_base_lshoulder = self.createPvect(msg.left_shoulder) 
        self.p_base_lelbow = self.createPvect(msg.left_elbow)
        self.p_base_lwrist = self.createPvect(msg.left_wrist)
        self.p_base_rshoulder = self.createPvect(msg.right_shoulder)
        self.p_base_relbow = self.createPvect(msg.right_elbow)
        self.p_base_rwrist = self.createPvect(msg.right_wrist)
        # Broadcast this vects as new TFs
        # p_base_shoulder - p_base_neck
        # Left arm
        self.p_thorax_lshoulder = (self.p_base_lshoulder - self.p_base_thorax) * (-1)
        self.p_shoulder_lelbow = (self.p_base_lelbow - self.p_base_lshoulder) * (-1)
        self.p_elbow_lwrist = (self.p_base_lwrist - self.p_base_lelbow) * (-1)
        # Right arm 
        self.p_thorax_rshoulder = (self.p_base_rshoulder - self.p_base_thorax) * (-1)
        self.p_shoulder_relbow = (self.p_base_relbow - self.p_base_rshoulder) * (-1)
        self.p_elbow_rwrist = (self.p_base_rwrist - self.p_base_relbow) * (-1)
        # recieved HPE 3D
        self.hpe3d_recv = True

    def publish_arm_angles(self): 

        self.shoulder_roll_angle_pub.publish(self.roll_angle)
        self.shoulder_pitch_angle_pub.publish(self.pitch_angle)
        self.shoulder_yaw_angle_pub.publish(self.yaw_angle)
        self.elbow_angle_pub.publish(self.elbow_angle)

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

    def get_arm_angles(self): 

        # Control of left arm 
        # 3 Shoulder angles 
        self.roll_angle = self.get_angle(self.p_shoulder_elbow, 'xz')  # anterior axis shoulder (xz)
        self.pitch_angle = self.get_angle(self.p_shoulder_elbow, 'yz') # mediolateral axis (yz) 
        self.yaw_angle = self.get_angle(self.p_shoulder_elbow, 'xy')   # longitudinal axis (xy)
        self.elbow_angle = self.get_angle(self.p_elbow_wrist, 'yz')

        rospy.logdebug("Shoulder roll angle: {}".format(self.roll_angle))
        rospy.logdebug("Shoulder pitch angle: {}".format(self.pitch_angle))
        rospy.logdebug("Shoulder yaw angle: {}".format(self.yaw_angle))
        rospy.logdebug("Sholder elbow angle: {}".format(self.elbow_angle))

    def get_angle(self, vectI, plane_="xy", format="degrees"): 

        # theta = cos-1 [ (a * b) / (abs(a) abs(b)) ]
        # Orthogonal projection of the vector to the wanted plane
        vectPlane = self.getOrthogonalVect(vectI, plane_)
        # Angle between orthogonal projection of the vector and the 

        if plane_ =="xy":
            vectBase = self.unit_x
        if plane_ == "xz": 
            vectBase = self.unit_x
        if plane_ == "yz": 
            vectBase = self.unit_y

        # All angles have 270deg offset -> radians      
        # TODO: Check if this is norm or absolute value
        value = np.dot(vectPlane, vectBase)/(np.linalg.norm(vectPlane) * np.linalg.norm(vectBase))
        angle = np.arccos(value)

        remove_offset = False 
        if remove_offset:
            angle = angle - 270 * np.pi/180 

        if format == "degrees": 
            angle = np.degrees(angle)

        return angle

    def createPvect(self, msg): 
        # Create position vector from Vector3
        return np.array([msg.x, msg.y, msg.z])

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
                # self.get_arm_angles()
                self.send_arm_transforms()

                #self.publish_arm_angles()
                
                measure_runtime = False; 
                if measure_runtime:
                    duration = rospy.Time.now().to_sec() - start_time
                    rospy.logdebug("Run t: {}".format(duration)) # --> very fast!

            else: 
                self.debug_print()
                
            self.rate.sleep()


def get_RotX(angle): 
    
    RX = np.array([[1, 0, 0], 
                   [0, np.cos(angle), -np.sin(angle)], 
                   [0, np.sin(angle), np.cos(angle)]])
    
    return RX

def get_RotY(angle): 
    
    RY = np.array([[np.cos(angle), 0, np.sin(angle)], 
                   [0, 1, 0], 
                   [-np.sin(angle), 0, np.cos(angle)]])
    return RY
    
def get_RotZ(angle): 
    
    RZ = np.array([[np.cos(angle), -np.sin(angle), 0],
                   [np.sin(angle), np.cos(angle), 0], 
                   [ 0, 0, 1]] )
    
    return RZ



if __name__ == "__main__": 

    hpe2cmd_ = hpe2cmd(sys.argv[1])
    hpe2cmd_.run()