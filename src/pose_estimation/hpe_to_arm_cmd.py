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
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Vector3

import sensor_msgs.point_cloud2 as pc2



# TODO:
# - Camera transformation https://www.cs.toronto.edu/~jepson/csc420/notes/imageProjection.pdf
# - Read camera_info 
# - add painting of a z measurements  

class HumanPose3D(): 

    def __init__(self, freq):

        rospy.init_node("hpe3d", log_level=rospy.DEBUG)

        self.rate = rospy.Rate(int(float(freq)))

        self.hpe3d_recv           = False

        # Initialize publishers and subscribers
        self._init_subscribers()
        self._init_publishers()

        # Obtain 4 DoF angles 
        self.unit_x = np.array([1, 0, 0])
        self.unit_y = np.array([0, 1, 0])
        self.unit_z = np.array([0, 0, 1])


        self.camera_frame_name = "camera_color_frame"
        # Initialize transform broadcaster                  
        self.tf_br = tf.TransformListener()

        rospy.loginfo("[Hpe3D] started!")

    def _init_subscribers(self):

        self.hpe_3d_sub         = rospy.Subscriber("camera/color/image_raw", Image, self.hpe3d_cb, queue_size=1)

    def _init_publishers(self): 

        # Publish array of messages
        # self.q_pos_cmd_pub = rospy.Publisher("")
        #
        pass

    def hpe3d_cb(self, msg):

        # Msg has header, use it for timestamping 
        self.hpe3d_recv = True
        self.hpe3d_recv_t = msg.header.stamp # -> to system time 
        # Extract data
        self.p_base_neck = self.createPvect(msg.BaseNeck)         # could be camera_link_frame
        self.p_neck_shoulder = self.createPvect(msg.NeckShoulder)
        # p_base_shoulder  - p_base_neck 
        self.p_shoulder_elbow = self.createPvect(msg.ShoulderElbow)
        # p_base_elbow - p_base_shoulder 
        self.p_elbow_wrist = self.createPvect(msg.ElbowWrist)
        # p_base_wrist - p_base_elbow

    def get_arm_angles(self): 


        # 3 Shoulder angles 
        self.roll_angle = self.get_angle(self.p_shoulder_elbow, 'xz') # anterior axis shoulder (xz)
        self.pitch_angle = self.get_angle(self.p_shoulder_elbow, 'yz') # mediolateral axis (yz) 
        self.yaw_angle = self.get_angle(self.p_shoulder_elbow, 'xy') # longitudinal axis (xy)
        self.elbow_angle = self.get_angle(self.elbow_wrist, 'yz')

    def get_angle(vectI, vectBase, plane_="xy"): 

        # θ = cos-1 [ (a · b) / (|a| |b|) ]
        # Orthogonal projection of the vectore to the wanted plane
        vectPlane = self.getOrthogonalVect(vectI, plane_)
        # Angle between orthogonal projection of the vector and the 

        if plane=="xy":
            vectBase = self.unit_x
        if plane == "xz": 
            vectBase = self.unit_x
        if plane == "yz": 
            vectBase = self.unit_y

        # All angles have 270° offset -> radians      
        angle = np.arcos(np.dot(vectPlane, vectBase), np.linalg.norm(vectPlane) * np.linalg.norm(vectBase))

        # remove offset = 
        if remove_offset:
            angle = angle - 270

        return angle


    def createPvect(self, msg): 
        # Create position vector
        return np.array([msg.data.x, msg.data.y, msg.data.z])

    def getOrthogonalVect(self, vect, plane="xy"): 

        if plane=="xy": 
            vect_ = np.array([vect[0], vect[1], 0])
        
        if plane=="xz": 
            vect_ = np.array([vect[0], 0, vect[2]])

        if plane=="yz"
            vect_ = np.array([0, vect[1], vect[2]])
        
        return vect_


    def debug_print(self): 

        if not self.hpe3d_recv:
            rospy.logwarn_throttle(1, "Human pose estimation from camera stream has not been recieved.")





    def run(self): 

        while not rospy.is_shutdown(): 
            
            run_ready = self.hpe3d_recv 

            if run_ready: 
                
                # Maybe save indices for easier debugging
                start_time = rospy.Time.now().to_sec()
                
                # TODO: 
                # - convert hpe 3d to joint commands 
                # Add time flag 
                self.get_arm_angles()
                
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

    hpe3D = HumanPose3D(sys.argv[1])
    hpe3D.run()