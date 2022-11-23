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

import sensor_msgs.point_cloud2 as pc2


# TODO:
# - Camera transformation https://www.cs.toronto.edu/~jepson/csc420/notes/imageProjection.pdf
# - Read camera_info 
# - add painting of a z measurements  

class HumanPose3D(): 

    def __init__(self, freq):

        rospy.init_node("hpe3d", log_level=rospy.DEBUG)

        self.rate = rospy.Rate(int(float(freq)))

        self.pcl_recv           = False
        self.img_recv           = False
        self.pred_recv          = False
        self.cinfo_recv         = False 

        # Initialize publishers and subscribers
        self._init_subscribers()
        self._init_publishers()

        # MPII keypoint indexing
        self.indexing = {0:"r_ankle", 1:"r_knee", 2:"r_hip", 3:"l_hip", 4: "l_knee", 5: "l_ankle",
                         6:"pelvis", 7:"thorax", 8:"upper_neck", 9:"head_top", 10:"r_wrist",
                         11:"r_elbow", 12:"r_shoulder", 13:"l_shoulder", 14:"l_elbow", 15:"l_wrist"}


        self.camera_frame_name = "camera_color_frame"
        # Initialize transform broadcaster                  
        self.tf_br = tf.TransformBroadcaster()

        rospy.loginfo("[Hpe3D] started!")

    def _init_subscribers(self):

        self.camera_sub         = rospy.Subscriber("camera/color/image_raw", Image, self.image_cb, queue_size=1)
        self.depth_sub          = rospy.Subscriber("camera/depth_registered/points", PointCloud2, self.pcl_cb, queue_size=1)
        self.depth_cinfo_sub    = rospy.Subscriber("camera/depth/camera_info", CameraInfo, self.cinfo_cb, queue_size=1)
        self.predictions_sub    = rospy.Subscriber("hpe_preds", Float64MultiArray, self.pred_cb, queue_size=1)

    def _init_publishers(self): 
        pass

    def image_cb(self, msg): 

        self.img_recv   = True
        
        self.img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)

    def pcl_cb(self, msg):

        self.pcl        = msg
        self.pcl_recv   = True

    def pred_cb(self, msg): 

        keypoints = msg.data
        # pair elements
        self.predictions = [(int(keypoints[i]), int(keypoints[i + 1])) for i in range(0, len(keypoints), 2)]
        self.pred_recv = True

        # Cut predictions on upper body only
        self.predictions = self.predictions[6:]

    def cinfo_cb(self, msg): 

        self.cinfo_recv = True

    def get_depths(self, pcl, indices, axis="z"):

        # Get current depths from depth cam
        depths = pc2.read_points(pcl, [axis], False, uvs=indices)

        return depths
    
    def get_coordinates(self, pcl, keypoints, axis): 

        ret = {}
        for i in axis: 
            # Get distances for each axis with get_depths method
            generator_depths = self.get_depths(pcl, keypoints, str(i))
            # Create dict with axis as key and list of values that represent coord of each keypoint
            ret["{}".format(i)] = [val for val in generator_depths]
        return ret

    def create_keypoint_tfs(self, coords): 

        cond = "x" in coords.keys() and "y" in coords.keys() and "z" in coords.keys()        
        assert(cond), "Not enough coordinates returned to create TF"

        kp_tf = {}
        for i, (x, y, z) in enumerate(zip(coords["x"], coords["y"], coords["z"])): 
            nan_cond =  not np.isnan(x) and not np.isnan(y) and not np.isnan(z)

            if nan_cond: 
                # Swapped z, y, x to hit correct dimension, and y is in wrong direction
                p = np.array([z[0], y[0], x[0]])
                # Needs to be rotated for 90 deg around X axis
                R = get_RotX(-np.pi/2) 
                rotP = np.matmul(R, p)
                print(rotP)
                
                kp_tf["{}".format(i)] = (rotP[0], -rotP[1], rotP[2])  

        return kp_tf

    def send_transforms(self, tfs):

        for index, tf in tfs.items():
            
            index_offset = 6    # Index offset is used to use only upper parts of the body for demonstration purposes
            x,y,z = tf[0], tf[1], tf[2]
            self.tf_br.sendTransform((x, y, z),
                                     (0, 0, 0, 1), # Hardcoded orientation for now
                                     rospy.Time.now(), 
                                     self.indexing[int(index) + index_offset], 
                                     self.camera_frame_name    # Should be camera but there's no transform from world to camera for now
            )   

    def plot_depths(self, keypoints, depths): 

        #pil_img = PILImage.fromarray(self.img.astype('uint8'), 'RGB')
        #draw = ImageDraw.Draw(pil_img)
        #draw.ellipse([(predictions[i][0] - point_r, predictions[i][1] - point_r), (predictions[i][0] + point_r, predictions[i][1] + point_r)], fill=fill_, width=2*point_r)
        
        pass

    def debug_print(self): 

        if not self.img_recv:
            rospy.logwarn_throttle(1, "Image is not recieved! Check camera and topic name.")
        if not self.pcl_recv: 
            rospy.logwarn_throttle(1, "Depth information is not recieved! Check camera type and topic name.")
        if not self.cinfo_recv: 
            rospy.logwarn_throttle(1, "Camera info is not recieved! Check camera and topic name.")
        if not self.pred_recv: 
            rospy.logwarn_throttle(1, "Prediction is not recieved! Check topic names, camera type and model initialization!")


    def run(self): 

        while not rospy.is_shutdown(): 
            
            run_ready = self.img_recv and self.cinfo_recv and self.pcl_recv and self.pred_recv

            if run_ready: 
                
                # Maybe save indices for easier debugging
                start_time = rospy.Time.now().to_sec()
                # Get X,Y,Z coordinates for predictions
                coords = self.get_coordinates(self.pcl, self.predictions, "xyz") # rospy.logdebug("coords: {}".format(coords))
                # Create coordinate frames
                tfs = self.create_keypoint_tfs(coords)
                # Send transforms
                self.send_transforms(tfs)

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