import rospy
import cv2

import os
import sys

from PIL import ImageDraw, ImageFont
from PIL import Image as PILImage

from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from std_msgs.msg import Float64MultiArray

import sensor_msgs.point_cloud2 as pc2


# TODO:
# - Add depth subscriber 
# - Read camera_info 
# - add tf2 broadcaster 
# - add painting of a z measurements  

class HumanPose3D(): 

    def __init__(self):

        self.pcl_recv           = False
        self.img_recv           = False
        self.pred_recv          = False
        self.camera_info_recv   = False 

        # Initialize publishers and subscribers
        self._init_subscribers()
        self._init_publishers()

    def _init_subscribers(self):

        self.camera_sub         = rospy.Subscriber("camera/rgb/image_raw",      Image,              self.image_cb,  queue_size=1)
        self.depth_sub          = rospy.Subscriber("camera/depth/points",       PointCloud2,        self.pcl_cb,    queue_size=1)
        self.depth_cinfo_sub    = rospy.Subscriber("camera/depth/camera_info",  CameraInfo,         self.cinfo_cb,  queue_size=1)
        self.predictions_sub    = rospy.Subscriber("hpe_preds",                 Float64MultiArray,  self.pred_cb,   queue_size=1)

    def _init_publishers(self): 
        pass

    def image_cb(self, msg): 

        self.img_recv   = True

    def depth_cb(self, msg):

        self.pcl_recv   = True
        self.pcl        = msg

    def pred_cb(self, msg): 

        self.pred_recv = True

    def get_depths(self, pcl, hpe_keypoints):

        # Get current depths from depth cam
        depths = pc2.read_points(self.depth_pcl_msg, ['z'], False, uvs=indices)

        return depths
    
    def run(self): 

        while not rospy.is_shutdown(): 
            
            run_ready = self.img_recv and self.camera_info_recv and self.pcl_recv and self.pred_recv

            if run_ready: 

                d = self.get_depths()




if __name__ == "__main__": 

    hpe3D = HumanPose3D(freq)
    hpe3D.run()