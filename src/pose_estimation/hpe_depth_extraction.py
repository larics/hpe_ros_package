#!/opt/conda/bin/python3

from tracemalloc import start
from venv import create
import numpy
import rospy
import cv2

import os
import sys

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

        rospy.init_node("hpe3d")

        print(freq)
        self.rate = rospy.Rate(int(float(freq)))

        self.pcl_recv           = False
        self.img_recv           = False
        self.pred_recv          = False
        self.camera_info_recv   = False 

        # Initialize publishers and subscribers
        self._init_subscribers()
        self._init_publishers()
        
        # Initialize transform broadcaster
        #self.br = TransformBroadcaster(self)

        rospy.loginfo("[Hpe3D] started!")

    def _init_subscribers(self):

        self.camera_sub         = rospy.Subscriber("camera/rgb/image_raw",      Image,              self.image_cb,  queue_size=1)
        self.depth_sub          = rospy.Subscriber("camera/depth/points",       PointCloud2,        self.pcl_cb,    queue_size=1)
        self.depth_cinfo_sub    = rospy.Subscriber("camera/depth/camera_info",  CameraInfo,         self.cinfo_cb,  queue_size=1)
        self.predictions_sub    = rospy.Subscriber("hpe_preds",                 Float64MultiArray,  self.pred_cb,   queue_size=1)

    def _init_publishers(self): 
        pass

    def image_cb(self, msg): 

        self.img_recv   = True
        
        self.img = numpy.frombuffer(msg.data, dtype=numpy.uint8).reshape(msg.height, msg.width, -1)

    def pcl_cb(self, msg):

        self.pcl        = msg
        self.pcl_recv   = True


    def pred_cb(self, msg): 

        keypoints = msg.data
        # pair elements
        self.predictions = [(int(keypoints[i]), int(keypoints[i + 1])) for i in range(0, len(keypoints), 2)]
        self.pred_recv = True

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

    def create_tf_list(self, coords): 

        cond = "x" in coords.keys() and "y" in coords.keys() and "z" in coords.keys()        
        assert(cond), "Not enough coordinates returned to create TF"

        kp_tf_list = []
        for x, y, z in zip(coords["x"], coords["y"], coords["z"]): 
            nan_cond =  not numpy.isnan(x) and not numpy.isnan(y) and numpy.isnan(z)
            kp_tf_list.append((x, y, z))

        return kp_tf_list

    def plot_depths(self, keypoints, depths): 

        #pil_img = PILImage.fromarray(self.img.astype('uint8'), 'RGB')
        #draw = ImageDraw.Draw(pil_img)
        #draw.ellipse([(predictions[i][0] - point_r, predictions[i][1] - point_r), (predictions[i][0] + point_r, predictions[i][1] + point_r)], fill=fill_, width=2*point_r)
        
        pass

    def run(self): 

        while not rospy.is_shutdown(): 
            
            run_ready = self.img_recv and self.cinfo_recv and self.pcl_recv and self.pred_recv

            if run_ready: 
                
                # Maybe save indices for easier debugging
                #start_time = rospy.Time.now().to_sec()
                coords = self.get_coordinates(self.pcl, self.predictions, "xyz")
                #duration = rospy.Time.now().to_sec() - start_time
                tf_list = self.create_tf_list.append(coords)



                #self.plot_depths(self.predictions, d)

            self.rate.sleep()




if __name__ == "__main__": 

    hpe3D = HumanPose3D(sys.argv[1])
    hpe3D.run()