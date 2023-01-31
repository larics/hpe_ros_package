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
from hpe_ros_package.msg import TorsoJointPositions

import sensor_msgs.point_cloud2 as pc2

openpose = True
if openpose: 
    from ros_openpose.msg import Frame



# TODO:
# - Camera transformation https://www.cs.toronto.edu/~jepson/csc420/notes/imageProjection.pdf
# - add painting of a z measurements  
# - Record bag of l shoulder, r shoulder and rest of the body parts 
# - 

class HumanPose3D(): 

    def __init__(self, freq):
        
        # TODO: Add LOG_LEVEL as argument
        rospy.init_node("hpe3d", log_level=rospy.INFO)

        self.rate = rospy.Rate(int(float(freq)))

        self.pcl_recv           = False
        self.img_recv           = False
        self.pred_recv          = False
        self.cinfo_recv         = False 

        # IF openpose: True, ELSE: False
        self.openpose = True   

        if self.openpose: 
            self.body25 = True
            self.coco = False

        # Initialize publishers and subscribers
        self._init_subscribers()
        self._init_publishers()

        # MPII keypoint indexing
        self.mpii_indexing = {0:"r_ankle", 1:"r_knee", 2:"r_hip", 3:"l_hip", 4: "l_knee", 5: "l_ankle",
                              6:"pelvis", 7:"thorax", 8:"upper_neck", 9:"head_top", 10:"r_wrist",
                              11:"r_elbow", 12:"r_shoulder", 13:"l_shoulder", 14:"l_elbow", 15:"l_wrist"}

        self.coco_indexing = {0: "nose", 1:"l_eye", 2:"r_eye", 3:"l_ear", 4:"r_ear", 5:"l_shoulder", 
                              6:"r_shoulder", 7:"l_elbow", 8:"r_elbow", 9:"l_wrist", 10:"r_wrist", 
                              11:"l_hip", 12:"r_hip", 13:"l_knee", 14:"r_knee", 15:"l_ankle", 16:"r_ankle"}

        self.body25_indexing = {0 :"nose", 1 :"neck", 2:  "r_shoulder", 3:  "r_elbow", 
                                4: "r_wrist", 5: "l_shoulder", 6 : "l_elbow", 7:"l_wrist", 8: "midhip", 9: "r_hip",
                                10: "r_knee", 11:"r_ankle", 12:"l_hip", 13: "l_knee", 14: "l_ankle", 15: "r_eye", 16: "l_eye", 
                                17: "r_ear", 18:"l_ear", 19:"l_big_toe", 20:"l_small_toe", 21: "l_heel", 22: "r_big_toe",
                                23: "r_small_toe", 24: "r_heel", 25: "background"}
        
        self.mpii = False

        # self.indexing = different indexing depending on weights that are used!
        if self.mpii: self.indexing = self.mpii_indexing
        if self.coco: self.indexing = self.coco_indexing   
        if self.body25: self.indexing = self.body25_indexing      

        self.camera_frame_name = "camera_color_frame"
        # Initialize transform broadcaster                  
        self.tf_br = tf.TransformBroadcaster()

        self.torso_pos_3d = TorsoJointPositions()
        rospy.loginfo("[Hpe3D] started!")

    def _init_subscribers(self):

        self.camera_sub         = rospy.Subscriber("camera/color/image_raw", Image, self.image_cb, queue_size=1)
        self.depth_sub          = rospy.Subscriber("camera/depth_registered/points", PointCloud2, self.pcl_cb, queue_size=1)
        self.depth_cinfo_sub    = rospy.Subscriber("camera/depth/camera_info", CameraInfo, self.cinfo_cb, queue_size=1)
       

        if self.openpose: 
            self.predictions_sub    = rospy.Subscriber("/frame", Frame, self.pred_cb, queue_size=1)
        else: 
            self.predictions_sub    = rospy.Subscriber("hpe_preds", Float64MultiArray, self.pred_cb, queue_size=1)
        
        rospy.loginfo("Initialized subscribers!")




    def _init_publishers(self): 
        self.left_wrist_pub     = rospy.Publisher("leftw_point", Vector3, queue_size=1)
        self.right_wrist_pub    = rospy.Publisher("rightw_point", Vector3, queue_size=1)
        self.upper_body_3d_pub  = rospy.Publisher("upper_body_3d", TorsoJointPositions, queue_size=1)

        rospy.loginfo("Initialized publishers!")

    def image_cb(self, msg): 

        self.img        = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        self.img_recv   = True


    def pcl_cb(self, msg):

        self.pcl        = msg
        self.pcl_recv   = True

    def pred_cb(self, msg): 

        #keypoints = msg.data
        persons = msg.persons
        self.predictions = []
        self.pose_predictions = []

        if self.openpose:
            for i, person in enumerate(persons): 
                if i == 0: 
                    for bodypart in person.bodyParts: 
                        self.predictions.append((int(bodypart.pixel.x), int(bodypart.pixel.y)))
                        self.pose_predictions.append((bodypart.point.x, bodypart.point.y, bodypart.point.z))
            
            self.predictions = self.predictions[:18]
            self.pred_recv = True
        else: 
            # pair elements
            self.predictions = [(int(keypoints[i]), int(keypoints[i + 1])) for i in range(0, len(keypoints), 2)]
            self.pred_recv = True
        # Cut predictions on upper body only 6+ --> don't cut predictions, performance speedup is not noticable
        # self.predictions = self.predictions[6:]

    def cinfo_cb(self, msg): 

        self.cinfo_recv = True
        # Color --> Params for homography
        # ------
        # K: [911.5259399414062, 0.0, 642.8853759765625, 0.0, 909.3442993164062, 357.6820068359375, 0.0, 0.0, 1.0]
        # Depth
        # ------
        # P: [886.5087280273438, 0.0, 645.0914306640625, 0.0, 0.0, 886.5087280273438, 358.6357116699219, 0.0, 0.0, 0.0, 1.0, 0.0]

    def get_depths(self, pcl, indices, axis="z"):

        # Get current depths from depth cam --> TODO: Change read_points with cam_homography
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
        pos_named = {}
        for i, (x, y, z) in enumerate(zip(coords["x"], coords["y"], coords["z"])): 
            nan_cond =  not np.isnan(x) and not np.isnan(y) and not np.isnan(z)
            if nan_cond: 
                # Swapped z, y, x to hit correct dimension
                p = np.array([z[0], y[0], x[0]])
                # Needs to be rotated for 90 deg around X axis
                R = get_RotX(-np.pi/2) 
                rotP = np.matmul(R, p)
                # Y is in wrong direction therefore -rotP
                kp_tf["{}".format(i)] = (rotP[0], -rotP[1], rotP[2])  
                pos_named["{}".format(self.indexing[i])] = (rotP[0], -rotP[1], rotP[2])

        return kp_tf, pos_named

    def send_transforms(self, tfs):

        for index, tf in tfs.items():
            
            x,y,z = tf[0], tf[1], tf[2]
            self.tf_br.sendTransform((x, y, z),
                                     (0, 0, 0, 1), # Hardcoded orientation for now
                                     rospy.Time.now(), 
                                     self.indexing[int(index)], 
                                     self.camera_frame_name)    # Should be camera but there's no transform from world to camera for now

            # Each of this tf-s is basically distance from camera_frame_name to some other coordinate frame :) 
            # use lookupTransform to fetch transform and estimate angles... 
            

    def plot_depths(self, keypoints, depths): 

        #pil_img = PILImage.fromarray(self.img.astype('uint8'), 'RGB')
        #draw = ImageDraw.Draw(pil_img)
        #draw.ellipse([(predictions[i][0] - point_r, predictions[i][1] - point_r), (predictions[i][0] + point_r, predictions[i][1] + point_r)], fill=fill_, width=2*point_r)
        
        pass

    def record_movement(self): 

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


    def create_ROSmsg(self, pos_named): 


        msg = TorsoJointPositions()
        msg.header          = self.pcl.header
        msg.frame_id.data        = "camera_color_frame"
        try:
            # COCO doesn't have THORAX!
            if self.coco or self.body25: 
                thorax = Vector3((pos_named["l_shoulder"][0] + pos_named["r_shoulder"][0])/2, 
                                 (pos_named["l_shoulder"][1] + pos_named["r_shoulder"][1])/2, 
                                 (pos_named["l_shoulder"][2] + pos_named["r_shoulder"][2])/2)
                msg.thorax = thorax
            else: 
                msg.thorax      = Vector3(pos_named["thorax"][0], pos_named["thorax"][1], pos_named["thorax"][2])
            msg.left_elbow      = Vector3(pos_named["l_elbow"][0], pos_named["l_elbow"][1], pos_named["l_elbow"][2])
            msg.right_elbow     = Vector3(pos_named["r_elbow"][0], pos_named["r_elbow"][1], pos_named["r_elbow"][2])
            msg.left_shoulder   = Vector3(pos_named["l_shoulder"][0], pos_named["l_shoulder"][1], pos_named["l_shoulder"][2])
            msg.right_shoulder  = Vector3(pos_named["r_shoulder"][0], pos_named["r_shoulder"][1], pos_named["r_shoulder"][2])
            msg.left_wrist      = Vector3(pos_named["l_wrist"][0], pos_named["l_wrist"][1], pos_named["l_wrist"][2])
            msg.right_wrist     = Vector3(pos_named["r_wrist"][0], pos_named["r_wrist"][1], pos_named["r_wrist"][2])
            msg.success.data = True
            rospy.logdebug("Created ROS msg!")
        except Exception as e:
            msg.success.data = False 
            rospy.logwarn_throttle(2, "Create ROS msg failed: {}".format(e))

        return msg
    

    def run(self): 

        while not rospy.is_shutdown(): 
            
            run_ready = self.img_recv and self.cinfo_recv and self.pcl_recv and self.pred_recv

            if run_ready: 
                rospy.loginfo_throttle(10, "Publishing HPE3d!")
                # Maybe save indices for easier debugging
                start_time = rospy.Time.now().to_sec()
                # Get X,Y,Z coordinates for predictions
                coords = self.get_coordinates(self.pcl, self.predictions, "xyz") # rospy.logdebug("coords: {}".format(coords))
                # Create coordinate frames
                tfs, pos_named = self.create_keypoint_tfs(coords)
                # Send transforms
                self.send_transforms(tfs)
                # Publish created ROS msg 
                msg = self.create_ROSmsg(pos_named)
                if msg: 
                    self.upper_body_3d_pub.publish(msg)

                measure_runtime = False; 
                if measure_runtime:
                    duration = rospy.Time.now().to_sec() - start_time
                    rospy.logdebug("Run t: {}".format(duration)) # --> very fast!

            else: 

                self.debug_print()
                
            self.rate.sleep()


# Create Rotation matrices
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