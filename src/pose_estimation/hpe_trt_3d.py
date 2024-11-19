#!/usr/bin/python3

import os
import sys

import rospy
import tf
import numpy as np

from PIL import ImageDraw, ImageFont
from PIL import Image as PILImage
from img_utils import convert_pil_to_ros_img

from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from std_msgs.msg import Int64MultiArray
from geometry_msgs.msg import Vector3
from hpe_ros_msgs.msg import TorsoJointPositions, HumanPose2D

import message_filters

import sensor_msgs.point_cloud2 as pc2

# TODO:
# - Camera transformation https://www.cs.toronto.edu/~jepson/csc420/notes/imageProjection.pdf
# - add painting of a z measurements  
# - Record bag of l shoulder, r shoulder and rest of the body parts 

class HumanPose3D(): 

    def __init__(self, freq, openpose):
        
        # TODO: Add LOG_LEVEL as argument
        rospy.init_node("hpe3d", log_level=rospy.INFO)

        self.rate = rospy.Rate(int(float(freq)))

        self.pcl_recv           = False
        self.img_recv           = False
        self.pred_recv          = False
        self.cinfo_recv         = False 

        # Camera CF for depth (LENSES - Z direction)
        #  A z
        #  |
        #  |
        #  X -----> x
        #  x goes right, y points down and z points from the camera viewpoint

        self.init_x_rot = -90 - 40 # - 30 due to pitch
        self.init_y_rot = 0
        self.init_z_rot = -90

        # IF openpose: True, ELSE: False
        self.openpose = openpose

        # Code pruning uneccessary indexing for TRT
        self.coco = True


        # Initialize publishers and subscribers
        self._init_subscribers()
        self._init_publishers()

        self.coco_indexing = {0: "nose", 1: "l_eye", 2: "r_eye", 3: "l_ear", 4: "r_ear",
                              5: "l_shoulder", 6: "r_shoulder", 7: "l_elbow", 8: "r_elbow",
                              9: "l_wrist", 10: "r_wrist", 11: "l_hip", 12: "r_hip",
                              13: "l_knee", 14: "r_knee", 15: "l_ankle", 16: "r_ankle", 17: "background"}

        
        self.indexing = self.coco_indexing

        self.camera_frame_name = "camera_color_frame"
        # Initialize transform broadcaster                  
        self.tf_br = tf.TransformBroadcaster()

        self.resize_predictions = True
        self.resized_predictions = []
        self.torso_pos_3d = TorsoJointPositions()
        rospy.loginfo("[Hpe3D] started!")

    def _init_subscribers(self):
        
        # Luxonis camera is used for now [without rs_compat values]
        self.camera_sub         = rospy.Subscriber("/oak/rgb/image_raw", Image, self.image_cb, queue_size=1)
        self.depth_sub          = rospy.Subscriber("/oak/points", PointCloud2, self.pcl_cb, queue_size=1)
        self.depth_cinfo_sub    = rospy.Subscriber("/oak/stereo/camera_info", CameraInfo, self.cinfo_cb, queue_size=1)
        self.predictions_sub    = rospy.Subscriber("/preds", Int64MultiArray, self.pred_cb, queue_size=1)
        self.hpe_2d_sub         = rospy.Subscriber("/hpe_2d", HumanPose2D, self.hpe2d_cb, queue_size=1)
        
        # Values with rs_compat = true
        self.camera_sub         = rospy.Subscriber("/camera/color/image_raw", Image, self.image_cb, queue_size=1)
        self.depth_sub          = rospy.Subscriber("/camera/depth/color/points", PointCloud2, self.pcl_cb, queue_size=1)
        self.depth_cinfo_sub    = rospy.Subscriber("/camera/depth/camera_info", CameraInfo, self.cinfo_cb, queue_size=1)


        rospy.loginfo("Initialized subscribers!")

    def _init_publishers(self): 
        self.left_wrist_pub     = rospy.Publisher("leftw_point", Vector3, queue_size=1)
        self.right_wrist_pub    = rospy.Publisher("rightw_point", Vector3, queue_size=1)
        self.upper_body_3d_pub  = rospy.Publisher("upper_body_3d", TorsoJointPositions, queue_size=1)
        rospy.loginfo("Initialized publishers!")


    def frame_pcl_cb(self, frame_msg, pcl_msg): 

        #keypoints = msg.data
        rospy.loginfo("Received frame and pcl!")
        persons = frame_msg.persons
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

        self.pcl        = pcl_msg
        self.pcl_recv   = True    


    def image_cb(self, msg): 

        self.img        = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        self.img_recv   = True

    def pcl_cb(self, msg):

        self.pcl        = msg
        self.pcl_recv   = True

    def hpe2d_cb(self, msg): 
        # TODO: Think how to make this shorter [Ugly, move it to the separate method in utils!]
        self.detections_pxs = []
        self.detections_pxs.append((msg.nose.x, msg.nose.y))
        self.detections_pxs.append((msg.l_eye.x, msg.l_eye.y))
        self.detections_pxs.append((msg.r_eye.x, msg.r_eye.y))
        self.detections_pxs.append((msg.l_ear.x, msg.l_ear.y))
        self.detections_pxs.append((msg.r_ear.x, msg.r_ear.y))
        self.detections_pxs.append((msg.l_shoulder.x, msg.l_shoulder.y))
        self.detections_pxs.append((msg.r_shoulder.x, msg.r_shoulder.y))
        self.detections_pxs.append((msg.l_elbow.x, msg.l_elbow.y))
        self.detections_pxs.append((msg.r_elbow.x, msg.r_elbow.y))
        self.detections_pxs.append((msg.l_wrist.x, msg.l_wrist.y))
        self.detections_pxs.append((msg.r_wrist.x, msg.r_wrist.y))
        self.detections_pxs.append((msg.l_hip.x, msg.l_hip.y))
        self.detections_pxs.append((msg.r_hip.x, msg.r_hip.y))
        self.detections_pxs.append((msg.l_knee.x, msg.l_knee.y))
        self.detections_pxs.append((msg.r_knee.x, msg.r_knee.y))
        self.detections_pxs.append((msg.l_ankle.x, msg.l_ankle.y))
        self.detections_pxs.append((msg.r_ankle.x, msg.r_ankle.y))
        self.resized_kp_preds = self.resize_preds_on_original_size(self.detections_pxs, (self.dpth_img_width, self.dpth_img_height))
    
    def pred_cb(self, msg): 

        keypoints = msg.data
        self.predictions = []
        self.resized_predictions = []
        self.pose_predictions = []
        # pair elements
        self.predictions = [(int(keypoints[i]), int(keypoints[i + 1])) for i in range(0, len(keypoints), 2)]

        if self.resize_predictions:
            self.resized_predictions = self.resize_preds_on_original_size(self.predictions, (self.dpth_img_width, self.dpth_img_height))

        self.pred_recv = True
        # Cut predictions on upper body only 6+ --> don't cut predictions, performance speedup is not noticable
        # self.predictions = self.predictions[6:]

    def cinfo_cb(self, msg): 

        self.cinfo_recv = True
        self.dpth_img_width = msg.width
        self.dpth_img_height = msg.height

    def get_depths(self, pcl, indices, axis="z"):

        # Get current depths from depth cam --> TODO: Change read_points with cam_homography
        # This works even with Luxonis! 
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
            rospy.loginfo("Nan condition: {}, i: {}, indexing: {}".format(nan_cond, i, self.indexing[i]))
            if nan_cond: 
                # OLD P 
                #p = np.array([z[0], y[0], x[0]]) # Swapped z, y, x to hit correct dimension
                #R = get_RotX(-np.pi/2)  # Needs to be rotated for 90 deg around X axis
                #rotP = np.matmul(R, p)
                #kp_tf["{}".format(i)] = (rotP[0], -rotP[1], rotP[2]) # Y is in wrong direction therefore -rotP
                #pos_named["{}".format(self.indexing[i])] = (rotP[0], -rotP[1], rotP[2])
                # NEW P CALC
                p = np.array([x[0], y[0], z[0]]) 

                x_rot = self.init_x_rot
                y_rot = self.init_y_rot
                z_rot = self.init_z_rot

                p = self.getP(p, x_rot, y_rot, z_rot, "xyz", "degrees") # TF from camera frame to the orientation human HAS!
                kp_tf["{}".format(i)] = p
                pos_named["{}".format(self.indexing[i])] = p
            
        rospy.loginfo(pos_named)

        return kp_tf, pos_named

    def getP(self, p, angle_x_axis, angle_y_axis, angle_z_axis, order, format="radians"):

        if format == "degrees": 
            angle_x_axis = np.radians(angle_x_axis)
            angle_y_axis = np.radians(angle_y_axis)
            angle_z_axis = np.radians(angle_z_axis)

        for i in order:
            if i == "x":
                p = np.matmul(get_RotX(angle_x_axis), p)
            elif i == "y":
                p = np.matmul(get_RotY(angle_y_axis), p)
            elif i == "z":
                p = np.matmul(get_RotZ(angle_z_axis), p)
        return (p[0], p[1], p[2])


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

    # Losing precision here
    def resize_preds_on_original_size(self, preds, img_size):
            resized_preds = []
            for pred in preds: 
                p_w, p_h = pred[0], pred[1]
                resized_preds.append([int(np.floor(p_w/224*img_size[0])),
                                    int(np.floor(p_h/224*img_size[1]))])
            return resized_preds

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
            try:
                if run_ready: 
                    rospy.loginfo_throttle(30, "Publishing HPE3d!")
                    # Maybe save indices for easier debugging
                    start_time = rospy.Time.now().to_sec()
                    # Get X,Y,Z coordinates for predictions
                    if self.resize_predictions:
                        coords = self.get_coordinates(self.pcl, self.resized_kp_preds, "xyz") 
                    else: 
                        coords = self.get_coordinates(self.pcl, self.predictions, "xyz")

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

                    self.rate.sleep()

                else: 
                    self.debug_print()
            
            except Exception as e: 
                rospy.logwarn("Run failed: {}".format(e))
                

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

    hpe3D = HumanPose3D(sys.argv[1], sys.argv[2])
    hpe3D.run()