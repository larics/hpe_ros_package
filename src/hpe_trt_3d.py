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
from geometry_msgs.msg import Vector3, PoseStamped
from hpe_ros_msgs.msg import TorsoJointPositions, HumanPose2D, HandPose2D, HumanPose3D

import message_filters

import sensor_msgs.point_cloud2 as pc2

# Import all methods from utils
from utils import *
from linalg_utils import get_RotX, get_RotY, get_RotZ   

from input_remapping import createOmatrix, createUmatrix, tfU2Vect3, tfU2Pose

# TODO:
# - Camera transformation https://www.cs.toronto.edu/~jepson/csc420/notes/imageProjection.pdf
# - add painting of a z measurements  
# - Record bag of l shoulder, r shoulder and rest of the body parts 

class HHPose3D(): 

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

        # Sync depth and prediction
        self.sync = True

        # Initialize publishers and subscribers
        self._init_subscribers()
        self._init_publishers()

        # TODO: move to some kind of json file and load with script :) 
        self.hpe_indexing = {0: "nose", 1: "l_eye", 2: "r_eye", 3: "l_ear", 4: "r_ear",
                              5: "l_shoulder", 6: "r_shoulder", 7: "l_elbow", 8: "r_elbow",
                              9: "l_wrist", 10: "r_wrist", 11: "l_hip", 12: "r_hip",
                              13: "l_knee", 14: "r_knee", 15: "l_ankle", 16: "r_ankle", 17: "background"}

        self.hand_indexing = {0: "wrist", 1: "thumb3", 2: "thumb2", 3: "thumb1", 4: "thumb0",
                              5: "index3", 6: "index2", 7: "index1", 8: "index0", 9: "middle3",
                             10: "middle2", 11: "middle1", 12: "middle0", 13: "ring3", 14: "ring2",
                             15: "ring1", 16: "ring0", 17: "pinky3", 18: "pinky2", 19: "pinky1", 20: "pinky0"}
        
        # Hand tree should be [How to include implicit knowledge about the hand in the prediction?]
        # That's one of the questions that could arise, and how to differentiate between left and right hand? 
        # 0 -> wrist -> 0 thumb  -> 1 thumb  -> 2 thumb 
        # 0 -> wrist -> 0 index  -> 1 index  -> 2 index 
        # 0 -> wrist -> 0 middle -> 1 middle -> 2 middle
        # 0 -> wrist -> 0 ring   -> 1 ring   -> 2 ring
        # 0 -> wrist -> 0 pinky  -> 1 pinky  -> 2 pinky

        # TODO: Check hand working 
        self.HPE = True; self.HAND = False
        self.camera_frame_name = "camera_color_frame"
        # Initialize transform broadcaster 
        # ATM just a stupid way to publish joint estimates              
        self.tf_br = tf.TransformBroadcaster()
        self.resize_predictions = True
        self.torso_pos_3d = TorsoJointPositions()
        rospy.loginfo("[Hpe3D] started!")

    def _init_subscribers(self):
        # Luxonis camera is used for now [with rs_compat values] # Can be removed if there is rs_compat setup
        self.camera_sub         = rospy.Subscriber("/oak/rgb/image_raw", Image, self.image_cb, queue_size=1)
        self.depth_sub          = rospy.Subscriber("/oak/points", PointCloud2, self.pcl_cb, queue_size=1)
        self.depth_cinfo_sub    = rospy.Subscriber("/oak/stereo/camera_info", CameraInfo, self.cinfo_cb, queue_size=1)
        # Subscription to predictions 
        self.hpe_2d_sub         = rospy.Subscriber("/hpe_2d", HumanPose2D, self.hpe2d_cb, queue_size=1)
        self.hand_2d_sub        = rospy.Subscriber("/hand_2d", HandPose2D, self.hand2d_cb, queue_size=1)
        # Values with rs_compat = true
        self.camera_sub         = rospy.Subscriber("/camera/color/image_raw", Image, self.image_cb, queue_size=1)
        self.depth_sub          = rospy.Subscriber("/camera/depth/color/points", PointCloud2, self.pcl_cb, queue_size=1)
        self.depth_cinfo_sub    = rospy.Subscriber("/camera/depth/camera_info", CameraInfo, self.cinfo_cb, queue_size=1)
        rospy.loginfo("Initialized subscribers!")

        if self.sync: 
            # It would make sense even to add HandPose2D to the sync_cb
            self.hpe_2d_sub = message_filters.Subscriber("/hpe_2d", HumanPose2D)
            self.depth_sub = message_filters.Subscriber("/camera/depth/color/points", PointCloud2)
            self.ts = message_filters.TimeSynchronizer([self.hpe_2d_sub, self.depth_sub], 5)
            self.ts.registerCallback(self.sync_cb)

    def _init_publishers(self):
        self.hpe_3d_pub  = rospy.Publisher("/hpe_3d", HumanPose3D, queue_size=1)
        self.torso3d_pub = rospy.Publisher("/torso_3d", TorsoJointPositions, queue_size=1)

        # Debug topics
        self.vect1_pub = rospy.Publisher("vect1", Vector3, queue_size=1)
        self.vect2_pub = rospy.Publisher("vect2", Vector3, queue_size=1)
        self.vect3_pub = rospy.Publisher("vect3", Vector3, queue_size=1)
        self.vect4_pub = rospy.Publisher("vect4", Vector3, queue_size=1)
        rospy.loginfo("Initialized publishers!")
        self.pose1_pub = rospy.Publisher("pose1", PoseStamped, queue_size=1)
        self.pose2_pub = rospy.Publisher("pose2", PoseStamped, queue_size=1)

    def image_cb(self, msg): 
        self.img        = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        self.img_recv   = True

    def pcl_cb(self, msg):
        self.pcl        = msg
        self.pcl_recv   = True

    def hpe2d_cb(self, msg): 
        hpe_pxs = unpackHumanPose2DMsg(msg)
        # r_prefix is resized!
        self.r_hpe_preds = resize_preds_on_original_size(hpe_pxs, (self.dpth_img_width, self.dpth_img_height))
        self.pred_recv = True

    def sync_cb(self, hpe_msg, pcl_msg):
        hpe_pxs = unpackHumanPose2DMsg(hpe_msg)
        self.r_hpe_preds = resize_preds_on_original_size(hpe_pxs, (self.dpth_img_width, self.dpth_img_height))
        self.pred_recv = True
        self.pcl = pcl_msg
        self.pcl_recv = True
        rospy.loginfo_throttle(1, "Synced callback!")

    def hand2d_cb(self, msg):
        hand_pxs = unpackHandPose2DMsg(msg)
        self.r_hand_preds = resize_preds_on_original_size(hand_pxs, (self.dpth_img_width, self.dpth_img_height))

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

    def create_keypoint_tfs(self, coords, indexing): 

        cond = "x" in coords.keys() and "y" in coords.keys() and "z" in coords.keys()        
        assert(cond), "Not enough coordinates returned to create TF"

        kp_tf = {}
        pos_named = {}
        for i, (x, y, z) in enumerate(zip(coords["x"], coords["y"], coords["z"])): 
            nan_cond =  not np.isnan(x) and not np.isnan(y) and not np.isnan(z)
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
                pos_named["{}".format(indexing[i])] = p
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

    def send_transforms(self, tfs, indexing):
        for index, tf in tfs.items():
            # At this point we just publish points not TFs (really, it is just a form of visualization)
            x,y,z = tf[0], tf[1], tf[2]
            self.tf_br.sendTransform((x, y, z),
                                     (0, 0, 0, 1), # Hardcoded orientation for now 
                                     rospy.Time.now(), 
                                     indexing[int(index)], 
                                     self.camera_frame_name)    
            # Should be camera but there's no transform from world to camera for now
            # Each of this tf-s is basically distance from camera_frame_name to some other coordinate frame :) 
            # use lookupTransform to fetch transform and estimate angles... 

    def debug_print(self): 
        if not self.img_recv:
            rospy.logwarn_throttle(1, "Image is not recieved! Check camera and topic name.")
        if not self.pcl_recv: 
            rospy.logwarn_throttle(1, "Depth information is not recieved! Check camera type and topic name.")
        if not self.cinfo_recv: 
            rospy.logwarn_throttle(1, "Camera info is not recieved! Check camera and topic name.")
        if not self.pred_recv: 
            rospy.logwarn_throttle(1, "Prediction is not recieved! Check topic names, camera type and model initialization!")

    def get_and_pub_keypoints(self, keypoints, indexing):
        # Get X,Y,Z coordinates for predictions
        coords = self.get_coordinates(self.pcl, keypoints, "xyz") 
        # Create coordinate frames
        hpe_tfs, hpe_pos_named = self.create_keypoint_tfs(coords, indexing)
        # Send transforms with tf broadcaster
        self.send_transforms(hpe_tfs, indexing)
        return coords
    
    # Debug methods
    def publish_vectors(self, vects_):
        self.vect1_pub.publish(vects_[0])
        self.vect2_pub.publish(vects_[1])
    
    def publish_poses(self, poses_):
        self.pose1_pub.publish(poses_[0])
        self.pose2_pub.publish(poses_[1])
    
    def remapping(self, p3D, indexing, ap_names, mp_names):
        """
            Remap the input to the desired output
            Input: 
                P3D: 3D points of the detections
                ap_names: Names of the anchor points
                mp_names: Names of the mapping points
            Output: 
                U: Remapped input
        """
        ap = [get_key_by_value(indexing, ap_name) for ap_name in ap_names]
        mp = [get_key_by_value(indexing, mp_name) for mp_name in mp_names]
        n_u = len(ap) # no of anchor points defines number of inputs
        n_k = p3D.shape[1] # no of motion points defines number of keypoitns [column in p3D]
        O_ = createOmatrix(n_k, n_u, ap, mp)
        U = createUmatrix(p3D.squeeze(), O_)
        return U

    def run(self): 
        while not rospy.is_shutdown(): 
            # Condition to run the code
            run_ready = self.img_recv and self.cinfo_recv and self.pcl_recv and self.pred_recv
            try:
                # TODO: Decouple hand and the rest of the body in some way
                if run_ready: 
                    rospy.loginfo_throttle(30, "Publishing HPE3d!")
                    start_time = rospy.Time.now().to_sec()
                    if self.HPE: 
                        pts = self.get_and_pub_keypoints(self.r_hpe_preds, self.hpe_indexing)
                        # These are measurements that could be given to the Kalman for example
                        P3D = dict_to_matrix(pts)
                        # Nans fu*k up the matrix multiplication
                        P3D = remove_nans(P3D) 
                        hpe3dMsg = packHumanPose3DMsg(rospy.Time.now(), P3D.squeeze())
                        self.hpe_3d_pub.publish(hpe3dMsg)

                        torso3dMsg = packTorsoPositionMsg(rospy.Time.now(), P3D.squeeze())
                        #print(torso3dMsg)
                        self.torso3d_pub.publish(torso3dMsg)

                        # This should publish relation of the r_shoulder with r_wrist and l_shoulder with l_wrist 
                        # This goes to the remapping part of the code [explain system how it works] 
                        # This is just basic remapping, there should be better options/solutions to do this
                        u = self.remapping(P3D, self.hpe_indexing,
                                          ["r_shoulder", "l_shoulder"],
                                          ["r_wrist", "l_wrist"])
                        vects_ = tfU2Vect3(u); self.publish_vectors(vects_)
                        poses_ = tfU2Pose(u); self.publish_poses(poses_)

                    if self.HAND: 
                        pts = self.get_and_pub_keypoints(self.r_hand_preds, self.hand_indexing)
                        H3D = dict_to_matrix(pts)
                    measure_runtime = False; 
                    if measure_runtime:
                        duration = rospy.Time.now().to_sec() - start_time
                        rospy.logdebug("Run t: {}".format(duration)) # --> very fast!
                    self.rate.sleep()
                else: 
                    self.debug_print()
            
            except Exception as e: 
                rospy.logwarn("Run failed: {}".format(e))


if __name__ == "__main__": 

    hpe3D = HHPose3D(sys.argv[1], sys.argv[2])
    hpe3D.run()