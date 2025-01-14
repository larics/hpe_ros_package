#!/usr/bin/python3

import os
import sys

import copy
import rospy
import tf
import numpy as np

from PIL import ImageDraw, ImageFont
from PIL import Image as PILImage
from img_utils import convert_pil_to_ros_img, convert_ros_to_pil_img, plot_hand_keypoints
from utils import *

from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Vector3
from hpe_ros_msgs.msg import TorsoJointPositions, HumanPose2D, HandPose2D, HumanPose3D, HandPose3D
from ros_openpose_msgs.msg import Frame
from visualization_msgs.msg import MarkerArray, Marker

import message_filters

import sensor_msgs.point_cloud2 as pc2

# TODO:
# - Camera transformation https://www.cs.toronto.edu/~jepson/csc420/notes/imageProjection.pdf
# - add painting of a z measurements  
# - Record bag of l shoulder, r shoulder and rest of the body parts 
# - Compare results 

class HPE2Dto3D(): 

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

        self.init_x_rot = 0 #-90 - 40 # - 30 due to pitch
        self.init_y_rot = 0
        self.init_z_rot = 0 #

        # IF openpose: True, ELSE: False
        self.openpose = openpose
        
        # If use hands, parse them from frame 
        # use hands could be part of the tmuxinator config? 
        self.use_hands = True
        self.r_hand_predictions, self.l_hand_predictions = [], []

        if self.openpose: 
            self.body25 = True
            self.coco = False
            self.mpii = False
        else: 
            self.coco = True
            self.body25 = False
            self.mpii = False

        # Initialize publishers and subscribers
        self._init_subscribers()
        self._init_publishers()

        # TODO: Add TRT indexing
        # MPII keypoint indexing
        # TODO: Move this indexing to the yaml file 
        # TODO: Add indexing for the hands
        self.mpii_indexing = {0:"r_ankle", 1:"r_knee", 2:"r_hip", 3:"l_hip", 4: "l_knee", 5: "l_ankle",
                              6:"pelvis", 7:"thorax", 8:"upper_neck", 9:"head_top", 10:"r_wrist",
                              11:"r_elbow", 12:"r_shoulder", 13:"l_shoulder", 14:"l_elbow", 15:"l_wrist"}

        self.coco_indexing = {0: "nose", 1:"l_eye", 2:"r_eye", 3:"l_ear", 4:"r_ear", 5:"l_shoulder", 
                              6:"r_shoulder", 7:"l_elbow", 8:"r_elbow", 9:"l_wrist", 10:"r_wrist", 
                              11:"l_hip", 12:"r_hip", 13:"l_knee", 14:"r_knee", 15:"l_ankle", 16:"r_ankle"}

        # TODO: Maybe plot stuff to see how this is indexed
        self.body25_indexing = {0 :"nose", 1 :"neck", 2:  "r_shoulder", 3:  "r_elbow", 
                                4: "r_wrist", 5: "l_shoulder", 6 : "l_elbow", 7:"l_wrist", 8: "midhip", 9: "r_hip",
                                10: "r_knee", 11:"r_ankle", 12:"l_hip", 13: "l_knee", 14: "l_ankle", 15: "r_eye", 16: "l_eye", 
                                17: "r_ear", 18:"l_ear", 19:"l_big_toe", 20:"l_small_toe", 21: "l_heel", 22: "r_big_toe",
                                23: "r_small_toe", 24: "r_heel", 25: "background"}
        
        self.hand_indexing = {0: "wrist", 1: "thumb0", 2: "thumb1", 3: "thumb2", 4: "thumb3",
                              5: "index0", 6: "index1", 7: "index2", 8: "index3",
                              9: "middle0", 10: "middle1", 11: "middle2", 12: "middle3",
                              13: "ring0", 14: "ring1", 15: "ring2", 16: "ring3",
                              17: "pinky0", 18: "pinky1", 19: "pinky2", 20: "pinky3"}
        
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

        self.camera_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_cb, queue_size=1)
        self.depth_sub = rospy.Subscriber("/camera/depth_registered/points", PointCloud2, self.pcl_cb, queue_size=1)
        self.depth_cinfo_sub    = rospy.Subscriber("/camera/depth/camera_info", CameraInfo, self.cinfo_cb, queue_size=1)
       
        if self.openpose: 
            #self.predictions_sub    = rospy.Subscriber("/frame", Frame, self.pred_cb, queue_size=1)
            self.predictions_sub    = message_filters.Subscriber("/frame", Frame)
            #self.predictions_sub    = message_filters.Subscriber("/hpe_2d", Frame)
            self.depth_sub          = message_filters.Subscriber("/camera/depth/color/points", PointCloud2)
            # Doesn't matter! 
            self.ats                = message_filters.TimeSynchronizer([self.predictions_sub, self.depth_sub], 10)
            self.ats.registerCallback(self.frame_pcl_cb)

        else: 
            self.predictions_sub    = rospy.Subscriber("hpe_preds", Float64MultiArray, self.pred_cb, queue_size=1)
        
        rospy.loginfo("Initialized subscribers!")

    def _init_publishers(self): 
        self.debug_plot         = rospy.Publisher("debug_plot", Image, queue_size=1)
        self.left_wrist_pub     = rospy.Publisher("leftw_point", Vector3, queue_size=1)
        self.right_wrist_pub    = rospy.Publisher("rightw_point", Vector3, queue_size=1)
        self.upper_body_3d_pub  = rospy.Publisher("upper_body_3d", TorsoJointPositions, queue_size=1)
        self.hpe3d_pub          = rospy.Publisher("hpe3d", HumanPose3D, queue_size=1)
        self.rhand_pub         = rospy.Publisher("rhand3d", HandPose3D, queue_size=1)
        self.lhand_pub          = rospy.Publisher("lhand3d", HandPose3D, queue_size=1)
        self.camera_est_pub = rospy.Publisher("camera_estimation", MarkerArray, queue_size=1)
        rospy.loginfo("Initialized publishers!")

    def frame_pcl_cb(self, frame_msg, pcl_msg): 
        # Used if OpenPose Sync is used 
        rospy.logdebug("Received frame and pcl!")
        persons = frame_msg.persons
        self.predictions = []
        self.pose_predictions = []

        # TODO: Add HumanPose2D and HumanHand2D to publish if necessary
        if self.openpose:
            for i, person in enumerate(persons): 
                if i == 0: 
                    for bodypart in person.bodyParts: 
                        self.predictions.append((int(bodypart.pixel.x), int(bodypart.pixel.y)))
                        self.pose_predictions.append((bodypart.point.x, bodypart.point.y, bodypart.point.z))
            self.pred_recv = True
        try:
            if self.openpose and self.use_hands: 
                for i, person in enumerate(persons): 
                    self.r_hand_predictions = []
                    self.l_hand_predictions = []
                    # TODO: Do it just for the first person for now
                    if i == 0: 
                        # Here is where problem arises, because sometimes openpose doesn't detect all fingers
                        # TODO: How are defined leftHandParts 
                        for rkp in person.leftHandParts: 
                            self.r_hand_predictions.append((int(rkp.pixel.x), int(rkp.pixel.y)))
                        for lkp in person.rightHandParts: 
                            self.l_hand_predictions.append((int(lkp.pixel.x), int(lkp.pixel.y)))
                            # TODO: Maybe plot ordering? :) 
                        # rospy.loginfo("Right hand predictions: {}" .format(self.r_hand_predictions))
                        # rospy.loginfo("Left hand predictions: {}".format(self.l_hand_predictions))
                
        except Exception as e: 
            rospy.logwarn("No hands were detected!")
        self.pcl        = pcl_msg
        self.pcl_recv   = True    

    def image_cb(self, msg): 
        
        self.ros_img = msg
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
            
            #self.predictions = self.predictions[:18]
            self.pred_recv = True
        else: 
            # pair elements
            self.predictions = [(int(keypoints[i]), int(keypoints[i + 1])) for i in range(0, len(keypoints), 2)]
            self.pred_recv = True
        # Cut predictions on upper body only 6+ --> don't cut predictions, performance speedup is not noticable
        # self.predictions = self.predictions[6:]

    def cinfo_cb(self, msg): 

        self.cinfo_recv = True

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
    
    # It is for human body for now only and for static TFs, which is wrong :) 
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
            else: 
                kp_tf["{}".format(i)] = (0, 0, 0)
                pos_named["{}".format(indexing[i])] = (0, 0, 0)

        return kp_tf, pos_named

    def getP(self, p, angle_x_axis, angle_y_axis, angle_z_axis, order, format="radians"):
        #TODO: This is disgusting! Change into ordinary matrix multiplication
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
        # TODO: move this to markers 
        for index, tf in tfs.items():
            
            x,y,z = tf[0], tf[1], tf[2]
            self.tf_br.sendTransform((x, y, z),
                                     (0, 0, 0, 1), # Hardcoded orientation for now
                                     rospy.Time.now(), 
                                     self.indexing[int(index)], 
                                     self.camera_frame_name)    # Should be camera but there's no transform from world to camera for now

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

    def packTorso3DMsg(self, pos_named): 
        # TODO: Pack this into packTorsoPoseMsg method
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
            rospy.logdebug("Created Torso ROS msg!")

        except Exception as e:
            msg.success.data = False 
            rospy.logwarn_throttle(2, "Create ROS msg failed: {}".format(e))

        return msg
    
    def get_hpe3d(self, predictions, publish_tfs=False): 
        # Here we extract x_i, y_i from the PCL values (measured in m)
        coords = self.get_coordinates(self.pcl, predictions, "xyz") 
        # It would make sense to add maybe confidence or something like that :)
        tfs, pos_named = self.create_keypoint_tfs(coords, self.body25_indexing)
        if publish_tfs:
            self.send_transforms(tfs)
        hpe3d_msg = packOPHumanPose3DMsg(rospy.Time.now(), pos_named)
        return hpe3d_msg
    
    def get_upper_body3d(self, hpe_preds): 
        coords = self.get_coordinates(self.pcl, hpe_preds, "xyz") 
        tfs, pos_named = self.create_keypoint_tfs(coords, self.body25_indexing)
        self.send_transforms(tfs)
        torso3d_msg = self.packTorso3DMsg(pos_named)
        return torso3d_msg
    
    def get_hand3d(self, hand_preds): 
        coords = self.get_coordinates(self.pcl, hand_preds, "xyz") 
        tfs, pos_named = self.create_keypoint_tfs(coords, self.hand_indexing)
        self.send_transforms(tfs)
        hand3d_msg = packHandPose3DMsg(rospy.Time.now(), pos_named)
        return hand3d_msg
    
    def proc_hand_pose_est(self):
        if self.use_hands:
            try:
                lhand3d_msg = self.get_hand3d(copy.deepcopy(self.l_hand_predictions))
                self.lhand_pub.publish(lhand3d_msg)
            except Exception as e:
                rospy.logwarn("Failed to generate or publish left hand message: {}".format(e))
            try: 
                rhand3d_msg = self.get_hand3d(copy.deepcopy(self.r_hand_predictions))
                self.rhand_pub.publish(rhand3d_msg)
            except Exception as e:
                rospy.logwarn("Failed to generate or publish right hand message: {}".format(e))

    def proc_hpe_est(self):
        try:
            # TODO: Add comparison of the openpose estimation and the 3D pose estimation 
            hpe3d_msg = self.get_hpe3d(copy.deepcopy(self.predictions))
            self.hpe3d_pub.publish(hpe3d_msg)

            # TODO: Get torso coordinate frame 
            c_d_ls = pointToArray(hpe3d_msg.l_shoulder)
            c_d_rs = pointToArray(hpe3d_msg.r_shoulder)
            c_d_t  = pointToArray(hpe3d_msg.neck)
            c_d_n  = pointToArray(hpe3d_msg.nose)
            c_d_le = pointToArray(hpe3d_msg.l_elbow)
            c_d_re = pointToArray(hpe3d_msg.r_elbow)
            c_d_rw = pointToArray(hpe3d_msg.r_wrist)
            c_d_lw = pointToArray(hpe3d_msg.l_wrist)

            cD = np.array([create_homogenous_vector(c_d_ls), 
                           create_homogenous_vector(c_d_rs), 
                           create_homogenous_vector(c_d_le), 
                           create_homogenous_vector(c_d_re), 
                           create_homogenous_vector(c_d_lw), 
                           create_homogenous_vector(c_d_rw)])


            # body in the camera coordinate frame 
            bRc = np.matmul(get_RotX(np.pi/2), get_RotY(np.pi/2))
            # thorax in the camera frame --> TODO: Fix transformations
            T = create_homogenous_matrix(bRc.T, -c_d_t)

            bD = np.matmul(T, cD.T).T
            self.publishMarkerArray(bD)             

        except Exception as e:
            rospy.logwarn("Failed to generate or publish HPE3d message: {}".format(e))

    def publishMarkerArray(self, bD):
        mA = MarkerArray()
        i = 0
        for v in bD:
            m_ = self.createMarker(v, i)
            i+=1 
            mA.markers.append(m_)
        print("len markers", len(mA.markers))
        self.camera_est_pub.publish(mA)

    def createMarker(self, v, i):
        m_ = Marker()
        m_.header.frame_id = "camera_color_frame"
        m_.header.stamp = rospy.Time.now()
        m_.type = m_.SPHERE
        m_.id = i
        m_.action = m_.ADD
        m_.scale.x = 0.1
        m_.scale.y = 0.1
        m_.scale.z = 0.1
        m_.color.a = 1.0
        m_.color.r = 0.0
        m_.color.g = 1.0
        m_.color.b = 0.0
        m_.pose.position.x = v[0]
        m_.pose.position.y = v[1]
        m_.pose.position.z = v[2]
        m_.pose.orientation.x = 0
        m_.pose.orientation.y = 0
        m_.pose.orientation.z = 0
        m_.pose.orientation.w = 1
        return m_

    def run(self): 
        cnt = 1; t_total = 0.0
        while not rospy.is_shutdown(): 
            
            run_ready = self.img_recv and self.cinfo_recv and self.pcl_recv and self.pred_recv
            if run_ready: 
                rospy.loginfo_throttle(30, "Publishing 3D pose of the human!")
                # Maybe save indices for easier debugging
                t_s = rospy.Time.now().to_sec()
                self.use_hpe = True
                if self.use_hpe:
                    self.proc_hpe_est()

                self.use_hands = False
                # TODO: Move to separate method
                if self.use_hands:
                    self.process_hand_estimations()
                
                # TODO: Move to the separate method
                debug_plot = False
                if debug_plot:
                    pil_img = convert_ros_to_pil_img(self.ros_img)
                    img = plot_hand_keypoints(pil_img, self.r_hand_predictions)
                    img = plot_hand_keypoints(pil_img, self.l_hand_predictions)
                    ros_img = convert_pil_to_ros_img(img)
                    self.debug_plot.publish(ros_img)
                    convert_pil_to_ros_img()

                measure_runtime = True; 
                # TODO: Should be moved to utils.py
                if measure_runtime:
                    cnt += 1
                    t_delta = rospy.Time.now().to_sec() - t_s
                    t_total += t_delta
                    t_avg = t_total/cnt
                
                rospy.loginfo_throttle(1, f"Average loop duration is {t_avg}")

                # Removed for test publishing of the 3D pose of the upper body
                # TODO: This used in Kalman filter for the upper body
                # upper_body_msg = self.get_upper_body3d(copy.deepcopy(self.predictions))
                # self.upper_body_3d_pub.publish(upper_body_msg)
                
            else: 
                self.debug_print()

            self.rate.sleep()


def create_homogenous_vector(v): 
    return np.array([v[0], v[1], v[2], 1])

# Create Rotation 
def create_homogenous_matrix(R, t):
    T = np.hstack((R, t.reshape(3, 1)))
    T = np.vstack((T, np.array([0, 0, 0, 1])))
    return T

def get_RotX(angle): # 
    
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

def pointToArray(msg): 
    return np.array([msg.x, msg.y, msg.z])


if __name__ == "__main__": 

    hpe3D = HPE2Dto3D(sys.argv[1], sys.argv[2])
    hpe3D.run()



"""
    def image_pcl_cb(self, img_msg, pcl_msg): 
        
        rospy.loginfo("Received image and pcl!")
        self.img        = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, -1)
        self.img_recv   = True

        self.pcl        = pcl_msg
        self.pcl_recv   = True

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

"""