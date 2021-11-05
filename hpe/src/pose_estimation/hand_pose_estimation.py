#!/opt/conda/bin/python3

import rospy
import numpy as np
import cv2

import argparse
import os
import sys
import pprint

from PIL import ImageDraw, ImageFont
from PIL import Image as PILImage

from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image
from std_msgs.msg import Bool

import mediapipe as mp
# This fails due to the AVX instructions tf has 
try:
    import tensorflow as tf
    print("Successfuly imported tensorflow!")
    print("Used python path is: {}".format(sys.path))
except Exception as e:
    print(str(e))
#from tensorflow.keras.models import load_model

class HandPoseEstimation():

    def __init__(self, frequency):
        
        self.initialized = False

        rospy.init_node("hand_pose_estimation", log_level=rospy.DEBUG)

        rospy.loginfo("Started hand pose estimation!")
                
        
        self.rate = rospy.Rate(int(frequency))

        
        # Initialize subscribers/publishers
        self._init_publishers()
        self._init_subscribers()

        # Initialize hand model 
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(model_complexity=0, 
                                        max_num_hands=1,
                                        min_detection_confidence=0.5,
                                        static_image_mode=True)
        self.mpDraw = mp.solutions.drawing_utils

        # Initialize computer vision bridge
        #self.cv_bridge = CvBridge()


        """
        
        self.nn_input = None
        
        # Initialize font
        self.font = ImageFont.truetype("/home/developer/catkin_ws/src/hpe_ros_package/hpe/include/arial.ttf", 20, encoding="unic")

        self.model = load_model('mp_hand_gesture')
        self.classNames = ['okay', 'peace', 'thumbs up', 'thumbs down', 'call me', 'stop', 'rock', 'live long', 'fist', 'smile']


        """

        # flags for starting 
        self.first_img_reciv = False
        self.detected_hand = False
        self.initialized = True

    def _init_subscribers(self):
        self.camera_sub = rospy.Subscriber("usb_camera/image_raw", Image, self.image_cb, queue_size=1)

     def _init_publishers(self):
        self.detected_points_pub = rospy.Publisher("/hand_keypoints", Image, queue_size=1)
        #self.dummy_pub = rospy.Publisher("/dummy_pub", Bool, queue_size=1)
        #self.image_pub = rospy.Publisher("/stickman", Image, queue_size=1)
        #self.pred_pub = rospy.Publisher("/hpe_preds", Float64MultiArray, queue_size=1)
        pass

    def image_cb(self, msg):

        start_time = rospy.Time.now().to_sec()

        self.first_img_reciv = True

        #try: 
        # cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")

        print(msg.height)
        print(msg.width)

        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.width, msg.height, -1)

        #except CvBridgeError as e:
        #    rospy.logerror(str(e))

        debug_img = False
        if debug_img:
            rospy.loginfo("Image width: {}".format(msg.width))
            rospy.loginfo("Image height: {}".format(msg.height))
            rospy.loginfo("Data is: {}".format(len(msg.data)))
            rospy.loginfo("Input shape is: {}".format(input.shape))

        cv_img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        print(cv_img_rgb)

        #cv_img_rgb = HandPoseEstimation.bgr2rgb(img)

        self.detected_hand = self.hands.process(cv_img_rgb)
        
        duration = rospy.Time.now().to_sec() - start_time 
        #rospy.loginfo("Duration of image_cb is: {}".format(duration)) # max --> 0.01s
                         
    def run(self):

        self.first = True
        
        while not rospy.is_shutdown(): 

            if (self.first_img_reciv and self.detected_hand and self.initialized):
               
                start_time = rospy.Time.now().to_sec()
                
                dir(self.detected_hand)

                if self.detected_hand.multi_hand_landmarks:
                    rospy.logdebug(hand_landmarks)

                    
                    for hand_landmarks in self.detected_hand.multi_hand_landmarks:
                        self.mpDraw.draw_landmarks(
                                                    image,
                                                    hand_landmarks,
                                                    mp_hands.HAND_CONNECTIONS,
                                                    mp_drawing_styles.get_default_hand_landmarks_style(),
                                                    mp_drawing_styles.get_default_hand_connections_style())

                            
            self.rate.sleep()

    @staticmethod
    def bgr2rgb(img_array):

        rgb = np.zeros(img_array.shape, dtype=np.float32)
        rgb[:, :, 0] = img_array[:, :, 2]/255.0
        rgb[:, :, 1] = img_array[:, :, 1]/255.0
        rgb[:, :, 2] = img_array[:, :, 0]/255.0  

        print(rgb)

        return rgb

    @staticmethod
    def draw_hand(img, predictions):
        pass     

    @staticmethod
    def convert_pil_to_ros_img(img):
        """Function for converting pillow to ros image
        Args:
            img (PIL.Image.Image): Pillow image that represents GUI
        Returns:
            sensor_msgs.msg._Image.Image: ROS image for image publisher
        """
        img = img.convert('RGB')
        msg = Image()
        stamp = rospy.Time.now()
        msg.height = img.height
        msg.width = img.width
        msg.encoding = "rgb8"
        msg.is_bigendian = False
        msg.step = 3 * img.width
        msg.data = numpy.array(img).tobytes()
        return msg   

def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers
    if args.use_detect_bbox:
        config.TEST.USE_GT_BBOX = not args.use_detect_bbox
    if args.shift_heatmap:
        config.TEST.SHIFT_HEATMAP = args.shift_heatmap
    if args.model_file:
        config.TEST.MODEL_FILE = args.model_file
    if args.coco_bbox_file:
        config.TEST.COCO_BBOX_FILE = args.coco_bbox_file

if __name__ == '__main__':

    print("==============Starting hand pose estimation!===========")
    handPoseROS = HandPoseEstimation(sys.argv[1])
    handPoseROS.run()
