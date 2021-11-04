#!/opt/conda/bin/python3

import rospy
import numpy
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
from cv_bridge import CvBridge, CvBridgeError

import mediapipe as mp
# This fails due to the AVX instructions tf has 
try:
    import tensorflow as tf
except Exception as e:
    print(str(e))

#from tensorflow.keras.models import load_model

class HandPoseEstimation():

    def __init__(self, frequency):
        
        print("Starting initialization!")

        rospy.init_node("hand_pose_estimation")

        rospy.loginfo("Started hand pose estimation!")
        
        
        self.rate = rospy.Rate(int(frequency))

        """
        # Initialize subscribers/publishers
        self._init_publishers()
        self._init_subscribers()

        self.nn_input = None
        
        # Initialize font
        self.font = ImageFont.truetype("/home/developer/catkin_ws/src/hpe_ros_package/hpe/include/arial.ttf", 20, encoding="unic")

        # Initialize hand model 
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.mpDraw = mp.solutions.drawing_utils

        self.model = load_model('mp_hand_gesture')
        self.classNames = ['okay', 'peace', 'thumbs up', 'thumbs down', 'call me', 'stop', 'rock', 'live long', 'fist', 'smile']

        self.cv_bridge = CvBridge()

        """

        # flags for starting 
        self.model_ready = True
        self.first_img_reciv = False


    def _init_subscribers(self):
        self.camera_sub = rospy.Subscriber("usb_camera/image_raw", Image, self.image_cb, queue_size=1)

 
    def _init_publishers(self):
        #self.dummy_pub = rospy.Publisher("/dummy_pub", Bool, queue_size=1)
        #self.image_pub = rospy.Publisher("/stickman", Image, queue_size=1)
        #self.pred_pub = rospy.Publisher("/hpe_preds", Float64MultiArray, queue_size=1)
        pass

    def _load_model(self, config):
        
        rospy.loginfo("Model name is: {}".format(config.MODEL.NAME))
        model = eval('models.' + config.MODEL.NAME + '.get_pose_net')(
        config, is_train=False)

        rospy.loginfo("Passed config is: {}".format(config))
        rospy.loginfo("config.TEST.MODEL.FILE")

        if config.TEST.MODEL_FILE:
            model_state_file = config.TEST.MODEL_FILE
            rospy.loginfo('=> loading model from {}'.format(config.TEST.MODEL_FILE))
            model.load_state_dict(torch.load(config.TEST.MODEL_FILE))
        else:
            model_state_file = os.path.join(final_output_dir,
                                        'final_state.pth.tar')
            rospy.loginfo('=> loading model from {}'.format(model_state_file))
            model.load_state_dict(torch.load(model_state_file))

        model.to(self.device)           
        
        return model

    def image_cb(self, msg):

        start_time = rospy.Time.now().to_sec()

        self.first_img_reciv = True

        try: 
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")

        except CvBridgeError as e:
            rospy.logerror(str(e))

        debug_img = False
        if debug_img:
            rospy.loginfo("Image width: {}".format(msg.width))
            rospy.loginfo("Image height: {}".format(msg.height))
            rospy.loginfo("Data is: {}".format(len(msg.data)))
            rospy.loginfo("Input shape is: {}".format(input.shape))

        cv_img_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        result = self.hands.process(cv_img_rgb)

        print(result)
        
        duration = rospy.Time.now().to_sec() - start_time 
        #rospy.loginfo("Duration of image_cb is: {}".format(duration)) # max --> 0.01s
                         

    #https://www.ros.org/news/2018/09/roscon-2017-determinism-in-ros---or-when-things-break-sometimes-and-how-to-fix-it----ingo-lutkebohle.html
    def run(self):

        self.first = True
        
        while not rospy.is_shutdown(): 

            if (self.first_img_reciv and self.nn_input_formed):
               
                start_time = rospy.Time.now().to_sec()
                
                # Convert ROS Image to PIL 
                start_time1 = rospy.Time.now().to_sec()
             
                duration = rospy.Time.now().to_sec() - start_time
                debug_runtime = False
                if debug_runtime:
                    rospy.loginfo("Run duration is: {}".format(duration))

            
            self.rate.sleep()
            

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
