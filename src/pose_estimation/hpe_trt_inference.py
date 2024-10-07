#!/usr/bin/python3.8

import queue

import numpy
from numpy.core.fromnumeric import compress
import rospy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
import cv2

import argparse
import sys
import statistics

from PIL import ImageDraw, ImageFont
from PIL import Image as PILImage


from std_msgs.msg import Bool

class HumanPoseEstimationROS():

    def __init__(self, frequency, args):

        # Init node 

        # Init subscribers     

        # Init publishers

        # Init model

        pass


    def _init_subscribers(self):
        
        if self.use_depth:
            # Xtion Cam
            self.camera_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_cb, queue_size=1)
        else:
            # USB Cam
            self.camera_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_cb, queue_size=1)
            self.camera_info_sub = rospy.Subscriber("camera/color/camera_info", CameraInfo, self.cinfo_cb, queue_size=1)
            
        #self.darknet_sub = rospy.Subscriber("/darknet_ros/bounding_boxes", BoundingBoxes, self.darknet_cb, queue_size=1)

    def _init_publishers(self):
        self.dummy_pub = rospy.Publisher("/dummy_pub", Bool, queue_size=1)
        self.image_pub = rospy.Publisher("/stickman", Image, queue_size=1)
        self.image_compressed_pub = rospy.Publisher("/stickman_compressed", CompressedImage, queue_size=1)
        self.pred_pub = rospy.Publisher("/hpe_preds", Float64MultiArray, queue_size=1)

    def _init_model(): 
        pass
    

    def cinfo_cb(self, msg): 

        # Get valid img width and height
        self.img_height = msg.height
        self.img_width = msg.width
        pass

    def image_cb(self, msg):

        start_time = rospy.Time.now().to_sec()

        self.first_img_reciv = True

        debug_img = False
        if debug_img:
            rospy.loginfo("Image width: {}".format(msg.width))
            rospy.loginfo("Image height: {}".format(msg.height))
            rospy.loginfo("Data is: {}".format(len(msg.data)))
            rospy.loginfo("Input shape is: {}".format(input.shape))

        # Transform img to numpy array        
        self.org_img = numpy.frombuffer(msg.data, dtype=numpy.uint8).reshape(msg.height, msg.width, -1)
        
        #self.logger.info("Scale is: {}".format(self.scale))
        self.center = 384/2, 384/2

        
        # Check this scaling
        self.scale = numpy.array([1, 1], dtype=numpy.float32) 
        
        # Transform img to 
        self.nn_input = transform(self.cam_img).unsqueeze(0).to(self.device)   

        self.nn_input_formed = True
        
        if debug_img:
            rospy.loginfo("NN_INPUT {}".format(self.nn_input))             

        duration = rospy.Time.now().to_sec() - start_time 
        #rospy.loginfo("Duration of image_cb is: {}".format(duration)) # max --> 0.01s
                         
 


    def filter_predictions(self, predictions, filter_type="avg", w_size=3): 

        preds_ = predictions[0]

        for i, prediction in enumerate(preds_): 
            
            # Keypoint 10 should be left wrist
            if i == 10: 
                l_x = preds_[i][0]; l_y = preds_[i][1]; 

            # Keypoint 15 should be right wrist
            if i == 15:
                r_x = preds_[i][0]; r_y = preds_[i][1]; 

        
        p_right = (r_x, r_y); p_left = (l_x, l_y)
        filtered_lx, filtered_ly, filtered_rx, filtered_ry = self.filtering(p_right, p_left, type=filter_type, window_size=w_size)

        preds_[10][0] = int(filtered_lx); preds_[10][1] = int(filtered_ly); 
        preds_[15][0] = int(filtered_rx); preds_[15][1] = int(filtered_ry); 

        return preds_

    # TODO: Add into other filtering scripts
    def filtering(self, p_right, p_left, type="avg", window_size=3):
        
        # TO STUPID WAY TO FILTER STUFF!
        if self.first_img_reciv and not self.filter_first_pass : 
            self.left_x = []; self.left_y = []; 
            self.right_x = []; self.right_y = [];          
            self.filter_first_pass = True 
        
        else:
            self.left_x.append(p_left[0]); self.left_y.append(p_left[1])
            self.right_x.append(p_right[0]); self.right_y.append(p_right[1])

            if len(self.left_x) >= window_size: 
                self.filtering_active = True
                # Cropped
                crop_l_x = self.left_x[-1 - window_size:].copy(); crop_l_y = self.left_y[-1 - window_size:].copy()
                crop_r_x = self.right_x[-1 - window_size:].copy(); crop_r_y = self.right_y[-1 - window_size:].copy()

                if type == "avg": 

                    filter_l_x = HumanPoseEstimationROS.avg_list(crop_l_x); filter_l_y = self.avg_list(crop_l_y)
                    filter_r_x = HumanPoseEstimationROS.avg_list(crop_r_x); filter_r_y = self.avg_list(crop_r_y)

                elif type == "median":

                    filter_l_x = HumanPoseEstimationROS.median_list(crop_l_x); filter_l_y = self.median_list(crop_l_y)
                    filter_r_x = HumanPoseEstimationROS.median_list(crop_r_x); filter_r_y = self.median_list(crop_r_y)

                buffer = 5
                if len(self.left_x) > buffer:
                    self.left_x = self.left_x[-buffer:]; self.left_y = self.left_y[-buffer:]
                    self.right_x = self.right_x[-buffer:]; self.right_y = self.right_y[-buffer:]              

        if self.filtering_active and self.first_img_reciv:

            return filter_l_x, filter_l_y, filter_r_x, filter_r_y

        else: 

            return p_left[0], p_left[1], p_right[0], p_right[1]
            
    #https://www.ros.org/news/2018/09/roscon-2017-determinism-in-ros---or-when-things-break-sometimes-and-how-to-fix-it----ingo-lutkebohle.html
    def run(self):

        self.first = True

        while not self.model_ready:
            self.logger.info("SimpleBaselines model for HPE not ready.")
        
        while not rospy.is_shutdown(): 

            if (self.first_img_reciv and self.nn_input_formed):
                pass          


    
    #TODO: add those methods to img_utils --> when I have all neccessary methods for img_utils
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

    @staticmethod
    def convert_pil_to_ros_compressed(img, compression_type="jpeg"):

        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()       
        msg.format = "{}".format(compression_type)
        np_img = numpy.array(img); #bgr 
        rgb_img = np_img[:, :, [2, 1, 0]]
        
        compressed_img = cv2.imencode(".{}".format(compression_type), rgb_img)[1]
        msg.data = compressed_img.tobytes()

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

    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    # training
    parser.add_argument('--frequent',
                        help='frequency of logging',
                        default=config.PRINT_FREQ,
                        type=int)
    parser.add_argument('--gpus',
                        help='gpus',
                        type=str)
    parser.add_argument('--workers',
                        help='num of dataloader workers',
                        type=int)
    parser.add_argument('--model-file',
                        help='model state file',
                        type=str)
    parser.add_argument('--use-detect-bbox',
                        help='use detect bbox',
                        action='store_true')
    parser.add_argument('--coco-bbox-file',
                        help='coco detection bbox file',
                        type=str)
    parser.add_argument('--shift-heatmap', 
                        help="shift heatmap", 
                        default=False, 
                        type=bool)
    parser.add_argument('--use-depth', 
                        help="use depth cam", 
                        default=False, 
                        type=bool)

    args, unknown = parser.parse_known_args(sys.argv[2:])
 

    HPERos = HumanPoseEstimationROS(20, args)
    HPERos.run()
