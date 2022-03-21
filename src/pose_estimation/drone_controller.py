#!/opt/conda/bin/python3
from audioop import avg
from multiprocessing import reduction
import queue
from turtle import width
import rospy
import rospkg
import sys
import cv2
import numpy 

from geometry_msgs.msg import Pose, PoseStamped
from std_msgs.msg import Float64MultiArray, Int32, Bool
from sensor_msgs.msg import Image, CompressedImage, Joy, PointCloud2
import sensor_msgs.point_cloud2 as pc2

from PIL import ImageDraw, ImageOps, ImageFont
from PIL import Image as PILImage

# TODO:
# - think of behavior when arm goes out of range! --> best to do nothing, just send references when there's signal from HPE 
# - test depth averaging 

class uavController:

    def __init__(self, frequency, use_calibration):

        nn_init_time_sec = 10
        rospy.init_node("uav_controller", log_level=rospy.DEBUG)
        rospy.sleep(nn_init_time_sec)

        # Available control types are: euler, euler2d 
        self.control_type = "euler2d" 

        self._init_publishers(); self._init_subscribers(); 

        # Define zones / dependent on input video image 
        self.height = 480; self.width = 640; 

        # Decoupled control  
        if self.control_type == "euler": 
            
            # 1D control zones
            self.height_rect, self.yaw_rect, self.pitch_rect, self.roll_rect = self.define_ctl_zones(self.width, self.height, 0.2, 0.05)
            self.start_joy_ctl = False

        # Coupled control --> Yaw/Height on same axis, Roll/Pitch on same axis
        if self.control_type == "euler2d": 

            # 2D control zones
            self.r_zone = self.define_ctl_zone( self.width/4, self.height/1.5, 3 * self.width/4, self.height/2)
            self.l_zone = self.define_ctl_zone( self.width/4, self.height/1.5, self.width/4, self.height/2)

            rospy.logdebug("Right zone: {}".format(self.r_zone))
            rospy.logdebug("Left zone: {}".format(self.l_zone))

            self.height_rect, self.yaw_rect, self.pitch_rect, self.roll_rect = self.define_2d_ctl_zones(self.l_zone, self.r_zone, 25)
            self.start_joy2d_ctl = False

        # Define deadzones
        self.l_deadzone = self.define_deadzones(self.height_rect, self.yaw_rect)
        self.r_deadzone = self.define_deadzones(self.pitch_rect, self.roll_rect)

        self.font = ImageFont.truetype("/home/developer/catkin_ws/src/hpe_ros_package/include/arial.ttf", 20, encoding="unic")

        self.start_position_ctl = False
        
        self.rate = rospy.Rate(int(frequency))     

        # Debugging arguments
        self.inspect_keypoints = False
        self.recv_pose_meas = False        

        # Image compression for human-machine interface
        self.hmi_compression = False
        # If calibration determine zone-centers
        self.start_calib = False
        # If use depth
        self.use_depth = True
        self.depth_recv = False
        self.depth_pcl_recv = False

        # Initialize start calib time to very large value to start calibration when i publish to topic
        self.calib_duration = 10
        self.rhand_calib_px, self.rhand_calib_py = [], []
        self.lhand_calib_px, self.lhand_calib_py = [], []
        
        # Flags for run method
        self.initialized = True
        self.prediction_started = False

        rospy.loginfo("Initialized!")   

    def _init_publishers(self): 
        
        #TODO: Add topics to yaml file
        if self.control_type == "position":
            self.pose_pub = rospy.Publisher("bebop/pos_ref", Pose, queue_size=1)

        if self.control_type == "euler" or self.control_type == "euler2d": 
            self.joy_pub = rospy.Publisher("/joy", Joy, queue_size=1)

        self.stickman_area_pub = rospy.Publisher("/stickman_cont_area", Image, queue_size=1)
        self.stickman_compressed_area_pub = rospy.Publisher("/stickman_compressed_ctl_area", CompressedImage, queue_size=1)

        self.lhand_x_pub = rospy.Publisher("hpe/lhand_x", Int32, queue_size=1)
        self.rhand_x_pub = rospy.Publisher("hpe/rhand_x", Int32, queue_size=1)
        self.lhand_y_pub = rospy.Publisher("hpe/lhand_y", Int32, queue_size=1)
        self.rhand_y_pub = rospy.Publisher("hpe/rhand_y", Int32, queue_size=1)

    def _init_subscribers(self): 

        self.preds_sub          = rospy.Subscriber("hpe_preds", Float64MultiArray, self.pred_cb, queue_size=1)
        self.stickman_sub       = rospy.Subscriber("stickman", Image, self.draw_zones_cb, queue_size=1)
        self.current_pose_sub   = rospy.Subscriber("uav/pose", PoseStamped, self.curr_pose_cb, queue_size=1)
        self.start_calib_sub    = rospy.Subscriber("start_calibration", Bool, self.calib_cb, queue_size=1)
        self.depth_sub          = rospy.Subscriber("camera/depth/image", Image, self.depth_cb, queue_size=1)
        self.depth_pcl_sub      = rospy.Subscriber("camera/depth/points", PointCloud2, self.depth_pcl_cb, queue_size=1)
           
    def publish_predicted_keypoints(self, rhand, lhand): 

        rhand_x, rhand_y = rhand[0], rhand[1]; 
        lhand_x, lhand_y = lhand[0], lhand[1]

        rospy.logdebug("rhand: \t x: {}\t y: {}".format(rhand_x, rhand_y))
        rospy.logdebug("lhand: \t x: {}\t y: {}".format(lhand_x, lhand_y))

        self.lhand_x_pub.publish(int(lhand_x))
        self.lhand_y_pub.publish(int(lhand_y))
        self.rhand_x_pub.publish(int(rhand_x))
        self.rhand_y_pub.publish(int(rhand_y))

    def curr_pose_cb(self, msg):
        
        self.recv_pose_meas = True; 
        self.current_pose = PoseStamped(); 
        self.current_pose.header = msg.header
        self.current_pose.pose.position = msg.pose.position
        self.current_pose.pose.orientation = msg.pose.orientation

    def pred_cb(self, converted_preds):
        preds = []

        start_time = rospy.Time().now().to_sec()
        # Why do we use switcher? 
        switcher = False
        for pred in converted_preds.data:
            if switcher is False:            
                preds.append([pred, 0])
            if switcher is True:
                preds[-1][1] = pred
            switcher = not switcher
        
        # Use info about right hand and left hand 
        self.rhand = preds[10]
        self.lhand = preds[15]

        self.prediction_started = True; 

        if self.inspect_keypoints:  
            self.publish_predicted_keypoints(self.rhand, self.lhand)

    def calib_cb(self, msg): 
        
        self.start_calib = msg.data

        self.start_calib_time = rospy.Time.now().to_sec()

    def draw_zones_cb(self, stickman_img):
        
        start_time = rospy.Time().now().to_sec()
        # Convert ROS Image to PIL
        img = numpy.frombuffer(stickman_img.data, dtype=numpy.uint8).reshape(stickman_img.height, stickman_img.width, -1)
        img = PILImage.fromarray(img.astype('uint8'), 'RGB')

        # Mirror image here 
        img = ImageOps.mirror(img) 
        
        # Draw rectangles which represent areas for control
        draw = ImageDraw.Draw(img, "RGBA")

        if self.control_type == "euler2d": 

            draw.rectangle(self.l_zone, width = 3)
            draw.rectangle(self.r_zone, width = 3)
        
        # Rect for yaw
        draw.rectangle(self.yaw_rect, fill=(178,34,34, 100), width=2)       

        # Rect for height
        draw.rectangle(self.height_rect, fill=(178,34,34, 100), width=2)

        # Rectangles for pitch
        draw.rectangle(self.pitch_rect, fill=(178,34,34, 100), width=2)
        
        # Rect for roll 
        draw.rectangle(self.roll_rect, fill=(178,34,34, 100), width=2)
       
        # Text for changing UAV height and yaw
        offset_x = 2; offset_y = 2; 
        # Text writing --> Think how to define this
        ###########################################################################################################################################
        #up_size = uavController.get_text_dimensions("UP", self.font); down_size = uavController.get_text_dimensions("DOWN", self.font)
        #yp_size = uavController.get_text_dimensions("Y+", self.font); ym_size = uavController.get_text_dimensions("Y-", self.font)
        #draw.text(((self.rotation_area[0] + self.rotation_area[1])/2 - up_size[0]/2, self.height_area[0]- up_size[1] ), "UP", font=self.font, fill="black")
        #draw.text(((self.rotation_area[0] + self.rotation_area[1])/2 - down_size[0]/2, self.height_area[1]), "DOWN", font=self.font, fill="black")
        
        ############################################################################################################################################
        #draw.text(((self.rotation_area[0] - ym_size[0], (self.height_area[0] + self.height_area[1])/2 - ym_size[1]/2)), "Y-", font=self.font)
        #draw.text(((self.rotation_area[1], (self.height_area[0] + self.height_area[1])/2 - yp_size[1]/2)), "Y+", font=self.font)

        ######################################################################################################################################
        
        # Text for moving UAV forward and backward 
        #fwd_size = uavController.get_text_dimensions("FWD", self.font); bwd_size = uavController.get_text_dimensions("BWD", self.font)
        #l_size = uavController.get_text_dimensions("L", self.font); r_size = uavController.get_text_dimensions("R", self.font)
        #draw.text(((self.x_area[0] + self.x_area[1])/2 - r_size[0]/2, self.y_area[0] - r_size[1]), "L", font=self.font, fill="black")
        #draw.text(((self.x_area[0] + self.x_area[1])/2 - l_size[0]/2, self.y_area[1]), "R", font=self.font, fill="black")
        #draw.text(((self.x_area[0] - bwd_size[0], (self.y_area[0] + self.y_area[1])/2 - bwd_size[1]/2)), "BWD", font=self.font, fill="black")
        #draw.text(((self.x_area[1], (self.y_area[0] + self.y_area[1])/2 - fwd_size[1]/2)), "FWD", font=self.font, fill="black")
        ########################################################################################################################################

        # Check what this mirroring does here! --> mirroring is neccessary to see ourselves when operating 
        #rospy.loginfo("Publishing stickman with zones!")

        if self.hmi_compression: 
            rospy.loginfo("Compressing zones")
            compressed_msg = uavController.convert_pil_to_ros_compressed(img, color_conversion="True")
            self.stickman_compressed_area_pub.publish(compressed_msg)            

        else:             
            ros_msg = uavController.convert_pil_to_ros_img(img) 
            self.stickman_area_pub.publish(ros_msg)

        duration = rospy.Time().now().to_sec() - start_time
        #rospy.loginfo("stickman_cb duration is: {}".format(duration))
    def depth_cb(self, msg): 
        
        self.depth_recv = True
        #self.depth_msg = numpy.frombuffer(msg.data, dtype=numpy.uint8).reshape(self.width, self.height, 4)

    def depth_pcl_cb(self, msg): 

        #https://answers.ros.org/question/191265/pointcloud2-access-data/

        self.depth_pcl_recv = True
        self.depth_pcl_msg = PointCloud2()
        self.depth_pcl_msg = msg

    def average_depth_cluster(self, px, py, k, config="WH"): 

        indices = []
        start_px = int(px - k); stop_px = int(px + k); 
        start_py = int(py - k); stop_py = int(py + k); 

        # Paired indices
        for px in range(start_px, stop_px, 1): 
                for py in range(start_py, stop_py, 1): 
                    # Row major indexing
                    if config == "WH": 
                        indices.append((px, py))
                    # Columnt major indexing
                    if config == "HW":
                        indices.append((py, px))
            
        # Fastest method for fetching specific indices!
        depths = pc2.read_points(self.depth_pcl_msg, ['z'], False, uvs=indices)
        
        try:

            depths = numpy.array(list(depths), dtype=numpy.float32)
            depth_no_nans = list(depths[~numpy.isnan(depths)])

            if len(depth_no_nans) > 0:                
                
                avg_depth = sum(depth_no_nans) / len(depth_no_nans)
                rospy.logdebug("{} Average depth is: {}".format(config, avg_depth))
                return avg_depth

            else: 

                return None
        
        except Exception as e:
            rospy.logwarn("Exception occured: {}".format(str(e))) 
            
            return None

    def define_ctl_zones(self, img_width, img_height, edge_offset, rect_width):
        
        # img center
        cx, cy = img_width/2, img_height/2
        # 1st zone
        cx1, cy1 = cx/2, cy/2
        # 2nd zone
        cx2, cy2 = cx + cx1, cy + cy1
        
        # Define offsets from edge
        if edge_offset < 1: 
            height_edge = edge_offset * img_height
            width_edge = edge_offset/2 * img_width

        # Zone definition 
        if rect_width < 1: 
            r_width = rect_width * img_width

        # Define rectangle for height control
        height_rect = ((cx1 - r_width, height_edge), (cx1 + r_width, img_height - height_edge))
        # Define rectangle for yaw control
        yaw_rect = ((width_edge, cy - r_width), (cx - width_edge, cy + r_width))
        # Define rectangle for pitch control
        pitch_rect = ((cx2 - r_width, height_edge), (cx2 + r_width, img_height - height_edge))
        # Define rectangle for roll control 
        roll_rect = ((cx + width_edge, cy-r_width), (img_width - width_edge, cy + r_width))
        
        return height_rect, yaw_rect, pitch_rect, roll_rect

    def define_calibrated_ctl_zones(self, calibration_points, img_w, img_h, w_perc=0.2, h_perc=0.3, rect_perc=0.05):

        cx1, cy1 = calibration_points[0][0], calibration_points[0][1]
        cx2, cy2 = calibration_points[1][0], calibration_points[1][1]

        # main_control dimensions
        a = img_w * w_perc
        b = img_h * h_perc

        # control_rect width
        c = img_w * rect_perc
        d = img_h * rect_perc

        # Define rectangles for heigh, yaw, pitch and roll
        height_rect     = ((cx1 - c, cy1 - b), (cx1 + c, cy1 + b))
        yaw_rect        = ((cx1 - a, cy1 - d), (cx1 + a, cy1 + d))
        pitch_rect      = ((cx2 - c, cy2 - b), (cx2 + c, cy2 + b))
        roll_rect       = ((cx2 - a, cy2 - d), (cx2 + a, cy2 + d))

        return height_rect, yaw_rect, pitch_rect, roll_rect    
    
    def define_deadzones(self, rect1, rect2):

        # valid if first rect vertical, second rect horizontal
        x01, y01 = rect1[0][0], rect1[0][1]
        x11, y11 = rect1[1][0], rect1[1][1]

        x02, y02 = rect2[0][0], rect2[0][1]
        x12, y12 = rect2[1][0], rect2[1][1]

        deadzone_rect = ((x01, y02), (x11, y12))

        return deadzone_rect      

    # 1D checking if in range  
    def check_if_in_range(self, value, min_value, max_value): 

        if (value >= min_value and value <= max_value): 
            return True

        else: 
            return False 
    
    # 2D checking inf in range
    def in_zone(self, point, rect): 

        x, y   = point[0], point[1]
        x0, y0 = rect[0][0], rect[0][1]
        x1, y1 = rect[1][0], rect[1][1]

        x_cond = True if (x >= x0 and x <= x1) else False
        y_cond = True if (y >= y0 and y <= y1) else False

        if x_cond and y_cond:
            return True
        else:
            return False

    def determine_center(self, rect):

        x0, y0 = rect[0][0], rect[0][1]
        x1, y1 = rect[1][0], rect[1][1]

        cx = (x1-x0)/2 + x0; 
        cy = (y1-y0)/2 + y0; 

        return (cx, cy)

    def in_ctl_zone(self, point, rect, deadzone, orientation): 

        x, y = point[0], point[1]
        x0, y0  = rect[0][0], rect[0][1]
        x1, y1  = rect[1][0], rect[1][1]
        cx, cy = self.determine_center(rect)

        if orientation == "vertical":
            rect1 = ((x0, y0), (cx + deadzone, cy - deadzone) )
            rect2 = ((cx - deadzone, cy + deadzone), (x1, y1))

        if orientation == "horizontal": 

            rect1 = ((x0, y0), (cx - deadzone, cy + deadzone))
            rect2 = ((cx + deadzone, cy - deadzone), (x1, y1))

        # Check in which rect is point located
        if self.in_zone(point, rect1) or self.in_zone(point, rect2): 

            norm_x_diff = (cx - x) / ((x1 - x0)/2)
            norm_y_diff = (cy - y) / ((y1 - y0)/2)

            return norm_x_diff, norm_y_diff

        else: 

            return 0.0, 0.0

    def compose_joy_msg(self, pitch, roll, yaw, height):

        joy_msg = Joy()

        joy_msg.header.stamp = rospy.Time.now()
        joy_msg.axes = [yaw, height, roll, pitch]
        joy_msg.buttons = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        return joy_msg

    def run_position_ctl(self, lhand, rhand):

        # Convert predictions into drone positions. Goes from [1, movement_available]
        # NOTE: image is mirrored, so left control area in preds corresponds to the right hand movements 
        pose_cmd = Pose()        
        if self.recv_pose_meas and not self.start_position_ctl:
            rospy.logdebug("Setting up initial value!")
            pose_cmd.position.x = self.current_pose.pose.position.x
            pose_cmd.position.y = self.current_pose.pose.position.y
            pose_cmd.position.z = self.current_pose.pose.position.z
            pose_cmd.orientation.z = self.current_pose.pose.orientation.z

        elif self.recv_pose_meas and self.start_position_ctl: 
            try:
                pose_cmd = self.prev_pose_cmd  # Doesn't exist in the situation where we've started the algorithm however, never entered some of the zones!
            except:
                pose_cmd.position = self.current_pose.pose.position
                pose_cmd.orientation = self.current_pose.pose.orientation
        
        increase = 0.03; decrease = 0.03; 
        
        if self.start_position_ctl:
            self.changed_cmd = False
            # Current predictions
            rospy.logdebug("Left hand: {}".format(lhand))
            rospy.logdebug("Right hand: {}".format(rhand))
            
    # TODO: Implement position change in same way it has been implemented for joy control     
    def run_joy_ctl(self, lhand, rhand): 

        joy_msg = Joy()

        height_w, height_h = self.in_ctl_zone(lhand, self.height_rect, 20, orientation="vertical")
        height_cmd = height_h 

        yaw_w, yaw_h = self.in_ctl_zone(lhand, self.yaw_rect, 20, orientation="horizontal")
        yaw_cmd = yaw_w

        pitch_w, pitch_h = self.in_ctl_zone(rhand, self.pitch_rect, 20, orientation="vertical")
        pitch_cmd = pitch_h

        roll_w, roll_h = self.in_ctl_zone(rhand, self.roll_rect, 20, orientation="horizontal")
        roll_cmd = roll_w 

        reverse_dir = -1
        # Added reverse because rc joystick implementation has reverse
        reverse = True 
        if reverse: 
            roll_cmd  *= reverse_dir

        rospy.logdebug("Height cmd: {}".format(height_cmd))
        rospy.logdebug("Yaw cmd: {}".format(yaw_cmd))
        rospy.logdebug("Pitch cmd: {}".format(pitch_cmd))
        rospy.logdebug("Roll cmd: {}".format(roll_cmd))
        
        # Compose from commands joy msg
        joy_msg = self.compose_joy_msg(pitch_cmd, roll_cmd, yaw_cmd, height_cmd)

        # Publish composed joy msg
        self.joy_pub.publish(joy_msg)
    
    # 2D control 
    def define_ctl_zone(self, w, h, cx, cy):

        ctl_rect = ((cx - w/2, cy - h/2), (cx + w/2, cy + h/2))

        return ctl_rect 

    # TODO: Fix this part!
    def define_2d_ctl_zones(self, l_zone, r_zone, deadzone): 

        cx1, cy1 = (l_zone[0][0] + l_zone[1][0])/2, (l_zone[0][1] + l_zone[1][1])/2
        cx2, cy2 = (r_zone[0][0] + r_zone[1][0])/2, (r_zone[0][1] + r_zone[1][1])/2

        rospy.logdebug("cx1, cy1: {}, {}".format(cx1, cy1))
        rospy.logdebug("cx2, cy2: {}, {}".format(cx2, cy2))
        
        height_rect = ((cx1 - deadzone, l_zone[0][1]), (cx1 + deadzone, l_zone[1][1]))
        pitch_rect = ((cx2 - deadzone, r_zone[0][1]), (cx2 + deadzone, r_zone[1][1]))

        roll_rect = ((r_zone[0][0], cy2 - deadzone), (r_zone[1][0], cy2 + deadzone))
        yaw_rect = ((l_zone[0][0], cy1 - deadzone), (l_zone[1][0], cy1 + deadzone))

        rospy.logdebug("Height: {}".format(height_rect))
        rospy.logdebug("Pitch: {}".format(pitch_rect))
        rospy.logdebug("Roll: {}".format(roll_rect))
        rospy.logdebug("Yaw: {}".format(yaw_rect))


        return height_rect, yaw_rect, pitch_rect, roll_rect

    def in_ctl2d_zone(self, point, rect, deadzone): 

        x, y = point[0], point[1]
        x0, y0 = rect[0][0], rect[0][1]
        x1, y1 = rect[1][0], rect[1][1]
        cx, cy = (x1 + x0) / 2, (y1 + y0) / 2

        rospy.logdebug("x0: {}\t x1: {}".format(x0, x1))
        rospy.logdebug("y0: {}\t y1: {}".format(y0, y1))
        rospy.logdebug("cx: {}".format(cx))
        rospy.logdebug("cy: {}".format(cy))
        
        if self.in_zone(point, rect): 
            
            rospy.logdebug("x: {}".format(x))

            rospy.logdebug("y: {}".format(y))

            if abs(cx - x) > deadzone: 
                norm_x_diff = (x - cx) / ((x1 - x0) / 2)
            else: 
                norm_x_diff = 0.0

            if abs(cy - y) > deadzone: 
                norm_y_diff = (y - cy) / ((y1 - y0) / 2)
            else: 
                norm_y_diff = 0.0
        
        else: 

            norm_x_diff, norm_y_diff = 0.0, 0.0

        return norm_x_diff, norm_y_diff

    def run_joy2d_ctl(self, lhand, rhand): 

        yaw_cmd, height_cmd = self.in_ctl2d_zone(lhand, self.l_zone, 25)
        pitch_cmd, roll_cmd = self.in_ctl2d_zone(rhand, self.r_zone, 25)

        reverse_dir = -1
        # Added reverse because rc joystick implementation has reverse
        reverse = True 
        if reverse: 
            roll_cmd  *= reverse_dir

        # Test!
        rospy.logdebug("Height cmd: {}".format(height_cmd))
        rospy.logdebug("Yaw cmd: {}".format(yaw_cmd))
        rospy.logdebug("Pitch cmd: {}".format(pitch_cmd))
        rospy.logdebug("Roll cmd: {}".format(roll_cmd))

        # Compose from commands joy msg
        joy_msg = self.compose_joy_msg(pitch_cmd, roll_cmd, yaw_cmd, height_cmd)

        # Publish composed joy msg
        self.joy_pub.publish(joy_msg)
         
    def run(self): 
        #rospy.spin()

        while not rospy.is_shutdown():
            if not self.initialized or not self.prediction_started: 
                rospy.logdebug("Waiting prediction")
                rospy.sleep(0.1)
            else:

                # Reverse mirroring operation: 
                lhand_ = (abs(self.lhand[0] - self.width), self.lhand[1])
                rhand_ = (abs(self.rhand[0] - self.width), self.rhand[1])

                if self.start_calib:

                    # Added dummy sleep to test calibration
                    duration = rospy.Time.now().to_sec() - self.start_calib_time
                    if duration < self.calib_duration:
                        # Disable calibration during execution  
                        self.control_type = "None"  
                        self.rhand_calib_px.append(rhand_[0]), self.rhand_calib_py.append(rhand_[1])
                        self.lhand_calib_px.append(lhand_[0]), self.lhand_calib_py.append(lhand_[1])         

                    
                    else:
                        avg_rhand = (int(sum(self.rhand_calib_px)/len(self.rhand_calib_px)), int(sum(self.rhand_calib_py)/len(self.rhand_calib_py)))
                        avg_lhand = (int(sum(self.lhand_calib_py)/len(self.lhand_calib_px)), int(sum(self.lhand_calib_py)/len(self.lhand_calib_py)))
                        calib_points = (avg_lhand, avg_rhand)
                        self.height_rect, self.yaw_rect, self.pitch_rect, self.roll_rect =  self.define_calibrated_ctl_zones(calib_points, self.width, self.height)
                        self.l_deadzone = self.define_deadzones(self.height_rect, self.yaw_rect)
                        self.r_deadzone = self.define_deadzones(self.pitch_rect, self.roll_rect)
                        self.control_type = "euler"
                        self.start_calib = False

                        # Calibration
                        zone_debug = False 
                        if zone_debug:
                            rospy.loginfo("Calib points are: {}".format(calib_points))
                            rospy.loginfo("Height rect: {}".format(self.height_rect))
                            rospy.loginfo("Yaw rect: {}".format(self.yaw_rect))
                            rospy.loginfo("Pitch rect: {}".format(self.pitch_rect))
                            rospy.loginfo("Roll rect: {}".format(self.roll_rect))
                            rospy.loginfo("Right deadzone: {}".format(self.r_deadzone))
                            rospy.loginfo("Left deadzone: {}".format(self.l_deadzone))


                if self.control_type == "position": 

                    self.run_position_ctl(lhand_, rhand_)

                if self.control_type == "euler": 

                    rospy.loginfo("rhand_: {}".format(rhand_))
                    rospy.loginfo("lhand_: {}".format(lhand_))
                    rospy.loginfo("r_deadzone: {}".format(self.r_deadzone))
                    rospy.loginfo("l_deadzone: {}".format(self.l_deadzone))                    

                    if self.in_zone(lhand_, self.l_deadzone) and self.in_zone(rhand_, self.r_deadzone):
                        self.start_joy_ctl = True 
                    else:
                        rospy.loginfo("Not in deadzones!")           
                    
                    if self.start_joy_ctl:
                        self.run_joy_ctl(lhand_, rhand_)

                if self.control_type == "euler2d": 

                    # Currently slow! --> Speed it up!
                    #pcl_matrix = self.get_current_depth()
                    #if self.depth_recv:
                    #    self.get_averaged_depth(pcl_matrix, rhand_[0], rhand_[1], 1)

                    if self.depth_recv:
                        # Using self.rhand and rhand_[1] because self.rhand is not mirrored (which makes it okay for depth!)
                        rospy.logdebug("Right hand!") 
                        self.average_depth_cluster(self.rhand[0], rhand_[1], 2, "WH")
                        rospy.logdebug("Left hand!")
                        self.average_depth_cluster(self.lhand[0], lhand_[1], 2, "WH")

                    if self.in_zone(lhand_, self.l_deadzone) and self.in_zone(rhand_, self.r_deadzone):
                        self.start_joy2d_ctl = True 
                    else:
                        rospy.loginfo("Not in deadzones!")        
                    
                    if self.start_joy2d_ctl: 
                        self.run_joy2d_ctl(lhand_, rhand_)

                self.rate.sleep()

    @staticmethod
    def convert_pil_to_ros_img(img):
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
    def convert_pil_to_ros_compressed(img, color_conversion = False, compression_type="jpeg"):

        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()       
        msg.format = "{}".format(compression_type)
        np_img = numpy.array(img); #bgr 

        if color_conversion: 
            np_img = uavController.bgr2rgb(np_img)            
        
        compressed_img = cv2.imencode(".{}".format(compression_type), np_img)[1]
        msg.data = compressed_img.tobytes()

        return msg

    @staticmethod
    def get_text_dimensions(text_string, font):

        ascent, descent = font.getmetrics()
        text_width = font.getmask(text_string).getbbox()[2]
        text_height = font.getmask(text_string).getbbox()[3] + descent

        return (text_width, text_height)
    
    @staticmethod
    def bgr2rgb(img):
        
        rgb_img = numpy.zeros_like(img)
        rgb_img[:, :, 0] = img[:, :, 2]
        rgb_img[:, :, 1] = img[:, :, 1]
        rgb_img[:, :, 2] = img[:, :, 0]
        
        return rgb_img

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")
        
if __name__ == '__main__':

    uC = uavController(sys.argv[1], sys.argv[2])
    uC.run()