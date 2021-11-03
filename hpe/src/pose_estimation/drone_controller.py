#!/opt/conda/bin/python3
import rospy
import rospkg
import sys
import cv2
import numpy

from geometry_msgs.msg import Pose, PoseStamped
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image

from PIL import ImageDraw, ImageOps, ImageFont
from PIL import Image as PILImage

from hpe_ros_inference import HumanPoseEstimationROS


class uavController:
    def __init__(self, frequency):


        nn_init_time_sec = 10
        rospy.init_node("uav_controller", log_level=rospy.DEBUG)
        rospy.sleep(nn_init_time_sec)

        self.current_x = 0
        self.current_y = 0
        self.current_z = 1
        self.current_rot = 0

        self._init_publishers(); self._init_subscribers(); 

        self.height = 480; self.width = 640; 

        # Config for control areas --> TODO: Scale to width, height to be parametric depending on image resolution 
        self.height_area = [50, 350]
        self.height_deadzone = [180, 220]
        self.rotation_area = [30, 250]
        self.rotation_deadzone = [120, 160]        
        self.x_area = [390, 610]
        self.x_deadzone = [480, 520]
        self.y_area = [50, 350]
        self.y_deadzone = [180, 220]
        self.font = ImageFont.truetype("/home/developer/catkin_ws/src/hpe_ros_package/hpe/include/arial.ttf", 20, encoding="unic")

        self.started = False
        self.rate = rospy.Rate(int(frequency))     

        self.recv_pose_meas = False

        rospy.loginfo("Initialized!")   


    def _init_publishers(self): 
        
        self.pose_pub = rospy.Publisher("uav/pose_ref", Pose, queue_size=1)
        self.stickman_area_pub = rospy.Publisher("/stickman_cont_area", Image, queue_size=1)


    def _init_subscribers(self): 

        self.preds_sub = rospy.Subscriber("hpe_preds", Float64MultiArray, self.pred_cb, queue_size=1)
        self.stickman_sub = rospy.Subscriber("stickman", Image, self.stickman_cb, queue_size=1)
        self.current_pose_sub = rospy.Subscriber("uav/pose", PoseStamped, self.curr_pose_cb, queue_size=1)
        
    
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
        
        # Convert predictions into drone positions. Goes from [1, movement_available]
        # NOTE: image is mirrored, so left control area in preds corresponds to the right hand movements 
        pose_cmd = Pose()        
        if self.recv_pose_meas and not self.started:
            rospy.logdebug("Setting up initial value!")
            pose_cmd.position.x = self.current_pose.pose.position.x
            pose_cmd.position.y = self.current_pose.pose.position.y
            pose_cmd.position.z = self.current_pose.pose.position.z
            pose_cmd.orientation.z = self.current_pose.pose.orientation.z

        elif self.recv_pose_meas and self.started: 
            try:
                pose_cmd = self.prev_pose_cmd  # Doesn't exist in the situation where we've started the algorithm however, never entered some of the zones!
            except:
                pose_cmd.position = self.current_pose.pose.position
                pose_cmd.orientation = self.current_pose.pose.orientation

        # Use info about right hand and left hand 
        rhand = preds[10]
        lhand = preds[15]

        increase = 0.03; decrease = 0.03; 
        
        if self.started:
            self.changed_cmd = False
            # Current predictions
            rospy.logdebug("Left hand: {}".format(lhand))
            rospy.logdebug("Right hand: {}".format(rhand))
             
            # Reversing mirroring operation!
            lhand[0] = abs(lhand[0] - self.width)
            rhand[0] = abs(rhand[0] - self.width)

            if self.check_if_in_range(lhand[0], self.rotation_deadzone[0], self.rotation_deadzone[1]):                                      
                rospy.logdebug("Left hand inside of rotation area!")
                if self.check_if_in_range(lhand[1], self.height_area[0], self.height_deadzone[0]):
                    pose_cmd.position.z += increase
                    rospy.logdebug("Increasing z!")
                elif self.check_if_in_range(lhand[1], self.height_deadzone[1], self.height_area[1]):
                    pose_cmd.position.z -= decrease  
                    rospy.logdebug("Decreasing z!")
     
            if self.check_if_in_range(lhand[1], self.height_area[0], self.height_area[1]):
                rospy.logdebug("Left hand inside of the height deadzone!")   
                if (self.check_if_in_range(lhand[0], self.rotation_deadzone[0], self.rotation_area[1])):
                    pose_cmd.orientation.z += increase # TODO: Generate orientation correctly for quaternion
                    rospy.logdebug("Increasing yaw!")
                
                elif(self.check_if_in_range(lhand[0], self.rotation_area[0], self.rotation_deadzone[1])):
                    pose_cmd.orientation.z -= decrease
                    rospy.logdebug("Decreasing yaw!")
            
            # Converter for x and y movements. Left hand is [15]   
            if self.check_if_in_range(rhand[0], self.x_area[0], self.x_area[1]):

                if (self.check_if_in_range(rhand[1], self.y_deadzone[1], self.y_area[1])):
                    pose_cmd.position.y += increase
                    rospy.logdebug("Increasing y!")

                elif (self.check_if_in_range(rhand[1], self.y_area[0], self.y_deadzone[0])):
                    pose_cmd.position.y -= decrease         
                    rospy.logdebug("Decreasing y!")   

            if self.check_if_in_range(rhand[1], self.y_area[0], self.y_area[1]):
                rospy.logdebug("Right hand inside of y area!")
                if self.check_if_in_range(rhand[0], self.x_deadzone[1], self.x_area[1]):
                    pose_cmd.position.x += increase    
                    rospy.logdebug("Increasing x!")

                elif self.check_if_in_range(rhand[0], self.x_area[0], self.x_deadzone[0]): 
                    pose_cmd.position.x -= decrease
                    rospy.logdebug("Decreasing x!")            
            

            rospy.loginfo("x:{0} \t y: {1} \t , z: {2} \t , rot: {3} \t".format(round(pose_cmd.position.x, 3),
                                                                                round(pose_cmd.position.y, 3),
                                                                                round(pose_cmd.position.z, 3),
                                                                                round(pose_cmd.orientation.z, 3)))
            self.prev_pose_cmd = pose_cmd
            self.pose_pub.publish(pose_cmd)


        # If not started yet, put both hand in the middle of the deadzones to start        
        else:
            # Good condition for starting 
            if rhand[0] > self.rotation_deadzone[0] and rhand[0] < self.rotation_deadzone[1] and rhand[1] > self.height_deadzone[0] and rhand[0] < self.height_deadzone[1]:
                if lhand[0] > self.x_deadzone[0] and lhand[1] < self.x_deadzone[1] and lhand[1] > self.y_deadzone[0] and lhand[1] < self.y_deadzone[1]:
                    rospy.loginfo("Started!")
                    self.started = True


        duration = rospy.Time.now().to_sec() - start_time
        #rospy.loginfo("Duration of pred_cb is: {}".format(duration))

        
     
    def stickman_cb(self, stickman_img):
        
        start_time = rospy.Time().now().to_sec()
        # Convert ROS Image to PIL
        img = numpy.frombuffer(stickman_img.data, dtype=numpy.uint8).reshape(stickman_img.height, stickman_img.width, -1)
        img = PILImage.fromarray(img.astype('uint8'), 'RGB')

        # Mirror image here 
        img = ImageOps.mirror(img) 
        
        # Draw rectangles which represent areas for control
        draw = ImageDraw.Draw(img, "RGBA")
        
        # Rectangles for height and rotation
        draw.rectangle([(self.rotation_area[0], self.height_deadzone[0]), (self.rotation_area[1], self.height_deadzone[1])],
                         fill=(178,34,34, 100), width=2)
        draw.rectangle([(self.rotation_deadzone[0], self.height_area[0]), (self.rotation_deadzone[1], self.height_area[1])],
                         fill=(178,34,34, 100), width=2)
       
        # Text for changing UAV height and yaw
        offset_x = 2; offset_y = 2; 
        up_size = uavController.get_text_dimensions("UP", self.font); down_size = uavController.get_text_dimensions("DOWN", self.font)
        yp_size = uavController.get_text_dimensions("Y+", self.font); ym_size = uavController.get_text_dimensions("Y-", self.font)
        draw.text(((self.rotation_area[0] + self.rotation_area[1])/2 - up_size[0]/2, self.height_area[0]- up_size[1] ), "UP", font=self.font)
        draw.text(((self.rotation_area[0] + self.rotation_area[1])/2 - down_size[0]/2, self.height_area[1]), "DOWN", font=self.font)
        draw.text(((self.rotation_area[0] - ym_size[0], (self.height_area[0] + self.height_area[1])/2 - ym_size[1]/2)), "Y-", font=self.font)
        draw.text(((self.rotation_area[1], (self.height_area[0] + self.height_area[1])/2 - yp_size[1]/2)), "Y+", font=self.font)

        ######################################################################################################################################
        # Rectangles for movement left-right and forward-backward
        draw.rectangle([(self.x_area[0], self.y_deadzone[0]), (self.x_area[1], self.y_deadzone[1])],
                        fill=(178,34,34, 100), width=2)
        draw.rectangle([(self.x_deadzone[0], self.y_area[0]), (self.x_deadzone[1], self.y_area[1])],
                        fill=(178,34,34, 100), width=2)
        
        # Text for moving UAV forward and backward 
        fwd_size = uavController.get_text_dimensions("FWD", self.font); bwd_size = uavController.get_text_dimensions("BWD", self.font)
        l_size = uavController.get_text_dimensions("L", self.font); r_size = uavController.get_text_dimensions("R", self.font)
        draw.text(((self.x_area[0] + self.x_area[1])/2 - fwd_size[0]/2, self.y_area[0] - fwd_size[1]), "FWD", font=self.font)
        draw.text(((self.x_area[0] + self.x_area[1])/2 - bwd_size[0]/2, self.y_area[1]), "BWD", font=self.font)
        draw.text(((self.x_area[0] - l_size[0], (self.y_area[0] + self.y_area[1])/2 - r_size[1]/2)), "L", font=self.font)
        draw.text(((self.x_area[1], (self.y_area[0] + self.y_area[1])/2 - l_size[1]/2)), "R", font=self.font)
        ########################################################################################################################################

        # Check what this mirroring does here! 
        ros_msg = uavController.convert_pil_to_ros_img(img) # Find better way to do this
        #rospy.loginfo("Publishing stickman with zones!")
        self.stickman_area_pub.publish(ros_msg)

        duration = rospy.Time().now().to_sec() - start_time
        #rospy.loginfo("stickman_cb duration is: {}".format(duration))


    def run(self): 
        #rospy.spin()
        while not rospy.is_shutdown():   
            #rospy.loginfo("CTL run")  
            self.rate.sleep()
    

    def check_if_in_range(self, value, min_value, max_value): 

        if (value >= min_value and value <= max_value): 
            return True

        else: 
            return False 



    
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
    def get_text_dimensions(text_string, font):

        ascent, descent = font.getmetrics()
        text_width = font.getmask(text_string).getbbox()[2]
        text_height = font.getmask(text_string).getbbox()[3] + descent

        return (text_width, text_height)
        

if __name__ == '__main__':

    uC = uavController(sys.argv[1])
    uC.run()