import sys
import os
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from hpe_ros_msgs.msg import CartesianArmCmd
import numpy as np
import rospy


class KalmanFilterPy():
    def __init__(self, freq): 

        rospy.init_node('kalman_filter', anonymous=True)

        self.leftKf = self.init_KF(freq)
        self.rightKf = self.init_KF(freq)        

        self._init_subscribers()
        self._init_subscribers()

        rospy.loginfo("Kalman filter started!")
        

    def init_KF(self, freq): 

        kf = KalmanFilter(dim_x=3, dim_z=3)

        kf = np.array([[0.], [0.], [0.]]) 

        # state transition matrix
        kf.F = np.array([[1., 0., 0.],
                         [0., 1., 0.],
                         [0., 0., 1.]]) 
        
        kf.P = np.array([[1.0, 0.0, 0.0,],
                         [0.0, 1.0, 0.0,], 
                         [0.0, 0.0, 1.0,]])
        
        kf.R = np.eye(3) * 500

        kf.Q = Q_discrete_white_noise(dim=3, dt=1/freq, var=0.5)

        kf.H = np.array([[1.0, 0.0, 0.0], 
                         [0.0, 1.0, 0.0], 
                         [0.0, 0.0, 1.0]])
        
        return kf
        
    def _init_subscribers(self): 

        self.cart_left_arm_sub = rospy.Subscriber("/cart_left_arm", CartesianArmCmd, self.left_arm_cb queue_size=1)
        self.cart_right_arm_sub = rospy.Subscriber("/cart_right_arm", CartesianArmCmd, self.right_arm_cb queue_size=1)

        rospy.loginfo("Subscribers intialized!")

    def _init_publishers(self): 
        # Should publish on cart_left_arm and cart_right_arm
        self.kalman_left_arm_pub = rospy.Publisher("cart_left_arm", CartesianArmCmd, queue_size=1)
        self.kalman_right_arm_pub = rospy.Publisher("cart_right_arm", CartesianArmCmd, queue_size=1)

        rospy.loginfo("Publishers intialized!")

    def left_arm_cb(self, msg): 
        self.recv_left_arm = True
        self.leftKf.x = np.array([[msg.positionEE.x],
                                  [msg.positionEE.y],
                                  [msg.positionEE.z]])
        self.leftKf.predict()
        self.leftKf.update(self.leftKf.x)
        kalmanLeftArmMsg = CartesianArmCmd()
        kalmanLeftArmMsg.header.stamp = rospy.Time().now()
        kalmanLeftArmMsg.positionEE.x = self.leftKf.x[0]
        kalmanLeftArmMsg.positionEE.y = self.leftKf.x[1]
        kalmanLeftArmMsg.positionEE.z = self.leftKf.x[2]
        self.kalman_left_arm_pub.publish(msg)


    def right_arm_cb(self, msg): 
        self.recv_right_arm = True
        self.rightKf.x = np.array([[msg.positionEE.x],
                                   [msg.positionEE.y],
                                   [msg.positionEE.z]])
        self.rightKf.predict()
        self.leftKf.update(self.rightKf.x)
        kalmanRightArmMsg = CartesianArmCmd()
        kalmanRightArmMsg.header.stamp = rospy.Time().now()
        kalmanRightArmMsg.positionEE.x = self.rightKf.x[0]
        kalmanRightArmMsg.positionEE.y = self.rightKf.x[1]
        kalmanRightArmMsg.positionEE.z = self.rightKf.x[2]
        self.kalman_right_arm_pub.publish(msg)

    def run(self): 

        while not rospy.is_shutdown():
            rospy.spin()



if __name__ == '__main__': 
    rospy.init_node('kalman_filter_py', anonymous=True)
    kalmanFilterPy = KalmanFilterPy(sys.argv[1])

