#!/usr/bin/python2

import sys
import os
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from hpe_ros_msgs.msg import CartesianArmCmd
import numpy as np
import rospy
import copy


class KalmanFilterPy():

    def __init__(self, freq): 

        rospy.init_node('kalman_filter', anonymous=True)

        self.leftKf     = self.init_KF()
        self.rightKf    = self.init_KF()

        self.recv_left_arm  = False
        self.recv_right_arm = False

        self._init_subscribers()
        self._init_publishers()

        self.rate = rospy.Rate(int(freq))

        self.kalmanLeftArmMsg   = CartesianArmCmd()
        self.kalmanRightArmMsg  = CartesianArmCmd()

        rospy.loginfo("Kalman filter started!")
        

    def init_KF(self): 

        kf = KalmanFilter(dim_x=6, dim_z=3)

        # position vector
        kf.x = np.array([[0],
                        [0], 
                        [0], 
                        [0], 
                        [0], 
                        [0]])

        # State transition 
        kf.F = np.array([[1., 0.04, 0., 0.00, 0., 0.00], 
                        [0., 1.00, 0., 0.00, 0., 0.00], 
                        [0., 0.00, 1., 0.04, 0., 0.00], 
                        [0., 0.00, 0., 1.00, 0., 0.00], 
                        [0., 0.00, 0., 0.00, 1., 0.04],
                        [0., 0.00, 0., 0.00, 0., 1.00]])

        # Covariance matrix
        kf.P = np.array([[1., 0., 0., 0., 0., 0.], 
                        [0., 1., 0., 0., 0., 0.], 
                        [0., 0., 1., 0., 0., 0.], 
                        [0., 0., 0., 1., 0., 0.], 
                        [0., 0., 0., 0., 1., 0.],
                        [0., 0., 0., 0., 0., 1.]])


        # Covariance matrix for the measurement noise
        kf.R = np.eye(3)*500

        # Observation matrix
        kf.H = np.array([[1., 0., 0., 0., 0., 0.], 
                        [0., 0., 1., 0., 0., 0.], 
                        [0., 0., 0., 0., 1., 0.]])
        # Process noise
        k = 5
        kf.Q = np.array([[k, 0., 0., 0., 0., 0.], 
                        [0., k/10, 0., 0., 0., 0.], 
                        [0., 0., k, 0., 0., 0.], 
                        [0., 0., 0., k/10, 0., 0.], 
                        [0., 0., 0., 0., k, 0.], 
                        [0, 0., 0., 0., 0., k/10]])
                
        return copy.deepcopy(kf)


    def _init_subscribers(self): 

        self.cart_left_arm_sub = rospy.Subscriber("cart_left_arm", CartesianArmCmd, self.left_arm_cb, queue_size=1)
        self.cart_right_arm_sub = rospy.Subscriber("cart_right_arm", CartesianArmCmd, self.right_arm_cb, queue_size=1)

        rospy.loginfo("Subscribers intialized!")


    def _init_publishers(self): 
        # Should publish on cart_left_arm and cart_right_arm
        self.kalman_left_arm_pub = rospy.Publisher("kalman_cart_left_arm", CartesianArmCmd, queue_size=1)
        self.kalman_right_arm_pub = rospy.Publisher("kalman_cart_right_arm", CartesianArmCmd, queue_size=1)

        rospy.loginfo("Publishers intialized!")


    def left_arm_cb(self, msg): 
        rospy.loginfo_once("Recieved meas for left arm")
        if not self.recv_left_arm: 
            self.leftKf.x = np.array([[msg.positionEE.x],
                                      [0],
                                      [msg.positionEE.y],
                                      [0],
                                      [msg.positionEE.z],
                                      [0]])
            self.recv_left_arm = True

        else:
            self.left_meas = np.array([[msg.positionEE.x],
                                       [msg.positionEE.y],
                                       [msg.positionEE.z]])
            self.leftKf.predict()
            self.leftKf.update(self.left_meas)

            self.kalmanLeftArmMsg.header.stamp = rospy.Time().now()
            self.kalmanLeftArmMsg.positionEE.x = self.leftKf.x[0]
            self.kalmanLeftArmMsg.positionEE.y = self.leftKf.x[2]
            self.kalmanLeftArmMsg.positionEE.z = self.leftKf.x[4]


    def right_arm_cb(self, msg): 
        rospy.loginfo_once("Recieved meas for left arm")
        if not self.recv_right_arm: 
            self.rightKf.x = np.array([[msg.positionEE.x],
                                      [0],
                                      [msg.positionEE.y],
                                      [0],
                                      [msg.positionEE.z],
                                      [0]])
            self.recv_right_arm = True

        else:
            self.right_meas = np.array([[msg.positionEE.x],
                                       [msg.positionEE.y],
                                       [msg.positionEE.z]])
            self.rightKf.predict()
            self.rightKf.update(self.right_meas)

            self.kalmanRightArmMsg.header.stamp = rospy.Time().now()
            self.kalmanRightArmMsg.positionEE.x = self.rightKf.x[0]
            self.kalmanRightArmMsg.positionEE.y = self.rightKf.x[2]
            self.kalmanRightArmMsg.positionEE.z = self.rightKf.x[4]
            

    def run(self): 

        while not rospy.is_shutdown():

            run_ready = self.recv_right_arm and self.recv_left_arm

            if run_ready: 
                rospy.loginfo_throttle(30, "Kalman filter running")
                self.kalman_right_arm_pub.publish(self.kalmanRightArmMsg)
                self.kalman_left_arm_pub.publish(self.kalmanLeftArmMsg) 
                self.rate.sleep()


if __name__ == '__main__': 
    kalmanFilterPy = KalmanFilterPy(sys.argv[1])
    kalmanFilterPy.run()

