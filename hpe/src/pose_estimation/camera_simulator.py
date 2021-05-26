#!/usr/bin/python
import rospy
import sys
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


def img_publisher(video):
    publisher = rospy.Publisher("camera", Image, queue_size=1)
    bridge = CvBridge()

    filename = video + "/rgb-000"
    this_filename = "bad_name"
    for i in range(1, 233):
        if i < 10:
            this_filename = filename + "00" + str(i) + ".jpg"
        elif i < 100:
            this_filename = filename + "0" + str(i) + ".jpg"
        elif i < 1000:
            this_filename = filename + str(i) + ".jpg"
        print(this_filename)
        img = cv2.imread(this_filename, flags=cv2.IMREAD_COLOR)
        if img is None:
            print("NULL img")
            return
        image_message = bridge.cv2_to_imgmsg(img, encoding="passthrough")
        publisher.publish(image_message)
        rospy.sleep(0.2)



if __name__ == '__main__':
    rospy.init_node("camera_sim")
    img_publisher(sys.argv[1])
