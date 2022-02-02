#!/usr/bin/python
from scipy.io import loadmat
import cv2
import rospy

# Load files
# annots = loadmat("preds.mat")
# img = cv2.imread("mpii/images/086310343.jpg")

move = 0
# Draw each point for the prediction
def draw_point(preds, img):
    for point in preds[0]:
        x = round(point[0])        # (1280 * point[0]) / norm_value_x
        y = round(point[1] - move)       # (720 * point[1]) / norm_value_y
        img = cv2.circle(img, (int(x), int(y)), radius=4, color=(0, 0, 255), thickness=6)
        # Show image
        cv2.imshow("Output", img)
        cv2.waitKey(1)

def draw_stickman(preds, img):
    right_foot = preds[0][0]
    right_knee = preds[0][1]
    right_ankle = preds[0][2]
    left_ankle = preds[0][3]
    left_knee = preds[0][4]
    left_foot = preds[0][5]
    center_ankles = preds[0][6]
    chest = preds[0][7]
    neck = preds[0][8]
    head = preds[0][9]
    right_hand = preds[0][10]
    right_elbow = preds[0][11]
    left_shoulder = preds[0][12]
    right_shoulder = preds[0][13]
    left_elbow = preds[0][14]
    left_hand = preds[0][15]

    right_foot = (int(right_foot[0]), int(right_foot[1]) -move)
    right_knee = (int(right_knee[0]), int(right_knee[1]) -move)
    right_ankle = (int(right_ankle[0]), int(right_ankle[1]) -move)
    left_ankle = (int(left_ankle[0]), int(left_ankle[1]) -move)
    left_knee = (int(left_knee[0]), int(left_knee[1]) -move)
    left_foot = (int(left_foot[0]), int(left_foot[1]) -move)
    center_ankles = (int(center_ankles[0]), int(center_ankles[1] -move))
    chest = (int(chest[0]), int(chest[1]) -move)
    neck = (int(neck[0]), int(neck[1]) -move)
    right_hand = (int(right_hand[0]), int(right_hand[1]) -move)
    right_elbow = (int(right_elbow[0]), int(right_elbow[1]) -move)
    right_shoulder = (int(right_shoulder[0]), int(right_shoulder[1]) -move)
    left_hand = (int(left_hand[0]), int(left_hand[1]) -move)
    left_elbow = (int(left_elbow[0]), int(left_elbow[1]) -move)
    left_shoulder = (int(left_shoulder[0]), int(left_shoulder[1]) -move)
    head = (int(head[0]), int(head[1]) -move)

    color = (0,255,0)

    #right_foot = (50,50)
    #right_knee = (350, 350)
    #print(right_foot)
    
    img = cv2.line(img, right_foot, right_knee, color, thickness=3)
    img = cv2.line(img, right_knee, right_ankle, color, thickness=3)

    img = cv2.line(img, right_ankle, center_ankles, color, thickness=3)
    img = cv2.line(img, center_ankles, left_ankle, color, thickness=3)

    img = cv2.line(img, left_ankle, left_knee, color, thickness=3)
    img = cv2.line(img, left_knee, left_foot, color, thickness=3)

    img = cv2.line(img, center_ankles, chest, color, thickness=3)
    img = cv2.line(img, chest, neck, color, thickness=3)
    img = cv2.line(img, neck, head, color, thickness=3)    

    img = cv2.line(img, chest, right_shoulder, color, thickness=3)
    img = cv2.line(img, right_shoulder, right_elbow, color, thickness=3)
    img = cv2.line(img, right_elbow, right_hand, color, thickness=3)

    img = cv2.line(img, chest, left_shoulder, color, thickness=3)
    img = cv2.line(img, left_shoulder, left_elbow, color, thickness=3)
    img = cv2.line(img, left_elbow, left_hand, color, thickness=3)
    

    draw_point(preds,img)
    cv2.imshow("Output", img)
    cv2.waitKey(1)
