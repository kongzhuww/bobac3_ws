#!/usr/bin/env python3
"""按 s 截图存到 dataset/ 目录，用于 YOLO 训练"""
import rospy
import cv2
import os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

SAVE_DIR = os.path.expanduser("~/yolo_dataset")
os.makedirs(SAVE_DIR, exist_ok=True)

bridge = CvBridge()
count = [0]
last_img = [None]

def cb(msg):
    last_img[0] = bridge.imgmsg_to_cv2(msg, "bgr8")

rospy.init_node("collect_data")
sub = rospy.Subscriber("/berxel_base/color/image_raw", Image, cb)

print(f"[Collect] 按 s 截图，按 q 退出，保存到 {SAVE_DIR}")
print(f"[Collect] 当前已有: {len(os.listdir(SAVE_DIR))} 张")

while not rospy.is_shutdown():
    key = input()
    if key == 'q':
        break
    elif key == 's':
        if last_img[0] is not None:
            count[0] += 1
            path = os.path.join(SAVE_DIR, f"frame_{count[0]:04d}.jpg")
            cv2.imwrite(path, last_img[0])
            print(f"  [+] {path}")
        else:
            print("  [-] 还没收到图片")

print(f"\n共 {len(os.listdir(SAVE_DIR))} 张图片，都在 {SAVE_DIR}")
