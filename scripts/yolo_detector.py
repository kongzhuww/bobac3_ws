#!/usr/bin/env python3
"""
YOLOv8 ROS Detector - subscribes to camera and runs object detection
Uses COCO pretrained model (includes fire_extinguisher class)
"""
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import String
from ultralytics import YOLO

# COCO class names that might interest you
WATCH_CLASSES = {
    "fire extinguisher": 73,
    "person": 0,
    "chair": 56,
    "bottle": 39,
    "backpack": 24,
}


class YoloDetector:
    def __init__(self):
        rospy.init_node("yolo_detector", anonymous=True)

        # Load YOLOv8 model (auto-downloads first time)
        model_size = rospy.get_param("~model", "n")  # n=nano, s=small, m=medium
        self.model = YOLO(f"yolov8{model_size}.pt")
        rospy.loginfo(f"YOLOv8{model_size} model loaded. Classes: {len(self.model.names)}")

        # Bridge
        self.bridge = CvBridge()

        # Subscriber
        self.image_sub = rospy.Subscriber(
            "/berxel_base/color/image_raw", Image, self.image_callback
        )

        # Publisher for annotated image
        self.annotated_pub = rospy.Publisher(
            "/yolo/annotated", Image, queue_size=10
        )

        # Publisher for detections (simple text)
        self.detection_pub = rospy.Publisher(
            "/yolo/detections", String, queue_size=10
        )

        rospy.loginfo(f"Listening on /berxel_base/color/image_raw")
        rospy.loginfo(f"Publishing annotated image to /yolo/annotated")

    def image_callback(self, msg):
        try:
            # Convert ROS Image -> OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Run YOLO inference
            results = self.model(cv_image, verbose=False)[0]

            # Build detection text
            detections = []
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = self.model.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw bounding box
                label = f"{name} {conf:.2f}"
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    cv_image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )

                detections.append(f"{name}({conf:.2f})")

            # Publish annotated image
            annotated_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            annotated_msg.header = msg.header
            self.annotated_pub.publish(annotated_msg)

            # Publish detection text
            if detections:
                rospy.loginfo(f"Detected: {', '.join(detections)}")
                self.detection_pub.publish(
                    String(data=", ".join(detections))
                )

        except Exception as e:
            rospy.logerr(f"Detection error: {e}")

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    detector = YoloDetector()
    detector.run()
