#!/usr/bin/env python3
"""
YOLOv8 检测节点 (ROS1)
  - 加载 models/best.pt
  - 订阅摄像头话题，执行推理
  - 将检测到的物体名称以逗号分隔的字符串发布到 /yolo/detections
  - 同时发布带标注框的图像到 /yolo/image_result
"""

import rospy
import os
import time
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# 将 ultralytics 的导入放在 try 中，方便检查依赖
try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False


class YoloDetector:
    def __init__(self):
        # ---- 参数 ----
        model_name = rospy.get_param("~model_name", "best.pt")
        self.conf_thresh = rospy.get_param("~conf_threshold", 0.5)
        self.infer_every_n = rospy.get_param("~infer_every_n", 3)   # 每 N 帧推理一次
        self.publish_image = rospy.get_param("~publish_image", True)
        self.camera_topic = rospy.get_param("~camera_topic", "/base_camera/image_raw")

        # ---- 模型路径 ----
        pkg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        self.model_path = os.path.join(pkg_dir, "models", model_name)

        if not os.path.exists(self.model_path):
            rospy.logerr("[YOLO] Model not found: %s" % self.model_path)
            rospy.logerr("[YOLO] Please place best.pt in bobac3_test/models/")
            raise FileNotFoundError(self.model_path)

        if not HAS_YOLO:
            rospy.logerr("[YOLO] ultralytics not installed!  Run: pip install ultralytics")
            raise ImportError("ultralytics")

        rospy.loginfo("[YOLO] Loading model: %s" % self.model_path)
        self.model = YOLO(self.model_path)
        rospy.loginfo("[YOLO] Model loaded. Classes: %s" % self.model.names)

        # ---- cv_bridge ----
        self.bridge = CvBridge()

        # ---- 状态 ----
        self.frame_count = 0
        self._last_result = ""

        # ---- 订阅 & 发布 ----
        self.image_sub = rospy.Subscriber(self.camera_topic, Image, self.image_cb, queue_size=1)
        self.detection_pub = rospy.Publisher("/yolo/detections", String, queue_size=10)
        if self.publish_image:
            self.image_pub = rospy.Publisher("/yolo/image_result", Image, queue_size=1)

        rospy.loginfo("[YOLO] Ready. Listening on %s" % self.camera_topic)

    def image_cb(self, msg):
        self.frame_count += 1
        if self.frame_count % self.infer_every_n != 0:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn("[YOLO] cv_bridge error: %s" % e)
            return

        t0 = time.time()
        results = self.model(cv_image, conf=self.conf_thresh, verbose=False)
        dt = time.time() - t0

        # 汇总所有检测到的类别名称
        detected_names = []
        annotated = cv_image

        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    name = self.model.names.get(cls_id, "unknown")
                    detected_names.append(name)

            # 绘制标注框（如有需要）
            if self.publish_image:
                annotated = r.plot()

        # 去重，保持顺序
        seen = set()
        unique_names = []
        for n in detected_names:
            if n not in seen:
                seen.add(n)
                unique_names.append(n)

        detection_str = ", ".join(unique_names) if unique_names else ""
        self._last_result = detection_str

        # 发布
        self.detection_pub.publish(String(data=detection_str))

        if self.publish_image:
            try:
                img_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
                img_msg.header = msg.header
                self.image_pub.publish(img_msg)
            except Exception as e:
                rospy.logwarn("[YOLO] publish image error: %s" % e)

        rospy.loginfo("[YOLO] frame=%d  detections=[%s]  dt=%.2fms" % (
            self.frame_count, detection_str, dt * 1000))


if __name__ == '__main__':
    rospy.init_node('yolo_detector')
    try:
        node = YoloDetector()
        rospy.spin()
    except Exception as e:
        rospy.logfatal("[YOLO] Init failed: %s" % e)
        rospy.signal_shutdown(str(e))
