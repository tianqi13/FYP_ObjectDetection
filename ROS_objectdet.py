#!/usr/bin/env python3
"""
rosbag_detector_node.py
ROS node that re-uses the *exact* pipeline you provided:
  • ObjectDetector  (YOLO-World)
  • DepthEstimator  (VITs)
  • BBox3d          (your 3-D visualiser)
Publishes:
  /detector/annotated_image   sensor_msgs/Image (bgr8)
  /detector/depth_map         sensor_msgs/Image (mono8)
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import rospy
import rosbag
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

# ────────────────────────────────────────────────
# CONFIG – change these in-code, no cmd-line args
# ────────────────────────────────────────────────
BAG_PATH      = "/home/user/bags/my_session.bag"
IMAGE_TOPIC   = "/camera/color/image_raw"

DET_PERIOD    = 0.07     # seconds between detection runs
DEPTH_PERIOD  = 0.14     # seconds between depth runs

SCORE_THR     = 0.65
NMS_THR       = 0.50

CLASS_NAMES = [
    "bottle","cone","cup","rubiks cube","soda can",
    "star","valve","weight","wooden cube"
]

COLOURS = [              # B G R
    (255,  0,  0), (  0,255,  0), (  0,  0,255),
    (255,255,  0), (255,  0,255), (  0,255,255),
    (255,165,  0), (128,  0,128), (191,255,  0),
    (  0,128,128)
]
# ────────────────────────────────────────────────

# add local modules
ROOT = Path(__file__).resolve().parent
sys.path.extend([str(ROOT / "YOLO_world"), str(ROOT / "DepthV2")])

from yolow_detect import ObjectDetector
from depth import DepthEstimator
from bbox3d import BBox3d


class RosbagBBox3DNode:
    def __init__(self):
        rospy.init_node("rosbag_bbox3d_node", anonymous=True)

        self.pub_img   = rospy.Publisher("/detector/annotated_image",
                                         Image, queue_size=1)
        self.pub_depth = rospy.Publisher("/detector/depth_map",
                                         Image, queue_size=1)

        rospy.loginfo("Loading models …")
        self.detector  = ObjectDetector(model_weights="finetuned",
                                        class_names=CLASS_NAMES)
        self.depth_est = DepthEstimator(model_config="vits")

        self.bridge          = CvBridge()
        self.last_depth_map  = None
        self.last_depth_time = None
        self.last_det_time   = None

    # ───────────────────────────────────────────
    def run(self):
        if not os.path.exists(BAG_PATH):
            rospy.logerr(f"Bag {BAG_PATH} not found"); return

        bag = rosbag.Bag(BAG_PATH, "r")

        for topic, msg, t in bag.read_messages(topics=[IMAGE_TOPIC]):
            if rospy.is_shutdown():
                break

            try:
                frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            except Exception as e:
                rospy.logwarn(f"cv_bridge error: {e}"); continue

            now = t.to_sec()

            # ── DEPTH every DEPTH_PERIOD ─────────────────────────────
            if (self.last_depth_time is None) or \
               (now - self.last_depth_time >= DEPTH_PERIOD):
                self.last_depth_map  = self.depth_est.estimate_depth(
                    frame, input_size=720)
                self.last_depth_time = now

                depth_vis = self.depth_est.colorize_depth(self.last_depth_map)
                self.pub_depth.publish(self.bridge.cv2_to_imgmsg(
                    depth_vis, encoding="mono8"))

            # ── DETECTION every DET_PERIOD ───────────────────────────
            if (self.last_det_time is None) or \
               (now - self.last_det_time >= DET_PERIOD):
                detections = self.detector.detect(
                    frame, max_num_boxes=100,
                    score_thr=SCORE_THR, nms_thr=NMS_THR)

                boxes_3d = []
                depth_map = self.last_depth_map  # may still be None

                for det in detections:
                    bbox, score, cls_id, obj_id = det
                    colour = COLOURS[cls_id % len(COLOURS)]

                    depth_val = None
                    if depth_map is not None:
                        depth_val = self.depth_est.get_depth_from_bbox(
                            depth_map, bbox, method="median")

                    boxes_3d.append({
                        "bbox_2d":      bbox,
                        "depth_value":  depth_val,
                        "class_name":   CLASS_NAMES[cls_id],
                        "object_id":    obj_id,
                        "score":        score,
                        "color":        colour
                    })

                # draw 3-D cuboids (and optional 2-D outlines) onto image
                vis = BBox3d(boxes_3d, frame).draw_box_3d()

                self.pub_img.publish(self.bridge.cv2_to_imgmsg(
                    vis, encoding="bgr8"))
                self.last_det_time = now

        bag.close()
        rospy.loginfo("Finished rosbag.")


# ───────────────────────────────────────────────
if __name__ == "__main__":
    RosbagBBox3DNode().run()
