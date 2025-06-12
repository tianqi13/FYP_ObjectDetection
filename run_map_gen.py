import sys
import os
import cv2
import time
import torch
import numpy as np
import open3d as o3d

sys.path.append(os.path.join(os.path.dirname(__file__), 'EfficientViTSAM'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'YOLO_world'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'DepthV2'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'StereoAnywhere'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'FoundationStereo'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'RoMa'))

from yolow_detect import ObjectDetector
from esam_seg import SegmentationModel
from process_keyframes import get_transformation_matrix
from stereo_disp import RoMaMatcher

# Configurations
# 1) for YOLO World
class_names=['bottle','cone','cup','rubiks cube','soda can','star','valve','weight','wooden cube']
score_thr = 0.5
nms_thr = 0.7
max_num_boxes = 100
# 2) for DAv2
depth_size = 720

# 3b) for RoMa
output_path = '/home/pro/Desktop/tianqi_FYP/FYP_ObjectDetection'
# 4) for EfficientViTSAM
colours = [
    (255,   0,   0),  # Red
    (  0, 255,   0),  # Lime/Green
    (  0,   0, 255),  # Blue
    (255, 255,   0),  # Yellow
    (255,   0, 255),  # Magenta
    (  0, 255, 255),  # Cyan
    (255, 165,   0),  # Orange
    (128,   0, 128),  # Purple
    (191, 255,   0),  # Chartreuse
    (  0, 128, 128)   # Teal
]

keyframe_txt = 'map_gen/keyframe_images.txt'
keyframe_L = 'map_gen/img_L_kp'
keyframe_R = 'map_gen/img_R_kp'
# Initialize models
detector = ObjectDetector(model_weights='finetuned', class_names=class_names)
# depth_estimator = DepthEstimator(model_config='vits')
segmenter = SegmentationModel(model_config='l0')
# stereo_model = DisparityEstimator(dtype=dtype)
# stereo_model = DisparityDetector(output_path, 'large')
matcher = RoMaMatcher(coarse_res=420, upsample_res=(720, 1280))

all_pcds = []

with open(keyframe_txt, 'r') as f:
    for idx, line in enumerate(f, start=1):
        print(f"Processing keyframe {idx}...")
        start_time = time.time()
        file_name, T = get_transformation_matrix(line)
        left_path  = os.path.join(keyframe_L, file_name)
        right_path = os.path.join(keyframe_R, file_name)
        left_image  = cv2.imread(left_path)
        right_image = cv2.imread(right_path)
    
        # Run Inference
        detections = detector.detect(left_image, max_num_boxes=max_num_boxes, score_thr=score_thr, nms_thr=nms_thr)
        if detections:
            boxes = [detection[0] for detection in detections]
            object_ids = [detection[2] for detection in detections]
            seg_masks = segmenter.segment_with_boxes(left_image, boxes)
            img_seg_masks = segmenter.overlay_masks(left_image, seg_masks, object_ids, colours)
        else:
            img_seg_masks = left_image.copy()
            

        # IF USING RoMa
        pred_disp = matcher.get_disparity(left_path, right_path, cert_thr=0.1)
        point_cloud = matcher.get_a_point_cloud(pred_disp, img_seg_masks, T, scale=1.0)
        print(f"Number of points in the point cloud: {len(np.asarray(point_cloud.points))}")
        all_pcds.append(point_cloud)
        
        end_time = time.time()
        print(f"Processed {file_name} in {end_time - start_time:.2f} seconds")

combined_pc = o3d.geometry.PointCloud()

for p in all_pcds:
    combined_pc += p

o3d.io.write_point_cloud(output_path+'/combined_pc.ply', combined_pc)

