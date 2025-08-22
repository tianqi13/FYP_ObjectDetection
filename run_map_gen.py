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
score_thr = 0.68
nms_thr = 0.5
max_num_boxes = 100
# 2) for DAv2
depth_size = 720
# 3b) for RoMa
output_path = '/home/pro/Desktop/tianqi_FYP/FYP_ObjectDetection/point_clouds'
# 4) for EfficientViTSAM
colours = [
    (255,  20, 147),  # Neon Pink
    ( 57, 255,  20),  # Electric Green
    (  0, 255, 255),  # Cyan / Neon Blue
    (255, 255,   0),  # Bright Yellow
    (255,  92,  51),  # Neon Orange
    (180,   0, 255),  # Vivid Purple
    (255,   0, 255),  # Magenta
    (  0, 255, 128),  # Spring Green / Aqua
    (  0, 255,  64),  # Lime Glow
    (  0, 128, 255),  # Vivid Sky Blue
]

keyframes = [0, 4, 5, 6, 8, 9, 10]

# Initialize models
detector = ObjectDetector(model_weights='finetuned', class_names=class_names)
segmenter = SegmentationModel(model_config='l0')
matcher = RoMaMatcher(coarse_res=420, upsample_res=(720, 1280))

for kf_num in keyframes:
    print(f"Processing keyframe set {kf_num}...")
    all_pcds = []
    keyframe_txt = f'/home/pro/Desktop/tianqi_FYP/ROS/keyframe_images{kf_num}.txt'
    keyframe_L = f'/home/pro/Desktop/tianqi_FYP/ROS/img_L_kp{kf_num}'
    keyframe_R = f'/home/pro/Desktop/tianqi_FYP/ROS/img_R_kp{kf_num}'

    with open(keyframe_txt, 'r') as f:
        for idx, line in enumerate(f, start=1):
            print(f"Processing keyframe {idx}...")
            file_name, T = get_transformation_matrix(line)
            left_path  = os.path.join(keyframe_L, file_name)
            right_path = os.path.join(keyframe_R, file_name)
            left_image  = cv2.imread(left_path) # BGR format
            right_image = cv2.imread(right_path)
        
            # Run Inference
            detections = detector.detect(left_image, max_num_boxes=max_num_boxes, score_thr=score_thr, nms_thr=nms_thr) # YOLO-World expects BGR images
            if detections:
                boxes = [detection[0] for detection in detections]
                object_ids = [detection[2] for detection in detections]
                seg_masks = segmenter.segment_with_boxes(left_image, boxes)
                img_seg_masks = segmenter.overlay_masks(left_image, seg_masks, object_ids, colours) # BGR
            else:
                img_seg_masks = left_image.copy() # BGR
                
            
            # IF USING RoMa
            pred_disp = matcher.get_disparity(left_path, right_path, cert_thr=0.3)
            point_cloud = matcher.get_a_point_cloud(pred_disp, img_seg_masks, T, scale=1.0)
            all_pcds.append(point_cloud)
            

    combined_pc = o3d.geometry.PointCloud()

    for p in all_pcds:
        combined_pc += p
    output_file = f'{output_path}/combined_pc{kf_num}.ply'
    o3d.io.write_point_cloud(output_file, combined_pc)

