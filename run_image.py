import sys
import os
import cv2
import time

sys.path.append(os.path.join(os.path.dirname(__file__), 'YOLO_world'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'DepthV2'))

from yolow_detect import ObjectDetector
from depth import DepthEstimator
from bbox3d import BBox3d

# ''' CHANGE CONFIGURATIONS IF NEEDED
path_to_image = 'img_L.png'
class_names = ['bottle','cone','cup','rubiks cube','soda can','star','valve','weight','wooden cube']
detector = ObjectDetector(model_weights='finetuned', class_names=class_names)   #can change between 'pretrained', 'finetuned' and 'prompt-tuned'
depth_estimator = DepthEstimator(model_config='vits')     
score_thr = 0.65 #reduce this if you want to detect more objects, but it will also increase false positives
nms_thr = 0.5                                   
# '''
 
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

# Load models
frame = cv2.imread(path_to_image) 
detections = detector.detect(frame, max_num_boxes=100, score_thr=score_thr, nms_thr=nms_thr)
boxes = [detection[0] for detection in detections]
depth_map = depth_estimator.estimate_depth(frame, input_size=367)

#3D Bounding Boxes
boxes_3d = []

for i, detection in enumerate(detections):
    try:
        bbox, score, class_id, obj_id = detection
    
        class_name = class_names[class_id]
    
        # Get depth in the region of the bounding box
        depth_value = depth_estimator.get_depth_from_bbox(depth_map, bbox, method='median')

        color = colours[class_id]

        box_3d = {
            'bbox_2d': bbox,
            'depth_value': depth_value,
            'class_name': class_name,
            'object_id': obj_id,
            'score': score,
            'color': color
        }
        # print(box_3d)
        boxes_3d.append(box_3d)
    
    except Exception as e:
        print(f"Error processing detection {detection}: {e}")
        continue

# Instantiate the BBox3D class
BBox3D_visualiser = BBox3d(boxes_3d, frame)
result_frame = BBox3D_visualiser.draw_box_3d()

cv2.imwrite('output_bbox_image.png', result_frame)





