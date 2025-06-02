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
class_names=['bottle', 'cup', 'soda can', 'cone']
detector = ObjectDetector(model_config='small', model_weights='finetuned', class_names=class_names)   #can change between 'pretrained' and 'finetuned'
depth_estimator = DepthEstimator(model_config='vits')                                               
# '''
 
colours = [(204, 40, 40), (216, 138, 21), (183, 229, 0), (87, 242, 48), (25, 255, 117), (0, 204, 204), (43, 112, 216), (64, 22, 229), (193, 0, 242), (255, 50, 173)]

# Load models
frame = cv2.imread(path_to_image) 
detections = detector.detect(frame, max_num_boxes=100, score_thr=0.05, nms_thr=0.2)
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





