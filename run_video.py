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
source = "test.mp4"  
output_path = "output_bbox_video.mp4"  
class_names=['water bottle', 'cup', 'soda can', 'cone']
detector = ObjectDetector(model_config='small', model_weights='finetuned', class_names=class_names) #can change between 'small' and 'medium', 'pretrained' and 'finetuned'
depth_estimator = DepthEstimator(model_config='vits')                                               #can change between 'vits' and 'vitb' and 'vitl'
# '''

colours = [(204, 40, 40), (216, 138, 21), (183, 229, 0), (87, 242, 48), (25, 255, 117), (0, 204, 204), (43, 112, 216), (64, 22, 229), (193, 0, 242), (255, 50, 173)]
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {DEVICE}")
print(f"Opening video source: {source}")
cap = cv2.VideoCapture(source)

if not cap.isOpened():
    print(f"Error: Could not open video source {source}")
    sys.exit(1)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Initialize variables for FPS calculation
frame_count = 0
start_time = time.time()
fps_display = "FPS: --"

print("Starting processing...")
    
# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    #YOLO World for Object Detection
    detections = detector.detect(frame, max_num_boxes=100, score_thr=0.1, nms_thr=0.2)
    
    # Extract bounding boxes from detections
    boxes = [detection[0] for detection in detections]

    #DEPTHV2 for Depth Estimation
    depth_map = depth_estimator.estimate_depth(frame, input_size=720)

    #3D Bounding Boxes
    boxes_3d = []

    for i, detection in enumerate(detections):
        try:
            bbox, score, class_id, obj_id = detection
            
            # Get class name
            class_name = class_names[class_id]
            
            # Get depth in the region of the bounding box
            depth_value = depth_estimator.get_depth_from_bbox(depth_map, bbox, method='median')

            color = colours[class_id]
            # Create a simplified 3D box representation
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

    frame_count += 1
    if frame_count % 10 == 0:  # Update FPS every 10 frames
        end_time = time.time()
        elapsed_time = end_time - start_time
        fps_value = frame_count / elapsed_time
        fps_display = f"FPS: {fps_value:.1f}"
        print(fps_display)
        
    # Add FPS and device info to the result frame
    cv2.putText(result_frame, f"{fps_display}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    out.write(result_frame)

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Output video saved to {output_path}")



