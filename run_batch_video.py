import sys
import os
import cv2
import time
import queue
import threading

sys.path.append(os.path.join(os.path.dirname(__file__), 'EfficientViTSAM'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'YOLO_world'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'DepthV2'))

from yolow_detect import ObjectDetector
from depth import DepthEstimator
from bbox3d import BBox3d

# WORKERS
def depth_worker(capture, depth_q, depth_estimator, depth_batch_size, input_size, depth_interval):
    """
    1) Read frames from capture
    2) Batch frames into `depth_batch_size` groups
    3) Within each batch, compute depth only every `depth_interval` frames,
       and reuse that map for the intermediate frames
    4) Enqueue (frame_id, frame, depth_map)

    Args:
        capture (cv2.VideoCapture): video source
        depth_q (queue.Queue): put (frame_id, frame, depth_map)
        depth_estimator: has `estimate_depth_batch(images, input_size)`
        depth_batch_size (int): number of frames per outer batch
        input_size (int): size for depth preprocessing
        depth_interval (int): compute depth every N frames
    """
    buffer = []
    frame_ids = []
    idx = 0
    total_frames = 0
    total_time = 0.0

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        buffer.append(frame)
        frame_ids.append(idx)
        idx += 1

        # process each outer batch
        if len(buffer) == depth_batch_size*depth_interval:
            # select every Nth frame for depth computation
            select_frames = buffer[::depth_interval]
            # time the depth batch compute
            t0 = time.time()
            depth_maps = depth_estimator.estimate_depth_batch(select_frames, input_size)
            t1 = time.time()

            batch_time = t1 - t0
            total_time += batch_time
            total_frames += len(buffer)

            # assign depth_map to each frame in buffer
            num_select = len(depth_maps)
            for i, (fid, frm) in enumerate(zip(frame_ids, buffer)):
                grp = min(i // depth_interval, num_select - 1)
                depth_q.put((fid, frm, depth_maps[grp]))

            buffer.clear()
            frame_ids.clear()

    # flush any remaining frames
    if buffer:
        select_frames = buffer[::depth_interval]
        t0 = time.time()
        depth_maps = depth_estimator.estimate_depth_batch(select_frames, input_size)
        t1 = time.time()

        total_time += (t1 - t0)
        total_frames += len(buffer)

        num_select = len(depth_maps)
        for i, (fid, frm) in enumerate(zip(frame_ids, buffer)):
            grp = min(i // depth_interval, num_select - 1)
            depth_q.put((fid, frm, depth_maps[grp]))

        buffer.clear()
        frame_ids.clear()

    # signal completion
    depth_q.put(None)

    # report timing
    if total_frames > 0:
        print(f"[Depth] Processed {total_frames} frames in {total_time:.2f}s "
              f"(avg {total_time/total_frames:.3f}s/frame)")


def detection_worker(depth_q, detect_q, detector, batch_size, score_thr=0.3, nms_thr=0.7):
    """
    1) consume (frame_id, frame, depth_map) from depth_q
    2) buffer until batch_size
    3) run object detection on the batch of frames
    4) merge detections + depth_map into metadata and enqueue to detect_q

    Args:
        depth_q (queue.Queue): yields (frame_id, frame, depth_map)
        detect_q (queue.Queue): queue to put (frame_id, frame, metadata)
        detector (ObjectDetector): has inference_detector(images) -> detections
        batch_size (int): number of frames per detection batch
    """
    frame_buffer = []
    depth_buffer = []
    id_buffer = []
    det_frames = 0
    det_time = 0.0

    while True:
        item = depth_q.get()
        if item is None:
            break
        frame_id, frame, depth_map = item
        id_buffer.append(frame_id)
        frame_buffer.append(frame)
        depth_buffer.append(depth_map)

        # once we have a batch, run detection
        if len(frame_buffer) == batch_size:
            t0 = time.time()
            dets_batch = detector.inference_detector_batch(frame_buffer, score_thr=score_thr, nms_thr=nms_thr)
            t1 = time.time()
            det_time += (t1 - t0)
            det_frames += len(frame_buffer)
            for fid, frm, dets, depth in zip(id_buffer, frame_buffer, dets_batch, depth_buffer):
                meta = {
                    'detections': dets,
                    'depth_map': depth
                }
                detect_q.put((fid, frm, meta))
            frame_buffer.clear()
            depth_buffer.clear()
            id_buffer.clear()

    # run any remaining frames
    if frame_buffer:
        t0 = time.time()
        dets_batch = detector.inference_detector(frame_buffer, score_thr=score_thr, nms_thr=nms_thr)
        t1 = time.time()
        det_time += (t1 - t0)
        det_frames += len(frame_buffer)
        for fid, frm, dets, depth in zip(id_buffer, frame_buffer, dets_batch, depth_buffer):
            meta = {
                'detections': dets,
                'depth_map': depth
            }
            detect_q.put((fid, frm, meta))
        frame_buffer.clear()
        depth_buffer.clear()
        id_buffer.clear()

    # signal completion
    detect_q.put(None)
    if det_frames > 0:
        print(f"[Detect] Processed {det_frames} frames in {det_time:.2f}s "
        f"(avg {det_time/det_frames:.3f}s/frame)")

def render_worker(detect_q, output_path, depth_estimator, fps, frame_size, class_names, colours):
    """
    1) consume (frame, metadata)
    2) draw boxes and depth overlay
    3) write to VideoWriter
    """
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        frame_size
    )

    frame_count = 0
    start_time = time.time()
    fps_display = "FPS: --"
    print("Starting processing...")
    render_time = 0.0

    while True:
        item = detect_q.get()
        if item is None:
            break
        fid, frame, meta = item

        # draw detections
        t0 = time.time()
        boxes_3d = []
        for detection in meta['detections']:
            try:
                bbox, score, class_id, obj_id = detection
                
                # Get class name
                class_name = class_names[class_id]
                
                # Get depth in the region of the bounding box
                depth_map = meta['depth_map']
                depth_value = depth_estimator.get_depth_from_bbox(depth_map, bbox, method='median')

                color = colours[class_id]
                color = (color[2], color[1], color[0])  # convert to BGR for OpenCV
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
        t1 = time.time()
        render_time += (t1 - t0)

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
        writer.write(result_frame)

    writer.release()
    if frame_count > 0:
        print(f"[Render] Processed {frame_count} frames in {render_time:.2f}s "
        f"(avg {render_time/frame_count:.3f}s/frame)")

if __name__ == "__main__":
    source = "test.mp4"  # Path to input video file
    output_path = "output.mp4"  # Path to output video file
    class_names=['bottle','cone','cup','rubiks cube','soda can','star','valve','weight','wooden cube']
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
    score_thr = 0.5
    nms_thr = 0.7
    detection_batch_size = 4  # Detection is faster than depth estimation, so use a smaller batch size than depth to reduce bottleneck
    depth_input_size = 720 # Smaller size for faster processing, but less accurate. Adjust as needed
    depth_interval = 4 # Compute depth every N frames to reduce load
    depth_batch_size = 8

    # Initialise models
    detector = ObjectDetector(model_weights='finetuned', class_names=class_names)
    depth_estimator = DepthEstimator(model_config='vits')

    # Open video source
    print(f"Opening video source: {source}")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        sys.exit(1)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialise queues
    depth_q  = queue.Queue(maxsize=depth_batch_size*2)
    detect_q = queue.Queue(maxsize=detection_batch_size*2)

    # threads
    t1 = threading.Thread(target=depth_worker,
                          args=(cap, depth_q, depth_estimator, depth_batch_size, depth_input_size, depth_interval),
                          daemon=True)
    t2 = threading.Thread(target=detection_worker,
                          args=(depth_q, detect_q, detector, detection_batch_size, score_thr, nms_thr),
                          daemon=True)
    t3 = threading.Thread(target=render_worker,
                          args=(detect_q, output_path, depth_estimator, fps, (width,height), class_names, colours),
                          daemon=True)

    # start & join
    t1.start()
    t2.start()
    t3.start()
    t1.join()
    t2.join()
    t3.join()

    cap.release()
    print("Processing complete! Saved to", output_path)



    