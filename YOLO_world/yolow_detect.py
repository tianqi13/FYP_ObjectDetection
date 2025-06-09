import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.runner import Runner
from mmengine.runner.amp import autocast
from mmyolo.registry import VISUALIZERS
from torchvision.ops import nms
import PIL.Image
import cv2
import supervision as sv
import os
from mmdet.apis import init_detector
from mmyolo.registry import VISUALIZERS

class ObjectDetector:
    """
    Object detection using YOLO-World
    """
    def __init__(self, model_weights='finetuned', class_names=None):
        """
        Initialize YOLO World
        
        Args:
            model_config (str): Model size ('small', 'medium', 'large', 'extra')
            model_weights (str): Model weights ('untrained', 'trained')
            class_names (list): List of class names to detect
        """
        HERE = os.path.dirname(__file__)
        
        CONFIG_PATH = os.path.join(HERE, 'configs', 'large.py')
        
        model_weight_paths = {
            'pretrained': os.path.join(HERE, 'weights', 'pre_train', 'l_stage2-b3e3dc3f.pth'),
            'finetuned': os.path.join(HERE, 'weights', 'finetune', 'l_finetuned.pth')
            'prompt-tuned': os.path.join(HERE, 'weights', 'prompt_tune', 'l_prompt_tuned.pth')
        }

        WEIGHTS_PATH = model_weight_paths.get(model_weights, model_weight_paths['finetuned'])
    
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.model = init_detector(CONFIG_PATH, WEIGHTS_PATH, device=DEVICE)
        self.test_pipeline = Compose(self.model.cfg.test_pipeline)

        if class_names is None:
            self.class_names = []
        else:
            self.class_names = class_names
        n = len(self.class_names)

        self.model.cfg.model.bbox_head.head_module.num_classes = n
        self.model.cfg.model.train_cfg.assigner.num_classes = n
        
        self.texts = [[t.strip()] for t in class_names] + [[" "]]
        self.model.reparameterize(self.texts)

        self.model.dataset_meta = {
            'classes': self.class_names
        }


    def inference_detector(self, image, max_num_boxes=100, score_thr=0.5, nms_thr=0.5):
        """
        Inference the model on the image
        Args:
            image (numpy.ndarray): Input image
            max_num_boxes (int): Maximum number of boxes to detect
            score_thr (float): Score threshold for detection
            nms_thr (float): NMS threshold for detection
        Returns:
            output (dict): Model output
            detections (list): List of detected objects with bounding boxes, scores, class IDs, and object IDs
        """
        
        data_info = dict(img_id=0, img=image, texts=self.texts)
        data_info = self.test_pipeline(data_info)
        data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                          data_samples=[data_info['data_samples']])

        with torch.no_grad():
            output = self.model.test_step(data_batch)[0]
            pred_instances = output.pred_instances
            pred_instances = pred_instances[pred_instances.scores > score_thr]

        keep_idxs = nms(pred_instances.bboxes, pred_instances.scores, iou_threshold=nms_thr)
        pred_instances = pred_instances[keep_idxs]

        if len(pred_instances.scores) > max_num_boxes:
            topk_indices = pred_instances.scores.topk(max_num_boxes).indices
            pred_instances = pred_instances[topk_indices]

        output.pred_instances = pred_instances

        detections = []
        bboxes = pred_instances.bboxes
        scores = pred_instances.scores
        labels = pred_instances.labels
        object_ids = getattr(pred_instances, 'object_ids', None)

        for i in range(len(bboxes)):
            bbox = bboxes[i].tolist()
            score = float(scores[i])
            class_id = int(labels[i])
            object_id = int(object_ids[i]) if object_ids is not None else None
            detections.append([bbox, score, class_id, object_id])

        return output, detections

    def detect_and_plot(self, input_frame, max_num_boxes=100, score_thr=0.5, nms_thr=0.5):
        """
        Detect objects in a frame and outputs a numpy array of the annotated image
        Args:
            input_frame (numpy.ndarray): Input image frame
            max_num_boxes (int): Maximum number of boxes to detect
            score_thr (float): Score threshold for detection
            nms_thr (float): NMS threshold for detection
        Returns:
            annotated_image (numpy.ndarray): Annotated image with detected objects
            detections (list): List of detected objects with bounding boxes, scores, class IDs, and object IDs
        """
        result, detections = self.inference_detector(input_frame,
                                                     max_num_boxes=max_num_boxes,
                                                     score_thr=score_thr,
                                                     nms_thr=nms_thr)

        visualizer = VISUALIZERS.build(self.model.cfg.visualizer)
        
        visualizer.dataset_meta = self.model.dataset_meta

        visualizer.add_datasample(name='video',
                                       image=input_frame,
                                       data_sample=result,
                                       draw_gt=False,
                                       show=False,
                                       pred_score_thr=score_thr)

        annotated_image = visualizer.get_image()

        return annotated_image, detections
    
    def detect(self, input_frame, max_num_boxes=100, score_thr=0.5, nms_thr=0.5):
        """
        Detect objects in a frame
        Args:
            input_frame (numpy.ndarray): Input image frame
            max_num_boxes (int): Maximum number of boxes to detect
            score_thr (float): Score threshold for detection
            nms_thr (float): NMS threshold for detection
        Returns:
            annotated_image (numpy.ndarray): Annotated image with detected objects
            detections (list): List of detected objects with bounding boxes, scores, class IDs, and object IDs
        """
        result, detections = self.inference_detector(input_frame,
                                                     max_num_boxes=max_num_boxes,
                                                     score_thr=score_thr,
                                                     nms_thr=nms_thr)

        return detections


    def get_class_names(self):
        """
        Get the names of the classes that the model can detect
        """
        return self.class_names

if __name__ == "__main__":
    # Example usage
    detector = ObjectDetector(model_weights='finetuned', class_names=['cone', 'bottle', 'cup'])
    image = cv2.imread('img_L.png')
    annotated_image, detections = detector.detect_and_plot(image, score_thr=0.2)
    cv2.imwrite('output.png', annotated_image)
    classes = detector.get_class_names()