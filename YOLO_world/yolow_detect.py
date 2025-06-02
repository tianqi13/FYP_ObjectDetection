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
    Object detection using YOLO World
    """
    def __init__(self, model_config='small', model_weights='untrained', class_names=None):
        """
        Initialize YOLO World
        
        Args:
            model_config (str): Model size ('small', 'medium')
            model_weights (str): Model weights ('pretrained', 'finetuned')
            class_names (list): List of class names to detect
        """
        HERE = os.path.dirname(__file__)
        
        model_config_paths = {
            'small':  os.path.join(HERE, 'configs', 'yolo_world_v2_s_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py'),
            'medium': os.path.join(HERE, 'configs', 'yolo_world_v2_m_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py')
        }
        
        CONFIG_PATH = model_config_paths.get(model_config, model_config_paths['small'])
        
        model_weight_paths = {
            'small': {
                'pretrained': os.path.join(HERE, 'weights', 'pre_train',  's_stage2-4466ab94.pth'),
                'finetuned':   os.path.join(HERE, 'weights', 'finetune',   's_finetuned.pth')
            },
            'medium': {
                'pretrained': os.path.join(HERE, 'weights', 'pre_train',  'm_stage1-7e1e5299.pth'),
                'finetuned':   os.path.join(HERE, 'weights', 'finetune',   'm_finetuned.pth')
            }
        }

        WEIGHTS_PATH = model_weight_paths.get(model_config, {}).get(model_weights, model_weight_paths['small']['pretrained'])
    
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

        self.model = init_detector(CONFIG_PATH, WEIGHTS_PATH, device=DEVICE)

        self.model.cfg.test_dataloader.dataset.pipeline[0].type = 'mmdet.LoadImageFromNDArray'
        self.test_pipeline = Compose(self.model.cfg.test_dataloader.dataset.pipeline)

        if class_names is None:
            self.class_names = []
        else:
            self.class_names = class_names
        
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
    
    def detect(self, input_frame, max_num_boxes=100, score_thr=0.5, nms_thr=0.5):
        """
        Detect objects in a frame and return detections 
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

    def detect_and_plot(self, input_frame, max_num_boxes=100, score_thr=0.5, nms_thr=0.5):
        """
        Detect objects in a frame and return annotated frame for printing 
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
    
    def get_class_names(self):
        """
        Get the names of the classes that the model can detect
        """
        return self.class_names

if __name__ == "__main__":
    # Example usage
    detector = ObjectDetector(model_config='small', model_weights='trained', class_names=['cone', 'cup', 'bottle'])
    image = cv2.imread('img_L.png')
    annotated_image, detections = detector.detect_and_plot(image, score_thr=0.2)
    cv2.imwrite('output_s.png', annotated_image)