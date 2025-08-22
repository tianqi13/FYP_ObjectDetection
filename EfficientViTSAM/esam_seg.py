import torch
import numpy as np
import cv2
import yaml
from efficientvit.sam_model_zoo import create_efficientvit_sam_model
from efficientvit.models.efficientvit.sam import EfficientViTSamAutomaticMaskGenerator, EfficientViTSamPredictor

class SegmentationModel:
    """
    Instance segmentation using EfficientSAM
    """
    def __init__(self, model_config="xl1", pred_iou_thresh=0.88, stability_score_thresh=0.95, min_mask_region_area=0):
        """
        Initialize the segmentation model
        
        Args:
            model_config (str): Model configuration ( 'l0', 'l1', 'l2', 'xl1') 
            pred_iou_thresh (float): IoU threshold for mask prediction
            stability_score_thresh (float): Stability score threshold for mask prediction
            min_mask_region_area (float): Minimum area for mask region
        """
        
        model_config_path = {
            'l0': 'efficientvit-sam-l0',
            'l1': 'efficientvit-sam-l1',
            'l2': 'efficientvit-sam-l2',
            'xl1': 'efficientvit-sam-xl1'
        }
        CONFIG_PATH = model_config_path.get(model_config.lower(), model_config_path['xl1'])
        
        # build model
        self.efficientvit_sam = create_efficientvit_sam_model(CONFIG_PATH, True, None).cuda().eval()
        self.efficientvit_sam_predictor = EfficientViTSamPredictor(self.efficientvit_sam)
        self.efficientvit_mask_generator = EfficientViTSamAutomaticMaskGenerator(
            self.efficientvit_sam,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            min_mask_region_area=min_mask_region_area,
            #**build_kwargs_from_config(opt, EfficientViTSamAutomaticMaskGenerator),
        )   
    
    def segment_with_boxes(self, image, boxes):
        """
        Generate segmentation masks for the given boxes
        
        Args:
            image (numpy.ndarray): Input image (RGB format)
            boxes (list): List of bounding boxes [[x1, y1, x2, y2], ...]
            
        Returns:
            results (list): List of binary masks
        """
        if len(boxes) == 0:
            return []

        # Set the image for the predictor
        self.efficientvit_sam_predictor.set_image(image)

        results = []
        for box in boxes:
            box_array = np.array([box])  # shape (1, 4)
            # Run prediction
            mask, _, _ = self.efficientvit_sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box_array,
                #TODO: check if toggling this to false might help
                multimask_output=False,
            )
            results.append(mask)

        return results
    
    def overlay_masks(self, frame, masks, object_ids, colours, alpha=0.9):
        """
        Overlay segmentation masks on frame with improved visibility
        
        Args:
            frame (numpy.ndarray): Input frame
            masks (list): List of dicts with key 'mask' containing boolean masks
            alpha (float): Transparency factor (lower value = more transparent masks)
            
        Returns:
            numpy.ndarray: Image with overlaid masks
        """
        out = frame.copy().astype(np.float32)

        for i, mask in enumerate(masks):
            # squeeze channel if needed
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask[0]

            # ensure boolean mask
            binary = mask.astype(bool)
            color = colours[object_ids[i]]
            color = (color[2], color[1], color[0])  # convert to BGR to match the BGR image
            
            # for each channel, blend only where mask is True
            for c in range(3):
                fg_val = color[c]
                bg_vals = out[..., c][binary]
                out[..., c][binary] = (1 - alpha) * bg_vals + alpha * fg_val

            # now draw the contours on the blended copy (they’ll go on top)
            # contours, _ = cv2.findContours(
            #     mask.astype(np.uint8),
            #     cv2.RETR_EXTERNAL,
            #     cv2.CHAIN_APPROX_SIMPLE
            # )
            # cv2.drawContours(out, contours, -1, color, 2)

        # convert back to uint8
        return out.clip(0, 255).astype(np.uint8)

# Example usage
if __name__ == "__main__":
    # Initialize model
    segmenter = SegmentationModel(model_config='l0')
    
    
    