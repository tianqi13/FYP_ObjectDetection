import cv2
import numpy as np
import torch
from depth_anything_v2.dpt import DepthAnythingV2
import os

class DepthEstimator:
    """
    Depth estimation using Depth Anything v2
    """
    def __init__(self, model_config='vits'):
        """
        Initialize the depth estimator
        
        Args:
            model_size (str): Model size ('vits', 'vitb', 'vitl')
        """
        HERE = os.path.dirname(__file__)
        
        model_weights = {
            'vits': os.path.join(HERE, 'checkpoints', 'depth_anything_v2_vits.pth'),
            'vitb': os.path.join(HERE, 'checkpoints', 'depth_anything_v2_vitb.pth'),
            'vitl': os.path.join(HERE, 'checkpoints', 'depth_anything_v2_vitl.pth')
        }
        WEIGHTS_PATH = model_weights.get(model_config.lower(), model_weights['vits'])

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }

        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

        self.depth_anything = DepthAnythingV2(**model_configs[model_config])
        self.depth_anything.load_state_dict(torch.load(WEIGHTS_PATH, map_location='cpu'))
        self.depth_anything = self.depth_anything.to(DEVICE).eval()   
        
    
    def estimate_depth(self, image, input_size, normalise=True):
        """
        Estimate depth from an image
        
        Args:
            image (numpy.ndarray): Input image (RGB format)
            input_size (int): Input size of image
        Returns:
            numpy.ndarray: Depth map (normalized to 0-1)
        """
        
        depth = self.depth_anything.infer_image(image, input_size)

        if normalise:
            # Normalize depth to 0-1
            depth = (depth - depth.min()) / (depth.max() - depth.min())
        
        return depth
    
    def colorize_depth(self, depth_map, cmap=cv2.COLORMAP_INFERNO):
        """
        Colorize depth map for visualization
        
        Args:
            depth_map (numpy.ndarray): Depth map (normalized to 0-1)
            cmap (int): OpenCV colormap
            
        Returns:
            numpy.ndarray: Colorized depth map (RGB format)
        """
        depth_map_uint8 = (depth_map * 255).astype(np.uint8)
        colored_depth = cv2.applyColorMap(depth_map_uint8, cmap)
        return colored_depth
    
    def get_depth_at_point(self, depth_map, x, y):
        """
        Get depth value at a specific point
        
        Args:
            depth_map (numpy.ndarray): Depth map
            x (int): X coordinate
            y (int): Y coordinate
            
        Returns:
            float: Depth value at (x, y)
        """
        if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
            return depth_map[y, x]
        
        return 0.0
    
    def get_depth_from_bbox(self, depth_map, bbox, method):
        """
        Get depth value in a region defined by a bounding box
        
        Args:
            depth_map (numpy.ndarray): Depth map
            bbox (list): Bounding box [x1, y1, x2, y2]
            method (str): Method to compute depth ('median', 'mean', 'min')
            
        Returns:
            float: Depth value in the region
        """
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Ensure coordinates are within image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(depth_map.shape[1] - 1, x2)
        y2 = min(depth_map.shape[0] - 1, y2)
        
        # Extract region
        region = depth_map[y1:y2, x1:x2]
        
        if region.size == 0:
            return 0.0
        
        # Compute depth based on method
        if method == 'median':
            return float(np.median(region))
        elif method == 'mean':
            return float(np.mean(region))
        elif method == 'min':
            return float(np.min(region))
        else:
            return float(np.median(region)) 
    
    def get_depth_all_bbox(self, depth_map, bboxes, method='median'):
        """
        Get depth values for all bounding boxes
        
        Args:
            depth_map (numpy.ndarray): Depth map
            bboxes (list): List of bounding boxes [[x1, y1, x2, y2], ...]
            method (str): Method to compute depth ('median', 'mean', 'min')
            
        Returns:
            list: Depth values for each bounding box
        """
        depths = []
        for bbox in bboxes:
            depth = self.get_depth_from_bbox(depth_map, bbox, method)
            depths.append(depth)
        
        return depths

if __name__ == "__main__":
    depth_estimator = DepthEstimator(model_config='vitb')
    image = cv2.imread('img_L.png')
    input_size = 672
    depth_map = depth_estimator.estimate_depth(image, input_size, normalise=True)
    colored_depth = depth_estimator.colorize_depth(depth_map)
    cv2.imwrite('img_L_depth.png', colored_depth)