import numpy as np
import cv2

class BBox3d:
    def __init__(self, box_3d, image):
        """
        Initialize the BBox3d object with 3D bounding box parameters
        
        Args:
            box_3d (list): List of 3D bounding box parameters (dict)
            image (numpy.ndarray): Image to draw on
        """
        self.box_3d = box_3d
        self.image = image

    def draw_box_3d(self, thickness=2):
            """
            Draw enhanced 3D bounding box on image with better depth perception
            
            Args:
                thickness (int): Line thickness
            Returns:
                numpy.ndarray: Image with 3D box drawn
            """
            for box in self.box_3d:
                # Get 2D box coordinates
                x1, y1, x2, y2 = [int(coord) for coord in box['bbox_2d']]
                
                # Get depth value for scaling
                depth_value = box.get('depth_value', 0.5)
                
                # Calculate box dimensions
                width = x2 - x1
                height = y2 - y1
                
                # Calculate the offset for the 3D effect (deeper objects have smaller offset)
                # Inverse relationship with depth - closer objects have larger offset
                offset_factor = 1.0 - depth_value
                offset_x = int(width * 0.3 * offset_factor)
                offset_y = int(height * 0.3 * offset_factor)
                
                # Ensure minimum offset for visibility
                offset_x = max(15, min(offset_x, 50))
                offset_y = max(15, min(offset_y, 50))
                
                # Create points for the 3D box
                # Front face (the 2D bounding box)
                front_tl = (x1, y1)
                front_tr = (x2, y1)
                front_br = (x2, y2)
                front_bl = (x1, y2)
                
                # Back face (offset by depth)
                back_tl = (x1 + offset_x, y1 - offset_y)
                back_tr = (x2 + offset_x, y1 - offset_y)
                back_br = (x2 + offset_x, y2 - offset_y)
                back_bl = (x1 + offset_x, y2 - offset_y)
                
                # Create a slightly transparent copy of the image for the 3D effect
                overlay = self.image.copy()

                # Get color for the box
                color = box.get('color', (0, 255, 0))
                
                # Draw the front face (2D bounding box)
                cv2.rectangle(self.image, front_tl, front_br, color, thickness)
                
                # Draw the connecting lines between front and back faces
                cv2.line(self.image, front_tl, back_tl, color, thickness)
                cv2.line(self.image, front_tr, back_tr, color, thickness)
                cv2.line(self.image, front_br, back_br, color, thickness)
                cv2.line(self.image, front_bl, back_bl, color, thickness)
                
                # Draw the back face
                cv2.line(self.image, back_tl, back_tr, color, thickness)
                cv2.line(self.image, back_tr, back_br, color, thickness)
                cv2.line(self.image, back_br, back_bl, color, thickness)
                cv2.line(self.image, back_bl, back_tl, color, thickness)
                
                # Fill the top face with a semi-transparent color to enhance 3D effect
                pts_top = np.array([front_tl, front_tr, back_tr, back_tl], np.int32)
                pts_top = pts_top.reshape((-1, 1, 2))
                cv2.fillPoly(overlay, [pts_top], color)
                
                # Fill the right face with a semi-transparent color
                pts_right = np.array([front_tr, front_br, back_br, back_tr], np.int32)
                pts_right = pts_right.reshape((-1, 1, 2))
                
                # Darken the right face color for better 3D effect
                right_color = (int(color[0] * 0.7), int(color[1] * 0.7), int(color[2] * 0.7))
                cv2.fillPoly(overlay, [pts_right], right_color)
                
                # Apply the overlay with transparency
                alpha = 0.3  # Transparency factor
                cv2.addWeighted(overlay, alpha, self.image, 1 - alpha, 0, self.image)
                
                # Get class name and object ID
                class_name = box['class_name']
                obj_id = box['object_id'] if 'object_id' in box else None
                
                # Draw text information
                text_y = y1 - 10
                if obj_id is not None:
                    cv2.putText(self.image, f"ID:{obj_id}", (x1, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    text_y -= 15
                
                cv2.putText(self.image, class_name, (x1, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                text_y -= 15
                
                # Get depth information if available
                if 'depth_value' in box:
                    depth_value = box['depth_value']
                    depth_text = f"D:{depth_value:.2f}"
                    cv2.putText(self.image, depth_text, (x1, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    text_y -= 15
                
                # Get score if available
                if 'score' in box:
                    score = box['score']
                    score_text = f"S:{score:.2f}"
                    cv2.putText(self.image, score_text, (x1, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            return self.image
        