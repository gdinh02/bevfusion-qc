import torch
import numpy as np
import cv2

OBJECT_CLASSES = {
    "car": (0, 158, 255),
    "truck": (71, 99, 255),
    "construction_vehicle": (70, 150, 233),
    "bus": (0, 69, 255),
    "trailer": (0, 140, 255),
    "barrier": (144, 128, 112),
    "motorcycle": (99, 61, 255),
    "bicycle": (60, 20, 220),
    "pedestrian": (230, 0, 0),
    "traffic_cone": (79, 79, 47),
}

# Ensure we have a list of keys to map the label integer to the class name
CLASS_NAMES = list(OBJECT_CLASSES.keys())

def generate_bev_map(bboxes: torch.Tensor, labels: torch.Tensor, max_range_meters=50, pixels_per_meter=10):
    """
    Draws a top-down BEV map using OpenCV.
    
    max_range_meters: How far to see in front/back/left/right (50m = 100x100m total area)
    pixels_per_meter: Resolution of the map (10px/m = 1000x1000 pixel image)
    """
    # 1. Setup the blank canvas
    img_size = int(max_range_meters * 2 * pixels_per_meter)
    canvas = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    
    center_x = img_size // 2
    center_y = img_size // 2
    
    # Draw Ego Vehicle in the center (assuming ~4m length, 2m width)
    ego_l, ego_w = int(4 * pixels_per_meter), int(2 * pixels_per_meter)
    cv2.rectangle(canvas, 
                  (center_x - ego_w//2, center_y - ego_l//2), 
                  (center_x + ego_w//2, center_y + ego_l//2), 
                  (255, 255, 255), -1)

    # Convert tensors to numpy arrays for easier math
    bboxes_np = bboxes.cpu().numpy()
    labels_np = labels.cpu().numpy()

    for i in range(len(bboxes_np)):
        box = bboxes_np[i]
        label = labels_np[i]
        
        # Extract attributes: [x, y, z, w, l, h, yaw, vx, vy]
        x, y = box[0], box[1]
        w, l = box[3], box[4]
        yaw = box[6]

        theta = -yaw - (np.pi / 2)
        
        # --- STEP 1: Create box corners at origin (0,0) ---
        # The car's length (l) runs along its local X-axis (forward)
        # The car's width (w) runs along its local Y-axis (left/right)
        dx = l / 2
        dy = w / 2
        
        # Front-left, Front-right, Rear-right, Rear-left
        corners = np.array([
            [ dx,  dy], 
            [ dx, -dy],
            [-dx, -dy],
            [-dx,  dy]
        ])
        
        # --- STEP 2: Apply rotation (yaw) and translation (x, y) ---
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], 
                      [s,  c]])
        
        # Rotate corners and move them to the object's physical location
        physical_corners = corners @ R.T + np.array([x, y])
        # physical_corners = corners + np.array([x, y])
        
        # --- STEP 3: Map physical meters to image pixels ---
        # In BEV Map: X points right, Y points forward (up)
        # In Image: X points right, Y points down
        # Therefore: Image X corresponds to +BEV X, Image Y corresponds to -BEV Y

        image_corners = np.zeros_like(physical_corners)
        image_corners[:, 0] = center_x + (physical_corners[:, 0] * pixels_per_meter) # X mapping (Direct)
        image_corners[:, 1] = center_y - (physical_corners[:, 1] * pixels_per_meter) # Y mapping (Inverted)
        
        image_corners = image_corners.astype(np.int32)
        
        # --- STEP 4: Render using OpenCV ---
        # Get color based on label index
        class_name = CLASS_NAMES[label] if label < len(CLASS_NAMES) else "car"
        color = OBJECT_CLASSES.get(class_name, (255, 255, 255))
        
        # Draw the bounding box
        cv2.polylines(canvas, [image_corners], isClosed=True, color=color, thickness=2)
        
        # Draw a line from the center to the front to indicate heading (yaw)
        front_midpoint = (image_corners[0] + image_corners[1]) // 2
        center_pixel = np.mean(image_corners, axis=0).astype(np.int32)
        cv2.line(canvas, tuple(center_pixel), tuple(front_midpoint), (0, 0, 255), 2)
        
    return canvas 