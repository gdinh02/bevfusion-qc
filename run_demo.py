import sys
import torch
import numpy as np
import cv2
sys.path.insert(0, './packages')
from nuscenes.nuscenes import NuScenes
from nuscenes_helper import get_bevfusion_dict

import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_device(device)

from typing import cast

from BEVFusionAppCustom import (
    BEVFusionAppCustom,
    BEVFusionEncoder1Custom
)

from qai_hub_models.models.bevfusion_det.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    BEVFusion,
    BEVFusionDecoder,
    BEVFusionEncoder1,
    BEVFusionEncoder2,
    BEVFusionEncoder3,
)

PIXELS_PER_METER = 10

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
        y, x = box[0], box[1]
        w, l = box[3], box[4]
        yaw = box[6]
        
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
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s], 
                      [s,  c]])
        
        # Rotate corners and move them to the object's physical location
        physical_corners = corners @ R.T + np.array([x, y])
        
        # --- STEP 3: Map physical meters to image pixels ---
        # In BEVFusion: Ego X points forward, Ego Y points left
        # In Image: X points right, Y points down
        # Therefore: Image Y corresponds to -Ego X, Image X corresponds to -Ego Y
        image_corners = np.zeros_like(physical_corners)
        image_corners[:, 0] = center_x - (physical_corners[:, 1] * pixels_per_meter) # X mapping
        image_corners[:, 1] = center_y - (physical_corners[:, 0] * pixels_per_meter) # Y mapping
        
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

VIZ_MODE = True

def main(is_test: bool = False) -> None:
    # Load the model
    model = BEVFusion.from_pretrained("checkpoints/camera-only-det.pth")

    enc1 = model.encoder1
    enc2 = model.encoder2
    enc3 = model.encoder3
    dec = model.decoder
    heads = model.decoder.heads

    enc1_shape = enc1.get_input_spec()["imgs"][0]
    input_shape = (enc1_shape[-2], enc1_shape[-1])
    app = BEVFusionAppCustom(
        cast(BEVFusionEncoder1Custom, enc1),
        cast(BEVFusionEncoder2, enc2),
        cast(BEVFusionEncoder3, enc3),
        cast(BEVFusionDecoder, dec),
        num_classes=heads.num_classes,
        task_heads=heads.task_heads,
        get_bboxes=heads.get_bboxes,
        model_input_shape=input_shape,
        score_threshold=0.3,
        device="cuda",
        class_filter=[0]
    )

    nusc = NuScenes('v1.0-mini', "./nuscenes")

    window_name = "BEVFusion Stream"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # cv2.resizeWindow(window_name, 1920, 1080)

    for sample in nusc.sample:
        token = sample['token']

        info, images, cam_paths = get_bevfusion_dict(nusc, token)
        inputs_json = info

        # predict
        bboxes, scores, labels = app.predict_3d_boxes_from_images(
            images,
            cam_paths,
            inputs_json,
            # raw_output=True
        )

        print(f"Sample {token} done")

        if VIZ_MODE:
            bev_image_np = generate_bev_map(
                bboxes=bboxes,
                labels=labels,
                pixels_per_meter=PIXELS_PER_METER
            )

            # 1. Map the PIL images to their camera names
            cam_names = list(cam_paths.keys())

            # 2. Setup target dimensions to align the camera grid with the BEV map height
            bev_h, bev_w, _ = bev_image_np.shape
            cam_h = bev_h // 2  # 2 rows of cameras to exactly match BEV map height
            
            # Calculate proportional width to maintain aspect ratio
            orig_w, orig_h = images[0].size 
            cam_w = int(cam_h * (orig_w / orig_h))

            # 3. Process each camera image: convert to OpenCV, resize, and draw labels
            processed_cams = {}
            for name, pil_img in zip(cam_names, images):
                # Convert PIL RGB to OpenCV BGR
                cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                
                # Resize image
                cv_img = cv2.resize(cv_img, (cam_w, cam_h))
                
                # Format text (e.g., turn "CAM_FRONT_LEFT" into "FRONT LEFT")
                text = name.replace("CAM_", "").replace("_", " ")
                
                # Draw label with a semi-transparent or solid black background for visibility
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.0
                thickness = 2
                (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
                
                cv2.rectangle(cv_img, (10, 10), (10 + text_w + 20, 10 + text_h + 20), (0, 0, 0), -1)
                cv2.putText(cv_img, text, (20, 10 + text_h + 10), font, font_scale, (255, 255, 255), thickness)
                
                processed_cams[name] = cv_img

            # 4. Construct the 2x3 camera grid using NumPy stacking
            row1 = np.hstack((
                processed_cams["CAM_FRONT_LEFT"], 
                processed_cams["CAM_FRONT"], 
                processed_cams["CAM_FRONT_RIGHT"]
            ))
            row2 = np.hstack((
                processed_cams["CAM_BACK_LEFT"], 
                processed_cams["CAM_BACK"], 
                processed_cams["CAM_BACK_RIGHT"]
            ))
            camera_grid = np.vstack((row1, row2))

            # 5. Concatenate the camera grid and the BEV map side-by-side
            final_combined_image = np.hstack((camera_grid, bev_image_np))

            # final_combined_bgr = cv2.cvtColor(final_combined_image, cv2.COLOR_RGB2BGR)

            cv2.imshow(window_name, final_combined_image)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("User Stopped, Exitting")

            elif key == ord('p'):
                print("Paused")
                cv2.waitKey(0)

if __name__ == "__main__":
    main()