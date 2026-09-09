import sys

from PIL.ImageMath import imagemath_convert
from shapely import boundary

from sympy import true
import torch
import numpy as np
import cv2
# import math
from collections import deque
from pyquaternion import Quaternion

sys.path.insert(0, './packages')
from nuscenes.nuscenes import NuScenes
from nuscenes_helper import get_bevfusion_dict, get_scene_samples
from bev_helper import (
    CLASS_NAMES,
    OBJECT_CLASSES,
    generate_bev_map
)

# --- LANE GRAPH IMPORTS ---
from lane_graph import (
    build_lane_compatibility_graph_from_vehicles,
    get_lane_streams,
    fit_lane_streams,                # NEW
    merge_compatible_lane_streams,   # NEW
    ensure_lead_vehicle_stream,      # NEW
    infer_lane_boundaries,           # NEW
    evaluate_lane_polynomial         # NEW (Needed to draw the curves)
)

from vehicle_tracking import accumulate_temporal_vehicle_evidence

from configs import(
    LaneGraphConfig,
    TemporalConfig,
    LeadVehicleConfig,
    LaneFitConfig,
    LaneMergeConfig,
    LaneBoundaryConfig,
    BoundaryTrackingConfig,
    RoadPlaneConfig,
    LaneProjectionConfig,
)

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
VIZ_MODE = True


# ==============================================================================
# PART 1: DATA EXTRACTION (Adapted for BEVFusion)
# ==============================================================================
def extract_vehicles_from_bevfusion(bboxes, scores, labels, cfg=None):
    """Converts BEVFusion 9-DoF arrays into the dictionary format lane_graph expects."""
    if cfg is None:
        cfg = LaneGraphConfig()

    vehicles = []
    
    # If tensors are returned, convert to numpy
    if torch.is_tensor(bboxes):
        bboxes = bboxes.cpu().numpy()
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()

    for i in range(len(bboxes)):
        score = float(scores[i])
        label = int(labels[i])
        
        # BEVFusion: [x, y, z, w, l, h, yaw, vx, vy]
        # Map BEVFusion (X=Right, Y=Forward, Z=Up) to FCOS3D Camera (X=Right, Y=Down, Z=Forward)
        x_val = float(bboxes[i, 0])     # Lateral (Right)
        y_val = float(bboxes[i, 1])     # Longitudinal (Forward)
        z_val = float(bboxes[i, 2])     # Vertical (Up)
        
        # lane_graph expects 'z' to be the forward depth, 'x' to be lateral
        z_depth = y_val 
        
        # if z_depth <= 0 or z_depth > cfg.max_depth:
        if abs(z_depth) > cfg.max_depth:
        
            continue

        passes_standard_score = score >= cfg.score_thresh
        passes_lead_score = (
            score >= cfg.lead_score_thresh
            and abs(x_val) <= cfg.lead_candidate_max_abs_x
        )
        if not (passes_standard_score or passes_lead_score):
            continue

        raw_yaw = float(bboxes[i, 6])
        fcos3d_yaw = raw_yaw + (np.pi / 2)

        vehicles.append({
            "original_idx": i,
            "x": x_val,              
            "y": -z_val,             # Invert Z-up to Y-down for FCOS3D height logic
            "z": z_depth,            
            "yaw": fcos3d_yaw, 
            "length": float(bboxes[i, 4]),
            "width": float(bboxes[i, 3]),
            "score": score,
            "evidence_weight": 1.0,
            "label": label,
            "below_standard_score": not passes_standard_score,
        })
    return vehicles


# ==============================================================================
# HELPER: DRAW GRAPH ON BEV MAP
# ==============================================================================
def draw_lane_graph_on_bev(canvas, graph, streams, pixels_per_meter):
    """Overlays the lane graph edges on the existing BEV map."""
    h, w, _ = canvas.shape
    center_x, center_y = w // 2, h // 2

    # Draw edges connecting vehicles
    for u, v, data in graph.edges(data=True):
        node_u = graph.nodes[u]
        node_v = graph.nodes[v]
        
        # Convert physical (x, z) back to image pixels
        img_u_x = int(center_x + (node_u['x'] * pixels_per_meter))
        img_u_y = int(center_y - (node_u['z'] * pixels_per_meter))
        
        img_v_x = int(center_x + (node_v['x'] * pixels_per_meter))
        img_v_y = int(center_y - (node_v['z'] * pixels_per_meter))
        
        # Draw line (Cyan color for graph connections)
        cv2.line(canvas, (img_u_x, img_u_y), (img_v_x, img_v_y), (255, 255, 0), 2, cv2.LINE_AA)

    return canvas

# ==============================================================================
# HELPER: DRAW BOUNDARIES ON BEV MAP
# ==============================================================================
def draw_lane_boundaries_on_bev(canvas, boundaries, pixels_per_meter):
    """Evaluates the mathematical polynomials and draws them on the BEV map."""
    h, w, _ = canvas.shape
    center_x, center_y = w // 2, h // 2

    for boundary in boundaries:
        # Get the start and end depth of the curve
        z_min = float(boundary["z_min"])
        z_max = float(boundary["z_max"])
        coeffs = boundary["coefficients"]

        # 1. Sample Z values (forward depth) along the boundary
        # If the curve is extremely short, skip it
        if z_max <= z_min:
            continue
        z_vals = np.linspace(z_min, z_max, num=50)
        
        # 2. Evaluate the polynomial to get X values (lateral position)
        x_vals = evaluate_lane_polynomial(coeffs, z_vals)

        # 3. Convert physical (x, z) meters to image pixels
        img_x = (center_x + (x_vals * pixels_per_meter)).astype(np.int32)
        img_y = (center_y - (z_vals * pixels_per_meter)).astype(np.int32)

        # Stack into OpenCV polyline format: shape (-1, 1, 2)
        pts = np.column_stack((img_x, img_y)).reshape((-1, 1, 2))

        # 4. Color coding based on boundary confidence/source
        is_provisional = boundary.get("source") == "single_stream"
        
        # Yellow for strong paired boundaries, Orange for provisional single-stream
        color = (0, 165, 255) if is_provisional else (0, 255, 255) 
        thickness = 2 if is_provisional else 4
        
        cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
        
    return canvas

# ==============================================================================
# HELPER: PROJECT BOUNDARIES & BOXES TO CAMERAS (BEVFUSION MATRICES)
# ==============================================================================
def project_scene_to_cameras(app, images, cam_paths, boundaries, bboxes, labels, inputs_json):
    """Draws 3D lane boundaries and 3D vehicle bounding boxes using BEVFusion's exact matrices."""
    
    intrins_list, sensor2keyegos_list = app.prepare_camera_inputs(cam_paths, inputs_json)
    
    # Calculate ground height for lane lines
    # lidar_z_offset = inputs_json.get("lidar2ego_translation")[2]
    # ground_z = -lidar_z_offset
    
    if torch.is_tensor(bboxes):
        bboxes = bboxes.cpu().numpy()
        labels = labels.cpu().numpy()

    lidar_z_offset = inputs_json.get("lidar2ego_translation", [0, 0, 1.84])[2]
    road_slope = 0.0
    road_intercept = -lidar_z_offset

    if len(bboxes) > 0:
        forward_dist = bboxes[:, 1]  # Forward distance of cars
        bottom_zs = bboxes[:, 2] - (bboxes[:, 5] / 2) # Height of car bottoms
        
        if len(bboxes) >= 2:
            # Fit a line (Z = slope * Y + intercept) to capture road tilt
            try:
                road_slope, road_intercept = np.polyfit(forward_dist, bottom_zs, 1)
            except np.linalg.LinAlgError:
                road_intercept = float(np.median(bottom_zs))
        else:
            # Only one car, fallback to flat median
            road_intercept = float(np.median(bottom_zs))
            
    processed_cams = {}
    
    for k, (name, pil_img) in enumerate(zip(cam_paths.keys(), images)):
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        s2k_inv = np.linalg.inv(sensor2keyegos_list[k])
        intrins = intrins_list[k]
        
        # ---------------------------------------------------------
        # 1. DRAW LANE BOUNDARIES
        # ---------------------------------------------------------
        for boundary in boundaries:
            z_min, z_max = float(boundary["z_min"]), float(boundary["z_max"])
            if z_max <= z_min:
                continue
                
            z_vals = np.linspace(z_min, z_max, num=100)
            x_vals = evaluate_lane_polynomial(boundary["coefficients"], z_vals)
            
            # --- NEW: Calculate the Z-height dynamically based on the road's slope ---
            lane_heights = (road_slope * z_vals) + road_intercept
            
            lidar_points = np.column_stack((x_vals, z_vals, lane_heights))
            lidar_points_hom = np.column_stack((lidar_points, np.ones_like(x_vals)))
            
            sensor_points = (lidar_points_hom @ s2k_inv.T)[:, :3]
            
            valid = sensor_points[:, 2] > 0.1
            if not np.any(valid):
                continue
            sensor_points = sensor_points[valid]
            
            projected = sensor_points @ intrins.T
            pixels = projected[:, :2] / projected[:, 2:3]
            pts = np.rint(pixels).astype(np.int32).reshape((-1, 1, 2))
            
            is_provisional = boundary.get("source") == "single_stream"
            color = (0, 165, 255) if is_provisional else (0, 255, 255) 
            cv2.polylines(cv_img, [pts], isClosed=False, color=color, thickness=8, lineType=cv2.LINE_AA)

        # ---------------------------------------------------------
        # 2. DRAW 3D BOUNDING BOXES
        # ---------------------------------------------------------
        for i in range(len(bboxes)):
            box = bboxes[i]
            label = labels[i]
            x, y, z, w, l, h, yaw = box[:7]
            
            # A. Generate the 8 corners centered at origin
            # Just like the BEV map: Length (l) on X-axis, Width (w) on Y-axis
            dx, dy, dz = l / 2, w / 2, h / 2
            
            x_corners = np.array([ dx,  dx, -dx, -dx,  dx,  dx, -dx, -dx])
            y_corners = np.array([ dy, -dy, -dy,  dy,  dy, -dy, -dy,  dy])
            z_corners = np.array([ dz,  dz,  dz,  dz, -dz, -dz, -dz, -dz])
            
            local_corners = np.vstack([x_corners, y_corners, z_corners])
            
            # B. Apply the exact same yaw correction as the BEV map
            # BEVFusion yaw: 0=Backward, Clockwise -> Standard theta: 0=Right, 90=Forward, Counter-Clockwise
            theta = -yaw - (np.pi / 2)
            c, s = np.cos(theta), np.sin(theta)
            
            R = np.array([
                [c, -s, 0],
                [s,  c, 0],
                [0,  0, 1]
            ])
            
            # Rotate and translate to actual location
            lidar_corners = (R @ local_corners).T + np.array([x, y, z])
            
            # C. Project through BEVFusion matrices
            lidar_corners_hom = np.column_stack((lidar_corners, np.ones(8)))
            sensor_corners = (lidar_corners_hom @ s2k_inv.T)[:, :3]
            
            # D. Ensure ALL 8 corners are in front of the camera
            if not np.all(sensor_corners[:, 2] > 0.1):
                continue
                
            projected = sensor_corners @ intrins.T
            pixels = (projected[:, :2] / projected[:, 2:3]).astype(np.int32)
            
            # E. Draw the lines that make up the box
            class_name = CLASS_NAMES[label] if label < len(CLASS_NAMES) else "car"
            box_color = OBJECT_CLASSES.get(class_name, (255, 255, 255))
            
            # 12 Edge Lines
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0), # Top
                (4, 5), (5, 6), (6, 7), (7, 4), # Bottom
                (0, 4), (1, 5), (2, 6), (3, 7)  # Pillars
            ]
            for start, end in edges:
                cv2.line(cv_img, tuple(pixels[start]), tuple(pixels[end]), box_color, 2, cv2.LINE_AA)
                
            # F. Draw an 'X' on the front face to indicate heading visually
            # Front face is corners 0, 1 (Top Front) and 4, 5 (Bottom Front)
            cv2.line(cv_img, tuple(pixels[0]), tuple(pixels[5]), (0, 0, 255), 2, cv2.LINE_AA) # Red X
            cv2.line(cv_img, tuple(pixels[1]), tuple(pixels[4]), (0, 0, 255), 2, cv2.LINE_AA)

        processed_cams[name] = cv_img
        
    return processed_cams

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
        class_filter=[0, 1, 2]
    )

    # nusc = NuScenes('v1.0-mini', "./nuscenes")
    # nusc = NuScenes('v1.0-trainval', "/home/gdtrinh/nuscenes")
    nusc = NuScenes('v1.0-trainval', "Z:/dataset/nuscenes")

    window_name = "BEVFusion Stream"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # --- PIPELINE CONFIG & HISTORY BUFFER ---
    temporal_cfg = TemporalConfig()
    graph_cfg = LaneGraphConfig()
    fit_cfg = LaneFitConfig()
    merge_cfg = LaneMergeConfig()
    lead_cfg = LeadVehicleConfig(enabled=true)
    boundary_cfg = LaneBoundaryConfig()

    history = deque(maxlen=temporal_cfg.history_frames)

    # Matrix mapping FCOS3D Camera coordinates back to Standard nuScenes Ego
    pseudo_cam_to_ego = np.array([
        [ 0,  0,  1,  0],  # Cam Z (Forward) -> Ego X
        [-1,  0,  0,  0],  # Cam X (Right) -> Ego -Y (Left)
        [ 0,  1,  0,  0],  # Cam Y (Down) -> Ego -Z (Up)
        [ 0,  0,  0,  1]
    ], dtype=np.float64)

    # for frame_id, sample in enumerate(nusc.sample):
    for frame_id, sample in enumerate(get_scene_samples(nusc, "scene-0095")):
    
        token = sample['token']

        # if token != "87e772078a494d42bd34cd16172808bc":
        #     continue

        info, images, cam_paths = get_bevfusion_dict(nusc, token)
        inputs_json = info

        # predict
        bboxes, scores, labels = app.predict_3d_boxes_from_images(
            images,
            cam_paths,
            inputs_json,
        )

        # --- PART 1: DATA EXTRACTION ---
        current_vehicles = extract_vehicles_from_bevfusion(bboxes, scores, labels, cfg=graph_cfg)

        # --- EGO-MOTION PREPARATION ---
        # Get ego2global from the sample inputs
        e2g_t = np.array(inputs_json["ego2global_translation"])
        e2g_r = Quaternion(inputs_json["ego2global_rotation"]).rotation_matrix
        ego2global = np.eye(4)
        ego2global[:3, :3] = e2g_r
        ego2global[:3, 3] = e2g_t
        
        # Combine to trick the tracker into handling our swapped axes perfectly
        pseudo_cam_to_global = ego2global @ pseudo_cam_to_ego

        history.append({
            "frame_index": frame_id,
            "vehicles": current_vehicles,
            "cam_to_global": pseudo_cam_to_global,
        })

        # --- PART 2: TEMPORAL TRACKING ---
        temporal_vehicles, tracks = accumulate_temporal_vehicle_evidence(
            list(history),
            reference_record_index=-1,
            cfg=temporal_cfg,
        )

        # --- PART 3: SPATIAL CLUSTERING (GRAPH) ---
        graph = build_lane_compatibility_graph_from_vehicles(
            temporal_vehicles,
            # current_vehicles,
            cfg=graph_cfg,
        )
        streams = get_lane_streams(graph, min_vehicles=2)

        # CURVE FITTING AND MERGING
        initial_fits = fit_lane_streams(graph, streams, cfg=fit_cfg)

        streams, lane_fits, merge_events = merge_compatible_lane_streams(
            graph, streams, lane_fits=initial_fits, merge_cfg=merge_cfg, fit_cfg=fit_cfg
        )

        streams, lane_fits, lead_diagnostic = ensure_lead_vehicle_stream(
            graph, streams, lane_fits, current_frame_index=0, cfg=lead_cfg
        )

        # LANE BOUDARY GENERATION
        boundaries = infer_lane_boundaries(lane_fits, cfg=boundary_cfg)

        # print(f"Sample {token} done | Vehicles: {len(current_vehicles)} | Temporal Tracks: {len(tracks)} | Lane Streams: {len(streams)}")
        print(f"Sample {token} done | Vehicles: {len(current_vehicles)} | Lane Streams: {len(streams)}")
                

        if VIZ_MODE:
            # 1. Base BEV Map with vehicle boxes
            bev_image_np = generate_bev_map(
                bboxes=bboxes,
                labels=labels,
                pixels_per_meter=PIXELS_PER_METER
            )
            
            # 2. Overlay the Lane Compatibility Graph edges
            bev_image_np = draw_lane_graph_on_bev(
                bev_image_np, 
                graph, 
                streams, 
                pixels_per_meter=PIXELS_PER_METER
            )

            bev_image_np = draw_lane_boundaries_on_bev(
                bev_image_np,
                boundaries,
                PIXELS_PER_METER
            )

            # 3. Map the PIL images to their camera names
            cam_names = list(cam_paths.keys())

            processed_cams_dict = project_scene_to_cameras(
                app,
                images,
                cam_paths,
                boundaries,
                bboxes,
                labels,
                inputs_json
            )

            # 4. Setup target dimensions to align the camera grid with the BEV map height
            bev_h, bev_w, _ = bev_image_np.shape
            cam_h = bev_h // 2  
            orig_w, orig_h = images[0].size 
            cam_w = int(cam_h * (orig_w / orig_h))

            # 5. Process each camera image
            processed_cams = {}
            # for name, pil_img in zip(cam_names, images):
            for name, pil_img in processed_cams_dict.items():
                # cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                cv_img = pil_img
                cv_img = cv2.resize(cv_img, (cam_w, cam_h))
                text = name.replace("CAM_", "").replace("_", " ")
                font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
                (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
                cv2.rectangle(cv_img, (10, 10), (10 + text_w + 20, 10 + text_h + 20), (0, 0, 0), -1)
                cv2.putText(cv_img, text, (20, 10 + text_h + 10), font, font_scale, (255, 255, 255), thickness)
                processed_cams[name] = cv_img

            # 6. Construct the grid
            row1 = np.hstack((processed_cams["CAM_FRONT_LEFT"], processed_cams["CAM_FRONT"], processed_cams["CAM_FRONT_RIGHT"]))
            row2 = np.hstack((processed_cams["CAM_BACK_LEFT"], processed_cams["CAM_BACK"], processed_cams["CAM_BACK_RIGHT"]))
            camera_grid = np.vstack((row1, row2))

            # 7. Concatenate and display
            final_combined_image = np.hstack((camera_grid, bev_image_np))
            cv2.imshow(window_name, final_combined_image)

            key = cv2.waitKey(1) & 0xFF  # Changed to 1ms to allow stream to play
            if key == ord("q"):
                print("User Stopped, Exiting")
                break
            elif key == ord('p'):
                print("Paused")
                cv2.waitKey(0)

if __name__ == "__main__":
    main()