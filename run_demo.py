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
    ensure_lead_vehicle_stream,      # NEW
)

from lane_boundaries import infer_lane_boundaries

from lane_fitting import (
    evaluate_lane_polynomial,
    fit_lane_streams,
    merge_compatible_lane_streams,
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

from visualisation import(
    draw_lane_graph_on_bev,
    draw_lane_boundaries_on_bev,
    project_scene_to_cameras,
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