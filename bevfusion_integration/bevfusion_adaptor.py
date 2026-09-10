import torch
import numpy as np

from lane_inference.configs import (
    LaneGraphConfig
)

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

def yaw_filter(bboxes, direction = np.pi, span = np.pi/12):
    # filter_check = [False] * len(bboxes)

    # forward = (direction - span, direction + span)
    # backward = ((direction - np.pi) - span, (direction - np.pi) + span)

    # for i, box in enumerate(bboxes):
    #     yaw = box[6] % (2 * np.pi)
    #     if (yaw > forward[0] and yaw < forward[1]) or (yaw > backward[0] and yaw < backward[1]):
    #         filter_check[i] = True

    # return filter_check
    return [
        abs((float(box[6]) - direction + np.pi / 2) % np.pi - np.pi / 2)
        <= span
        for box in bboxes
    ]
