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

def yaw_filter(
    bboxes,
    direction=np.pi,
    span=np.pi / 12,
):
    result = []

    opposite_direction = direction + np.pi

    for box in bboxes:
        yaw = float(box[6])

        forward_error = abs(
            (yaw - direction + np.pi)
            % (2.0 * np.pi)
            - np.pi
        )

        reverse_error = abs(
            (yaw - opposite_direction + np.pi)
            % (2.0 * np.pi)
            - np.pi
        )

        result.append(
            min(forward_error, reverse_error) <= span
        )

    return result

def vectorized_yaw_filter(
    bboxes: torch.Tensor | np.ndarray,
    direction: float = np.pi,
    span: float = np.pi / 12,
) -> torch.Tensor | np.ndarray:
    """Vectorised equivalent of bevfusion_adaptor.yaw_filter.

    For CUDA tensors, this stays entirely on the GPU and returns a CUDA boolean
    tensor.  It avoids the original Python loop's ``float(box[6])`` conversion,
    which synchronises the GPU once per detection.
    """
    if torch.is_tensor(bboxes):
        yaw = bboxes[:, 6]

        direction_t = yaw.new_tensor(direction)
        opposite_t = direction_t + yaw.new_tensor(np.pi)

        pi = yaw.new_tensor(np.pi)
        two_pi = yaw.new_tensor(2.0 * np.pi)

        forward_diff = (
            torch.remainder(
                yaw - direction_t + pi,
                two_pi,
            )
            - pi
        )

        reverse_diff = (
            torch.remainder(
                yaw - opposite_t + pi,
                two_pi,
            )
            - pi
        )

        return (
            (torch.abs(forward_diff) <= span)
            | (torch.abs(reverse_diff) <= span)
        )


    bboxes_np = np.asarray(bboxes)
    yaw = bboxes_np[:, 6]

    opposite_direction = direction + np.pi

    forward_diff = (
        np.remainder(
            yaw - direction + np.pi,
            2.0 * np.pi,
        )
        - np.pi
    )

    reverse_diff = (
        np.remainder(
            yaw - opposite_direction + np.pi,
            2.0 * np.pi,
        )
        - np.pi
    )

    return (
        (np.abs(forward_diff) <= span)
        | (np.abs(reverse_diff) <= span)
    )
