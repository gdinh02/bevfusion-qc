import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Polygon


@dataclass
class LaneGraphConfig:
    """Thresholds used to build the vehicle lane-compatibility graph."""

    score_thresh: float = 0.25
    max_depth: float = 60.0
    max_cross_track: float = 1.6
    max_yaw_diff_deg: float = 15.0
    max_along_track: float = 30.0
    sigma_cross_track: float = 0.8
    sigma_yaw_deg: float = 8.0


@dataclass
class LaneFitConfig:
    """Settings for robustly fitting a lane centreline x(z)."""

    degree: int = 2
    residual_threshold: float = 0.75
    max_trials: int = 200
    random_seed: int = 0


@dataclass
class LaneBoundaryConfig:
    """Settings for inferring lane boundaries between fitted lane streams."""

    min_overlap: float = 10.0
    min_lane_width: float = 2.5
    max_lane_width: float = 5.0
    sample_count: int = 50


@dataclass
class RoadPlaneConfig:
    """Settings for estimating camera-coordinate road height y(x, z)."""

    residual_threshold: float = 0.35
    max_trials: int = 200
    random_seed: int = 0


@dataclass
class LaneProjectionConfig:
    """Settings for projecting inferred lane boundaries into the camera image."""

    sample_count: int = 200
    min_depth: float = 1.0
    clip_to_image: bool = True


@dataclass
class TemporalConfig:
    """Settings for temporal ego-motion compensation and vehicle tracking."""

    history_frames: int = 5
    max_track_distance: float = 12.0
    max_track_yaw_diff_deg: float = 30.0
    max_track_frame_gap: int = 1
    temporal_decay: float = 0.90
    min_track_observations: int = 1
    max_reference_distance: float = 60.0
    max_time_gap_s: float = 3.0

def axial_angle_diff(a, b):
    """Return orientation difference in [0, pi/2], treating theta and theta+pi as equivalent."""
    diff = np.mod(np.abs(a - b), np.pi)
    return np.minimum(diff, np.pi - diff)


def heading_from_yaw(yaw):
    """Convert BEV yaw to a unit heading vector [x, z]."""
    return np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float64)

def pair_metrics(p_i, yaw_i, p_j, yaw_j):
    """Compute symmetric cross-track, yaw, and along-track differences for a vehicle pair."""
    p_i = np.asarray(p_i, dtype=np.float64)
    p_j = np.asarray(p_j, dtype=np.float64)
    delta = p_j - p_i

    h_i = heading_from_yaw(yaw_i)
    h_j = heading_from_yaw(yaw_j)
    n_i = np.array([-h_i[1], h_i[0]])
    n_j = np.array([-h_j[1], h_j[0]])

    cross_track = 0.5 * (
        abs(np.dot(n_i, delta))
        + abs(np.dot(n_j, -delta))
    )
    yaw_diff = axial_angle_diff(yaw_i, yaw_j)
    along_track = 0.5 * (
        abs(np.dot(h_i, delta))
        + abs(np.dot(h_j, -delta))
    )

    return {
        "cross_track": float(cross_track),
        "yaw_diff": float(yaw_diff),
        "along_track": float(along_track),
    }

def extract_vehicles_from_arrays(bboxes, labels, scores=None, vehicle_label_ids=None, cfg=None):
    """Extract vehicles from raw bounding box arrays instead of mmdet3d objects."""
    if cfg is None:
        cfg = LaneGraphConfig()
        
    # If no scores are provided, assume perfect confidence (1.0)
    if scores is None:
        scores = [1.0] * len(bboxes)
        
    if vehicle_label_ids is not None:
        vehicle_label_ids = set(vehicle_label_ids)

    vehicles = []

    for original_idx, box in enumerate(bboxes):
        score = float(scores[original_idx])
        label = int(labels[original_idx])

        # Filter by score and label
        if score < cfg.score_thresh:
            continue
        if vehicle_label_ids is not None and label not in vehicle_label_ids:
            continue

        # Unpack the first 7 elements (ignoring vx, vy if they exist)
        x, y, z, w, l, h, yaw = box[:7]

        # Filter out objects behind the camera or too far away
        if z <= 0 or z > cfg.max_depth:
            continue

        vehicles.append({
            "original_idx": original_idx,
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "yaw": float(yaw),
            "width": float(w),
            "length": float(l),
            "height": float(h),  # Saved for road plane estimation
            "score": score,
            "evidence_weight": 1.0,
            "label": label,
        })

    return vehicles

def build_lane_compatibility_graph_from_vehicles(vehicles, cfg=None):
    """Build a lane-compatibility graph from vehicle dictionaries."""
    if cfg is None:
        cfg = LaneGraphConfig()

    graph = nx.Graph()
    for node_id, vehicle in enumerate(vehicles):
        graph.add_node(node_id, **vehicle)

    max_yaw = math.radians(cfg.max_yaw_diff_deg)
    sigma_yaw = math.radians(cfg.sigma_yaw_deg)

    for i, vi in enumerate(vehicles):
        p_i = np.array([vi["x"], vi["z"]])

        for j in range(i + 1, len(vehicles)):
            vj = vehicles[j]
            p_j = np.array([vj["x"], vj["z"]])
            metrics = pair_metrics(p_i, vi["yaw"], p_j, vj["yaw"])

            cross = metrics["cross_track"]
            yaw_diff = metrics["yaw_diff"]
            along = metrics["along_track"]

            if (
                cross > cfg.max_cross_track
                or yaw_diff > max_yaw
                or along > cfg.max_along_track
            ):
                continue

            cross_term = (cross / cfg.sigma_cross_track) ** 2
            yaw_term = (yaw_diff / sigma_yaw) ** 2
            compatibility = np.exp(-0.5 * (cross_term + yaw_term))

            score_i = vi["score"] * vi.get("evidence_weight", 1.0)
            score_j = vj["score"] * vj.get("evidence_weight", 1.0)
            confidence = math.sqrt(max(score_i, 0.0) * max(score_j, 0.0))

            graph.add_edge(
                i,
                j,
                weight=float(compatibility * confidence),
                cross_track=cross,
                yaw_diff_deg=math.degrees(yaw_diff),
                along_track=along,
            )

    return graph

def build_lane_compatibility_graph_from_arrays(bboxes, labels, scores=None, vehicle_label_ids=None, cfg=None):
    """Build a graph directly from raw detection arrays."""
    vehicles = extract_vehicles_from_arrays(
        bboxes=bboxes,
        labels=labels,
        scores=scores,
        vehicle_label_ids=vehicle_label_ids,
        cfg=cfg,
    )
    graph = build_lane_compatibility_graph_from_vehicles(vehicles, cfg=cfg)
    return graph, vehicles

def get_lane_streams(graph, min_vehicles=2):
    """Return connected components large enough to be candidate lane streams."""
    return [
        sorted(component)
        for component in nx.connected_components(graph)
        if len(component) >= min_vehicles
    ]