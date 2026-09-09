import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Polygon

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

from road_plane import (
    road_plane_y,
    estimate_road_plane,
    transform_road_plane_to_camera,
)

from geometry import (
    axial_angle_diff,
    heading_from_yaw,
    pair_metrics,
)

from lane_fitting import (
    evaluate_lane_polynomial,
    lane_polynomial_slope,
)

def extract_vehicles_from_prediction(pred_instances_3d, vehicle_label_ids, cfg=None):
    if cfg is None:
        cfg = LaneGraphConfig()

    boxes = pred_instances_3d.bboxes_3d
    scores = pred_instances_3d.scores_3d.detach().cpu()
    labels = pred_instances_3d.labels_3d.detach().cpu()
    bev = boxes.bev.detach().cpu().numpy()

    if hasattr(boxes, "gravity_center"):
        centres = boxes.gravity_center.detach().cpu().numpy()
    else:
        centres = np.column_stack(
            [bev[:, 0], np.zeros(len(bev), dtype=np.float64), bev[:, 1]]
        )

    vehicle_label_ids = set(vehicle_label_ids)
    vehicles = []

    for original_idx in range(len(boxes)):
        score = float(scores[original_idx])
        label = int(labels[original_idx])
        if label not in vehicle_label_ids:
            continue

        x = float(bev[original_idx, 0])
        z = float(bev[original_idx, 1])
        if z <= 0 or z > cfg.max_depth:
            continue

        passes_standard_score = score >= cfg.score_thresh
        passes_lead_score = (
            score >= cfg.lead_score_thresh
            and abs(x) <= cfg.lead_candidate_max_abs_x
        )
        if not (passes_standard_score or passes_lead_score):
            continue

        vehicles.append(
            {
                "original_idx": original_idx,
                "x": x,
                "y": float(centres[original_idx, 1]),
                "z": z,
                "yaw": float(bev[original_idx, 4]),
                "length": float(bev[original_idx, 2]),
                "width": float(bev[original_idx, 3]),
                "score": score,
                "evidence_weight": 1.0,
                "label": label,
                "below_standard_score": not passes_standard_score,
            }
        )

    return vehicles


def build_lane_compatibility_graph_from_vehicles(vehicles, cfg=None):
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

            compatibility = np.exp(
                -0.5
                * (
                    (cross / cfg.sigma_cross_track) ** 2
                    + (yaw_diff / sigma_yaw) ** 2
                )
            )
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


def build_lane_compatibility_graph(pred_instances_3d, vehicle_label_ids, cfg=None):
    vehicles = extract_vehicles_from_prediction(
        pred_instances_3d, vehicle_label_ids, cfg=cfg
    )
    return build_lane_compatibility_graph_from_vehicles(vehicles, cfg=cfg), vehicles



def get_lane_streams(graph, min_vehicles=2):
    return [
        sorted(component)
        for component in nx.connected_components(graph)
        if len(component) >= min_vehicles
    ]


def print_graph_edges(graph):
    for i, j, data in graph.edges(data=True):
        print(
            f"{i:2d} <-> {j:2d} | "
            f"cross={data['cross_track']:.2f} m | "
            f"yaw={data['yaw_diff_deg']:.1f} deg | "
            f"along={data['along_track']:.1f} m | "
            f"weight={data['weight']:.3f}"
        )




def ensure_lead_vehicle_stream(
    graph,
    streams,
    lane_fits,
    current_frame_index,
    cfg=None,
):
    """Guarantee that the current front-centre vehicle supports a lane fit.

    Ordinary stream fitting needs at least two observations at distinct
    forward depths. A stopped vehicle can produce several ego-compensated
    observations at essentially one point, so it may have a valid track but no
    fit. In that case its FCOS3D yaw supplies the tangent of a conservative
    one-vehicle anchor line.

    The guarantee starts after detection: if FCOS3D produces no eligible
    current-frame vehicle, this function reports that fact and cannot invent
    one.
    """
    if cfg is None:
        cfg = LeadVehicleConfig()

    streams = [list(stream) for stream in streams]
    lane_fits = [dict(fit) for fit in lane_fits]
    diagnostic = {
        "status": "disabled" if not cfg.enabled else "no_current_vehicle",
        "selected_node": None,
        "track_id": None,
        "stream_id": None,
        "graph_degree": 0,
        "component_size": 0,
        "used_anchor": False,
    }
    if not cfg.enabled:
        return streams, lane_fits, diagnostic

    current_nodes = [
        node
        for node, vehicle in graph.nodes(data=True)
        if vehicle.get("frame_index") == current_frame_index
        and 0.0 < float(vehicle["z"]) <= cfg.max_depth
        and abs(float(vehicle["x"])) <= cfg.max_abs_x
    ]
    if not current_nodes:
        return streams, lane_fits, diagnostic

    forward_yaw = -0.5 * math.pi
    max_yaw_diff = math.radians(cfg.max_forward_yaw_diff_deg)
    aligned_nodes = [
        node
        for node in current_nodes
        if axial_angle_diff(graph.nodes[node]["yaw"], forward_yaw)
        <= max_yaw_diff
    ]
    candidates = aligned_nodes or current_nodes

    # Angular proximity to the optical axis identifies "the car in front"
    # more reliably than x alone at different depths. Prefer the nearer car
    # when angular offsets are effectively tied.
    lead_node = min(
        candidates,
        key=lambda node: (
            abs(
                math.atan2(
                    float(graph.nodes[node]["x"]),
                    float(graph.nodes[node]["z"]),
                )
            ),
            float(graph.nodes[node]["z"]),
        ),
    )
    lead = graph.nodes[lead_node]
    diagnostic.update(
        {
            "selected_node": int(lead_node),
            "track_id": lead.get("track_id"),
            "x": float(lead["x"]),
            "z": float(lead["z"]),
            "yaw_deg": float(math.degrees(lead["yaw"])),
            "score": float(lead["score"]),
            "below_standard_score": bool(
                lead.get("below_standard_score", False)
            ),
            "track_observations": int(lead.get("track_observations", 1)),
            "track_confirmed": bool(lead.get("track_confirmed", True)),
            "graph_degree": int(graph.degree[lead_node]),
            "component_size": int(
                len(nx.node_connected_component(graph, lead_node))
            ),
        }
    )

    for fit in lane_fits:
        if lead_node not in fit.get("inliers", []):
            continue
        fit.update(
            {
                "is_lead_stream": True,
                "forced_lead": False,
                "lead_node_id": int(lead_node),
                "lead_track_id": lead.get("track_id"),
                "lead_score": float(lead["score"]),
            }
        )
        diagnostic.update(
            {
                "status": "existing_fit",
                "stream_id": int(fit["stream_id"]),
            }
        )
        return streams, lane_fits, diagnostic

    heading = heading_from_yaw(float(lead["yaw"]))
    if abs(float(heading[1])) < 1e-6:
        raw_slope = 0.0
        slope_fallback = True
    else:
        raw_slope = float(heading[0] / heading[1])
        slope_fallback = abs(raw_slope) > cfg.max_abs_slope

    slope = (
        0.0
        if slope_fallback
        else float(np.clip(raw_slope, -cfg.max_abs_slope, cfg.max_abs_slope))
    )
    intercept = float(lead["x"] - slope * lead["z"])
    z_min = float(min(lead["z"], max(cfg.near_depth, 1.0)))
    z_max = float(
        min(cfg.max_depth, max(lead["z"] + cfg.forward_extension, z_min + 1.0))
    )

    stream_id = max(
        [int(fit.get("stream_id", -1)) for fit in lane_fits],
        default=-1,
    ) + 1
    track_id = lead.get("track_id")
    anchor_fit = {
        "stream_id": stream_id,
        "source_stream_ids": [],
        "degree": 1,
        "coefficients": np.array([intercept, slope], dtype=np.float64),
        "inliers": [int(lead_node)],
        "outliers": [],
        "rmse": 0.0,
        "z_min": z_min,
        "z_max": z_max,
        "num_observations": 1,
        "track_ids": [] if track_id is None else [int(track_id)],
        "num_tracks": 0 if track_id is None else 1,
        "has_track_info": track_id is not None,
        "track_support": "lead_anchor",
        "source": "lead_vehicle_anchor",
        "is_lead_stream": True,
        "forced_lead": True,
        "lead_node_id": int(lead_node),
        "lead_track_id": track_id,
        "lead_score": float(lead["score"]),
        "yaw_slope_fallback": bool(slope_fallback),
    }
    streams.append([int(lead_node)])
    lane_fits.append(anchor_fit)
    diagnostic.update(
        {
            "status": "anchored_fit",
            "stream_id": int(stream_id),
            "used_anchor": True,
            "yaw_slope_fallback": bool(slope_fallback),
            "anchor_slope": float(slope),
            "z_min": z_min,
            "z_max": z_max,
        }
    )
    return streams, lane_fits, diagnostic


def transform_lane_boundary_to_camera(
    boundary,
    source_road_plane,
    source_cam_to_global,
    target_cam_to_global,
    cfg=None,
):
    """
    Ego-motion compensate a lane boundary into a new camera frame.

    The boundary is sampled as 3D road points in the source camera, rigidly
    transformed through global coordinates, then re-fitted as x(z) in the
    target camera. This avoids directly transforming polynomial coefficients.
    """
    if cfg is None:
        cfg = BoundaryTrackingConfig()
    if source_road_plane is None:
        return None

    z = np.linspace(
        float(boundary["z_min"]),
        float(boundary["z_max"]),
        max(cfg.sample_count, cfg.min_points_after_transform),
    )
    x = evaluate_lane_polynomial(boundary["coefficients"], z)
    y = road_plane_y(source_road_plane, x, z)

    source_points = np.column_stack(
        [x, y, z, np.ones_like(z, dtype=np.float64)]
    )

    source_cam_to_global = np.asarray(source_cam_to_global, dtype=np.float64)
    target_cam_to_global = np.asarray(target_cam_to_global, dtype=np.float64)
    target_from_source = (
        np.linalg.inv(target_cam_to_global) @ source_cam_to_global
    )
    target_points = source_points @ target_from_source.T

    target_x = target_points[:, 0]
    target_z = target_points[:, 2]
    valid = (
        np.isfinite(target_x)
        & np.isfinite(target_z)
        & (target_z >= cfg.min_depth)
    )
    target_x = target_x[valid]
    target_z = target_z[valid]

    if len(target_z) < cfg.min_points_after_transform:
        return None

    order = np.argsort(target_z)
    target_z = target_z[order]
    target_x = target_x[order]

    degree = min(
        max(1, len(np.asarray(boundary["coefficients"])) - 1),
        len(target_z) - 1,
    )
    if np.unique(target_z).size <= degree:
        return None

    try:
        fit_desc = np.polyfit(target_z, target_x, degree)
    except (np.linalg.LinAlgError, ValueError):
        return None

    transformed = dict(boundary)
    transformed.update(
        {
            "coefficients": fit_desc[::-1].copy(),
            "z_min": float(target_z.min()),
            "z_max": float(target_z.max()),
            "overlap": float(target_z.max() - target_z.min()),
            "ego_motion_compensated": True,
        }
    )
    return transformed


def _boundary_temporal_metrics(previous_boundary, current_boundary, cfg):
    z_min = max(previous_boundary["z_min"], current_boundary["z_min"])
    z_max = min(previous_boundary["z_max"], current_boundary["z_max"])
    overlap = float(z_max - z_min)

    if overlap < cfg.min_overlap:
        return {
            "compatible": False,
            "overlap": overlap,
            "median_lateral_distance": np.inf,
            "max_tangent_diff_deg": np.inf,
            "cost": np.inf,
        }

    z = np.linspace(z_min, z_max, max(3, cfg.sample_count))
    x_previous = evaluate_lane_polynomial(previous_boundary["coefficients"], z)
    x_current = evaluate_lane_polynomial(current_boundary["coefficients"], z)
    lateral = np.abs(x_previous - x_current)

    previous_slope = lane_polynomial_slope(previous_boundary["coefficients"], z)
    current_slope = lane_polynomial_slope(current_boundary["coefficients"], z)
    previous_angle = np.arctan(previous_slope)
    current_angle = np.arctan(current_slope)
    tangent_diff = np.abs(previous_angle - current_angle)
    tangent_diff = np.minimum(tangent_diff, np.pi - tangent_diff)

    median_lateral = float(np.median(lateral))
    max_tangent_deg = float(np.degrees(np.max(tangent_diff)))

    compatible = (
        median_lateral <= cfg.max_lateral_distance
        and max_tangent_deg <= cfg.max_tangent_diff_deg
    )

    previous_extent = max(
        1e-6,
        float(previous_boundary["z_max"] - previous_boundary["z_min"]),
    )
    current_extent = max(
        1e-6,
        float(current_boundary["z_max"] - current_boundary["z_min"]),
    )
    overlap_fraction = min(1.0, overlap / min(previous_extent, current_extent))

    cost = (
        median_lateral / max(cfg.max_lateral_distance, 1e-6)
        + max_tangent_deg / max(cfg.max_tangent_diff_deg, 1e-6)
        + 0.25 * (1.0 - overlap_fraction)
    )

    return {
        "compatible": bool(compatible),
        "overlap": overlap,
        "median_lateral_distance": median_lateral,
        "max_tangent_diff_deg": max_tangent_deg,
        "cost": float(cost),
    }


def _smooth_boundary_geometry(previous_boundary, current_boundary, cfg):
    """Smooth sampled x(z) geometry, then refit the current polynomial model."""
    z = np.linspace(
        current_boundary["z_min"],
        current_boundary["z_max"],
        max(6, cfg.sample_count),
    )
    x_current = evaluate_lane_polynomial(current_boundary["coefficients"], z)
    x_smoothed = x_current.copy()

    overlap_mask = (
        (z >= previous_boundary["z_min"])
        & (z <= previous_boundary["z_max"])
    )

    if np.any(overlap_mask):
        x_previous = evaluate_lane_polynomial(
            previous_boundary["coefficients"], z[overlap_mask]
        )
        alpha = float(np.clip(cfg.smoothing_alpha, 0.0, 1.0))
        x_smoothed[overlap_mask] = (
            alpha * x_current[overlap_mask]
            + (1.0 - alpha) * x_previous
        )

    degree = min(
        max(1, len(np.asarray(current_boundary["coefficients"])) - 1),
        len(z) - 1,
    )
    try:
        fit_desc = np.polyfit(z, x_smoothed, degree)
    except (np.linalg.LinAlgError, ValueError):
        return dict(current_boundary)

    result = dict(current_boundary)
    result["coefficients"] = fit_desc[::-1].copy()

    alpha = float(np.clip(cfg.smoothing_alpha, 0.0, 1.0))
    previous_confidence = float(previous_boundary.get("confidence", 1.0))
    current_confidence = float(current_boundary.get("confidence", 1.0))
    result["confidence"] = float(
        alpha * current_confidence + (1.0 - alpha) * previous_confidence
    )
    result["smoothed"] = True
    return result


def update_temporal_lane_tracks(
    state,
    current_boundaries,
    current_cam_to_global,
    current_road_plane,
    frame_index,
    cfg=None,
):
    """
    Associate, smooth and persist lane boundaries across scene frames.

    Parameters
    ----------
    state : dict or None
        Persistent tracker state returned by the previous call.
    current_boundaries : list[dict]
        Current-frame measurements from infer_lane_boundaries().
    current_cam_to_global : ndarray (4, 4)
        Pose of the current camera.
    current_road_plane : dict or None
        Current road plane estimate. If unavailable, transformed previous
        planes are used for carried tracks when possible.
    frame_index : int
        Scene frame index.

    Returns
    -------
    output_boundaries : list[dict]
        Confirmed current measurements and, when configured, short-lived
        predictions or unconfirmed measurements. All live tracks remain in
        state even when they are suppressed from this output.
    state : dict
        Updated persistent tracker state.
    events : list[dict]
        Association diagnostics for logging/evaluation.
    """
    if cfg is None:
        cfg = BoundaryTrackingConfig()
    if state is None:
        state = {"tracks": {}, "next_track_id": 0}

    current_cam_to_global = np.asarray(current_cam_to_global, dtype=np.float64)
    previous_tracks = dict(state.get("tracks", {}))
    next_track_id = int(state.get("next_track_id", 0))

    transformed_tracks = {}
    for track_id, track in previous_tracks.items():
        source_plane = track.get("road_plane")
        if source_plane is None:
            continue

        transformed_boundary = transform_lane_boundary_to_camera(
            track["boundary"],
            source_plane,
            track["cam_to_global"],
            current_cam_to_global,
            cfg=cfg,
        )
        if transformed_boundary is None:
            continue

        transformed_plane = transform_road_plane_to_camera(
            source_plane,
            track["cam_to_global"],
            current_cam_to_global,
        )
        transformed_tracks[track_id] = {
            "track": track,
            "boundary": transformed_boundary,
            "road_plane": transformed_plane,
        }

    candidates = []
    for track_id, predicted in transformed_tracks.items():
        for measurement_index, boundary in enumerate(current_boundaries):
            metrics = _boundary_temporal_metrics(
                predicted["boundary"], boundary, cfg
            )
            if metrics["compatible"]:
                candidates.append(
                    (metrics["cost"], track_id, measurement_index, metrics)
                )

    candidates.sort(key=lambda item: item[0])
    matched_tracks = set()
    matched_measurements = set()
    accepted_matches = []

    for _, track_id, measurement_index, metrics in candidates:
        if track_id in matched_tracks or measurement_index in matched_measurements:
            continue
        matched_tracks.add(track_id)
        matched_measurements.add(measurement_index)
        accepted_matches.append((track_id, measurement_index, metrics))

    output_boundaries = []
    new_tracks = {}
    events = []

    # Matched measurements: ego compensate previous geometry, smooth in x(z),
    # and preserve the same persistent boundary ID.
    for track_id, measurement_index, metrics in accepted_matches:
        previous_record = transformed_tracks[track_id]
        old_track = previous_record["track"]
        current = current_boundaries[measurement_index]
        fused = _smooth_boundary_geometry(
            previous_record["boundary"], current, cfg
        )

        age = int(old_track.get("age", 1)) + 1
        hits = int(old_track.get("hits", 1)) + 1
        fused.update(
            {
                "boundary_track_id": int(track_id),
                "temporal_status": "matched",
                "track_age": age,
                "track_hits": hits,
                "missed_frames": 0,
                "is_predicted": False,
            }
        )

        track_plane = (
            current_road_plane
            if current_road_plane is not None
            else previous_record["road_plane"]
        )
        new_tracks[track_id] = {
            "track_id": int(track_id),
            "boundary": dict(fused),
            "cam_to_global": current_cam_to_global.copy(),
            "road_plane": track_plane,
            "last_frame_index": int(frame_index),
            "age": age,
            "hits": hits,
            "missed_frames": 0,
        }
        confirmed = hits >= cfg.min_confirmed_hits
        force_emit = bool(fused.get("forced_lead", False))
        if confirmed or cfg.emit_unconfirmed or force_emit:
            output_boundaries.append(fused)
        events.append(
            {
                "type": "matched",
                "boundary_track_id": int(track_id),
                "measurement_index": int(measurement_index),
                **metrics,
            }
        )

    # New measurements start new persistent lane-boundary tracks.
    for measurement_index, boundary in enumerate(current_boundaries):
        if measurement_index in matched_measurements:
            continue

        track_id = next_track_id
        next_track_id += 1
        new_boundary = dict(boundary)
        new_boundary.update(
            {
                "boundary_track_id": int(track_id),
                "temporal_status": "new",
                "track_age": 1,
                "track_hits": 1,
                "missed_frames": 0,
                "is_predicted": False,
                "smoothed": False,
            }
        )
        new_tracks[track_id] = {
            "track_id": int(track_id),
            "boundary": dict(new_boundary),
            "cam_to_global": current_cam_to_global.copy(),
            "road_plane": current_road_plane,
            "last_frame_index": int(frame_index),
            "age": 1,
            "hits": 1,
            "missed_frames": 0,
        }
        confirmed = 1 >= cfg.min_confirmed_hits
        force_emit = bool(new_boundary.get("forced_lead", False))
        if confirmed or cfg.emit_unconfirmed or force_emit:
            output_boundaries.append(new_boundary)
        events.append(
            {
                "type": "new",
                "boundary_track_id": int(track_id),
                "measurement_index": int(measurement_index),
            }
        )

    # Unmatched prior tracks are carried for a short period with decaying
    # confidence. This bridges occasional missed FCOS3D / lane-fit frames.
    for track_id, previous_record in transformed_tracks.items():
        if track_id in matched_tracks:
            continue

        old_track = previous_record["track"]
        missed_frames = int(old_track.get("missed_frames", 0)) + 1
        if missed_frames > cfg.max_missed_frames:
            events.append(
                {
                    "type": "expired",
                    "boundary_track_id": int(track_id),
                    "missed_frames": missed_frames,
                }
            )
            continue

        carried = dict(previous_record["boundary"])
        carried["confidence"] = float(
            carried.get("confidence", 1.0)
            * cfg.missing_confidence_decay
        )
        age = int(old_track.get("age", 1)) + 1
        hits = int(old_track.get("hits", 1))
        carried.update(
            {
                "boundary_track_id": int(track_id),
                "temporal_status": "predicted",
                "track_age": age,
                "track_hits": hits,
                "missed_frames": missed_frames,
                "is_predicted": True,
                "smoothed": bool(carried.get("smoothed", False)),
            }
        )

        carried_plane = previous_record["road_plane"]
        new_tracks[track_id] = {
            "track_id": int(track_id),
            "boundary": dict(carried),
            "cam_to_global": current_cam_to_global.copy(),
            "road_plane": carried_plane,
            "last_frame_index": int(frame_index),
            "age": age,
            "hits": hits,
            "missed_frames": missed_frames,
        }
        confirmed = hits >= cfg.min_confirmed_hits
        if cfg.emit_predicted and (confirmed or cfg.emit_unconfirmed):
            output_boundaries.append(carried)
        events.append(
            {
                "type": "predicted",
                "boundary_track_id": int(track_id),
                "missed_frames": missed_frames,
            }
        )

    def lateral_key(boundary):
        z_mid = 0.5 * (boundary["z_min"] + boundary["z_max"])
        return float(
            evaluate_lane_polynomial(boundary["coefficients"], z_mid)
        )

    output_boundaries.sort(key=lateral_key)
    state = {
        "tracks": new_tracks,
        "next_track_id": next_track_id,
    }
    return output_boundaries, state, events



def project_camera_points(points_3d, cam2img):
    points_3d = np.asarray(points_3d, dtype=np.float64)
    cam2img = np.asarray(cam2img, dtype=np.float64)
    if cam2img.shape == (3, 3):
        projected = points_3d @ cam2img.T
    elif cam2img.shape == (3, 4):
        projected = np.column_stack([points_3d, np.ones(len(points_3d))]) @ cam2img.T
    elif cam2img.shape == (4, 4):
        projected = (
            np.column_stack([points_3d, np.ones(len(points_3d))]) @ cam2img.T
        )[:, :3]
    else:
        raise ValueError(f"Unsupported cam2img shape: {cam2img.shape}")

    pixels = np.full((len(points_3d), 2), np.nan, dtype=np.float64)
    valid = np.abs(projected[:, 2]) > 1e-8
    pixels[valid] = projected[valid, :2] / projected[valid, 2, None]
    return pixels


def project_lane_boundaries_to_image(
    lane_boundaries, road_plane, cam2img, image_shape=None, cfg=None
):
    if cfg is None:
        cfg = LaneProjectionConfig()
    if road_plane is None:
        return []

    projected_boundaries = []
    for boundary_id, boundary in enumerate(lane_boundaries):
        z = np.linspace(boundary["z_min"], boundary["z_max"], cfg.sample_count)
        x = evaluate_lane_polynomial(boundary["coefficients"], z)
        y = road_plane_y(road_plane, x, z)
        points_3d = np.column_stack([x, y, z])
        pixels = project_camera_points(points_3d, cam2img)

        valid = (
            np.isfinite(pixels[:, 0])
            & np.isfinite(pixels[:, 1])
            & (z >= cfg.min_depth)
        )
        if cfg.clip_to_image and image_shape is not None:
            height, width = image_shape[:2]
            valid &= (
                (pixels[:, 0] >= 0)
                & (pixels[:, 0] < width)
                & (pixels[:, 1] >= 0)
                & (pixels[:, 1] < height)
            )
        if not np.any(valid):
            continue

        projected_boundaries.append(
            {
                "boundary_id": boundary_id,
                "left_stream_id": boundary["left_stream_id"],
                "right_stream_id": boundary["right_stream_id"],
                "pixels": pixels[valid],
                "points_3d": points_3d[valid],
                "lane_width": boundary["lane_width"],
                "source": boundary.get("source", "paired_streams"),
                "confidence": boundary.get("confidence", 1.0),
                "side": boundary.get("side", "between"),
                "source_stream_id": boundary.get("source_stream_id"),
                "boundary_track_id": boundary.get("boundary_track_id"),
                "temporal_status": boundary.get("temporal_status", "measurement"),
                "track_age": boundary.get("track_age"),
                "track_hits": boundary.get("track_hits"),
                "missed_frames": boundary.get("missed_frames", 0),
                "is_predicted": boundary.get("is_predicted", False),
                "smoothed": boundary.get("smoothed", False),
                "forced_lead": bool(boundary.get("forced_lead", False)),
            }
        )

    return projected_boundaries


def plot_projected_lane_boundaries(
    image_path, projected_boundaries, save_path=None, show=True, title=None
):
    image = plt.imread(image_path)
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(image)

    for boundary in projected_boundaries:
        pixels = boundary["pixels"]
        if len(pixels) < 2:
            continue

        is_provisional = boundary.get("source") == "single_stream"
        is_predicted = bool(boundary.get("is_predicted", False))
        track_id = boundary.get("boundary_track_id")
        status = boundary.get("temporal_status", "measurement")

        if is_predicted:
            linestyle = "--"
            linewidth = 2
        elif is_provisional:
            linestyle = ":"
            linewidth = 5
        else:
            linestyle = "-"
            linewidth = 7

        if track_id is not None:
            label = f"B{track_id} {status}"
        elif is_provisional:
            label = (
                f"provisional S{boundary.get('source_stream_id')} "
                f"{boundary.get('side', '')}"
            )
        else:
            label = (
                f"boundary S{boundary['left_stream_id']}|"
                f"S{boundary['right_stream_id']}"
            )

        ax.plot(
            pixels[:, 0],
            pixels[:, 1],
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=max(0.20, float(boundary.get("confidence", 1.0))),
            label=label,
        )

    ax.set_title(title or "Projected temporally tracked lane boundaries")
    ax.axis("off")
    if projected_boundaries:
        ax.legend(loc="best")
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved image-space lane projection to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

def _vehicle_rectangle(x, z, yaw, length, width):
    heading = heading_from_yaw(yaw)
    normal = np.array([-heading[1], heading[0]])
    centre = np.array([x, z])
    half_length = length / 2.0
    half_width = width / 2.0
    return np.stack(
        [
            centre + half_length * heading + half_width * normal,
            centre + half_length * heading - half_width * normal,
            centre - half_length * heading - half_width * normal,
            centre - half_length * heading + half_width * normal,
        ]
    )


def plot_lane_graph(
    graph,
    streams=None,
    lane_fits=None,
    lane_boundaries=None,
    max_depth=60.0,
    x_range=(-15.0, 15.0),
    show_labels=True,
    save_path=None,
    show=True,
):
    if streams is None:
        streams = [list(c) for c in nx.connected_components(graph)]
    node_to_stream = {
        node: stream_id
        for stream_id, stream in enumerate(streams)
        for node in stream
    }

    fig, ax = plt.subplots(figsize=(9, 12))
    ax.scatter(0, 0, marker="^", s=150, label="Camera")
    ax.text(0.3, 0.5, "camera")

    for i, j, edge in graph.edges(data=True):
        vi, vj = graph.nodes[i], graph.nodes[j]
        ax.plot(
            [vi["x"], vj["x"]],
            [vi["z"], vj["z"]],
            linewidth=0.5 + 3.0 * edge.get("weight", 1.0),
            alpha=0.5,
        )

    for node, vehicle in graph.nodes(data=True):
        x, z, yaw = vehicle["x"], vehicle["z"], vehicle["yaw"]
        corners = _vehicle_rectangle(
            x, z, yaw, vehicle["length"], vehicle["width"]
        )
        ax.add_patch(Polygon(corners, closed=True, fill=False, linewidth=2))
        ax.scatter(x, z, s=50)
        ax.arrow(
            x,
            z,
            2.5 * np.cos(yaw),
            2.5 * np.sin(yaw),
            width=0.025,
            head_width=0.35,
            head_length=0.5,
            length_includes_head=True,
        )
        if show_labels:
            label = (
                f"{node}\nS{node_to_stream.get(node, -1)}\n{vehicle['score']:.2f}"
            )
            if "track_id" in vehicle:
                label += f"\nT{vehicle['track_id']} F{vehicle['frame_index']}"
            ax.text(x + 0.25, z + 0.25, label, fontsize=8)

    if lane_fits:
        for fit in lane_fits:
            z_curve = np.linspace(fit["z_min"], fit["z_max"], 100)
            x_curve = evaluate_lane_polynomial(fit["coefficients"], z_curve)
            source_ids = fit.get("source_stream_ids", [fit["stream_id"]])
            track_text = (
                f" T={fit.get('num_tracks', 0)}"
                if fit.get("has_track_info")
                else ""
            )
            source_text = (
                f" src={source_ids}"
                if len(source_ids) > 1
                else ""
            )
            ax.plot(
                x_curve,
                z_curve,
                linewidth=3,
                label=(
                    f"centre S{fit['stream_id']}"
                    f"{track_text}{source_text}"
                ),
            )

    if lane_boundaries:
        for boundary in lane_boundaries:
            z_curve = np.linspace(boundary["z_min"], boundary["z_max"], 100)
            x_curve = evaluate_lane_polynomial(boundary["coefficients"], z_curve)
            is_provisional = boundary.get("source") == "single_stream"
            is_predicted = bool(boundary.get("is_predicted", False))
            track_id = boundary.get("boundary_track_id")
            status = boundary.get("temporal_status", "measurement")

            if is_predicted:
                linestyle = "--"
                linewidth = 2
            elif is_provisional:
                linestyle = ":"
                linewidth = 5
            else:
                linestyle = "--"
                linewidth = 7
                

            if track_id is not None:
                label = f"B{track_id} {status}"
            elif is_provisional:
                label = (
                    f"provisional S{boundary.get('source_stream_id')} "
                    f"{boundary.get('side', '')}"
                )
            else:
                label = (
                    f"boundary S{boundary['left_stream_id']}|"
                    f"S{boundary['right_stream_id']}"
                )

            ax.plot(
                x_curve,
                z_curve,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=max(0.20, float(boundary.get("confidence", 1.0))),
                label=label,
            )

    ax.set_xlim(*x_range)
    ax.set_ylim(0, max_depth)
    ax.set_xlabel("Lateral x [m]")
    ax.set_ylabel("Forward z [m]")
    ax.set_title("Temporal FCOS3D lane graph")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    if lane_fits or lane_boundaries:
        ax.legend(loc="best")
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved BEV visualisation to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
