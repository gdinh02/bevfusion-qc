import numpy as np

from configs import BoundaryTrackingConfig
from lane_fitting import (
    evaluate_lane_polynomial,
    lane_polynomial_slope,
)
from road_plane import (
    road_plane_y,
    transform_road_plane_to_camera,
)

'''
from boundary_tracking import (
    transform_lane_boundary_to_camera,
    _boundary_temporal_metrics,
    _smooth_boundary_geometry,
    update_temporal_lane_tracks,
)
'''

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

