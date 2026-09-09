import numpy as np
import math

from configs import(
    TemporalConfig,
)

from geometry import axial_angle_diff

'''
Code to track past vehicle detections and move them into the 
current detection frame to infer traffic streams with.
'''

def transform_vehicles_to_reference(
    vehicles,
    current_cam_to_global,
    reference_cam_to_global,
    frame_index=None,
    frame_age=0,
    temporal_decay=1.0,
):
    current_cam_to_global = np.asarray(current_cam_to_global, dtype=np.float64)
    reference_cam_to_global = np.asarray(reference_cam_to_global, dtype=np.float64)
    reference_from_current = (
        np.linalg.inv(reference_cam_to_global) @ current_cam_to_global
    )
    rotation = reference_from_current[:3, :3]

    transformed = []
    for vehicle in vehicles:
        point = np.array(
            [vehicle["x"], vehicle.get("y", 0.0), vehicle["z"], 1.0]
        )
        point_ref = reference_from_current @ point

        heading = np.array(
            [np.cos(vehicle["yaw"]), 0.0, np.sin(vehicle["yaw"])]
        )
        heading_ref = rotation @ heading
        if np.hypot(heading_ref[0], heading_ref[2]) < 1e-8:
            continue

        updated = dict(vehicle)
        updated.update(
            {
                "x": float(point_ref[0]),
                "y": float(point_ref[1]),
                "z": float(point_ref[2]),
                "yaw": float(np.arctan2(heading_ref[2], heading_ref[0])),
                "frame_index": frame_index,
                "frame_age": int(frame_age),
                "evidence_weight": float(temporal_decay**frame_age),
            }
        )
        transformed.append(updated)

    return transformed


def _track_cost(track, detection, max_yaw_diff):
    if track["label"] != detection["label"]:
        return None
    if axial_angle_diff(track["yaw"], detection["yaw"]) > max_yaw_diff:
        return None
    return float(
        np.hypot(
            detection["x"] - track["x"],
            detection["z"] - track["z"],
        )
    )


def assign_temporal_tracks(frame_vehicles, cfg=None):
    if cfg is None:
        cfg = TemporalConfig()

    max_yaw_diff = math.radians(cfg.max_track_yaw_diff_deg)
    tracks = {}
    next_track_id = 0
    all_detections = []

    for frame_order, (frame_index, detections) in enumerate(frame_vehicles):
        candidates = []
        for det_idx, detection in enumerate(detections):
            for track_id, track in tracks.items():
                frame_gap = frame_order - track["last_frame_order"]
                if frame_gap <= 0 or frame_gap > cfg.max_track_frame_gap:
                    continue
                cost = _track_cost(track, detection, max_yaw_diff)
                if cost is not None and cost <= cfg.max_track_distance:
                    candidates.append((cost, track_id, det_idx))

        candidates.sort(key=lambda item: item[0])
        matched_tracks = set()
        matched_detections = set()

        for _, track_id, det_idx in candidates:
            if track_id in matched_tracks or det_idx in matched_detections:
                continue
            detection = detections[det_idx]
            detection["track_id"] = track_id
            matched_tracks.add(track_id)
            matched_detections.add(det_idx)
            tracks[track_id].update(
                {
                    "x": detection["x"],
                    "z": detection["z"],
                    "yaw": detection["yaw"],
                    "last_frame_order": frame_order,
                    "observations": tracks[track_id]["observations"] + 1,
                }
            )

        for det_idx, detection in enumerate(detections):
            if det_idx not in matched_detections:
                track_id = next_track_id
                next_track_id += 1
                detection["track_id"] = track_id
                tracks[track_id] = {
                    "label": detection["label"],
                    "x": detection["x"],
                    "z": detection["z"],
                    "yaw": detection["yaw"],
                    "last_frame_order": frame_order,
                    "observations": 1,
                }
            all_detections.append(detection)

    for detection in all_detections:
        observations = tracks[detection["track_id"]]["observations"]
        detection["track_observations"] = int(observations)
        detection["track_confirmed"] = bool(
            observations >= cfg.min_track_observations
        )

    if cfg.min_track_observations <= 1:
        return all_detections, tracks

    keep = {
        track_id
        for track_id, track in tracks.items()
        if track["observations"] >= cfg.min_track_observations
    }
    latest_frame_index = frame_vehicles[-1][0] if frame_vehicles else None
    return [
        detection
        for detection in all_detections
        if detection["track_id"] in keep
        or (
            cfg.keep_latest_frame_detections
            and detection.get("frame_index") == latest_frame_index
        )
    ], tracks


def accumulate_temporal_vehicle_evidence(
    frame_records, reference_record_index=-1, cfg=None
):
    if cfg is None:
        cfg = TemporalConfig()
    if not frame_records:
        return [], {}

    ref_position = (
        len(frame_records) + reference_record_index
        if reference_record_index < 0
        else reference_record_index
    )
    reference_transform = frame_records[ref_position]["cam_to_global"]

    transformed_frames = []
    for position, record in enumerate(frame_records):
        frame_age = abs(ref_position - position)
        transformed = transform_vehicles_to_reference(
            record["vehicles"],
            record["cam_to_global"],
            reference_transform,
            frame_index=record["frame_index"],
            frame_age=frame_age,
            temporal_decay=cfg.temporal_decay,
        )
        transformed_frames.append((record["frame_index"], transformed))

    return assign_temporal_tracks(transformed_frames, cfg=cfg)

