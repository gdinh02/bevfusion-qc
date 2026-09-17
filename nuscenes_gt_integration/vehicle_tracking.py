import numpy as np

from lane_inference.configs import (
    TemporalConfig,
)

from lane_inference.vehicle_tracking import (
    transform_vehicles_to_reference,
)


def assign_gt_tracks(frame_vehicles, cfg=None):
    """
    Attach perfect nuScenes GT track metadata to temporally accumulated
    detections.

    Unlike assign_temporal_tracks(), this performs NO association.
    Each detection already has its correct scene-local track_id derived
    from the nuScenes instance_token.

    Parameters
    ----------
    frame_vehicles:
        List of:
            (frame_index, [vehicle_dict, ...])

        Vehicles must already contain:
            track_id
            instance_token

    cfg:
        TemporalConfig.

    Returns
    -------
    detections:
        Flattened list of detections with:
            track_observations
            track_confirmed

    tracks:
        Dictionary indexed by GT track_id containing diagnostics.
    """
    if cfg is None:
        cfg = TemporalConfig()

    tracks = {}
    all_detections = []

    for frame_order, (frame_index, detections) in enumerate(frame_vehicles):
        seen_track_ids = set()

        for detection in detections:
            if "track_id" not in detection:
                raise ValueError(
                    "GT temporal accumulation requires every vehicle "
                    "to contain track_id"
                )

            if "instance_token" not in detection:
                raise ValueError(
                    "GT temporal accumulation requires every vehicle "
                    "to contain instance_token"
                )

            track_id = int(detection["track_id"])
            instance_token = detection["instance_token"]

            # A nuScenes instance can appear only once in a given sample.
            if track_id in seen_track_ids:
                raise ValueError(
                    f"Duplicate GT track_id {track_id} in frame "
                    f"{frame_index}"
                )

            seen_track_ids.add(track_id)

            if track_id not in tracks:
                tracks[track_id] = {
                    "track_id": track_id,
                    "instance_token": instance_token,
                    "label": detection.get("label"),
                    "class_name": detection.get("class_name"),
                    "first_frame_index": frame_index,
                    "last_frame_index": frame_index,
                    "first_frame_order": frame_order,
                    "last_frame_order": frame_order,
                    "observations": 0,
                }
            else:
                track = tracks[track_id]

                if track["instance_token"] != instance_token:
                    raise ValueError(
                        f"GT track_id {track_id} maps to multiple "
                        f"instance tokens"
                    )

                track["last_frame_index"] = frame_index
                track["last_frame_order"] = frame_order

            tracks[track_id]["observations"] += 1

            all_detections.append(detection)

    # Add track-level support information to every observation.
    for detection in all_detections:
        track_id = int(detection["track_id"])
        observations = int(
            tracks[track_id]["observations"]
        )

        detection["track_observations"] = observations
        detection["track_confirmed"] = bool(
            observations >= cfg.min_track_observations
        )

    # Keep behaviour consistent with the existing temporal path.
    if cfg.min_track_observations <= 1:
        return all_detections, tracks

    confirmed_track_ids = {
        track_id
        for track_id, track in tracks.items()
        if track["observations"] >= cfg.min_track_observations
    }

    latest_frame_index = (
        frame_vehicles[-1][0]
        if frame_vehicles
        else None
    )

    filtered = [
        detection
        for detection in all_detections
        if (
            int(detection["track_id"])
            in confirmed_track_ids
        )
        or (
            cfg.keep_latest_frame_detections
            and detection.get("frame_index")
            == latest_frame_index
        )
    ]

    return filtered, tracks


def accumulate_temporal_gt_vehicle_evidence(
    frame_records,
    reference_record_index=-1,
    cfg=None,
):
    """
    Accumulate nuScenes GT vehicle evidence in a common reference frame,
    while preserving perfect GT identity association.

    Historical observations are ego-motion compensated into the selected
    reference frame, but vehicle motion itself is NOT removed.

    This is intentional: repeated observations of a moving vehicle form
    spatial evidence describing the traffic stream.

    Parameters
    ----------
    frame_records:
        List of dictionaries containing:

            {
                "frame_index": int,
                "vehicles": list[dict],
                "cam_to_global": 4x4 ndarray,
            }

        Despite the historical key name ``cam_to_global``, the GT pipeline
        should supply the canonical lane-frame-to-global transform returned
        by get_lane_to_global().

    reference_record_index:
        Which record provides the target lane coordinate frame.
        -1 means the newest/current frame.

    cfg:
        TemporalConfig.

    Returns
    -------
    vehicles:
        All transformed GT observations passing temporal support rules.

    tracks:
        Perfect GT track metadata keyed by track_id.
    """
    if cfg is None:
        cfg = TemporalConfig()

    if not frame_records:
        return [], {}

    ref_position = (
        len(frame_records) + reference_record_index
        if reference_record_index < 0
        else reference_record_index
    )

    if not (
        0 <= ref_position < len(frame_records)
    ):
        raise IndexError(
            "reference_record_index is outside frame_records"
        )

    reference_transform = np.asarray(
        frame_records[ref_position]["cam_to_global"],
        dtype=np.float64,
    )

    if reference_transform.shape != (4, 4):
        raise ValueError(
            "reference cam_to_global must be 4x4"
        )

    transformed_frames = []

    for position, record in enumerate(frame_records):
        current_transform = np.asarray(
            record["cam_to_global"],
            dtype=np.float64,
        )

        if current_transform.shape != (4, 4):
            raise ValueError(
                "cam_to_global transforms must be 4x4"
            )

        frame_age = abs(
            ref_position - position
        )

        transformed = (
            transform_vehicles_to_reference(
                record["vehicles"],
                current_transform,
                reference_transform,
                frame_index=record["frame_index"],
                frame_age=frame_age,
                temporal_decay=cfg.temporal_decay,
            )
        )

        transformed_frames.append(
            (
                record["frame_index"],
                transformed,
            )
        )

    return assign_gt_tracks(
        transformed_frames,
        cfg=cfg,
    )