import numpy as np
import matplotlib.pyplot as plt

from lane_graph import (
    LaneBoundaryConfig,
    LaneFitConfig,
    LaneGraphConfig,
    LaneProjectionConfig,
    RoadPlaneConfig,
    TemporalConfig,
    accumulate_temporal_vehicle_evidence,
    build_lane_compatibility_graph_from_vehicles,
    estimate_road_plane,
    extract_vehicles_from_prediction,
    fit_lane_streams,
    get_lane_streams,
    infer_lane_boundaries,
    plot_lane_graph,
    plot_projected_lane_boundaries,
    print_graph_edges,
    project_lane_boundaries_to_image,
)

SEQUENCE_INFO_FILE = (
    "demo/data/nuscenes/"
    "n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151621912404.pkl"
)
DATA_ROOT = "data/nuscenes"
REFERENCE_FRAME_INDEX = -1

DEVICE = "cuda:0"
CAM_TYPE = "CAM_FRONT"
WANTED_VEHICLE_CLASSES = {"car", "truck", "bus"}

GRAPH_CONFIG = LaneGraphConfig(
    score_thresh=0.30,
    max_depth=50.0,
    max_cross_track=1.5,
    max_yaw_diff_deg=15.0,
    max_along_track=25.0,
    sigma_cross_track=0.8,
    sigma_yaw_deg=8.0,
)

TEMPORAL_CONFIG = TemporalConfig(
    history_frames=5,
    max_track_distance=12.0,
    max_track_yaw_diff_deg=30.0,
    max_track_frame_gap=1,
    temporal_decay=0.90,
    min_track_observations=1,
    max_reference_distance=60.0,
    max_time_gap_s=3.0,
)

FIT_CONFIG = LaneFitConfig(
    degree=2,
    residual_threshold=0.75,
    max_trials=200,
    random_seed=0,
)

BOUNDARY_CONFIG = LaneBoundaryConfig(
    min_overlap=10.0,
    min_lane_width=2.5,
    max_lane_width=5.0,
    sample_count=50,
)

ROAD_PLANE_CONFIG = RoadPlaneConfig(
    residual_threshold=0.35,
    max_trials=200,
    random_seed=0,
)

PROJECTION_CONFIG = LaneProjectionConfig(
    sample_count=200,
    min_depth=1.0,
    clip_to_image=True,
)


def _resolve_reference_index(length, index):
    resolved = index if index >= 0 else length + index
    if resolved < 0 or resolved >= length:
        raise IndexError(f"REFERENCE_FRAME_INDEX {index} is outside 0..{length - 1}")
    return resolved


def _camera_to_global(sample, cam_type):
    """Return the 4x4 camera-to-global pose for one nuScenes sample."""
    ego2global = np.asarray(sample["ego2global"], dtype=np.float64)
    cam2ego = np.asarray(sample["images"][cam_type]["cam2ego"], dtype=np.float64)

    if ego2global.shape != (4, 4) or cam2ego.shape != (4, 4):
        raise ValueError("ego2global and cam2ego must both be 4x4 matrices")

    return ego2global @ cam2ego


def _timestamp_seconds(sample):
    timestamp = sample.get("timestamp")
    if timestamp is None:
        return None

    # nuScenes timestamps are normally in microseconds.
    timestamp = float(timestamp)
    return timestamp / 1e6 if abs(timestamp) > 1e9 else timestamp


def _select_temporal_samples(sequence_info, reference_index, cfg):
    """Select recent frames that plausibly belong to the reference scene."""
    data_list = sequence_info["data_list"]
    ref_idx = _resolve_reference_index(len(data_list), reference_index)
    reference = data_list[ref_idx]
    reference_pose = np.asarray(reference["ego2global"], dtype=np.float64)
    reference_position = reference_pose[:3, 3]
    reference_time = _timestamp_seconds(reference)
    reference_scene = reference.get("scene_token")

    start = max(0, ref_idx - cfg.history_frames + 1)
    selected = []

    for frame_idx in range(start, ref_idx + 1):
        sample = data_list[frame_idx]

        if CAM_TYPE not in sample.get("images", {}):
            continue

        sample_scene = sample.get("scene_token")
        if reference_scene is not None and sample_scene is not None:
            if sample_scene != reference_scene:
                continue

        pose = np.asarray(sample["ego2global"], dtype=np.float64)
        ego_distance = float(np.linalg.norm(pose[:3, 3] - reference_position))
        if ego_distance > cfg.max_reference_distance:
            continue

        sample_time = _timestamp_seconds(sample)
        if reference_time is not None and sample_time is not None:
            if abs(reference_time - sample_time) > cfg.max_time_gap_s:
                continue

        selected.append((frame_idx, sample))

    if not selected or selected[-1][0] != ref_idx:
        selected.append((ref_idx, reference))

    selected.sort(key=lambda item: item[0])
    return selected, ref_idx


def _resolve_image_path(sample, cam_type, info_file, data_root):
    image_path = Path(sample["images"][cam_type]["img_path"])
    candidates = [
        image_path,
        Path(data_root) / image_path,
        Path(info_file).resolve().parent / image_path,
    ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        "Could not resolve image path from info entry. Tried:\n  "
        + "\n  ".join(str(candidate) for candidate in candidates)
    )


def _write_single_sample_info(sequence_info, sample, path):
    """Create the one-sample annotation file expected by mono-3D inference."""
    single = {
        key: value
        for key, value in sequence_info.items()
        if key != "data_list"
    }
    single["data_list"] = [sample]
    mmengine.dump(single, path)


def _run_temporal_inference(model, sequence_info, selected_frames, vehicle_label_ids):
    frame_records = []

    with tempfile.TemporaryDirectory(prefix="fcos3d_temporal_") as temp_dir:
        for order, (frame_idx, sample) in enumerate(selected_frames):
            image_path = _resolve_image_path(
                sample,
                CAM_TYPE,
                SEQUENCE_INFO_FILE,
                DATA_ROOT,
            )
            one_frame_info = Path(temp_dir) / f"frame_{frame_idx}.pkl"
            _write_single_sample_info(sequence_info, sample, one_frame_info)

            print(
                f"Frame {order + 1}/{len(selected_frames)} "
                f"(data_list[{frame_idx}]): {image_path}"
            )
            result = inference_mono_3d_detector(
                model,
                image_path,
                str(one_frame_info),
                cam_type=CAM_TYPE,
            )
            pred = result.pred_instances_3d
            vehicles = extract_vehicles_from_prediction(
                pred,
                vehicle_label_ids,
                cfg=GRAPH_CONFIG,
            )

            frame_records.append({
                "frame_index": frame_idx,
                "sample": sample,
                "image_path": image_path,
                "pred": pred,
                "vehicles": vehicles,
                "cam_to_global": _camera_to_global(sample, CAM_TYPE),
            })
            print(
                f"  raw detections={len(pred.bboxes_3d)} | "
                f"usable vehicles={len(vehicles)}"
            )

    return frame_records