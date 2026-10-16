#!/usr/bin/env python3
"""nuScenes-GT lane demo using the existing square BEV visualisation.

The exported scene PKL is the only inference input. BEVFusion model loading and
inference are not used. The existing BEV renderer is retained by adapting the
canonical GT vehicles to the box layout it already accepts.

Examples:
    python run_live_demo_profiled_gt_bevonly.py \
        --input Z:/dataset/scene-0956 --output gt_bev.mp4

    python run_live_demo_profiled_gt_bevonly.py `
        --input Z:/dataset/scene-0399 --output gt_bev.mp4 `
        --mask-dir Z:/dataset/mask --no-display `
        --pred-dir Z:/dataset/scene-0399/pred `
        --vehicles-dir Z:/dataset/scene-0399/vehicles
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import time
import json
from collections import defaultdict
from statistics import mean

import cv2
import numpy as np
import torch

import run_live_demo_bev as base
from nuscenes_gt_integration.gt_adaptor import extract_vehicles_from_gt
from nuscenes_gt_integration.pipeline import GTLanePipeline
from run_live_demo_profiled_prefetch import DirectFFmpegWriter, _ms
from lane_inference.configs import LaneBoundaryConfig
from sweep_config import (
    make_pipeline_kwargs,
    load_overrides,
    
)

PROFILE_ORDER = (
    "gt_lane_pipeline",
    "gt_visualisation_boxes",
    "bev_render",
    "video_submit",
    "cpu_post_wall",
    "pipeline_interval",
)


# generate_bev_map() uses the BEVFusion class ordering from bev_helper.py,
# which differs from GT_CLASS_LABELS for bus/construction/trailer.
VISUALISATION_LABELS = {
    "car": 0,
    "truck": 1,
    "construction": 2,
    "bus": 3,
    "trailer": 4,
}


def gt_vehicles_to_visualisation_boxes(
    vehicles: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """Adapt canonical GT vehicles for the unchanged BEV box renderer.

    Canonical lane coordinates are x-right/y-down/z-forward. The existing
    renderer accepts [x-right, y-forward, z-up, width, length, height, yaw,
    vx, vy]. Its yaw drawing convention is also inverted, hence the explicit
    conversion below.
    """
    boxes = np.zeros((len(vehicles), 9), dtype=np.float32)
    labels = np.zeros(len(vehicles), dtype=np.int64)

    for index, vehicle in enumerate(vehicles):
        boxes[index, 0] = float(vehicle["x"])
        boxes[index, 1] = float(vehicle["z"])
        boxes[index, 2] = -float(vehicle["y"])
        boxes[index, 3] = float(vehicle["width"])
        boxes[index, 4] = float(vehicle["length"])
        boxes[index, 5] = float(vehicle["height"])
        boxes[index, 6] = -float(vehicle["yaw"]) - (np.pi / 2.0)

        class_name = str(vehicle["class_name"])
        try:
            labels[index] = VISUALISATION_LABELS[class_name]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported GT visualisation class: {class_name}"
            ) from exc

    return boxes, labels


def render_gt_frame(
    lane_pipeline: GTLanePipeline,
    frame: dict,
    lane_only: bool = False # Fixed type hint (was `lane_only: False`)
):
    """Run GT lane inference and render the existing 40 x 40 metre BEV."""
    timings: dict[str, float] = {}
    post_start = time.perf_counter()

    t0 = time.perf_counter()
    graph, streams, boundaries, vehicle_count = lane_pipeline.process(frame)

    if LaneBoundaryConfig.verbose == True:
        frame_id = int(frame["frame_index"])
        graph = lane_pipeline.last_graph
        streams = lane_pipeline.last_streams
        fits = lane_pipeline.last_fits

        stream_for_node = {
            node: stream_id
            for stream_id, stream in enumerate(streams)
            for node in stream
        }

        fit_inliers = {
            node
            for fit in fits
            for node in fit.get("inliers", [])
        }

        print("\nCURRENT GT VEHICLES IN GRAPH")

        for node, v in graph.nodes(data=True):
            if int(v.get("frame_index", -1)) != frame_id:
                continue

            print(
                f"node={node:3d} "
                f"track={v.get('track_id')} "
                f"x={v['x']:6.2f} "
                f"z={v['z']:6.2f} "
                f"yaw={np.degrees(v['yaw']):6.1f} "
                f"degree={graph.degree[node]:2d} "
                f"stream={stream_for_node.get(node)} "
                f"fit_inlier={node in fit_inliers}"
            )

    timings["gt_lane_pipeline"] = _ms(t0)

    # Draw every exported vehicle, not only the subset admitted as lane
    # evidence by GTLanePipeline's depth/yaw filters.
    t0 = time.perf_counter()
    all_gt_vehicles = extract_vehicles_from_gt(
        frame["gt_vehicles"],
        frame["inputs_json"],
    )
    bboxes, labels = gt_vehicles_to_visualisation_boxes(all_gt_vehicles)
    timings["gt_visualisation_boxes"] = _ms(t0)

    t0 = time.perf_counter()
    
    if not lane_only:
        bev = base.generate_bev_map(
            bboxes=torch.from_numpy(bboxes),
            labels=torch.from_numpy(labels),
            pixels_per_meter=base.PIXELS_PER_METER,
            max_range_meters=20,
        )
        bev = base.draw_lane_graph_on_bev(
            bev,
            graph,
            streams,
            pixels_per_meter=base.PIXELS_PER_METER,
        )
    else:
        img_size = int(20 * 2 * base.PIXELS_PER_METER)
        bev = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    bev = base.draw_lane_boundaries_on_bev(
        bev,
        boundaries,
        base.PIXELS_PER_METER,
    )
    timings["bev_render"] = _ms(t0)
    timings["cpu_post_wall"] = _ms(post_start)

    # Modified to return the vehicle arrays for saving
    return bev, vehicle_count, len(streams), all_gt_vehicles, bboxes, labels, timings


def print_frame_profile(frame_id: int, timings: dict[str, float]) -> None:
    keys = [key for key in PROFILE_ORDER if key in timings]
    text = " | ".join(f"{key}={timings[key]:.1f}ms" for key in keys)
    print(f"PROFILE frame={frame_id:03d} | {text}", flush=True)


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def print_summary(samples: list[dict[str, float]], warmup_frames: int = 3) -> None:
    if not samples:
        return

    selected = samples
    label = "all frames"
    if len(samples) > warmup_frames + 2:
        selected = samples[warmup_frames:]
        label = f"excluding first {warmup_frames} warm-up frames"

    values: dict[str, list[float]] = defaultdict(list)
    for sample in selected:
        for key, value in sample.items():
            values[key].append(value)

    print("\n=== Profiling summary: " + label + " ===")
    print(f"Frames measured: {len(selected)}")
    for key in PROFILE_ORDER:
        if key not in values:
            continue
        avg = mean(values[key])
        med = _median(values[key])
        print(f"{key:30s} avg={avg:8.2f} ms   median={med:8.2f} ms")


def overlay_transparent_mask(
    main_frame,
    mask_frame,
    color=(0, 0, 255),
    alpha=0.5,
):
    colored_mask = np.zeros_like(main_frame)
    condition = mask_frame > 127
    colored_mask[condition] = color

    blended = cv2.addWeighted(colored_mask, alpha, main_frame, 1.0, 0)
    result = main_frame.copy()
    result[condition] = blended[condition]
    return result


def _open_mask_videos(mask_dir: str | None, input_path):
    """Open every matching MP4 and map absolute 0-based scene indices to each segment."""
    if not mask_dir:
        return []

    scene_name = os.path.basename(os.path.normpath(input_path))
    matched_files = sorted(glob.glob(os.path.join(mask_dir, f"{scene_name}_frames-*.mp4")))
    segments = []
    for video_path in matched_files:
        filename = os.path.basename(video_path)
        match = re.fullmatch(rf"{re.escape(scene_name)}_frames-(\d+)-(\d+)\.mp4", filename)
        if match is None:
            continue
        start_frame, end_frame = map(int, match.groups())
        if start_frame < 0 or end_frame < start_frame:
            raise ValueError(f"Invalid GT segment range: {filename}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Could not open GT mask video: {video_path}")
        segments.append((start_frame, end_frame, cap, video_path))

    segments.sort(key=lambda segment: (segment[0], segment[1]))
    for previous, current in zip(segments, segments[1:]):
        if current[0] <= previous[1]:
            for _, _, cap, _ in segments:
                cap.release()
            raise ValueError(f"Overlapping GT mask frame ranges: {previous[3]} and {current[3]}")

    for first, last, _, video_path in segments:
        print(f"Found GT mask video: {video_path} (scene frames {first}-{last})")
    return segments


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--mask-dir",
        type=str,
        default=None,
        help="Directory containing mask videos",
    )
    parser.add_argument("--pred-dir", type=str, default=None, help="Directory to save prediction bitmasks")
    parser.add_argument("--vehicles-dir", type=str, default=None, help="Directory to save per-frame vehicle locations in JSON format")
    parser.add_argument("--sweep-config", type=str, default=None, help="JSON file containing sweep parameter overrides")
    # parser.add_argument("--render_dist", type=int, default=None, help="Forward render distance in metres")
    parser.add_argument(
        "--evaluation-only",
        action="store_true",
        help="Generate evaluation masks without demo rendering or video"
    )

    custom_args, remaining_argv = parser.parse_known_args(argv)
    args = base.parse_args(remaining_argv)
    if custom_args.evaluation_only:
        if not custom_args.mask_dir or not custom_args.pred_dir:
            raise ValueError("--evaluation-only requires --mask-dir and --pred-dir")
        if not args.no_display:
            raise ValueError("--evaluation-only requires --no-display")

    root, _camera_order, frames, timestamps, info_path = base.load_scene(args.input)
    print(
        f"Loaded {len(frames)} total frames from {info_path.name}; "
        f"timestamp span {timestamps[-1] - timestamps[0]:.3f}s"
    )

    scene_name = os.path.basename(os.path.normpath(args.input))

    for frame in frames:
        if not isinstance(frame.get("gt_vehicles"), list):
            raise ValueError(
                "Scene PKL does not contain GT vehicles; re-export it with "
                "the schema-v2 create_bevfusion_scene.py"
            )

    mask_segments = _open_mask_videos(custom_args.mask_dir, root)
    if custom_args.mask_dir and not mask_segments:
        raise RuntimeError(f"No valid GT mask videos for scene {scene_name}")
    if mask_segments:
        if mask_segments[-1][1] >= len(frames):
            raise ValueError(
                f"GT mask segment ends at frame {mask_segments[-1][1]}, "
                f"but scene only has {len(frames)} frames"
            )
        # Preserve temporal history across unlabeled gaps by processing frames from zero.
        last_annotated_frame = mask_segments[-1][1]
        frames = frames[:last_annotated_frame + 1]
        timestamps = timestamps[:last_annotated_frame + 1]
        print(f"Processing frames 0-{last_annotated_frame}; scoring annotated frames only")
    segment_index = 0

    lane_pipeline = GTLanePipeline(**make_pipeline_kwargs(
        load_overrides(custom_args.sweep_config)
    ))

    display_enabled = not args.no_display
    window = "nuScenes GT Lane Inference (BEV only)"
    wall_start = None
    profile_samples: list[dict[str, float]] = []
    writer: DirectFFmpegWriter | None = None
    previous_completion: float | None = None

    if display_enabled:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        print("Live controls: p = pause/resume, q = stop")

    if custom_args.pred_dir:
        os.makedirs(custom_args.pred_dir, exist_ok=True)
        print(f"Saving bitmasks to: {custom_args.pred_dir}")
        
    if custom_args.vehicles_dir:
        os.makedirs(custom_args.vehicles_dir, exist_ok=True)
        print(f"Saving per-frame JSON vehicle locations to: {custom_args.vehicles_dir}")

    try:
        for playback_index, frame in enumerate(frames):
            # Scene-index ranges are inclusive and zero-based, as in the MP4 names.
            while segment_index < len(mask_segments) and playback_index > mask_segments[segment_index][1]:
                segment_index += 1
            active_segment = (
                mask_segments[segment_index]
                if segment_index < len(mask_segments)
                and mask_segments[segment_index][0] <= playback_index <= mask_segments[segment_index][1]
                else None
            )
            combined, vehicles, streams, all_gt_vehicles, bboxes, labels, timings = render_gt_frame(
                lane_pipeline,
                frame,
                True if custom_args.pred_dir else False
            )

            # --- SAVE CAR LOCATIONS ---
            if custom_args.vehicles_dir and active_segment is not None:
                frame_vehicles = []
                for v_raw, bbox, label in zip(all_gt_vehicles, bboxes, labels):
                    frame_vehicles.append({
                        "class_name": str(v_raw["class_name"]),
                        "visualisation_label": int(label),
                        "canonical": {
                            "x": float(v_raw.get("x", 0.0)),
                            "y": float(v_raw.get("y", 0.0)),
                            "z": float(v_raw.get("z", 0.0)),
                            "width": float(v_raw.get("width", 0.0)),
                            "length": float(v_raw.get("length", 0.0)),
                            "height": float(v_raw.get("height", 0.0)),
                            "yaw": float(v_raw.get("yaw", 0.0)),
                        },
                        "bev_transformed": {
                            "x_right": float(bbox[0]),
                            "y_forward": float(bbox[1]),
                            "z_up": float(bbox[2]),
                            "width": float(bbox[3]),
                            "length": float(bbox[4]),
                            "height": float(bbox[5]),
                            "yaw_inverted": float(bbox[6]),
                            "vx": float(bbox[7]),
                            "vy": float(bbox[8]),
                        }
                    })
                
                frame_data = {
                    "playback_index": playback_index,
                    "frame_index": int(frame["frame_index"]),
                    "token": frame["token"],
                    "vehicles_count": len(frame_vehicles),
                    "vehicles": frame_vehicles
                }
                
                json_path = os.path.join(custom_args.vehicles_dir, f"{scene_name}_{playback_index:04d}_vehicles.json")
                with open(json_path, "w") as f:
                    json.dump(frame_data, f, indent=4)
            # ---------------------------

            if custom_args.pred_dir and active_segment is not None:
                inf_gray = cv2.cvtColor(combined, cv2.COLOR_BGR2GRAY)
                _, inf_bitmask = cv2.threshold(inf_gray, 1, 255, cv2.THRESH_BINARY)
                
                inf_save_path = os.path.join(custom_args.pred_dir, f"{scene_name}_{playback_index:04d}_pred.png")
                cv2.imwrite(inf_save_path, inf_bitmask)


            if active_segment is not None:
                _, _, mask_cap, mask_path = active_segment
                ret, mask_frame = mask_cap.read()
                if not ret:
                    raise RuntimeError(f"GT mask video ended early: {mask_path} at scene frame {playback_index}")
                if ret:
                    if mask_frame.ndim == 3:
                        mask_frame = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
                    h, w = combined.shape[:2]
                    mask_frame = cv2.resize(
                        mask_frame,
                        (w, h),
                        interpolation=cv2.INTER_NEAREST,
                    )

                    if custom_args.pred_dir:
                        _, gt_bitmask = cv2.threshold(mask_frame, 127, 255, cv2.THRESH_BINARY)
                        gt_save_path = os.path.join(custom_args.pred_dir, f"{scene_name}_{playback_index:04d}_gt.png")
                        cv2.imwrite(gt_save_path, gt_bitmask)

                    combined = overlay_transparent_mask(
                        combined,
                        mask_frame,
                        color=(0, 0, 255),
                        alpha=0.5,
                    )
                else:
                    raise RuntimeError(f"GT mask video ended early: {mask_path}")

            if writer is None:
                writer = DirectFFmpegWriter(
                    args.output,
                    combined.shape,
                    timestamps,
                    args.fps,
                )

            t0 = time.perf_counter()
            writer.submit(combined)
            timings["video_submit"] = _ms(t0)

            completion = time.perf_counter()
            if previous_completion is not None:
                timings["pipeline_interval"] = (
                    completion - previous_completion
                ) * 1000.0
            previous_completion = completion

            profile_samples.append(timings)
            frame_id = int(frame["frame_index"])
            print_frame_profile(frame_id, timings)

            if display_enabled:
                if wall_start is None:
                    wall_start = time.monotonic()
                wall_start, keep_running = base.show_live_frame(
                    window,
                    combined,
                    playback_index,
                    timestamps,
                    args.fps,
                    wall_start,
                )
                if not keep_running:
                    print("Live display stopped; finalising frames produced so far.")
                    break

            print(
                f"[{playback_index + 1:>3}/{len(frames)}] "
                f"{frame['token']} | exported GT vehicles: {len(all_gt_vehicles)} | "
                f"lane-evidence vehicles: {vehicles} | lane streams: {streams}",
                flush=True,
            )

        if writer is None:
            raise RuntimeError("No frames were rendered")

        print_summary(profile_samples)

        output, _writer_wall_ms = writer.close()
        writer = None

        timing = f"{args.fps:g} FPS" if args.fps is not None else "scene timestamps (VFR)"
        print(f"Saved {output} using {timing}")

    finally:
        for _, _, cap, _ in mask_segments:
            cap.release()
        if writer is not None:
            try:
                writer.close()
            except Exception as exc:
                print(f"Warning: video writer cleanup failed: {exc}")
        if display_enabled:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()