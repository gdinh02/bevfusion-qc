#!/usr/bin/env python3
"""nuScenes-GT lane demo using the existing square BEV visualisation.

The exported scene PKL is the only inference input. BEVFusion model loading and
inference are not used. The existing BEV renderer is retained by adapting the
canonical GT vehicles to the box layout it already accepts.

Examples:
    python run_live_demo_profiled_gt_bevonly.py \
        --input Z:/dataset/scene-0956 --output gt_bev.mp4

    python run_live_demo_profiled_gt_bevonly.py `
        --input Z:/dataset/scene-0956 --output gt_bev.mp4 `
        --mask-dir Z:/dataset/mask --no-display 
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import time
from collections import defaultdict
from statistics import mean

import cv2
import numpy as np
import torch

import run_live_demo_bev as base
from nuscenes_gt_integration.gt_adaptor import extract_vehicles_from_gt
from nuscenes_gt_integration.pipeline import GTLanePipeline
from run_live_demo_profiled_prefetch import DirectFFmpegWriter, _ms


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
):
    """Run GT lane inference and render the existing 40 x 40 metre BEV."""
    timings: dict[str, float] = {}
    post_start = time.perf_counter()

    t0 = time.perf_counter()
    graph, streams, boundaries, vehicle_count = lane_pipeline.process(frame)
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
    bev = base.draw_lane_boundaries_on_bev(
        bev,
        boundaries,
        base.PIXELS_PER_METER,
    )
    timings["bev_render"] = _ms(t0)
    timings["cpu_post_wall"] = _ms(post_start)

    return bev, vehicle_count, len(streams), len(all_gt_vehicles), timings


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


def _open_mask_video(mask_dir: str | None, input_path):
    """Open the existing optional scene mask and return its frame range."""
    if not mask_dir:
        return None, None

    scene_name = os.path.basename(os.path.normpath(input_path))
    search_pattern = os.path.join(mask_dir, f"{scene_name}_frames-*.mp4")
    matched_files = sorted(glob.glob(search_pattern))

    if not matched_files:
        print(
            f"No mask video found in {mask_dir} matching {scene_name}. "
            "Proceeding without mask."
        )
        return None, None

    mask_video_path = matched_files[0]
    print(f"Found mask video: {mask_video_path}")

    match = re.search(r"frames-(\d+)-(\d+)\.mp4$", os.path.basename(mask_video_path))
    if match is None:
        print(
            "Could not parse start/end frames from mask filename. "
            "Proceeding without mask."
        )
        return None, None

    frame_range = (int(match.group(1)), int(match.group(2)))
    mask_cap = cv2.VideoCapture(mask_video_path)
    if not mask_cap.isOpened():
        mask_cap.release()
        raise RuntimeError(f"Could not open mask video: {mask_video_path}")

    return mask_cap, frame_range


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--mask-dir",
        type=str,
        default=None,
        help="Directory containing mask videos",
    )
    custom_args, remaining_argv = parser.parse_known_args(argv)
    args = base.parse_args(remaining_argv)

    root, _camera_order, frames, timestamps, info_path = base.load_scene(args.input)
    print(
        f"Loaded {len(frames)} total frames from {info_path.name}; "
        f"timestamp span {timestamps[-1] - timestamps[0]:.3f}s"
    )

    for frame in frames:
        if not isinstance(frame.get("gt_vehicles"), list):
            raise ValueError(
                "Scene PKL does not contain GT vehicles; re-export it with "
                "the schema-v2 create_bevfusion_scene.py"
            )

    mask_cap, frame_range = _open_mask_video(custom_args.mask_dir, root)
    if frame_range is not None:
        start_frame, end_frame = frame_range
        if start_frame < 0 or end_frame < start_frame or end_frame >= len(frames):
            raise ValueError(
                f"Mask frame range {start_frame}-{end_frame} is outside "
                f"the exported scene range 0-{len(frames) - 1}"
            )
        frames = frames[start_frame : end_frame + 1]
        timestamps = timestamps[start_frame : end_frame + 1]
        print(
            f"Slicing pipeline to process frames {start_frame} to {end_frame} "
            f"(total: {len(frames)} frames)"
        )

    lane_pipeline = GTLanePipeline()

    display_enabled = not args.no_display
    window = "nuScenes GT Lane Inference (BEV only)"
    wall_start = None
    profile_samples: list[dict[str, float]] = []
    writer: DirectFFmpegWriter | None = None
    previous_completion: float | None = None

    if display_enabled:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        print("Live controls: p = pause/resume, q = stop")

    try:
        for playback_index, frame in enumerate(frames):
            combined, vehicles, streams, exported_vehicles, timings = render_gt_frame(
                lane_pipeline,
                frame,
            )

            if mask_cap is not None and mask_cap.isOpened():
                ret, mask_frame = mask_cap.read()
                if ret:
                    if mask_frame.ndim == 3:
                        mask_frame = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
                    h, w = combined.shape[:2]
                    mask_frame = cv2.resize(
                        mask_frame,
                        (w, h),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    combined = overlay_transparent_mask(
                        combined,
                        mask_frame,
                        color=(0, 0, 255),
                        alpha=0.5,
                    )
                else:
                    print("Warning: Mask video ran out of frames before the pipeline.")

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
                f"{frame['token']} | exported GT vehicles: {exported_vehicles} | "
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
        if mask_cap is not None:
            mask_cap.release()
        if writer is not None:
            try:
                writer.close()
            except Exception as exc:
                print(f"Warning: video writer cleanup failed: {exc}")
        if display_enabled:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
