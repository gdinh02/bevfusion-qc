#!/usr/bin/env python3
"""Profile the existing BEVFusion live demo stage by stage.


    python run_live_demo_profiled.py --input Z:/dataset/scene-0095 --output Z:/dataset/scene-0095/demo_profiled.mp4
"""
from __future__ import annotations

import tempfile
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

import cv2
import torch

import run_live_demo as base
from bevfusion_profiler import ProfiledBEVFusionAppCustom


PROFILE_ORDER = (
    "load",
    "camera_geometry_cpu",
    "preprocess_cpu",
    "input_setup_h2d_gpu",
    "encoder1_gpu",
    "encoder2_gpu",
    "cumsum_segment_gpu",
    "encoder3_gpu",
    "decoder_gpu",
    "decode_nms_filter_gpu",
    "postprocess_wall",
    "model_wall",
    "inference_wall",
    "lanes",
    "bev_render",
    "camera_projection",
    "compose",
    "png_write",
    "compute_wall",
)


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def render_frame_profiled(
    app,
    lane_pipeline,
    frame_id,
    images,
    cam_paths,
    inputs_json,
):
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    with torch.inference_mode():
        bboxes, scores, labels = app.predict_3d_boxes_from_images(
            images,
            cam_paths,
            inputs_json,
        )
    timings["inference_wall"] = _ms(t0)
    timings.update(app.last_profile)

    t0 = time.perf_counter()
    graph, streams, boundaries, vehicle_count = lane_pipeline.process(
        frame_id,
        inputs_json,
        bboxes,
        scores,
        labels,
    )
    timings["lanes"] = _ms(t0)

    t0 = time.perf_counter()
    bev = base.generate_bev_map(
        bboxes=bboxes,
        labels=labels,
        pixels_per_meter=base.PIXELS_PER_METER,
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

    t0 = time.perf_counter()
    cameras = base.project_scene_to_cameras(
        app,
        images,
        cam_paths,
        boundaries,
        bboxes,
        labels,
        inputs_json,
    )
    timings["camera_projection"] = _ms(t0)

    t0 = time.perf_counter()
    combined = base.compose_layout(cameras, bev)
    timings["compose"] = _ms(t0)

    return combined, vehicle_count, len(streams), timings


def print_frame_profile(frame_id: int, timings: dict[str, float]) -> None:
    keys = [key for key in PROFILE_ORDER if key in timings]
    text = " | ".join(f"{key}={timings[key]:.1f}ms" for key in keys)
    print(f"PROFILE frame={frame_id:03d} | {text}", flush=True)


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
        med = sorted(values[key])[len(values[key]) // 2]
        print(f"{key:24s} avg={avg:8.2f} ms   median={med:8.2f} ms")

    if "compute_wall" in values:
        avg_wall = mean(values["compute_wall"])
        if avg_wall > 0:
            print(f"Estimated compute throughput: {1000.0 / avg_wall:.2f} FPS")


def main(argv=None):
    args = base.parse_args(argv)

    root, camera_order, frames, timestamps, info_path = base.load_scene(args.input)
    print(
        f"Loaded {len(frames)} frames from {info_path.name}; "
        f"timestamp span {timestamps[-1] - timestamps[0]:.3f}s"
    )

    # build_app() resolves BEVFusionAppCustom from run_live_demo's module globals
    # at call time, so swapping this symbol gives us a profiled app while keeping
    # the repo's existing model-loading path exactly the same.
    base.BEVFusionAppCustom = ProfiledBEVFusionAppCustom
    app = base.build_app()
    lane_pipeline = base.LanePipeline()

    display_enabled = not args.no_display
    window = "BEVFusion Live (profiled)"
    wall_start = None
    profile_samples: list[dict[str, float]] = []

    if display_enabled:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        print("Live controls: p = pause/resume, q = stop")

    try:
        with tempfile.TemporaryDirectory(prefix="bevfusion-demo-profiled-") as temp:
            frame_dir = Path(temp)
            rendered_paths = []
            rendered_timestamps = []

            for frame_id, frame in enumerate(frames):
                compute_start = time.perf_counter()

                t0 = time.perf_counter()
                inputs_json, images, cam_paths = base.load_frame(
                    root,
                    camera_order,
                    frame,
                )
                load_ms = _ms(t0)

                combined, vehicles, streams, timings = render_frame_profiled(
                    app,
                    lane_pipeline,
                    frame_id,
                    images,
                    cam_paths,
                    inputs_json,
                )
                timings["load"] = load_ms

                path = frame_dir / f"frame_{frame_id:06d}.png"
                t0 = time.perf_counter()
                if not cv2.imwrite(str(path), combined):
                    raise RuntimeError(f"Could not save rendered frame: {path}")
                timings["png_write"] = _ms(t0)
                timings["compute_wall"] = _ms(compute_start)

                rendered_paths.append(path)
                rendered_timestamps.append(timestamps[frame_id])
                profile_samples.append(timings)

                print_frame_profile(frame_id, timings)

                if display_enabled:
                    if wall_start is None:
                        wall_start = time.monotonic()
                    wall_start, keep_running = base.show_live_frame(
                        window,
                        combined,
                        frame_id,
                        timestamps,
                        args.fps,
                        wall_start,
                    )
                    if not keep_running:
                        print(
                            "Live display stopped; encoding frames produced so far."
                        )
                        break

                print(
                    f"[{frame_id + 1:>3}/{len(frames)}] "
                    f"{frame['token']} | vehicles: {vehicles} | "
                    f"lane streams: {streams}",
                    flush=True,
                )

            if not rendered_paths:
                raise RuntimeError("No frames were rendered")

            print_summary(profile_samples)

            encode_start = time.perf_counter()
            output = base.encode_mp4(
                rendered_paths,
                rendered_timestamps,
                args.output,
                args.fps,
            )
            print(f"MP4 encoding wall time: {_ms(encode_start):.1f} ms")

            timing = (
                f"{args.fps:g} FPS"
                if args.fps is not None
                else "scene timestamps (VFR)"
            )
            print(f"Saved {output} using {timing}")

    finally:
        if display_enabled:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
