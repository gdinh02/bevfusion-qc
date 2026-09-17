#!/usr/bin/env python3
"""BEVFusion live demo with a three-stage inference pipeline and mask overlay.

Pipeline:
    CPU prepare worker: load/decode + BEVFusion preprocessing for frame N+2
    GPU inference worker: BEVFusion inference + one detection CPU snapshot for N+1
    Main CPU thread: lane inference + render + mask overlay + direct FFmpeg submission for frame N

    Example Usage: python run_live_demo_profiled_bevonly.py --input Z:/dataset/scene-0956 --output bev_masked.mp4 --mask-dir Z:/dataset/mask
    Example Usage: python run_live_demo_profiled_bevonly.py --input <path-to-nuscenes>/scene-0956 --output bev_masked.mp4 --no-display --mask-dir <path-to-gt-dir>
"""
from __future__ import annotations

import argparse
import glob
import os
import queue
import re
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean

import cv2
import numpy as np
import torch

import run_live_demo_bev as base
from bevfusion_profiler_prefetch import ProfiledPrefetchBEVFusionAppCustom
from pipeline import ProfiledOptimizedLanePipeline
from run_live_demo_profiled_prefetch import (
    DirectFFmpegWriter,
    FramePrefetcher,
    PrefetchedFrame,
    _ms,
    snapshot_detections_once,
)


PROFILE_ORDER = (
    "prefetch_wait_worker",
    "load",
    "camera_geometry_cpu",
    "preprocess_cpu",
    "prepare_wall",
    "input_setup_h2d_gpu",
    "encoder1_gpu",
    "encoder2_gpu",
    "cumsum_segment_gpu",
    "encoder3_gpu",
    "decoder_gpu",
    "decode_nms_filter_gpu",
    "postprocess_wall",
    "inference_wall",
    "model_work_total",
    "yaw_filter_gpu",
    "detections_to_cpu",
    "inference_stage_wall",
    "inference_result_queue_block",
    "inference_wait_main",
    "lane_road_plane",
    "lane_vehicle_extract",
    "lane_pose_history",
    "lane_temporal_evidence",
    "lane_graph_build",
    "lane_stream_components",
    "lane_fit",
    "lane_merge",
    "lane_lead_stream",
    "lane_boundary_infer",
    "lane_boundary_track",
    "lanes",
    "bev_render",
    "camera_projection",
    "compose",
    "video_submit",
    "cpu_post_wall",
    "pipeline_interval",
)


@dataclass
class InferenceResult:
    item: PrefetchedFrame
    bboxes: np.ndarray
    scores: np.ndarray
    labels: np.ndarray
    yaw_mask: np.ndarray
    profile: dict[str, float]


class InferenceWorker:
    """Consume prepared frames and run all CUDA/model work on one thread."""

    _END = object()

    def __init__(
        self,
        app: ProfiledPrefetchBEVFusionAppCustom,
        prefetcher: FramePrefetcher,
        depth: int = 2,
    ) -> None:
        self.app = app
        self.prefetcher = prefetcher
        self.queue: queue.Queue[object] = queue.Queue(maxsize=max(1, depth))
        self.stop_event = threading.Event()
        self.error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._worker,
            name="bevfusion-gpu-inference",
            daemon=True,
        )
        self._thread.start()

    def _put(self, item: object) -> tuple[bool, float]:
        block_start = time.perf_counter()
        while not self.stop_event.is_set():
            try:
                self.queue.put(item, timeout=0.1)
                return True, _ms(block_start)
            except queue.Full:
                continue
        return False, _ms(block_start)

    def _worker(self) -> None:
        try:
            worker_device = torch.device(self.app.device)
            if worker_device.type == "cuda":
                cuda_index = (
                    worker_device.index
                    if worker_device.index is not None
                    else torch.cuda.current_device()
                )
                torch.cuda.set_device(cuda_index)
                worker_device = torch.device("cuda", cuda_index)
            torch.set_default_device(worker_device)

            while not self.stop_event.is_set():
                t0 = time.perf_counter()
                item = self.prefetcher.get()
                prefetch_wait_ms = _ms(t0)
                if item is None:
                    self._put(self._END)
                    return

                timings = dict(item.profile)
                timings["prefetch_wait_worker"] = prefetch_wait_ms
                stage_start = time.perf_counter()

                with torch.inference_mode():
                    bboxes, scores, labels = self.app.predict_prepared(item.prepared)
                timings.update(dict(self.app.last_profile))

                bboxes_np, scores_np, labels_np, yaw_mask_np = snapshot_detections_once(
                    bboxes,
                    scores,
                    labels,
                    timings,
                )
                timings["inference_stage_wall"] = _ms(stage_start)

                result = InferenceResult(
                    item=item,
                    bboxes=bboxes_np,
                    scores=scores_np,
                    labels=labels_np,
                    yaw_mask=np.asarray(yaw_mask_np, dtype=bool),
                    profile=timings,
                )

                ok, block_ms = self._put(result)
                if not ok:
                    return
                result.profile["inference_result_queue_block"] = block_ms
        except BaseException as exc:
            self.error = exc
            self._put(self._END)

    def get(self) -> InferenceResult | None:
        item = self.queue.get()
        try:
            if item is self._END:
                if self.error is not None:
                    raise RuntimeError("Inference worker failed") from self.error
                return None
            assert isinstance(item, InferenceResult)
            return item
        finally:
            self.queue.task_done()

    def close(self) -> None:
        self.stop_event.set()
        self._thread.join(timeout=10.0)


def render_inference_result(
    app: ProfiledPrefetchBEVFusionAppCustom,
    lane_pipeline: ProfiledOptimizedLanePipeline,
    result: InferenceResult,
):
    """CPU-only ordered post-processing for one completed inference result.
    Modified to output ONLY a 20x20m BEV map.
    """
    item = result.item
    timings = dict(result.profile)
    post_start = time.perf_counter()

    graph, streams, boundaries, vehicle_count = lane_pipeline.process(
        item.frame_id,
        item.inputs_json,
        result.bboxes,
        result.scores,
        result.labels,
        yaw_mask=result.yaw_mask,
    )
    timings.update(lane_pipeline.last_profile)

    t0 = time.perf_counter()
    bev = base.generate_bev_map(
        bboxes=torch.from_numpy(result.bboxes),
        labels=torch.from_numpy(result.labels),
        pixels_per_meter=base.PIXELS_PER_METER,
        max_range_meters=20
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
    
    # 3. Crop BEV to exactly 20m x 20m
    target_size_px = int(20.0 * base.PIXELS_PER_METER)
    h, w = bev.shape[:2]
    
    # start_y = max(0, (h - target_size_px) // 2)
    # end_y = min(h, start_y + target_size_px)
    # start_x = max(0, (w - target_size_px) // 2)
    # end_x = min(w, start_x + target_size_px)
    
    # Perform the crop
    bev_20x20 = bev#[start_y:end_y, start_x:end_x]
    
    timings["bev_render"] = _ms(t0)
    timings["cpu_post_wall"] = _ms(post_start)

    return bev_20x20, vehicle_count, len(streams), timings


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


def overlay_transparent_mask(main_frame, mask_frame, color=(0, 0, 255), alpha=0.5):
    colored_mask = np.zeros_like(main_frame)
    condition = mask_frame > 127
    colored_mask[condition] = color
    
    blended = cv2.addWeighted(colored_mask, alpha, main_frame, 1.0, 0)
    result = main_frame.copy()
    result[condition] = blended[condition]
    return result


def main(argv=None):
    # Pre-parse specifically for this script's added arguments
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mask-dir", type=str, default=None, help="Directory containing mask videos")
    custom_args, remaining_argv = parser.parse_known_args(argv)
    
    # Parse standard arguments using remaining argv
    args = base.parse_args(remaining_argv)
    
    # --- Override FPS ---
    args.fps = 2

    root, camera_order, frames, timestamps, info_path = base.load_scene(args.input)
    print(
        f"Loaded {len(frames)} total frames from {info_path.name}; "
        f"timestamp span {timestamps[-1] - timestamps[0]:.3f}s"
    )

    # --- Match mask video and slice frame range based on image_2a0c85.png format ---
    mask_cap = None
    if custom_args.mask_dir:
        scene_name = os.path.basename(os.path.normpath(args.input)) # e.g., "scene-0030"
        
        # Search for files matching the pattern in the provided image (e.g. scene-0030_frames-06-18.mp4)
        search_pattern = os.path.join(custom_args.mask_dir, f"{scene_name}_frames-*.mp4")
        matched_files = glob.glob(search_pattern)
        
        if matched_files:
            mask_video_path = matched_files[0]
            print(f"Found mask video: {mask_video_path}")
            
            # Extract start and end frames from filename using regex
            filename = os.path.basename(mask_video_path)
            match = re.search(r"frames-(\d+)-(\d+)\.mp4", filename)
            
            if match:
                start_frame = int(match.group(1))
                end_frame = int(match.group(2))
                
                # Slice the frames and timestamps to ONLY process this range
                frames = frames[start_frame : end_frame + 1]
                timestamps = timestamps[start_frame : end_frame + 1]
                print(f"Slicing pipeline to process frames {start_frame} to {end_frame} (Total: {len(frames)} frames)")
                
                # Initialize video capture for the mask
                mask_cap = cv2.VideoCapture(mask_video_path)
            else:
                print("Could not parse start/end frames from mask filename. Proceeding with all frames.")
        else:
            print(f"No mask video found in {custom_args.mask_dir} matching {scene_name}. Proceeding without mask.")

    base.BEVFusionAppCustom = ProfiledPrefetchBEVFusionAppCustom
    app = base.build_app()
    lane_pipeline = ProfiledOptimizedLanePipeline()

    prefetcher = FramePrefetcher(app, root, camera_order, frames, depth=3) # type:ignore
    inference_worker = InferenceWorker(app, prefetcher, depth=2) # type:ignore

    display_enabled = not args.no_display
    window = "BEVFusion Live (full pipeline)"
    wall_start = None
    profile_samples: list[dict[str, float]] = []
    writer: DirectFFmpegWriter | None = None
    previous_completion: float | None = None

    if display_enabled:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        print("Live controls: p = pause/resume, q = stop")

    try:
        while True:
            wait_start = time.perf_counter()
            result = inference_worker.get()
            inference_wait_ms = _ms(wait_start)
            if result is None:
                break

            combined, vehicles, streams, timings = render_inference_result(
                app, # type:ignore
                lane_pipeline,
                result,
            )
            timings["inference_wait_main"] = inference_wait_ms

            # --- Mask Overlay Logic ---
            if mask_cap is not None and mask_cap.isOpened():
                ret, mask_frame = mask_cap.read()
                if ret:
                    if len(mask_frame.shape) == 3:
                        mask_frame_gray = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
                    else:
                        mask_frame_gray = mask_frame
                        
                    # Resize mask to exactly match the BEV output dimensions just in case
                    h, w = combined.shape[:2]
                    mask_frame_gray = cv2.resize(mask_frame_gray, (w, h), interpolation=cv2.INTER_NEAREST)
                    
                    # Overlay the mask (Red color with 0.5 opacity)
                    combined = overlay_transparent_mask(combined, mask_frame_gray, color=(0, 0, 255), alpha=0.5)
                else:
                    print("Warning: Mask video ran out of frames before pipeline finished.")

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
            print_frame_profile(result.item.frame_id, timings)

            if display_enabled:
                if wall_start is None:
                    wall_start = time.monotonic()
                wall_start, keep_running = base.show_live_frame(
                    window,
                    combined,
                    result.item.frame_id,
                    timestamps,
                    args.fps,
                    wall_start,
                )
                if not keep_running:
                    print("Live display stopped; finalising frames produced so far.")
                    break

            print(
                f"[{result.item.frame_id + 1:>3}/{len(frames)}] "
                f"{result.item.frame['token']} | vehicles: {vehicles} | "
                f"lane streams: {streams}",
                flush=True,
            )

        if writer is None:
            raise RuntimeError("No frames were rendered")

        print_summary(profile_samples)

        finalize_start = time.perf_counter()
        output, writer_wall_ms = writer.close()
        writer = None
        
        timing = f"{args.fps:g} FPS" if args.fps is not None else "scene timestamps (VFR)"
        print(f"Saved {output} using {timing}")

    finally:
        inference_worker.close()
        prefetcher.close()
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