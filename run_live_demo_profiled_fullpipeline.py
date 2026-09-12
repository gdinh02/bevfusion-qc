#!/usr/bin/env python3
"""BEVFusion live demo with a three-stage inference pipeline.

Pipeline:
    CPU prepare worker: load/decode + BEVFusion preprocessing for frame N+2
    GPU inference worker: BEVFusion inference + one detection CPU snapshot for N+1
    Main CPU thread: lane inference + render + direct FFmpeg submission for frame N

This preserves frame ordering and keeps the temporal lane tracker on one thread.
Place beside:
    run_live_demo.py
    run_live_demo_profiled_prefetch.py
    bevfusion_profiler_prefetch.py
    lane_profile_optimized.py

Example:
    python run_live_demo_profiled_fullpipeline.py --input Z:/dataset/scene-0095 --output Z:/dataset/scene-0095/demo_profiled_pipeline.mp4 --no-display
"""
from __future__ import annotations

import queue
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean

import cv2
import numpy as np
import torch

import run_live_demo as base
from bevfusion_profiler_prefetch import ProfiledPrefetchBEVFusionAppCustom
from lane_profile_optimized import ProfiledOptimizedLanePipeline
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
            # torch.set_default_device() is thread-local for the purposes that
            # matter here. build_app() configures CUDA on the main thread, but
            # BEVFusion/get_bboxes also creates some intermediate tensors without
            # an explicit device. Establish the same device context in this
            # inference worker before running any model code.
            worker_device = torch.device(self.app.device)
            if worker_device.type == "cuda":
                # torch.cuda.set_device() requires an explicit CUDA index.
                # demo_settings.DEVICE is commonly just "cuda", so resolve that
                # to the process' current CUDA device (normally cuda:0).
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

                # All model forward passes stay on this single inference thread.
                with torch.inference_mode():
                    bboxes, scores, labels = self.app.predict_prepared(item.prepared)
                timings.update(dict(self.app.last_profile))

                # Make one ordered CPU snapshot. Once this completes, frame N no
                # longer depends on CUDA and the GPU can start frame N+1 while
                # the main thread handles lane/render work for N.
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
                # Queue blocking happens after the result's inference stage has
                # completed, but is useful to diagnose CPU post-processing being
                # slower than inference.
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
    """CPU-only ordered post-processing for one completed inference result."""
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
    # generate_bev_map currently accepts torch tensors, but these are CPU views
    # over the already-copied NumPy snapshot: there is no CUDA transfer here.
    bev = base.generate_bev_map(
        bboxes=torch.from_numpy(result.bboxes),
        labels=torch.from_numpy(result.labels),
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
        item.images,
        item.cam_paths,
        boundaries,
        result.bboxes,
        result.labels,
        item.inputs_json,
    )
    timings["camera_projection"] = _ms(t0)

    t0 = time.perf_counter()
    combined = base.compose_layout(cameras, bev)
    timings["compose"] = _ms(t0)
    timings["cpu_post_wall"] = _ms(post_start)

    return combined, vehicle_count, len(streams), timings


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

    if "pipeline_interval" in values:
        avg_interval = mean(values["pipeline_interval"])
        if avg_interval > 0:
            print(
                "Estimated overlapped throughput: "
                f"{1000.0 / avg_interval:.2f} FPS"
            )

    print(
        "Note: load/prepare, GPU inference, and cpu_post_wall are concurrent "
        "pipeline stages. Do not add them together. pipeline_interval is the "
        "steady-state frame-completion interval and is the throughput metric."
    )


def main(argv=None):
    args = base.parse_args(argv)

    root, camera_order, frames, timestamps, info_path = base.load_scene(args.input)
    print(
        f"Loaded {len(frames)} frames from {info_path.name}; "
        f"timestamp span {timestamps[-1] - timestamps[0]:.3f}s"
    )

    base.BEVFusionAppCustom = ProfiledPrefetchBEVFusionAppCustom
    app = base.build_app()
    lane_pipeline = ProfiledOptimizedLanePipeline()

    # A slightly deeper prepare queue keeps the inference worker fed while the
    # main thread performs the relatively heavy lane fitting stage.
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
        print(f"FFmpeg finalisation wait: {_ms(finalize_start):.1f} ms")
        print(
            f"Direct MP4 writer wall span: {writer_wall_ms:.1f} ms "
            "(overlapped with pipeline)"
        )

        timing = (
            f"{args.fps:g} FPS"
            if args.fps is not None
            else "scene timestamps (VFR)"
        )
        print(f"Saved {output} using {timing}")

    finally:
        inference_worker.close()
        prefetcher.close()
        if writer is not None:
            try:
                writer.close()
            except Exception as exc:
                print(f"Warning: video writer cleanup failed: {exc}")
        if display_enabled:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
