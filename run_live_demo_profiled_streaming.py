#!/usr/bin/env python3
"""Profile BEVFusion while streaming rendered frames directly to FFmpeg.

Place this file beside ``run_live_demo.py``, ``bevfusion_profiler.py`` and
``lane_profile_optimized.py`` in the repository root.

This is based on ``run_live_demo_profiled_laneopt.py`` but removes the temporary
PNG sequence entirely. Rendered BGR frames are queued to a background writer
thread, which feeds raw video directly to FFmpeg while the next frame is being
processed.

Timestamp behaviour is preserved:
  * with ``--fps``, FFmpeg receives a constant-rate raw-video stream;
  * without ``--fps``, explicit per-frame PTS values are assigned from the
    exported scene timestamps, preserving variable frame timing.

Example:
    python run_live_demo_profiled_streaming.py --input Z:/dataset/scene-0095 --output Z:/dataset/scene-0095/demo_profiled_opt.mp4 --no-display
"""
from __future__ import annotations

import queue
import shutil
import subprocess
import threading
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

import cv2
import numpy as np
import torch

import run_live_demo as base
from bevfusion_profiler import ProfiledBEVFusionAppCustom
from lane_profile_optimized import (
    ProfiledOptimizedLanePipeline,
    vectorized_yaw_filter,
)


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
    "yaw_filter_gpu",
    "detections_to_cpu",
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
    "compute_wall",
)


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _setpts_expression(pts_us: list[int]) -> str:
    """Build an FFmpeg setpts expression mapping frame number to microseconds."""
    if not pts_us:
        raise ValueError("pts_us must not be empty")

    expression = str(int(pts_us[-1]))
    for index in range(len(pts_us) - 2, -1, -1):
        expression = f"if(eq(N,{index}),{int(pts_us[index])},{expression})"
    return expression


def _choose_encoder(ffmpeg: str) -> list[str]:
    """Use the same preferred encoder as the original demo when available."""
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        check=False,
    )
    if "libx264" in result.stdout:
        return ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "18"]
    return ["-c:v", "mpeg4", "-q:v", "2"]


class DirectFFmpegWriter:
    """Background raw-BGR -> FFmpeg -> MP4 writer.

    ``submit`` normally returns almost immediately because FFmpeg encoding occurs
    on a separate thread. The queue is bounded so a pathologically slow encoder
    cannot consume unbounded RAM; if FFmpeg falls behind far enough, ``submit``
    will provide intentional back-pressure and that time will appear in the
    ``video_submit`` profile stage.
    """

    _STOP = object()

    def __init__(
        self,
        output: Path,
        frame_shape: tuple[int, int, int],
        timestamps: list[float],
        fps: float | None,
        queue_size: int = 4,
    ) -> None:
        height, width, channels = frame_shape
        if channels != 3:
            raise ValueError(f"Expected BGR frame with 3 channels, got {frame_shape}")
        if width % 2 or height % 2:
            raise ValueError("MP4 yuv420p output requires even frame dimensions")

        self.output = output.expanduser().resolve()
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.height = height
        self.timestamps = timestamps
        self.fps = fps
        self.queue: queue.Queue[object] = queue.Queue(maxsize=queue_size)
        self.error: BaseException | None = None
        self.frames_written = 0
        self._last_frame: np.ndarray | None = None
        self._closed = False
        self._start_time = time.perf_counter()

        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg is required and was not found on PATH")

        command = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s:v",
            f"{width}x{height}",
        ]

        if fps is not None:
            command += ["-framerate", f"{fps:.12g}", "-i", "pipe:0"]
            output_timing_args = ["-fps_mode", "cfr"]
            self._duplicate_final_frame = False
        else:
            # Input cadence is arbitrary because setpts below replaces the PTS.
            command += ["-framerate", "30", "-i", "pipe:0"]

            durations = base.frame_durations(timestamps, None)
            first_ts = timestamps[0]
            pts_us = [
                int(round((timestamp - first_ts) * 1_000_000.0))
                for timestamp in timestamps
            ]
            # As in the old concat writer, append the last frame a second time
            # so the final real frame receives its intended display duration.
            final_end_us = int(
                round(
                    (timestamps[-1] - first_ts + durations[-1])
                    * 1_000_000.0
                )
            )
            pts_us.append(final_end_us)

            expression = _setpts_expression(pts_us)
            vf = f"settb=expr=1/1000000,setpts='{expression}'"
            output_timing_args = ["-vf", vf, "-fps_mode", "vfr"]
            self._duplicate_final_frame = True

        command += output_timing_args
        command += _choose_encoder(ffmpeg)
        command += [
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self.output),
        ]

        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if self._process.stdin is None or self._process.stderr is None:
            self._process.kill()
            raise RuntimeError("Could not open FFmpeg stdin/stderr pipes")

        self._thread = threading.Thread(
            target=self._writer_loop,
            name="ffmpeg-raw-video-writer",
            daemon=True,
        )
        self._thread.start()

    def _write_frame(self, frame: np.ndarray) -> None:
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(
                "Frame dimensions changed during stream: "
                f"expected {(self.height, self.width, 3)}, got {frame.shape}"
            )
        if frame.dtype != np.uint8:
            raise ValueError(f"Expected uint8 BGR frame, got {frame.dtype}")

        contiguous = np.ascontiguousarray(frame)
        assert self._process.stdin is not None
        self._process.stdin.write(contiguous.tobytes())
        self.frames_written += 1

    def _writer_loop(self) -> None:
        try:
            while True:
                item = self.queue.get()
                try:
                    if item is self._STOP:
                        break
                    frame = item
                    assert isinstance(frame, np.ndarray)
                    self._write_frame(frame)
                    self._last_frame = frame
                finally:
                    self.queue.task_done()

            if self._duplicate_final_frame and self._last_frame is not None:
                self._write_frame(self._last_frame)

            assert self._process.stdin is not None
            self._process.stdin.close()
            stderr = self._process.stderr.read() # type: ignore
            return_code = self._process.wait()
            if return_code != 0:
                message = stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"ffmpeg failed: {message}")
        except BaseException as exc:  # surfaced on submit/close in main thread
            self.error = exc
            try:
                if self._process.stdin is not None and not self._process.stdin.closed:
                    self._process.stdin.close()
            except OSError:
                pass
            if self._process.poll() is None:
                self._process.kill()
                self._process.wait()

    def submit(self, frame: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("Cannot submit to a closed video writer")
        if self.error is not None:
            raise RuntimeError("FFmpeg writer failed") from self.error
        self.queue.put(frame)
        if self.error is not None:
            raise RuntimeError("FFmpeg writer failed") from self.error

    def close(self) -> tuple[Path, float]:
        if self._closed:
            return self.output, (time.perf_counter() - self._start_time) * 1000.0
        self._closed = True

        self.queue.put(self._STOP)
        self._thread.join()
        if self.error is not None:
            raise RuntimeError("FFmpeg writer failed") from self.error
        return self.output, (time.perf_counter() - self._start_time) * 1000.0


def snapshot_detections_once(
    bboxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    timings: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorise yaw filtering, then make one reusable CPU snapshot."""
    use_cuda_events = bboxes.is_cuda and torch.cuda.is_available()

    yaw_start = yaw_end = None
    if use_cuda_events:
        yaw_start = torch.cuda.Event(enable_timing=True)
        yaw_end = torch.cuda.Event(enable_timing=True)
        yaw_start.record()

    yaw_mask = vectorized_yaw_filter(bboxes, np.pi, np.pi / 8)

    if use_cuda_events:
        yaw_end.record() # type:ignore

    t0 = time.perf_counter()
    bboxes_np = bboxes.detach().cpu().numpy()
    scores_np = scores.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    if torch.is_tensor(yaw_mask):
        yaw_mask_np = yaw_mask.detach().cpu().numpy()
    else:
        yaw_mask_np = np.asarray(yaw_mask)
    timings["detections_to_cpu"] = _ms(t0)

    if use_cuda_events and yaw_start is not None and yaw_end is not None:
        timings["yaw_filter_gpu"] = yaw_start.elapsed_time(yaw_end)

    return bboxes_np, scores_np, labels_np, yaw_mask_np


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

    bboxes_np, scores_np, labels_np, yaw_mask_np = snapshot_detections_once(
        bboxes,
        scores,
        labels,
        timings,
    )

    graph, streams, boundaries, vehicle_count = lane_pipeline.process(
        frame_id,
        inputs_json,
        bboxes_np,
        scores_np,
        labels_np,
        yaw_mask=yaw_mask_np,
    )
    timings.update(lane_pipeline.last_profile)

    t0 = time.perf_counter()
    bev = base.generate_bev_map(
        bboxes=torch.from_numpy(bboxes_np),
        labels=torch.from_numpy(labels_np),
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
        bboxes_np,
        labels_np,
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

    base.BEVFusionAppCustom = ProfiledBEVFusionAppCustom
    app = base.build_app()
    lane_pipeline = ProfiledOptimizedLanePipeline()

    display_enabled = not args.no_display
    window = "BEVFusion Live (profiled direct MP4)"
    wall_start = None
    profile_samples: list[dict[str, float]] = []
    writer: DirectFFmpegWriter | None = None
    output: Path | None = None

    if display_enabled:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        print("Live controls: p = pause/resume, q = stop")

    try:
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
            timings["compute_wall"] = _ms(compute_start)

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
                    print("Live display stopped; finalising frames produced so far.")
                    break

            print(
                f"[{frame_id + 1:>3}/{len(frames)}] "
                f"{frame['token']} | vehicles: {vehicles} | "
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
        print(f"Direct MP4 writer wall span: {writer_wall_ms:.1f} ms (overlapped with compute)")

        timing = (
            f"{args.fps:g} FPS"
            if args.fps is not None
            else "scene timestamps (VFR)"
        )
        print(f"Saved {output} using {timing}")

    finally:
        if writer is not None:
            try:
                writer.close()
            except Exception as exc:
                print(f"Warning: failed to finalise FFmpeg writer: {exc}")
        if display_enabled:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
