#!/usr/bin/env python3
"""Run the six-camera BEVFusion demo live on an exported scene folder.

Expected input:
    scene-folder/
      samples/CAM_*/...
      bevfusion_infos_<scene>.pkl

Frames are displayed as soon as inference/visualisation for that frame finishes.
Without --fps, display pacing follows the exported timestamps on a best-effort
basis: if inference is already behind schedule, the frame is shown immediately.
--fps overrides that cadence. The saved MP4 preserves the exported timestamps
(or the FPS override).

python run_live_demo.py --input Z:/dataset/scene-0095 --output Z:/dataset/scene-0095/demo.mp4

python run_live_demo.py --input example --output demo.mp4 --fps 10
"""
from __future__ import annotations

# import warnings
# warnings.filterwarnings("error", message="To copy construct from a tensor")

import argparse
import math
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
from collections import deque
from pathlib import Path
from statistics import median
from typing import cast

import cv2
import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion

sys.path.insert(0, "./packages")

from bevfusion_integration.BEVFusionAppCustom import (
    BEVFusionAppCustom,
    BEVFusionEncoder1Custom,
)
from bevfusion_integration.bev_helper import generate_bev_map
from bevfusion_integration.bevfusion_adaptor import (
    extract_vehicles_from_bevfusion,
    yaw_filter,
)
from demo_settings import DEVICE, PIXELS_PER_METER
from lane_inference.boundary_tracking import update_temporal_lane_tracks
from lane_inference.configs import (
    BoundaryTrackingConfig,
    LaneBoundaryConfig,
    LaneFitConfig,
    LaneGraphConfig,
    LaneMergeConfig,
    LeadVehicleConfig,
    TemporalConfig,
)
from lane_inference.lane_boundaries import infer_lane_boundaries
from lane_inference.lane_fitting import (
    fit_lane_streams,
    merge_compatible_lane_streams,
)
from lane_inference.lane_graph import (
    build_lane_compatibility_graph_from_vehicles,
    ensure_lead_vehicle_stream,
    get_lane_streams,
)
from lane_inference.road_plane import estimate_road_height_from_boxes
from lane_inference.vehicle_tracking import accumulate_temporal_vehicle_evidence
from qai_hub_models.models.bevfusion_det.model import (
    BEVFusion,
    BEVFusionDecoder,
    BEVFusionEncoder2,
    BEVFusionEncoder3,
)
from visualisation import (
    draw_lane_boundaries_on_bev,
    draw_lane_graph_on_bev,
    project_scene_to_cameras,
)

CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
    "CAM_BACK_LEFT",
)
TOP_ROW = ("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT")
BOTTOM_ROW = ("CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT")
CHECKPOINT = Path("checkpoints/camera-only-det.pth")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Scene folder containing samples/ and bevfusion_infos_*.pkl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("demo.mp4"),
        help="Output MP4 path",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override timestamp timing for both saved video and live display",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Save without opening the live display window",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed lane-inference diagnostics",
    )
    args = parser.parse_args(argv)

    if args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0):
        parser.error("--fps must be a finite value greater than zero")
    if args.output.suffix.lower() != ".mp4":
        parser.error("--output must use the .mp4 extension")
    return args


def safe_scene_path(root: Path, relative_value: str) -> Path:
    relative = Path(relative_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe scene-relative path: {relative_value}")

    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path escapes scene directory: {relative_value}") from exc
    return resolved


def load_scene(folder: Path):
    root = folder.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Scene folder not found: {root}")

    info_files = sorted(root.glob("bevfusion_infos_*.pkl"))
    if len(info_files) != 1:
        raise ValueError(
            f"Expected exactly one bevfusion_infos_*.pkl in {root}, "
            f"found {len(info_files)}"
        )

    # Only load PKLs produced by this project / from a trusted source.
    with info_files[0].open("rb") as handle:
        payload = pickle.load(handle)

    meta = payload.get("metainfo", {})
    frames = payload.get("data_list", [])
    if meta.get("schema") != "bevfusion-qc.scene" or meta.get("schema_version") != 1:
        raise ValueError(
            "Unsupported scene PKL; re-export it with create_bevfusion_scene.py"
        )
    if not frames:
        raise ValueError("Scene PKL contains no frames")

    camera_order = tuple(meta.get("camera_order", ()))
    if len(camera_order) != 6 or set(camera_order) != set(CAMERAS):
        raise ValueError("camera_order must contain exactly the six nuScenes cameras")

    timestamps = []
    previous = None
    for index, frame in enumerate(frames):
        if frame.get("frame_index") != index:
            raise ValueError(f"Invalid frame_index at data_list[{index}]")

        if "timestamp" in frame:
            timestamp = float(frame["timestamp"])
        elif "timestamp_us" in frame:
            timestamp = float(frame["timestamp_us"]) / 1e6
        else:
            raise ValueError(f"Frame {index} is missing timestamp/timestamp_us")

        if not math.isfinite(timestamp) or (
            previous is not None and timestamp <= previous
        ):
            raise ValueError("Frame timestamps must increase strictly")

        previous = timestamp
        timestamps.append(timestamp)

        if set(frame.get("cam_paths", {})) != set(CAMERAS):
            raise ValueError(f"Frame {index} does not contain all six camera paths")
        if not isinstance(frame.get("inputs_json"), dict):
            raise ValueError(f"Frame {index} is missing inputs_json")

        for camera in camera_order:
            path = safe_scene_path(root, frame["cam_paths"][camera])
            if not path.is_file():
                raise FileNotFoundError(path)

    return root, camera_order, frames, timestamps, info_files[0]


def load_frame(root: Path, camera_order: tuple[str, ...], frame: dict):
    images = []
    cam_paths = {}

    for camera in camera_order:
        path = safe_scene_path(root, frame["cam_paths"][camera])
        cam_paths[camera] = str(path)
        with Image.open(path) as image:
            images.append(image.convert("RGB").copy())

    return frame["inputs_json"], images, cam_paths


def build_app():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    torch.set_default_device(DEVICE)
    model = BEVFusion.from_pretrained(str(CHECKPOINT))

    enc1 = model.encoder1
    enc2 = model.encoder2
    enc3 = model.encoder3
    dec = model.decoder
    heads = dec.heads
    enc1_shape = enc1.get_input_spec()["imgs"][0]

    return BEVFusionAppCustom(
        cast(BEVFusionEncoder1Custom, enc1),
        cast(BEVFusionEncoder2, enc2),
        cast(BEVFusionEncoder3, enc3),
        cast(BEVFusionDecoder, dec),
        num_classes=heads.num_classes,
        task_heads=heads.task_heads,
        get_bboxes=heads.get_bboxes,
        model_input_shape=(enc1_shape[-2], enc1_shape[-1]),
        score_threshold=0.3,
        device=DEVICE,  # type: ignore[arg-type]
        class_filter=[0, 1, 2],
    )


class LanePipeline:
    def __init__(self):
        self.temporal_cfg = TemporalConfig()
        self.graph_cfg = LaneGraphConfig()
        self.fit_cfg = LaneFitConfig()
        self.merge_cfg = LaneMergeConfig()
        self.lead_cfg = LeadVehicleConfig(enabled=False)
        self.boundary_cfg = LaneBoundaryConfig(verbose=False)
        self.tracking_cfg = BoundaryTrackingConfig(
            emit_unconfirmed=False,
            emit_predicted=False,
        )
        self.history = deque(maxlen=self.temporal_cfg.history_frames)
        self.tracker_state = None
        self.pseudo_cam_to_ego = np.array(
            [
                [0, 0, 1, 0],
                [-1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float64,
        )

    def process(self, frame_id, inputs_json, bboxes, scores, labels):
        slope, intercept = estimate_road_height_from_boxes(
            bboxes.detach().cpu().numpy(),
            inputs_json,
        )
        road_plane = {
            "coefficients": np.array(
                [0.0, -slope, -intercept],
                dtype=np.float64,
            ),
            "model": "bevfusion_height_line",
        }

        mask = torch.tensor(
            yaw_filter(bboxes, np.pi, np.pi / 8),
            dtype=torch.bool,
            device=bboxes.device,
        )

        vehicles = extract_vehicles_from_bevfusion(
            bboxes[mask],
            scores[mask],
            labels[mask],
            cfg=self.graph_cfg,
        )

        ego2global = np.eye(4)
        ego2global[:3, :3] = Quaternion(
            inputs_json["ego2global_rotation"]
        ).rotation_matrix
        ego2global[:3, 3] = np.asarray(
            inputs_json["ego2global_translation"]
        )
        pseudo_cam_to_global = ego2global @ self.pseudo_cam_to_ego

        self.history.append(
            {
                "frame_index": frame_id,
                "vehicles": vehicles,
                "cam_to_global": pseudo_cam_to_global,
            }
        )

        temporal_vehicles, _ = accumulate_temporal_vehicle_evidence(
            list(self.history),
            reference_record_index=-1,
            cfg=self.temporal_cfg,
        )

        graph = build_lane_compatibility_graph_from_vehicles(
            temporal_vehicles,
            cfg=self.graph_cfg,
        )
        streams = get_lane_streams(graph, min_vehicles=2)
        fits = fit_lane_streams(graph, streams, cfg=self.fit_cfg)

        streams, fits, _ = merge_compatible_lane_streams(
            graph,
            streams,
            lane_fits=fits,
            merge_cfg=self.merge_cfg,
            fit_cfg=self.fit_cfg,
        )

        streams, fits, _ = ensure_lead_vehicle_stream(
            graph,
            streams,
            fits,
            current_frame_index=frame_id,
            cfg=self.lead_cfg,
        )

        raw_boundaries = infer_lane_boundaries(
            fits,
            cfg=self.boundary_cfg,
        )

        boundaries, self.tracker_state, _ = update_temporal_lane_tracks(
            self.tracker_state,
            raw_boundaries,
            pseudo_cam_to_global,
            road_plane,
            frame_id,
            cfg=self.tracking_cfg,
        )

        return graph, streams, boundaries, len(vehicles)


def resize_keep_ratio(image: np.ndarray, target_height: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = target_height / h
    width = max(1, int(round(w * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(
        image,
        (width, target_height),
        interpolation=interpolation,
    )


def label_camera(image: np.ndarray, camera: str) -> np.ndarray:
    text = camera.replace("CAM_", "").replace("_", " ")
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        font,
        scale,
        thickness,
    )

    cv2.rectangle(
        image,
        (8, 8),
        (text_width + 24, text_height + baseline + 24),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        image,
        text,
        (16, text_height + 16),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return image


def pad_right(image: np.ndarray, width: int) -> np.ndarray:
    return cv2.copyMakeBorder(
        image,
        0,
        0,
        0,
        max(0, width - image.shape[1]),
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )


def compose_layout(
    cameras: dict[str, np.ndarray],
    bev: np.ndarray,
) -> np.ndarray:
    """2x3 camera grid on the left and BEV on the right, with no stretching."""
    cam_height = bev.shape[0] // 2

    views = {
        name: label_camera(
            resize_keep_ratio(cameras[name], cam_height),
            name,
        )
        for name in CAMERAS
    }

    top = np.hstack([views[name] for name in TOP_ROW])
    bottom = np.hstack([views[name] for name in BOTTOM_ROW])
    width = max(top.shape[1], bottom.shape[1])

    grid = np.vstack(
        (
            pad_right(top, width),
            pad_right(bottom, width),
        )
    )

    if grid.shape[0] < bev.shape[0]:
        grid = cv2.copyMakeBorder(
            grid,
            0,
            bev.shape[0] - grid.shape[0],
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

    combined = np.hstack((grid, bev))

    # yuv420p requires even dimensions. Padding preserves all view ratios.
    if combined.shape[0] % 2 or combined.shape[1] % 2:
        combined = cv2.copyMakeBorder(
            combined,
            0,
            combined.shape[0] % 2,
            0,
            combined.shape[1] % 2,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

    return combined


def render_frame(
    app,
    lane_pipeline,
    frame_id,
    images,
    cam_paths,
    inputs_json,
):
    with torch.inference_mode():
        bboxes, scores, labels = app.predict_3d_boxes_from_images(
            images,
            cam_paths,
            inputs_json,
        )

    graph, streams, boundaries, vehicle_count = lane_pipeline.process(
        frame_id,
        inputs_json,
        bboxes,
        scores,
        labels,
    )

    bev = generate_bev_map(
        bboxes=bboxes,
        labels=labels,
        pixels_per_meter=PIXELS_PER_METER,
    )
    bev = draw_lane_graph_on_bev(
        bev,
        graph,
        streams,
        pixels_per_meter=PIXELS_PER_METER,
    )
    bev = draw_lane_boundaries_on_bev(
        bev,
        boundaries,
        PIXELS_PER_METER,
    )

    cameras = project_scene_to_cameras(
        app,
        images,
        cam_paths,
        boundaries,
        bboxes,
        labels,
        inputs_json,
    )

    return compose_layout(cameras, bev), vehicle_count, len(streams)


def frame_durations(
    timestamps: list[float],
    fps: float | None,
):
    if fps is not None:
        return [1.0 / fps] * len(timestamps)

    if len(timestamps) == 1:
        return [0.5]

    intervals = [
        later - earlier
        for earlier, later in zip(timestamps, timestamps[1:])
    ]
    return intervals + [median(intervals)]


def encode_mp4(
    frame_paths,
    timestamps,
    output: Path,
    fps: float | None,
):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg is required for timestamp-accurate MP4 output "
            "and was not found on PATH"
        )

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    durations = frame_durations(timestamps, fps)
    frame_dir = frame_paths[0].parent
    manifest = frame_dir / "frames.ffconcat"

    with manifest.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("ffconcat version 1.0\n")
        for path, duration in zip(frame_paths, durations):
            handle.write(f"file {path.name}\n")
            handle.write("option framerate 1000000\n")
            handle.write(f"duration {duration:.9f}\n")

        # concat requires the final file twice to apply its duration.
        handle.write(f"file {frame_paths[-1].name}\n")
        handle.write("option framerate 1000000\n")

    base = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        manifest.name,
        "-fps_mode",
        "vfr",
    ]

    encoders = (
        ["-c:v", "libx264", "-preset", "medium", "-crf", "18"],
        ["-c:v", "mpeg4", "-q:v", "2"],
    )

    error = ""
    for encoder in encoders:
        result = subprocess.run(
            base
            + encoder
            + [
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output),
            ],
            cwd=frame_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return output
        error = result.stderr.strip()

    raise RuntimeError(f"ffmpeg failed: {error}")


def live_target_offset(
    frame_index: int,
    timestamps: list[float],
    fps: float | None,
) -> float:
    if fps is not None:
        return frame_index / fps
    return timestamps[frame_index] - timestamps[0]


def wait_until_due(
    window: str,
    wall_start: float,
    target_offset: float,
) -> tuple[float, bool]:
    """Wait only when inference is ahead of the requested playback timeline."""
    while True:
        remaining = wall_start + target_offset - time.monotonic()
        if remaining <= 0:
            return wall_start, True

        key = cv2.waitKey(
            max(1, min(50, math.ceil(remaining * 1000)))
        ) & 0xFF

        if key == ord("q"):
            return wall_start, False

        if key == ord("p"):
            pause_started = time.monotonic()
            key = cv2.waitKey(0) & 0xFF
            wall_start += time.monotonic() - pause_started

            if key == ord("q"):
                return wall_start, False

        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            return wall_start, False


def show_live_frame(
    window: str,
    image: np.ndarray,
    frame_index: int,
    timestamps: list[float],
    fps: float | None,
    wall_start: float,
) -> tuple[float, bool]:
    target_offset = live_target_offset(
        frame_index,
        timestamps,
        fps,
    )

    wall_start, keep_running = wait_until_due(
        window,
        wall_start,
        target_offset,
    )
    if not keep_running:
        return wall_start, False

    # If inference is behind schedule, this happens immediately.
    cv2.imshow(window, image)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        return wall_start, False

    if key == ord("p"):
        pause_started = time.monotonic()
        key = cv2.waitKey(0) & 0xFF
        wall_start += time.monotonic() - pause_started

        if key == ord("q"):
            return wall_start, False

    if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
        return wall_start, False

    return wall_start, True


def main(argv=None):
    args = parse_args(argv)

    root, camera_order, frames, timestamps, info_path = load_scene(
        args.input
    )

    print(
        f"Loaded {len(frames)} frames from {info_path.name}; "
        f"timestamp span {timestamps[-1] - timestamps[0]:.3f}s"
    )

    app = build_app()
    lane_pipeline = LanePipeline()

    display_enabled = not args.no_display
    window = "BEVFusion Live"
    wall_start = None

    if display_enabled:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        print("Live controls: p = pause/resume, q = stop")

    try:
        with tempfile.TemporaryDirectory(
            prefix="bevfusion-demo-"
        ) as temp:
            frame_dir = Path(temp)
            rendered_paths = []
            rendered_timestamps = []

            for frame_id, frame in enumerate(frames):
                inputs_json, images, cam_paths = load_frame(
                    root,
                    camera_order,
                    frame,
                )

                combined, vehicles, streams = render_frame(
                    app,
                    lane_pipeline,
                    frame_id,
                    images,
                    cam_paths,
                    inputs_json,
                )

                # Store the rendered frame for timestamp-accurate MP4 encoding.
                path = frame_dir / f"frame_{frame_id:06d}.png"
                if not cv2.imwrite(str(path), combined):
                    raise RuntimeError(
                        f"Could not save rendered frame: {path}"
                    )

                rendered_paths.append(path)
                rendered_timestamps.append(timestamps[frame_id])

                # Display this frame now, rather than replaying after inference.
                if display_enabled:
                    if wall_start is None:
                        wall_start = time.monotonic()

                    wall_start, keep_running = show_live_frame(
                        window,
                        combined,
                        frame_id,
                        timestamps,
                        args.fps,
                        wall_start,
                    )

                    if not keep_running:
                        print(
                            "Live display stopped; encoding frames "
                            "produced so far."
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

            output = encode_mp4(
                rendered_paths,
                rendered_timestamps,
                args.output,
                args.fps,
            )

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
