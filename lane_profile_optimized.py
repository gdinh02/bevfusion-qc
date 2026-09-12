from __future__ import annotations

import time
from collections import deque

import numpy as np
import torch
from pyquaternion import Quaternion

from bevfusion_integration.bevfusion_adaptor import extract_vehicles_from_bevfusion
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


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def vectorized_yaw_filter(
    bboxes: torch.Tensor | np.ndarray,
    direction: float = np.pi,
    span: float = np.pi / 12,
) -> torch.Tensor | np.ndarray:
    """Vectorised equivalent of bevfusion_adaptor.yaw_filter.

    For CUDA tensors, this stays entirely on the GPU and returns a CUDA boolean
    tensor.  It avoids the original Python loop's ``float(box[6])`` conversion,
    which synchronises the GPU once per detection.
    """
    if torch.is_tensor(bboxes):
        yaw = bboxes[:, 6]
        direction_t = yaw.new_tensor(direction)
        half_pi = yaw.new_tensor(np.pi / 2)
        pi = yaw.new_tensor(np.pi)
        wrapped = torch.remainder(yaw - direction_t + half_pi, pi) - half_pi
        return torch.abs(wrapped) <= span

    bboxes_np = np.asarray(bboxes)
    yaw = bboxes_np[:, 6]
    wrapped = np.remainder(yaw - direction + np.pi / 2, np.pi) - np.pi / 2
    return np.abs(wrapped) <= span


class ProfiledOptimizedLanePipeline:
    """Lane pipeline that consumes one CPU/NumPy detection snapshot per frame.

    This preserves the original lane logic while avoiding repeated CUDA->CPU
    conversions inside road estimation, vehicle extraction, BEV rendering and
    camera projection.  Per-stage timing is exposed in ``last_profile``.
    """

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
        self.last_profile: dict[str, float] = {}

    def process(
        self,
        frame_id: int,
        inputs_json: dict,
        bboxes: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
        yaw_mask: np.ndarray | None = None,
    ):
        timings: dict[str, float] = {}
        total_start = time.perf_counter()

        t0 = time.perf_counter()
        slope, intercept = estimate_road_height_from_boxes(bboxes, inputs_json)
        road_plane = {
            "coefficients": np.array(
                [0.0, -slope, -intercept],
                dtype=np.float64,
            ),
            "model": "bevfusion_height_line",
        }
        timings["lane_road_plane"] = _ms(t0)

        t0 = time.perf_counter()
        if yaw_mask is None:
            yaw_mask = vectorized_yaw_filter(bboxes, np.pi, np.pi / 8)  # type: ignore
        yaw_mask = np.asarray(yaw_mask, dtype=bool)
        vehicles = extract_vehicles_from_bevfusion(
            bboxes[yaw_mask],
            scores[yaw_mask],
            labels[yaw_mask],
            cfg=self.graph_cfg,
        )
        timings["lane_vehicle_extract"] = _ms(t0)

        t0 = time.perf_counter()
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
        timings["lane_pose_history"] = _ms(t0)

        t0 = time.perf_counter()
        temporal_vehicles, _ = accumulate_temporal_vehicle_evidence(
            list(self.history),
            reference_record_index=-1,
            cfg=self.temporal_cfg,
        )
        timings["lane_temporal_evidence"] = _ms(t0)

        t0 = time.perf_counter()
        graph = build_lane_compatibility_graph_from_vehicles(
            temporal_vehicles,
            cfg=self.graph_cfg,
        )
        timings["lane_graph_build"] = _ms(t0)

        t0 = time.perf_counter()
        streams = get_lane_streams(graph, min_vehicles=2)
        timings["lane_stream_components"] = _ms(t0)

        t0 = time.perf_counter()
        fits = fit_lane_streams(graph, streams, cfg=self.fit_cfg)
        timings["lane_fit"] = _ms(t0)

        t0 = time.perf_counter()
        streams, fits, _ = merge_compatible_lane_streams(
            graph,
            streams,
            lane_fits=fits,
            merge_cfg=self.merge_cfg,
            fit_cfg=self.fit_cfg,
        )
        timings["lane_merge"] = _ms(t0)

        t0 = time.perf_counter()
        streams, fits, _ = ensure_lead_vehicle_stream(
            graph,
            streams,
            fits,
            current_frame_index=frame_id,
            cfg=self.lead_cfg,
        )
        timings["lane_lead_stream"] = _ms(t0)

        t0 = time.perf_counter()
        raw_boundaries = infer_lane_boundaries(
            fits,
            cfg=self.boundary_cfg,
        )
        timings["lane_boundary_infer"] = _ms(t0)

        t0 = time.perf_counter()
        boundaries, self.tracker_state, _ = update_temporal_lane_tracks(
            self.tracker_state,
            raw_boundaries,
            pseudo_cam_to_global,
            road_plane,
            frame_id,
            cfg=self.tracking_cfg,
        )
        timings["lane_boundary_track"] = _ms(t0)

        timings["lanes"] = _ms(total_start)
        self.last_profile = timings
        return graph, streams, boundaries, len(vehicles)
