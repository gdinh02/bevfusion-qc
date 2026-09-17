from __future__ import annotations

import time
from collections import deque

import numpy as np
import torch
from pyquaternion import Quaternion

from bevfusion_integration.bevfusion_adaptor import (
    extract_vehicles_from_bevfusion,
    vectorized_yaw_filter,
)
from lane_inference.boundary_tracking import update_temporal_lane_tracks
from lane_inference.configs import (
    BEVFusionConfig,
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


def _as_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class LanePipeline:
    def __init__(self):
        self.bevfusion_cfg = BEVFusionConfig()
        self.temporal_cfg = TemporalConfig()
        self.graph_cfg = LaneGraphConfig()
        self.fit_cfg = LaneFitConfig()
        self.merge_cfg = LaneMergeConfig()
        self.lead_cfg = LeadVehicleConfig()
        self.boundary_cfg = LaneBoundaryConfig()
        self.tracking_cfg = BoundaryTrackingConfig()

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

    def _process_numpy(
        self,
        frame_id: int,
        inputs_json: dict,
        bboxes: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
        yaw_mask: np.ndarray | None = None,
        timings: dict[str, float] | None = None,
    ):
        total_start = time.perf_counter()

        def stage_start() -> float:
            return time.perf_counter()

        def stage_end(name: str, start: float) -> None:
            if timings is not None:
                timings[name] = _ms(start)

        t0 = stage_start()
        slope, intercept = estimate_road_height_from_boxes(
            bboxes,
            inputs_json,
        )
        road_plane = {
            "coefficients": np.array(
                [0.0, -slope, -intercept],
                dtype=np.float64,
            ),
            "model": "bevfusion_height_line",
        }
        stage_end("lane_road_plane", t0)

        t0 = stage_start()
        if yaw_mask is None:
            yaw_mask = vectorized_yaw_filter(
                bboxes,
                self.bevfusion_cfg.yaw_filter_direction,
                self.bevfusion_cfg.yaw_filter_span,
            ) # type:ignore
        yaw_mask = np.asarray(yaw_mask, dtype=bool)

        vehicles = extract_vehicles_from_bevfusion(
            bboxes[yaw_mask],
            scores[yaw_mask],
            labels[yaw_mask],
            cfg=self.graph_cfg,
        )
        stage_end("lane_vehicle_extract", t0)

        t0 = stage_start()
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
        stage_end("lane_pose_history", t0)

        t0 = stage_start()
        temporal_vehicles, _ = accumulate_temporal_vehicle_evidence(
            list(self.history),
            reference_record_index=-1,
            cfg=self.temporal_cfg,
        )
        stage_end("lane_temporal_evidence", t0)

        t0 = stage_start()
        graph = build_lane_compatibility_graph_from_vehicles(
            temporal_vehicles,
            cfg=self.graph_cfg,
        )
        stage_end("lane_graph_build", t0)

        t0 = stage_start()
        streams = get_lane_streams(
            graph,
            min_vehicles=self.graph_cfg.min_observations_per_stream,
            min_tracks=self.graph_cfg.min_tracks_per_stream,
            single_track_min_z_span=self.graph_cfg.single_track_min_z_span,
        )

        stage_end("lane_stream_components", t0)

        t0 = stage_start()
        fits = fit_lane_streams(
            graph,
            streams,
            cfg=self.fit_cfg,
        )
        stage_end("lane_fit", t0)

        t0 = stage_start()
        streams, fits, _ = merge_compatible_lane_streams(
            graph,
            streams,
            lane_fits=fits,
            merge_cfg=self.merge_cfg,
            fit_cfg=self.fit_cfg,
        )
        stage_end("lane_merge", t0)

        t0 = stage_start()
        streams, fits, _ = ensure_lead_vehicle_stream(
            graph,
            streams,
            fits,
            current_frame_index=frame_id,
            cfg=self.lead_cfg,
        )
        stage_end("lane_lead_stream", t0)

        t0 = stage_start()
        raw_boundaries = infer_lane_boundaries(
            fits,
            cfg=self.boundary_cfg,
        )
        stage_end("lane_boundary_infer", t0)

        t0 = stage_start()
        boundaries, self.tracker_state, _ = update_temporal_lane_tracks(
            self.tracker_state,
            raw_boundaries,
            pseudo_cam_to_global,
            road_plane,
            frame_id,
            cfg=self.tracking_cfg,
        )
        stage_end("lane_boundary_track", t0)

        if timings is not None:
            timings["lanes"] = _ms(total_start)

        return graph, streams, boundaries, len(vehicles)

    def process(
        self,
        frame_id: int,
        inputs_json: dict,
        bboxes: torch.Tensor | np.ndarray,
        scores: torch.Tensor | np.ndarray,
        labels: torch.Tensor | np.ndarray,
    ):
        return self._process_numpy(
            frame_id=frame_id,
            inputs_json=inputs_json,
            bboxes=_as_numpy(bboxes),
            scores=_as_numpy(scores),
            labels=_as_numpy(labels),
        )


class ProfiledOptimizedLanePipeline(LanePipeline):
    """Lane pipeline with per-stage profiling.

    The core lane logic is inherited from ``LanePipeline`` so the normal and
    profiled paths cannot drift apart. This variant expects the caller to have
    already snapshotted detections to CPU/NumPy, and it may receive a precomputed
    yaw mask.
    """

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

        result = self._process_numpy(
            frame_id=frame_id,
            inputs_json=inputs_json,
            bboxes=np.asarray(bboxes),
            scores=np.asarray(scores),
            labels=np.asarray(labels),
            yaw_mask=yaw_mask,
            timings=timings,
        )

        self.last_profile = timings
        return result
