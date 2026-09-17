from __future__ import annotations

import math
from collections import deque

from lane_inference.boundary_tracking import (
    update_temporal_lane_tracks,
)
from lane_inference.configs import (
    BoundaryTrackingConfig,
    LaneBoundaryConfig,
    LaneFitConfig,
    LaneGraphConfig,
    LaneMergeConfig,
    LeadVehicleConfig,
    TemporalConfig,
)
from lane_inference.geometry import axial_angle_diff
from lane_inference.lane_boundaries import (
    infer_lane_boundaries,
)
from lane_inference.lane_fitting import (
    fit_lane_streams,
    merge_compatible_lane_streams,
)
from lane_inference.lane_graph import (
    build_lane_compatibility_graph_from_vehicles,
    ensure_lead_vehicle_stream,
    get_lane_streams,
)

from nuscenes_gt_integration.configs import (
    GTLaneEvidenceConfig,
)
from nuscenes_gt_integration.gt_adaptor import (
    extract_vehicles_from_gt,
    get_lane_to_global,
)
from nuscenes_gt_integration.road_plane import (
    estimate_road_plane_from_gt_vehicles,
)
from nuscenes_gt_integration.vehicle_tracking import (
    accumulate_temporal_gt_vehicle_evidence,
)


class GTLanePipeline:
    """
    Lane inference using perfect nuScenes GT vehicle detections and
    perfect nuScenes instance-token temporal association.

    Detection and association errors are therefore removed from the
    experiment, leaving the lane post-processing stack as the primary
    system under evaluation.
    """

    def __init__(
        self,
        *,
        evidence_cfg=None,
        temporal_cfg=None,
        graph_cfg=None,
        fit_cfg=None,
        merge_cfg=None,
        lead_cfg=None,
        boundary_cfg=None,
        tracking_cfg=None,
    ):
        self.evidence_cfg = (
            evidence_cfg
            or GTLaneEvidenceConfig()
        )
        self.temporal_cfg = (
            temporal_cfg
            or TemporalConfig()
        )
        self.graph_cfg = (
            graph_cfg
            or LaneGraphConfig()
        )
        self.fit_cfg = (
            fit_cfg
            or LaneFitConfig()
        )
        self.merge_cfg = (
            merge_cfg
            or LaneMergeConfig()
        )
        self.lead_cfg = (
            lead_cfg
            or LeadVehicleConfig()
        )
        self.boundary_cfg = (
            boundary_cfg
            or LaneBoundaryConfig()
        )
        self.tracking_cfg = (
            tracking_cfg
            or BoundaryTrackingConfig()
        )

        self.history = deque(
            maxlen=self.temporal_cfg.history_frames
        )

        self.tracker_state = None
        self.scene_token = None

        # Diagnostics useful for later evaluation.
        self.last_tracks = {}
        self.last_road_plane = None
        self.last_current_vehicles = []
        self.last_temporal_vehicles = []
        self.last_graph = None
        self.last_streams = []
        self.last_fits = []
        self.last_raw_boundaries = []
        self.last_boundaries = []

    def reset(self):
        """
        Clear all scene-temporal state.

        This is important during dataset evaluation: evidence must never
        leak between nuScenes scenes.
        """
        self.history = deque(
            maxlen=self.temporal_cfg.history_frames
        )

        self.tracker_state = None
        self.scene_token = None

        self.last_tracks = {}
        self.last_road_plane = None
        self.last_current_vehicles = []
        self.last_temporal_vehicles = []
        self.last_graph = None
        self.last_streams = []
        self.last_fits = []
        self.last_raw_boundaries = []
        self.last_boundaries = []

    def _handle_scene_boundary(
        self,
        frame: dict,
    ) -> None:
        scene_token = frame.get("scene_token")

        if self.scene_token is None:
            self.scene_token = scene_token
            return

        if (
            scene_token is not None
            and scene_token != self.scene_token
        ):
            self.reset()
            self.scene_token = scene_token

    def _filter_lane_evidence(
        self,
        vehicles: list[dict],
    ) -> list[dict]:
        """
        Apply lane-evidence gates to otherwise perfect detections.

        No confidence filtering is performed: GT score is always 1.
        """

        max_yaw_diff = math.radians(
            self.evidence_cfg.max_forward_yaw_diff_deg
        )

        # Canonical x-right/z-forward convention:
        # yaw = pi/2 points along +z.
        forward_yaw = 0.5 * math.pi

        filtered = []

        for vehicle in vehicles:
            z = float(vehicle["z"])

            # Preserve the existing BEVFusion lane behaviour, which uses
            # evidence both ahead and behind the reference vehicle.
            if abs(z) > self.graph_cfg.max_depth:
                continue

            if (
                self.evidence_cfg.enable_yaw_filter
                and axial_angle_diff(
                    float(vehicle["yaw"]),
                    forward_yaw,
                )
                > max_yaw_diff
            ):
                continue

            filtered.append(vehicle)

        return filtered

    def process(
        self,
        frame: dict,
    ):
        """
        Process one exported nuScenes frame.

        Expected frame fields:
            frame_index
            scene_token
            inputs_json
            gt_vehicles

        Returns the same main tuple shape as the existing LanePipeline:
            graph, streams, boundaries, current_vehicle_count
        """

        self._handle_scene_boundary(frame)

        frame_id = int(frame["frame_index"])
        inputs_json = frame["inputs_json"]

        # -----------------------------------------------------
        # 1. Perfect GT detection -> canonical lane coordinates
        # -----------------------------------------------------

        all_gt_vehicles = extract_vehicles_from_gt(
            frame["gt_vehicles"],
            inputs_json,
        )

        # -----------------------------------------------------
        # 2. Estimate road plane from GT bounding-box bottoms
        # -----------------------------------------------------

        road_plane = (
            estimate_road_plane_from_gt_vehicles(
                all_gt_vehicles,
                inputs_json,
            )
        )

        # -----------------------------------------------------
        # 3. Lane-evidence filtering
        # -----------------------------------------------------

        current_vehicles = (
            self._filter_lane_evidence(
                all_gt_vehicles
            )
        )

        # -----------------------------------------------------
        # 4. Exact canonical-frame pose
        # -----------------------------------------------------

        lane_to_global = get_lane_to_global(
            inputs_json
        )

        # -----------------------------------------------------
        # 5. Temporal history
        # -----------------------------------------------------

        self.history.append(
            {
                "frame_index": frame_id,
                "vehicles": current_vehicles,

                # Historical name retained because the shared transform
                # function currently expects this dictionary key.
                "cam_to_global": lane_to_global,
            }
        )

        temporal_vehicles, tracks = (
            accumulate_temporal_gt_vehicle_evidence(
                list(self.history),
                reference_record_index=-1,
                cfg=self.temporal_cfg,
            )
        )

        # -----------------------------------------------------
        # 6. Existing lane post-processing
        # -----------------------------------------------------

        graph = (
            build_lane_compatibility_graph_from_vehicles(
                temporal_vehicles,
                cfg=self.graph_cfg,
            )
        )

        streams = get_lane_streams(
            graph,
            min_vehicles=self.graph_cfg.min_observations_per_stream,
            min_tracks=self.graph_cfg.min_tracks_per_stream,
            single_track_min_z_span=self.graph_cfg.single_track_min_z_span,
        )

        fits = fit_lane_streams(
            graph,
            streams,
            cfg=self.fit_cfg,
        )

        streams, fits, _ = (
            merge_compatible_lane_streams(
                graph,
                streams,
                lane_fits=fits,
                merge_cfg=self.merge_cfg,
                fit_cfg=self.fit_cfg,
            )
        )

        streams, fits, _ = (
            ensure_lead_vehicle_stream(
                graph,
                streams,
                fits,
                current_frame_index=frame_id,
                cfg=self.lead_cfg,
            )
        )

        raw_boundaries = infer_lane_boundaries(
            fits,
            cfg=self.boundary_cfg,
        )

        boundaries, self.tracker_state, _ = (
            update_temporal_lane_tracks(
                self.tracker_state,
                raw_boundaries,
                lane_to_global,
                road_plane,
                frame_id,
                cfg=self.tracking_cfg,
            )
        )

        # -----------------------------------------------------
        # Evaluation / validation diagnostics
        # -----------------------------------------------------

        self.last_tracks = tracks
        self.last_road_plane = road_plane
        self.last_current_vehicles = list(current_vehicles)
        self.last_temporal_vehicles = list(temporal_vehicles)

        self.last_graph = graph
        self.last_streams = [
            list(stream)
            for stream in streams
        ]
        self.last_fits = [
            dict(fit)
            for fit in fits
        ]
        self.last_raw_boundaries = [
            dict(boundary)
            for boundary in raw_boundaries
        ]
        self.last_boundaries = [
            dict(boundary)
            for boundary in boundaries
        ]

        return (
            graph,
            streams,
            boundaries,
            len(current_vehicles),
        )