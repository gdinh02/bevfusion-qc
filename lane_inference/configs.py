from dataclasses import dataclass

'''
from configs import(
    LaneGraphConfig,
    TemporalConfig,
    LeadVehicleConfig,
    LaneFitConfig,
    LaneMergeConfig,
    LaneBoundaryConfig,
    BoundaryTrackingConfig,
    RoadPlaneConfig,
    LaneProjectionConfig,
)
'''

@dataclass
class LaneGraphConfig:
    score_thresh: float = 0.25
    # A centred lead vehicle is valuable lane evidence even when FCOS3D gives
    # it a slightly lower score than the general graph threshold.
    lead_score_thresh: float = 0.15
    lead_candidate_max_abs_x: float = 3.0
    max_depth: float = 60.0
    max_cross_track: float = 1.0
    max_yaw_diff_deg: float = 10.0
    max_along_track: float = 20.0
    sigma_cross_track: float = 0.8
    sigma_yaw_deg: float = 8.0

@dataclass
class TemporalConfig:
    history_frames: int = 5
    max_track_distance: float = 12.0
    max_track_yaw_diff_deg: float = 30.0
    max_track_frame_gap: int = 1
    temporal_decay: float = 0.90
    min_track_observations: int = 1
    # Never discard detections from the reference/latest frame merely because
    # their temporal track has not accumulated enough observations yet.
    keep_latest_frame_detections: bool = True

@dataclass
class LeadVehicleConfig:
    enabled: bool = True
    max_abs_x: float = 3.0
    max_depth: float = 45.0
    max_forward_yaw_diff_deg: float = 45.0
    near_depth: float = 3.0
    forward_extension: float = 8.0
    max_abs_slope: float = 0.75


@dataclass
class LaneFitConfig:
    degree: int = 1
    residual_threshold: float = 0.75
    max_trials: int = 200
    random_seed: int = 0


@dataclass
class LaneMergeConfig:
    # Merge fragmented fits only when their longitudinal ranges are close.
    max_longitudinal_gap: float = 6.0

    # Maximum centreline disagreement over the overlap/gap comparison interval.
    max_lateral_disagreement: float = 0.6

    # Maximum tangent-angle disagreement between the two fitted centrelines.
    max_tangent_diff_deg: float = 6.0

    # Number of points used when comparing two fitted stream fragments.
    sample_count: int = 15

    # Safety cap for iterative pairwise merging.
    max_iterations: int = 50


@dataclass
class LaneBoundaryConfig:
    min_overlap: float = 3.0
    min_lane_width: float = 2.5
    max_lane_width: float = 5.0
    sample_count: int = 50

    # Single-stream boundary inference
    enable_single_stream_boundaries: bool = True
    single_stream_only_when_no_paired: bool =True
    default_lane_width: float = 3.5
    single_stream_min_inliers: int = 3
    single_stream_max_rmse: float = 0.75
    single_stream_confidence: float = 0.45
    single_stream_min_tracks: int = 1
    single_stream_min_span: float = 3.0
    max_single_stream_fits: int = 2

    # Avoid drawing a provisional boundary on top of a stronger paired one
    provisional_dedup_distance: float = 0.75

    verbose: bool = False


@dataclass
class BoundaryTrackingConfig:
    # Minimum shared forward range required to associate two boundaries.
    min_overlap: float = 3.0

    # Maximum median lateral disagreement after ego-motion compensation.
    max_lateral_distance: float = 1.0

    # Maximum tangent-angle disagreement over the common range.
    max_tangent_diff_deg: float = 10.0

    # Current measurement weight used for temporal smoothing.
    # 1.0 = no smoothing; smaller values retain more previous geometry.
    smoothing_alpha: float = 0.30

    # A track is confirmed only after this many associated measurements.
    # Unconfirmed tracks remain in tracker state so that they can mature.
    min_confirmed_hits: int = 2

    # Control which tracker states are emitted for projection/display.
    emit_unconfirmed: bool = False
    emit_predicted: bool = False

    # Keep an unmatched boundary alive briefly to bridge missed detections.
    max_missed_frames: int = 2
    missing_confidence_decay: float = 0.75

    sample_count: int = 60
    min_depth: float = 1.0
    min_points_after_transform: int = 6


@dataclass
class RoadPlaneConfig:
    residual_threshold: float = 0.35
    max_trials: int = 200
    random_seed: int = 0


@dataclass
class LaneProjectionConfig:
    sample_count: int = 200
    min_depth: float = 1.0
    clip_to_image: bool = True

