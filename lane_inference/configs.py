from dataclasses import dataclass
import math


# ==============================================================================
# BEVFusion detection / post-processing
# ==============================================================================

@dataclass
class BEVFusionConfig:
    """Tunable BEVFusion post-processing settings used before lane inference."""

    # Minimum BEVFusion confidence retained before downstream lane processing.
    score_threshold: float = 0.50

    # BEVFusion NMS settings.
    nms_threshold: float = 4.0
    nms_post_max_size: int = 83

    # Vehicle-orientation filter applied before lane inference.
    yaw_filter_direction: float = math.pi
    yaw_filter_span: float = math.pi / 8


# ==============================================================================
# Temporal vehicle evidence
# ==============================================================================

@dataclass
class TemporalConfig:
    """Controls temporal accumulation and association of vehicle detections."""

    history_frames: int = 10
    max_track_distance: float = 12.0
    max_track_yaw_diff_deg: float = 15.0
    max_track_frame_gap: int = 5
    temporal_decay: float = 1.0
    min_track_observations: int = 1

    # Keep detections from the newest frame even when their track has not yet
    # accumulated min_track_observations.
    keep_latest_frame_detections: bool = True


# ==============================================================================
# Lane-stream graph construction
# ==============================================================================

@dataclass
class LaneGraphConfig:
    """Controls vehicle filtering and lane-compatibility graph construction."""

    # Vehicle evidence filtering.
    score_thresh: float = 0.45

    # A centred lead vehicle can remain useful lane evidence even when its
    # confidence is below the normal graph threshold.
    lead_score_thresh: float = 0.15
    lead_candidate_max_abs_x: float = 3.0

    # Spatial limits for evidence used by the lane graph.
    max_depth: float = 30.0

    # Pairwise compatibility gates.
    max_cross_track: float = 1.0
    max_yaw_diff_deg: float = 15.0
    max_along_track: float = 20.0

    # Compatibility-score scaling.
    sigma_cross_track: float = 0.8
    sigma_yaw_deg: float = 8.0

    # Minimum connected-component size required to form a lane stream.
    min_vehicles_per_stream: int = 2


# ==============================================================================
# Lane fitting
# ==============================================================================

@dataclass
class LaneFitConfig:
    """Controls polynomial fitting of each inferred lane stream."""

    degree: int = 2
    residual_threshold: float = 0.75
    max_trials: int = 100
    random_seed: int = 0


# ==============================================================================
# Lane-stream merging
# ==============================================================================

@dataclass
class LaneMergeConfig:
    """Controls merging of fragmented lane-stream fits."""

    # Merge fragmented fits only when their longitudinal ranges are close.
    max_longitudinal_gap: float = 6.0

    # Maximum centreline disagreement over the overlap/gap comparison interval.
    max_lateral_disagreement: float = 0.6

    # Maximum tangent-angle disagreement between fitted centrelines.
    max_tangent_diff_deg: float = 6.0

    # Number of points used when comparing fitted stream fragments.
    sample_count: int = 15

    # Safety cap for iterative pairwise merging.
    max_iterations: int = 50


# ==============================================================================
# Lead-vehicle fallback
# ==============================================================================

@dataclass
class LeadVehicleConfig:
    """Controls the optional lead-vehicle lane-stream fallback."""

    enabled: bool = False
    max_abs_x: float = 3.0
    max_depth: float = 30.0
    max_forward_yaw_diff_deg: float = 45.0
    near_depth: float = 3.0
    forward_extension: float = 16.0
    max_abs_slope: float = 0.45


# ==============================================================================
# Lane-boundary inference
# ==============================================================================

@dataclass
class LaneBoundaryConfig:
    """Controls conversion of lane-centre fits into lane-boundary hypotheses."""

    # Paired-stream boundary inference.
    min_overlap: float = 3.0
    min_lane_width: float = 2.5
    max_lane_width: float = 5.0
    sample_count: int = 50

    # Single-stream boundary inference.
    enable_single_stream_boundaries: bool = True
    single_stream_only_when_no_paired: bool = False
    default_lane_width: float = 3.5
    single_stream_min_inliers: int = 1
    single_stream_max_rmse: float = 0.75
    single_stream_confidence: float = 0.35
    single_stream_min_tracks: int = 1
    single_stream_min_span: float = 0.0
    max_single_stream_fits: int = 100

    # Avoid emitting a provisional boundary on top of a stronger boundary.
    provisional_dedup_distance: float = 0.75

    # Diagnostic logging.
    verbose: bool = False


# ==============================================================================
# Road-plane estimation
# ==============================================================================

@dataclass
class RoadPlaneConfig:
    """Controls robust estimation of the road plane."""

    residual_threshold: float = 0.35
    max_trials: int = 100
    random_seed: int = 0


# ==============================================================================
# Temporal lane-boundary tracking
# ==============================================================================

@dataclass
class BoundaryTrackingConfig:
    """Controls temporal association, smoothing and persistence of boundaries."""

    # Association gates.
    min_overlap: float = 1.0
    max_lateral_distance: float = 1.0
    max_tangent_diff_deg: float = 5.0

    # Current-measurement weight used for temporal smoothing.
    # 1.0 = no smoothing; smaller values retain more previous geometry.
    smoothing_alpha: float = 0.30

    # Confirmation and output behaviour.
    min_confirmed_hits: int = 2
    emit_unconfirmed: bool = False
    emit_predicted: bool = False

    # Miss handling.
    max_missed_frames: int = 2
    missing_confidence_decay: float = 0.75

    # Numerical settings for transforming/re-fitting tracked boundaries.
    sample_count: int = 60
    min_depth: float = 1.0
    min_points_after_transform: int = 6


# ==============================================================================
# Projection / display geometry
# ==============================================================================

@dataclass
class LaneProjectionConfig:
    """Controls sampling and clipping when projecting lane geometry to images."""

    sample_count: int = 200
    min_depth: float = 1.0
    clip_to_image: bool = True
