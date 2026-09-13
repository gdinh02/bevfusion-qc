# Lane Inference Configuration Reference

This file documents the tunable parameters in `configs.py`.

The configuration blocks are ordered to follow the inference pipeline:

1. BEVFusion detection and post-processing
2. Temporal vehicle evidence
3. Lane-stream graph construction
4. Lane fitting
5. Lane-stream merging
6. Lead-vehicle fallback
7. Lane-boundary inference
8. Road-plane estimation
9. Temporal boundary tracking
10. Projection

The values in `configs.py` are the defaults for one run. They are intended to become the single source of truth for parameters that can change inference behaviour.

---

## `BEVFusionConfig`

Controls BEVFusion post-processing before detections enter lane inference.

### `score_threshold`
Minimum BEVFusion confidence required for a detection to survive the model's initial confidence filter.

- Lower values increase recall but admit more low-confidence detections.
- Higher values improve detection precision but may remove useful lane evidence.

This threshold is upstream of `LaneGraphConfig.score_thresh`, so detections removed here cannot be recovered later.

### `nms_threshold`
Threshold used by BEVFusion's non-maximum suppression/post-processing.

It controls how aggressively overlapping 3D detections are suppressed.

### `nms_post_max_size`
Maximum number of detections retained after NMS.

A larger value permits more detections to reach downstream processing.

### `yaw_filter_direction`
Reference yaw direction used by the vehicle orientation filter, in radians.

The default is `pi`.

### `yaw_filter_span`
Maximum angular deviation from the accepted yaw direction, in radians.

The default is `pi / 8`, or 22.5 degrees.

- Smaller spans keep only strongly aligned vehicles.
- Larger spans retain more vehicles but may admit vehicles that do not provide useful lane-direction evidence.

---

## `TemporalConfig`

Controls how vehicle detections from previous frames are transformed into the current frame and associated into tracks.

### `history_frames`
Number of frames retained in the temporal history.

- More frames provide more geometric evidence.
- Too many frames can accumulate stale or incorrectly associated evidence.

### `max_track_distance`
Maximum spatial distance, in metres, allowed when associating a detection with an existing vehicle track.

### `max_track_yaw_diff_deg`
Maximum yaw difference, in degrees, allowed for temporal vehicle association.

### `max_track_frame_gap`
Maximum frame-order gap over which a previous track can still be matched.

### `temporal_decay`
Weight decay applied to older detections.

The evidence weight is approximately:

`temporal_decay ** frame_age`

- `1.0` gives all retained frames equal weight.
- Values below `1.0` progressively down-weight older evidence.

### `min_track_observations`
Minimum number of observations required for a temporal vehicle track to be considered confirmed.

### `keep_latest_frame_detections`
When `True`, detections from the newest frame remain available even if they have not yet reached `min_track_observations`.

This prevents the confirmation requirement from eliminating all newly appearing vehicles.

---

## `LaneGraphConfig`

Controls which vehicle detections become graph nodes and when two vehicle observations are considered geometrically compatible.

### `score_thresh`
Normal confidence threshold for vehicle evidence entering the lane graph.

### `lead_score_thresh`
Lower confidence threshold permitted for potential lead-vehicle evidence.

This allows a centred vehicle to remain useful even when it does not meet `score_thresh`.

### `lead_candidate_max_abs_x`
Maximum absolute lateral displacement, in metres, for a low-confidence detection to qualify for the lead-vehicle exception.

### `max_depth`
Maximum absolute forward depth, in metres, for vehicle evidence used by the lane graph.

### `max_cross_track`
Maximum cross-track separation, in metres, allowed when adding an edge between two graph nodes.

This is a major control on whether observations are grouped into the same traffic stream.

### `max_yaw_diff_deg`
Maximum yaw difference, in degrees, allowed when connecting two vehicle observations.

### `max_along_track`
Maximum along-track separation, in metres, allowed when connecting two observations.

### `sigma_cross_track`
Scale used when converting cross-track disagreement into the graph edge compatibility score.

Smaller values penalise lateral disagreement more strongly.

### `sigma_yaw_deg`
Scale, in degrees, used when converting yaw disagreement into the edge compatibility score.

Smaller values penalise angular disagreement more strongly.

### `min_vehicles_per_stream`
Minimum number of graph nodes required for a connected component to be accepted as a lane stream.

- Lower values improve recall for sparse scenes.
- Higher values suppress weak or isolated stream hypotheses.

---

## `LaneFitConfig`

Controls polynomial fitting of vehicle streams into lane centrelines.

### `degree`
Polynomial degree used for the lane model.

- `1` gives a straight line.
- `2` permits curvature.
- Higher degrees are generally more flexible but can become unstable with sparse evidence.

### `residual_threshold`
Maximum lateral residual, in metres, for an observation to count as a RANSAC inlier.

### `max_trials`
Maximum number of RANSAC sampling attempts.

Increasing this improves the chance of finding a good fit when many outliers are present, at the cost of computation.

### `random_seed`
Seed used by the fitting RANSAC process.

Keeping this fixed makes experiments reproducible.

---

## `LaneMergeConfig`

Controls whether separately fitted lane-stream fragments should be consolidated into one lane model.

### `max_longitudinal_gap`
Maximum forward gap, in metres, between two fitted fragments that may still be merged.

### `max_lateral_disagreement`
Maximum lateral disagreement, in metres, permitted between candidate fits.

### `max_tangent_diff_deg`
Maximum tangent-angle difference, in degrees, permitted between candidate fits.

### `sample_count`
Number of positions sampled when comparing fitted lane fragments.

### `max_iterations`
Maximum number of iterative pairwise merges.

This prevents pathological cases from causing excessive repeated merging.

---

## `LeadVehicleConfig`

Controls the special fallback that attempts to form useful lane evidence around the current lead vehicle when ordinary multi-vehicle fitting is insufficient.

### `enabled`
Enables or disables the lead-vehicle fallback.

### `max_abs_x`
Maximum absolute lateral position, in metres, for a vehicle to be considered a lead-vehicle candidate.

### `max_depth`
Maximum forward depth, in metres, for the lead-vehicle candidate.

### `max_forward_yaw_diff_deg`
Maximum allowed yaw difference from the expected forward direction.

### `near_depth`
Near-distance limit used when creating the fallback lane geometry.

### `forward_extension`
Distance, in metres, by which a lead-vehicle anchor is extended forward.

### `max_abs_slope`
Maximum absolute lane slope permitted for the fallback anchor.

This limits implausibly steep lateral changes with forward distance.

---

## `LaneBoundaryConfig`

Controls conversion of fitted lane centrelines into lane-boundary hypotheses.

### Paired-stream inference

#### `min_overlap`
Minimum longitudinal overlap, in metres, required between two lane fits before they can define a boundary between them.

#### `min_lane_width`
Minimum valid lane width, in metres.

#### `max_lane_width`
Maximum valid lane width, in metres.

#### `sample_count`
Number of longitudinal samples used when comparing lane fits and estimating boundary geometry.

### Single-stream inference

#### `enable_single_stream_boundaries`
Enables boundary generation from a single fitted traffic stream by offsetting the centreline using an estimated/default lane width.

#### `single_stream_only_when_no_paired`
When `True`, ordinary single-stream boundaries are used only when no paired-stream boundaries were found.

When `False`, single-stream boundaries may coexist with paired-stream boundaries.

#### `default_lane_width`
Lane width, in metres, used when the system cannot estimate width from paired streams.

#### `single_stream_min_inliers`
Minimum number of fit inliers required before an ordinary single-stream fit may generate boundaries.

#### `single_stream_max_rmse`
Maximum fit RMSE, in metres, allowed for ordinary single-stream boundary generation.

#### `single_stream_confidence`
Base confidence assigned to single-stream boundary hypotheses before other evidence factors are applied.

#### `single_stream_min_tracks`
Minimum number of distinct temporal vehicle tracks required for an ordinary single-stream fit.

#### `single_stream_min_span`
Minimum longitudinal span, in metres, required for an ordinary single-stream fit.

#### `max_single_stream_fits`
Maximum number of ordinary single-stream fits allowed to produce provisional boundaries.

### Deduplication

#### `provisional_dedup_distance`
If two boundary hypotheses are closer than this distance, in metres, over their shared range, the weaker duplicate may be suppressed.

### Diagnostics

#### `verbose`
Enables detailed diagnostic printing from lane-boundary inference.

---

## `RoadPlaneConfig`

Controls robust road-plane estimation.

### `residual_threshold`
Maximum residual, in metres, for a point to count as an inlier during RANSAC road-plane fitting.

### `max_trials`
Maximum number of RANSAC trials.

### `random_seed`
Seed used by the road-plane RANSAC process for reproducibility.

---

## `BoundaryTrackingConfig`

Controls temporal association and persistence of inferred lane boundaries.

### Association

#### `min_overlap`
Minimum shared longitudinal range, in metres, required to associate a previous boundary with a current one.

#### `max_lateral_distance`
Maximum median lateral disagreement, in metres, allowed between a predicted boundary and a current measurement.

#### `max_tangent_diff_deg`
Maximum tangent-angle difference, in degrees, allowed during boundary association.

### Smoothing

#### `smoothing_alpha`
Weight assigned to the current boundary measurement during temporal smoothing.

- `1.0` means no smoothing from the previous boundary.
- Lower values retain more previous geometry.

### Confirmation and output

#### `min_confirmed_hits`
Number of associated measurements required before a boundary track is considered confirmed.

#### `emit_unconfirmed`
When `True`, newly created but not-yet-confirmed boundary tracks may be emitted.

#### `emit_predicted`
When `True`, unmatched but still-live tracked boundaries may be emitted as predictions.

### Miss handling

#### `max_missed_frames`
Maximum number of consecutive unmatched frames for which a boundary track remains alive.

#### `missing_confidence_decay`
Multiplicative confidence decay applied each time an unmatched boundary is carried forward.

### Numerical settings

#### `sample_count`
Number of points used when transforming and comparing tracked boundary geometry.

#### `min_depth`
Minimum forward depth, in metres, retained after transforming a boundary into a new camera frame.

#### `min_points_after_transform`
Minimum number of transformed boundary points required before the boundary may be re-fitted.

---

## `LaneProjectionConfig`

Controls sampling and clipping when inferred lane geometry is projected into camera images.

### `sample_count`
Number of points sampled along each lane boundary for projection.

### `min_depth`
Minimum forward depth, in metres, eligible for projection.

### `clip_to_image`
When `True`, projected geometry is clipped to the image bounds.

---

## Parameter categories for optimisation

For F1 optimisation, the highest-impact parameters are generally those that change whether evidence or boundaries exist:

- detection confidence and yaw filtering
- temporal association
- graph compatibility
- minimum stream size
- fitting residual threshold
- merge gates
- paired/single-stream boundary gates
- boundary confirmation and emission behaviour

Numerical parameters such as `sample_count`, `max_trials`, and `random_seed` can affect stability and runtime, but they are usually lower-priority optimisation dimensions unless there is evidence that the current numerical resolution is insufficient.
