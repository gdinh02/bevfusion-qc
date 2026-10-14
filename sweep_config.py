"""Explicit, validated lane-parameter selection for GT experiments."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from lane_inference.configs import (
    BoundaryTrackingConfig, LaneBoundaryConfig, LaneFitConfig,
    LaneGraphConfig, LaneMergeConfig, TemporalConfig,
)
from nuscenes_gt_integration.configs import GTLaneEvidenceConfig

# Manually selected subset. Add new paths only after verifying their effects.
CANDIDATES = {
    "graph_cfg.max_cross_track": [0.6, 0.8, 1.0, 1.2, 1.5],
    "graph_cfg.max_yaw_diff_deg": [8.0, 12.0, 15.0, 20.0],
    "fit_cfg.residual_threshold": [0.25, 0.4, 0.5, 0.75, 1.0],
    "merge_cfg.max_lateral_disagreement": [0.3, 0.45, 0.6, 0.8],
    "boundary_cfg.default_lane_width": [3.0, 3.25, 3.5, 3.75, 4.0],
    "tracking_cfg.smoothing_alpha": [0.15, 0.3, 0.5, 0.75, 1.0],
}

CONFIG_CLASSES = {
    "evidence_cfg": GTLaneEvidenceConfig,
    "temporal_cfg": TemporalConfig,
    "graph_cfg": LaneGraphConfig,
    "fit_cfg": LaneFitConfig,
    "merge_cfg": LaneMergeConfig,
    "boundary_cfg": LaneBoundaryConfig,
    "tracking_cfg": BoundaryTrackingConfig,
}


def baseline() -> dict[str, float]:
    """Defaults for only the parameters explicitly included in the sweep."""
    return {
        name: getattr(CONFIG_CLASSES[name.split('.')[0]](), name.split('.')[1])
        for name in CANDIDATES
    }


def validate(overrides: dict) -> dict:
    """Reject unknown settings, invalid values and invalid dataclass fields."""
    if not isinstance(overrides, dict):
        raise TypeError("Parameter overrides must be a JSON object")
    unknown = set(overrides) - set(CANDIDATES)
    if unknown:
        raise ValueError(f"Parameters not on the sweep allow-list: {sorted(unknown)}")
    validated = {}
    for key, value in overrides.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} requires a numeric value")
        import math
        if not math.isfinite(float(value)):
            raise ValueError(f"{key} must be finite")
        if value <= 0:
            raise ValueError(f"{key} must be positive")
        if key == "tracking_cfg.smoothing_alpha" and value > 1:
            raise ValueError("smoothing_alpha must be <= 1")
        validated[key] = float(value)
    return validated


def make_pipeline_kwargs(overrides: dict) -> dict:
    """Construct new independent config instances for each pipeline run."""
    overrides = validate(overrides)
    configurations = {name: cls() for name, cls in CONFIG_CLASSES.items()}
    for dotted_name, value in overrides.items():
        config_name, field_name = dotted_name.split(".", 1)
        setattr(configurations[config_name], field_name, value)
    return configurations


def load_overrides(path: str | Path | None) -> dict:
    if path is None:
        return {}
    with open(path, encoding="utf-8") as stream:
        return validate(json.load(stream))


def single_parameter_trials() -> list[dict]:
    """Baseline + OAT variants; duplicate baseline candidates skipped."""
    trials = [{}]
    defaults = baseline()
    for name, values in CANDIDATES.items():
        for value in values:
            if value != defaults[name]:
                trials.append({name: value})
    return trials
