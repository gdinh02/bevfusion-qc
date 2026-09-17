from dataclasses import dataclass


@dataclass
class GTLaneEvidenceConfig:
    """
    Filtering applied to perfect GT detections before lane inference.

    These remain configurable because they are lane-evidence choices,
    not detector limitations.
    """

    enable_yaw_filter: bool = True

    # Canonical lane-frame forward is yaw = +/- pi/2.
    # 22.5 deg matches the existing BEVFusion pi/8 filter.
    max_forward_yaw_diff_deg: float = 22.5