from dataclasses import dataclass


@dataclass
class GTLaneEvidenceConfig:
    """
    Filtering applied to perfect GT detections before lane inference.

    These remain configurable because they are lane-evidence choices,
    not detector limitations.
    """

    enable_yaw_filter: bool = True

    # Accept vehicles travelling approximately along either +z or -z.
    # The two directions remain distinct downstream.
    max_forward_yaw_diff_deg: float = 22.5