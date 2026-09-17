from __future__ import annotations

import numpy as np
from pyquaternion import Quaternion


# ---------------------------------------------------------------------------
# Canonical downstream lane coordinate convention:
#
#     +x = right
#     +y = down
#     +z = forward
#
# nuScenes / LIDAR_TOP convention:
#
#     +x = forward
#     +y = left
#     +z = up
#
# Therefore:
#
#     lane_x = -lidar_y
#     lane_y = -lidar_z
#     lane_z =  lidar_x
# ---------------------------------------------------------------------------

LANE_FROM_LIDAR = np.array(
    [
        [0.0, -1.0, 0.0],
        [0.0,  0.0, -1.0],
        [1.0,  0.0, 0.0],
    ],
    dtype=np.float64,
)

LIDAR_FROM_LANE = LANE_FROM_LIDAR.T


GT_CLASS_LABELS = {
    "car": 0,
    "truck": 1,
    "bus": 2,
    "trailer": 3,
    "construction": 4,
}


def _make_transform(
    translation,
    rotation,
) -> np.ndarray:
    """
    Construct a homogeneous transform from quaternion + translation.

    Quaternion order is nuScenes [w, x, y, z].
    """
    transform = np.eye(4, dtype=np.float64)

    transform[:3, :3] = Quaternion(
        rotation
    ).rotation_matrix

    transform[:3, 3] = np.asarray(
        translation,
        dtype=np.float64,
    )

    return transform


def get_lane_to_global(
    inputs_json: dict,
) -> np.ndarray:
    """
    Return the homogeneous transform from the canonical lane frame
    to the nuScenes global frame.

    The canonical lane frame is rigidly attached to LIDAR_TOP for the
    current keyframe but uses the camera-like x-right/y-down/z-forward
    axis convention expected by the downstream lane code.
    """
    lidar_to_ego = _make_transform(
        inputs_json["lidar2ego_translation"],
        inputs_json["lidar2ego_rotation"],
    )

    ego_to_global = _make_transform(
        inputs_json["ego2global_translation"],
        inputs_json["ego2global_rotation"],
    )

    lane_to_lidar = np.eye(4, dtype=np.float64)
    lane_to_lidar[:3, :3] = LIDAR_FROM_LANE

    return (
        ego_to_global
        @ lidar_to_ego
        @ lane_to_lidar
    )


def get_global_to_lane(
    inputs_json: dict,
) -> np.ndarray:
    """Inverse of get_lane_to_global()."""
    return np.linalg.inv(
        get_lane_to_global(inputs_json)
    )


def _global_point_to_lane(
    point_global,
    global_to_lane: np.ndarray,
) -> np.ndarray:
    point_global_h = np.array(
        [
            float(point_global[0]),
            float(point_global[1]),
            float(point_global[2]),
            1.0,
        ],
        dtype=np.float64,
    )

    return (
        global_to_lane
        @ point_global_h
    )[:3]


def _global_heading_to_lane(
    rotation_global,
    global_to_lane: np.ndarray,
) -> np.ndarray:
    """
    Convert the vehicle's forward direction into the lane frame.

    nuScenes Box convention uses local +x as the forward/length axis,
    so [1, 0, 0] is transformed by the annotation quaternion.
    """
    box_rotation_global = Quaternion(
        rotation_global
    ).rotation_matrix

    heading_global = (
        box_rotation_global
        @ np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float64,
        )
    )

    heading_lane = (
        global_to_lane[:3, :3]
        @ heading_global
    )

    norm_xz = np.hypot(
        heading_lane[0],
        heading_lane[2],
    )

    if norm_xz < 1e-8:
        raise ValueError(
            "GT vehicle heading has degenerate x-z projection"
        )

    return heading_lane / np.linalg.norm(
        heading_lane
    )


def gt_annotation_to_vehicle(
    annotation: dict,
    inputs_json: dict,
    original_idx: int = 0,
) -> dict:
    """
    Convert one exported nuScenes GT annotation into the dictionary
    format expected by the downstream lane inference code.

    This performs no configurable evidence filtering. It is only a
    deterministic coordinate/data representation adapter.
    """
    global_to_lane = get_global_to_lane(
        inputs_json
    )

    centre_lane = _global_point_to_lane(
        annotation["translation_global"],
        global_to_lane,
    )

    heading_lane = _global_heading_to_lane(
        annotation["rotation_global"],
        global_to_lane,
    )

    # Downstream geometry uses:
    #
    # heading_from_yaw(yaw) = [cos(yaw), sin(yaw)]
    #
    # corresponding to [lane_x, lane_z].
    yaw = float(
        np.arctan2(
            heading_lane[2],
            heading_lane[0],
        )
    )

    width, length, height = map(
        float,
        annotation["size"],
    )

    class_name = annotation["class_name"]

    if class_name not in GT_CLASS_LABELS:
        raise ValueError(
            f"Unsupported GT class: {class_name}"
        )

    return {
        "original_idx": int(original_idx),

        "annotation_token": annotation[
            "annotation_token"
        ],
        "instance_token": annotation[
            "instance_token"
        ],
        "track_id": int(annotation["track_id"]),

        "category_name": annotation[
            "category_name"
        ],
        "class_name": class_name,

        # Canonical lane coordinates.
        "x": float(centre_lane[0]),
        "y": float(centre_lane[1]),
        "z": float(centre_lane[2]),
        "yaw": yaw,

        "length": length,
        "width": width,
        "height": height,

        # Perfect GT detection confidence.
        "score": 1.0,
        "evidence_weight": 1.0,

        # Preserve numeric label compatibility with existing code.
        "label": GT_CLASS_LABELS[class_name],

        "below_standard_score": False,

        "source": "nuscenes_gt",
    }


def extract_vehicles_from_gt(
    gt_vehicles: list[dict],
    inputs_json: dict,
) -> list[dict]:
    """
    Convert all exported GT vehicles from one frame.

    No max-depth, yaw, visibility or confidence filtering is performed
    here. Those belong to the post-processing/evaluation configuration.
    """
    vehicles = []

    for index, annotation in enumerate(
        gt_vehicles
    ):
        vehicle = gt_annotation_to_vehicle(
            annotation,
            inputs_json,
            original_idx=index,
        )

        values = np.array(
            [
                vehicle["x"],
                vehicle["y"],
                vehicle["z"],
                vehicle["yaw"],
                vehicle["length"],
                vehicle["width"],
                vehicle["height"],
            ],
            dtype=np.float64,
        )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                "Non-finite canonical GT vehicle: "
                f"{annotation['annotation_token']}"
            )

        vehicles.append(vehicle)

    return vehicles