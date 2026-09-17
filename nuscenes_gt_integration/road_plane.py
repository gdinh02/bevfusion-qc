from __future__ import annotations

import numpy as np


def estimate_road_plane_from_gt_vehicles(
    vehicles: list[dict],
    inputs_json: dict,
) -> dict:
    """
    Estimate road height in the canonical lane frame.

    Canonical coordinates:
        +x = right
        +y = down
        +z = forward

    Vehicle centre y is therefore above its ground contact point, so:

        bottom_y = centre_y + height / 2

    We fit:

        y = slope * z + intercept

    which directly matches the downstream road-plane representation:

        y = a*x + b*z + c
    """

    lidar_height = float(
        inputs_json.get(
            "lidar2ego_translation",
            [0.0, 0.0, 1.84],
        )[2]
    )

    if not vehicles:
        return {
            "coefficients": np.array(
                [0.0, 0.0, lidar_height],
                dtype=np.float64,
            ),
            "rmse": 0.0,
            "inlier_count": 0,
            "total_count": 0,
            "model": "gt_lidar_height_fallback",
        }

    points = []

    for vehicle in vehicles:
        z = float(vehicle["z"])
        bottom_y = (
            float(vehicle["y"])
            + 0.5 * float(vehicle["height"])
        )

        if np.isfinite(z) and np.isfinite(bottom_y):
            points.append((z, bottom_y))

    if not points:
        return {
            "coefficients": np.array(
                [0.0, 0.0, lidar_height],
                dtype=np.float64,
            ),
            "rmse": 0.0,
            "inlier_count": 0,
            "total_count": 0,
            "model": "gt_lidar_height_fallback",
        }

    points = np.asarray(points, dtype=np.float64)
    z = points[:, 0]
    bottom_y = points[:, 1]

    if len(points) == 1:
        slope = 0.0
        intercept = float(bottom_y[0])
        prediction = np.full_like(bottom_y, intercept)
        model = "gt_constant"
    else:
        try:
            slope, intercept = np.polyfit(
                z,
                bottom_y,
                1,
            )
            prediction = slope * z + intercept
            model = "gt_vehicle_bottom_line"
        except np.linalg.LinAlgError:
            slope = 0.0
            intercept = float(np.median(bottom_y))
            prediction = np.full_like(
                bottom_y,
                intercept,
            )
            model = "gt_median_fallback"

    rmse = float(
        np.sqrt(
            np.mean(
                (bottom_y - prediction) ** 2
            )
        )
    )

    return {
        "coefficients": np.array(
            [
                0.0,
                float(slope),
                float(intercept),
            ],
            dtype=np.float64,
        ),
        "rmse": rmse,
        "inlier_count": len(points),
        "total_count": len(points),
        "model": model,
    }