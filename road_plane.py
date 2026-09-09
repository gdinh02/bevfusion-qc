import numpy as np

from configs import RoadPlaneConfig

def estimate_road_plane(pred_instances_3d, vehicles, cfg=None):
    if cfg is None:
        cfg = RoadPlaneConfig()
    if not vehicles:
        return None

    boxes = pred_instances_3d.bboxes_3d
    if not hasattr(boxes, "bottom_center"):
        return None

    all_bottom = boxes.bottom_center.detach().cpu().numpy()
    indices = [vehicle["original_idx"] for vehicle in vehicles]
    points = np.asarray(all_bottom[indices], dtype=np.float64)
    points = points[np.all(np.isfinite(points), axis=1) & (points[:, 2] > 0)]
    if len(points) == 0:
        return None

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    if len(points) == 1:
        return {
            "coefficients": np.array([0.0, 0.0, float(y[0])]),
            "rmse": 0.0,
            "inlier_count": 1,
            "total_count": 1,
            "model": "constant",
        }

    if len(points) == 2:
        design = np.column_stack([z, np.ones_like(z)])
        b, c = np.linalg.lstsq(design, y, rcond=None)[0]
        prediction = design @ np.array([b, c])
        return {
            "coefficients": np.array([0.0, b, c]),
            "rmse": float(np.sqrt(np.mean((y - prediction) ** 2))),
            "inlier_count": 2,
            "total_count": 2,
            "model": "z_line",
        }

    design_all = np.column_stack([x, z, np.ones_like(x)])
    rng = np.random.default_rng(cfg.random_seed)
    best_mask = None
    best_count = -1
    best_rmse = np.inf

    for _ in range(cfg.max_trials):
        idx = rng.choice(len(points), size=3, replace=False)
        if np.linalg.matrix_rank(design_all[idx]) < 3:
            continue
        coeff = np.linalg.lstsq(design_all[idx], y[idx], rcond=None)[0]
        prediction = design_all @ coeff
        mask = np.abs(y - prediction) <= cfg.residual_threshold
        count = int(mask.sum())
        if count < 3:
            continue
        rmse = float(np.sqrt(np.mean((y[mask] - prediction[mask]) ** 2)))
        if count > best_count or (count == best_count and rmse < best_rmse):
            best_mask, best_count, best_rmse = mask, count, rmse

    if best_mask is None:
        design = np.column_stack([z, np.ones_like(z)])
        b, c = np.linalg.lstsq(design, y, rcond=None)[0]
        prediction = design @ np.array([b, c])
        return {
            "coefficients": np.array([0.0, b, c]),
            "rmse": float(np.sqrt(np.mean((y - prediction) ** 2))),
            "inlier_count": len(points),
            "total_count": len(points),
            "model": "z_line_fallback",
        }

    coeff = np.linalg.lstsq(
        design_all[best_mask], y[best_mask], rcond=None
    )[0]
    prediction = design_all[best_mask] @ coeff
    return {
        "coefficients": coeff,
        "rmse": float(np.sqrt(np.mean((y[best_mask] - prediction) ** 2))),
        "inlier_count": int(best_mask.sum()),
        "total_count": len(points),
        "model": "plane",
    }


def road_plane_y(road_plane, x, z):
    a, b, c = np.asarray(road_plane["coefficients"], dtype=np.float64)
    return a * np.asarray(x) + b * np.asarray(z) + c


def transform_road_plane_to_camera(
    road_plane,
    source_cam_to_global,
    target_cam_to_global,
):
    """Transform y = a*x + b*z + c from one camera frame to another."""
    if road_plane is None:
        return None

    source_cam_to_global = np.asarray(source_cam_to_global, dtype=np.float64)
    target_cam_to_global = np.asarray(target_cam_to_global, dtype=np.float64)
    if source_cam_to_global.shape != (4, 4) or target_cam_to_global.shape != (4, 4):
        raise ValueError("camera-to-global transforms must be 4x4")

    a, b, c = np.asarray(road_plane["coefficients"], dtype=np.float64)

    # Plane in homogeneous form:
    #     a*x - y + b*z + c = 0
    plane_source = np.array([a, -1.0, b, c], dtype=np.float64)

    target_from_source = (
        np.linalg.inv(target_cam_to_global) @ source_cam_to_global
    )

    # If x_target = T * x_source, then plane_target = T^-T plane_source.
    plane_target = np.linalg.inv(target_from_source).T @ plane_source

    y_coefficient = float(plane_target[1])
    if abs(y_coefficient) < 1e-8:
        return None

    transformed_coefficients = np.array(
        [
            -plane_target[0] / y_coefficient,
            -plane_target[2] / y_coefficient,
            -plane_target[3] / y_coefficient,
        ],
        dtype=np.float64,
    )

    return {
        "coefficients": transformed_coefficients,
        "rmse": float(road_plane.get("rmse", 0.0)),
        "inlier_count": int(road_plane.get("inlier_count", 0)),
        "total_count": int(road_plane.get("total_count", 0)),
        "model": "transformed_" + str(road_plane.get("model", "plane")),
    }

