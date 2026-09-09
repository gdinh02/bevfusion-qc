import numpy as np

from configs import LaneProjectionConfig
from lane_fitting import evaluate_lane_polynomial
from road_plane import road_plane_y


def project_camera_points(points_3d, cam2img):
    points_3d = np.asarray(points_3d, dtype=np.float64)
    cam2img = np.asarray(cam2img, dtype=np.float64)
    if cam2img.shape == (3, 3):
        projected = points_3d @ cam2img.T
    elif cam2img.shape == (3, 4):
        projected = np.column_stack([points_3d, np.ones(len(points_3d))]) @ cam2img.T
    elif cam2img.shape == (4, 4):
        projected = (
            np.column_stack([points_3d, np.ones(len(points_3d))]) @ cam2img.T
        )[:, :3]
    else:
        raise ValueError(f"Unsupported cam2img shape: {cam2img.shape}")

    pixels = np.full((len(points_3d), 2), np.nan, dtype=np.float64)
    valid = np.abs(projected[:, 2]) > 1e-8
    pixels[valid] = projected[valid, :2] / projected[valid, 2, None]
    return pixels


def project_lane_boundaries_to_image(
    lane_boundaries, road_plane, cam2img, image_shape=None, cfg=None
):
    if cfg is None:
        cfg = LaneProjectionConfig()
    if road_plane is None:
        return []

    projected_boundaries = []
    for boundary_id, boundary in enumerate(lane_boundaries):
        z = np.linspace(boundary["z_min"], boundary["z_max"], cfg.sample_count)
        x = evaluate_lane_polynomial(boundary["coefficients"], z)
        y = road_plane_y(road_plane, x, z)
        points_3d = np.column_stack([x, y, z])
        pixels = project_camera_points(points_3d, cam2img)

        valid = (
            np.isfinite(pixels[:, 0])
            & np.isfinite(pixels[:, 1])
            & (z >= cfg.min_depth)
        )
        if cfg.clip_to_image and image_shape is not None:
            height, width = image_shape[:2]
            valid &= (
                (pixels[:, 0] >= 0)
                & (pixels[:, 0] < width)
                & (pixels[:, 1] >= 0)
                & (pixels[:, 1] < height)
            )
        if not np.any(valid):
            continue

        projected_boundaries.append(
            {
                "boundary_id": boundary_id,
                "left_stream_id": boundary["left_stream_id"],
                "right_stream_id": boundary["right_stream_id"],
                "pixels": pixels[valid],
                "points_3d": points_3d[valid],
                "lane_width": boundary["lane_width"],
                "source": boundary.get("source", "paired_streams"),
                "confidence": boundary.get("confidence", 1.0),
                "side": boundary.get("side", "between"),
                "source_stream_id": boundary.get("source_stream_id"),
                "boundary_track_id": boundary.get("boundary_track_id"),
                "temporal_status": boundary.get("temporal_status", "measurement"),
                "track_age": boundary.get("track_age"),
                "track_hits": boundary.get("track_hits"),
                "missed_frames": boundary.get("missed_frames", 0),
                "is_predicted": boundary.get("is_predicted", False),
                "smoothed": boundary.get("smoothed", False),
                "forced_lead": bool(boundary.get("forced_lead", False)),
            }
        )

    return projected_boundaries
