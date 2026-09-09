import cv2
import numpy as np
import torch

from lane_inference.lane_fitting import evaluate_lane_polynomial

from bevfusion_integration.bev_helper import (
    CLASS_NAMES,
    OBJECT_CLASSES,
    generate_bev_map
)

'''
from visualisation import(
    draw_lane_graph_on_bev,
    draw_lane_boundaries_on_bev,
    estimate_road_height_from_boxes,
    project_lane_boundary_to_camera,
    draw_lane_boundary,
    project_3d_box_to_camera,
    draw_3d_box,
    project_scene_to_cameras,
)
'''

# ==============================================================================
# HELPER: DRAW GRAPH ON BEV MAP
# ==============================================================================
def draw_lane_graph_on_bev(canvas, graph, streams, pixels_per_meter):
    """Overlays the lane graph edges on the existing BEV map."""
    h, w, _ = canvas.shape
    center_x, center_y = w // 2, h // 2

    # Draw edges connecting vehicles
    for u, v, data in graph.edges(data=True):
        node_u = graph.nodes[u]
        node_v = graph.nodes[v]
        
        # Convert physical (x, z) back to image pixels
        img_u_x = int(center_x + (node_u['x'] * pixels_per_meter))
        img_u_y = int(center_y - (node_u['z'] * pixels_per_meter))
        
        img_v_x = int(center_x + (node_v['x'] * pixels_per_meter))
        img_v_y = int(center_y - (node_v['z'] * pixels_per_meter))
        
        # Draw line (Cyan color for graph connections)
        cv2.line(canvas, (img_u_x, img_u_y), (img_v_x, img_v_y), (255, 255, 0), 2, cv2.LINE_AA)

    return canvas


# ==============================================================================
# HELPER: DRAW BOUNDARIES ON BEV MAP
# ==============================================================================
def draw_lane_boundaries_on_bev(canvas, boundaries, pixels_per_meter):
    """Evaluates the mathematical polynomials and draws them on the BEV map."""
    h, w, _ = canvas.shape
    center_x, center_y = w // 2, h // 2

    for boundary in boundaries:
        # Get the start and end depth of the curve
        z_min = float(boundary["z_min"])
        z_max = float(boundary["z_max"])
        coeffs = boundary["coefficients"]

        # 1. Sample Z values (forward depth) along the boundary
        # If the curve is extremely short, skip it
        if z_max <= z_min:
            continue
        z_vals = np.linspace(z_min, z_max, num=50)
        
        # 2. Evaluate the polynomial to get X values (lateral position)
        x_vals = evaluate_lane_polynomial(coeffs, z_vals)

        # 3. Convert physical (x, z) meters to image pixels
        img_x = (center_x + (x_vals * pixels_per_meter)).astype(np.int32)
        img_y = (center_y - (z_vals * pixels_per_meter)).astype(np.int32)

        # Stack into OpenCV polyline format: shape (-1, 1, 2)
        pts = np.column_stack((img_x, img_y)).reshape((-1, 1, 2))

        # 4. Color coding based on boundary confidence/source
        is_provisional = boundary.get("source") == "single_stream"
        
        # Yellow for strong paired boundaries, Orange for provisional single-stream
        color = (0, 165, 255) if is_provisional else (0, 255, 255) 
        thickness = 2 if is_provisional else 4
        
        cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
        
    return canvas


# ==============================================================================
# HELPER: PROJECT BOUNDARIES & BOXES TO CAMERAS (BEVFUSION MATRICES)
# ==============================================================================
def estimate_road_height_from_boxes(bboxes, inputs_json):
    """Estimate road height as Z = slope * Y + intercept."""
    lidar_z_offset = inputs_json.get(
        "lidar2ego_translation",
        [0, 0, 1.84]
    )[2]

    road_slope = 0.0
    road_intercept = -lidar_z_offset

    if len(bboxes) > 0:
        forward_dist = bboxes[:, 1]
        bottom_zs = bboxes[:, 2] - (bboxes[:, 5] / 2)

        if len(bboxes) >= 2:
            try:
                road_slope, road_intercept = np.polyfit(
                    forward_dist,
                    bottom_zs,
                    1,
                )
            except np.linalg.LinAlgError:
                road_intercept = float(np.median(bottom_zs))
        else:
            road_intercept = float(np.median(bottom_zs))

    return road_slope, road_intercept


def project_lane_boundary_to_camera(
    boundary,
    s2k_inv,
    intrins,
    road_slope,
    road_intercept,
):
    """Project one BEV lane boundary into camera pixel coordinates."""
    z_min = float(boundary["z_min"])
    z_max = float(boundary["z_max"])

    if z_max <= z_min:
        return None

    z_vals = np.linspace(z_min, z_max, num=100)
    x_vals = evaluate_lane_polynomial(
        boundary["coefficients"],
        z_vals,
    )

    lane_heights = (road_slope * z_vals) + road_intercept

    lidar_points = np.column_stack(
        (x_vals, z_vals, lane_heights)
    )
    lidar_points_hom = np.column_stack(
        (lidar_points, np.ones_like(x_vals))
    )

    sensor_points = (lidar_points_hom @ s2k_inv.T)[:, :3]

    valid = sensor_points[:, 2] > 0.1
    if not np.any(valid):
        return None

    sensor_points = sensor_points[valid]

    projected = sensor_points @ intrins.T
    pixels = projected[:, :2] / projected[:, 2:3]

    return np.rint(pixels).astype(np.int32).reshape((-1, 1, 2))


def draw_lane_boundary(cv_img, pts, boundary):
    """Draw one projected lane boundary."""
    is_provisional = boundary.get("source") == "single_stream"

    color = (
        (0, 165, 255)
        if is_provisional
        else (0, 255, 255)
    )

    cv2.polylines(
        cv_img,
        [pts],
        isClosed=False,
        color=color,
        thickness=8,
        lineType=cv2.LINE_AA,
    )


def project_3d_box_to_camera(box, s2k_inv, intrins):
    """Project a BEVFusion 3D bounding box into camera pixels."""
    x, y, z, w, l, h, yaw = box[:7]

    dx = l / 2
    dy = w / 2
    dz = h / 2

    x_corners = np.array([
         dx,  dx, -dx, -dx,
         dx,  dx, -dx, -dx,
    ])

    y_corners = np.array([
         dy, -dy, -dy,  dy,
         dy, -dy, -dy,  dy,
    ])

    z_corners = np.array([
         dz,  dz,  dz,  dz,
        -dz, -dz, -dz, -dz,
    ])

    local_corners = np.vstack(
        [x_corners, y_corners, z_corners]
    )

    # Same BEVFusion yaw correction as before.
    theta = -yaw - (np.pi / 2)
    c = np.cos(theta)
    s = np.sin(theta)

    rotation = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1],
    ])

    lidar_corners = (
        rotation @ local_corners
    ).T + np.array([x, y, z])

    lidar_corners_hom = np.column_stack(
        (lidar_corners, np.ones(8))
    )

    sensor_corners = (
        lidar_corners_hom @ s2k_inv.T
    )[:, :3]

    # Preserve original behaviour: reject box unless every
    # corner is in front of the camera.
    if not np.all(sensor_corners[:, 2] > 0.1):
        return None

    projected = sensor_corners @ intrins.T

    return (
        projected[:, :2] / projected[:, 2:3]
    ).astype(np.int32)


def draw_3d_box(cv_img, pixels, label):
    """Draw one projected 3D vehicle bounding box."""
    class_name = (
        CLASS_NAMES[label]
        if label < len(CLASS_NAMES)
        else "car"
    )

    box_color = OBJECT_CLASSES.get(
        class_name,
        (255, 255, 255),
    )

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    for start, end in edges:
        cv2.line(
            cv_img,
            tuple(pixels[start]),
            tuple(pixels[end]),
            box_color,
            2,
            cv2.LINE_AA,
        )

    # Red X on the vehicle's front face.
    cv2.line(
        cv_img,
        tuple(pixels[0]),
        tuple(pixels[5]),
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.line(
        cv_img,
        tuple(pixels[1]),
        tuple(pixels[4]),
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )


def project_scene_to_cameras(
    app,
    images,
    cam_paths,
    boundaries,
    bboxes,
    labels,
    inputs_json,
):
    """Project and draw lane boundaries and vehicle boxes on all cameras."""

    intrins_list, sensor2keyegos_list = (
        app.prepare_camera_inputs(
            cam_paths,
            inputs_json,
        )
    )

    if torch.is_tensor(bboxes):
        bboxes = bboxes.cpu().numpy()
        labels = labels.cpu().numpy()

    road_slope, road_intercept = (
        estimate_road_height_from_boxes(
            bboxes,
            inputs_json,
        )
    )

    processed_cams = {}

    for k, (name, pil_img) in enumerate(
        zip(cam_paths.keys(), images)
    ):
        cv_img = cv2.cvtColor(
            np.array(pil_img),
            cv2.COLOR_RGB2BGR,
        )

        s2k_inv = np.linalg.inv(
            sensor2keyegos_list[k]
        )
        intrins = intrins_list[k]

        # Lane boundaries
        for boundary in boundaries:
            pts = project_lane_boundary_to_camera(
                boundary,
                s2k_inv,
                intrins,
                road_slope,
                road_intercept,
            )

            if pts is None:
                continue

            draw_lane_boundary(
                cv_img,
                pts,
                boundary,
            )

        # Vehicle boxes
        for i in range(len(bboxes)):
            pixels = project_3d_box_to_camera(
                bboxes[i],
                s2k_inv,
                intrins,
            )

            if pixels is None:
                continue

            draw_3d_box(
                cv_img,
                pixels,
                labels[i],
            )

        processed_cams[name] = cv_img

    return processed_cams