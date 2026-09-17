from __future__ import annotations

import cv2
import numpy as np

from lane_inference.lane_fitting import (
    evaluate_lane_polynomial,
)


def _world_to_pixel(
    x,
    z,
    x_min,
    z_max,
    pixels_per_meter,
):
    px = int(
        round(
            (float(x) - x_min)
            * pixels_per_meter
        )
    )

    py = int(
        round(
            (z_max - float(z))
            * pixels_per_meter
        )
    )

    return px, py


def _vehicle_corners(vehicle):
    """
    Return vehicle rectangle corners in canonical x-z coordinates.

    yaw convention:
        heading = [cos(yaw), sin(yaw)]
        coordinates = [x, z]
    """
    x = float(vehicle["x"])
    z = float(vehicle["z"])
    yaw = float(vehicle["yaw"])

    length = float(vehicle["length"])
    width = float(vehicle["width"])

    heading = np.array(
        [
            np.cos(yaw),
            np.sin(yaw),
        ],
        dtype=np.float64,
    )

    normal = np.array(
        [
            -heading[1],
            heading[0],
        ],
        dtype=np.float64,
    )

    centre = np.array(
        [x, z],
        dtype=np.float64,
    )

    half_length = 0.5 * length
    half_width = 0.5 * width

    return np.stack(
        [
            centre
            + half_length * heading
            + half_width * normal,

            centre
            + half_length * heading
            - half_width * normal,

            centre
            - half_length * heading
            - half_width * normal,

            centre
            - half_length * heading
            + half_width * normal,
        ]
    )


def render_gt_pipeline_bev(
    pipeline,
    *,
    x_range=(-15.0, 15.0),
    z_range=(-10.0, 40.0),
    pixels_per_meter=20,
    show_track_ids=True,
):
    """
    Render the latest GT lane-pipeline state in canonical BEV coordinates.

    Displays:
        - all temporally accumulated vehicle observations
        - trajectories grouped by GT track_id
        - current-frame GT vehicle boxes
        - compatibility graph edges
        - accepted traffic streams
        - fitted stream centrelines
        - final temporally tracked lane boundaries
        - road-plane diagnostics

    Canonical axes:
        +x = right
        +z = forward
    """
    x_min, x_max = map(float, x_range)
    z_min, z_max = map(float, z_range)

    ppm = float(pixels_per_meter)

    width = int(
        round((x_max - x_min) * ppm)
    )
    height = int(
        round((z_max - z_min) * ppm)
    )

    canvas = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )

    # ---------------------------------------------------------
    # Grid
    # ---------------------------------------------------------

    grid_step = 5.0

    x_grid = np.arange(
        np.ceil(x_min / grid_step) * grid_step,
        x_max + 1e-6,
        grid_step,
    )

    z_grid = np.arange(
        np.ceil(z_min / grid_step) * grid_step,
        z_max + 1e-6,
        grid_step,
    )

    for x in x_grid:
        p1 = _world_to_pixel(
            x,
            z_min,
            x_min,
            z_max,
            ppm,
        )
        p2 = _world_to_pixel(
            x,
            z_max,
            x_min,
            z_max,
            ppm,
        )

        cv2.line(
            canvas,
            p1,
            p2,
            (35, 35, 35),
            1,
        )

    for z in z_grid:
        p1 = _world_to_pixel(
            x_min,
            z,
            x_min,
            z_max,
            ppm,
        )
        p2 = _world_to_pixel(
            x_max,
            z,
            x_min,
            z_max,
            ppm,
        )

        cv2.line(
            canvas,
            p1,
            p2,
            (35, 35, 35),
            1,
        )

    # ---------------------------------------------------------
    # Ego vehicle / origin
    # ---------------------------------------------------------

    ego = _world_to_pixel(
        0.0,
        0.0,
        x_min,
        z_max,
        ppm,
    )

    cv2.drawMarker(
        canvas,
        ego,
        (255, 255, 255),
        markerType=cv2.MARKER_TRIANGLE_UP,
        markerSize=18,
        thickness=2,
    )

    # ---------------------------------------------------------
    # Temporal GT trajectories
    # ---------------------------------------------------------

    temporal = pipeline.last_temporal_vehicles

    by_track = {}

    for vehicle in temporal:
        track_id = vehicle.get("track_id")

        if track_id is None:
            continue

        by_track.setdefault(
            int(track_id),
            [],
        ).append(vehicle)

    for track_id, observations in by_track.items():
        observations = sorted(
            observations,
            key=lambda vehicle: vehicle.get(
                "frame_index",
                -1,
            ),
        )

        points = []

        for vehicle in observations:
            point = _world_to_pixel(
                vehicle["x"],
                vehicle["z"],
                x_min,
                z_max,
                ppm,
            )

            points.append(point)

            # Historical observation.
            cv2.circle(
                canvas,
                point,
                3,
                (180, 120, 180),
                -1,
                cv2.LINE_AA,
            )

        if len(points) >= 2:
            pts = np.asarray(
                points,
                dtype=np.int32,
            ).reshape((-1, 1, 2))

            cv2.polylines(
                canvas,
                [pts],
                False,
                (120, 80, 120),
                1,
                cv2.LINE_AA,
            )

    # ---------------------------------------------------------
    # Compatibility graph
    # ---------------------------------------------------------

    graph = pipeline.last_graph

    if graph is not None:
        node_to_stream = {}

        for stream_id, stream in enumerate(
            pipeline.last_streams
        ):
            for node in stream:
                node_to_stream[node] = stream_id

        for u, v in graph.edges():
            a = graph.nodes[u]
            b = graph.nodes[v]

            p1 = _world_to_pixel(
                a["x"],
                a["z"],
                x_min,
                z_max,
                ppm,
            )

            p2 = _world_to_pixel(
                b["x"],
                b["z"],
                x_min,
                z_max,
                ppm,
            )

            same_accepted_stream = (
                u in node_to_stream
                and v in node_to_stream
                and node_to_stream[u]
                == node_to_stream[v]
            )

            if same_accepted_stream:
                colour = (255, 255, 0)
                thickness = 2
            else:
                colour = (65, 65, 65)
                thickness = 1

            cv2.line(
                canvas,
                p1,
                p2,
                colour,
                thickness,
                cv2.LINE_AA,
            )

    # ---------------------------------------------------------
    # Fitted traffic-stream centrelines
    # ---------------------------------------------------------

    for fit in pipeline.last_fits:
        z_start = max(
            float(fit["z_min"]),
            z_min,
        )
        z_end = min(
            float(fit["z_max"]),
            z_max,
        )

        if z_end <= z_start:
            continue

        z = np.linspace(
            z_start,
            z_end,
            80,
        )

        x = evaluate_lane_polynomial(
            fit["coefficients"],
            z,
        )

        points = [
            _world_to_pixel(
                xi,
                zi,
                x_min,
                z_max,
                ppm,
            )
            for xi, zi in zip(x, z)
        ]

        pts = np.asarray(
            points,
            dtype=np.int32,
        ).reshape((-1, 1, 2))

        cv2.polylines(
            canvas,
            [pts],
            False,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # ---------------------------------------------------------
    # Final lane boundaries
    # ---------------------------------------------------------

    for boundary in pipeline.last_boundaries:
        z_start = max(
            float(boundary["z_min"]),
            z_min,
        )
        z_end = min(
            float(boundary["z_max"]),
            z_max,
        )

        if z_end <= z_start:
            continue

        z = np.linspace(
            z_start,
            z_end,
            100,
        )

        x = evaluate_lane_polynomial(
            boundary["coefficients"],
            z,
        )

        points = [
            _world_to_pixel(
                xi,
                zi,
                x_min,
                z_max,
                ppm,
            )
            for xi, zi in zip(x, z)
        ]

        pts = np.asarray(
            points,
            dtype=np.int32,
        ).reshape((-1, 1, 2))

        is_single_stream = (
            boundary.get("source")
            == "single_stream"
        )

        colour = (
            (0, 165, 255)
            if is_single_stream
            else (0, 255, 255)
        )

        cv2.polylines(
            canvas,
            [pts],
            False,
            colour,
            3,
            cv2.LINE_AA,
        )

    # ---------------------------------------------------------
    # Current-frame vehicle boxes
    # ---------------------------------------------------------

    for vehicle in pipeline.last_current_vehicles:
        corners = _vehicle_corners(
            vehicle
        )

        pixels = np.asarray(
            [
                _world_to_pixel(
                    x,
                    z,
                    x_min,
                    z_max,
                    ppm,
                )
                for x, z in corners
            ],
            dtype=np.int32,
        ).reshape((-1, 1, 2))

        cv2.polylines(
            canvas,
            [pixels],
            True,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        centre = _world_to_pixel(
            vehicle["x"],
            vehicle["z"],
            x_min,
            z_max,
            ppm,
        )

        # Heading marker.
        heading_length = 2.0

        heading_end = _world_to_pixel(
            float(vehicle["x"])
            + heading_length
            * np.cos(float(vehicle["yaw"])),
            float(vehicle["z"])
            + heading_length
            * np.sin(float(vehicle["yaw"])),
            x_min,
            z_max,
            ppm,
        )

        cv2.arrowedLine(
            canvas,
            centre,
            heading_end,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
            tipLength=0.25,
        )

        if show_track_ids:
            label = (
                f"T{vehicle['track_id']} "
                f"{vehicle.get('class_name', '')}"
            )

            cv2.putText(
                canvas,
                label,
                (
                    centre[0] + 4,
                    centre[1] - 4,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

    # ---------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------

    road_plane = pipeline.last_road_plane or {}

    lines = [
        (
            f"current={len(pipeline.last_current_vehicles)}  "
            f"temporal={len(pipeline.last_temporal_vehicles)}"
        ),
        (
            f"tracks={len(pipeline.last_tracks)}  "
            f"streams={len(pipeline.last_streams)}  "
            f"fits={len(pipeline.last_fits)}  "
            f"boundaries={len(pipeline.last_boundaries)}"
        ),
        (
            f"road={road_plane.get('model', 'none')}  "
            f"rmse={road_plane.get('rmse', 0.0):.3f} m"
        ),
    ]

    y = 20

    for text in lines:
        cv2.putText(
            canvas,
            text,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        y += 18

    return canvas