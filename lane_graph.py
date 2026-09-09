import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Polygon

from configs import(
    LaneGraphConfig,
    TemporalConfig,
    LeadVehicleConfig,
    LaneFitConfig,
    LaneMergeConfig,
    LaneBoundaryConfig,
    BoundaryTrackingConfig,
    RoadPlaneConfig,
    LaneProjectionConfig,
)

from geometry import (
    axial_angle_diff,
    heading_from_yaw,
    pair_metrics,
)

from lane_fitting import (
    evaluate_lane_polynomial,
    lane_polynomial_slope,
)

def extract_vehicles_from_prediction(pred_instances_3d, vehicle_label_ids, cfg=None):
    if cfg is None:
        cfg = LaneGraphConfig()

    boxes = pred_instances_3d.bboxes_3d
    scores = pred_instances_3d.scores_3d.detach().cpu()
    labels = pred_instances_3d.labels_3d.detach().cpu()
    bev = boxes.bev.detach().cpu().numpy()

    if hasattr(boxes, "gravity_center"):
        centres = boxes.gravity_center.detach().cpu().numpy()
    else:
        centres = np.column_stack(
            [bev[:, 0], np.zeros(len(bev), dtype=np.float64), bev[:, 1]]
        )

    vehicle_label_ids = set(vehicle_label_ids)
    vehicles = []

    for original_idx in range(len(boxes)):
        score = float(scores[original_idx])
        label = int(labels[original_idx])
        if label not in vehicle_label_ids:
            continue

        x = float(bev[original_idx, 0])
        z = float(bev[original_idx, 1])
        if z <= 0 or z > cfg.max_depth:
            continue

        passes_standard_score = score >= cfg.score_thresh
        passes_lead_score = (
            score >= cfg.lead_score_thresh
            and abs(x) <= cfg.lead_candidate_max_abs_x
        )
        if not (passes_standard_score or passes_lead_score):
            continue

        vehicles.append(
            {
                "original_idx": original_idx,
                "x": x,
                "y": float(centres[original_idx, 1]),
                "z": z,
                "yaw": float(bev[original_idx, 4]),
                "length": float(bev[original_idx, 2]),
                "width": float(bev[original_idx, 3]),
                "score": score,
                "evidence_weight": 1.0,
                "label": label,
                "below_standard_score": not passes_standard_score,
            }
        )

    return vehicles


def build_lane_compatibility_graph_from_vehicles(vehicles, cfg=None):
    if cfg is None:
        cfg = LaneGraphConfig()

    graph = nx.Graph()
    for node_id, vehicle in enumerate(vehicles):
        graph.add_node(node_id, **vehicle)

    max_yaw = math.radians(cfg.max_yaw_diff_deg)
    sigma_yaw = math.radians(cfg.sigma_yaw_deg)

    for i, vi in enumerate(vehicles):
        p_i = np.array([vi["x"], vi["z"]])
        for j in range(i + 1, len(vehicles)):
            vj = vehicles[j]
            p_j = np.array([vj["x"], vj["z"]])
            metrics = pair_metrics(p_i, vi["yaw"], p_j, vj["yaw"])

            cross = metrics["cross_track"]
            yaw_diff = metrics["yaw_diff"]
            along = metrics["along_track"]
            if (
                cross > cfg.max_cross_track
                or yaw_diff > max_yaw
                or along > cfg.max_along_track
            ):
                continue

            compatibility = np.exp(
                -0.5
                * (
                    (cross / cfg.sigma_cross_track) ** 2
                    + (yaw_diff / sigma_yaw) ** 2
                )
            )
            score_i = vi["score"] * vi.get("evidence_weight", 1.0)
            score_j = vj["score"] * vj.get("evidence_weight", 1.0)
            confidence = math.sqrt(max(score_i, 0.0) * max(score_j, 0.0))

            graph.add_edge(
                i,
                j,
                weight=float(compatibility * confidence),
                cross_track=cross,
                yaw_diff_deg=math.degrees(yaw_diff),
                along_track=along,
            )

    return graph


def build_lane_compatibility_graph(pred_instances_3d, vehicle_label_ids, cfg=None):
    vehicles = extract_vehicles_from_prediction(
        pred_instances_3d, vehicle_label_ids, cfg=cfg
    )
    return build_lane_compatibility_graph_from_vehicles(vehicles, cfg=cfg), vehicles



def get_lane_streams(graph, min_vehicles=2):
    return [
        sorted(component)
        for component in nx.connected_components(graph)
        if len(component) >= min_vehicles
    ]


def print_graph_edges(graph):
    for i, j, data in graph.edges(data=True):
        print(
            f"{i:2d} <-> {j:2d} | "
            f"cross={data['cross_track']:.2f} m | "
            f"yaw={data['yaw_diff_deg']:.1f} deg | "
            f"along={data['along_track']:.1f} m | "
            f"weight={data['weight']:.3f}"
        )




def ensure_lead_vehicle_stream(
    graph,
    streams,
    lane_fits,
    current_frame_index,
    cfg=None,
):
    """Guarantee that the current front-centre vehicle supports a lane fit.

    Ordinary stream fitting needs at least two observations at distinct
    forward depths. A stopped vehicle can produce several ego-compensated
    observations at essentially one point, so it may have a valid track but no
    fit. In that case its FCOS3D yaw supplies the tangent of a conservative
    one-vehicle anchor line.

    The guarantee starts after detection: if FCOS3D produces no eligible
    current-frame vehicle, this function reports that fact and cannot invent
    one.
    """
    if cfg is None:
        cfg = LeadVehicleConfig()

    streams = [list(stream) for stream in streams]
    lane_fits = [dict(fit) for fit in lane_fits]
    diagnostic = {
        "status": "disabled" if not cfg.enabled else "no_current_vehicle",
        "selected_node": None,
        "track_id": None,
        "stream_id": None,
        "graph_degree": 0,
        "component_size": 0,
        "used_anchor": False,
    }
    if not cfg.enabled:
        return streams, lane_fits, diagnostic

    current_nodes = [
        node
        for node, vehicle in graph.nodes(data=True)
        if vehicle.get("frame_index") == current_frame_index
        and 0.0 < float(vehicle["z"]) <= cfg.max_depth
        and abs(float(vehicle["x"])) <= cfg.max_abs_x
    ]
    if not current_nodes:
        return streams, lane_fits, diagnostic

    forward_yaw = -0.5 * math.pi
    max_yaw_diff = math.radians(cfg.max_forward_yaw_diff_deg)
    aligned_nodes = [
        node
        for node in current_nodes
        if axial_angle_diff(graph.nodes[node]["yaw"], forward_yaw)
        <= max_yaw_diff
    ]
    candidates = aligned_nodes or current_nodes

    # Angular proximity to the optical axis identifies "the car in front"
    # more reliably than x alone at different depths. Prefer the nearer car
    # when angular offsets are effectively tied.
    lead_node = min(
        candidates,
        key=lambda node: (
            abs(
                math.atan2(
                    float(graph.nodes[node]["x"]),
                    float(graph.nodes[node]["z"]),
                )
            ),
            float(graph.nodes[node]["z"]),
        ),
    )
    lead = graph.nodes[lead_node]
    diagnostic.update(
        {
            "selected_node": int(lead_node),
            "track_id": lead.get("track_id"),
            "x": float(lead["x"]),
            "z": float(lead["z"]),
            "yaw_deg": float(math.degrees(lead["yaw"])),
            "score": float(lead["score"]),
            "below_standard_score": bool(
                lead.get("below_standard_score", False)
            ),
            "track_observations": int(lead.get("track_observations", 1)),
            "track_confirmed": bool(lead.get("track_confirmed", True)),
            "graph_degree": int(graph.degree[lead_node]),
            "component_size": int(
                len(nx.node_connected_component(graph, lead_node))
            ),
        }
    )

    for fit in lane_fits:
        if lead_node not in fit.get("inliers", []):
            continue
        fit.update(
            {
                "is_lead_stream": True,
                "forced_lead": False,
                "lead_node_id": int(lead_node),
                "lead_track_id": lead.get("track_id"),
                "lead_score": float(lead["score"]),
            }
        )
        diagnostic.update(
            {
                "status": "existing_fit",
                "stream_id": int(fit["stream_id"]),
            }
        )
        return streams, lane_fits, diagnostic

    heading = heading_from_yaw(float(lead["yaw"]))
    if abs(float(heading[1])) < 1e-6:
        raw_slope = 0.0
        slope_fallback = True
    else:
        raw_slope = float(heading[0] / heading[1])
        slope_fallback = abs(raw_slope) > cfg.max_abs_slope

    slope = (
        0.0
        if slope_fallback
        else float(np.clip(raw_slope, -cfg.max_abs_slope, cfg.max_abs_slope))
    )
    intercept = float(lead["x"] - slope * lead["z"])
    z_min = float(min(lead["z"], max(cfg.near_depth, 1.0)))
    z_max = float(
        min(cfg.max_depth, max(lead["z"] + cfg.forward_extension, z_min + 1.0))
    )

    stream_id = max(
        [int(fit.get("stream_id", -1)) for fit in lane_fits],
        default=-1,
    ) + 1
    track_id = lead.get("track_id")
    anchor_fit = {
        "stream_id": stream_id,
        "source_stream_ids": [],
        "degree": 1,
        "coefficients": np.array([intercept, slope], dtype=np.float64),
        "inliers": [int(lead_node)],
        "outliers": [],
        "rmse": 0.0,
        "z_min": z_min,
        "z_max": z_max,
        "num_observations": 1,
        "track_ids": [] if track_id is None else [int(track_id)],
        "num_tracks": 0 if track_id is None else 1,
        "has_track_info": track_id is not None,
        "track_support": "lead_anchor",
        "source": "lead_vehicle_anchor",
        "is_lead_stream": True,
        "forced_lead": True,
        "lead_node_id": int(lead_node),
        "lead_track_id": track_id,
        "lead_score": float(lead["score"]),
        "yaw_slope_fallback": bool(slope_fallback),
    }
    streams.append([int(lead_node)])
    lane_fits.append(anchor_fit)
    diagnostic.update(
        {
            "status": "anchored_fit",
            "stream_id": int(stream_id),
            "used_anchor": True,
            "yaw_slope_fallback": bool(slope_fallback),
            "anchor_slope": float(slope),
            "z_min": z_min,
            "z_max": z_max,
        }
    )
    return streams, lane_fits, diagnostic


def _average_polynomials(a, b):
    degree = max(len(a), len(b))
    a = np.pad(np.asarray(a, dtype=np.float64), (0, degree - len(a)))
    b = np.pad(np.asarray(b, dtype=np.float64), (0, degree - len(b)))
    return 0.5 * (a + b)


def infer_lane_boundaries(lane_fits, cfg=None):
    if cfg is None:
        cfg = LaneBoundaryConfig()

    boundaries = []
    paired_boundaries = []

    def boundary_separation(a, b):
        """Median lateral separation over the shared z-range."""
        z_min = max(a["z_min"], b["z_min"])
        z_max = min(a["z_max"], b["z_max"])
        if z_max <= z_min:
            return None

        z = np.linspace(z_min, z_max, cfg.sample_count)
        x_a = evaluate_lane_polynomial(a["coefficients"], z)
        x_b = evaluate_lane_polynomial(b["coefficients"], z)
        return float(np.median(np.abs(x_a - x_b)))

    def add_boundary(candidate):
        """Add a boundary unless it duplicates a stronger existing one."""
        for index, existing in enumerate(boundaries):
            separation = boundary_separation(candidate, existing)
            if (
                separation is not None
                and separation < cfg.provisional_dedup_distance
            ):
                if candidate["confidence"] > existing["confidence"]:
                    boundaries[index] = candidate
                return

        boundaries.append(candidate)

    def make_offset_boundary(fit, side, lane_width):
        """Offset one fitted lane centreline by half a lane width."""
        z = np.linspace(fit["z_min"], fit["z_max"], cfg.sample_count)
        centre_x = evaluate_lane_polynomial(fit["coefficients"], z)
        slope = lane_polynomial_slope(fit["coefficients"], z)

        half_width = lane_width / 2.0
        horizontal_offset = half_width * np.sqrt(1.0 + slope**2)

        if side == "left":
            boundary_x = centre_x - horizontal_offset
        elif side == "right":
            boundary_x = centre_x + horizontal_offset
        else:
            raise ValueError("side must be 'left' or 'right'")

        degree = min(fit.get("degree", 2), len(z) - 1)
        poly_desc = np.polyfit(z, boundary_x, degree)
        return poly_desc[::-1].copy()

    # ---------------------------------------------------------
    # 1. High-confidence paired-stream boundaries
    # ---------------------------------------------------------
    for i in range(len(lane_fits)):
        for j in range(i + 1, len(lane_fits)):
            a, b = lane_fits[i], lane_fits[j]

            z_min = max(a["z_min"], b["z_min"])
            z_max = min(a["z_max"], b["z_max"])
            overlap = z_max - z_min

            print(
                f"\nS{a['stream_id']} vs S{b['stream_id']}: "
                f"z=[{z_min:.1f}, {z_max:.1f}], "
                f"overlap={overlap:.2f} m"
            )

            if overlap < cfg.min_overlap:
                print(
                    f"  -> rejected: insufficient overlap "
                    f"({overlap:.2f} < {cfg.min_overlap:.2f} m)"
                )
                continue

            z = np.linspace(z_min, z_max, cfg.sample_count)
            x_a = evaluate_lane_polynomial(a["coefficients"], z)
            x_b = evaluate_lane_polynomial(b["coefficients"], z)

            delta = x_b - x_a
            median_delta = float(np.median(delta))
            median_x_gap = float(np.median(np.abs(delta)))

            if abs(median_delta) < 1e-6:
                print("  -> rejected: median lateral separation is effectively zero")
                continue

            sign_consistency = float(
                np.mean(np.sign(delta) == np.sign(median_delta))
            )

            print(
                f"  median x-gap={median_x_gap:.2f} m | "
                f"left/right consistency={sign_consistency:.2%}"
            )

            if sign_consistency < 0.9:
                print(
                    "  -> rejected: streams do not maintain a consistent "
                    "left/right ordering"
                )
                continue

            if median_delta > 0:
                left, right = a, b
                x_left, x_right = x_a, x_b
            else:
                left, right = b, a
                x_left, x_right = x_b, x_a

            slope_left = lane_polynomial_slope(left["coefficients"], z)
            slope_right = lane_polynomial_slope(right["coefficients"], z)
            mean_slope = 0.5 * (slope_left + slope_right)

            widths = (x_right - x_left) / np.sqrt(1.0 + mean_slope**2)
            median_width = float(np.median(widths))
            width_std = float(np.std(widths))

            print(
                f"  normal lane width={median_width:.2f} m | "
                f"std={width_std:.2f} m"
            )

            if not (
                cfg.min_lane_width <= median_width <= cfg.max_lane_width
            ):
                print(
                    f"  -> rejected: lane width outside allowed range "
                    f"[{cfg.min_lane_width:.2f}, {cfg.max_lane_width:.2f}] m"
                )
                continue

            overlap_score = min(
                1.0,
                overlap / max(2.0 * cfg.min_overlap, 1e-6),
            )
            width_score = float(np.exp(-width_std / 0.5))

            # Two or more distinct tracked vehicles per centreline is stronger
            # evidence than a trajectory formed from repeated observations of
            # only one vehicle. If no track metadata exists, do not penalise.
            if left.get("has_track_info") and right.get("has_track_info"):
                track_score = min(
                    1.0,
                    min(left.get("num_tracks", 0), right.get("num_tracks", 0)) / 2.0,
                )
            else:
                track_score = 1.0

            confidence = float(
                0.55 + 0.45 * overlap_score * width_score * track_score
            )

            boundary = {
                "left_stream_id": left["stream_id"],
                "right_stream_id": right["stream_id"],
                "coefficients": _average_polynomials(
                    left["coefficients"], right["coefficients"]
                ),
                "z_min": float(z_min),
                "z_max": float(z_max),
                "overlap": float(overlap),
                "lane_width": median_width,
                "lane_width_std": width_std,
                "source": "paired_streams",
                "confidence": confidence,
                "side": "between",
                "forced_lead": bool(
                    left.get("is_lead_stream") or right.get("is_lead_stream")
                ),
            }

            paired_boundaries.append(boundary)
            add_boundary(boundary)

            print(
                f"  -> ACCEPTED paired boundary "
                f"(confidence={confidence:.2f})"
            )

    # ---------------------------------------------------------
    # 2. Estimate representative lane width
    # ---------------------------------------------------------
    if paired_boundaries:
        estimated_lane_width = float(
            np.median([b["lane_width"] for b in paired_boundaries])
        )
    else:
        estimated_lane_width = cfg.default_lane_width

    print(
        f"\nLane width used for single-stream inference: "
        f"{estimated_lane_width:.2f} m"
    )

    if not cfg.enable_single_stream_boundaries:
        boundaries.sort(
            key=lambda boundary: float(
                evaluate_lane_polynomial(
                    boundary["coefficients"],
                    0.5 * (boundary["z_min"] + boundary["z_max"]),
                )
            )
        )
        print("Single-stream provisional boundaries: disabled")
        print(f"\nTotal inferred lane boundaries: {len(boundaries)}")
        return boundaries

    # ---------------------------------------------------------
    # 3. Lower-confidence single-stream boundaries
    # ---------------------------------------------------------
    eligible_fits = []
    for fit in lane_fits:
        inlier_count = len(fit.get("inliers", []))
        rmse = float(fit.get("rmse", np.inf))
        span = float(fit["z_max"] - fit["z_min"])
        forced_lead = bool(fit.get("forced_lead", False))
        lead_stream = bool(fit.get("is_lead_stream", False))

        # A forced lead anchor is the deliberate exception to the ordinary
        # multi-observation quality gates. It is created from one current car
        # plus its FCOS3D yaw precisely because no normal line fit was possible.
        if lead_stream:
            eligible_fits.append(fit)
            continue

        if cfg.single_stream_only_when_no_paired and paired_boundaries:
            print(
                f"S{fit['stream_id']}: single-stream fallback suppressed "
                "because paired-stream boundaries exist"
            )
            continue

        if inlier_count < cfg.single_stream_min_inliers:
            print(
                f"S{fit['stream_id']}: not enough inliers for "
                f"single-stream boundaries ({inlier_count})"
            )
            continue

        if span < cfg.single_stream_min_span:
            print(
                f"S{fit['stream_id']}: fitted span too short "
                f"({span:.2f} m)"
            )
            continue

        if (
            fit.get("has_track_info")
            and fit.get("num_tracks", 0) < cfg.single_stream_min_tracks
        ):
            print(
                f"S{fit['stream_id']}: insufficient distinct tracks "
                f"({fit.get('num_tracks', 0)})"
            )
            continue

        if rmse > cfg.single_stream_max_rmse:
            print(
                f"S{fit['stream_id']}: fit RMSE too high "
                f"({rmse:.2f} m)"
            )
            continue

        eligible_fits.append(fit)

    forced_fits = [fit for fit in eligible_fits if fit.get("is_lead_stream")]
    ordinary_fits = [fit for fit in eligible_fits if not fit.get("is_lead_stream")]
    ordinary_fits.sort(
        key=lambda fit: (
            -len(fit.get("inliers", [])),
            float(fit.get("rmse", np.inf)),
        )
    )
    ordinary_fits = ordinary_fits[: max(0, cfg.max_single_stream_fits)]

    for fit in forced_fits + ordinary_fits:
        inlier_count = len(fit.get("inliers", []))
        rmse = float(fit.get("rmse", 0.0))
        forced_lead = bool(fit.get("forced_lead", False))
        lead_stream = bool(fit.get("is_lead_stream", False))

        observation_score = 1.0 if lead_stream else min(1.0, inlier_count / 5.0)
        fit_score = float(
            np.exp(
                -rmse / max(cfg.single_stream_max_rmse, 1e-6)
            )
        )

        if fit.get("has_track_info"):
            num_tracks = fit.get("num_tracks", 0)
            # One tracked car remains useful trajectory evidence, but two or
            # more independent cars provide full support.
            track_score = min(1.0, num_tracks / 2.0)
        else:
            track_score = 1.0

        if lead_stream:
            provisional_confidence = float(
                cfg.single_stream_confidence
                * max(0.5, float(fit.get("lead_score", 1.0)))
            )
            boundary_source = (
                "lead_vehicle_anchor"
                if forced_lead
                else "lead_vehicle_stream"
            )
        else:
            provisional_confidence = float(
                cfg.single_stream_confidence
                * observation_score
                * fit_score
                * track_score
            )
            boundary_source = "single_stream"

        left_boundary = {
            "left_stream_id": None,
            "right_stream_id": fit["stream_id"],
            "coefficients": make_offset_boundary(
                fit, "left", estimated_lane_width
            ),
            "z_min": float(fit["z_min"]),
            "z_max": float(fit["z_max"]),
            "overlap": float(fit["z_max"] - fit["z_min"]),
            "lane_width": estimated_lane_width,
            "lane_width_std": 0.0,
            "source": boundary_source,
            "confidence": provisional_confidence,
            "side": "left",
            "source_stream_id": fit["stream_id"],
            "forced_lead": lead_stream,
        }
        add_boundary(left_boundary)

        right_boundary = {
            "left_stream_id": fit["stream_id"],
            "right_stream_id": None,
            "coefficients": make_offset_boundary(
                fit, "right", estimated_lane_width
            ),
            "z_min": float(fit["z_min"]),
            "z_max": float(fit["z_max"]),
            "overlap": float(fit["z_max"] - fit["z_min"]),
            "lane_width": estimated_lane_width,
            "lane_width_std": 0.0,
            "source": boundary_source,
            "confidence": provisional_confidence,
            "side": "right",
            "source_stream_id": fit["stream_id"],
            "forced_lead": lead_stream,
        }
        add_boundary(right_boundary)

        print(
            f"S{fit['stream_id']}: added provisional left/right boundaries | "
            f"tracks={fit.get('num_tracks', 0)} | "
            f"support={fit.get('track_support', 'unknown')} | "
            f"confidence={provisional_confidence:.2f}"
        )

    boundaries.sort(
        key=lambda boundary: float(
            evaluate_lane_polynomial(
                boundary["coefficients"],
                0.5 * (boundary["z_min"] + boundary["z_max"]),
            )
        )
    )

    print(f"\nTotal inferred lane boundaries: {len(boundaries)}")
    return boundaries



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


def transform_lane_boundary_to_camera(
    boundary,
    source_road_plane,
    source_cam_to_global,
    target_cam_to_global,
    cfg=None,
):
    """
    Ego-motion compensate a lane boundary into a new camera frame.

    The boundary is sampled as 3D road points in the source camera, rigidly
    transformed through global coordinates, then re-fitted as x(z) in the
    target camera. This avoids directly transforming polynomial coefficients.
    """
    if cfg is None:
        cfg = BoundaryTrackingConfig()
    if source_road_plane is None:
        return None

    z = np.linspace(
        float(boundary["z_min"]),
        float(boundary["z_max"]),
        max(cfg.sample_count, cfg.min_points_after_transform),
    )
    x = evaluate_lane_polynomial(boundary["coefficients"], z)
    y = road_plane_y(source_road_plane, x, z)

    source_points = np.column_stack(
        [x, y, z, np.ones_like(z, dtype=np.float64)]
    )

    source_cam_to_global = np.asarray(source_cam_to_global, dtype=np.float64)
    target_cam_to_global = np.asarray(target_cam_to_global, dtype=np.float64)
    target_from_source = (
        np.linalg.inv(target_cam_to_global) @ source_cam_to_global
    )
    target_points = source_points @ target_from_source.T

    target_x = target_points[:, 0]
    target_z = target_points[:, 2]
    valid = (
        np.isfinite(target_x)
        & np.isfinite(target_z)
        & (target_z >= cfg.min_depth)
    )
    target_x = target_x[valid]
    target_z = target_z[valid]

    if len(target_z) < cfg.min_points_after_transform:
        return None

    order = np.argsort(target_z)
    target_z = target_z[order]
    target_x = target_x[order]

    degree = min(
        max(1, len(np.asarray(boundary["coefficients"])) - 1),
        len(target_z) - 1,
    )
    if np.unique(target_z).size <= degree:
        return None

    try:
        fit_desc = np.polyfit(target_z, target_x, degree)
    except (np.linalg.LinAlgError, ValueError):
        return None

    transformed = dict(boundary)
    transformed.update(
        {
            "coefficients": fit_desc[::-1].copy(),
            "z_min": float(target_z.min()),
            "z_max": float(target_z.max()),
            "overlap": float(target_z.max() - target_z.min()),
            "ego_motion_compensated": True,
        }
    )
    return transformed


def _boundary_temporal_metrics(previous_boundary, current_boundary, cfg):
    z_min = max(previous_boundary["z_min"], current_boundary["z_min"])
    z_max = min(previous_boundary["z_max"], current_boundary["z_max"])
    overlap = float(z_max - z_min)

    if overlap < cfg.min_overlap:
        return {
            "compatible": False,
            "overlap": overlap,
            "median_lateral_distance": np.inf,
            "max_tangent_diff_deg": np.inf,
            "cost": np.inf,
        }

    z = np.linspace(z_min, z_max, max(3, cfg.sample_count))
    x_previous = evaluate_lane_polynomial(previous_boundary["coefficients"], z)
    x_current = evaluate_lane_polynomial(current_boundary["coefficients"], z)
    lateral = np.abs(x_previous - x_current)

    previous_slope = lane_polynomial_slope(previous_boundary["coefficients"], z)
    current_slope = lane_polynomial_slope(current_boundary["coefficients"], z)
    previous_angle = np.arctan(previous_slope)
    current_angle = np.arctan(current_slope)
    tangent_diff = np.abs(previous_angle - current_angle)
    tangent_diff = np.minimum(tangent_diff, np.pi - tangent_diff)

    median_lateral = float(np.median(lateral))
    max_tangent_deg = float(np.degrees(np.max(tangent_diff)))

    compatible = (
        median_lateral <= cfg.max_lateral_distance
        and max_tangent_deg <= cfg.max_tangent_diff_deg
    )

    previous_extent = max(
        1e-6,
        float(previous_boundary["z_max"] - previous_boundary["z_min"]),
    )
    current_extent = max(
        1e-6,
        float(current_boundary["z_max"] - current_boundary["z_min"]),
    )
    overlap_fraction = min(1.0, overlap / min(previous_extent, current_extent))

    cost = (
        median_lateral / max(cfg.max_lateral_distance, 1e-6)
        + max_tangent_deg / max(cfg.max_tangent_diff_deg, 1e-6)
        + 0.25 * (1.0 - overlap_fraction)
    )

    return {
        "compatible": bool(compatible),
        "overlap": overlap,
        "median_lateral_distance": median_lateral,
        "max_tangent_diff_deg": max_tangent_deg,
        "cost": float(cost),
    }


def _smooth_boundary_geometry(previous_boundary, current_boundary, cfg):
    """Smooth sampled x(z) geometry, then refit the current polynomial model."""
    z = np.linspace(
        current_boundary["z_min"],
        current_boundary["z_max"],
        max(6, cfg.sample_count),
    )
    x_current = evaluate_lane_polynomial(current_boundary["coefficients"], z)
    x_smoothed = x_current.copy()

    overlap_mask = (
        (z >= previous_boundary["z_min"])
        & (z <= previous_boundary["z_max"])
    )

    if np.any(overlap_mask):
        x_previous = evaluate_lane_polynomial(
            previous_boundary["coefficients"], z[overlap_mask]
        )
        alpha = float(np.clip(cfg.smoothing_alpha, 0.0, 1.0))
        x_smoothed[overlap_mask] = (
            alpha * x_current[overlap_mask]
            + (1.0 - alpha) * x_previous
        )

    degree = min(
        max(1, len(np.asarray(current_boundary["coefficients"])) - 1),
        len(z) - 1,
    )
    try:
        fit_desc = np.polyfit(z, x_smoothed, degree)
    except (np.linalg.LinAlgError, ValueError):
        return dict(current_boundary)

    result = dict(current_boundary)
    result["coefficients"] = fit_desc[::-1].copy()

    alpha = float(np.clip(cfg.smoothing_alpha, 0.0, 1.0))
    previous_confidence = float(previous_boundary.get("confidence", 1.0))
    current_confidence = float(current_boundary.get("confidence", 1.0))
    result["confidence"] = float(
        alpha * current_confidence + (1.0 - alpha) * previous_confidence
    )
    result["smoothed"] = True
    return result


def update_temporal_lane_tracks(
    state,
    current_boundaries,
    current_cam_to_global,
    current_road_plane,
    frame_index,
    cfg=None,
):
    """
    Associate, smooth and persist lane boundaries across scene frames.

    Parameters
    ----------
    state : dict or None
        Persistent tracker state returned by the previous call.
    current_boundaries : list[dict]
        Current-frame measurements from infer_lane_boundaries().
    current_cam_to_global : ndarray (4, 4)
        Pose of the current camera.
    current_road_plane : dict or None
        Current road plane estimate. If unavailable, transformed previous
        planes are used for carried tracks when possible.
    frame_index : int
        Scene frame index.

    Returns
    -------
    output_boundaries : list[dict]
        Confirmed current measurements and, when configured, short-lived
        predictions or unconfirmed measurements. All live tracks remain in
        state even when they are suppressed from this output.
    state : dict
        Updated persistent tracker state.
    events : list[dict]
        Association diagnostics for logging/evaluation.
    """
    if cfg is None:
        cfg = BoundaryTrackingConfig()
    if state is None:
        state = {"tracks": {}, "next_track_id": 0}

    current_cam_to_global = np.asarray(current_cam_to_global, dtype=np.float64)
    previous_tracks = dict(state.get("tracks", {}))
    next_track_id = int(state.get("next_track_id", 0))

    transformed_tracks = {}
    for track_id, track in previous_tracks.items():
        source_plane = track.get("road_plane")
        if source_plane is None:
            continue

        transformed_boundary = transform_lane_boundary_to_camera(
            track["boundary"],
            source_plane,
            track["cam_to_global"],
            current_cam_to_global,
            cfg=cfg,
        )
        if transformed_boundary is None:
            continue

        transformed_plane = transform_road_plane_to_camera(
            source_plane,
            track["cam_to_global"],
            current_cam_to_global,
        )
        transformed_tracks[track_id] = {
            "track": track,
            "boundary": transformed_boundary,
            "road_plane": transformed_plane,
        }

    candidates = []
    for track_id, predicted in transformed_tracks.items():
        for measurement_index, boundary in enumerate(current_boundaries):
            metrics = _boundary_temporal_metrics(
                predicted["boundary"], boundary, cfg
            )
            if metrics["compatible"]:
                candidates.append(
                    (metrics["cost"], track_id, measurement_index, metrics)
                )

    candidates.sort(key=lambda item: item[0])
    matched_tracks = set()
    matched_measurements = set()
    accepted_matches = []

    for _, track_id, measurement_index, metrics in candidates:
        if track_id in matched_tracks or measurement_index in matched_measurements:
            continue
        matched_tracks.add(track_id)
        matched_measurements.add(measurement_index)
        accepted_matches.append((track_id, measurement_index, metrics))

    output_boundaries = []
    new_tracks = {}
    events = []

    # Matched measurements: ego compensate previous geometry, smooth in x(z),
    # and preserve the same persistent boundary ID.
    for track_id, measurement_index, metrics in accepted_matches:
        previous_record = transformed_tracks[track_id]
        old_track = previous_record["track"]
        current = current_boundaries[measurement_index]
        fused = _smooth_boundary_geometry(
            previous_record["boundary"], current, cfg
        )

        age = int(old_track.get("age", 1)) + 1
        hits = int(old_track.get("hits", 1)) + 1
        fused.update(
            {
                "boundary_track_id": int(track_id),
                "temporal_status": "matched",
                "track_age": age,
                "track_hits": hits,
                "missed_frames": 0,
                "is_predicted": False,
            }
        )

        track_plane = (
            current_road_plane
            if current_road_plane is not None
            else previous_record["road_plane"]
        )
        new_tracks[track_id] = {
            "track_id": int(track_id),
            "boundary": dict(fused),
            "cam_to_global": current_cam_to_global.copy(),
            "road_plane": track_plane,
            "last_frame_index": int(frame_index),
            "age": age,
            "hits": hits,
            "missed_frames": 0,
        }
        confirmed = hits >= cfg.min_confirmed_hits
        force_emit = bool(fused.get("forced_lead", False))
        if confirmed or cfg.emit_unconfirmed or force_emit:
            output_boundaries.append(fused)
        events.append(
            {
                "type": "matched",
                "boundary_track_id": int(track_id),
                "measurement_index": int(measurement_index),
                **metrics,
            }
        )

    # New measurements start new persistent lane-boundary tracks.
    for measurement_index, boundary in enumerate(current_boundaries):
        if measurement_index in matched_measurements:
            continue

        track_id = next_track_id
        next_track_id += 1
        new_boundary = dict(boundary)
        new_boundary.update(
            {
                "boundary_track_id": int(track_id),
                "temporal_status": "new",
                "track_age": 1,
                "track_hits": 1,
                "missed_frames": 0,
                "is_predicted": False,
                "smoothed": False,
            }
        )
        new_tracks[track_id] = {
            "track_id": int(track_id),
            "boundary": dict(new_boundary),
            "cam_to_global": current_cam_to_global.copy(),
            "road_plane": current_road_plane,
            "last_frame_index": int(frame_index),
            "age": 1,
            "hits": 1,
            "missed_frames": 0,
        }
        confirmed = 1 >= cfg.min_confirmed_hits
        force_emit = bool(new_boundary.get("forced_lead", False))
        if confirmed or cfg.emit_unconfirmed or force_emit:
            output_boundaries.append(new_boundary)
        events.append(
            {
                "type": "new",
                "boundary_track_id": int(track_id),
                "measurement_index": int(measurement_index),
            }
        )

    # Unmatched prior tracks are carried for a short period with decaying
    # confidence. This bridges occasional missed FCOS3D / lane-fit frames.
    for track_id, previous_record in transformed_tracks.items():
        if track_id in matched_tracks:
            continue

        old_track = previous_record["track"]
        missed_frames = int(old_track.get("missed_frames", 0)) + 1
        if missed_frames > cfg.max_missed_frames:
            events.append(
                {
                    "type": "expired",
                    "boundary_track_id": int(track_id),
                    "missed_frames": missed_frames,
                }
            )
            continue

        carried = dict(previous_record["boundary"])
        carried["confidence"] = float(
            carried.get("confidence", 1.0)
            * cfg.missing_confidence_decay
        )
        age = int(old_track.get("age", 1)) + 1
        hits = int(old_track.get("hits", 1))
        carried.update(
            {
                "boundary_track_id": int(track_id),
                "temporal_status": "predicted",
                "track_age": age,
                "track_hits": hits,
                "missed_frames": missed_frames,
                "is_predicted": True,
                "smoothed": bool(carried.get("smoothed", False)),
            }
        )

        carried_plane = previous_record["road_plane"]
        new_tracks[track_id] = {
            "track_id": int(track_id),
            "boundary": dict(carried),
            "cam_to_global": current_cam_to_global.copy(),
            "road_plane": carried_plane,
            "last_frame_index": int(frame_index),
            "age": age,
            "hits": hits,
            "missed_frames": missed_frames,
        }
        confirmed = hits >= cfg.min_confirmed_hits
        if cfg.emit_predicted and (confirmed or cfg.emit_unconfirmed):
            output_boundaries.append(carried)
        events.append(
            {
                "type": "predicted",
                "boundary_track_id": int(track_id),
                "missed_frames": missed_frames,
            }
        )

    def lateral_key(boundary):
        z_mid = 0.5 * (boundary["z_min"] + boundary["z_max"])
        return float(
            evaluate_lane_polynomial(boundary["coefficients"], z_mid)
        )

    output_boundaries.sort(key=lateral_key)
    state = {
        "tracks": new_tracks,
        "next_track_id": next_track_id,
    }
    return output_boundaries, state, events


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


def plot_projected_lane_boundaries(
    image_path, projected_boundaries, save_path=None, show=True, title=None
):
    image = plt.imread(image_path)
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(image)

    for boundary in projected_boundaries:
        pixels = boundary["pixels"]
        if len(pixels) < 2:
            continue

        is_provisional = boundary.get("source") == "single_stream"
        is_predicted = bool(boundary.get("is_predicted", False))
        track_id = boundary.get("boundary_track_id")
        status = boundary.get("temporal_status", "measurement")

        if is_predicted:
            linestyle = "--"
            linewidth = 2
        elif is_provisional:
            linestyle = ":"
            linewidth = 5
        else:
            linestyle = "-"
            linewidth = 7

        if track_id is not None:
            label = f"B{track_id} {status}"
        elif is_provisional:
            label = (
                f"provisional S{boundary.get('source_stream_id')} "
                f"{boundary.get('side', '')}"
            )
        else:
            label = (
                f"boundary S{boundary['left_stream_id']}|"
                f"S{boundary['right_stream_id']}"
            )

        ax.plot(
            pixels[:, 0],
            pixels[:, 1],
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=max(0.20, float(boundary.get("confidence", 1.0))),
            label=label,
        )

    ax.set_title(title or "Projected temporally tracked lane boundaries")
    ax.axis("off")
    if projected_boundaries:
        ax.legend(loc="best")
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved image-space lane projection to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

def _vehicle_rectangle(x, z, yaw, length, width):
    heading = heading_from_yaw(yaw)
    normal = np.array([-heading[1], heading[0]])
    centre = np.array([x, z])
    half_length = length / 2.0
    half_width = width / 2.0
    return np.stack(
        [
            centre + half_length * heading + half_width * normal,
            centre + half_length * heading - half_width * normal,
            centre - half_length * heading - half_width * normal,
            centre - half_length * heading + half_width * normal,
        ]
    )


def plot_lane_graph(
    graph,
    streams=None,
    lane_fits=None,
    lane_boundaries=None,
    max_depth=60.0,
    x_range=(-15.0, 15.0),
    show_labels=True,
    save_path=None,
    show=True,
):
    if streams is None:
        streams = [list(c) for c in nx.connected_components(graph)]
    node_to_stream = {
        node: stream_id
        for stream_id, stream in enumerate(streams)
        for node in stream
    }

    fig, ax = plt.subplots(figsize=(9, 12))
    ax.scatter(0, 0, marker="^", s=150, label="Camera")
    ax.text(0.3, 0.5, "camera")

    for i, j, edge in graph.edges(data=True):
        vi, vj = graph.nodes[i], graph.nodes[j]
        ax.plot(
            [vi["x"], vj["x"]],
            [vi["z"], vj["z"]],
            linewidth=0.5 + 3.0 * edge.get("weight", 1.0),
            alpha=0.5,
        )

    for node, vehicle in graph.nodes(data=True):
        x, z, yaw = vehicle["x"], vehicle["z"], vehicle["yaw"]
        corners = _vehicle_rectangle(
            x, z, yaw, vehicle["length"], vehicle["width"]
        )
        ax.add_patch(Polygon(corners, closed=True, fill=False, linewidth=2))
        ax.scatter(x, z, s=50)
        ax.arrow(
            x,
            z,
            2.5 * np.cos(yaw),
            2.5 * np.sin(yaw),
            width=0.025,
            head_width=0.35,
            head_length=0.5,
            length_includes_head=True,
        )
        if show_labels:
            label = (
                f"{node}\nS{node_to_stream.get(node, -1)}\n{vehicle['score']:.2f}"
            )
            if "track_id" in vehicle:
                label += f"\nT{vehicle['track_id']} F{vehicle['frame_index']}"
            ax.text(x + 0.25, z + 0.25, label, fontsize=8)

    if lane_fits:
        for fit in lane_fits:
            z_curve = np.linspace(fit["z_min"], fit["z_max"], 100)
            x_curve = evaluate_lane_polynomial(fit["coefficients"], z_curve)
            source_ids = fit.get("source_stream_ids", [fit["stream_id"]])
            track_text = (
                f" T={fit.get('num_tracks', 0)}"
                if fit.get("has_track_info")
                else ""
            )
            source_text = (
                f" src={source_ids}"
                if len(source_ids) > 1
                else ""
            )
            ax.plot(
                x_curve,
                z_curve,
                linewidth=3,
                label=(
                    f"centre S{fit['stream_id']}"
                    f"{track_text}{source_text}"
                ),
            )

    if lane_boundaries:
        for boundary in lane_boundaries:
            z_curve = np.linspace(boundary["z_min"], boundary["z_max"], 100)
            x_curve = evaluate_lane_polynomial(boundary["coefficients"], z_curve)
            is_provisional = boundary.get("source") == "single_stream"
            is_predicted = bool(boundary.get("is_predicted", False))
            track_id = boundary.get("boundary_track_id")
            status = boundary.get("temporal_status", "measurement")

            if is_predicted:
                linestyle = "--"
                linewidth = 2
            elif is_provisional:
                linestyle = ":"
                linewidth = 5
            else:
                linestyle = "--"
                linewidth = 7
                

            if track_id is not None:
                label = f"B{track_id} {status}"
            elif is_provisional:
                label = (
                    f"provisional S{boundary.get('source_stream_id')} "
                    f"{boundary.get('side', '')}"
                )
            else:
                label = (
                    f"boundary S{boundary['left_stream_id']}|"
                    f"S{boundary['right_stream_id']}"
                )

            ax.plot(
                x_curve,
                z_curve,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=max(0.20, float(boundary.get("confidence", 1.0))),
                label=label,
            )

    ax.set_xlim(*x_range)
    ax.set_ylim(0, max_depth)
    ax.set_xlabel("Lateral x [m]")
    ax.set_ylabel("Forward z [m]")
    ax.set_title("Temporal FCOS3D lane graph")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    if lane_fits or lane_boundaries:
        ax.legend(loc="best")
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved BEV visualisation to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
