import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Polygon

from lane_inference.configs import(
    LaneGraphConfig,
    LeadVehicleConfig,
)

from lane_inference.road_plane import (
    road_plane_y,
    estimate_road_plane,
    transform_road_plane_to_camera,
)

from lane_inference.geometry import (
    axial_angle_diff,
    heading_from_yaw,
    pair_metrics,
)

from lane_inference.lane_fitting import (
    evaluate_lane_polynomial,
    lane_polynomial_slope,
)

from lane_inference.boundary_tracking import(
    transform_lane_boundary_to_camera,
    _boundary_temporal_metrics,
    _smooth_boundary_geometry,
    update_temporal_lane_tracks,
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
