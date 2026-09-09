import math
import numpy as np

from configs import LaneFitConfig, LaneMergeConfig

'''
from lane_fitting import (
    evaluate_lane_polynomial,
    lane_polynomial_slope,
    fit_lane_stream,
    fit_lane_streams,
    _fit_linear_merge_proxy,
    _lane_fit_comparison,
    merge_compatible_lane_streams,
)
'''

def evaluate_lane_polynomial(coefficients, z):
    z = np.asarray(z, dtype=np.float64)
    x = np.zeros_like(z, dtype=np.float64)
    for power, coefficient in enumerate(coefficients):
        x += coefficient * z**power
    return x


def lane_polynomial_slope(coefficients, z):
    z = np.asarray(z, dtype=np.float64)
    slope = np.zeros_like(z, dtype=np.float64)
    for power, coefficient in enumerate(coefficients[1:], start=1):
        slope += power * coefficient * z ** (power - 1)
    return slope


def fit_lane_stream(graph, stream, cfg=None):
    if cfg is None:
        cfg = LaneFitConfig()

    nodes = sorted(stream, key=lambda node: graph.nodes[node]["z"])
    if len(nodes) < 2:
        return None

    z = np.array([graph.nodes[n]["z"] for n in nodes], dtype=np.float64)
    x = np.array([graph.nodes[n]["x"] for n in nodes], dtype=np.float64)
    scores = np.array(
        [
            graph.nodes[n]["score"] * graph.nodes[n].get("evidence_weight", 1.0)
            for n in nodes
        ],
        dtype=np.float64,
    )

    degree = min(cfg.degree, len(nodes) - 1)
    sample_size = degree + 1
    trials = 1 if len(nodes) == sample_size else cfg.max_trials
    rng = np.random.default_rng(cfg.random_seed)
    best_mask = None
    best_count = -1
    best_rmse = np.inf

    for _ in range(trials):
        sample_idx = (
            np.arange(len(nodes))
            if len(nodes) == sample_size
            else rng.choice(len(nodes), size=sample_size, replace=False)
        )
        if np.unique(z[sample_idx]).size < sample_size:
            continue
        try:
            poly_desc = np.polyfit(z[sample_idx], x[sample_idx], degree)
        except (np.linalg.LinAlgError, ValueError):
            continue

        predicted = np.polyval(poly_desc, z)
        mask = np.abs(x - predicted) <= cfg.residual_threshold
        count = int(mask.sum())
        if count < sample_size:
            continue
        rmse = float(np.sqrt(np.mean((x[mask] - predicted[mask]) ** 2)))
        if count > best_count or (count == best_count and rmse < best_rmse):
            best_mask, best_count, best_rmse = mask, count, rmse

    if best_mask is None:
        return None

    inlier_z = z[best_mask]
    inlier_x = x[best_mask]
    weights = np.sqrt(np.clip(scores[best_mask], 1e-6, None))
    try:
        final_desc = np.polyfit(inlier_z, inlier_x, degree, w=weights)
    except (np.linalg.LinAlgError, ValueError):
        return None

    prediction = np.polyval(final_desc, inlier_z)
    inlier_nodes = [n for n, keep in zip(nodes, best_mask) if keep]
    outlier_nodes = [n for n, keep in zip(nodes, best_mask) if not keep]

    has_track_info = any("track_id" in graph.nodes[n] for n in inlier_nodes)
    track_ids = sorted(
        {
            int(graph.nodes[n]["track_id"])
            for n in inlier_nodes
            if graph.nodes[n].get("track_id") is not None
        }
    )
    num_tracks = len(track_ids)

    if not has_track_info:
        track_support = "unknown"
    elif num_tracks >= 2:
        track_support = "strong"
    else:
        track_support = "weak"

    return {
        "degree": degree,
        "coefficients": final_desc[::-1].copy(),
        "inliers": inlier_nodes,
        "outliers": outlier_nodes,
        "rmse": float(np.sqrt(np.mean((inlier_x - prediction) ** 2))),
        "z_min": float(inlier_z.min()),
        "z_max": float(inlier_z.max()),
        "num_observations": len(inlier_nodes),
        "track_ids": track_ids,
        "num_tracks": num_tracks,
        "has_track_info": has_track_info,
        "track_support": track_support,
    }


def fit_lane_streams(graph, streams, cfg=None):
    fits = []
    for stream_id, stream in enumerate(streams):
        fit = fit_lane_stream(graph, stream, cfg=cfg)
        if fit is not None:
            fit["stream_id"] = stream_id
            fit["source_stream_ids"] = [stream_id]
            fits.append(fit)
    return fits


def _fit_linear_merge_proxy(graph, fit):
    """
    Build a stable local x(z) line from a fit's inlier observations.

    Short quadratic fits can have unstable curvature outside their observed
    range. Stream merging therefore uses this local linear proxy only for the
    compatibility test; the final consolidated lane is still refitted with
    the configured RANSAC polynomial model.
    """
    nodes = fit.get("inliers", [])
    if len(nodes) >= 2:
        z = np.array([graph.nodes[n]["z"] for n in nodes], dtype=np.float64)
        x = np.array([graph.nodes[n]["x"] for n in nodes], dtype=np.float64)
        if np.unique(z).size >= 2:
            weights = np.sqrt(
                np.clip(
                    np.array(
                        [
                            graph.nodes[n]["score"]
                            * graph.nodes[n].get("evidence_weight", 1.0)
                            for n in nodes
                        ],
                        dtype=np.float64,
                    ),
                    1e-6,
                    None,
                )
            )
            try:
                slope, intercept = np.polyfit(z, x, 1, w=weights)
                return float(intercept), float(slope)
            except (np.linalg.LinAlgError, ValueError):
                pass

    # Fallback to the fitted polynomial's local tangent at its midpoint.
    z_mid = 0.5 * (fit["z_min"] + fit["z_max"])
    x_mid = float(evaluate_lane_polynomial(fit["coefficients"], z_mid))
    slope = float(lane_polynomial_slope(fit["coefficients"], z_mid))
    intercept = x_mid - slope * z_mid
    return float(intercept), float(slope)


def _lane_fit_comparison(graph, a, b, cfg):
    """Compare two fitted stream fragments for same-lane compatibility."""
    if a["z_max"] < b["z_min"]:
        longitudinal_gap = float(b["z_min"] - a["z_max"])
        z_start, z_end = a["z_max"], b["z_min"]
    elif b["z_max"] < a["z_min"]:
        longitudinal_gap = float(a["z_min"] - b["z_max"])
        z_start, z_end = b["z_max"], a["z_min"]
    else:
        longitudinal_gap = 0.0
        z_start = max(a["z_min"], b["z_min"])
        z_end = min(a["z_max"], b["z_max"])

    if longitudinal_gap > cfg.max_longitudinal_gap:
        return {
            "compatible": False,
            "longitudinal_gap": longitudinal_gap,
            "max_lateral_disagreement": np.inf,
            "median_lateral_disagreement": np.inf,
            "max_tangent_diff_deg": np.inf,
            "score": np.inf,
        }

    if abs(z_end - z_start) < 1e-9:
        z = np.array([z_start], dtype=np.float64)
    else:
        z = np.linspace(z_start, z_end, max(2, cfg.sample_count))

    intercept_a, slope_a = _fit_linear_merge_proxy(graph, a)
    intercept_b, slope_b = _fit_linear_merge_proxy(graph, b)

    x_a = intercept_a + slope_a * z
    x_b = intercept_b + slope_b * z
    lateral = np.abs(x_a - x_b)

    tangent_a = math.atan(slope_a)
    tangent_b = math.atan(slope_b)
    tangent_diff = abs(tangent_a - tangent_b)
    tangent_diff = min(tangent_diff, math.pi - tangent_diff)

    max_lateral = float(np.max(lateral))
    median_lateral = float(np.median(lateral))
    max_tangent_deg = float(math.degrees(tangent_diff))

    compatible = (
        max_lateral <= cfg.max_lateral_disagreement
        and max_tangent_deg <= cfg.max_tangent_diff_deg
    )

    score = (
        longitudinal_gap / max(cfg.max_longitudinal_gap, 1e-6)
        + max_lateral / max(cfg.max_lateral_disagreement, 1e-6)
        + max_tangent_deg / max(cfg.max_tangent_diff_deg, 1e-6)
    )

    return {
        "compatible": bool(compatible),
        "longitudinal_gap": longitudinal_gap,
        "max_lateral_disagreement": max_lateral,
        "median_lateral_disagreement": median_lateral,
        "max_tangent_diff_deg": max_tangent_deg,
        "score": float(score),
    }

def merge_compatible_lane_streams(
    graph,
    streams,
    lane_fits=None,
    merge_cfg=None,
    fit_cfg=None,
):
    """
    Merge fragmented fitted streams that appear to describe the same lane.

    Merging is deliberately iterative. After each accepted merge the original
    graph observations are combined and RANSAC is run again. This prevents us
    from merely averaging two polynomial coefficient sets and forces the
    merged lane model to be supported by the underlying vehicle observations.

    Returns
    -------
    merged_streams : list[list[int]]
        Node groups after consolidation.
    merged_fits : list[dict]
        Re-fitted lane models with distinct-track support metadata.
    merge_events : list[dict]
        Diagnostics for every accepted merge.
    """
    if merge_cfg is None:
        merge_cfg = LaneMergeConfig()
    if fit_cfg is None:
        fit_cfg = LaneFitConfig()
    if lane_fits is None:
        lane_fits = fit_lane_streams(graph, streams, cfg=fit_cfg)

    # Only streams with a valid fit can participate in geometric merging.
    groups = []
    for fit in lane_fits:
        source_ids = list(fit.get("source_stream_ids", [fit["stream_id"]]))
        nodes = sorted(
            {
                node
                for source_id in source_ids
                for node in streams[source_id]
            }
        )
        current_fit = dict(fit)
        current_fit["source_stream_ids"] = sorted(source_ids)
        groups.append(
            {
                "nodes": nodes,
                "fit": current_fit,
                "source_stream_ids": sorted(source_ids),
            }
        )

    merge_events = []

    for _ in range(merge_cfg.max_iterations):
        candidates = []
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                metrics = _lane_fit_comparison(
                    graph, groups[i]["fit"], groups[j]["fit"], merge_cfg
                )
                if metrics["compatible"]:
                    candidates.append((metrics["score"], i, j, metrics))

        if not candidates:
            break

        candidates.sort(key=lambda item: item[0])
        merged_this_iteration = False

        for _, i, j, metrics in candidates:
            group_a = groups[i]
            group_b = groups[j]
            merged_nodes = sorted(set(group_a["nodes"]) | set(group_b["nodes"]))
            merged_fit = fit_lane_stream(graph, merged_nodes, cfg=fit_cfg)
            if merged_fit is None:
                continue

            # A merge is only useful if the refitted model actually retains
            # evidence from both original groups rather than treating one
            # whole fragment as RANSAC outliers.
            inlier_set = set(merged_fit["inliers"])
            if not (inlier_set & set(group_a["nodes"])):
                continue
            if not (inlier_set & set(group_b["nodes"])):
                continue

            source_ids = sorted(
                set(group_a["source_stream_ids"])
                | set(group_b["source_stream_ids"])
            )
            merged_fit["source_stream_ids"] = source_ids

            merge_events.append(
                {
                    "source_stream_ids_a": list(group_a["source_stream_ids"]),
                    "source_stream_ids_b": list(group_b["source_stream_ids"]),
                    "merged_source_stream_ids": source_ids,
                    **metrics,
                    "refit_rmse": merged_fit["rmse"],
                    "refit_inliers": len(merged_fit["inliers"]),
                    "refit_num_tracks": merged_fit["num_tracks"],
                }
            )

            new_group = {
                "nodes": merged_nodes,
                "fit": merged_fit,
                "source_stream_ids": source_ids,
            }

            groups = [
                group
                for index, group in enumerate(groups)
                if index not in (i, j)
            ]
            groups.append(new_group)
            merged_this_iteration = True
            break

        if not merged_this_iteration:
            break

    # Give consolidated lanes fresh compact IDs. Sorting by the fitted lateral
    # position at each fit's midpoint makes the numbering reasonably stable.
    def lateral_key(group):
        fit = group["fit"]
        z_mid = 0.5 * (fit["z_min"] + fit["z_max"])
        return float(evaluate_lane_polynomial(fit["coefficients"], z_mid))

    groups.sort(key=lateral_key)

    merged_streams = []
    merged_fits = []
    for stream_id, group in enumerate(groups):
        fit = group["fit"]
        fit["stream_id"] = stream_id
        fit["source_stream_ids"] = list(group["source_stream_ids"])
        merged_streams.append(sorted(group["nodes"]))
        merged_fits.append(fit)

    return merged_streams, merged_fits, merge_events