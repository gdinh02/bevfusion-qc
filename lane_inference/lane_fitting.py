import math
import numpy as np

from lane_inference.configs import LaneFitConfig, LaneMergeConfig
from lane_inference.geometry import (
    axial_angle_diff,
    heading_angle_diff,
)

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

def _fit_position_and_yaw(
    z,
    x,
    yaw,
    scores,
    degree,
    yaw_constraint_length,
):
    """
    Fit x(z) using both:
        x(z_i)  ~= x_i
        x'(z_i) ~= cot(yaw_i)

    Coefficients are returned in ascending order:
        [a0, a1, a2, ...]
    """
    z = np.asarray(z, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    yaw = np.asarray(yaw, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)

    # Position equations:
    # x = a0 + a1*z + a2*z^2 + ...
    position_A = np.vander(
        z,
        N=degree + 1,
        increasing=True,
    )

    # Derivative equations:
    # dx/dz = a1 + 2*a2*z + 3*a3*z^2 + ...
    derivative_A = np.zeros_like(position_A)
    for power in range(1, degree + 1):
        derivative_A[:, power] = power * z ** (power - 1)

    # Vehicle heading [cos(yaw), sin(yaw)] in x-z coordinates.
    # Therefore dx/dz = cos(yaw) / sin(yaw).
    sin_yaw = np.sin(yaw)
    valid_yaw = np.abs(sin_yaw) > 1e-3

    observed_slope = np.zeros_like(yaw)
    observed_slope[valid_yaw] = (
        np.cos(yaw[valid_yaw]) / sin_yaw[valid_yaw]
    )

    weights = np.sqrt(np.clip(scores, 1e-6, None))

    A_position = position_A * weights[:, None]
    b_position = x * weights

    if np.any(valid_yaw):
        yaw_weights = (
            weights[valid_yaw] * yaw_constraint_length
        )

        A_yaw = (
            derivative_A[valid_yaw]
            * yaw_weights[:, None]
        )
        b_yaw = (
            observed_slope[valid_yaw]
            * yaw_weights
        )

        A = np.vstack((A_position, A_yaw))
        b = np.concatenate((b_position, b_yaw))
    else:
        A = A_position
        b = b_position

    coefficients, _, rank, _ = np.linalg.lstsq(
        A,
        b,
        rcond=None,
    )

    # The combined position + yaw equations must contain enough
    # independent information to determine every polynomial coefficient.
    if rank < degree + 1:
        raise ValueError(
            "Position/yaw observations do not fully constrain polynomial"
        )

    return coefficients

def fit_lane_stream(graph, stream, cfg=None):
    if cfg is None:
        cfg = LaneFitConfig()

    nodes = sorted(
        stream,
        key=lambda node: graph.nodes[node]["z"],
    )

    if len(nodes) < 2:
        return None

    z = np.array(
        [graph.nodes[n]["z"] for n in nodes],
        dtype=np.float64,
    )
    x = np.array(
        [graph.nodes[n]["x"] for n in nodes],
        dtype=np.float64,
    )
    yaw = np.array(
        [graph.nodes[n]["yaw"] for n in nodes],
        dtype=np.float64,
    )

    scores = np.array(
        [
            graph.nodes[n]["score"]
            * graph.nodes[n].get("evidence_weight", 1.0)
            for n in nodes
        ],
        dtype=np.float64,
    )

    degree = min(cfg.degree, len(nodes) - 1)

    # Each oriented observation supplies two constraints:
    #   x(z_i)  = x_i
    #   x'(z_i) = cot(yaw_i)
    #
    # Therefore a quadratic (3 coefficients) can be constrained
    # by two observations at different longitudinal positions.
    sample_size = max(2, math.ceil((degree + 1) / 2))
    trials = (
        1
        if len(nodes) == sample_size
        else cfg.max_trials
    )

    rng = np.random.default_rng(cfg.random_seed)

    best_mask = None
    best_count = -1
    best_rmse = np.inf
    best_yaw_error = np.inf

    max_tangent_error = math.radians(
        cfg.max_tangent_error_deg
    )

    for _ in range(trials):
        sample_idx = (
            np.arange(len(nodes))
            if len(nodes) == sample_size
            else rng.choice(
                len(nodes),
                size=sample_size,
                replace=False,
            )
        )

        sample_z = z[sample_idx]

        # We no longer require degree + 1 distinct z positions because yaw
        # contributes derivative constraints. We still require some longitudinal
        # separation; observations all at exactly the same z cannot determine
        # the curvature of a quadratic.
        if np.ptp(sample_z) < cfg.min_sample_z_span:
            continue

        try:
            coefficients = _fit_position_and_yaw(
                z[sample_idx],
                x[sample_idx],
                yaw[sample_idx],
                scores[sample_idx],
                degree,
                cfg.yaw_constraint_length,
            )
        except (np.linalg.LinAlgError, ValueError):
            continue

        predicted_x = evaluate_lane_polynomial(
            coefficients,
            z,
        )

        predicted_slope = lane_polynomial_slope(
            coefficients,
            z,
        )

        # Tangent direction of x(z):
        # vector = [dx/dz, 1]
        predicted_yaw = np.arctan2(
            np.ones_like(predicted_slope),
            predicted_slope,
        )

        lateral_error = np.abs(x - predicted_x)

        # INTENTIONAL axial comparison:
        # x(z) describes directionless lane geometry, so its tangent cannot
        # distinguish yaw from yaw + pi. Directed traffic identity is enforced
        # by the graph and retained separately in the fit metadata.
        yaw_error = axial_angle_diff(
            predicted_yaw,
            yaw,
        )

        mask = (
            (lateral_error <= cfg.residual_threshold)
            & (yaw_error <= max_tangent_error)
        )

        count = int(mask.sum())

        if count < sample_size:
            continue

        rmse = float(
            np.sqrt(
                np.mean(
                    (x[mask] - predicted_x[mask]) ** 2
                )
            )
        )

        mean_yaw_error = float(
            np.mean(yaw_error[mask])
        )

        if (
            count > best_count
            or (
                count == best_count
                and rmse < best_rmse
            )
            or (
                count == best_count
                and np.isclose(rmse, best_rmse)
                and mean_yaw_error < best_yaw_error
            )
        ):
            best_mask = mask
            best_count = count
            best_rmse = rmse
            best_yaw_error = mean_yaw_error

    if best_mask is None:
        return None

    inlier_z = z[best_mask]
    inlier_x = x[best_mask]
    inlier_yaw = yaw[best_mask]
    inlier_scores = scores[best_mask]

    if np.ptp(inlier_z) < cfg.min_sample_z_span:
        return None

    # Final fit using all RANSAC inliers, with yaw constraints.
    try:
        final_coefficients = _fit_position_and_yaw(
            inlier_z,
            inlier_x,
            inlier_yaw,
            inlier_scores,
            degree,
            cfg.yaw_constraint_length,
        )
    except (np.linalg.LinAlgError, ValueError):
        return None

    prediction = evaluate_lane_polynomial(
        final_coefficients,
        inlier_z,
    )

    final_slope = lane_polynomial_slope(
        final_coefficients,
        inlier_z,
    )

    final_yaw = np.arctan2(
        np.ones_like(final_slope),
        final_slope,
    )

    final_yaw_error = axial_angle_diff(
        final_yaw,
        inlier_yaw,
    )

    inlier_nodes = [
        n
        for n, keep in zip(nodes, best_mask)
        if keep
    ]

    outlier_nodes = [
        n
        for n, keep in zip(nodes, best_mask)
        if not keep
    ]

    has_track_info = any(
        "track_id" in graph.nodes[n]
        for n in inlier_nodes
    )

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

    # Preserve the directed traffic heading separately from the
    # directionless x(z) polynomial geometry.
    heading_weights = np.clip(
        inlier_scores,
        1e-6,
        None,
    )

    heading_x = float(
        np.sum(
            heading_weights
            * np.cos(inlier_yaw)
        )
    )

    heading_z = float(
        np.sum(
            heading_weights
            * np.sin(inlier_yaw)
        )
    )

    heading_norm = float(
        np.hypot(heading_x, heading_z)
    )

    if heading_norm < 1e-8:
        # This would indicate contradictory directed headings within
        # what is supposed to be one traffic stream.
        return None

    mean_heading_yaw = float(
        np.arctan2(
            heading_z,
            heading_x,
        )
    )

    heading_resultant = float(
        heading_norm
        / np.sum(heading_weights)
    )

    return {
        "degree": degree,
        "coefficients": final_coefficients.copy(),
        "inliers": inlier_nodes,
        "outliers": outlier_nodes,
        "rmse": float(
            np.sqrt(
                np.mean(
                    (inlier_x - prediction) ** 2
                )
            )
        ),
        "mean_tangent_error_deg": float(
            np.degrees(np.mean(final_yaw_error))
        ),
        "max_tangent_error_deg": float(
            np.degrees(np.max(final_yaw_error))
        ),
        "z_min": float(inlier_z.min()),
        "z_max": float(inlier_z.max()),
        "num_observations": len(inlier_nodes),
        "track_ids": track_ids,
        "num_tracks": num_tracks,
        "has_track_info": has_track_info,
        "track_support": track_support,
        "mean_heading_yaw": mean_heading_yaw,
        "heading_resultant": heading_resultant,
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
    heading_diff = float(
        heading_angle_diff(
            a["mean_heading_yaw"],
            b["mean_heading_yaw"],
        )
    )

    heading_diff_deg = math.degrees(
        heading_diff
    )

    if heading_diff_deg > cfg.max_heading_diff_deg:
        return {
            "compatible": False,
            "longitudinal_gap": np.inf,
            "max_lateral_disagreement": np.inf,
            "median_lateral_disagreement": np.inf,
            "max_tangent_diff_deg": np.inf,
            "heading_diff_deg": float(
                heading_diff_deg
            ),
            "score": np.inf,
        }

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
            "heading_diff_deg": float(heading_diff_deg),
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
        heading_diff_deg <= cfg.max_heading_diff_deg
        and max_lateral <= cfg.max_lateral_disagreement
        and max_tangent_deg <= cfg.max_tangent_diff_deg
    )

    score = (
        longitudinal_gap
        / max(cfg.max_longitudinal_gap, 1e-6)

        + max_lateral
        / max(cfg.max_lateral_disagreement, 1e-6)

        + max_tangent_deg
        / max(cfg.max_tangent_diff_deg, 1e-6)

        + heading_diff_deg
        / max(cfg.max_heading_diff_deg, 1e-6)
    )

    return {
        "compatible": bool(compatible),
        "longitudinal_gap": longitudinal_gap,
        "max_lateral_disagreement": max_lateral,
        "median_lateral_disagreement": median_lateral,
        "max_tangent_diff_deg": max_tangent_deg,
        "score": float(score),
        "heading_diff_deg": float(heading_diff_deg),
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