import numpy as np

from lane_inference.configs import (
    LaneBoundaryConfig
)

from lane_inference.lane_fitting import (
    evaluate_lane_polynomial,
    lane_polynomial_slope,
)

def _average_polynomials(a, b):
    degree = max(len(a), len(b))
    a = np.pad(np.asarray(a, dtype=np.float64), (0, degree - len(a)))
    b = np.pad(np.asarray(b, dtype=np.float64), (0, degree - len(b)))
    return 0.5 * (a + b)


def infer_lane_boundaries(lane_fits, cfg=None):

    if cfg is None:
        cfg = LaneBoundaryConfig()

    debug_print = print if cfg.verbose else lambda *args, **kwargs: None

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

            debug_print(
                f"\nS{a['stream_id']} vs S{b['stream_id']}: "
                f"z=[{z_min:.1f}, {z_max:.1f}], "
                f"overlap={overlap:.2f} m"
            )

            if overlap < cfg.min_overlap:
                debug_print(
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
                debug_print("  -> rejected: median lateral separation is effectively zero")
                continue

            sign_consistency = float(
                np.mean(np.sign(delta) == np.sign(median_delta))
            )

            debug_print(
                f"  median x-gap={median_x_gap:.2f} m | "
                f"left/right consistency={sign_consistency:.2%}"
            )

            if sign_consistency < 0.9:
                debug_print(
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

            debug_print(
                f"  normal lane width={median_width:.2f} m | "
                f"std={width_std:.2f} m"
            )

            if not (
                cfg.min_lane_width <= median_width <= cfg.max_lane_width
            ):
                debug_print(
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

            debug_print(
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

    debug_print(
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
        debug_print("Single-stream provisional boundaries: disabled")
        debug_print(f"\nTotal inferred lane boundaries: {len(boundaries)}")
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
            debug_print(
                f"S{fit['stream_id']}: single-stream fallback suppressed "
                "because paired-stream boundaries exist"
            )
            continue

        if inlier_count < cfg.single_stream_min_inliers:
            debug_print(
                f"S{fit['stream_id']}: not enough inliers for "
                f"single-stream boundaries ({inlier_count})"
            )
            continue

        if span < cfg.single_stream_min_span:
            debug_print(
                f"S{fit['stream_id']}: fitted span too short "
                f"({span:.2f} m)"
            )
            continue

        if (
            fit.get("has_track_info")
            and fit.get("num_tracks", 0) < cfg.single_stream_min_tracks
        ):
            debug_print(
                f"S{fit['stream_id']}: insufficient distinct tracks "
                f"({fit.get('num_tracks', 0)})"
            )
            continue

        if rmse > cfg.single_stream_max_rmse:
            debug_print(
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

        debug_print(
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

    debug_print(f"\nTotal inferred lane boundaries: {len(boundaries)}")
    return boundaries

