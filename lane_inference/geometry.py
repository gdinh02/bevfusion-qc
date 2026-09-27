import numpy as np


def signed_heading_angle_diff(a, b):
    """
    Signed directed angular difference a - b in [-pi, pi).

    Unlike axial logic, headings separated by pi remain opposite:
        90 deg vs 270 deg -> 180 deg
    """
    return (
        np.asarray(a) - np.asarray(b) + np.pi
    ) % (2.0 * np.pi) - np.pi


def heading_angle_diff(a, b):
    """
    Smallest directed-heading separation in [0, pi].
    """
    return np.abs(signed_heading_angle_diff(a, b))


def axial_angle_diff(a, b):
    """
    Directionless orientation difference.

    Keep this ONLY for geometry which fundamentally cannot distinguish
    yaw from yaw + pi, such as an x(z) polynomial tangent.
    """
    diff = np.mod(np.abs(a - b), np.pi)
    return np.minimum(diff, np.pi - diff)


def heading_from_yaw(yaw):
    return np.array(
        [np.cos(yaw), np.sin(yaw)],
        dtype=np.float64,
    )


def pair_metrics(p_i, yaw_i, p_j, yaw_j):
    p_i = np.asarray(p_i, dtype=np.float64)
    p_j = np.asarray(p_j, dtype=np.float64)
    delta = p_j - p_i

    # Preserve direction. Find the shortest DIRECTED angular displacement
    # from yaw_i to yaw_j, then take its midpoint.
    yaw_delta = float(
        signed_heading_angle_diff(yaw_j, yaw_i)
    )

    mean_yaw = float(
        (yaw_i + 0.5 * yaw_delta + np.pi)
        % (2.0 * np.pi)
        - np.pi
    )

    heading = heading_from_yaw(mean_yaw)
    normal = np.array(
        [-heading[1], heading[0]],
        dtype=np.float64,
    )

    cross_track = abs(np.dot(normal, delta))
    along_track = abs(np.dot(heading, delta))

    # IMPORTANT: directed heading difference, not axial difference.
    yaw_diff = float(
        heading_angle_diff(yaw_i, yaw_j)
    )

    return {
        "cross_track": float(cross_track),
        "yaw_diff": yaw_diff,
        "along_track": float(along_track),
        "mean_heading_yaw": mean_yaw,
    }