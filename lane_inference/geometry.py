import numpy as np

'''
Just some helper functions to do math u know how it is

from geometry import (
    axial_angle_diff,
    heading_from_yaw,
    pair_metrics,
)
'''

def axial_angle_diff(a, b):
    diff = np.mod(np.abs(a - b), np.pi)
    return np.minimum(diff, np.pi - diff)


def heading_from_yaw(yaw):
    return np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float64)


def pair_metrics(p_i, yaw_i, p_j, yaw_j):
    p_i = np.asarray(p_i, dtype=np.float64)
    p_j = np.asarray(p_j, dtype=np.float64)
    delta = p_j - p_i

    # Yaw represents an axis rather than a directed vector for lane
    # compatibility: yaw and yaw + pi describe the same traffic axis.
    # Compute the pair's mean axial heading using double-angle averaging.
    mean_yaw = 0.5 * np.arctan2(
        np.sin(2.0 * yaw_i) + np.sin(2.0 * yaw_j),
        np.cos(2.0 * yaw_i) + np.cos(2.0 * yaw_j),
    )

    heading = heading_from_yaw(mean_yaw)
    normal = np.array(
        [-heading[1], heading[0]],
        dtype=np.float64,
    )

    # Measure both distances relative to the shared traffic direction.
    cross_track = abs(np.dot(normal, delta))
    along_track = abs(np.dot(heading, delta))

    # Heading disagreement remains an independent compatibility gate.
    yaw_diff = axial_angle_diff(yaw_i, yaw_j)

    return {
        "cross_track": float(cross_track),
        "yaw_diff": float(yaw_diff),
        "along_track": float(along_track),
    }