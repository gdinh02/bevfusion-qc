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

    h_i = heading_from_yaw(yaw_i)
    h_j = heading_from_yaw(yaw_j)
    n_i = np.array([-h_i[1], h_i[0]])
    n_j = np.array([-h_j[1], h_j[0]])

    cross_track = 0.5 * (
        abs(np.dot(n_i, delta)) + abs(np.dot(n_j, -delta))
    )
    yaw_diff = axial_angle_diff(yaw_i, yaw_j)
    along_track = 0.5 * (
        abs(np.dot(h_i, delta)) + abs(np.dot(h_j, -delta))
    )

    return {
        "cross_track": float(cross_track),
        "yaw_diff": float(yaw_diff),
        "along_track": float(along_track),
    }

