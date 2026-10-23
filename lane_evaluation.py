"""Geometry masks and symmetric point-to-boundary distance metrics."""
import math
import cv2
import numpy as np


def thin_mask(mask):
    """Zhang-Suen thinning; no opencv-contrib dependency required."""
    image = np.pad(np.asarray(mask, dtype=bool), 1)
    while True:
        changed = False
        for phase in (0, 1):
            p = [image[:-2, 1:-1], image[:-2, 2:], image[1:-1, 2:],
                 image[2:, 2:], image[2:, 1:-1], image[2:, :-2],
                 image[1:-1, :-2], image[:-2, :-2]]
            neighbours = sum(v.astype(np.uint8) for v in p)
            transitions = sum((~p[i] & p[(i + 1) % 8]).astype(np.uint8)
                              for i in range(8))
            if phase == 0:
                gates = ~(p[0] & p[2] & p[4]) & ~(p[2] & p[4] & p[6])
            else:
                gates = ~(p[0] & p[2] & p[6]) & ~(p[0] & p[4] & p[6])
            remove = image[1:-1, 1:-1] & (neighbours >= 2) & (neighbours <= 6)
            remove &= (transitions == 1) & gates
            if remove.any():
                image[1:-1, 1:-1][remove] = False
                changed = True
        if not changed:
            return image[1:-1, 1:-1]


def rasterize_boundaries(boundaries, shape, ppm):
    """Draw equal-width binary polylines directly from x(z), without styling."""
    from lane_inference.lane_fitting import evaluate_lane_polynomial
    h, w = shape[:2]
    mask = np.zeros((h, w), np.uint8)
    for boundary in boundaries:
        lo = max(float(boundary['z_min']), -(h - 1 - h // 2) / ppm)
        hi = min(float(boundary['z_max']), (h // 2) / ppm)
        if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
            continue
        # Subpixel depth sampling avoids the display renderer's fixed 50 samples.
        z = np.linspace(lo, hi, max(2, int(math.ceil((hi - lo) * ppm * 4)) + 1))
        x = evaluate_lane_polynomial(boundary['coefficients'], z)
        points = np.column_stack((w // 2 + x * ppm, h // 2 - z * ppm))
        if not np.isfinite(points).all():
            raise ValueError('Non-finite predicted boundary geometry')
        # Clip each segment before conversion to OpenCV integer coordinates.
        points = np.rint(np.clip(points, -10000000, 10000000)).astype(np.int32)
        cv2.polylines(mask, [points.reshape(-1, 1, 2)], False, 255, 1, cv2.LINE_8)
    return mask


def match_masks(gt, pred, ppm=33.0, tolerance_m=0.1):
    """Match each skeleton pixel to the nearest opposite skeleton in metres.

    Precision and recall have separate matched-point counts. Duplicate nearby
    predictions are not penalised as separate lane instances by this metric.
    """
    if not math.isfinite(ppm) or ppm <= 0:
        raise ValueError('ppm must be finite and positive')
    if not math.isfinite(tolerance_m) or tolerance_m < 0:
        raise ValueError('tolerance_m must be finite and nonnegative')
    if gt.ndim != 2 or pred.ndim != 2 or gt.shape != pred.shape:
        raise ValueError('GT and prediction masks must have identical 2D shapes')
    gt = thin_mask(gt > 127)
    pred = thin_mask(pred > 127)

    def within(source, target):
        if not target.any():
            return np.zeros_like(source)
        distance = cv2.distanceTransform((~target).astype(np.uint8),
                                         cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        return source & (distance <= tolerance_m * ppm + 1e-6)

    matched_pred, matched_gt = within(pred, gt), within(gt, pred)
    counts = dict(predicted=int(pred.sum()), ground_truth=int(gt.sum()),
                  matched_predicted=int(matched_pred.sum()),
                  matched_ground_truth=int(matched_gt.sum()),
                  intersection=int((gt & pred).sum()), union=int((gt | pred).sum()))
    return counts, gt, pred, matched_gt, matched_pred


def metrics(counts):
    # Empty denominators are reported as zero, with counts printed by the CLI.
    p = counts['matched_predicted'] / counts['predicted'] if counts['predicted'] else 0.0
    r = counts['matched_ground_truth'] / counts['ground_truth'] if counts['ground_truth'] else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    iou = counts['intersection'] / counts['union'] if counts['union'] else 0.0
    return p, r, f1, iou
