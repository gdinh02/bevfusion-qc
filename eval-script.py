import cv2
import numpy as np
import os
import glob
import argparse
import json

def draw_vehicles_on_vis(vis_img, json_path, ppm=20.0):
    """
    Draws oriented vehicle bounding boxes onto the visualization image.
    """
    if not os.path.exists(json_path):
        return
        
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    H, W = vis_img.shape[:2]
    vehicles = data.get("vehicles", [])
    
    for v in vehicles:
        bev = v.get("bev_transformed", {})
        x_right = bev.get("x_right", 0.0)
        y_forward = bev.get("y_forward", 0.0)
        width = bev.get("width", 2.0)
        length = bev.get("length", 4.0)
        yaw = bev.get("yaw_inverted", 0.0)
        
        px = W / 2.0 + x_right * ppm
        py = H / 2.0 - y_forward * ppm
        
        w_px = width * ppm
        l_px = length * ppm
        
        cos_y = np.cos(yaw)
        sin_y = np.sin(yaw)
        
        # Local unrotated corners (centered)
        corners = [
            [-w_px/2, -l_px/2],
            [ w_px/2, -l_px/2],
            [ w_px/2,  l_px/2],
            [-w_px/2,  l_px/2]
        ]
        
        pts = []
        for cx, cy in corners:
            # 2D Rotation and translation
            rx = cx * cos_y - cy * sin_y
            ry = cx * sin_y + cy * cos_y
            pts.append([int(px + rx), int(py + ry)])
            
        pts = np.array(pts, np.int32).reshape((-1, 1, 2))
        
        # Draw the bounding box polygon in yellow (BGR: 0, 255, 255)
        cv2.polylines(vis_img, [pts], isClosed=True, color=(0, 255, 255), thickness=1)
        # Draw a small center dot
        cv2.circle(vis_img, (int(px), int(py)), 2, (0, 255, 255), -1)

def evaluate_and_visualize(gt_dir, pred_dir, vehicles_dir, output_dir=None,
                           tolerance_m=0.3, ppm=33.0):
    from lane_evaluation import match_masks, metrics
    gt_paths = sorted(glob.glob(os.path.join(gt_dir, '*_gt.png')))
    if not gt_paths:
        raise ValueError(f"No ground truth images found in {gt_dir}")
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    totals = dict(predicted=0, ground_truth=0, matched_predicted=0,
                  matched_ground_truth=0, intersection=0, union=0)
    for gt_path in gt_paths:
        filename = os.path.basename(gt_path)
        pred_path = os.path.join(pred_dir, filename.replace('_gt.png', '_pred.png'))
        if not os.path.exists(pred_path):
            raise FileNotFoundError(f"Missing prediction mask: {pred_path}")
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        if gt_mask is None or pred_mask is None:
            raise ValueError(f"Unreadable mask pair: {filename}")
        counts, gt, pred, matched_gt, matched_pred = match_masks(
            gt_mask, pred_mask, ppm, tolerance_m)
        for key in totals:
            totals[key] += counts[key]
        if output_dir:
            vis = np.zeros((*gt.shape, 3), dtype=np.uint8)
            vis[gt & ~matched_gt] = [255, 0, 0]
            vis[matched_gt] = [0, 100, 0]
            vis[pred & ~matched_pred] = [0, 0, 255]
            vis[matched_pred] = [0, 255, 0]
            draw_vehicles_on_vis(vis, os.path.join(vehicles_dir,
                filename.replace('_gt.png', '_vehicles.json')), ppm)
            cv2.imwrite(os.path.join(output_dir, filename.replace('_gt.png', '_vis.png')), vis)
    precision, recall, f1, iou = metrics(totals)
    print(f"Evaluation: skeleton distance matching | tolerance={tolerance_m:g} m | ppm={ppm:g}")
    print(f"Total Evaluated : {len(gt_paths)} frames")
    print(f"Precision       : {precision:.4f}")
    print(f"Recall          : {recall:.4f}")
    print(f"F1 Score        : {f1:.4f}")
    # Exact skeleton IoU is retained for batch-parser compatibility. It does
    # not use the distance tolerance and is not derived from tolerant P/R.
    print(f"IoU             : {iou:.4f}")
    print("IoU definition  : exact skeleton pixel intersection / union")
    print("Counts          : " + json.dumps(totals, sort_keys=True))
    print("Empty-denominator convention: 0; inspect Counts for empty predictions/GT")
    return precision, recall, f1, iou


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate thin lane boundaries by Euclidean distance.')
    parser.add_argument('--gt', required=True)
    parser.add_argument('--pred', required=True)
    parser.add_argument('--vehicles', required=True)
    parser.add_argument('--vis', default=None)
    parser.add_argument('--tolerance-m', type=float, default=0.1)
    parser.add_argument('--ppm', type=float, default=33.0)
    parser.add_argument('--kernel', type=int, default=None,
                        help='Removed: use --tolerance-m instead')
    args = parser.parse_args()
    if args.kernel is not None:
        parser.error('--kernel was removed; use --tolerance-m 0.1')
    evaluate_and_visualize(args.gt, args.pred, args.vehicles, args.vis,
                           tolerance_m=args.tolerance_m, ppm=args.ppm)
