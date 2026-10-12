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
                           dilation_kernel=5, ppm=20.0):
    """
    Evaluates GT and Pred masks with structural dilation over the full image area.
    """
    kernel = np.ones((dilation_kernel, dilation_kernel), np.uint8)
    gt_paths = glob.glob(os.path.join(gt_dir, '*_gt.png'))
    
    if not gt_paths:
        print(f"No ground truth images found in {gt_dir}")
        return
        
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    total_tp, total_fp, total_fn = 0, 0, 0
    
    for gt_path in gt_paths:
        filename = os.path.basename(gt_path)
        pred_filename = filename.replace('_gt.png', '_pred.png')
        json_filename = filename.replace('_gt.png', '_vehicles.json')
        
        pred_path = os.path.join(pred_dir, pred_filename)
        json_path = os.path.join(vehicles_dir, json_filename)
        
        if not os.path.exists(pred_path):
            print(f"Warning: Missing prediction mask for {filename}")
            continue
            
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        
        _, gt_bin = cv2.threshold(gt_mask, 127, 255, cv2.THRESH_BINARY)
        _, pred_bin = cv2.threshold(pred_mask, 127, 255, cv2.THRESH_BINARY)
        
        gt_dilated = cv2.dilate(gt_bin, kernel, iterations=1)
        pred_dilated = cv2.dilate(pred_bin, kernel, iterations=1)
        
        # Evaluate on the full mask
        gt_bool = gt_dilated > 0
        pred_bool = pred_dilated > 0
        
        tp_mask = gt_bool & pred_bool
        fp_mask = pred_bool & ~gt_bool
        fn_mask = gt_bool & ~pred_bool
        
        total_tp += np.sum(tp_mask)
        total_fp += np.sum(fp_mask)
        total_fn += np.sum(fn_mask)
        
        if output_dir:
            vis_img = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=np.uint8)
            
            vis_img[fn_mask] = [255, 0, 0]    # Blue: False Negatives
            vis_img[tp_mask] = [0, 255, 0]    # Green: True Positives
            vis_img[fp_mask] = [0, 0, 255]    # Red: False Positives
            
            # Draw the vehicle bounding boxes in yellow
            draw_vehicles_on_vis(vis_img, json_path, ppm=ppm)
            
            vis_filename = filename.replace('_gt.png', '_vis.png')
            cv2.imwrite(os.path.join(output_dir, vis_filename), vis_img)

    epsilon = 1e-7
    precision = total_tp / (total_tp + total_fp + epsilon)
    recall = total_tp / (total_tp + total_fn + epsilon)
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
    iou = total_tp / (total_tp + total_fp + total_fn + epsilon)
    
    print(f"Evaluation Results (Dilation: {dilation_kernel}x{dilation_kernel} | Full Mask)")
    print("-" * 65)
    print(f"Total Evaluated : {len(gt_paths)} frames")
    print(f"Precision       : {precision:.4f}")
    print(f"Recall          : {recall:.4f}")
    print(f"F1 Score        : {f1_score:.4f}")
    print(f"IoU             : {iou:.4f}")
    if output_dir:
        print(f"Visualisations  : Saved to '{output_dir}'")
    
    return precision, recall, f1_score, iou

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate segmentation masks on the full image.")
    parser.add_argument("--gt", type=str, required=True, help="Directory containing ground truth *_gt.png masks.")
    parser.add_argument("--pred", type=str, required=True, help="Directory containing prediction *_pred.png masks.")
    parser.add_argument("--vehicles", type=str, required=True, help="Directory containing *_vehicles.json files.")
    parser.add_argument("--vis", type=str, default=None, help="Directory to save output visualisation RGB masks.")
    parser.add_argument("--kernel", type=int, default=5, help="Kernel size for structural dilation (default: 5).")
    parser.add_argument("--ppm", type=float, default=20.0, help="Pixels per meter used for BEV mapping (default: 20.0).")
    
    args = parser.parse_args()
    
    evaluate_and_visualize(
        gt_dir=args.gt, 
        pred_dir=args.pred, 
        vehicles_dir=args.vehicles,
        output_dir=args.vis, 
        dilation_kernel=args.kernel,
        ppm=args.ppm
    )