import os
import re
import shutil
import subprocess
import datetime

def main():
    # Configuration based on your commands
    MASK_DIR = "./ground_truth_mask/"
    SCENES_PKL_BASE_DIR = "/home/gdtrinh/nuscenes/scenes-pkl/"
    PRED_DIR = "./pred-dir"
    VIZ_DIR = "./eval-viz"
    SUMMARY_FILE = "evaluation_summary.txt"
    MIN_FRAMES = 15

    # Initialize the summary file with a timestamp
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(SUMMARY_FILE, 'w') as f:
        f.write(f"Batch Evaluation Summary ({current_time})\n")
        f.write("="*60 + "\n\n")

    # 1. Find all scene files and extract unique scene numbers
    if not os.path.exists(MASK_DIR):
        print(f"Error: Mask directory '{MASK_DIR}' not found.")
        return

    mask_files = [f for f in os.listdir(MASK_DIR) if f.endswith('.mp4')]
    
    scene_ids = set()
    for f in mask_files:
        # Match pattern like: scene-1046_frames-00-39.mp4
        match = re.search(r'(scene-\d+)_frames-(\d+)-(\d+)\.mp4', f)
        if match:
            scene_id = match.group(1)
            start_frame = int(match.group(2))
            end_frame = int(match.group(3))
            frame_count = (end_frame - start_frame) + 1
            
            if frame_count >= MIN_FRAMES:
                scene_ids.add(scene_id)
            else:
                print(f"Skipping {scene_id}: Contains only {frame_count} frames (minimum {MIN_FRAMES} required).")
        else:
            # Fallback if filename doesn't have the frames format, but still matches 'scene-XXXX'
            match_fallback = re.search(r'(scene-\d+)', f)
            if match_fallback:
                print(f"Warning: Could not parse frame count from {f}. Including {match_fallback.group(1)} by default.")
                scene_ids.add(match_fallback.group(1))
            
    scene_ids = sorted(list(scene_ids))
    
    if not scene_ids:
        print(f"No valid scenes with >={MIN_FRAMES} frames found in {MASK_DIR}")
        return

    print(f"\nFound {len(scene_ids)} valid scenes to evaluate.")

    recalls = []

    # 2. Iterate through each scene
    for scene_id in scene_ids:
        print(f"\n" + "="*40)
        print(f"--- Processing {scene_id} ---")
        print("="*40)
        
        # Ensure fresh temporary directories
        for d in [PRED_DIR, VIZ_DIR]:
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)
        
        scene_pkl_path = os.path.join(SCENES_PKL_BASE_DIR, scene_id)
        
        # Command 1: Run Generation
        cmd1 = [
            "python", "run_live_demo_profiled_bevonly.py",
            "--input", scene_pkl_path,
            "--output", "gt_bev.mp4",
            "--mask-dir", MASK_DIR,
            "--pred-dir", PRED_DIR,
            "--vehicles-dir", PRED_DIR,
            "--no-display"
        ]
        
        print(f"[{scene_id}] Generating masks and vehicle blobs...")
        res1 = subprocess.run(cmd1)
        if res1.returncode != 0:
            print(f"[{scene_id}] Error: Generation script failed. Skipping to next scene.")
            continue
            
        # Command 2: Run Evaluation
        cmd2 = [
            "python", "eval-script.py",
            "--gt", PRED_DIR,
            "--pred", PRED_DIR,
            "--vis", VIZ_DIR,
            "--vehicles", PRED_DIR,
            "--kernel", "25",
            "--ppm", "33",
            "--padding", "1"
        ]
        
        print(f"[{scene_id}] Running evaluation...")
        res2 = subprocess.run(cmd2, capture_output=True, text=True)
        
        if res2.returncode != 0:
            print(f"[{scene_id}] Error: Evaluation script failed. Skipping to next scene.")
            print(res2.stderr)
            continue
            
        # 3. Parse the output metrics using regex
        output = res2.stdout
        
        p_match = re.search(r"Precision\s*:\s*([\d.]+)", output)
        r_match = re.search(r"Recall\s*:\s*([\d.]+)", output)
        f1_match = re.search(r"F1 Score\s*:\s*([\d.]+)", output)
        iou_match = re.search(r"IoU\s*:\s*([\d.]+)", output)
        
        if p_match and r_match and f1_match and iou_match:
            p = float(p_match.group(1))
            r = float(r_match.group(1))
            f1 = float(f1_match.group(1))
            iou = float(iou_match.group(1))
            
            recalls.append(r)
            
            result_text = f"{scene_id:<12} | Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f} | IoU: {iou:.4f}"
            print(f"[{scene_id}] Success: {result_text}")
            
            # Append result to summary file immediately
            with open(SUMMARY_FILE, 'a') as f:
                f.write(result_text + "\n")
        else:
            print(f"[{scene_id}] Error: Could not parse metrics. Script output was:\n{output}")
            
        # 4. Clean up temp directories for the next iteration
        print(f"[{scene_id}] Cleaning up temporary folders...")
        shutil.rmtree(PRED_DIR, ignore_errors=True)
        shutil.rmtree(VIZ_DIR, ignore_errors=True)

    # 5. Calculate and print final metrics
    if recalls:
        avg_recall = sum(recalls) / len(recalls)
        summary_msg = f"\n=== BATCH COMPLETE ===\nTotal scenes evaluated: {len(recalls)}\nAverage Recall      : {avg_recall:.4f}"
        
        print(summary_msg)
        with open(SUMMARY_FILE, 'a') as f:
            f.write("\n" + "-"*60)
            f.write(summary_msg + "\n")
    else:
        print("\nNo scenes were successfully evaluated.")

if __name__ == "__main__":
    main()