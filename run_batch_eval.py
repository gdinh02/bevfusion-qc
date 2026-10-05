import os
import re
import shutil
import subprocess
import datetime

def main():
    MASK_DIR = "./ground_truth_mask/"
    SCENES_PKL_BASE_DIR = "/home/gdtrinh/nuscenes/scenes-pkl/"
    PRED_DIR = "./pred-dir"
    VIZ_DIR = "./eval-viz"
    SUMMARY_FILE = "evaluation_summary.txt"
    MIN_FRAMES = 15

    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(SUMMARY_FILE, 'w') as f:
        f.write(f"Batch Evaluation Summary ({current_time})\n")
        f.write("="*95 + "\n\n")

    if not os.path.exists(MASK_DIR):
        print(f"Error: Mask directory '{MASK_DIR}' not found.")
        return

    mask_files = [f for f in os.listdir(MASK_DIR) if f.endswith('.mp4')]
    
    scene_ids = set()
    for f in mask_files:
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
            match_fallback = re.search(r'(scene-\d+)', f)
            if match_fallback:
                scene_ids.add(match_fallback.group(1))
            
    scene_ids = sorted(list(scene_ids))
    
    if not scene_ids:
        print(f"No valid scenes with >={MIN_FRAMES} frames found in {MASK_DIR}")
        return

    print(f"\nFound {len(scene_ids)} valid scenes to evaluate.")
    recalls = []

    for scene_id in scene_ids:
        print(f"\n" + "="*40)
        print(f"--- Processing {scene_id} ---")
        print("="*40)
        
        for d in [PRED_DIR, VIZ_DIR]:
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)
        
        scene_pkl_path = os.path.join(SCENES_PKL_BASE_DIR, scene_id)
        
        cmd1 = [
            "python", "run_live_demo_profiled_gt_bevonly.py",
            "--input", scene_pkl_path,
            "--output", "gt_bev.mp4",
            "--mask-dir", MASK_DIR,
            "--pred-dir", PRED_DIR,
            "--vehicles-dir", PRED_DIR,
            "--no-display"
        ]
        
        print(f"[{scene_id}] Generating masks and vehicle blobs...")
        
        # Stream the output live to the console while capturing it for regex
        process = subprocess.Popen(cmd1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        cmd1_output = []
        for line in process.stdout:
            print(line, end='')
            cmd1_output.append(line)
        process.wait()
        
        if process.returncode != 0:
            print(f"[{scene_id}] Error: Generation script failed. Skipping to next scene.")
            continue
            
        cmd1_text = "".join(cmd1_output)
        
        # Parse Profiling Metrics
        model_work_match = re.search(r"model_work_total\s+avg=\s+([\d.]+)\s*ms", cmd1_text)
        pipeline_match = re.search(r"pipeline_interval\s+avg=\s+([\d.]+)\s*ms", cmd1_text)
        
        model_time = float(model_work_match.group(1)) if model_work_match else None
        pipeline_time = float(pipeline_match.group(1)) if pipeline_match else None
            
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
            print(f"[{scene_id}] Error: Evaluation script failed. Skipping.")
            print(res2.stderr)
            continue
            
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
            
            # Format times for display
            model_str = f"{model_time:.2f}ms" if model_time is not None else "N/A"
            pipe_str = f"{pipeline_time:.2f}ms" if pipeline_time is not None else "N/A"
            
            result_text = (f"{scene_id:<12} | Precision: {p:.4f} | Recall: {r:.4f} | "
                           f"F1: {f1:.4f} | IoU: {iou:.4f} | Model: {model_str:<8} | Pipeline: {pipe_str:<8}")
                           
            print(f"[{scene_id}] Success: {result_text}")
            
            with open(SUMMARY_FILE, 'a') as f:
                f.write(result_text + "\n")
        else:
            print(f"[{scene_id}] Error: Could not parse metrics.")
            
        print(f"[{scene_id}] Cleaning up temporary folders...")
        shutil.rmtree(PRED_DIR, ignore_errors=True)
        shutil.rmtree(VIZ_DIR, ignore_errors=True)

    if recalls:
        avg_recall = sum(recalls) / len(recalls)
        summary_msg = f"\n=== BATCH COMPLETE ===\nTotal scenes evaluated: {len(recalls)}\nAverage Recall      : {avg_recall:.4f}"
        print(summary_msg)
        with open(SUMMARY_FILE, 'a') as f:
            f.write("\n" + "-"*95)
            f.write(summary_msg + "\n")
    else:
        print("\nNo scenes were successfully evaluated.")

if __name__ == "__main__":
    main()