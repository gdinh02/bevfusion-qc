import re
import argparse

def calculate_averages(file_path):
    precisions = []
    recalls = []
    f1s = []
    ious = []
    model_times = []
    pipeline_times = []

    # Regex updated to optionally match Model and Pipeline times, handling "N/A" cases
    pattern = re.compile(
        r"Precision:\s*([\d.]+)\s*\|\s*Recall:\s*([\d.]+)\s*\|\s*F1:\s*([\d.]+)\s*\|\s*IoU:\s*([\d.]+)"
        r"(?:\s*\|\s*Model:\s*([0-9.]+|N/A)\s*m?s?)?(?:\s*\|\s*Pipeline:\s*([0-9.]+|N/A)\s*m?s?)?"
    )

    try:
        with open(file_path, 'r') as file:
            for line in file:
                match = pattern.search(line)
                if match:
                    precisions.append(float(match.group(1)))
                    recalls.append(float(match.group(2)))
                    f1s.append(float(match.group(3)))
                    ious.append(float(match.group(4)))
                    
                    # Extract profiling times if they exist and are not "N/A"
                    if match.group(5) and match.group(5) != "N/A":
                        model_times.append(float(match.group(5)))
                    if match.group(6) and match.group(6) != "N/A":
                        pipeline_times.append(float(match.group(6)))
                        
    except FileNotFoundError:
        print(f"Error: Could not find {file_path}")
        return

    count = len(precisions)
    if count == 0:
        print("No metrics found in the file.")
        return

    avg_precision = sum(precisions) / count
    avg_recall = sum(recalls) / count
    avg_f1 = sum(f1s) / count
    avg_iou = sum(ious) / count

    print(f"Total scenes evaluated : {count}")
    print(f"Average Precision      : {avg_precision:.4f}")
    print(f"Average Recall         : {avg_recall:.4f}")
    print(f"Average F1 Score       : {avg_f1:.4f}")
    print(f"Average IoU            : {avg_iou:.4f}")
    
    if model_times:
        avg_model = sum(model_times) / len(model_times)
        print(f"Average Model Time     : {avg_model:.2f} ms")
    if pipeline_times:
        avg_pipeline = sum(pipeline_times) / len(pipeline_times)
        print(f"Average Pipeline Time  : {avg_pipeline:.2f} ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate average metrics from batch evaluation summary.")
    parser.add_argument("--file", type=str, default="evaluation_summary.txt", help="Path to the summary text file.")
    args = parser.parse_args()
    
    calculate_averages(args.file)