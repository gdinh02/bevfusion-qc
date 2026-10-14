"""Evaluate one selected configuration on GT-vehicle nuScenes scenes.

Uses exactly the repo's mask producer and eval-script.py. The sweep controller
can call evaluate_configuration(...) repeatedly. This stage does not itself
launch a grid sweep.
}

* Accepts any number of non-overlapping `scene-XXXX_frames-START-END.mp4` videos for a scene.
* Frame numbers are zero-based and inclusive.
* Runs lane inference from frame 0 through the last GT segment, retaining the temporal state during unlabeled gaps.
* Writes `*_gt.png`, `*_pred.png`, and `*_vehicles.json` only for annotated frames; filenames use absolute zero-based scene indices.
* Treats `--min-frames` as the minimum total number of annotated frames in a scene, across all segments. Default is 15.
* Retains the original eval-script.py mask dilation and vehicle blob scoring. Default kernel=25, ppm=33, padding=10 m.

Example:

python gt_sweep_eval.py --out-dir Z:/dataset/sweep_results/baseline --mask-dir Z:/dataset/mask --scenes-root Z:/dataset/scenes-pkl
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from statistics import mean

from sweep_config import load_overrides

SCENE_PATTERN = re.compile(r"^(scene-\d+)_frames-(\d+)-(\d+)\.mp4$")
METRICS = {
    "precision": re.compile(r"^Precision\s*:\s*([\d.]+)\s*$", re.M),
    "recall": re.compile(r"^Recall\s*:\s*([\d.]+)\s*$", re.M),
    "f1": re.compile(r"^F1 Score\s*:\s*([\d.]+)\s*$", re.M),
    "iou": re.compile(r"^IoU\s*:\s*([\d.]+)\s*$", re.M),
}


def discover_scenes(mask_dir: Path, min_frames: int = 15) -> list[str]:
    """Discover scenes with enough annotated MP4 frames across all their segments."""
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"GT mask directory missing: {mask_dir}")
    segments: dict[str, list[tuple[int, int]]] = {}
    for video in sorted(mask_dir.glob("*.mp4")):
        match = SCENE_PATTERN.fullmatch(video.name)
        if match is None:
            continue
        scene, first, last = match.groups()
        first, last = int(first), int(last)
        if last < first:
            raise ValueError(f"Invalid mask frame range: {video.name}")
        segments.setdefault(scene, []).append((first, last))
    scenes = []
    for scene, ranges in sorted(segments.items()):
        ranges.sort()
        for prior, following in zip(ranges, ranges[1:]):
            if following[0] <= prior[1]:
                raise ValueError(f"Overlapping mask ranges for {scene}: {prior}, {following}")
        if sum(last - first + 1 for first, last in ranges) >= min_frames:
            scenes.append(scene)
    return scenes


def _run(args: list[str], root: Path) -> str:
    result = subprocess.run(args, cwd=root, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode}): {' '.join(args)}\n{result.stdout[-3500:]}")
    return result.stdout


def _parse_metrics(output: str) -> dict[str, float]:
    values = {}
    for name, pattern in METRICS.items():
        match = pattern.search(output)
        if match is None:
            raise ValueError(f"Metric {name} not found in eval output:\n{output[-1500:]}")
        values[name] = float(match.group(1))
    return values


def evaluate_configuration(overrides: dict, *, repo: Path, mask_dir: Path,
                           scenes_root: Path, out_dir: Path,
                           min_frames: int = 15, kernel: int = 25,
                           ppm: float = 33.0, padding: float = 10.0,
                           scene_ids: list[str] | None = None) -> dict:
    """Return scene-level and macro-mean results; failures remain visible."""
    from sweep_config import validate
    overrides = validate(overrides)
    repo, mask_dir, scenes_root, out_dir = map(
        lambda p: Path(p).expanduser().resolve(),
        (repo, mask_dir, scenes_root, out_dir)
    )
    discovered = discover_scenes(mask_dir, min_frames)
    scenes = sorted(set(scene_ids)) if scene_ids is not None else discovered
    if any(scene not in discovered for scene in scenes):
        raise ValueError("Requested scene lacks an eligible GT mask")
    if not scenes:
        raise ValueError("No eligible scenes")
    if kernel < 1 or ppm <= 0 or padding < 0:
        raise ValueError("Invalid evaluation settings")
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "config.json"
    config_path.write_text(json.dumps(overrides, indent=2, sort_keys=True), encoding="utf-8")
    results = []
    for scene in scenes:
        scene_dir = out_dir / scene
        scene_dir.mkdir(parents=True, exist_ok=True)
        work_dir = scene_dir / "masks"
        work_dir.mkdir(parents=True, exist_ok=True)
        row = {"scene": scene, "status": "failed", "error": ""}
        try:
            scene_path = scenes_root / scene
            if not scene_path.is_dir():
                raise FileNotFoundError(scene_path)
            output_video = scene_dir / "gt_bev.mp4"
            _run([sys.executable, "run_live_demo_profiled_gt_bevonly.py",
                  "--input", str(scene_path),
                #   "--output", str(output_video),
                  "--mask-dir", str(mask_dir), "--pred-dir", str(work_dir),
                  "--vehicles-dir", str(work_dir), "--sweep-config", str(config_path),
                  "--no-display"], repo)
            if not list(work_dir.glob("*_gt.png")) or not list(work_dir.glob("*_pred.png")):
                raise RuntimeError("Mask producer generated no GT or prediction PNGs")
            evaluated = _run([sys.executable, "eval-script.py",
                              "--gt", str(work_dir), "--pred", str(work_dir),
                              "--vehicles", str(work_dir),
                              "--kernel", str(kernel), "--ppm", str(ppm),
                            #   "--padding", str(padding)
                              ], repo)
            row.update(_parse_metrics(evaluated))
            row["status"] = "ok"
        except Exception as exc:
            row["error"] = str(exc)
        results.append(row)
        print(scene, row["status"], row.get("f1", row["error"]), flush=True)
        _save_results(out_dir, results)
    successful = [row for row in results if row["status"] == "ok"]
    summary = {
        "parameters": overrides, "settings": {"kernel": kernel, "ppm": ppm,
                                                  "padding": padding, "min_frames": min_frames},
        "scenes_attempted": len(scenes), "scenes_succeeded": len(successful),
        "macro_metrics": {name: mean(row[name] for row in successful) if successful else None
                          for name in METRICS},
        "per_scene": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _save_results(out_dir: Path, results: list[dict]) -> None:
    with (out_dir / "per_scene.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["scene", "status", "precision", "recall", "f1", "iou", "error"])
        writer.writeheader()
        writer.writerows(results)


def main():
    parser = argparse.ArgumentParser(description="Evaluate one GT lane configuration")
    parser.add_argument("--config", type=Path, help="JSON object of selected dotted parameter overrides")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--mask-dir", type=Path, default=Path("ground_truth_mask"))
    parser.add_argument("--scenes-root", type=Path, default=Path("Z:/dataset/scenes-pkl"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-frames", type=int, default=15)
    parser.add_argument("--kernel", type=int, default=25)
    parser.add_argument("--ppm", type=float, default=33)
    parser.add_argument("--padding", type=float, default=10)
    args = parser.parse_args()
    # Resolve relative mask path against repository root, not invocation folder.
    mask_dir = args.mask_dir if args.mask_dir.is_absolute() else args.repo / args.mask_dir
    result = evaluate_configuration(load_overrides(args.config), repo=args.repo,
        mask_dir=mask_dir, scenes_root=args.scenes_root, out_dir=args.out_dir,
        min_frames=args.min_frames, kernel=args.kernel, ppm=args.ppm, padding=args.padding)
    print(json.dumps({"scenes_succeeded": result["scenes_succeeded"],
                      "macro_metrics": result["macro_metrics"]}, indent=2))
    if result["scenes_succeeded"] != result["scenes_attempted"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
