#!/usr/bin/env python3
"""Select dense nuScenes training scenes using ground-truth metadata only.

Selection rule (defaults):
- Official nuScenes v1.0-trainval ``train`` split only.
- Target BEVFusion labels/classes:
    0 -> vehicle.car
    1 -> vehicle.truck
    2 -> vehicle.construction
- A keyframe qualifies when it contains at least 5 target annotations.
- A scene qualifies when at least 50% of its keyframes qualify.

This script reads nuScenes JSON metadata/annotations only. It does NOT copy,
export, or read camera images, LiDAR point clouds, radar, or sweeps.

Example:
    python tools/select_nuscenes_eval_scenes.py --root Z:/dataset/nuscenes --output-json selected_nuscenes_train_scenes.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterator


DATASET_VERSION = "v1.0-trainval"
SPLIT_NAME = "train"

# These are the classes used by the repo's camera-only BEVFusion path.
TARGET_LABEL_TO_CATEGORY = {
    0: "vehicle.car",
    1: "vehicle.truck",
    2: "vehicle.construction",
}
CATEGORY_TO_TARGET_LABEL = {
    category: label for label, category in TARGET_LABEL_TO_CATEGORY.items()
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="nuScenes dataset root containing v1.0-trainval/",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("selected_nuscenes_train_scenes.json"),
        help="JSON manifest containing selected scenes and per-keyframe counts",
    )
    parser.add_argument(
        "--output-txt",
        type=Path,
        default=None,
        help="Optional text manifest; defaults next to --output-json",
    )
    parser.add_argument(
        "--min-vehicles",
        type=int,
        default=5,
        help="Minimum number of target GT vehicles required in a keyframe",
    )
    parser.add_argument(
        "--min-keyframe-fraction",
        type=float,
        default=0.50,
        help="Minimum fraction of a scene's keyframes that must qualify",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every selected scene while scanning",
    )
    args = parser.parse_args(argv)

    if args.min_vehicles < 1:
        parser.error("--min-vehicles must be >= 1")
    if not 0.0 < args.min_keyframe_fraction <= 1.0:
        parser.error("--min-keyframe-fraction must be in (0, 1]")

    if args.output_txt is None:
        if args.output_json.suffix:
            args.output_txt = args.output_json.with_suffix(".txt")
        else:
            args.output_txt = Path(str(args.output_json) + ".txt")

    return args


def iter_scene_samples(nusc: Any, scene: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield all annotated nuScenes samples/keyframes in chronological order."""
    token = scene["first_sample_token"]
    seen: set[str] = set()

    while token:
        if token in seen:
            raise ValueError(f"Cycle detected in scene {scene['name']}: {token}")
        seen.add(token)

        sample = nusc.get("sample", token)
        if sample["scene_token"] != scene["token"]:
            raise ValueError(
                f"Sample {token} does not belong to scene {scene['name']}"
            )

        yield sample
        token = sample["next"]


def annotation_category_name(nusc: Any, annotation: dict[str, Any]) -> str:
    """Return a sample annotation's nuScenes category name.

    Recent nuScenes-devkit versions add ``category_name`` to annotation records.
    The fallback derives it from the raw instance/category tables so the selector
    does not depend on that convenience field being present.
    """
    category_name = annotation.get("category_name")
    if category_name is not None:
        return str(category_name)

    instance = nusc.get("instance", annotation["instance_token"])
    category = nusc.get("category", instance["category_token"])
    return str(category["name"])


def count_target_annotations(
    nusc: Any,
    sample: dict[str, Any],
) -> tuple[int, dict[int, int]]:
    """Count GT objects corresponding to BEVFusion labels 0, 1 and 2."""
    class_counts = {label: 0 for label in TARGET_LABEL_TO_CATEGORY}

    for annotation_token in sample["anns"]:
        annotation = nusc.get("sample_annotation", annotation_token)
        category_name = annotation_category_name(nusc, annotation)
        label = CATEGORY_TO_TARGET_LABEL.get(category_name)
        if label is not None:
            class_counts[label] += 1

    return sum(class_counts.values()), class_counts


def required_qualifying_keyframes(total: int, fraction: float) -> int:
    """Return the integer number of qualifying frames required by the fraction."""
    if total <= 0:
        raise ValueError("A scene must contain at least one keyframe")
    return max(1, math.ceil(total * fraction - 1e-12))


def analyse_scene(
    nusc: Any,
    scene: dict[str, Any],
    min_vehicles: int,
    min_keyframe_fraction: float,
) -> tuple[bool, dict[str, Any]]:
    keyframes: list[dict[str, Any]] = []
    qualifying_keyframes = 0
    total_target_annotations = 0
    scene_class_counts = {label: 0 for label in TARGET_LABEL_TO_CATEGORY}
    max_target_vehicle_count = 0

    for frame_index, sample in enumerate(iter_scene_samples(nusc, scene)):
        target_count, class_counts = count_target_annotations(nusc, sample)
        qualifies = target_count >= min_vehicles

        if qualifies:
            qualifying_keyframes += 1
        total_target_annotations += target_count
        max_target_vehicle_count = max(max_target_vehicle_count, target_count)

        for label, count in class_counts.items():
            scene_class_counts[label] += count

        keyframes.append(
            {
                "frame_index": frame_index,
                "sample_token": sample["token"],
                "timestamp_us": int(sample["timestamp"]),
                "target_vehicle_count": target_count,
                "class_counts": {
                    str(label): class_counts[label]
                    for label in sorted(class_counts)
                },
                "qualifies": qualifies,
            }
        )

    num_keyframes = len(keyframes)
    if num_keyframes != int(scene["nbr_samples"]):
        raise ValueError(
            f"Scene {scene['name']} reports {scene['nbr_samples']} samples but "
            f"the sample chain contains {num_keyframes}"
        )

    required = required_qualifying_keyframes(
        num_keyframes,
        min_keyframe_fraction,
    )
    selected = qualifying_keyframes >= required

    record = {
        "scene_name": scene["name"],
        "scene_token": scene["token"],
        "num_keyframes": num_keyframes,
        "required_qualifying_keyframes": required,
        "qualifying_keyframes": qualifying_keyframes,
        "qualifying_fraction": qualifying_keyframes / num_keyframes,
        "max_target_vehicle_count": max_target_vehicle_count,
        "mean_target_vehicle_count": total_target_annotations / num_keyframes,
        "class_annotation_counts_across_keyframes": {
            str(label): scene_class_counts[label]
            for label in sorted(scene_class_counts)
        },
        "keyframes": keyframes,
    }
    return selected, record


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".tmp")
    with pending.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    pending.replace(path)


def write_text_atomic(path: Path, scene_names: list[str]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".tmp")
    with pending.open("w", encoding="utf-8", newline="\n") as handle:
        for scene_name in scene_names:
            handle.write(scene_name + "\n")
    pending.replace(path)


def select_scenes(
    nusc: Any,
    train_scene_names: list[str],
    min_vehicles: int,
    min_keyframe_fraction: float,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Analyse the official training split and return every qualifying scene."""
    scene_by_name = {scene["name"]: scene for scene in nusc.scene}
    missing = sorted(set(train_scene_names) - set(scene_by_name))
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"{len(missing)} official training scenes are missing from the loaded "
            f"dataset (first: {preview}). Is this the full {DATASET_VERSION} dataset?"
        )

    selected: list[dict[str, Any]] = []

    for index, scene_name in enumerate(train_scene_names, start=1):
        scene = scene_by_name[scene_name]
        is_selected, record = analyse_scene(
            nusc,
            scene,
            min_vehicles=min_vehicles,
            min_keyframe_fraction=min_keyframe_fraction,
        )

        if is_selected:
            selected.append(record)
            if verbose:
                print(
                    f"SELECT {scene_name}: "
                    f"{record['qualifying_keyframes']}/{record['num_keyframes']} "
                    f"keyframes >= {min_vehicles} target vehicles"
                )

        if index % 50 == 0:
            print(
                f"Scanned {index}/{len(train_scene_names)} training scenes; "
                f"selected {len(selected)}",
                flush=True,
            )

    selected.sort(key=lambda record: record["scene_name"])
    return selected


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.expanduser().resolve()

    metadata_dir = root / DATASET_VERSION
    if not metadata_dir.is_dir():
        raise FileNotFoundError(
            f"nuScenes metadata directory not found: {metadata_dir}"
        )

    # Imported here so --help remains usable even if nuscenes-devkit is absent.
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.splits import create_splits_scenes

    split_scenes = create_splits_scenes(verbose=False)
    train_scene_names = list(split_scenes[SPLIT_NAME])

    nusc = NuScenes(
        version=DATASET_VERSION,
        dataroot=str(root),
        verbose=False,
    )

    selected = select_scenes(
        nusc,
        train_scene_names=train_scene_names,
        min_vehicles=args.min_vehicles,
        min_keyframe_fraction=args.min_keyframe_fraction,
        verbose=args.verbose,
    )

    manifest = {
        "schema": "bevfusion-qc.nuscenes-scene-selection",
        "schema_version": 1,
        "dataset_version": DATASET_VERSION,
        "split": SPLIT_NAME,
        "target_classes": {
            str(label): category
            for label, category in sorted(TARGET_LABEL_TO_CATEGORY.items())
        },
        "selection_criteria": {
            "min_target_vehicles_per_keyframe": args.min_vehicles,
            "min_qualifying_keyframe_fraction": args.min_keyframe_fraction,
        },
        "num_scenes_examined": len(train_scene_names),
        "num_scenes_selected": len(selected),
        "selected_scene_names": [record["scene_name"] for record in selected],
        "selected_scenes": selected,
    }

    write_json_atomic(args.output_json, manifest)
    write_text_atomic(
        args.output_txt,
        [record["scene_name"] for record in selected],
    )

    print(
        f"Selected {len(selected)} of {len(train_scene_names)} training scenes."
    )
    print(f"JSON manifest: {args.output_json.expanduser().resolve()}")
    print(f"Text manifest: {args.output_txt.expanduser().resolve()}")
    print("No scene data, images, point clouds, radar, or sweeps were copied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
