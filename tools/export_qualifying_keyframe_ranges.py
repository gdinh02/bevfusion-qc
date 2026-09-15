#!/usr/bin/env python3
"""Export contiguous qualifying keyframe ranges from a scene-selection manifest.

This script consumes the JSON written by ``select_nuscenes_eval_scenes.py``.
It does not read or copy any nuScenes sensor data.

For each selected scene, consecutive keyframes whose manifest entry has
``qualifies == true`` are grouped into one CSV row:

    scene,start,end,comment
    38,5,32,
    47,-,31,

Scene names such as ``scene-0038`` are written as integer ``38``. Frame numbers
use the manifest's ``frame_index`` values (0-based by default). When
``--boundary-style hyphen`` is used, a run that starts at the first keyframe has
``start=-`` and a run that reaches the last keyframe has ``end=-``. This matches
the hand-authored CSV style shown in the project notes. Use
``--boundary-style numeric`` to always emit integer start/end values.

Examples:
    python tools/export_qualifying_keyframe_ranges.py  --manifest selected_nuscenes_train_scenes.json  --output selected_keyframe_ranges.csv

    python tools/export_qualifying_keyframe_ranges.py \
        --manifest selected_nuscenes_train_scenes.json \
        --output selected_keyframe_ranges.csv \
        --boundary-style numeric \
        --frame-base 1
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


EXPECTED_SCHEMA = "bevfusion-qc.nuscenes-scene-selection"
SCENE_NAME_RE = re.compile(r"^scene-(\d+)$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="JSON manifest produced by select_nuscenes_eval_scenes.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("selected_keyframe_ranges.csv"),
        help="CSV file to write",
    )
    parser.add_argument(
        "--frame-base",
        type=int,
        choices=(0, 1),
        default=0,
        help="Output frame numbering: 0 keeps manifest indices; 1 adds one",
    )
    parser.add_argument(
        "--boundary-style",
        choices=("hyphen", "numeric"),
        default="hyphen",
        help=(
            "hyphen writes '-' when a run touches a scene boundary; "
            "numeric always writes integer frame numbers"
        ),
    )
    return parser.parse_args(argv)


def load_manifest(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)

    with resolved.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    if not isinstance(manifest, dict):
        raise ValueError("Manifest root must be a JSON object")
    if manifest.get("schema") != EXPECTED_SCHEMA:
        raise ValueError(
            f"Expected manifest schema {EXPECTED_SCHEMA!r}, "
            f"got {manifest.get('schema')!r}"
        )
    if manifest.get("schema_version") != 1:
        raise ValueError(
            f"Unsupported manifest schema version: {manifest.get('schema_version')!r}"
        )
    if not isinstance(manifest.get("selected_scenes"), list):
        raise ValueError("Manifest is missing selected_scenes")

    return manifest


def scene_number(scene_name: str) -> int:
    match = SCENE_NAME_RE.fullmatch(scene_name)
    if match is None:
        raise ValueError(f"Unexpected nuScenes scene name: {scene_name!r}")
    return int(match.group(1))


def qualifying_runs(keyframes: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """Return inclusive contiguous ranges of qualifying manifest frame indices."""
    if not keyframes:
        return []

    ordered = sorted(keyframes, key=lambda frame: int(frame["frame_index"]))
    indices = [int(frame["frame_index"]) for frame in ordered]

    expected = list(range(len(ordered)))
    if indices != expected:
        raise ValueError(
            "Expected consecutive zero-based frame_index values; "
            f"got {indices[:10]}{'...' if len(indices) > 10 else ''}"
        )

    runs: list[tuple[int, int]] = []
    run_start: int | None = None

    for frame in ordered:
        index = int(frame["frame_index"])
        qualifies = frame.get("qualifies")
        if not isinstance(qualifies, bool):
            raise ValueError(f"Frame {index} has non-boolean qualifies value")

        if qualifies and run_start is None:
            run_start = index
        elif not qualifies and run_start is not None:
            runs.append((run_start, index - 1))
            run_start = None

    if run_start is not None:
        runs.append((run_start, indices[-1]))

    return runs


def csv_rows(
    manifest: dict[str, Any],
    frame_base: int,
    boundary_style: str,
) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []

    for scene in manifest["selected_scenes"]:
        if not isinstance(scene, dict):
            raise ValueError("Each selected_scenes entry must be an object")

        name = scene.get("scene_name")
        keyframes = scene.get("keyframes")
        if not isinstance(name, str) or not isinstance(keyframes, list):
            raise ValueError("Selected scene is missing scene_name/keyframes")

        number = scene_number(name)
        if not keyframes:
            continue

        last_index = len(keyframes) - 1
        for start_index, end_index in qualifying_runs(keyframes):
            start_value: int | str = start_index + frame_base
            end_value: int | str = end_index + frame_base

            if boundary_style == "hyphen":
                if start_index == 0:
                    start_value = "-"
                if end_index == last_index:
                    end_value = "-"

            rows.append(
                {
                    "scene": number,
                    "start": start_value,
                    "end": end_value,
                    "comment": "",
                }
            )

    return rows


def write_csv_atomic(path: Path, rows: list[dict[str, int | str]]) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    pending = resolved.with_name(resolved.name + ".tmp")

    with pending.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("scene", "start", "end", "comment"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    pending.replace(resolved)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = load_manifest(args.manifest)
    rows = csv_rows(
        manifest,
        frame_base=args.frame_base,
        boundary_style=args.boundary_style,
    )
    write_csv_atomic(args.output, rows)

    distinct_scenes = len({int(row["scene"]) for row in rows})
    print(
        f"Wrote {len(rows)} qualifying keyframe ranges from "
        f"{distinct_scenes} scenes to {args.output.expanduser().resolve()}"
    )
    print("Comments are blank by default. No nuScenes sensor data was read or copied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
