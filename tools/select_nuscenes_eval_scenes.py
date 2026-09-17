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

Optional map filter:
- ``--exclude-intersections-and-parking-lots`` rejects an otherwise qualifying
  scene when its ego trajectory intersects either:
    * a nuScenes ``road_segment`` with ``is_intersection == True``; or
    * a nuScenes ``carpark_area`` polygon.
- ``--map-buffer-meters`` can expand the ego trajectory before that test.
  The default 0.0 means the trajectory itself must intersect the polygon.

This script reads nuScenes JSON metadata/annotations and, when the optional map
filter is enabled, the nuScenes map-expansion JSON directly. It does NOT copy, export, or
read camera images, LiDAR point clouds, radar, or sweeps.

Examples:
    python tools/select_nuscenes_eval_scenes.py \
        --root Z:/dataset/nuscenes \
        --output-json selected_nuscenes_train_scenes.json

    python tools/select_nuscenes_eval_scenes.py --root Z:/dataset/nuscenes --output-json selected_nuscenes_train_scenes.json --min-vehicles 10
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
        default=0.1,
        help="Minimum fraction of a scene's keyframes that must qualify",
    )
    parser.add_argument(
        "--exclude-intersections-and-parking-lots",
        action="store_true",
        help=(
            "After density selection, reject scenes whose ego trajectory "
            "intersects a nuScenes intersection or carpark_area polygon"
        ),
    )
    parser.add_argument(
        "--map-buffer-meters",
        type=float,
        default=0.0,
        help=(
            "Optional buffer around the ego trajectory for the map filter; "
            "only valid with --exclude-intersections-and-parking-lots"
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every selected or map-rejected scene while scanning",
    )
    args = parser.parse_args(argv)

    if args.min_vehicles < 1:
        parser.error("--min-vehicles must be >= 1")
    if not 0.0 < args.min_keyframe_fraction <= 1.0:
        parser.error("--min-keyframe-fraction must be in (0, 1]")
    if not math.isfinite(args.map_buffer_meters) or args.map_buffer_meters < 0:
        parser.error("--map-buffer-meters must be a finite value >= 0")
    if (
        args.map_buffer_meters != 0.0
        and not args.exclude_intersections_and_parking_lots
    ):
        parser.error(
            "--map-buffer-meters requires "
            "--exclude-intersections-and-parking-lots"
        )

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
    """Return a sample annotation's nuScenes category name."""
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


def scene_ego_positions(
    nusc: Any,
    scene: dict[str, Any],
) -> list[tuple[float, float]]:
    """Return keyframe ego x/y positions in global map coordinates.

    Only sample_data and ego_pose metadata are read; the LIDAR_TOP point-cloud
    files themselves are never opened.
    """
    positions: list[tuple[float, float]] = []

    for sample in iter_scene_samples(nusc, scene):
        lidar_token = sample["data"].get("LIDAR_TOP")
        if not lidar_token:
            raise ValueError(
                f"Sample {sample['token']} in {scene['name']} has no "
                "LIDAR_TOP metadata"
            )

        sample_data = nusc.get("sample_data", lidar_token)
        ego_pose = nusc.get("ego_pose", sample_data["ego_pose_token"])
        x, y = ego_pose["translation"][:2]
        positions.append((float(x), float(y)))

    if not positions:
        raise ValueError(f"Scene {scene['name']} contains no keyframes")

    return positions


class IntersectionParkingFilter:
    """Reject scenes entering nuScenes intersection/carpark map polygons.

    The nuScenes map-expansion JSON is parsed directly rather than importing
    ``nuscenes.map_expansion.map_api.NuScenesMap``. This keeps metadata-only
    filtering independent of Matplotlib/seaborn plotting dependencies.
    """

    def __init__(
        self,
        nusc: Any,
        root: Path,
        buffer_meters: float = 0.0,
    ) -> None:
        self.nusc = nusc
        self.root = root
        self.buffer_meters = buffer_meters
        self._cache: dict[str, tuple[Any, Any]] = {}

    @staticmethod
    def _polygon_from_record(
        polygon_record: dict[str, Any],
        node_by_token: dict[str, dict[str, Any]],
    ) -> Any:
        from shapely.geometry import Polygon

        exterior = [
            (
                float(node_by_token[token]["x"]),
                float(node_by_token[token]["y"]),
            )
            for token in polygon_record["exterior_node_tokens"]
        ]

        interiors = []
        for hole in polygon_record.get("holes", []):
            coords = [
                (
                    float(node_by_token[token]["x"]),
                    float(node_by_token[token]["y"]),
                )
                for token in hole.get("node_tokens", [])
            ]
            if coords:
                interiors.append(coords)

        polygon = Polygon(exterior, interiors)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)

        return polygon

    def _load_map_geometries(self, map_name: str) -> tuple[Any, Any]:
        cached = self._cache.get(map_name)
        if cached is not None:
            return cached

        from shapely.geometry import GeometryCollection
        from shapely.ops import unary_union

        map_path = self.root / "maps" / "expansion" / f"{map_name}.json"
        if not map_path.is_file():
            raise FileNotFoundError(f"nuScenes map file not found: {map_path}")

        with map_path.open("r", encoding="utf-8") as handle:
            map_data = json.load(handle)

        node_by_token = {
            record["token"]: record
            for record in map_data.get("node", [])
        }
        polygon_by_token = {
            record["token"]: record
            for record in map_data.get("polygon", [])
        }

        def extract_polygon(polygon_token: str) -> Any:
            try:
                polygon_record = polygon_by_token[polygon_token]
            except KeyError as exc:
                raise ValueError(
                    f"Map {map_name!r} references unknown polygon "
                    f"{polygon_token!r}"
                ) from exc

            try:
                return self._polygon_from_record(
                    polygon_record,
                    node_by_token,
                )
            except KeyError as exc:
                raise ValueError(
                    f"Map {map_name!r} polygon {polygon_token!r} references "
                    f"an unknown node {exc.args[0]!r}"
                ) from exc

        intersection_polygons = [
            extract_polygon(record["polygon_token"])
            for record in map_data.get("road_segment", [])
            if bool(record.get("is_intersection", False))
        ]

        carpark_polygons = [
            extract_polygon(record["polygon_token"])
            for record in map_data.get("carpark_area", [])
        ]

        intersections = (
            unary_union(intersection_polygons)
            if intersection_polygons
            else GeometryCollection()
        )
        carparks = (
            unary_union(carpark_polygons)
            if carpark_polygons
            else GeometryCollection()
        )

        cached = (intersections, carparks)
        self._cache[map_name] = cached
        return cached

    @staticmethod
    def _trajectory_geometry(
        positions: list[tuple[float, float]],
        buffer_meters: float,
    ) -> Any:
        from shapely.geometry import LineString, Point

        trajectory = (
            Point(positions[0])
            if len(positions) == 1
            else LineString(positions)
        )
        if buffer_meters > 0.0:
            trajectory = trajectory.buffer(buffer_meters)
        return trajectory

    @staticmethod
    def _keyframe_hits(
        positions: list[tuple[float, float]],
        geometry: Any,
    ) -> list[int]:
        from shapely.geometry import Point

        return [
            index
            for index, position in enumerate(positions)
            if geometry.intersects(Point(position))
        ]

    def classify(self, scene: dict[str, Any]) -> dict[str, Any]:
        log = self.nusc.get("log", scene["log_token"])
        map_name = str(log["location"])
        intersections, carparks = self._load_map_geometries(map_name)

        positions = scene_ego_positions(self.nusc, scene)
        trajectory = self._trajectory_geometry(
            positions,
            self.buffer_meters,
        )

        hits_intersection = bool(trajectory.intersects(intersections))
        hits_parking_lot = bool(trajectory.intersects(carparks))

        return {
            "scene_name": scene["name"],
            "scene_token": scene["token"],
            "map_name": map_name,
            "reject": hits_intersection or hits_parking_lot,
            "hits_intersection": hits_intersection,
            "hits_parking_lot": hits_parking_lot,
            "intersection_keyframe_indices": self._keyframe_hits(
                positions,
                intersections,
            ),
            "parking_lot_keyframe_indices": self._keyframe_hits(
                positions,
                carparks,
            ),
        }


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
    map_filter: IntersectionParkingFilter | None = None,
    verbose: bool = False,
) -> tuple[list[dict[str, Any]], int, list[dict[str, Any]]]:
    """Analyse the official training split and return every qualifying scene.

    Returns:
        selected: final selected scene records.
        density_qualified_count: number passing the GT vehicle-density rule.
        map_rejected: map-filter audit records for scenes removed after density
            selection.
    """
    scene_by_name = {scene["name"]: scene for scene in nusc.scene}
    missing = sorted(set(train_scene_names) - set(scene_by_name))
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"{len(missing)} official training scenes are missing from the loaded "
            f"dataset (first: {preview}). Is this the full {DATASET_VERSION} dataset?"
        )

    selected: list[dict[str, Any]] = []
    map_rejected: list[dict[str, Any]] = []
    density_qualified_count = 0

    for index, scene_name in enumerate(train_scene_names, start=1):
        scene = scene_by_name[scene_name]
        density_selected, record = analyse_scene(
            nusc,
            scene,
            min_vehicles=min_vehicles,
            min_keyframe_fraction=min_keyframe_fraction,
        )

        if density_selected:
            density_qualified_count += 1

            if map_filter is not None:
                map_result = map_filter.classify(scene)
                if map_result["reject"]:
                    map_rejected.append(map_result)
                    if verbose:
                        reasons = []
                        if map_result["hits_intersection"]:
                            reasons.append("intersection")
                        if map_result["hits_parking_lot"]:
                            reasons.append("parking lot")
                        print(f"REJECT {scene_name}: {', '.join(reasons)}")
                    continue

                record["map_filter"] = {
                    "map_name": map_result["map_name"],
                    "hits_intersection": False,
                    "hits_parking_lot": False,
                }

            selected.append(record)
            if verbose:
                print(
                    f"SELECT {scene_name}: "
                    f"{record['qualifying_keyframes']}/{record['num_keyframes']} "
                    f"keyframes >= {min_vehicles} target vehicles"
                )

        if index % 50 == 0:
            progress = (
                f"Scanned {index}/{len(train_scene_names)} training scenes; "
                f"density-qualified {density_qualified_count}; "
                f"selected {len(selected)}"
            )
            if map_filter is not None:
                progress += f"; map-rejected {len(map_rejected)}"
            print(progress, flush=True)

    selected.sort(key=lambda record: record["scene_name"])
    map_rejected.sort(key=lambda record: record["scene_name"])
    return selected, density_qualified_count, map_rejected


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.expanduser().resolve()

    metadata_dir = root / DATASET_VERSION
    if not metadata_dir.is_dir():
        raise FileNotFoundError(
            f"nuScenes metadata directory not found: {metadata_dir}"
        )

    if args.exclude_intersections_and_parking_lots:
        map_dir = root / "maps" / "expansion"
        if not map_dir.is_dir():
            raise FileNotFoundError(
                "Map filtering was requested, but the nuScenes map expansion "
                f"directory was not found: {map_dir}"
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

    map_filter: IntersectionParkingFilter | None = None
    if args.exclude_intersections_and_parking_lots:
        map_filter = IntersectionParkingFilter(
            nusc=nusc,
            root=root,
            buffer_meters=args.map_buffer_meters,
        )

    selected, density_qualified_count, map_rejected = select_scenes(
        nusc,
        train_scene_names=train_scene_names,
        min_vehicles=args.min_vehicles,
        min_keyframe_fraction=args.min_keyframe_fraction,
        map_filter=map_filter,
        verbose=args.verbose,
    )

    manifest = {
        "schema": "bevfusion-qc.nuscenes-scene-selection",
        # Keep version 1 so export_qualifying_keyframe_ranges.py remains
        # compatible. The added fields are optional extensions.
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
            "exclude_intersections_and_parking_lots": (
                args.exclude_intersections_and_parking_lots
            ),
            "map_trajectory_buffer_meters": (
                args.map_buffer_meters
                if args.exclude_intersections_and_parking_lots
                else None
            ),
        },
        "map_filter": {
            "enabled": args.exclude_intersections_and_parking_lots,
            "definition": (
                "Reject a density-qualified scene when its keyframe ego "
                "trajectory intersects a road_segment with is_intersection=true "
                "or a carpark_area polygon."
                if args.exclude_intersections_and_parking_lots
                else None
            ),
            "trajectory_buffer_meters": (
                args.map_buffer_meters
                if args.exclude_intersections_and_parking_lots
                else None
            ),
        },
        "num_scenes_examined": len(train_scene_names),
        "num_scenes_density_qualified": density_qualified_count,
        "num_scenes_rejected_by_map_filter": len(map_rejected),
        "num_scenes_selected": len(selected),
        "selected_scene_names": [record["scene_name"] for record in selected],
        "selected_scenes": selected,
        "map_filter_rejected_scenes": map_rejected,
    }

    write_json_atomic(args.output_json, manifest)
    write_text_atomic(
        args.output_txt,
        [record["scene_name"] for record in selected],
    )

    print(
        f"Density rule qualified {density_qualified_count} of "
        f"{len(train_scene_names)} training scenes."
    )
    if args.exclude_intersections_and_parking_lots:
        print(
            f"Map filter rejected {len(map_rejected)} density-qualified scenes; "
            f"{len(selected)} remain."
        )
    else:
        print(f"Selected {len(selected)} scenes; map filter disabled.")

    print(f"JSON manifest: {args.output_json.expanduser().resolve()}")
    print(f"Text manifest: {args.output_txt.expanduser().resolve()}")
    print("No scene data, images, point clouds, radar, or sweeps were copied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
