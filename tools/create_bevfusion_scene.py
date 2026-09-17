#!/usr/bin/env python3
"""Export one nuScenes scene for gdinh02/bevfusion-qc camera-only inference.

python tools/create_bevfusion_scene.py --scene scene-0956 --root Z:/dataset/nuscenes --output-base Z:/dataset/

Usage (run in an environment with nuscenes-devkit installed):
    python create_bevfusion_scene.py --scene scene-0095 \
        --root /path/to/nuscenes --output-base /path/to/exports

Output:
    scene-0095/samples/<CAMERA>/<image.jpg>
    scene-0095/bevfusion_infos_scene-0095.pkl

The PKL contains plain Python dictionaries/lists (no MMDetection3D objects):
    metainfo: schema, schema_version, scene_name, scene_token, dataset_version,
              camera_order, timestamp_unit, quaternion_order, reference_sensor
    data_list: one record per keyframe, in scene order:
        frame_index, token, scene_token, timestamp (seconds), timestamp_us,
        sample_timestamp_us, camera_timestamps_us, cam_paths, inputs_json

cam_paths are POSIX paths relative to the exported scene directory. inputs_json
matches get_bevfusion_dict() in bevfusion-qc: each camera has intrins,
sensor2ego_translation/rotation and its own ego2global_translation/rotation;
the top-level lidar2ego_* and ego2global_* use LIDAR_TOP calibration/pose.
Quaternions are preserved as nuScenes [w, x, y, z]. No transforms are changed.

Only keyframe images are copied. No point clouds, radar, sweeps, annotations,
full infos PKL or nuScenes JSON directory are copied. nuScenes JSON metadata
is required at export time, but not for reading the exported scene.
Existing scene folders are reused: existing images and unrelated files remain
untouched, missing images are copied from the source, and only the BEVFusion
PKL is replaced. nuScenes JSON metadata is still required to rebuild the PKL.
Image reuse checks file existence, not image contents or calibration accuracy.
A failed rebuild preserves the old PKL; successfully copied missing images may
remain in an existing folder. New scene exports are staged before publication.

Reader outline (load only PKLs you generated/trust):
    payload = pickle.load(open(info_path, 'rb'))
    for record in payload['data_list']:
        paths = {cam: str(scene_root / record['cam_paths'][cam])
                 for cam in payload['metainfo']['camera_order']}
        # Open each image as RGB in that same camera order.
        # Pass images, paths, record['inputs_json'] to the existing app.

The current run_demo.py must be adapted to this reader; it does not already
consume this PKL. Model loading, detection and tracking are not run here.

Adapted from gdinh02/mmdetection3d-fyp/nuscenes_tools/create_fcos3d_scene.py
and gdinh02/bevfusion-qc/bevfusion_integration/nuscenes_helper.py.
"""
from __future__ import annotations

import argparse
import math
import pickle
import shutil
import tempfile
from pathlib import Path

# Match the BEVFusion helper exactly; never rely on incidental dict ordering.
CAMERAS = (
    'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
    'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT',
)

GT_VEHICLE_CLASS_MAP = {
    "vehicle.car": "car",
    "vehicle.truck": "truck",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.trailer": "trailer",
    "vehicle.construction": "construction",
}

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scene', required=True, help='For example scene-0095')
    parser.add_argument('--root', required=True, type=Path, help='Source nuScenes root')
    parser.add_argument('--output-base', required=True, type=Path,
                        help='Parent directory containing the new or existing scene folder')
    parser.add_argument('--version', default='v1.0-trainval',
                        choices=['v1.0-trainval', 'v1.0-mini', 'v1.0-test'])
    return parser.parse_args(argv)


def finite_vector(values, length, name):
    result = [float(value) for value in values]
    if len(result) != length or not all(math.isfinite(v) for v in result):
        raise ValueError(f'Invalid {name}: expected {length} finite values')
    if length == 4 and sum(v * v for v in result) < 1e-12:
        raise ValueError(f'Invalid zero quaternion: {name}')
    return result


def pose_fields(calibration, pose, prefix):
    return {
        f'{prefix}2ego_translation': finite_vector(calibration['translation'], 3, 'translation'),
        f'{prefix}2ego_rotation': finite_vector(calibration['rotation'], 4, 'rotation'),
        'ego2global_translation': finite_vector(pose['translation'], 3, 'ego translation'),
        'ego2global_rotation': finite_vector(pose['rotation'], 4, 'ego rotation'),
    }


def sensor_records(nusc, token):
    data = nusc.get('sample_data', token)
    calibration = nusc.get('calibrated_sensor', data['calibrated_sensor_token'])
    pose = nusc.get('ego_pose', data['ego_pose_token'])
    return data, calibration, pose


def iterate_scene_samples(nusc, scene):
    token = scene['first_sample_token']
    seen = set()
    while token:
        if token in seen:
            raise ValueError(f'Cycle in scene sample chain: {token}')
        seen.add(token)
        sample = nusc.get('sample', token)
        if sample['scene_token'] != scene['token']:
            raise ValueError(f'Sample {token} belongs to another scene')
        yield sample
        token = sample['next']


def build_scene_gt_track_map(nusc, scene):
    """
    Assign a deterministic scene-local integer track ID to every relevant
    nuScenes vehicle instance.

    Sorting the instance tokens makes the mapping independent of annotation
    iteration order.
    """
    instance_tokens = set()

    for sample in iterate_scene_samples(nusc, scene):
        for annotation_token in sample["anns"]:
            annotation = nusc.get("sample_annotation", annotation_token)

            if annotation["category_name"] not in GT_VEHICLE_CLASS_MAP:
                continue

            instance_tokens.add(annotation["instance_token"])

    return {
        instance_token: track_id
        for track_id, instance_token in enumerate(sorted(instance_tokens))
    }


def extract_gt_vehicles(nusc, sample, instance_track_ids):
    """
    Export relevant nuScenes GT vehicle annotations without changing their
    coordinate frame.

    translation and rotation remain in the nuScenes global frame.
    size remains nuScenes [width, length, height].
    """
    vehicles = []

    for annotation_token in sample["anns"]:
        annotation = nusc.get("sample_annotation", annotation_token)

        category_name = annotation["category_name"]
        class_name = GT_VEHICLE_CLASS_MAP.get(category_name)

        if class_name is None:
            continue

        instance_token = annotation["instance_token"]

        if instance_token not in instance_track_ids:
            raise ValueError(
                f"Missing GT track mapping for instance {instance_token}"
            )

        vehicles.append(
            {
                "annotation_token": annotation["token"],
                "instance_token": instance_token,
                "track_id": int(instance_track_ids[instance_token]),

                # Keep both granular nuScenes category and coarse class.
                "category_name": category_name,
                "class_name": class_name,

                # IMPORTANT: these stay in the nuScenes GLOBAL frame.
                "translation_global": finite_vector(
                    annotation["translation"],
                    3,
                    "GT translation",
                ),
                "rotation_global": finite_vector(
                    annotation["rotation"],
                    4,
                    "GT rotation",
                ),

                # nuScenes order: width, length, height.
                "size": finite_vector(
                    annotation["size"],
                    3,
                    "GT size",
                ),

                # Useful diagnostic/evaluation metadata.
                "visibility_token": str(
                    annotation.get("visibility_token", "")
                ),
                "num_lidar_pts": int(
                    annotation.get("num_lidar_pts", 0)
                ),
                "num_radar_pts": int(
                    annotation.get("num_radar_pts", 0)
                ),
            }
        )

    # Makes the output ordering reproducible as well.
    vehicles.sort(
        key=lambda vehicle: (
            vehicle["track_id"],
            vehicle["annotation_token"],
        )
    )

    return vehicles


def build_frame(
    nusc,
    sample,
    frame_index,
    output_root,
    image_counts,
    instance_track_ids,
):
    """Copy six images and preserve the current BEVFusion input dictionary."""
    lidar_token = sample['data'].get('LIDAR_TOP')
    if not lidar_token:
        raise ValueError(f"Sample {sample['token']} lacks LIDAR_TOP reference metadata")
    lidar_data, lidar_calibration, lidar_pose = sensor_records(nusc, lidar_token)
    inputs = {}
    paths = {}
    camera_times = {}
    for camera in CAMERAS:
        data, calibration, pose = sensor_records(nusc, sample['data'][camera])
        if not data.get('is_key_frame', False):
            raise ValueError(f"Expected keyframe image: {data['token']}")
        intrinsic = calibration['camera_intrinsic']
        if len(intrinsic) != 3:
            raise ValueError(f'Invalid intrinsic matrix for {camera}')
        inputs[camera] = pose_fields(calibration, pose, 'sensor')
        inputs[camera]['intrins'] = [finite_vector(row, 3, 'intrinsic row') for row in intrinsic]
        # devkit resolves the original filename; only camera image bytes are read.
        source = Path(nusc.get_sample_data_path(data['token']))
        relative = Path('samples') / camera / source.name
        destination = output_root / relative
        if destination.is_file():
            image_counts['reused'] += 1
        elif destination.exists():
            raise FileExistsError(f'Image destination is not a file: {destination}')
        else:
            if not source.is_file():
                raise FileNotFoundError(f'Image missing from export and source: {source}')
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Copy to a temporary file first, avoiding partial images after failure.
            with tempfile.TemporaryDirectory(prefix='.image-', dir=destination.parent) as tmp:
                copied_image = Path(tmp) / source.name
                shutil.copy2(source, copied_image)
                copied_image.replace(destination)
            image_counts['copied'] += 1
        paths[camera] = relative.as_posix()
        camera_times[camera] = int(data['timestamp'])

    inputs.update(pose_fields(lidar_calibration, lidar_pose, 'lidar'))
    reference_time = int(lidar_data['timestamp'])
    gt_vehicles = extract_gt_vehicles(
        nusc,
        sample,
        instance_track_ids,
    )
    return {
        'frame_index': frame_index,
        'token': sample['token'],
        'scene_token': sample['scene_token'],
        'timestamp': reference_time / 1e6,
        'timestamp_us': reference_time,
        'sample_timestamp_us': int(sample['timestamp']),
        'camera_timestamps_us': camera_times,
        'cam_paths': paths,
        'inputs_json': inputs,
        'gt_vehicles': gt_vehicles,
    }


def validate_export(payload, scene_root):
    frames = payload['data_list']
    if not frames:
        raise ValueError('Scene contains no frames')
    previous_timestamp = None
    tokens = set()
    track_to_instance = {}
    instance_to_track = {}
    annotation_tokens = set()

    for index, frame in enumerate(frames):
        if "gt_vehicles" not in frame:
            raise ValueError(
                f"Frame {index} has no gt_vehicles field"
            )

        for vehicle in frame["gt_vehicles"]:
            annotation_token = vehicle["annotation_token"]
            instance_token = vehicle["instance_token"]
            track_id = vehicle["track_id"]
            category_name = vehicle["category_name"]

            if category_name not in GT_VEHICLE_CLASS_MAP:
                raise ValueError(
                    f"Unexpected GT category: {category_name}"
                )

            if (
                not isinstance(track_id, int)
                or isinstance(track_id, bool)
                or track_id < 0
            ):
                raise ValueError(
                    f"Invalid GT track ID: {track_id}"
                )

            if annotation_token in annotation_tokens:
                raise ValueError(
                    f"Duplicate GT annotation token: "
                    f"{annotation_token}"
                )
            annotation_tokens.add(annotation_token)

            previous_instance = track_to_instance.get(track_id)
            if (
                previous_instance is not None
                and previous_instance != instance_token
            ):
                raise ValueError(
                    f"GT track {track_id} maps to multiple instances"
                )
            track_to_instance[track_id] = instance_token

            previous_track = instance_to_track.get(instance_token)
            if (
                previous_track is not None
                and previous_track != track_id
            ):
                raise ValueError(
                    f"GT instance {instance_token} maps to multiple tracks"
                )
            instance_to_track[instance_token] = track_id

            translation = vehicle["translation_global"]
            rotation = vehicle["rotation_global"]
            size = vehicle["size"]

            if (
                len(translation) != 3
                or not all(math.isfinite(x) for x in translation)
            ):
                raise ValueError(
                    f"Invalid GT translation: {translation}"
                )

            if (
                len(rotation) != 4
                or not all(math.isfinite(x) for x in rotation)
            ):
                raise ValueError(
                    f"Invalid GT rotation: {rotation}"
                )

            if (
                len(size) != 3
                or not all(
                    math.isfinite(x) and x > 0
                    for x in size
                )
            ):
                raise ValueError(
                    f"Invalid GT box size: {size}"
                )
        if frame['frame_index'] != index or frame['token'] in tokens:
            raise ValueError('Invalid frame ordering or duplicate sample token')
        tokens.add(frame['token'])
        timestamp = frame['timestamp_us']
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise ValueError('Reference timestamps must increase strictly')
        previous_timestamp = timestamp
        if set(frame['cam_paths']) != set(CAMERAS):
            raise ValueError('Frame must have all six cameras')
        for camera in CAMERAS:
            relative = Path(frame['cam_paths'][camera])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Image paths must stay relative to the scene folder')
            if not (scene_root / relative).is_file():
                raise FileNotFoundError(scene_root / relative)
            if camera not in frame['inputs_json'] or camera not in frame['camera_timestamps_us']:
                raise ValueError(f'Missing camera metadata: {camera}')

        expected_tracks = payload["metainfo"].get("num_gt_tracks")

    if expected_tracks != len(instance_to_track):
        raise ValueError(
            f"GT track count mismatch: metadata says "
            f"{expected_tracks}, found {len(instance_to_track)}"
        )

def export_scene(nusc, scene_name, output_base):
    scene = next((s for s in nusc.scene if s['name'] == scene_name), None)

    if scene is None:
        raise ValueError(f'Scene {scene_name!r} not found in {nusc.version}')
    if Path(scene_name).name != scene_name or scene_name in ('.', '..'):
        raise ValueError('Scene name must be a single directory name')
    output_base = Path(output_base).expanduser().resolve()
    target = output_base / scene_name
    reuse_existing = target.is_dir()

    if target.exists() and not reuse_existing:
        raise FileExistsError(f'Scene destination is not a directory: {target}')
    if reuse_existing:
        print(f'Reusing images in {target}; rebuilding the BEVFusion PKL.')
    output_base.mkdir(parents=True, exist_ok=True)
    instance_track_ids = build_scene_gt_track_map(nusc, scene)

    # Build in a temporary directory so a failed copy cannot look like a finished export.
    with tempfile.TemporaryDirectory(prefix=f'.{scene_name}-', dir=output_base) as temporary:
        staging = target if reuse_existing else Path(temporary) / scene_name
        if not reuse_existing:
            staging.mkdir()
        image_counts = {'reused': 0, 'copied': 0}
        frames = []
        for index, sample in enumerate(iterate_scene_samples(nusc, scene)):
            frames.append(
                build_frame(
                    nusc,
                    sample,
                    index,
                    staging,
                    image_counts,
                    instance_track_ids,
                )
            )
            print(f'Exported frame {index + 1}: {sample["token"]}', flush=True)
        if len(frames) != int(scene['nbr_samples']):
            raise ValueError('Scene sample count differs from its metadata')
        payload = {
            "metainfo": {
                "schema": "bevfusion-qc.scene",
                "schema_version": 2,
                "scene_name": scene_name,
                "scene_token": scene["token"],
                "dataset_version": nusc.version,
                "camera_order": list(CAMERAS),
                "timestamp_unit": "timestamp: seconds; *_us: microseconds",
                "quaternion_order": "wxyz",
                "reference_sensor": "LIDAR_TOP",
                "path_root": "scene_directory",
                "num_frames": len(frames),

                "ground_truth": {
                    "coordinate_frame": "global",
                    "size_order": ["width", "length", "height"],
                    "quaternion_order": "wxyz",
                    "tracking_source": "nuscenes_instance_token",
                    "track_id_scope": "scene",
                    "vehicle_categories": sorted(GT_VEHICLE_CLASS_MAP.keys()),
                },
                "num_gt_tracks": len(instance_track_ids),
            },
            'data_list': frames,
        }
        validate_export(payload, staging)
        filename = f'bevfusion_infos_{scene_name}.pkl'
        pending_pkl = Path(temporary) / filename
        with pending_pkl.open('wb') as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # Check serialization of our own generated file before publishing the folder.
        with pending_pkl.open('rb') as handle:
            validate_export(pickle.load(handle), staging)
        # Temporary file is on the same filesystem: replacement is atomic.
        pending_pkl.replace(staging / filename)
        if not reuse_existing:
            if target.exists():
                raise FileExistsError(f'Output appeared during export: {target}')
            staging.rename(target)
    print(f"Saved {len(frames)} frames to {target}; "
          f"images reused: {image_counts['reused']}, copied: {image_counts['copied']}")
    return target / filename


def main(argv=None):
    args = parse_args(argv)
    root = args.root.expanduser().resolve()
    if not (root / args.version).is_dir():
        raise FileNotFoundError(f'nuScenes metadata directory missing: {root / args.version}')
    # Keep --help usable without installing model or dataset packages.
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=args.version, dataroot=str(root), verbose=False)
    export_scene(nusc, args.scene, args.output_base)


if __name__ == '__main__':
    main()
