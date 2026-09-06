from nuscenes.nuscenes import NuScenes
from PIL import Image
import json

# Initialize the nuScenes object (update paths as needed)

def get_bevfusion_dict(nusc, sample_token):
    """
    Constructs the BEVFusion intrinsic/extrinsic dictionary for a given sample.
    """
    sample = nusc.get('sample', sample_token)
    info = {}
    images_list = []
    cam_paths_dict = {}
    
    # 1. Process all 6 cameras
    camera_names = [
        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 
        'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT'
    ]
    
    for cam in camera_names:
        # 1. Get sensor data record
        cam_data_token = sample['data'][cam]
        cam_data = nusc.get('sample_data', cam_data_token)
        
        # 2. Extract spatial metadata for the JSON
        calib_sensor = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        ego_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])
        
        info[cam] = {
            "sensor2ego_translation": calib_sensor['translation'],
            "sensor2ego_rotation": calib_sensor['rotation'],
            "ego2global_translation": ego_pose['translation'],
            "ego2global_rotation": ego_pose['rotation'],
            "intrins": calib_sensor['camera_intrinsic']
        }
        
        # 3. Get image absolute path and load PIL Image
        file_path = nusc.get_sample_data_path(cam_data_token)
        cam_paths_dict[cam] = file_path
        
        # Load and ensure RGB format
        img = Image.open(file_path).convert('RGB')
        images_list.append(img)
        
    # Process LiDAR (LIDAR_TOP) metadata
    lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    lidar_calib = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
    lidar_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    
    info["lidar2ego_translation"] = lidar_calib['translation']
    info["lidar2ego_rotation"] = lidar_calib['rotation']
    info["ego2global_translation"] = lidar_pose['translation']
    info["ego2global_rotation"] = lidar_pose['rotation']
    
    return info, images_list, cam_paths_dict


def get_scene_samples(nusc, scene_name):
    """Yields all samples for a specific scene in chronological order."""
    
    # 1. Find the scene dictionary by its name
    scene = next((s for s in nusc.scene if s['name'] == scene_name), None)
    if not scene:
        raise ValueError(f"Scene '{scene_name}' not found in the loaded dataset!")
        
    # 2. Start at the first keyframe (sample) of the scene
    current_token = scene['first_sample_token']
    
    # 3. Traverse the linked list until the scene ends
    while current_token:
        sample = nusc.get('sample', current_token)
        yield sample
        
        # 'next' contains the token for the next frame, or "" if it's the last frame
        current_token = sample['next']