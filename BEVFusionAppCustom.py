import sys
sys.path.insert(0, './packages')

import torch
from PIL import Image
import numpy as np

import time

# from qai_hub_models.models.bevfusion_det.model import BEVFusion
from qai_hub_models.models.bevfusion_det.app import BEVFusionApp
from qai_hub_models.models.bevfusion_det.model import (
    BEVFusion,
    BEVFusionEncoder1,
    BEVFusionEncoder2,
    BEVFusionEncoder3,
    BEVFusionDecoder   
)
from qai_hub_models.utils.optimization import optimized_cumsum

def normalize_image_torchvision(
    image_tensor: torch.Tensor,
    image_tensor_has_batch: bool = True,
    is_video: bool = False,
) -> torch.Tensor:
    """
    Stand in function to deal with tensors on different devices
    """
    shape = [-1, 1, 1]
    if image_tensor_has_batch:
        shape.insert(0, 1)
    if is_video:
        shape.append(1)
    mean = torch.tensor([0.485, 0.456, 0.406], device=image_tensor.device).reshape(*shape)
    std = torch.tensor([[0.229, 0.224, 0.225]], device=image_tensor.device).reshape(*shape)
    return (image_tensor - mean) / std

class BEVFusionEncoder1Custom(BEVFusionEncoder1):
    def forward(self, imgs):
        B, NC, H, W = imgs.size()
        imgs_reshaped = imgs.reshape(B * (NC // 3), 3, H, W)
        imgs_normalized = normalize_image_torchvision(imgs_reshaped)
        x = self.backbone(imgs_normalized)
        x = self.neck(x)
        return x[0]

class BEVFusionAppCustom(BEVFusionApp):
    """
    Custom BEVFusionApp to extract the wlhxyz tensors instead of annotated_imgs/box_corners
    """

    def __init__(self, 
                 encoder1, encoder2, encoder3, decoder, 
                 num_classes, task_heads, get_bboxes, 
                 model_input_shape = (256, 704), 
                 score_threshold = 0.4, 
                 nms_threshold = 4, 
                 nms_post_max_size = 83,
                 device: torch.device = "cpu", # Device to run the model on
                 class_filter: list[int] | None = None
                 ):

        self.device = device
        if class_filter:
            self.class_filter = torch.tensor(class_filter, device='cpu')

        # Casting hack cuz dev be dumdum and use deprecated init, which force the tensor on CPU
        encoder1.__class__ = BEVFusionEncoder1Custom

        for module in [encoder1, encoder2, encoder3, decoder]:
            if hasattr(module, 'to'):
                module.to(self.device)

        if isinstance(task_heads, list):
            for head in task_heads:
                if hasattr(head, 'to'):
                    head.to(self.device)
        else:
            if hasattr(task_heads, 'to'):
                task_heads.to(self.device)

        super().__init__(encoder1, encoder2, encoder3, decoder, num_classes, task_heads, get_bboxes, model_input_shape, score_threshold, nms_threshold, nms_post_max_size)

    def get_post_rot_and_tran(
        self, resize: float, crop: tuple[int, int, int, int], rotate: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        post_rot = torch.eye(3)
        post_tran = torch.zeros(3)

        # post-homography transformation
        post_rot[:2, :2] *= resize
        post_tran[:2] -= torch.tensor(crop[:2])

        rotate_angle = torch.tensor(rotate / 180 * np.pi)
        rot_sin = torch.sin(rotate_angle)
        rot_cos = torch.cos(rotate_angle)
        A = torch.tensor([[rot_cos, rot_sin], [-rot_sin, rot_cos]])
        b = torch.tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot[:2, :2] = A.matmul(post_rot[:2, :2])
        post_tran[:2] = A.matmul(post_tran[:2]) + b

        return post_rot, post_tran

    def predict_3d_boxes_from_images(
            self,
            images_list: list[Image.Image],
            cam_paths: dict[str, str],
            inputs_json: dict,
            # raw_output: bool = False,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Run the BEVFusion model and predict 3D bounding boxes.
    
            Parameters
            ----------
            images_list
                List of PIL Images in RGB format.
            cam_paths
                Dictionary mapping camera names to image file paths.
            inputs_json
                JSON dictionary containing intrinsics and transformation data.
    
            Returns
            -------
            bbox, scores and labels tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            """
            intrins_list, sensor2keyegos_list = self.prepare_camera_inputs(
                cam_paths, inputs_json
            )

            # Create Tensor on device
            sensor2keyegos = torch.tensor(sensor2keyegos_list, device=self.device).unsqueeze(0)
            inv_intrins = torch.inverse(torch.tensor(intrins_list, device=self.device))

            imgs, inv_post_rots, post_trans = self.preprocess_images(images_list)

            # Move tensor to device
            imgs = imgs.to(self.device)
            inv_post_rots = inv_post_rots.to(self.device)
            post_trans = post_trans.to(self.device)
    
            assert imgs.shape[0] == 1, "Model supports only single batch"
            x = self.encoder1(imgs)
    
            x, lengths, geom_feats = self.encoder2(
                x.unsqueeze(0),
                inv_intrins.unsqueeze(0),
                sensor2keyegos,
                inv_post_rots,
                post_trans,
            )
            segment_indices = torch.cumsum(lengths, dim=0).long()
    
            x_reshaped = x.unsqueeze(0).reshape(96, 118, 176, 80).float()
            x = optimized_cumsum(x_reshaped).reshape(x.shape)
            x = x.reshape(-1, 80)
            segment = x[segment_indices, :]
            diff = segment[1:] - segment[:-1]
            segment = torch.cat([segment[:1], diff], dim=0)
    
            x = self.encoder3(segment.unsqueeze(0), geom_feats.unsqueeze(0))
            pred_tensor = self.decoder(x)

            # unpack predictions into dict
            pred_dicts = []
            start = 0
            head_order = ["reg", "height", "dim", "rot", "vel", "heatmap"]
            for i, nc in enumerate(self.num_classes):
                task_head = self.task_heads[i]
                reg_heads = getattr(task_head, "heads", None)
                channels = []
                for key in head_order:
                    if key == "heatmap":
                        channels.append(nc)
                    else:
                        if reg_heads is None:
                            raise ValueError(f"reg_heads is None for task_head {i}")
                        channels.append(reg_heads[key][0])
    
                total = sum(channels)
                split_slice = pred_tensor[:, start : start + total]
                split_tensors = torch.split(split_slice, channels, dim=1)
                start += total
    
                pred_dict = dict(zip(head_order, split_tensors, strict=False))
                pred_dicts.append([pred_dict])

            # Pre NMS filtering
            if self.class_filter is not None:
                allowed_classes = self.class_filter.tolist()
                global_class_idx = 0
                for i, nc in enumerate(self.num_classes):
                    for local_channel_idx in range(nc):
                        if global_class_idx not in allowed_classes:
                            pred_dicts[i][0]["heatmap"][:, local_channel_idx, :, :] = -1e4
                        global_class_idx += 1

            bboxes, scores, labels = self.get_bboxes(pred_dicts)[0]

            if not isinstance(scores, torch.Tensor):
                scores = torch.tensor(scores, device=self.device)
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, device=self.device)

            # Threshold filter
            conf_indices = scores >= self.score_threshold
            # if self.class_filter is not None:
            #     class_indices = torch.isin(labels, self.class_filter)
            #     indices = conf_indices & class_indices
            # else:
            #     indices = conf_indices
                
            bboxes, scores, labels = (
                bboxes[conf_indices],
                scores[conf_indices],
                labels[conf_indices]
            )

            return bboxes, scores, labels