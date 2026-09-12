from __future__ import annotations

import time

import numpy as np
import torch
from PIL import Image

from bevfusion_integration.BEVFusionAppCustom import BEVFusionAppCustom
from qai_hub_models.utils.optimization import optimized_cumsum


class ProfiledBEVFusionAppCustom(BEVFusionAppCustom):
    """BEVFusionAppCustom with per-stage CPU/CUDA inference timing.

    The most recent timing sample is exposed in ``last_profile`` in milliseconds.
    CUDA events are used for GPU stages and one synchronization is performed at
    the end of each prediction so that the measurements are accurate.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_profile: dict[str, float] = {}

    def predict_3d_boxes_from_images(
        self,
        images_list: list[Image.Image],
        cam_paths: dict[str, str],
        inputs_json: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        profile: dict[str, float] = {}
        model_wall_start = time.perf_counter()

        use_cuda_events = (
            torch.cuda.is_available()
            and str(self.device).lower().startswith("cuda")
        )
        events: dict[str, torch.cuda.Event] = {}

        def mark(name: str) -> None:
            if use_cuda_events:
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                events[name] = event

        # CPU-side camera geometry preparation.
        t0 = time.perf_counter()
        intrins_list, sensor2keyegos_list = self.prepare_camera_inputs(
            cam_paths, inputs_json
        )
        profile["camera_geometry_cpu"] = (time.perf_counter() - t0) * 1000.0

        intrins_list = np.asarray(intrins_list)
        sensor2keyegos_list = np.asarray(sensor2keyegos_list)

        # PIL resize/crop/tensor construction is CPU-side in the base app.
        t0 = time.perf_counter()
        imgs, inv_post_rots, post_trans = self.preprocess_images(images_list)
        profile["preprocess_cpu"] = (time.perf_counter() - t0) * 1000.0

        mark("gpu_start")

        # Input construction / host-to-device copies.
        sensor2keyegos = torch.tensor(
            sensor2keyegos_list, device=self.device
        ).unsqueeze(0)
        inv_intrins = torch.inverse(
            torch.tensor(intrins_list, device=self.device)
        )
        imgs = imgs.to(self.device)
        inv_post_rots = inv_post_rots.to(self.device)
        post_trans = post_trans.to(self.device)
        mark("input_setup_h2d")

        assert imgs.shape[0] == 1, "Model supports only single batch"

        x = self.encoder1(imgs)
        mark("encoder1")

        x, lengths, geom_feats = self.encoder2(
            x.unsqueeze(0),
            inv_intrins.unsqueeze(0),
            sensor2keyegos,
            inv_post_rots,
            post_trans,
        )
        mark("encoder2")

        segment_indices = torch.cumsum(lengths, dim=0).long()
        x_reshaped = x.unsqueeze(0).reshape(96, 118, 176, 80).float()
        x = optimized_cumsum(x_reshaped).reshape(x.shape)
        x = x.reshape(-1, 80)
        segment = x[segment_indices, :]
        diff = segment[1:] - segment[:-1]
        segment = torch.cat([segment[:1], diff], dim=0)
        mark("cumsum_segment")

        x = self.encoder3(segment.unsqueeze(0), geom_feats.unsqueeze(0))
        mark("encoder3")

        pred_tensor = self.decoder(x)
        mark("decoder")

        # Unpack predictions into the structure expected by CenterHead decoding.
        post_wall_start = time.perf_counter()
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
            pred_dicts.append([dict(zip(head_order, split_tensors, strict=False))])

        # Preserve the repo's pre-NMS class filtering behaviour.
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

        conf_indices = scores >= self.score_threshold
        bboxes, scores, labels = (
            bboxes[conf_indices],
            scores[conf_indices],
            labels[conf_indices],
        )
        mark("decode_nms_filter")

        # Synchronize only once at the end. Synchronizing at every stage would
        # distort the workload and add artificial serialization.
        if use_cuda_events:
            torch.cuda.synchronize()
            order = [
                "gpu_start",
                "input_setup_h2d",
                "encoder1",
                "encoder2",
                "cumsum_segment",
                "encoder3",
                "decoder",
                "decode_nms_filter",
            ]
            for start_name, end_name in zip(order, order[1:]):
                profile[f"{end_name}_gpu"] = events[start_name].elapsed_time(
                    events[end_name]
                )

        profile["postprocess_wall"] = (time.perf_counter() - post_wall_start) * 1000.0
        profile["model_wall"] = (time.perf_counter() - model_wall_start) * 1000.0
        self.last_profile = profile

        return bboxes, scores, labels
