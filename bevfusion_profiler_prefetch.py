from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from bevfusion_integration.BEVFusionAppCustom import BEVFusionAppCustom
from qai_hub_models.utils.optimization import optimized_cumsum


@dataclass
class PreparedBEVFusionInputs:
    """Inputs prepared off the model-inference critical path."""

    intrins: np.ndarray
    sensor2keyegos: np.ndarray
    imgs: torch.Tensor
    inv_post_rots: torch.Tensor
    post_trans: torch.Tensor
    profile: dict[str, float]


class ProfiledPrefetchBEVFusionAppCustom(BEVFusionAppCustom):
    """BEVFusion app with split CPU preparation and GPU inference profiling.

    ``prepare_frame_inputs`` is safe to run in the prefetch worker: it only
    performs camera-geometry preparation and image preprocessing. Model forward
    passes remain in ``predict_prepared`` on the main thread.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_profile: dict[str, float] = {}

    def prepare_frame_inputs(
        self,
        images_list: list[Image.Image],
        cam_paths: dict[str, str],
        inputs_json: dict,
    ) -> PreparedBEVFusionInputs:
        profile: dict[str, float] = {}
        prepare_start = time.perf_counter()

        t0 = time.perf_counter()
        intrins_list, sensor2keyegos_list = self.prepare_camera_inputs(
            cam_paths, inputs_json
        )
        profile["camera_geometry_cpu"] = (time.perf_counter() - t0) * 1000.0

        intrins = np.asarray(intrins_list)
        sensor2keyegos = np.asarray(sensor2keyegos_list)

        t0 = time.perf_counter()
        imgs, inv_post_rots, post_trans = self.preprocess_images(images_list)
        profile["preprocess_cpu"] = (time.perf_counter() - t0) * 1000.0
        profile["prepare_wall"] = (time.perf_counter() - prepare_start) * 1000.0

        return PreparedBEVFusionInputs(
            intrins=intrins,
            sensor2keyegos=sensor2keyegos,
            imgs=imgs,
            inv_post_rots=inv_post_rots,
            post_trans=post_trans,
            profile=profile,
        )

    def predict_prepared(
        self,
        prepared: PreparedBEVFusionInputs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        profile = dict(prepared.profile)
        inference_start = time.perf_counter()

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

        mark("gpu_start")

        sensor2keyegos = torch.tensor(
            prepared.sensor2keyegos, device=self.device
        ).unsqueeze(0)
        inv_intrins = torch.inverse(
            torch.tensor(prepared.intrins, device=self.device)
        )
        imgs = prepared.imgs.to(self.device)
        inv_post_rots = prepared.inv_post_rots.to(self.device)
        post_trans = prepared.post_trans.to(self.device)
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

        profile["postprocess_wall"] = (
            time.perf_counter() - post_wall_start
        ) * 1000.0
        profile["inference_wall"] = (
            time.perf_counter() - inference_start
        ) * 1000.0
        profile["model_work_total"] = (
            profile.get("prepare_wall", 0.0) + profile["inference_wall"]
        )
        self.last_profile = profile

        return bboxes, scores, labels

    def predict_3d_boxes_from_images(
        self,
        images_list: list[Image.Image],
        cam_paths: dict[str, str],
        inputs_json: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Backward-compatible non-prefetched path."""
        prepared = self.prepare_frame_inputs(images_list, cam_paths, inputs_json)
        return self.predict_prepared(prepared)
