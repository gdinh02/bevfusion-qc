from __future__ import annotations

import os
from pathlib import Path
import queue
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
import torch

import run_live_demo_bev as base
from bevfusion_profiler_prefetch import ProfiledPrefetchBEVFusionAppCustom
from pipeline import ProfiledOptimizedLanePipeline
from run_live_demo_profiled_prefetch import (
    FramePrefetcher,
    PrefetchedFrame,
    _ms,
    snapshot_detections_once,
)

from lane_inference.configs import (
    BEVFusionConfig,
    BoundaryTrackingConfig,
    LaneBoundaryConfig,
    LaneFitConfig,
    LaneGraphConfig,
    LaneMergeConfig,
    LeadVehicleConfig,
    TemporalConfig,
)


@dataclass
class InferenceResult:
    item: PrefetchedFrame
    bboxes: np.ndarray
    scores: np.ndarray
    labels: np.ndarray
    yaw_mask: np.ndarray
    profile: dict[str, float]


class InferenceWorker:
    """Consume prepared frames and run all CUDA/model work on one thread."""
    _END = object()

    def __init__(
        self,
        app: ProfiledPrefetchBEVFusionAppCustom,
        prefetcher: FramePrefetcher,
        depth: int = 2,
    ) -> None:
        self.app = app
        self.prefetcher = prefetcher
        self.queue: queue.Queue[object] = queue.Queue(maxsize=max(1, depth))
        self.stop_event = threading.Event()
        self.error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._worker,
            name="bevfusion-gpu-inference",
            daemon=True,
        )
        self._thread.start()

    def _put(self, item: object) -> tuple[bool, float]:
        block_start = time.perf_counter()
        while not self.stop_event.is_set():
            try:
                self.queue.put(item, timeout=0.1)
                return True, _ms(block_start)
            except queue.Full:
                continue
        return False, _ms(block_start)

    def _worker(self) -> None:
        try:
            worker_device = torch.device(self.app.device)
            if worker_device.type == "cuda":
                cuda_index = (
                    worker_device.index
                    if worker_device.index is not None
                    else torch.cuda.current_device()
                )
                torch.cuda.set_device(cuda_index)
                worker_device = torch.device("cuda", cuda_index)
            torch.set_default_device(worker_device)

            while not self.stop_event.is_set():
                t0 = time.perf_counter()
                item = self.prefetcher.get()
                prefetch_wait_ms = _ms(t0)
                if item is None:
                    self._put(self._END)
                    return

                timings = dict(item.profile)
                timings["prefetch_wait_worker"] = prefetch_wait_ms
                stage_start = time.perf_counter()

                with torch.inference_mode():
                    bboxes, scores, labels = self.app.predict_prepared(item.prepared)
                timings.update(dict(self.app.last_profile))

                bboxes_np, scores_np, labels_np, yaw_mask_np = snapshot_detections_once(
                    bboxes,
                    scores,
                    labels,
                    timings,
                )
                timings["inference_stage_wall"] = _ms(stage_start)

                result = InferenceResult(
                    item=item,
                    bboxes=bboxes_np,
                    scores=scores_np,
                    labels=labels_np,
                    yaw_mask=np.asarray(yaw_mask_np, dtype=bool),
                    profile=timings,
                )

                ok, block_ms = self._put(result)
                if not ok:
                    return
                result.profile["inference_result_queue_block"] = block_ms
        except BaseException as exc:
            self.error = exc
            self._put(self._END)

    def get(self) -> InferenceResult | None:
        item = self.queue.get()
        try:
            if item is self._END:
                if self.error is not None:
                    raise RuntimeError("Inference worker failed") from self.error
                return None
            assert isinstance(item, InferenceResult)
            return item
        finally:
            self.queue.task_done()

    def close(self) -> None:
        self.stop_event.set()
        self._thread.join(timeout=10.0)


def render_inference_result(
    app: ProfiledPrefetchBEVFusionAppCustom,
    lane_pipeline: ProfiledOptimizedLanePipeline,
    result: InferenceResult,
    lane_only: bool = True
):
    """CPU-only ordered post-processing for one completed inference result."""
    item = result.item
    timings = dict(result.profile)
    post_start = time.perf_counter()

    graph, streams, boundaries, vehicle_count = lane_pipeline.process(
        item.frame_id,
        item.inputs_json,
        result.bboxes,
        result.scores,
        result.labels,
        yaw_mask=result.yaw_mask,
    )
    timings.update(lane_pipeline.last_profile)

    t0 = time.perf_counter()
    if not lane_only:
        bev = base.generate_bev_map(
            bboxes=torch.from_numpy(result.bboxes),
            labels=torch.from_numpy(result.labels),
            pixels_per_meter=base.PIXELS_PER_METER,
            max_range_meters=20
        )
        bev = base.draw_lane_graph_on_bev(
            bev,
            graph,
            streams,
            pixels_per_meter=base.PIXELS_PER_METER,
        )
    else:
        img_size = int(20 * 2 * base.PIXELS_PER_METER)
        bev = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    
    bev = base.draw_lane_boundaries_on_bev(
        bev,
        boundaries,
        base.PIXELS_PER_METER,
    )
    
    timings["bev_render"] = _ms(t0)
    timings["cpu_post_wall"] = _ms(post_start)

    return bev, vehicle_count, len(streams), timings


class BEVFusionLiveProcessor:
    """Class to manage the asynchronous BEVFusion pipeline, returning bitmasks one frame at a time."""

    def __init__(self, 
        input_dir: str, 
        lane_width_meter=None,
        bevfusion_cfg  = BEVFusionConfig(),
        temporal_cfg   = TemporalConfig(),
        graph_cfg      = LaneGraphConfig(),
        fit_cfg        = LaneFitConfig(),
        merge_cfg      = LaneMergeConfig(),
        lead_cfg       = LeadVehicleConfig(),
        boundary_cfg   = LaneBoundaryConfig(),
        tracking_cfg   = BoundaryTrackingConfig()
    ):
        self.input_dir = input_dir
        
        # 1. Initialize models
        base.BEVFusionAppCustom = ProfiledPrefetchBEVFusionAppCustom
        self.app = base.build_app()
        self.lane_width_meter = lane_width_meter

        self.bevfusion_cfg  = bevfusion_cfg
        self.temporal_cfg   = temporal_cfg
        self.graph_cfg      = graph_cfg
        self.fit_cfg        = fit_cfg
        self.merge_cfg      = merge_cfg
        self.lead_cfg       = lead_cfg
        self.boundary_cfg   = boundary_cfg
        self.tracking_cfg   = tracking_cfg
        
        # Track state
        self.expected_frame_id = 0
        self.current_scene = None

    def set_scene(self, scene_name: str, start_idx: int = 0):
        """Loads a new scene and starts the pipeline from the specified frame."""
        print(f"\n[Scene Change] Initializing scene: {scene_name} at frame {start_idx}")
        
        # Stop existing workers if we are switching to a new scene
        self._stop_workers()
        
        # UPDATE THE TRACKER so we don't infinitely reload
        self.current_scene = scene_name 
        
        self.root, self.camera_order, self.frames, self.timestamps, self.info_path = base.load_scene(
            Path(os.path.join(self.input_dir, scene_name))
        )
        
        # Start workers at requested frame
        self._start_workers(start_idx=start_idx)

    def _start_workers(self, start_idx: int):
        """Starts (or restarts) the prefetch and inference queues at a specific index."""
        # Initialize (or reset) the lane pipeline to clear temporal tracking history
        self.lane_pipeline = ProfiledOptimizedLanePipeline(
            bevfusion_cfg = self.bevfusion_cfg,
            temporal_cfg = self.temporal_cfg,
            graph_cfg = self.graph_cfg,
            fit_cfg = self.fit_cfg,
            merge_cfg = self.merge_cfg,
            lead_cfg = self.lead_cfg,
            boundary_cfg = self.boundary_cfg,
            tracking_cfg = self.tracking_cfg
        )
        
        if self.lane_width_meter is not None:
            self.lane_pipeline.boundary_cfg.default_lane_width = self.lane_width_meter

        self.prefetcher = FramePrefetcher(
            self.app, self.root, self.camera_order, self.frames, depth=3, start_index=start_idx
        ) # type: ignore
        self.inference_worker = InferenceWorker(self.app, self.prefetcher, depth=2) # type: ignore
        
        self.expected_frame_id = start_idx

    def _stop_workers(self):
        """Gracefully shuts down current workers and flushes queues."""
        if hasattr(self, 'inference_worker'):
            self.inference_worker.close()
        if hasattr(self, 'prefetcher'):
            self.prefetcher.close()

    def process(self, scene_name: str, frame_idx: int | None = None) -> tuple[int, np.ndarray, dict] | None:
        """
        Pulls the next inference result from the worker thread, renders the lane prediction, 
        and extracts a 2D bitmask.
        
        Args:
            scene_name: The folder name of the scene to process.
            frame_idx: If provided, dynamically seeks to this frame by resetting queues.
            
        Returns:
            Tuple containing:
            - frame_id (int)
            - inf_bitmask (np.ndarray): 1-channel binary image (0 or 255)
            - timings (dict): Profiling timestamps
            
            Returns None if the sequence is complete.
        """
        
        # --- SCENE SWITCH LOGIC ---
        if scene_name != self.current_scene:
            # If changing scene and frame_idx is provided, start there. Otherwise start at 0.
            start = frame_idx if frame_idx is not None else 0
            self.set_scene(scene_name, start_idx=start)
            
        # --- SEEK LOGIC (Within the same scene) ---
        elif frame_idx is not None and frame_idx != self.expected_frame_id:
            if frame_idx < 0 or frame_idx >= len(self.frames):
                raise IndexError(f"Frame index {frame_idx} out of bounds (0-{len(self.frames)-1})")
            
            print(f"\nJumping to frame {frame_idx}. Flushing queues and resetting temporal state...")
            self._stop_workers()
            self._start_workers(start_idx=frame_idx)
        # ------------------

        wait_start = time.perf_counter()
        
        # 1. Block and wait for the GPU worker thread to yield a completed frame
        result = self.inference_worker.get()
        inference_wait_ms = _ms(wait_start)
        
        if result is None:
            return None # End of frames

        processed_frame_id = result.item.frame_id
        
        # 2. Run post-processing on CPU thread
        combined, vehicles_count, streams, timings = render_inference_result(
            self.app,  # type:ignore
            self.lane_pipeline,
            result,
            lane_only=True
        )
        timings["inference_wait_main"] = inference_wait_ms

        # 3. Convert rendered BEV image to a pure 1-channel bitmask
        inf_gray = cv2.cvtColor(combined, cv2.COLOR_BGR2GRAY)
        _, inf_bitmask = cv2.threshold(inf_gray, 1, 255, cv2.THRESH_BINARY)

        # Update the expected pointer for the next loop
        self.expected_frame_id = processed_frame_id + 1

        return processed_frame_id, inf_bitmask, vehicles_count, timings

    def close(self):
        """Safely shut down the worker threads."""
        print("Shutting down worker threads...")
        self._stop_workers()


def main():
    # Only requires the dataset input directory
    processor = BEVFusionLiveProcessor(input_dir="/home/gdtrinh/nuscenes/scenes-pkl")
    output_folder = "./test_masks_output"
    os.makedirs(output_folder, exist_ok=True)

    # Define the scene we are testing
    current_scene = "scene-0277"

    print(f"Pipeline started. Testing scene: {current_scene}")
    try:
        # --- TEST 1: Normal sequential processing ---
        print("\n--- Testing Sequential Processing (Frames 0-2) ---")
        for _ in range(3):
            # Notice we MUST pass scene_name now
            result = processor.process(scene_name=current_scene)
            
            if result is None:
                break
                
            frame_id, bev_bitmask, timings = result
            print(f"Processed frame {frame_id:04d} | Inference Time: {timings.get('inference_stage_wall', 0):.1f}ms")

            save_path = os.path.join(output_folder, f"{current_scene}_mask_{frame_id:04d}.png")
            cv2.imwrite(save_path, bev_bitmask)


        # --- TEST 2: Seek logic ---
        print("\n--- Testing Seek Logic: Jumping to Frame 15 ---")
        # Passing frame_idx=15 will trigger the internal flush and restart
        result = processor.process(scene_name=current_scene, frame_idx=15)
        if result is not None:
            frame_id, bev_bitmask, timings = result
            print(f"Jumped to frame {frame_id:04d} | Inference Time: {timings.get('inference_stage_wall', 0):.1f}ms")

            save_path = os.path.join(output_folder, f"{current_scene}_mask_{frame_id:04d}.png")
            cv2.imwrite(save_path, bev_bitmask)


        # --- TEST 3: Resume sequential from new position ---
        print("\n--- Testing Resume Sequential (Frames 16+) ---")
        while True:
            # Leaving frame_idx=None continues sequentially from 16
            result = processor.process(scene_name=current_scene)
            
            if result is None:
                print("End of sequence.")
                break
                
            frame_id, bev_bitmask, timings = result
            print(f"Processed frame {frame_id:04d} | Inference Time: {timings.get('inference_stage_wall', 0):.1f}ms")

            save_path = os.path.join(output_folder, f"{current_scene}_mask_{frame_id:04d}.png")
            cv2.imwrite(save_path, bev_bitmask)
            
            # For testing purposes, let's stop at frame 20 so it doesn't run forever
            if frame_id >= 20:
                print("Test threshold reached. Stopping.")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"\nError occurred: {e}")
    finally:
        processor.close()
        # cv2.destroyAllWindows()
        print("Pipeline closed cleanly.")

if __name__ == "__main__":
    main()