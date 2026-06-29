"""Setup helpers for sensors, visualization, memory, and planner state."""

from dataclasses import dataclass

import cv2

from model.proprio.salon_cost import SalonCostCalculator
from model.vision.dinov3_infer import Dinov3Infer
from polt_runtime import visualization
from sensors.lidar import Lidar
from sensors.proprio import Prorio


@dataclass
class SingleFeedbackContext:
    lidar_sample: Lidar
    proprio_sample: Prorio
    dino_sampler: Dinov3Infer
    salon_sampler: object
    memory_backend: object


@dataclass
class DualFeedbackContext:
    lidar_sample: Lidar
    proprio_sample: Prorio
    dino_sampler: Dinov3Infer
    salon_sampler: SalonCostCalculator
    mechanical_memory: object
    roughness_memory: object


def init_single_feedback_context(runner, root, config, feedback_mode, memory_builder):
    lidar_sample = Lidar()
    proprio_sample = Prorio()
    if feedback_mode == "mechanical":
        if not config.proprio_checkpoint:
            raise ValueError("mechanical feedback requires proprio_checkpoint")
        proprio_sample.model_init(config.proprio_checkpoint)
    dino_sampler = Dinov3Infer(use_vlad=config.use_vlad, vlad_clusters=config.vlad_clusters)
    salon_sampler = SalonCostCalculator() if feedback_mode == "roughness" else None
    memory_backend = memory_builder(config, dino_sampler, feedback_mode)

    runner.data_init(root)
    proprio_sample.data_init(runner.proprio_files)
    return SingleFeedbackContext(lidar_sample, proprio_sample, dino_sampler, salon_sampler, memory_backend)


def init_dual_feedback_context(runner, root, config, memory_builder):
    lidar_sample = Lidar()
    proprio_sample = Prorio()
    if not config.proprio_checkpoint:
        raise ValueError("dual feedback experiment requires proprio_checkpoint")
    proprio_sample.model_init(config.proprio_checkpoint)
    salon_sampler = SalonCostCalculator()
    dino_sampler = Dinov3Infer(use_vlad=config.use_vlad, vlad_clusters=config.vlad_clusters)
    mechanical_memory = memory_builder(config, dino_sampler, "mechanical")
    roughness_memory = memory_builder(config, dino_sampler, "roughness")

    runner.data_init(root)
    proprio_sample.data_init(runner.proprio_files)
    return DualFeedbackContext(
        lidar_sample,
        proprio_sample,
        dino_sampler,
        salon_sampler,
        mechanical_memory,
        roughness_memory,
    )


def setup_learning_visualization(runner, config, title, include_bev=False):
    if not config.vis:
        return None, None

    vis = visualization.create_continuous_visualizer(runner, title)
    colorbar_img = (
        visualization.create_rough_vertical_cost_colorbar()
        if config.colorbar_kind == "roughness"
        else visualization.create_vertical_cost_colorbar()
    )
    cv2.namedWindow("Front Camera Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Front Camera Image", 800, 600)
    cv2.namedWindow("Traversability Color Bar", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Traversability Color Bar", 200, 600)
    if include_bev:
        cv2.namedWindow("BEV Risk Map", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("BEV Risk Map", 1500, 500)
    return vis, colorbar_img


def teardown_learning_visualization(config, vis):
    if config.vis and vis is not None:
        vis.destroy_window()
