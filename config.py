"""Experiment presets and runtime configuration for the POLT entrypoint."""

from dataclasses import dataclass, replace
from typing import Optional


FEEDBACK_MODES = ("mechanical", "roughness", "both")
MEMORY_MODES = ("dynamic_memory", "max_similiarity_out")
UPDATE_MODES = ("online", "offline")
LEGACY_MEMORY_MODE_ALIASES = {"fixed_size_data_buffer": "max_similiarity_out"}

DEFAULT_PROPRIO_CHECKPOINT = "weights/proprio"
DEFAULT_MECHANICAL_MEM_BUFFER = "mem_buffer/meta_memory"
DEFAULT_ROUGHNESS_MEM_BUFFER = "data_buffer_salon/baotou"
DEFAULT_VLAD_CLUSTERS = 32
DEFAULT_PLANNER_REFERENCE_MODE = "future_odom_debug"


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    feedback_mode: str
    memory_mode: str
    update_mode: str
    proprio_checkpoint: Optional[str] = None
    memory_buffer_path: Optional[str] = None
    mechanical_memory_buffer_path: Optional[str] = None
    roughness_memory_buffer_path: Optional[str] = None
    use_vlad: bool = True
    vlad_clusters: int = DEFAULT_VLAD_CLUSTERS
    vis: bool = False
    feature_voxel_size: Optional[float] = None
    min_speed_for_roughness: float = 3.0
    save_log_name: str = "log.json"
    colorbar_kind: str = "cost"
    enable_planner: bool = True
    planner_reference_mode: str = DEFAULT_PLANNER_REFERENCE_MODE
    max_frames: Optional[int] = None
    planner_debug: bool = False
    profile_timing: bool = True
    save_planner_traces: bool = True


PRESET_EXPERIMENTS = {
    "online_learning": ExperimentConfig(
        name="online_learning",
        feedback_mode="mechanical",
        memory_mode="dynamic_memory",
        update_mode="online",
        save_log_name="log.json",
        colorbar_kind="cost",
        enable_planner=False,
    ),
    "offline_learning": ExperimentConfig(
        name="offline_learning",
        feedback_mode="mechanical",
        memory_mode="dynamic_memory",
        update_mode="offline",
        vis=True,
        save_log_name="log_offline.json",
        colorbar_kind="cost",
        enable_planner=False,
    ),
    "online_learning_with_databuffer": ExperimentConfig(
        name="online_learning_with_databuffer",
        feedback_mode="mechanical",
        memory_mode="max_similiarity_out",
        update_mode="online",
        save_log_name="log_databuffer.json",
        colorbar_kind="cost",
        enable_planner=False,
    ),
    "salon": ExperimentConfig(
        name="salon",
        feedback_mode="roughness",
        memory_mode="max_similiarity_out",
        update_mode="online",
        feature_voxel_size=None,
        save_log_name="log_salon.json",
        colorbar_kind="roughness",
        enable_planner=False,
    ),
    "online_learning_dual_cost": ExperimentConfig(
        name="online_learning_dual_cost",
        feedback_mode="both",
        memory_mode="dynamic_memory",
        update_mode="online",
        save_log_name="log_dual_cost.json",
        colorbar_kind="cost",
        enable_planner=True,
    ),
}


def get_experiment_config(name: str, **overrides) -> ExperimentConfig:
    if name not in PRESET_EXPERIMENTS:
        raise KeyError(f"Unknown experiment preset: {name}")
    if "memory_mode" in overrides:
        overrides["memory_mode"] = LEGACY_MEMORY_MODE_ALIASES.get(overrides["memory_mode"], overrides["memory_mode"])
    config = PRESET_EXPERIMENTS[name]
    return replace(config, **overrides)


__all__ = [
    "ExperimentConfig",
    "FEEDBACK_MODES",
    "MEMORY_MODES",
    "PRESET_EXPERIMENTS",
    "UPDATE_MODES",
    "LEGACY_MEMORY_MODE_ALIASES",
    "DEFAULT_PROPRIO_CHECKPOINT",
    "DEFAULT_MECHANICAL_MEM_BUFFER",
    "DEFAULT_ROUGHNESS_MEM_BUFFER",
    "DEFAULT_VLAD_CLUSTERS",
    "DEFAULT_PLANNER_REFERENCE_MODE",
    "get_experiment_config",
]
