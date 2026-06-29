"""POLT runtime coordinator that loads models, data, and experiment pipelines."""

import torch
from config import get_experiment_config

from .data import (
    data_init as runtime_data_init,
    data_loader as runtime_data_loader,
    free_cache as runtime_free_cache,
    initialize_runtime_state,
    read_file as runtime_read_file,
    read_folder as runtime_read_folder,
    time_match as runtime_time_match,
)
from polt_runtime.online_learning import run_experiment as run_experiment_impl


class POLT:
    """Runtime coordinator shared by CLI presets and compatibility wrappers."""

    def __init__(self):
        self.device = torch.device("cuda")
        initialize_runtime_state(self)

        self.visualizer_paused = False
        self.visualizer_should_exit = False
        self.mppi_planner = None
        self.planner_reference_config = None
        self._last_salon_proprio_idx = -1

    def read_folder(self, folder_path):
        return runtime_read_folder(folder_path)

    def read_file(self, file_path):
        return runtime_read_file(self, file_path)

    def data_loader(self, path):
        return runtime_data_loader(self, path)

    def time_match(self, stamps, target_stamp, tss_gap=None):
        if tss_gap is None:
            return runtime_time_match(stamps, target_stamp)
        return runtime_time_match(stamps, target_stamp, tss_gap)

    def data_init(self, root):
        return runtime_data_init(self, root)

    def _free_cache(self):
        return runtime_free_cache(self)

    def build_experiment_config(
        self,
        preset,
        *,
        proprio_checkpoint=None,
        mem_buffer=None,
        mechanical_mem_buffer=None,
        roughness_mem_buffer=None,
        VIS=False,
        use_vlad=True,
        vlad_clusters=32,
        **overrides,
    ):
        """Merge CLI overrides into a preset-defined ``ExperimentConfig``."""
        config_overrides = dict(
            proprio_checkpoint=proprio_checkpoint,
            memory_buffer_path=mem_buffer,
            mechanical_memory_buffer_path=mechanical_mem_buffer,
            roughness_memory_buffer_path=roughness_mem_buffer,
            vis=VIS,
            use_vlad=use_vlad,
            vlad_clusters=vlad_clusters,
        )
        config_overrides.update(overrides)
        config_overrides = {key: value for key, value in config_overrides.items() if value is not None}
        return get_experiment_config(preset, **config_overrides)

    def run_experiment(
        self,
        root,
        preset,
        proprio_checkpoint=None,
        mem_buffer=None,
        mechanical_mem_buffer=None,
        roughness_mem_buffer=None,
        VIS=False,
        use_vlad=True,
        vlad_clusters=32,
        **overrides,
    ):
        """Create the final config and delegate execution to ``online_learning``."""
        config = self.build_experiment_config(
            preset,
            proprio_checkpoint=proprio_checkpoint,
            mem_buffer=mem_buffer,
            mechanical_mem_buffer=mechanical_mem_buffer,
            roughness_mem_buffer=roughness_mem_buffer,
            VIS=VIS,
            use_vlad=use_vlad,
            vlad_clusters=vlad_clusters,
            **overrides,
        )
        return run_experiment_impl(self, root, config)

    def _run_preset_compat(
        self,
        root,
        preset,
        *,
        proprio_checkpoint=None,
        mem_buffer=None,
        mechanical_mem_buffer=None,
        roughness_mem_buffer=None,
        VIS=False,
        use_vlad=True,
        vlad_clusters=32,
        **overrides,
    ):
        return self.run_experiment(
            root,
            preset,
            proprio_checkpoint=proprio_checkpoint,
            mem_buffer=mem_buffer,
            mechanical_mem_buffer=mechanical_mem_buffer,
            roughness_mem_buffer=roughness_mem_buffer,
            VIS=VIS,
            use_vlad=use_vlad,
            vlad_clusters=vlad_clusters,
            **overrides,
        )

    def run_offline_learning(self, root, proprio_checkpoint, mem_buffer, VIS=True, use_vlad=True, vlad_clusters=32):
        return self._run_preset_compat(
            root,
            "offline_learning",
            proprio_checkpoint=proprio_checkpoint,
            mem_buffer=mem_buffer,
            VIS=VIS,
            use_vlad=use_vlad,
            vlad_clusters=vlad_clusters,
        )

    def run_online_learning(self, root, proprio_checkpoint, mem_buffer, VIS=False, use_vlad=True, vlad_clusters=32, enable_planner=True):
        return self._run_preset_compat(
            root,
            "online_learning",
            proprio_checkpoint=proprio_checkpoint,
            mem_buffer=mem_buffer,
            VIS=VIS,
            use_vlad=use_vlad,
            vlad_clusters=vlad_clusters,
        )

    def run_online_learning_dual_cost(
        self,
        root,
        proprio_checkpoint,
        mechanical_mem_buffer,
        roughness_mem_buffer,
        VIS=False,
        use_vlad=True,
        vlad_clusters=32,
        enable_planner=True,
        planner_reference_mode="future_odom_debug",
        max_frames=None,
        planner_debug=False,
        profile_timing=True,
        save_planner_traces=True,
    ):
        return self._run_preset_compat(
            root,
            "online_learning_dual_cost",
            proprio_checkpoint=proprio_checkpoint,
            mechanical_mem_buffer=mechanical_mem_buffer,
            roughness_mem_buffer=roughness_mem_buffer,
            VIS=VIS,
            use_vlad=use_vlad,
            vlad_clusters=vlad_clusters,
            enable_planner=enable_planner,
            planner_reference_mode=planner_reference_mode,
            max_frames=max_frames,
            planner_debug=planner_debug,
            profile_timing=profile_timing,
            save_planner_traces=save_planner_traces,
        )

    def run_online_learning_with_databuffer(self, root, proprio_checkpoint, mem_buffer, VIS=False, use_vlad=True, vlad_clusters=32):
        return self._run_preset_compat(
            root,
            "online_learning_with_databuffer",
            proprio_checkpoint=proprio_checkpoint,
            mem_buffer=mem_buffer,
            VIS=VIS,
            use_vlad=use_vlad,
            vlad_clusters=vlad_clusters,
        )

    def run_salon(self, root, mem_buffer, VIS=False, use_vlad=True, vlad_clusters=32):
        return self._run_preset_compat(
            root,
            "salon",
            mem_buffer=mem_buffer,
            VIS=VIS,
            use_vlad=use_vlad,
            vlad_clusters=vlad_clusters,
        )


__all__ = ["POLT"]
