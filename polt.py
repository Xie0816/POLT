"""Command-line entrypoint for running POLT experiment presets."""

import argparse

from config import (
    DEFAULT_MECHANICAL_MEM_BUFFER,
    DEFAULT_PLANNER_REFERENCE_MODE,
    DEFAULT_PROPRIO_CHECKPOINT,
    DEFAULT_ROUGHNESS_MEM_BUFFER,
    DEFAULT_VLAD_CLUSTERS,
    FEEDBACK_MODES,
    MEMORY_MODES,
    PRESET_EXPERIMENTS,
    UPDATE_MODES,
)
from polt_runtime.runtime.coordinator import POLT


def build_parser():
    experiment_presets = sorted(PRESET_EXPERIMENTS.keys())
    parser = argparse.ArgumentParser(description="POLT runtime entrypoint")
    parser.add_argument("--root", default = "/home/xie/Data/Terrain_Dataset/2025-02-25-14-40-20",help="dataset root path")
    parser.add_argument(
        "--mode",
        default="experiment",
        choices=[
            "experiment",
            "offline_learning",
            "online_learning",
            "online_learning_dual_cost",
            "online_learning_with_databuffer",
            "salon",
        ],
        help="which POLT pipeline to run",
    )
    parser.add_argument(
        "--preset",
        default="online_learning_dual_cost",
        choices=experiment_presets,
        help="experiment preset",
    )
    parser.add_argument("--feedback-mode", choices=FEEDBACK_MODES, default="mechanical")
    parser.add_argument("--memory-mode", choices=MEMORY_MODES, default="dynamic_memory")
    parser.add_argument("--update-mode", choices=UPDATE_MODES, default="online")
    parser.add_argument("--proprio-checkpoint", default=DEFAULT_PROPRIO_CHECKPOINT)
    parser.add_argument("--mechanical-mem-buffer", default=DEFAULT_MECHANICAL_MEM_BUFFER)
    parser.add_argument("--roughness-mem-buffer", default=DEFAULT_ROUGHNESS_MEM_BUFFER)
    parser.add_argument("--mem-buffer", default=DEFAULT_MECHANICAL_MEM_BUFFER)
    parser.add_argument("--vis", default=True)
    parser.add_argument("--use-vlad", action="store_true", default=True)
    parser.add_argument("--disable-vlad", dest="use_vlad", action="store_false")
    parser.add_argument("--vlad-clusters", type=int, default=DEFAULT_VLAD_CLUSTERS)
    parser.add_argument("--enable-planner", action="store_true", default=True)
    parser.add_argument("--disable-planner", dest="enable_planner", action="store_false")
    parser.add_argument("--planner-reference-mode", default=DEFAULT_PLANNER_REFERENCE_MODE)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--planner-debug", action="store_true")
    parser.add_argument("--disable-profile-timing", dest="profile_timing", action="store_false")
    parser.add_argument("--disable-save-planner-traces", dest="save_planner_traces", action="store_false")
    parser.set_defaults(profile_timing=True, save_planner_traces=True)
    return parser


def _resolve_preset(mode, preset):
    if mode == "experiment":
        return preset
    return mode


def main():
    args = build_parser().parse_args()
    runner = POLT()
    preset = _resolve_preset(args.mode, args.preset)

    experiment_overrides = {
        "feedback_mode": args.feedback_mode,
        "memory_mode": args.memory_mode,
        "update_mode": args.update_mode,
        "enable_planner": args.enable_planner,
        "planner_reference_mode": args.planner_reference_mode,
        "max_frames": args.max_frames,
        "planner_debug": args.planner_debug,
        "profile_timing": args.profile_timing,
        "save_planner_traces": args.save_planner_traces,
    }

    if experiment_overrides["feedback_mode"] == "both" or preset == "online_learning_dual_cost":
        runner.run_experiment(
            args.root,
            preset,
            proprio_checkpoint=args.proprio_checkpoint,
            mechanical_mem_buffer=args.mechanical_mem_buffer,
            roughness_mem_buffer=args.roughness_mem_buffer,
            VIS=args.vis,
            use_vlad=args.use_vlad,
            vlad_clusters=args.vlad_clusters,
            **experiment_overrides,
        )
        return

    mem_buffer = args.roughness_mem_buffer if preset == "salon" else args.mem_buffer
    runner.run_experiment(
        args.root,
        preset,
        proprio_checkpoint=args.proprio_checkpoint,
        mem_buffer=mem_buffer,
        VIS=args.vis,
        use_vlad=args.use_vlad,
        vlad_clusters=args.vlad_clusters,
        **experiment_overrides,
    )


if __name__ == "__main__":
    main()
