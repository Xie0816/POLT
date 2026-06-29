"""Experiment-level online learning flows for POLT.

The functions here keep the frame loop orchestration in one place and delegate
sensor synchronization, feedback estimation, memory operations, visualization,
and optional planning to smaller modules.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from planning import planner as planning_ops
from polt_runtime.memory import build_memory_backend, resolve_memory_buffer_path
from polt_runtime import perception, traversability, visualization
from polt_runtime.runtime.experiment_setup import (
    init_dual_feedback_context,
    init_single_feedback_context,
    setup_learning_visualization,
    teardown_learning_visualization,
)
from polt_runtime.runtime.frame_pipeline import (
    load_feature_frame,
    log_dual_cost_frame,
    record_timing_log,
    update_dual_feedback_memories,
    update_single_feedback_memory,
)
from polt_runtime.runtime.profiling import _print_timing_summary, _profile_now, _profile_record


def _estimate_feedback(runner, feedback_mode, config, proprio_sample, salon_sampler, proprio_idx):
    """Dispatch one proprioceptive feedback sample to the selected estimator."""
    if feedback_mode == "mechanical":
        return traversability.estimate_mechanical_feedback(proprio_sample, proprio_idx)
    if feedback_mode == "roughness":
        return traversability.estimate_roughness_feedback(
            runner,
            proprio_sample,
            salon_sampler,
            proprio_idx,
            min_speed=config.min_speed_for_roughness,
        )
    raise ValueError(f"Unsupported feedback mode: {feedback_mode}")


def run_experiment(runner, root, config):
    """Run the experiment selected by ``ExperimentConfig.feedback_mode``."""
    if config.feedback_mode == "both":
        return _run_dual_feedback_experiment(runner, root, config)
    return _run_single_feedback_experiment(runner, root, config)


def _save_log(log_data, log_path):
    with open(log_path, "w", encoding="utf-8") as handle:
        json.dump(log_data, handle, indent=4)


def _print_user_exit():
    print("用户请求退出，停止处理后续帧...")


def _resolve_experiment_log_path(root, config):
    if config.enable_planner:
        log_path = Path("planner_traces") / Path(root).name / config.save_log_name
        log_path.parent.mkdir(parents=True, exist_ok=True)
        return log_path
    return Path(root) / config.save_log_name


def _finalize_experiment(runner, root, config, vis, timing_records=None, dual_cost=False):
    """Close visualization/profiling resources and persist the experiment log."""
    teardown_learning_visualization(config, vis)
    if config.profile_timing and timing_records is not None:
        _print_timing_summary(runner, timing_records)
    log_path = _resolve_experiment_log_path(root, config)
    _save_log(runner.log, log_path)
    if dual_cost:
        print(f"dual-cost log saved: {log_path}")
    return log_path


def _should_visualize_predictions(config, accumulated_feats_points, predicted_costs):
    return config.vis and accumulated_feats_points is not None and len(predicted_costs) > 0


def _visualize_single_feedback_frame(
    runner,
    config,
    vis,
    colorbar_img,
    synced_frame,
    accumulated_feats_points,
    predicted_costs,
    accumulated_colored_proprio,
):
    cv2.imshow("Traversability Color Bar", colorbar_img)
    cur_img_show = cv2.cvtColor(np.array(synced_frame.image), cv2.COLOR_RGB2BGR)
    cv2.imshow("Front Camera Image", cur_img_show)
    cv2.waitKey(1)
    window_name = f"Frame: {synced_frame.img_stamp}, Points: {len(accumulated_feats_points)}"
    if config.colorbar_kind == "roughness":
        return visualization.update_continuous_roughness_visualizer(
            runner,
            vis,
            accumulated_feats_points,
            predicted_costs,
            accumulated_colored_proprio=accumulated_colored_proprio,
            window_name=window_name,
        )
    return visualization.update_continuous_visualizer(
        runner,
        vis,
        accumulated_feats_points,
        predicted_costs,
        accumulated_colored_proprio=accumulated_colored_proprio,
        window_name=window_name,
    )


def _run_single_feedback_frame(
    runner,
    config,
    feedback_mode,
    lidar_sample,
    proprio_sample,
    dino_sampler,
    salon_sampler,
    memory_backend,
    img_stamp,
    vis,
    colorbar_img,
):
    """Process one synchronized frame for mechanical-only or roughness-only modes."""
    print(f"##################处理{img_stamp}帧传感器结果##################")
    synced_frame, feature_frame = load_feature_frame(
        runner,
        lidar_sample,
        dino_sampler,
        img_stamp,
        config.use_vlad,
        config.feature_voxel_size,
    )
    if synced_frame is None:
        return True

    accumulated_feats_points = feature_frame.accumulated_feats_points
    print(f"累积带特征的点云数量: {len(accumulated_feats_points)}")
    feedback_cost = _estimate_feedback(
        runner,
        feedback_mode,
        config,
        proprio_sample,
        salon_sampler,
        synced_frame.proprio_idx,
    )
    update_result = update_single_feedback_memory(
        runner,
        config,
        synced_frame,
        accumulated_feats_points,
        memory_backend,
        feedback_cost,
    )
    if update_result is not None:
        print(
            f"{config.name}: cost={feedback_cost:.4f}, "
            f"pred={float(update_result['pred_info']['gpr_prediction']['mu']):.4f}, "
            f"distance={update_result['min_distance']:.4f}m"
        )

    accumulated_colored_proprio = traversability.accumulate_proprio_history(runner, lidar_sample, synced_frame.odom)
    predicted_costs = memory_backend.predict_batch(accumulated_feats_points, feedback_mode)
    if _should_visualize_predictions(config, accumulated_feats_points, predicted_costs):
        should_continue = _visualize_single_feedback_frame(
            runner,
            config,
            vis,
            colorbar_img,
            synced_frame,
            accumulated_feats_points,
            predicted_costs,
            accumulated_colored_proprio,
        )
        if not should_continue:
            _print_user_exit()
            return False

    runner._free_cache()
    return True


def _validate_dual_memory_paths(config):
    """Ensure dual-cost mode does not accidentally share one writable buffer."""
    mechanical_mem_buffer = resolve_memory_buffer_path(config, "mechanical")
    roughness_mem_buffer = resolve_memory_buffer_path(config, "roughness")
    if (
        mechanical_mem_buffer
        and roughness_mem_buffer
        and Path(mechanical_mem_buffer).resolve() == Path(roughness_mem_buffer).resolve()
    ):
        raise ValueError("mechanical_mem_buffer and roughness_mem_buffer must be different paths.")


def _load_dual_feature_frame(runner, lidar_sample, dino_sampler, img_stamp, config):
    """Load one synced frame and reuse it for feature projection in dual mode."""
    synced_frame = perception.load_synced_frame(runner, lidar_sample, img_stamp)
    if synced_frame is None:
        return None, None
    _, feature_frame = load_feature_frame(
        runner,
        lidar_sample,
        dino_sampler,
        img_stamp,
        config.use_vlad,
        config.feature_voxel_size,
        synced_frame=synced_frame,
    )
    return synced_frame, feature_frame


def _estimate_dual_feedbacks(runner, config, proprio_sample, salon_sampler, synced_frame, accumulated_feats_points):
    """Estimate mechanical and roughness feedback for the same contact point."""
    mechanical_feedback = _estimate_feedback(
        runner, "mechanical", config, proprio_sample, salon_sampler, synced_frame.proprio_idx
    )
    roughness_feedback = _estimate_feedback(
        runner, "roughness", config, proprio_sample, salon_sampler, synced_frame.proprio_idx
    )
    nearest_point_feat, nearest_distance = traversability.nearest_feature_to_contact(runner.device, accumulated_feats_points)
    return mechanical_feedback, roughness_feedback, nearest_point_feat, nearest_distance


def _predict_dual_costs(accumulated_feats_points, mechanical_memory, roughness_memory):
    """Predict both cost channels over the accumulated feature cloud."""
    mechanical_predicted_costs = mechanical_memory.predict_batch(accumulated_feats_points, "mechanical")
    roughness_predicted_costs = roughness_memory.predict_batch(accumulated_feats_points, "roughness")
    return mechanical_predicted_costs, roughness_predicted_costs


def _build_dual_risk_map(runner, accumulated_feats_points, mechanical_predicted_costs, roughness_predicted_costs):
    """Convert dual cost predictions into the layered map consumed by planning."""
    risk_map = planning_ops.build_dual_cost_risk_map_from_predictions(
        accumulated_feats_points,
        mechanical_predicted_costs,
        roughness_predicted_costs,
    )
    if risk_map is not None:
        runner.mechanical_cost_map = risk_map.layers["mechanical_cost"]
        runner.roughness_cost_map = risk_map.layers["roughness_cost"]
    collision_ratio = 0.0
    if risk_map is not None and "collision_layer" in risk_map.layers:
        collision_ratio = float(np.mean(risk_map.layers["collision_layer"] > 0.0))
    return risk_map, collision_ratio


def _planner_debug_lengths(planner_result):
    ref_path = planner_result.get("reference_path")
    visual_ref_path = planner_result.get("visual_reference_path")
    reachable_ref_path = planner_result.get("reachable_reference_path")
    return (
        planning_ops.reference_path_length(ref_path),
        planning_ops.reference_path_length(visual_ref_path),
        planning_ops.reference_path_length(reachable_ref_path),
    )


def _update_planner_debug_state(planner_result, prev_best_endpoint, prev_command):
    planner_command = planner_result["command"]
    planner_diag = planner_result["diagnostics"]
    best_endpoint_jump = None
    best_trajectory = np.asarray(planner_diag.best_trajectory, dtype=np.float64)
    if len(best_trajectory) > 0:
        best_endpoint = best_trajectory[-1, :2]
        if prev_best_endpoint is not None:
            best_endpoint_jump = float(np.linalg.norm(best_endpoint - prev_best_endpoint))
        prev_best_endpoint = best_endpoint
    steer_delta = None if prev_command is None else abs(float(planner_command.steering_angle) - prev_command)
    prev_command = float(planner_command.steering_angle)
    return planner_command, planner_diag, best_trajectory, best_endpoint_jump, steer_delta, prev_best_endpoint, prev_command


def _log_planner_result(runner, root, config, synced_frame, planner_result, collision_ratio, frame_profile):
    planner_command = planner_result["command"]
    planner_diag = planner_result["diagnostics"]
    planner_trace_path = None
    if config.save_planner_traces:
        stage_start = _profile_now(runner, config.profile_timing)
        planner_trace_path = planning_ops.save_planner_trace(runner, root, synced_frame.lidar_stamp, planner_result)
        _profile_record(runner, frame_profile, "planner_trace_save", stage_start, config.profile_timing)
    runner.log[int(synced_frame.lidar_stamp)]["planner_command"] = {
        "steering_angle": float(planner_command.steering_angle),
        "speed": float(planner_command.speed),
        "accel": float(getattr(planner_command, "accel", 0.0)),
    }
    runner.log[int(synced_frame.lidar_stamp)]["planner_diagnostics"] = {
        "reference_cost": float(planner_diag.reference_cost),
        "roughness_cost": float(planner_diag.roughness_cost),
        "mechanical_cost": float(planner_diag.mechanical_cost),
        "map_cost": float(planner_diag.map_cost),
        "total_cost": float(planner_diag.total_cost),
        "runtime_ms": float(planner_diag.runtime_ms),
        "reference_mode": config.planner_reference_mode,
        "cost_mode": "dual_cost_only",
        "planner_backend": planner_result.get("planner_backend", "unknown"),
        "trace_path": planner_trace_path,
        "collision_ratio": float(collision_ratio),
    }


def _run_dual_planner(
    runner,
    root,
    config,
    synced_frame,
    accumulated_feats_points,
    mechanical_predicted_costs,
    roughness_predicted_costs,
    risk_map,
    collision_ratio,
    frame_profile,
    prev_best_endpoint,
    prev_command,
):
    if not config.enable_planner or risk_map is None:
        return None, prev_best_endpoint, prev_command

    stage_start = _profile_now(runner, config.profile_timing)
    planner_result = planning_ops.run_mppi_planner(
        runner,
        synced_frame.odom_idx,
        accumulated_feats_points,
        mechanical_predicted_costs,
        roughness_predicted_costs,
        prebuilt_risk_map=risk_map,
        reference_mode=config.planner_reference_mode,
    )
    _profile_record(runner, frame_profile, "planner_total", stage_start, config.profile_timing)
    if planner_result is None:
        return None, prev_best_endpoint, prev_command

    (
        planner_command,
        planner_diag,
        best_trajectory,
        best_endpoint_jump,
        steer_delta,
        prev_best_endpoint,
        prev_command,
    ) = _update_planner_debug_state(planner_result, prev_best_endpoint, prev_command)

    if config.planner_debug:
        ref_len, visual_ref_len, reachable_ref_len = _planner_debug_lengths(planner_result)
        best_end_text = "(nan,nan)"
        if len(best_trajectory) > 0:
            best_end_text = f"({best_trajectory[-1, 0]:.3f},{best_trajectory[-1, 1]:.3f})"
        print(
            "Planner debug: "
            f"ref_len={ref_len:.3f}m, "
            f"visual_ref_len={visual_ref_len:.3f}m, "
            f"reachable_ref_len={reachable_ref_len:.3f}m, "
            f"best_end={best_end_text} "
            f"end_jump={best_endpoint_jump}, "
            f"steer_delta={steer_delta}, "
            f"rough={planner_diag.roughness_cost:.3f}, "
            f"mech={planner_diag.mechanical_cost:.3f}, "
            f"map={planner_diag.map_cost:.3f}, "
            f"collision_ratio={collision_ratio:.4f}, "
            f"ref_cost={planner_diag.reference_cost:.3f}"
        )

    print(
        "Portable planner dual-cost: "
        f"steer={planner_command.steering_angle:.4f}, "
        f"speed={planner_command.speed:.4f}, "
        f"map_cost={planner_diag.map_cost:.4f}, "
        f"total_cost={planner_diag.total_cost:.4f}"
    )
    _log_planner_result(runner, root, config, synced_frame, planner_result, collision_ratio, frame_profile)
    return planner_result, prev_best_endpoint, prev_command


def _visualize_dual_cost_frame(
    runner,
    config,
    vis,
    colorbar_img,
    synced_frame,
    accumulated_feats_points,
    mechanical_predicted_costs,
    risk_map,
    planner_result,
    frame_profile,
):
    if not _should_visualize_predictions(config, accumulated_feats_points, mechanical_predicted_costs):
        return True

    stage_start = _profile_now(runner, config.profile_timing)
    cv2.imshow("Traversability Color Bar", colorbar_img)
    cur_img_show = cv2.cvtColor(np.array(synced_frame.image), cv2.COLOR_RGB2BGR)
    cv2.imshow("Front Camera Image", cur_img_show)
    bev_risk_img = visualization.render_bev_risk_map(risk_map, planner_result=planner_result)
    if bev_risk_img is not None:
        cv2.imshow("BEV Risk Map", bev_risk_img)
    cv2.waitKey(1)
    _profile_record(runner, frame_profile, "opencv_bev_visualization", stage_start, config.profile_timing)

    stage_start = _profile_now(runner, config.profile_timing)
    window_name = f"Frame: {synced_frame.img_stamp}, Points: {len(accumulated_feats_points)}"
    should_continue = visualization.update_continuous_visualizer(
        runner,
        vis,
        accumulated_feats_points,
        mechanical_predicted_costs,
        planner_result=planner_result,
        window_name=window_name,
    )
    _profile_record(runner, frame_profile, "open3d_visualization", stage_start, config.profile_timing)
    return should_continue


def _finalize_dual_frame(runner, config, synced_frame, frame_profile, frame_start, timing_records):
    stage_start = _profile_now(runner, config.profile_timing)
    runner._free_cache()
    _profile_record(runner, frame_profile, "free_cache", stage_start, config.profile_timing)
    _profile_record(runner, frame_profile, "frame_total", frame_start, config.profile_timing)
    if not config.profile_timing:
        return
    timing_records.append(frame_profile.copy())
    record_timing_log(runner, synced_frame, frame_profile)
    top_items = sorted(
        (
            (key, value)
            for key, value in frame_profile.items()
            if key not in {"stamp", "frame_total", "accumulated_points"}
        ),
        key=lambda item: item[1],
        reverse=True,
    )[:6]
    top_text = ", ".join(f"{key}={value:.1f}ms" for key, value in top_items)
    total_ms = frame_profile.get("frame_total", 0.0)
    print(f"Timing: total={total_ms:.1f}ms, fps={1000.0 / max(total_ms, 1e-6):.2f}, {top_text}")


def _run_single_feedback_experiment(runner, root, config):
    feedback_mode = config.feedback_mode
    context = init_single_feedback_context(runner, root, config, feedback_mode, build_memory_backend)
    lidar_sample = context.lidar_sample
    proprio_sample = context.proprio_sample
    dino_sampler = context.dino_sampler
    salon_sampler = context.salon_sampler
    memory_backend = context.memory_backend

    vis, colorbar_img = setup_learning_visualization(
        runner,
        config,
        "Online Learning - Continuous Visualization",
    )

    for img_stamp in runner.img_stamps:
        should_continue = _run_single_feedback_frame(
            runner,
            config,
            feedback_mode,
            lidar_sample,
            proprio_sample,
            dino_sampler,
            salon_sampler,
            memory_backend,
            img_stamp,
            vis,
            colorbar_img,
        )
        if not should_continue:
            break

    _finalize_experiment(runner, root, config, vis)


def _run_dual_feedback_experiment(runner, root, config):
    _validate_dual_memory_paths(config)
    context = init_dual_feedback_context(runner, root, config, build_memory_backend)
    lidar_sample = context.lidar_sample
    proprio_sample = context.proprio_sample
    salon_sampler = context.salon_sampler
    dino_sampler = context.dino_sampler
    mechanical_memory = context.mechanical_memory
    roughness_memory = context.roughness_memory

    vis, colorbar_img = setup_learning_visualization(
        runner,
        config,
        "Dual Cost Online Learning",
        include_bev=True,
    )

    processed_frames = 0
    prev_best_endpoint = None
    prev_command = None
    timing_records = []
    for img_stamp in runner.img_stamps:
        frame_profile = {"stamp": int(img_stamp)}
        frame_start = _profile_now(runner, config.profile_timing)
        stage_start = _profile_now(runner, config.profile_timing)
        print(f"##################处理{img_stamp}帧 dual-cost online learning##################")
        synced_frame, feature_frame = _load_dual_feature_frame(runner, lidar_sample, dino_sampler, img_stamp, config)
        _profile_record(runner, frame_profile, "time_match", stage_start, config.profile_timing)
        if synced_frame is None:
            continue

        _profile_record(runner, frame_profile, "dino_vlad_infer", stage_start, config.profile_timing)
        accumulated_feats_points = feature_frame.accumulated_feats_points
        print(f"累积带特征的点云数量: {len(accumulated_feats_points)}")
        frame_profile["accumulated_points"] = float(len(accumulated_feats_points))
        _profile_record(runner, frame_profile, "feature_accumulation", stage_start, config.profile_timing)

        stage_start = _profile_now(runner, config.profile_timing)
        mechanical_feedback, roughness_feedback, nearest_point_feat, nearest_distance = _estimate_dual_feedbacks(
            runner,
            config,
            proprio_sample,
            salon_sampler,
            synced_frame,
            accumulated_feats_points,
        )
        _profile_record(runner, frame_profile, "feedback_estimation", stage_start, config.profile_timing)

        stage_start = _profile_now(runner, config.profile_timing)
        if nearest_point_feat is not None:
            update_dual_feedback_memories(
                config,
                nearest_point_feat,
                nearest_distance,
                mechanical_memory,
                roughness_memory,
                mechanical_feedback,
                roughness_feedback,
            )
        elif nearest_distance is not None:
            print(f"最近接触特征点距离={nearest_distance:.4f}m，跳过 memory update")
        _profile_record(runner, frame_profile, "memory_update", stage_start, config.profile_timing)

        stage_start = _profile_now(runner, config.profile_timing)
        mechanical_predicted_costs, roughness_predicted_costs = _predict_dual_costs(
            accumulated_feats_points,
            mechanical_memory,
            roughness_memory,
        )
        _profile_record(runner, frame_profile, "gpr_predict_mechanical", stage_start, config.profile_timing)
        _profile_record(runner, frame_profile, "gpr_predict_roughness", stage_start, config.profile_timing)

        stage_start = _profile_now(runner, config.profile_timing)
        risk_map, collision_ratio = _build_dual_risk_map(
            runner,
            accumulated_feats_points,
            mechanical_predicted_costs,
            roughness_predicted_costs,
        )
        _profile_record(runner, frame_profile, "risk_map_build", stage_start, config.profile_timing)

        stage_start = _profile_now(runner, config.profile_timing)
        log_dual_cost_frame(
            runner,
            synced_frame,
            mechanical_feedback,
            roughness_feedback,
            mechanical_predicted_costs,
            roughness_predicted_costs,
            mechanical_memory,
            roughness_memory,
            collision_ratio,
        )
        _profile_record(runner, frame_profile, "log_dual_cost", stage_start, config.profile_timing)

        planner_result, prev_best_endpoint, prev_command = _run_dual_planner(
            runner,
            root,
            config,
            synced_frame,
            accumulated_feats_points,
            mechanical_predicted_costs,
            roughness_predicted_costs,
            risk_map,
            collision_ratio,
            frame_profile,
            prev_best_endpoint,
            prev_command,
        )

        should_continue = _visualize_dual_cost_frame(
            runner,
            config,
            vis,
            colorbar_img,
            synced_frame,
            accumulated_feats_points,
            mechanical_predicted_costs,
            risk_map,
            planner_result,
            frame_profile,
        )
        if not should_continue:
            _print_user_exit()
            break

        _finalize_dual_frame(runner, config, synced_frame, frame_profile, frame_start, timing_records)
        processed_frames += 1
        if config.max_frames is not None and processed_frames >= config.max_frames:
            print(f"达到 max_frames={config.max_frames}，停止 dual-cost 诊断运行。")
            break

    _finalize_experiment(runner, root, config, vis, timing_records=timing_records, dual_cost=True)
