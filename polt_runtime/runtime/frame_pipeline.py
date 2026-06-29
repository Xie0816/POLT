"""Reusable per-frame operations shared by single and dual feedback modes."""

import torch

from common_struct import RESOLUTION
from polt_runtime import perception, traversability, visualization


def load_feature_frame(runner, lidar_sample, dino_sampler, img_stamp, use_vlad, voxel_size, synced_frame=None):
    """Load synchronized sensors and build the current feature frame."""
    if synced_frame is None:
        synced_frame = perception.load_synced_frame(runner, lidar_sample, img_stamp)
    if synced_frame is None:
        return None, None
    feature_frame = perception.build_feature_frame(
        runner,
        lidar_sample,
        dino_sampler,
        synced_frame,
        use_vlad,
        voxel_size=voxel_size,
    )
    return synced_frame, feature_frame


def update_single_feedback_memory(
    runner,
    config,
    synced_frame,
    accumulated_feats_points,
    memory_backend,
    feedback_cost,
):
    """Update or query the single feedback memory from the nearest contact feature."""
    if feedback_cost is None:
        return None

    color_fn = visualization.rough_cost_to_color if config.colorbar_kind == "roughness" else visualization.cost_to_color
    proprio_pose, proprio_xyzrgb = traversability.make_proprio_history_point(runner.device, feedback_cost, color_fn)
    runner.proprio_color_history[synced_frame.lidar_stamp] = proprio_xyzrgb

    if len(accumulated_feats_points) == 0:
        return None

    proprio_pose_tensor = proprio_pose.unsqueeze(0)
    feat_points_xyz = accumulated_feats_points[:, :3]
    distances = torch.norm(feat_points_xyz - proprio_pose_tensor, dim=1)
    min_distance, min_index = torch.min(distances, dim=0)
    if min_distance >= RESOLUTION:
        return None

    nearest_point_feat = accumulated_feats_points[min_index, 3:]
    if config.update_mode == "online":
        pred_info = memory_backend.update(nearest_point_feat, feedback_cost)
    else:
        pred_info = {"gpr_prediction": memory_backend.predict_point(nearest_point_feat), "train_info": {}}

    runner.log[int(synced_frame.lidar_stamp)] = {
        "gpr_cost": float(pred_info["gpr_prediction"]["mu"]),
        "proprio_cost": float(feedback_cost),
        "error": abs(float(feedback_cost) - float(pred_info["gpr_prediction"]["mu"])),
        "nodes_num": memory_backend.nodes_num(),
        "train_time": float(pred_info.get("train_info", {}).get("duration", 0)),
        "train_epochs": int(pred_info.get("train_info", {}).get("epochs", 0)),
    }
    return {
        "pred_info": pred_info,
        "min_distance": float(min_distance),
    }


def update_dual_feedback_memories(
    config,
    nearest_point_feat,
    nearest_distance,
    mechanical_memory,
    roughness_memory,
    mechanical_feedback,
    roughness_feedback,
):
    """Update/query both mechanical and roughness memories at one feature point."""
    if nearest_point_feat is None:
        return
    if mechanical_feedback is not None:
        if config.update_mode == "online":
            mechanical_memory.update(nearest_point_feat, mechanical_feedback)
        else:
            mechanical_memory.predict_point(nearest_point_feat)
        print(f"mechanical_memory 更新: cost={mechanical_feedback:.4f}, distance={nearest_distance:.4f}m")
    if roughness_feedback is not None:
        if config.update_mode == "online":
            roughness_memory.update(nearest_point_feat, roughness_feedback)
        else:
            roughness_memory.predict_point(nearest_point_feat)
        print(f"roughness_memory 更新: cost={roughness_feedback:.4f}, distance={nearest_distance:.4f}m")


def log_dual_cost_frame(
    runner,
    synced_frame,
    mechanical_feedback,
    roughness_feedback,
    mechanical_predicted_costs,
    roughness_predicted_costs,
    mechanical_memory,
    roughness_memory,
    collision_ratio,
):
    """Store per-frame dual-cost statistics in the runtime log."""
    mechanical_pred_min, mechanical_pred_max = traversability.prediction_min_max(mechanical_predicted_costs)
    roughness_pred_min, roughness_pred_max = traversability.prediction_min_max(roughness_predicted_costs)
    runner.log.setdefault(int(synced_frame.lidar_stamp), {})
    runner.log[int(synced_frame.lidar_stamp)]["dual_cost"] = {
        "mechanical_feedback": mechanical_feedback,
        "roughness_feedback": roughness_feedback,
        "mechanical_pred_mean": traversability.prediction_mean(mechanical_predicted_costs),
        "roughness_pred_mean": traversability.prediction_mean(roughness_predicted_costs),
        "mechanical_pred_min": mechanical_pred_min,
        "mechanical_pred_max": mechanical_pred_max,
        "roughness_pred_min": roughness_pred_min,
        "roughness_pred_max": roughness_pred_max,
        "mechanical_nodes_num": mechanical_memory.nodes_num(),
        "roughness_nodes_num": roughness_memory.nodes_num(),
        "collision_ratio": collision_ratio,
    }


def record_timing_log(runner, synced_frame, frame_profile):
    """Attach profiling measurements to the frame log entry."""
    runner.log.setdefault(int(synced_frame.lidar_stamp), {})
    runner.log[int(synced_frame.lidar_stamp)]["timing_ms"] = {
        key: float(value)
        for key, value in frame_profile.items()
        if key not in {"stamp", "accumulated_points"}
    }
    runner.log[int(synced_frame.lidar_stamp)]["profile_points"] = {
        "accumulated_points": float(frame_profile.get("accumulated_points", 0.0))
    }
