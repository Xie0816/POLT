"""Feedback and traversability cost helpers for POLT."""

import time

import numpy as np
import torch

from common_struct import INFER_SLIP_LIMIT, LIDAR_H, PROPRIO_WINDOWS, RESOLUTION


def extract_imu_vector(proprio_frame):
    """Extract the IMU vector used by the roughness feedback estimator."""
    imu_data = proprio_frame.get("imu", {})
    imu_keys = [
        "angular_velocity_x",
        "angular_velocity_y",
        "angular_velocity_z",
        "linear_acceleration_x",
        "linear_acceleration_y",
        "linear_acceleration_z",
    ]
    if any(key not in imu_data for key in imu_keys):
        return None
    return np.asarray([imu_data[key] for key in imu_keys], dtype=np.float64)


def estimate_mechanical_feedback(proprio_sample, proprio_idx):
    """Estimate force/mechanical traversability feedback from proprioception."""
    s = proprio_sample.compute_slip(proprio_idx)
    proprio_temporal_feats = proprio_sample._prepare_temporal_feats(proprio_idx)
    if s > INFER_SLIP_LIMIT and s < 0.4 and len(proprio_temporal_feats) >= PROPRIO_WINDOWS:
        cost = proprio_sample.model_infer(proprio_temporal_feats)
        return float(np.clip(cost, 0.0, 1.0))
    return None


def estimate_roughness_feedback(runner, proprio_sample, salon_sampler, proprio_idx, imu_window=10, min_speed=3.0):
    """Estimate terrain roughness feedback from recent IMU samples."""
    start_idx = max(0, runner._last_salon_proprio_idx + 1, proprio_idx - imu_window + 1)
    roughness_cost = None
    inserted = 0
    for frame_idx in range(start_idx, proprio_idx + 1):
        frame_stamp = proprio_sample.tss[frame_idx]
        imu_vec = extract_imu_vector(proprio_sample.proprio_data[frame_stamp])
        if imu_vec is None:
            return None
        v_local, _ = proprio_sample._extract_globalpose_data(frame_idx)
        speed = float(np.hypot(v_local[0], v_local[1]))
        if speed < min_speed:
            continue
        roughness_cost = salon_sampler.calculate_cost(imu_vec, velocity=speed)
        inserted += 1

    runner._last_salon_proprio_idx = max(runner._last_salon_proprio_idx, proprio_idx)
    warmup_count = int(np.count_nonzero(salon_sampler.bufferY.data))
    if inserted == 0 or warmup_count < salon_sampler.buffer_size:
        return None
    return float(np.clip(roughness_cost, 0.0, 1.0))


def nearest_feature_to_contact(device, accumulated_feats_points):
    """Find the feature vector nearest to the robot-ground contact proxy."""
    if accumulated_feats_points is None or len(accumulated_feats_points) == 0:
        return None, None
    proprio_pose = torch.tensor([0.0, 0.0, -LIDAR_H], device=device)
    distances = torch.norm(accumulated_feats_points[:, :3] - proprio_pose.unsqueeze(0), dim=1)
    min_distance, min_index = torch.min(distances, dim=0)
    if min_distance >= RESOLUTION * 2:
        return None, float(min_distance)
    return accumulated_feats_points[min_index, 3:], float(min_distance)


def make_proprio_history_point(device, cost, color_fn):
    """Create a colored contact point for visualization history."""
    color = color_fn([cost])[0]
    proprio_color = (color * 255).astype(np.uint8)
    proprio_pose = torch.tensor([0.0, 0.0, -LIDAR_H], device=device)
    proprio_rgb = torch.tensor(proprio_color, device=device)
    return proprio_pose, torch.cat([proprio_pose, proprio_rgb])


def accumulate_proprio_history(runner, lidar_sample, odom):
    """Project historical contact points into the current LiDAR frame."""
    if len(runner.proprio_color_history) == 0:
        return None
    return lidar_sample.projection_accumulation(
        runner.proprio_color_history,
        runner.odom_data,
        odom,
        accumulate_gap=1,
        voxel_size=None,
    )


def predict_costs_for_points(memory_sampler, accumulated_feats_points, label):
    """Run batch memory inference for all accumulated feature points."""
    if accumulated_feats_points is None or len(accumulated_feats_points) == 0:
        return []
    feat_vectors = accumulated_feats_points[:, 3:]
    semantic_features_batch = feat_vectors.detach().cpu().numpy()
    start_time = time.perf_counter()
    gpr_results = memory_sampler.predict_cost_batch(semantic_features_batch, batch_size=50000)
    predicted_costs = [
        {
            "point_index": i,
            "semantic_features": semantic_features_batch[i],
            "predicted_cost": gpr_result["mu"],
            "predicted_variance": gpr_result["var"],
        }
        for i, gpr_result in enumerate(gpr_results)
    ]
    inference_time = time.perf_counter() - start_time
    costs = np.asarray([item["predicted_cost"] for item in predicted_costs], dtype=np.float64)
    valid_costs = costs[np.isfinite(costs) & (costs >= 0.0)]
    if valid_costs.size > 0:
        print(
            f"{label} 批量GPR推理完成: {len(predicted_costs)}个点, 耗时: {inference_time:.4f}秒, "
            f"cost[min/mean/max]={valid_costs.min():.4f}/{valid_costs.mean():.4f}/{valid_costs.max():.4f}"
        )
    else:
        print(f"{label} 批量GPR推理完成: {len(predicted_costs)}个点, 耗时: {inference_time:.4f}秒, 无有效预测")
    return predicted_costs


def prediction_mean(predicted_costs):
    """Compute the mean valid predicted cost."""
    if not predicted_costs:
        return None
    costs = np.asarray([item["predicted_cost"] for item in predicted_costs], dtype=np.float64)
    valid = costs[np.isfinite(costs) & (costs >= 0.0)]
    return float(np.mean(valid)) if valid.size > 0 else None


def prediction_min_max(predicted_costs):
    """Compute the valid predicted cost range."""
    if not predicted_costs:
        return None, None
    costs = np.asarray([item["predicted_cost"] for item in predicted_costs], dtype=np.float64)
    valid = costs[np.isfinite(costs) & (costs >= 0.0)]
    if valid.size == 0:
        return None, None
    return float(np.min(valid)), float(np.max(valid))


def memory_nodes_num(memory_sampler):
    """Count nodes in the hierarchical memory backend."""
    return int(sum(len(nodes) for nodes in memory_sampler.hierarchical_memory.values()))
