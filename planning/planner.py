"""Planning utilities for the optional dual-cost POLT experiment.

This module converts POLT frame predictions into local risk maps, builds a
short reference path, and calls the local MPPI backend in ``planning.mppi``.
"""

from pathlib import Path

import numpy as np

from planning.mppi import BYFMPPIConfig, BYFMPPIPlanner
from planning.types import GridMap2D, ReferenceMapBuilder, ReferencePath, ReferencePathConfig, RobotState


def quat_wxyz_to_rotmat(quat):
    """Convert a wxyz quaternion from odometry into a rotation matrix."""
    quat = np.asarray(quat, dtype=np.float64)
    w, x, y, z = quat
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ], dtype=np.float64)


def quat_wxyz_to_yaw(quat):
    """Extract planar yaw from a wxyz quaternion."""
    quat = np.asarray(quat, dtype=np.float64)
    w, x, y, z = quat
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def estimate_speed(runner, stamp_index):
    """Estimate local speed from consecutive LiDAR odometry poses."""
    if stamp_index <= 0 or stamp_index >= len(runner.lidar_odom_stamps):
        return 0.0
    current_stamp = runner.lidar_odom_stamps[stamp_index]
    previous_stamp = runner.lidar_odom_stamps[stamp_index - 1]
    current_pose = runner.lidar_odom_files[current_stamp][:3].detach().cpu().numpy()
    previous_pose = runner.lidar_odom_files[previous_stamp][:3].detach().cpu().numpy()
    dt = max((current_stamp - previous_stamp) / 1000.0, 1e-3)
    return float(np.linalg.norm(current_pose - previous_pose) / dt)


def build_reference_path_from_future_odom(runner, odom_idx, horizon_points, step_stride, max_length):
    """Build a debug-only reference from future odometry samples."""
    print("WARNING: using future odom as debug reference, not valid for online planning evaluation.")
    if odom_idx is None or odom_idx >= len(runner.lidar_odom_stamps):
        return None
    current_stamp = runner.lidar_odom_stamps[odom_idx]
    current_odom = runner.lidar_odom_files[current_stamp]
    current_pose = current_odom[:3].detach().cpu().numpy()
    current_quat = current_odom[3:].detach().cpu().numpy()
    current_rot = quat_wxyz_to_rotmat(current_quat)

    future_stamps = runner.lidar_odom_stamps[
        odom_idx : min(len(runner.lidar_odom_stamps), odom_idx + horizon_points * step_stride) : step_stride
    ]
    if len(future_stamps) < 3:
        return None

    local_points = []
    speeds = []
    previous_global_pose = None
    previous_stamp = None
    for stamp in future_stamps:
        odom = runner.lidar_odom_files[stamp]
        global_pose = odom[:3].detach().cpu().numpy()
        delta = global_pose - current_pose
        local_pose = delta @ current_rot
        if len(local_points) >= 3 and np.linalg.norm(local_pose[:2]) > max_length:
            break
        local_points.append(local_pose[:2])

        if previous_global_pose is None:
            speeds.append(estimate_speed(runner, odom_idx))
        else:
            dt = max((stamp - previous_stamp) / 1000.0, 1e-3)
            speeds.append(float(np.linalg.norm(global_pose - previous_global_pose) / dt))
        previous_global_pose = global_pose
        previous_stamp = stamp

    if len(local_points) < 3:
        return None

    local_points = np.asarray(local_points, dtype=np.float64)
    speeds = np.asarray(speeds, dtype=np.float64)
    if np.allclose(speeds, 0.0):
        speeds[:] = 1.0
    return ReferencePath(x=local_points[:, 0], y=local_points[:, 1], v=speeds)


def build_reference_path_local_goal(length, interval, speed):
    """Build a straight local reference path in the robot frame."""
    xs = np.arange(0.0, length + interval, interval, dtype=np.float64)
    ys = np.zeros_like(xs)
    speeds = np.full_like(xs, speed, dtype=np.float64)
    return ReferencePath(x=xs, y=ys, v=speeds)


def clip_reference_path_by_length(reference_path, max_length):
    """Clip a reference path to the requested arc length."""
    if reference_path is None or len(reference_path.x) < 2:
        return reference_path
    xy = np.column_stack((reference_path.x, reference_path.y))
    segment_lengths = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    max_length = float(max(max_length, 0.0))
    if max_length >= cumulative[-1]:
        return reference_path

    end_idx = int(np.searchsorted(cumulative, max_length, side="right"))
    end_idx = max(1, min(end_idx, len(reference_path.x) - 1))
    clipped_x = list(reference_path.x[:end_idx])
    clipped_y = list(reference_path.y[:end_idx])
    clipped_v = list(reference_path.v[:end_idx])
    prev_s = cumulative[end_idx - 1]
    next_s = cumulative[end_idx]
    ratio = 0.0 if next_s <= prev_s else (max_length - prev_s) / (next_s - prev_s)
    clipped_x.append(float(reference_path.x[end_idx - 1] + ratio * (reference_path.x[end_idx] - reference_path.x[end_idx - 1])))
    clipped_y.append(float(reference_path.y[end_idx - 1] + ratio * (reference_path.y[end_idx] - reference_path.y[end_idx - 1])))
    clipped_v.append(float(reference_path.v[end_idx - 1] + ratio * (reference_path.v[end_idx] - reference_path.v[end_idx - 1])))
    return ReferencePath(x=np.asarray(clipped_x), y=np.asarray(clipped_y), v=np.asarray(clipped_v))


def reference_path_length(reference_path):
    """Return the cumulative 2D length of a reference path."""
    if reference_path is None or len(reference_path.x) < 2:
        return 0.0
    xy = np.column_stack((reference_path.x, reference_path.y))
    return float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))


def reference_path_to_array(reference_path):
    if reference_path is None:
        return np.zeros((0, 3), dtype=np.float64)
    return np.column_stack((reference_path.x, reference_path.y, reference_path.v)).astype(np.float64)


def save_planner_trace(runner, root, lidar_stamp, planner_result):
    if planner_result is None:
        return None
    diagnostics = planner_result.get("diagnostics")
    if diagnostics is None:
        return None

    trace_dir = Path("planner_traces") / Path(root).name
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / f"{int(lidar_stamp)}.npz"

    candidate_trajectories = getattr(diagnostics, "candidate_trajectories", None) or []
    if candidate_trajectories:
        candidates = np.asarray(candidate_trajectories, dtype=np.float64)
    else:
        candidates = np.zeros((0, 0, 5), dtype=np.float64)
    candidate_weights = np.asarray(getattr(diagnostics, "candidate_weights", []), dtype=np.float64)

    np.savez_compressed(
        trace_path,
        reference_path=reference_path_to_array(planner_result.get("reference_path")),
        visual_reference_path=reference_path_to_array(planner_result.get("visual_reference_path")),
        reachable_reference_path=reference_path_to_array(planner_result.get("reachable_reference_path")),
        best_trajectory=np.asarray(getattr(diagnostics, "best_trajectory", []), dtype=np.float64),
        candidate_trajectories=candidates,
        candidate_weights=candidate_weights,
        control_sequence=np.asarray(getattr(runner.mppi_planner, "last_control_sequence", []), dtype=np.float64),
        sample_costs=np.asarray(getattr(runner.mppi_planner, "last_sample_costs", []), dtype=np.float64),
    )
    return str(trace_path)


def build_dense_bev_cost_map_from_points(points_xyz, point_costs, grid_size=1000, resolution=0.2, ego_center=True, max_fill_distance=2.0, default_cost=0.5, return_valid_mask=False):
    points_xyz = np.asarray(points_xyz, dtype=np.float64)
    point_costs = np.asarray(point_costs, dtype=np.float64).reshape(-1)
    dense_map = np.full((grid_size, grid_size), float(default_cost), dtype=np.float64)
    valid_map = np.zeros((grid_size, grid_size), dtype=np.float64)
    if points_xyz.size == 0 or point_costs.size == 0:
        return (dense_map, valid_map) if return_valid_mask else dense_map

    valid_length = min(len(points_xyz), len(point_costs))
    points_xyz = points_xyz[:valid_length]
    point_costs = np.clip(point_costs[:valid_length], 0.0, 1.0)

    half = grid_size // 2 if ego_center else 0
    ix = np.floor(points_xyz[:, 0] / resolution).astype(np.int64) + half
    iy = np.floor(points_xyz[:, 1] / resolution).astype(np.int64) + half
    valid = (ix >= 0) & (ix < grid_size) & (iy >= 0) & (iy < grid_size) & np.isfinite(point_costs)
    if not np.any(valid):
        return (dense_map, valid_map) if return_valid_mask else dense_map

    ix = ix[valid]
    iy = iy[valid]
    costs = point_costs[valid]
    cost_sum = np.zeros((grid_size, grid_size), dtype=np.float64)
    counts = np.zeros((grid_size, grid_size), dtype=np.float64)
    np.add.at(cost_sum, (ix, iy), costs)
    np.add.at(counts, (ix, iy), 1.0)
    observed = counts > 0
    dense_map[observed] = cost_sum[observed] / counts[observed]
    valid_map[observed] = 1.0

    try:
        from scipy.ndimage import distance_transform_edt

        distances, nearest_indices = distance_transform_edt(~observed, return_indices=True)
        fill_mask = (~observed) & ((distances * resolution) <= max_fill_distance)
        dense_map[fill_mask] = dense_map[nearest_indices[0][fill_mask], nearest_indices[1][fill_mask]]
        valid_map[fill_mask] = 1.0
    except Exception as exc:
        print(f"WARNING: nearest-neighbor BEV fill skipped: {exc}")

    dense_map = np.clip(dense_map, 0.0, 1.0)
    return (dense_map, valid_map) if return_valid_mask else dense_map


def predicted_cost_array(predicted_costs):
    """Normalize prediction records into a finite cost array."""
    costs = np.asarray([result["predicted_cost"] for result in predicted_costs], dtype=np.float64)
    costs[~np.isfinite(costs)] = 0.5
    costs[costs < 0.0] = 0.5
    return costs


def build_dual_cost_risk_map_from_predictions(accumulated_feats_points, mechanical_predicted_costs, roughness_predicted_costs, grid_size=1000, resolution=0.2, default_cost=0.5, collision_threshold=0.7, collision_inflation_m=0.4):
    """Rasterize mechanical and roughness predictions into a layered BEV risk map."""
    if accumulated_feats_points is None or len(accumulated_feats_points) == 0 or len(mechanical_predicted_costs) == 0 or len(roughness_predicted_costs) == 0:
        return None

    points_xyz = accumulated_feats_points[:, :3].detach().cpu().numpy()
    mechanical_costs = predicted_cost_array(mechanical_predicted_costs)
    roughness_costs = predicted_cost_array(roughness_predicted_costs)
    valid_length = min(len(points_xyz), len(mechanical_costs), len(roughness_costs))
    if valid_length == 0:
        return None

    points_xyz = points_xyz[:valid_length]
    mechanical_costs = mechanical_costs[:valid_length]
    roughness_costs = roughness_costs[:valid_length]
    mechanical_cost_map, mechanical_valid = build_dense_bev_cost_map_from_points(points_xyz, mechanical_costs, grid_size=grid_size, resolution=resolution, default_cost=default_cost, return_valid_mask=True)
    roughness_cost_map, roughness_valid = build_dense_bev_cost_map_from_points(points_xyz, roughness_costs, grid_size=grid_size, resolution=resolution, default_cost=default_cost, return_valid_mask=True)
    fused_cost = np.clip(0.5 * mechanical_cost_map + 0.5 * roughness_cost_map, 0.0, 1.0)
    valid_mask = np.maximum(mechanical_valid, roughness_valid)
    collision_layer = ((fused_cost >= float(collision_threshold)) & (valid_mask > 0.0)).astype(np.float64)
    if collision_inflation_m > 0.0 and np.any(collision_layer > 0.0):
        try:
            from scipy.ndimage import binary_dilation

            iterations = max(int(np.ceil(float(collision_inflation_m) / float(resolution))), 1)
            collision_layer = binary_dilation(collision_layer > 0.0, iterations=iterations).astype(np.float64)
        except Exception as exc:
            print(f"WARNING: collision_layer inflation skipped: {exc}")
    return GridMap2D(
        layers={
            "mechanical_cost": mechanical_cost_map,
            "roughness_cost": roughness_cost_map,
            "fused_cost": fused_cost,
            "valid_mask": valid_mask,
            "collision_layer": collision_layer,
        },
        resolution=resolution,
        center_x=0.0,
        center_y=0.0,
    )


def build_risk_map_from_predictions(accumulated_feats_points, predicted_costs, map_width=20.0, map_resolution=0.2):
    """Build a single-cost risk map through the dual-cost map representation."""
    grid_size = int(round(map_width / map_resolution))
    return build_dual_cost_risk_map_from_predictions(
        accumulated_feats_points,
        mechanical_predicted_costs=predicted_costs,
        roughness_predicted_costs=predicted_costs,
        grid_size=grid_size,
        resolution=map_resolution,
    )


def init_mppi_planner(runner, planner_reference_max_length_m, planner_horizon_step_t):
    """Initialize the local MPPI backend and its reference-map configuration."""
    if runner.mppi_planner is not None:
        return runner.mppi_planner

    planner_config = BYFMPPIConfig(
        delta_t=0.1,
        wheel_base=2.5,
        max_steer_abs=0.523,
        max_accel_abs=2.0,
        horizon_step_T=planner_horizon_step_t,
        number_of_samples_K=512,
        param_exploration=0.05,
        param_lambda=15000.0,
        param_alpha=0.98,
        sigma_steer=0.06,
        sigma_accel=1.0,
        stage_cost_weight=(300.0, 300.0, 8.0, 35.0),
        terminal_cost_weight=(450.0, 450.0, 12.0, 50.0),
        risk_weight=0.2,
        collision_weight=1.0e10,
        moving_average_window=8,
        final_smoothing_window=9,
        steer_rate_weight=500.0,
        accel_rate_weight=20.0,
        control_magnitude_weight=1.0,
        min_speed=0.0,
        max_speed=15.0,
        seed=42,
    )
    runner.planner_reference_config = ReferencePathConfig(
        num_waypoints=120,
        backward_margin_num=0,
        waypoint_interval=0.5,
        ref_path_map_resolution=0.2,
        ref_path_map_width=planner_reference_max_length_m * 2.0,
        ref_path_map_height=20.0,
        reference_speed_scale=1.0,
        max_speed=15.0,
    )
    runner.mppi_planner = BYFMPPIPlanner(planner_config)
    return runner.mppi_planner


def run_mppi_planner(runner, odom_idx, accumulated_feats_points, mechanical_predicted_costs, roughness_predicted_costs=None, prebuilt_risk_map=None, reference_mode="future_odom_debug", planner_reference_max_length_m=50.0, planner_horizon_step_t=60):
    """Run one MPPI planning step from POLT cost predictions."""
    planner = init_mppi_planner(runner, planner_reference_max_length_m, planner_horizon_step_t)
    if reference_mode == "future_odom_debug":
        reference_path = build_reference_path_from_future_odom(runner, odom_idx, horizon_points=360, step_stride=1, max_length=planner_reference_max_length_m)
    elif reference_mode == "local_goal":
        reference_path = build_reference_path_local_goal(speed=max(estimate_speed(runner, odom_idx), 1.0), length=planner_reference_max_length_m, interval=0.3)
    else:
        raise ValueError(f"Unsupported planner_reference_mode: {reference_mode}")
    reference_path = clip_reference_path_by_length(reference_path, planner_reference_max_length_m)

    if prebuilt_risk_map is not None:
        risk_map = prebuilt_risk_map
    elif roughness_predicted_costs is None:
        risk_map = build_risk_map_from_predictions(accumulated_feats_points, mechanical_predicted_costs)
    else:
        risk_map = build_dual_cost_risk_map_from_predictions(accumulated_feats_points, mechanical_predicted_costs, roughness_predicted_costs)
    if reference_path is None or risk_map is None:
        return None

    reference_builder = ReferenceMapBuilder(reference_path, runner.planner_reference_config)
    robot_state = RobotState(x=0.0, y=0.0, yaw=0.0, vel=estimate_speed(runner, odom_idx), steer=0.0)
    reference_map = reference_builder.build(robot_state)
    command, diagnostics = planner.compute_command(robot_state=robot_state, reference_path=reference_path, risk_map=risk_map)
    reachable_ref_length = min(
        reference_path_length(reference_path),
        max(robot_state.vel, float(reference_path.v[0]) if len(reference_path.v) > 0 else 0.0, 0.0) * planner.config.delta_t * planner.config.horizon_step_T,
    )
    reachable_reference_path = clip_reference_path_by_length(reference_path, reachable_ref_length)
    return {
        "command": command,
        "diagnostics": diagnostics,
        "risk_map": risk_map,
        "reference_path": reference_path,
        "visual_reference_path": reference_path,
        "reachable_reference_path": reachable_reference_path,
        "reference_map": reference_map,
        "planner_backend": "planning.mppi",
        "reference_mode": reference_mode,
    }
