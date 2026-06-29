"""Self-contained MPPI backend used by the optional POLT planning branch."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from planning.types import Diagnostics, GridMap2D, ReferencePath, RobotState


def wrap_angle(angle: float) -> float:
    """Normalize an angle to [-pi, pi]."""
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


@dataclass
class BYFMPPIConfig:
    delta_t: float = 0.1
    wheel_base: float = 2.5
    max_steer_abs: float = 0.523
    max_accel_abs: float = 2.0
    horizon_step_T: int = 60
    number_of_samples_K: int = 512
    param_exploration: float = 0.05
    param_lambda: float = 15000.0
    param_alpha: float = 0.98
    sigma_steer: float = 0.06
    sigma_accel: float = 1.0
    stage_cost_weight: tuple[float, float, float, float] = (300.0, 300.0, 8.0, 35.0)
    terminal_cost_weight: tuple[float, float, float, float] = (450.0, 450.0, 12.0, 50.0)
    risk_weight: float = 0.2
    collision_weight: float = 1.0e10
    moving_average_window: int = 8
    final_smoothing_window: int = 9
    steer_rate_weight: float = 500.0
    accel_rate_weight: float = 20.0
    control_magnitude_weight: float = 1.0
    min_speed: float = 0.0
    max_speed: float = 15.0
    seed: int = 42
    reset_reference_progress_each_call: bool = True
    use_time_indexed_reference: bool = True


@dataclass
class BYFCommand:
    """First control command returned by the MPPI rollout optimization."""

    steering_angle: float
    speed: float
    accel: float


class BYFMPPIPlanner:
    """Lightweight MPPI planner used by POLT's optional dual-cost planning branch."""

    def __init__(self, config: BYFMPPIConfig | None = None):
        self.config = config or BYFMPPIConfig()
        self.dim_x = 4
        self.dim_u = 2
        self.u_prev = np.zeros((self.config.horizon_step_T, self.dim_u), dtype=np.float64)
        self.prev_waypoints_idx = 0
        self.ref_path = np.asarray([[0.0, 0.0, 0.0, self.config.max_speed]], dtype=np.float64)
        self.ref_s = np.zeros((1,), dtype=np.float64)
        self.ref_time_s = np.zeros((1,), dtype=np.float64)
        self.ref_yaw_unwrapped = np.zeros((1,), dtype=np.float64)
        self.risk_map: GridMap2D | None = None
        self.rng = np.random.default_rng(self.config.seed)
        self.sigma = np.array(
            [
                [self.config.sigma_steer, 0.0],
                [0.0, self.config.sigma_accel],
            ],
            dtype=np.float64,
        )
        self.sigma_inv = np.linalg.inv(self.sigma)
        self.param_gamma = self.config.param_lambda * (1.0 - self.config.param_alpha)

    def compute_command(
        self,
        robot_state: RobotState,
        reference_path: ReferencePath,
        risk_map: GridMap2D | None = None,
        num_visualized_samples: int = 12,
    ) -> tuple[BYFCommand, Diagnostics]:
        """Optimize a control sequence for the current state and risk map."""
        start_time = time.perf_counter()
        self.ref_path = self._reference_to_array(reference_path)
        self.risk_map = risk_map
        if self.config.reset_reference_progress_each_call:
            self.prev_waypoints_idx = 0
        else:
            self.prev_waypoints_idx = min(self.prev_waypoints_idx, max(len(self.ref_path) - 1, 0))

        state = np.asarray([robot_state.x, robot_state.y, robot_state.yaw, robot_state.vel], dtype=np.float64)
        control, control_seq, optimal_traj, sampled_trajs, sample_costs = self.calc_control_input(state)
        self.last_control_sequence = control_seq.copy()
        self.last_sample_costs = sample_costs.copy()
        speed_cmd = float(np.clip(state[3] + control[1] * self.config.delta_t, self.config.min_speed, self.config.max_speed))
        command = BYFCommand(
            steering_angle=float(np.clip(control[0], -self.config.max_steer_abs, self.config.max_steer_abs)),
            speed=speed_cmd,
            accel=float(np.clip(control[1], -self.config.max_accel_abs, self.config.max_accel_abs)),
        )

        costs = self.evaluate_trajectory_costs(optimal_traj)
        sorted_idx = np.argsort(sample_costs)
        candidate_indices = sorted_idx[: min(num_visualized_samples, len(sorted_idx))]
        candidate_trajectories = [self._state4_to_state5(sampled_trajs[index]) for index in candidate_indices]
        candidate_weights = self._compute_weights(sample_costs)[candidate_indices]
        diagnostics = Diagnostics(
            reference_cost=costs["reference_cost"],
            collision_cost=costs["collision_cost"],
            risk_cost=costs["map_cost"],
            total_cost=costs["reference_cost"] + costs["collision_cost"] + costs["map_cost"],
            collision_rate=float(np.count_nonzero(sample_costs >= self.config.collision_weight) / max(len(sample_costs), 1)),
            input_error=float(np.sum(np.linalg.norm(control_seq, axis=1))),
            best_trajectory=self._state4_to_state5(optimal_traj),
            candidate_trajectories=candidate_trajectories,
            candidate_weights=candidate_weights,
            roughness_cost=costs["roughness_cost"],
            mechanical_cost=costs["mechanical_cost"],
            map_cost=costs["map_cost"],
            runtime_ms=(time.perf_counter() - start_time) * 1000.0,
        )
        return command, diagnostics

    def calc_control_input(self, state: np.ndarray):
        """Sample MPPI rollouts and update the receding-horizon control sequence."""
        self._get_nearest_waypoint(state[0], state[1], update_prev_idx=True)
        if self.prev_waypoints_idx >= self.ref_path.shape[0] - 1:
            self.prev_waypoints_idx = max(self.ref_path.shape[0] - 2, 0)

        u = self.u_prev.copy()
        epsilon = self._calc_epsilon()
        sample_costs = np.zeros((self.config.number_of_samples_K,), dtype=np.float64)
        sampled_controls = np.zeros(
            (self.config.number_of_samples_K, self.config.horizon_step_T, self.dim_u),
            dtype=np.float64,
        )
        sampled_trajs = np.zeros(
            (self.config.number_of_samples_K, self.config.horizon_step_T, self.dim_x),
            dtype=np.float64,
        )

        biased_count = int((1.0 - self.config.param_exploration) * self.config.number_of_samples_K)
        for sample_idx in range(self.config.number_of_samples_K):
            x = state.copy()
            if sample_idx < biased_count:
                sampled_controls[sample_idx] = u + epsilon[sample_idx]
            else:
                sampled_controls[sample_idx] = epsilon[sample_idx]
            sampled_controls[sample_idx] = self.control_bound(sampled_controls[sample_idx])

            for t_idx in range(self.config.horizon_step_T):
                x = self.update(x, sampled_controls[sample_idx, t_idx])
                sampled_trajs[sample_idx, t_idx] = x
                sample_costs[sample_idx] += self.cost(x, t_idx)
                sample_costs[sample_idx] += self.control_smoothness_cost(sampled_controls[sample_idx], t_idx)
                sample_costs[sample_idx] += self.param_gamma * u[t_idx].T @ self.sigma_inv @ sampled_controls[sample_idx, t_idx]
            sample_costs[sample_idx] += self.terminal_cost(x, self.config.horizon_step_T - 1)

        weights = self._compute_weights(sample_costs)
        weighted_epsilon = np.einsum("k,ktu->tu", weights, epsilon)
        weighted_epsilon = self._moving_average_filter(weighted_epsilon, self.config.moving_average_window)
        u = self.control_bound(u + weighted_epsilon)
        u = self._smooth_control_sequence(u)

        optimal_traj = np.zeros((self.config.horizon_step_T, self.dim_x), dtype=np.float64)
        x = state.copy()
        for t_idx in range(self.config.horizon_step_T):
            x = self.update(x, u[t_idx])
            optimal_traj[t_idx] = x

        self.u_prev[:-1] = u[1:]
        self.u_prev[-1] = u[-1]
        return u[0], u, optimal_traj, sampled_trajs, sample_costs

    def _reference_to_array(self, reference_path: ReferencePath) -> np.ndarray:
        x_values = np.asarray(reference_path.x, dtype=np.float64)
        y_values = np.asarray(reference_path.y, dtype=np.float64)
        v_values = np.asarray(reference_path.v, dtype=np.float64)
        if len(x_values) < 2:
            fallback = np.asarray([[0.0, 0.0, 0.0, self.config.max_speed], [1.0, 0.0, 0.0, self.config.max_speed]])
            self.ref_s = np.asarray([0.0, 1.0], dtype=np.float64)
            self.ref_time_s = np.asarray([0.0, self.config.max_speed * self.config.delta_t], dtype=np.float64)
            self.ref_yaw_unwrapped = fallback[:, 2].copy()
            return fallback
        v_values = np.clip(np.nan_to_num(v_values, nan=self.config.max_speed), self.config.min_speed, self.config.max_speed)

        yaw_values = np.zeros_like(x_values)
        for idx in range(len(x_values)):
            next_idx = min(idx + 1, len(x_values) - 1)
            prev_idx = max(idx - 1, 0)
            yaw_values[idx] = math.atan2(y_values[next_idx] - y_values[prev_idx], x_values[next_idx] - x_values[prev_idx])
        ref_array = np.column_stack((x_values, y_values, yaw_values, v_values))
        segment_lengths = np.linalg.norm(np.diff(ref_array[:, :2], axis=0), axis=1)
        self.ref_s = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        time_progress = np.cumsum(v_values[:-1] * self.config.delta_t)
        self.ref_time_s = np.concatenate(([0.0], time_progress))
        self.ref_yaw_unwrapped = np.unwrap(yaw_values)
        return ref_array

    def _get_time_indexed_reference(self, t_idx: int) -> tuple[float, float, float, float]:
        if len(self.ref_path) == 0:
            return 0.0, 0.0, 0.0, self.config.max_speed
        if len(self.ref_path) == 1 or self.ref_s[-1] <= 1e-6:
            ref = self.ref_path[0]
            return float(ref[0]), float(ref[1]), float(ref[2]), float(ref[3])

        if len(self.ref_time_s) > 1:
            time_idx = min(max(int(t_idx), 0), len(self.ref_time_s) - 1)
            target_s = float(self.ref_time_s[time_idx])
        else:
            base_speed = float(np.clip(self.ref_path[0, 3], self.config.min_speed, self.config.max_speed))
            target_s = float(t_idx) * self.config.delta_t * base_speed
        target_s = min(max(target_s, 0.0), float(self.ref_s[-1]))
        ref_x = float(np.interp(target_s, self.ref_s, self.ref_path[:, 0]))
        ref_y = float(np.interp(target_s, self.ref_s, self.ref_path[:, 1]))
        ref_yaw = float(wrap_angle(np.interp(target_s, self.ref_s, self.ref_yaw_unwrapped)))
        ref_v = float(np.interp(target_s, self.ref_s, self.ref_path[:, 3]))
        return ref_x, ref_y, ref_yaw, ref_v

    def _calc_epsilon(self) -> np.ndarray:
        sample_count = self.config.number_of_samples_K
        zero_sample = np.zeros((1, self.config.horizon_step_T, self.dim_u), dtype=np.float64)
        pair_count = max((sample_count - 1) // 2, 0)
        paired = self.rng.multivariate_normal(
            np.zeros((self.dim_u,), dtype=np.float64),
            self.sigma,
            (pair_count, self.config.horizon_step_T),
        )
        epsilon = np.concatenate((zero_sample, paired, -paired), axis=0)
        remaining = sample_count - epsilon.shape[0]
        if remaining > 0:
            extra = np.zeros((remaining, self.config.horizon_step_T, self.dim_u), dtype=np.float64)
            epsilon = np.concatenate((epsilon, extra), axis=0)
        return epsilon

    def update(self, x_t: np.ndarray, u_t: np.ndarray) -> np.ndarray:
        x, y, yaw, velocity = x_t
        steer, accel = self.control_bound(np.asarray(u_t, dtype=np.float64).copy())
        next_velocity = np.clip(
            velocity + accel * self.config.delta_t,
            self.config.min_speed,
            self.config.max_speed,
        )
        next_x = x + velocity * np.cos(yaw) * self.config.delta_t
        next_y = y + velocity * np.sin(yaw) * self.config.delta_t
        next_yaw = wrap_angle(yaw + velocity / self.config.wheel_base * np.tan(steer) * self.config.delta_t)
        return np.asarray([next_x, next_y, next_yaw, next_velocity], dtype=np.float64)

    def control_bound(self, u: np.ndarray) -> np.ndarray:
        bounded = np.asarray(u, dtype=np.float64).copy()
        bounded[..., 0] = np.clip(bounded[..., 0], -self.config.max_steer_abs, self.config.max_steer_abs)
        bounded[..., 1] = np.clip(bounded[..., 1], -self.config.max_accel_abs, self.config.max_accel_abs)
        return bounded

    def control_smoothness_cost(self, control_seq: np.ndarray, t_idx: int) -> float:
        control = control_seq[t_idx]
        cost = self.config.control_magnitude_weight * float(control @ control)
        if t_idx > 0:
            delta = control_seq[t_idx] - control_seq[t_idx - 1]
            cost += self.config.steer_rate_weight * float(delta[0] ** 2)
            cost += self.config.accel_rate_weight * float(delta[1] ** 2)
        return cost

    def cost(self, x_t: np.ndarray, t_idx: int | None = None) -> float:
        ref_cost = self.reference_cost(x_t, terminal=False, t_idx=t_idx)
        map_cost, _, _, collision_cost = self.risk_cost(x_t)
        return ref_cost + self.config.risk_weight * map_cost + collision_cost

    def terminal_cost(self, x_t: np.ndarray, t_idx: int | None = None) -> float:
        ref_cost = self.reference_cost(x_t, terminal=True, t_idx=t_idx)
        map_cost, _, _, collision_cost = self.risk_cost(x_t)
        return ref_cost + self.config.risk_weight * map_cost + collision_cost

    def reference_cost(self, x_t: np.ndarray, terminal: bool, t_idx: int | None = None) -> float:
        x, y, yaw, velocity = x_t
        if self.config.use_time_indexed_reference and t_idx is not None:
            ref_x, ref_y, ref_yaw, ref_v = self._get_time_indexed_reference(t_idx)
        else:
            _, ref_x, ref_y, ref_yaw, ref_v = self._get_nearest_waypoint(x, y, update_prev_idx=False)
        weights = self.config.terminal_cost_weight if terminal else self.config.stage_cost_weight
        return float(
            weights[0] * (x - ref_x) ** 2
            + weights[1] * (y - ref_y) ** 2
            + weights[2] * wrap_angle(yaw - ref_yaw) ** 2
            + weights[3] * (velocity - ref_v) ** 2
        )

    def risk_cost(self, x_t: np.ndarray) -> tuple[float, float, float, float]:
        """Evaluate map-derived risk at one state."""
        if self.risk_map is None:
            return 0.0, 0.0, 0.0, 0.0
        x_value, y_value = float(x_t[0]), float(x_t[1])
        if not self.risk_map.is_inside(x_value, y_value):
            return 1.0, 0.0, 0.0, self.config.collision_weight

        mechanical = self._layer_at("mechanical_cost", x_value, y_value)
        roughness = self._layer_at("roughness_cost", x_value, y_value)
        map_cost = 0.5 * mechanical + 0.5 * roughness
        collision = 0.0
        if self.risk_map.exists("collision_layer") and self.risk_map.at_position("collision_layer", x_value, y_value) > 0.0:
            collision = self.config.collision_weight
        return map_cost, roughness, mechanical, collision

    def _layer_at(self, layer_name: str, x_value: float, y_value: float) -> float:
        if self.risk_map is None or not self.risk_map.exists(layer_name):
            return 0.0
        return float(np.clip(self.risk_map.at_position(layer_name, x_value, y_value), 0.0, 1.0))

    def evaluate_trajectory_costs(self, trajectory: np.ndarray) -> dict[str, float]:
        reference_cost = 0.0
        map_cost = 0.0
        roughness_cost = 0.0
        mechanical_cost = 0.0
        collision_cost = 0.0
        for t_idx, state in enumerate(trajectory):
            reference_cost += self.reference_cost(state, terminal=False, t_idx=t_idx)
            cur_map, cur_rough, cur_mech, cur_collision = self.risk_cost(state)
            map_cost += self.config.risk_weight * cur_map
            roughness_cost += cur_rough
            mechanical_cost += cur_mech
            collision_cost += cur_collision
        return {
            "reference_cost": float(reference_cost),
            "map_cost": float(map_cost),
            "roughness_cost": float(roughness_cost),
            "mechanical_cost": float(mechanical_cost),
            "collision_cost": float(collision_cost),
        }

    def _get_nearest_waypoint(self, x: float, y: float, update_prev_idx: bool = True):
        search_idx_len = 200
        prev_idx = min(self.prev_waypoints_idx, max(len(self.ref_path) - 1, 0))
        search_path = self.ref_path[prev_idx : min(prev_idx + search_idx_len, len(self.ref_path))]
        if len(search_path) == 0:
            prev_idx = 0
            search_path = self.ref_path
        distances = (search_path[:, 0] - x) ** 2 + (search_path[:, 1] - y) ** 2
        nearest_idx = int(np.argmin(distances)) + prev_idx
        if update_prev_idx:
            self.prev_waypoints_idx = nearest_idx
        ref = self.ref_path[nearest_idx]
        return nearest_idx, float(ref[0]), float(ref[1]), float(ref[2]), float(ref[3])

    def _compute_weights(self, costs: np.ndarray) -> np.ndarray:
        rho = float(np.min(costs))
        exp_costs = np.exp(-(costs - rho) / self.config.param_lambda)
        return exp_costs / (np.sum(exp_costs) + 1e-10)

    def _moving_average_filter(self, values: np.ndarray, window_size: int) -> np.ndarray:
        if window_size <= 1:
            return values
        b = np.ones(window_size, dtype=np.float64) / float(window_size)
        smoothed = np.zeros_like(values)
        for dim in range(values.shape[1]):
            smoothed[:, dim] = np.convolve(values[:, dim], b, mode="same")
            n_conv = math.ceil(window_size / 2)
            smoothed[0, dim] *= window_size / n_conv
            for idx in range(1, n_conv):
                smoothed[idx, dim] *= window_size / (idx + n_conv)
                smoothed[-idx, dim] *= window_size / (idx + n_conv - (window_size % 2))
        return smoothed

    def _smooth_control_sequence(self, control_seq: np.ndarray) -> np.ndarray:
        smoothed = self._moving_average_filter(control_seq, self.config.final_smoothing_window)
        if len(smoothed) > 0:
            smoothed[0] = control_seq[0]
        return self.control_bound(smoothed)

    def _state4_to_state5(self, trajectory: np.ndarray) -> np.ndarray:
        trajectory = np.asarray(trajectory, dtype=np.float64)
        if trajectory.size == 0:
            return np.zeros((0, 5), dtype=np.float64)
        steer = np.zeros((trajectory.shape[0], 1), dtype=np.float64)
        return np.column_stack((trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], trajectory[:, 3], steer[:, 0]))


__all__ = ["BYFCommand", "BYFMPPIConfig", "BYFMPPIPlanner", "wrap_angle"]
