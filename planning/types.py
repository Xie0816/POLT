"""Typed data containers and helpers for the optional planning pipeline."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass
class GridMap2D:
    """Small layered 2D grid used by the POLT planning branch."""

    layers: dict[str, np.ndarray]
    resolution: float
    center_x: float = 0.0
    center_y: float = 0.0

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("GridMap2D requires at least one layer.")
        first_shape = next(iter(self.layers.values())).shape
        if len(first_shape) != 2:
            raise ValueError("GridMap2D layers must be 2D arrays.")
        for name, layer in self.layers.items():
            if layer.shape != first_shape:
                raise ValueError(f"Layer '{name}' shape {layer.shape} does not match {first_shape}.")
            self.layers[name] = np.asarray(layer, dtype=np.float64)
        self.size_x, self.size_y = first_shape
        self.length_x = self.size_x * self.resolution
        self.length_y = self.size_y * self.resolution
        self.min_x = self.center_x - self.length_x / 2.0
        self.min_y = self.center_y - self.length_y / 2.0

    @classmethod
    def empty(
        cls,
        layer_names: Iterable[str],
        size_x: int,
        size_y: int,
        resolution: float,
        center_x: float = 0.0,
        center_y: float = 0.0,
        fill_value: float = 0.0,
    ) -> "GridMap2D":
        layers = {name: np.full((size_x, size_y), fill_value, dtype=np.float64) for name in layer_names}
        return cls(layers=layers, resolution=resolution, center_x=center_x, center_y=center_y)

    def copy(self) -> "GridMap2D":
        return GridMap2D(
            layers={name: values.copy() for name, values in self.layers.items()},
            resolution=self.resolution,
            center_x=self.center_x,
            center_y=self.center_y,
        )

    def exists(self, layer_name: str) -> bool:
        return layer_name in self.layers

    def is_inside(self, x: float, y: float) -> bool:
        return self.min_x <= x < self.min_x + self.length_x and self.min_y <= y < self.min_y + self.length_y

    def position_to_index(self, x: float, y: float) -> tuple[int, int]:
        ix = int(np.floor((x - self.min_x) / self.resolution))
        iy = int(np.floor((y - self.min_y) / self.resolution))
        return ix, iy

    def at_position(self, layer_name: str, x: float, y: float) -> float:
        if layer_name not in self.layers:
            raise KeyError(f"Layer '{layer_name}' does not exist.")
        if not self.is_inside(x, y):
            raise ValueError(f"Position ({x}, {y}) is outside the map.")
        ix, iy = self.position_to_index(x, y)
        ix = min(max(ix, 0), self.size_x - 1)
        iy = min(max(iy, 0), self.size_y - 1)
        return float(self.layers[layer_name][ix, iy])

    def cell_centers(self) -> tuple[np.ndarray, np.ndarray]:
        xs = self.min_x + (np.arange(self.size_x) + 0.5) * self.resolution
        ys = self.min_y + (np.arange(self.size_y) + 0.5) * self.resolution
        return np.meshgrid(xs, ys, indexing="ij")


@dataclass
class ReferencePath:
    """Local reference path consumed by the MPPI planner."""

    x: np.ndarray
    y: np.ndarray
    v: np.ndarray

    @staticmethod
    def from_csv(
        path: str | Path,
        x_label: str = "opt_x",
        y_label: str = "opt_y",
        v_label: str = "ref_v",
    ) -> "ReferencePath":
        with open(path, "r", encoding="utf-8") as file_obj:
            reader = csv.DictReader(file_obj)
            x_values = []
            y_values = []
            v_values = []
            for row in reader:
                x_values.append(float(row[x_label]))
                y_values.append(float(row[y_label]))
                v_values.append(float(row[v_label]))
        return ReferencePath(np.asarray(x_values), np.asarray(y_values), np.asarray(v_values))


@dataclass
class ReferencePathConfig:
    num_waypoints: int = 80
    backward_margin_num: int = 0
    waypoint_interval: float = 0.1
    ref_path_map_resolution: float = 0.1
    ref_path_map_width: float = 20.0
    ref_path_map_height: float = 20.0
    reference_speed_scale: float = 1.0
    max_speed: float = 100.0
    distance_field_layer_name: str = "distance_field"
    angle_field_layer_name: str = "angle_field"
    speed_field_layer_name: str = "speed_field"


@dataclass
class RobotState:
    x: float
    y: float
    yaw: float
    vel: float
    steer: float = 0.0

    def as_array(self) -> list[float]:
        return [self.x, self.y, self.yaw, self.vel, self.steer]


@dataclass
class Diagnostics:
    reference_cost: float
    collision_cost: float
    risk_cost: float
    total_cost: float
    collision_rate: float
    input_error: float
    best_trajectory: Any
    candidate_trajectories: Any
    candidate_weights: Any
    roughness_cost: float = 0.0
    mechanical_cost: float = 0.0
    map_cost: float = 0.0
    runtime_ms: float = 0.0


class ReferenceMapBuilder:
    """Build distance, heading, and speed fields around the current robot pose."""

    def __init__(self, reference_path: ReferencePath, config: ReferencePathConfig | None = None):
        self.reference_path = reference_path
        self.config = config or ReferencePathConfig()
        self.path_xy = np.column_stack((self.reference_path.x, self.reference_path.y))

    def build(self, robot_state: RobotState) -> GridMap2D:
        waypoints = self._calc_waypoints(robot_state)
        size_x = int(round(self.config.ref_path_map_width / self.config.ref_path_map_resolution))
        size_y = int(round(self.config.ref_path_map_height / self.config.ref_path_map_resolution))
        grid_map = GridMap2D.empty(
            layer_names=[
                self.config.distance_field_layer_name,
                self.config.angle_field_layer_name,
                self.config.speed_field_layer_name,
            ],
            size_x=size_x,
            size_y=size_y,
            resolution=self.config.ref_path_map_resolution,
            center_x=robot_state.x,
            center_y=robot_state.y,
        )
        xs, ys = grid_map.cell_centers()
        waypoint_xy = np.asarray([[wp["x"], wp["y"]] for wp in waypoints], dtype=np.float64)
        waypoint_yaw = np.asarray([wp["yaw"] for wp in waypoints], dtype=np.float64)
        waypoint_vel = np.asarray([wp["vel"] for wp in waypoints], dtype=np.float64)

        for ix in range(grid_map.size_x):
            for iy in range(grid_map.size_y):
                position = np.array([xs[ix, iy], ys[ix, iy]])
                distances = np.linalg.norm(waypoint_xy - position, axis=1)
                nearest_index = int(np.argmin(distances))
                grid_map.layers[self.config.distance_field_layer_name][ix, iy] = distances[nearest_index]
                grid_map.layers[self.config.angle_field_layer_name][ix, iy] = waypoint_yaw[nearest_index]
                grid_map.layers[self.config.speed_field_layer_name][ix, iy] = min(
                    waypoint_vel[nearest_index] * self.config.reference_speed_scale,
                    self.config.max_speed,
                )
        return grid_map

    def _calc_waypoints(self, robot_state: RobotState) -> list[dict[str, float]]:
        nearest_idx = self._find_nearest_index(robot_state)
        current_idx = max(nearest_idx - self.config.backward_margin_num, 0)
        waypoints = [self._get_waypoint(current_idx)]
        for _ in range(1, self.config.num_waypoints):
            current_idx = self._find_lookahead_index(current_idx, self.config.waypoint_interval)
            waypoints.append(self._get_waypoint(current_idx))
        return waypoints

    def _find_nearest_index(self, robot_state: RobotState) -> int:
        distances = np.linalg.norm(self.path_xy - np.array([robot_state.x, robot_state.y]), axis=1)
        return int(np.argmin(distances))

    def _find_lookahead_index(self, nearest_index: int, lookahead_dist: float) -> int:
        index = nearest_index
        while index < len(self.path_xy) - 1:
            index += 1
            distance = np.linalg.norm(self.path_xy[index] - self.path_xy[nearest_index])
            if distance >= lookahead_dist:
                return index
        return len(self.path_xy) - 1

    def _get_waypoint(self, index: int) -> dict[str, float]:
        if index == len(self.path_xy) - 1:
            next_index = index - 1
            yaw = np.arctan2(
                self.reference_path.y[index] - self.reference_path.y[next_index],
                self.reference_path.x[index] - self.reference_path.x[next_index],
            )
        else:
            next_index = index + 1
            yaw = np.arctan2(
                self.reference_path.y[next_index] - self.reference_path.y[index],
                self.reference_path.x[next_index] - self.reference_path.x[index],
            )
        return {
            "x": float(self.reference_path.x[index]),
            "y": float(self.reference_path.y[index]),
            "yaw": float(yaw),
            "vel": float(self.reference_path.v[index]),
        }


__all__ = [
    "Diagnostics",
    "GridMap2D",
    "ReferenceMapBuilder",
    "ReferencePath",
    "ReferencePathConfig",
    "RobotState",
]
