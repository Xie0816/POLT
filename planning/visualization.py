"""Visualization helpers for planning trajectories and local risk maps."""

import cv2
import numpy as np
import open3d as o3d


def trajectory_to_lineset(trajectory_xy, color, z=0.15):
    trajectory_xy = np.asarray(trajectory_xy, dtype=np.float64)
    if trajectory_xy.ndim != 2 or trajectory_xy.shape[0] < 2:
        return None
    points = np.column_stack([trajectory_xy[:, 0], trajectory_xy[:, 1], np.full((trajectory_xy.shape[0],), z, dtype=np.float64)])
    lines = np.column_stack([np.arange(trajectory_xy.shape[0] - 1, dtype=np.int32), np.arange(1, trajectory_xy.shape[0], dtype=np.int32)])
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(np.tile(np.asarray(color, dtype=np.float64), (len(lines), 1)))
    return line_set


def add_planner_geometries_to_visualizer(vis, planner_result):
    if planner_result is None:
        return
    diagnostics = planner_result.get("diagnostics")
    if diagnostics is None:
        return

    candidate_trajectories = getattr(diagnostics, "candidate_trajectories", None) or []
    for trajectory in candidate_trajectories[:20]:
        trajectory = np.asarray(trajectory, dtype=np.float64)
        line_set = trajectory_to_lineset(trajectory[:, :2], color=[0.55, 0.55, 0.55], z=0.08)
        if line_set is not None:
            vis.add_geometry(line_set)

    best_trajectory = getattr(diagnostics, "best_trajectory", None)
    if best_trajectory is not None:
        best_trajectory = np.asarray(best_trajectory, dtype=np.float64)
        line_set = trajectory_to_lineset(best_trajectory[:, :2], color=[0.0, 0.25, 1.0], z=0.2)
        if line_set is not None:
            vis.add_geometry(line_set)

    reference_path = planner_result.get("visual_reference_path") or planner_result.get("reference_path")
    if reference_path is not None:
        ref_xy = np.column_stack((reference_path.x, reference_path.y))
        line_set = trajectory_to_lineset(ref_xy, color=[0.0, 0.85, 0.2], z=0.25)
        if line_set is not None:
            vis.add_geometry(line_set)


def risk_position_to_pixel(risk_map, x_value, y_value):
    if risk_map is None or not risk_map.is_inside(float(x_value), float(y_value)):
        return None
    ix, iy = risk_map.position_to_index(float(x_value), float(y_value))
    row = risk_map.size_x - 1 - ix
    col = risk_map.size_y - 1 - iy
    if row < 0 or row >= risk_map.size_x or col < 0 or col >= risk_map.size_y:
        return None
    return int(col), int(row)


def draw_trajectory_on_bev(image, risk_map, trajectory_xy, color, thickness=2):
    trajectory_xy = np.asarray(trajectory_xy, dtype=np.float64)
    if trajectory_xy.ndim != 2 or trajectory_xy.shape[0] < 2:
        return
    pixels = []
    for point in trajectory_xy:
        pixel = risk_position_to_pixel(risk_map, point[0], point[1])
        if pixel is not None:
            pixels.append(pixel)
    for start, end in zip(pixels[:-1], pixels[1:]):
        cv2.line(image, start, end, color, thickness, lineType=cv2.LINE_AA)
