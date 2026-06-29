"""Open3D and Matplotlib visualization helpers for POLT runtime outputs."""

import time

import cv2
import matplotlib.cm as cm
import numpy as np
import open3d as o3d
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

from common_struct import O3DVIS_CAMERA_PARAMS
from planning import visualization as planner_visualization


def rough_cost_to_color(costs):
    normalized_costs = [max(0, min(cost, 1)) for cost in costs]
    custom_cmap = cm.get_cmap("RdYlGn_r")
    return custom_cmap(normalized_costs)[:, :3]


def cost_to_color(costs):
    normalized_costs = [max(0, min(cost, 1)) for cost in costs]
    custom_cmap = cm.get_cmap("RdYlGn_r")
    return custom_cmap(normalized_costs)[:, :3]


def create_continuous_visualizer(runner, window_name="Continuous Cost Point Cloud Visualization"):
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=window_name, width=600, height=600)

    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1])
    opt.point_size = 5.0

    def space_callback(_vis):
        runner.visualizer_paused = not runner.visualizer_paused
        print(f"可视化 {'已暂停' if runner.visualizer_paused else '已继续'}")
        return False

    def esc_callback(_vis):
        runner.visualizer_should_exit = True
        print("收到ESC键，准备退出可视化...")
        return False

    vis.register_key_callback(32, space_callback)
    vis.register_key_callback(256, esc_callback)
    return vis


def make_bev_cost_panel(cost_layer, output_size=500, valid_mask=None):
    cost_layer = np.clip(np.asarray(cost_layer, dtype=np.float64), 0.0, 1.0)
    display = np.fliplr(np.flipud(cost_layer))
    color_rgb = cost_to_color(display.reshape(-1)).reshape(display.shape[0], display.shape[1], 3)
    panel = (np.clip(color_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    panel = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
    if valid_mask is not None:
        valid_display = np.fliplr(np.flipud(np.asarray(valid_mask, dtype=np.float64))) > 0.0
        panel[~valid_display] = (255, 255, 255)
    panel = cv2.resize(panel, (output_size, output_size), interpolation=cv2.INTER_NEAREST)
    return panel


def render_bev_risk_map(risk_map, planner_result=None, output_size=500):
    if risk_map is None:
        return None
    mechanical = risk_map.layers.get("mechanical_cost")
    roughness = risk_map.layers.get("roughness_cost")
    if mechanical is None or roughness is None:
        return None

    map_cost = np.clip(0.5 * mechanical + 0.5 * roughness, 0.0, 1.0)
    valid_mask = risk_map.layers.get("valid_mask")
    map_panel = make_bev_cost_panel(map_cost, output_size=output_size, valid_mask=valid_mask)
    mechanical_panel = make_bev_cost_panel(mechanical, output_size=output_size, valid_mask=valid_mask)
    roughness_panel = make_bev_cost_panel(roughness, output_size=output_size, valid_mask=valid_mask)

    overlay = cv2.resize(map_panel, (risk_map.size_y, risk_map.size_x), interpolation=cv2.INTER_NEAREST)
    if planner_result is not None:
        diagnostics = planner_result.get("diagnostics")
        if diagnostics is not None:
            candidate_trajectories = getattr(diagnostics, "candidate_trajectories", None) or []
            for trajectory in candidate_trajectories[:20]:
                trajectory = np.asarray(trajectory, dtype=np.float64)
                planner_visualization.draw_trajectory_on_bev(overlay, risk_map, trajectory[:, :2], color=(160, 160, 160), thickness=2)
            best_trajectory = getattr(diagnostics, "best_trajectory", None)
            if best_trajectory is not None:
                best_trajectory = np.asarray(best_trajectory, dtype=np.float64)
                planner_visualization.draw_trajectory_on_bev(overlay, risk_map, best_trajectory[:, :2], color=(255, 0, 0), thickness=4)
        reference_path = planner_result.get("visual_reference_path") or planner_result.get("reference_path")
        if reference_path is not None:
            ref_xy = np.column_stack((reference_path.x, reference_path.y))
            planner_visualization.draw_trajectory_on_bev(overlay, risk_map, ref_xy, color=(0, 255, 0), thickness=4)

    ego_pixel = planner_visualization.risk_position_to_pixel(risk_map, 0.0, 0.0)
    if ego_pixel is not None:
        cv2.circle(overlay, ego_pixel, 5, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(overlay, ego_pixel, 7, (0, 0, 0), 1, lineType=cv2.LINE_AA)
    overlay = cv2.resize(overlay, (output_size, output_size), interpolation=cv2.INTER_NEAREST)
    return np.hstack([overlay, mechanical_panel, roughness_panel])


def update_continuous_visualizer(runner, vis, accumulated_feats_points, predicted_costs, accumulated_colored_proprio=None, window_name=None, planner_result=None):
    if len(accumulated_feats_points) == 0 or len(predicted_costs) == 0:
        print("警告: 没有点云数据或代价数据可更新")
        return True

    vis.poll_events()
    if runner.visualizer_should_exit:
        print("收到退出信号，准备退出可视化...")
        return False

    if runner.visualizer_paused:
        print("可视化已暂停，按空格键继续...")
        while runner.visualizer_paused and not runner.visualizer_should_exit:
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.1)
        if runner.visualizer_should_exit:
            print("在暂停状态下收到退出信号，准备退出可视化...")
            return False
        print("可视化已继续")
        return True

    vis.clear_geometries()
    cost_points_3d = accumulated_feats_points[:, :3].cpu().numpy()
    cost_pcd = o3d.geometry.PointCloud()
    cost_pcd.points = o3d.utility.Vector3dVector(cost_points_3d)
    costs = [result["predicted_cost"] for result in predicted_costs]
    cost_pcd.colors = o3d.utility.Vector3dVector(cost_to_color(costs))
    vis.add_geometry(cost_pcd)

    if accumulated_colored_proprio is not None and len(accumulated_colored_proprio) > 0:
        proprio_points_3d = accumulated_colored_proprio[:, :3].cpu().numpy()
        proprio_colors = accumulated_colored_proprio[:, 3:].cpu().numpy().astype(np.float32)
        sphere_points = []
        sphere_colors = []
        for point, color in zip(proprio_points_3d, proprio_colors):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
            sphere.translate(point)
            sphere.paint_uniform_color(color / 255.0)
            sphere_points.extend(np.asarray(sphere.vertices))
            sphere_colors.extend(np.asarray(sphere.vertex_colors))
        proprio_pcd = o3d.geometry.PointCloud()
        proprio_pcd.points = o3d.utility.Vector3dVector(sphere_points)
        proprio_pcd.colors = o3d.utility.Vector3dVector(sphere_colors)
        vis.add_geometry(proprio_pcd)

    planner_visualization.add_planner_geometries_to_visualizer(vis, planner_result)
    vis.update_renderer()

    ctr = vis.get_view_control()
    view_mode = "top_down"
    if view_mode == "top_down":
        ctr.set_front([0.0, 0.0, 1.0])
        ctr.set_lookat([1.0, 0.0, 0.0])
        ctr.set_up([1.0, 0.0, 0.0])
        ctr.set_zoom(0.4)
    elif view_mode == "front_view":
        ctr.set_front([-1.0, 0.0, 0.2])
        ctr.set_lookat([15.0, 0.0, 0.0])
        ctr.set_up([0.0, 0.0, 1.0])
        ctr.set_zoom(0.1)
    else:
        ctr.set_front(O3DVIS_CAMERA_PARAMS["front"])
        ctr.set_lookat(O3DVIS_CAMERA_PARAMS["lookat"])
        ctr.set_up(O3DVIS_CAMERA_PARAMS["up"])
        ctr.set_zoom(O3DVIS_CAMERA_PARAMS["zoom"])

    if window_name:
        print(f"更新可视化窗口: {window_name}")
    if accumulated_colored_proprio is not None and len(accumulated_colored_proprio) > 0:
        print(f"本体感知点云数量: {len(accumulated_colored_proprio)}")
    return True


def update_continuous_roughness_visualizer(runner, vis, accumulated_feats_points, predicted_costs, accumulated_colored_proprio=None, window_name=None):
    if len(accumulated_feats_points) == 0 or len(predicted_costs) == 0:
        print("警告: 没有点云数据或代价数据可更新")
        return True

    vis.poll_events()
    if runner.visualizer_should_exit:
        print("收到退出信号，准备退出可视化...")
        return False

    if runner.visualizer_paused:
        print("可视化已暂停，按空格键继续...")
        while runner.visualizer_paused and not runner.visualizer_should_exit:
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.1)
        if runner.visualizer_should_exit:
            print("在暂停状态下收到退出信号，准备退出可视化...")
            return False
        print("可视化已继续")
        return True

    vis.clear_geometries()
    cost_points_3d = accumulated_feats_points[:, :3].cpu().numpy()
    cost_pcd = o3d.geometry.PointCloud()
    cost_pcd.points = o3d.utility.Vector3dVector(cost_points_3d)
    costs = [result["predicted_cost"] for result in predicted_costs]
    cost_pcd.colors = o3d.utility.Vector3dVector(rough_cost_to_color(costs))
    vis.add_geometry(cost_pcd)
    vis.update_renderer()

    ctr = vis.get_view_control()
    ctr.set_front(O3DVIS_CAMERA_PARAMS["front"])
    ctr.set_lookat(O3DVIS_CAMERA_PARAMS["lookat"])
    ctr.set_up(O3DVIS_CAMERA_PARAMS["up"])
    ctr.set_zoom(O3DVIS_CAMERA_PARAMS["zoom"])

    if window_name:
        print(f"更新可视化窗口: {window_name}")
    if accumulated_colored_proprio is not None and len(accumulated_colored_proprio) > 0:
        print(f"本体感知点云数量: {len(accumulated_colored_proprio)}")
    return True


def _render_colorbar_figure(title, tick_label, annotations, save_path=None, figsize=(1.5, 8), left=0.3, right=0.6):
    fig = Figure(figsize=figsize)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=left, right=right)
    custom_cmap = cm.get_cmap("RdYlGn_r")
    norm = Normalize(vmin=0, vmax=1)
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=custom_cmap), cax=ax, orientation="vertical")
    ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{tick:.1f}" for tick in ticks])
    if tick_label:
        cb.set_label(tick_label, fontsize=12, fontweight="bold", rotation=270, labelpad=20)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    for text, y, color, x in annotations:
        ax.text(x, y, text, transform=ax.transAxes, fontsize=10, ha="left", color=color, fontweight="bold")
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
        print(f"竖着颜色映射图注已保存为: {save_path}")
    canvas.draw()
    colorbar_img = np.asarray(canvas.buffer_rgba())[:, :, :3]
    colorbar_img_bgr = cv2.cvtColor(colorbar_img, cv2.COLOR_RGB2BGR)
    return colorbar_img_bgr


def create_rough_vertical_cost_colorbar(save_path=None):
    return _render_colorbar_figure(
        title="roughness \ncolor bar",
        tick_label="semantic cost",
        annotations=[
            ("high roughness", 1.0, "red", 1.5),
            ("mid roughness", 0.5, "orange", 1.5),
            ("low roughness", 0.0, "green", 1.5),
        ],
        save_path=save_path,
        figsize=(2, 8),
        left=0.2,
        right=0.8,
    )


def create_vertical_cost_colorbar(save_path=None):
    return _render_colorbar_figure(
        title="Cost Bar",
        tick_label=None,
        annotations=[
            ("high cost", 1.0, "red", 1.8),
            ("mid cost", 0.5, "orange", 1.8),
            ("low cost", 0.0, "green", 1.8),
        ],
        save_path=save_path,
    )
