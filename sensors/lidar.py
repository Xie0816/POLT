"""LiDAR loading, motion compensation, projection, and point-cloud accumulation."""

from dataclasses import dataclass

import math
import os
import time
import numpy as np
import torch
from torch_scatter import scatter_min
import open3d as o3d
from common_struct import *
import cv2
import matplotlib.pyplot as plt 
from polt_runtime import projection


class Lidar():
    """Stateful LiDAR processor used by the POLT frame pipeline."""
    
    def __init__(self):
        """Initialize device placement and scan-angle resolution."""
        self.device = torch.device("cuda")
        self.horizontal_angular_resolution = HORIZONTAL_ANGULAR_RESOLUTION

    def load_lidarbin(self, lidar_path):
        """Load one raw LiDAR binary file and return valid XYZ points in meters."""
        # Raw files store int32 centimeters; load on CPU before moving to CUDA.
        points_cpu = torch.from_numpy(np.fromfile(lidar_path, dtype=np.int32)).view(-1, 4)

        # Convert centimeters to meters and filter invalid or out-of-range points.
        points_gpu = points_cpu.float().to(self.device) / 100.0

        x, y, z, _ = points_gpu.unbind(dim=1)

        mask = (
                torch.isfinite(x).float() *
                torch.isfinite(y).float() *
                torch.isfinite(z).float() *
                (~((x == 0) & (y == 0) & (z == 0))).float() *
                (x.abs() < 1000).float() *
                (y.abs() < 1000).float() *
                (z < Z_max).float() *
                (z > Z_min).float() *
                ((x.pow(2) + y.pow(2)).sqrt() >= 4).float()
        ).bool()

        return points_gpu[mask, :3]

    def angle_index(self, points):
        """Append the horizontal scan-angle bin to each point as the fourth column."""
        x, y = points[:, 0], points[:, 1]
        # Horizontal angle convention follows the original LiDAR scan order.
        angles_deg = torch.atan2(y, -x) * (180 / math.pi)  # [-180°, 180°]
        angles_deg = torch.where(angles_deg < -180, angles_deg + 360, angles_deg)

        # 0.2-degree angular resolution gives bins in [0, 1800].
        angle_indices = (angles_deg / self.horizontal_angular_resolution).long().clamp(0, 1800)  # [N]
        return torch.cat([points, angle_indices.unsqueeze(-1)], dim=-1)  # [N, 4]

    def inter_pose(self, pre_pose, pre_quat, cur_pose, cur_quat, t):
        """Interpolate pose and quaternion for per-point motion compensation."""

        # Position uses linear interpolation; orientation uses a stable SLERP path.
        inter_pose = pre_pose + t.unsqueeze(-1) * (cur_pose - pre_pose)  # [N,3]

        dot = (pre_quat * cur_quat).sum(dim=-1, keepdim=True)

        # Flip quaternion sign when needed so interpolation follows the short arc.
        mask = dot < 0
        cur_quat = torch.where(mask, -cur_quat, cur_quat)
        dot = torch.where(mask, -dot, dot)

        theta = torch.acos(torch.clamp(dot, -1 + 1e-6, 1 - 1e-6))

        # Near-identical orientations fall back to linear interpolation.
        small_angle = theta < 1e-6
        interp_linear = (1 - t.unsqueeze(-1)) * pre_quat + t.unsqueeze(-1) * cur_quat

        sin_theta = torch.sin(theta)
        interp_slerp = (torch.sin((1 - t.unsqueeze(-1)) * theta) / sin_theta) * pre_quat + \
                       (torch.sin(t.unsqueeze(-1) * theta) / sin_theta) * cur_quat

        inter_quat = torch.where(small_angle, interp_linear, interp_slerp)

        # Normalize to protect downstream rotation matrix construction.
        inter_quat = inter_quat / torch.norm(inter_quat, dim=-1, keepdim=True)

        return inter_pose, inter_quat

    def motion_compensation(self, points, pre_odom, cur_odom):
        """Compensate one LiDAR scan into the current frame using odometry."""
        # Step 1: assign scan-angle bins for per-point timing.
        points_with_indices = self.angle_index(points)  # [N,4]
        angle_indices = points_with_indices[:, -1].float()  # [N]

        # Step 2: approximate time ratio within the scan, normalized to [0, 1].
        t = angle_indices / 1800.0

        # Step 3: interpolate each point's acquisition pose.
        pre_pose = pre_odom[:3] # [3]
        pre_quat = pre_odom[3:]  # [4]
        cur_pose = cur_odom[:3]  # [3]
        cur_quat = cur_odom[3:]  # [4]

        inter_pose, inter_quat = self.inter_pose(
            pre_pose, pre_quat, cur_pose, cur_quat, t
        )  # inter_pose [N,3], inter_quat [N,4]

        # Step 4: project all points to the current LiDAR frame.
        compensated_points = self.projection(
            points_with_indices[:, :3],  # [N,3]
            inter_pose,  # [N,3]
            inter_quat,  # [N,4]
            cur_pose.unsqueeze(0),  # [1,3]
            cur_quat.unsqueeze(0)  # [1,4]
        )
        return compensated_points

    def quat_to_mat(self, quat):
        """Convert batched wxyz quaternions into rotation matrices."""
        w, x, y, z = quat.unbind(-1)
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        rot_mat = torch.stack([
            1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
            2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
            2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)
        ], dim=-1).view(*quat.shape[:-1], 3, 3)
        return rot_mat

    def projection(self, points, ori_pose, ori_quat, target_pose, target_quat):
        """Project points from their source pose into the target pose frame."""

        # Compute rotation matrices
        ori_R = self.quat_to_mat(ori_quat)  # [N,3,3] or [3,3]
        target_R_inv = self.quat_to_mat(target_quat).transpose(-2, -1)  # [N,3,3] or [3,3]

        # Project to world coordinates
        world_points = torch.matmul(points.unsqueeze(-2), ori_R.transpose(-2, -1)).squeeze(-2) + ori_pose

        # Project to target coordinates
        target_points = torch.matmul((world_points - target_pose).unsqueeze(-2),
                                     target_R_inv.transpose(-2, -1)).squeeze(-2)
        return target_points

    def projection_accumulation(self, points_xyzrgb, odom_data, cur_odom=None, time_windows=TIME_LEN, accumulate_gap=TIME_GAP, voxel_size=VOXEL_SIZE):
        """
        Accumulate recent point clouds into the current odometry frame.

        Args:
            points_xyzrgb: Timestamp-indexed point clouds with attached RGB/features.
            odom_data: Timestamp-indexed odometry poses.
            cur_odom: Optional current pose override.
            time_windows: Number of candidate timestamps to inspect.
            accumulate_gap: Temporal stride for accumulation.
            voxel_size: Voxel size in meters; ``None`` disables downsampling.

        Returns:
            CUDA tensor containing accumulated and optionally downsampled points.
        """
        start = time.perf_counter()
        torch.cuda.synchronize()
        # Use the newest timestamp as the local reference frame.
        time_stamps = sorted([key for key in points_xyzrgb.keys()], reverse=True)
        cur_stamp = time_stamps[0]

        # Resolve the current pose used as accumulation target.
        if cur_odom is None:
            cur_odom = odom_data[cur_stamp]
        cur_pose = cur_odom[:3]
        cur_quat = cur_odom[3:]

        # Project selected historical scans into the current frame.
        accumulated_points_list = []
        end = min(time_windows, len(time_stamps))
        for i in range(0, end, accumulate_gap):
            # Fetch the source scan and any attached channels.
            stamp = time_stamps[i]
            points_i = points_xyzrgb[stamp][..., :3]
            colors_i = points_xyzrgb[stamp][..., 3:]

            # Use the source scan pose for rigid projection.
            odom_i = odom_data[stamp]
            pose_i = odom_i[:3]
            quat_i = odom_i[3:]

            projected_points = self.projection(points_i, pose_i, quat_i, cur_pose, cur_quat)

            projected_points = torch.cat((projected_points, colors_i), dim=-1)

            filtered_points = self.points_filter(projected_points)

            accumulated_points_list.append(filtered_points)

        # Merge all selected scans on GPU.
        if len(accumulated_points_list) == 0:
            return torch.empty((0, 6), device=self.device)
        
        accumulated_points = torch.cat(accumulated_points_list, dim=0)
        
        # Downsample after accumulation to cap runtime for projection/GPR.
        if voxel_size is not None:
            accumulated_points = self.voxel_downsample(accumulated_points, voxel_size=voxel_size)
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        print(f"点云累积投影计算耗时{(end - start):.4f} s")
        return accumulated_points

    def points_filter(self, points, origin='center'):
        """
        Filter a point cloud to the configured local map bounds.

        Args:
            points: Input points, with at least XYZ columns.
            origin: ``center`` for ego-centered maps or ``bottom`` for forward maps.

        Returns:
            Filtered point cloud as a CUDA tensor.
        """
        # Convert numpy input to the same CUDA device used by LiDAR processing.
        if isinstance(points, np.ndarray):
            points = torch.from_numpy(points).float().to(self.device)

        distance = MAP_SIZE * RESOLUTION
        if origin == 'center':
            mask = (points[..., 0] > -distance / 2 + RESOLUTION) & (points[..., 0] < distance / 2) & \
                   (points[..., 1] > -distance / 2 + RESOLUTION) & (points[..., 1] < distance / 2)
        elif origin == 'bottom':
            mask = (points[..., 0] >= 0) & (points[..., 0] < distance) & \
                   (points[..., 1] > -distance / 2 + RESOLUTION) & (points[..., 1] < distance / 2)
        else:
            raise ValueError(f"Invalid origin option: {origin}")
        return points[mask]


    def voxel_downsample(self, points, voxel_size=0.1):
        """Voxel downsample points while preserving attached color/feature columns."""
        if len(points) == 0:
            return points
        
        num_points, dim = points.shape
        
        # Split geometry from any attached RGB or semantic feature channels.
        points_xyz = points[:, :3]  # [N, 3]
        points_info = points[:, 3:]  # [N, dim-3]

        voxel_indices = torch.floor(points_xyz / voxel_size).long()
        
        # Shift voxel indices into non-negative space before hashing.
        min_indices = voxel_indices.min(dim=0)[0]  # [3]
        offset = -min_indices
        
        shifted_indices = voxel_indices + offset
        
        # Build collision-free integer keys from 3D voxel indices.
        max_indices = shifted_indices.max(dim=0)[0]  # [3]
        range_x = max_indices[0].item() + 1
        range_y = max_indices[1].item() + 1
        range_z = max_indices[2].item() + 1
        
        base_y = range_x
        base_z = range_x * range_y
        
        voxel_keys = shifted_indices[:, 0] + shifted_indices[:, 1] * base_y + shifted_indices[:, 2] * base_z
        
        unique_keys, inverse_indices, counts = torch.unique(voxel_keys, return_inverse=True, return_counts=True)
        
        # Keep the first point assigned to each voxel as the representative point.
        cum_counts = torch.cumsum(counts, dim=0)
        first_indices = cum_counts - counts
        
        representative_mask = torch.zeros_like(voxel_keys, dtype=torch.bool)
        representative_mask[first_indices] = True
        
        downsampled_points_xyz = points_xyz[representative_mask]
        downsampled_points_info = points_info[representative_mask]
        
        downsampled_points = torch.cat([downsampled_points_xyz, downsampled_points_info], dim=1)
        
        print(f"PyTorch3D体素降采样: {len(points)} -> {len(downsampled_points)} 个点 (体素大小: {voxel_size}m)")
        print(f"体素索引范围: x[{min_indices[0].item()}, {max_indices[0].item()-offset[0].item()}], "
              f"y[{min_indices[1].item()}, {max_indices[1].item()-offset[1].item()}], "
              f"z[{min_indices[2].item()}, {max_indices[2].item()-offset[2].item()}]")
        print(f"体素数量: {len(unique_keys)}")
        
        return downsampled_points

    def accumulated_show(self, accumulated_points):
        """Display accumulated points in an Open3D viewer."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(accumulated_points.cpu().numpy())
        o3d.visualization.draw_geometries([pcd])

    def compute_geometric_roughness_cost(self, points, grid_res=RESOLUTION):
        """Estimate a geometry-only roughness cost for each accumulated point."""
        # Morphological post-processing is simpler on CPU numpy arrays.
        if torch.is_tensor(points):
            pts = points[:, :3].cpu().numpy()
        else:
            pts = points[:, :3]
            
        if len(pts) == 0:
            return []

        # Step 1: rasterize the accumulated point cloud into local XY cells.
        voxel_coords = np.floor(pts[:, :2] / grid_res).astype(np.int32)
        unique_coords, inverse_indices = np.unique(voxel_coords, axis=0, return_inverse=True)
        
        K = len(unique_coords)
        z = pts[:, 2]
        
        counts = np.bincount(inverse_indices, minlength=K)
        valid_mask = counts > 0
        safe_counts = np.clip(counts, 1, None)
        
        # Local elevation variance is used as a roughness proxy.
        sum_z = np.bincount(inverse_indices, weights=z, minlength=K)
        mean_z = sum_z / safe_counts
        
        sum_sq_z = np.bincount(inverse_indices, weights=z**2, minlength=K)
        var_z = (sum_sq_z / safe_counts) - mean_z**2
        std_z = np.sqrt(np.maximum(var_z, 0))
        
        # Elevation range captures steps, curbs, and obstacle edges.
        max_z = np.full(K, -np.inf)
        np.maximum.at(max_z, inverse_indices, z)
        min_z = np.full(K, np.inf)
        np.minimum.at(min_z, inverse_indices, z)
        delta_z = max_z - min_z
        delta_z[~valid_mask] = 0.0 
        
        # Step 2: map local geometry statistics to traversability cost.
        MAX_STD = 0.5
        MAX_STEP = 0.80
        
        roughness_cost = np.clip(std_z / MAX_STD, 0.0, 1.0)
        step_cost = np.clip(delta_z / MAX_STEP, 0.0, 1.0)
        
        # Conservative fusion: use the worse of roughness and step penalties.
        grid_costs = np.maximum(roughness_cost, step_cost)
        
        # Step 3: fill sparse LiDAR holes and smooth local discontinuities.
        min_x, min_y = np.min(unique_coords, axis=0)
        max_x, max_y = np.max(unique_coords, axis=0)
        width = max_x - min_x + 1
        height = max_y - min_y + 1
        
        cost_img = np.zeros((width, height), dtype=np.float32)
        grid_x_idx = unique_coords[:, 0] - min_x
        grid_y_idx = unique_coords[:, 1] - min_y
        cost_img[grid_x_idx, grid_y_idx] = grid_costs
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated_cost = cv2.dilate(cost_img, kernel, iterations=1)
        
        smoothed_cost = cv2.GaussianBlur(dilated_cost, (3, 3), 0)
        
        # Step 4: map rasterized costs back to the original 3D points.
        pt_x_idx = voxel_coords[:, 0] - min_x
        pt_y_idx = voxel_coords[:, 1] - min_y
        
        final_costs = smoothed_cost[pt_x_idx, pt_y_idx]
        
        # Keep the same prediction-record format used by learned memory outputs.
        predicted_costs = [{'predicted_cost': float(c)} for c in final_costs]
        
        return predicted_costs





    def generate_geometric_bev(self, pointsxyzc, grid_res=RESOLUTION, origin='bottom', VISUALIZE=False, agg_method='mean'):
        """Rasterize point-wise geometric costs into a local BEV cost map."""
        if torch.is_tensor(pointsxyzc):
            pointsxyzc = pointsxyzc.to(self.device)
        else:
            pointsxyzc = torch.from_numpy(pointsxyzc).float().to(self.device)
        valid_mask = pointsxyzc[:, 3] <= LIDAR_H
        pointsxyzc = pointsxyzc[valid_mask]
        pointsxyzc = self.points_filter(pointsxyzc, origin=origin)
        if len(pointsxyzc) == 0:
            empty_grid = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)
            return {'roughness': empty_grid.copy(), 'slope': empty_grid.copy(), 'step': empty_grid.copy(), 'cost': empty_grid.copy()}

        if origin == 'center':
            grid_y = (-pointsxyzc[:, 0] / grid_res + MAP_SIZE // 2).long()
            grid_x = (-pointsxyzc[:, 1] / grid_res + MAP_SIZE // 2).long()
        elif origin == 'bottom':
            grid_y = (MAP_SIZE - 1) - (pointsxyzc[:, 0] / grid_res).long()
            grid_x = (-pointsxyzc[:, 1] / grid_res + MAP_SIZE // 2).long()
        else:
            raise ValueError(f"Invalid origin option: {origin}")

        valid_mask = (grid_x >= 0) & (grid_x < MAP_SIZE) & (grid_y >= 0) & (grid_y < MAP_SIZE)
        grid_x_valid = grid_x[valid_mask]
        grid_y_valid = grid_y[valid_mask]

        cost_values = pointsxyzc[valid_mask, 3]
        flat_indices = grid_y_valid * MAP_SIZE + grid_x_valid

        bev_cost = torch.zeros((MAP_SIZE, MAP_SIZE), device=self.device)

        if agg_method == 'mean':
            weight_grid = torch.zeros((MAP_SIZE, MAP_SIZE), device=self.device)
            weight_grid.view(-1).scatter_add_(0, flat_indices, torch.ones_like(cost_values))
            bev_cost.view(-1).scatter_add_(0, flat_indices, cost_values)
            valid_weights = weight_grid > 0
            bev_cost[valid_weights] = bev_cost[valid_weights] / weight_grid[valid_weights]
        elif agg_method == 'max':
            bev_cost.view(-1).scatter_reduce_(0, flat_indices, cost_values, reduce='amax')
            valid_weights = bev_cost > 0
        else:
            raise ValueError(f"Invalid aggregation method: {agg_method}")

        if valid_weights.any():
            from torch.nn.functional import max_pool2d
            mask = valid_weights.float()
            bev_grids = [bev_cost]
            for _ in range(3):
                mask = max_pool2d(mask.unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0).squeeze(0)
                for i, bev_grid in enumerate(bev_grids):
                    bev_grid_expanded = max_pool2d(bev_grid.unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0).squeeze(0)
                    fill_mask = (mask > 0) & (bev_grid == 0)
                    bev_grid[fill_mask] = bev_grid_expanded[fill_mask]
                    bev_grids[i] = bev_grid

        bev_grids_np = {
            'cost': bev_cost.detach().cpu().numpy(),
        }
        self.cost_bev = bev_grids_np['cost']
        self.bev_grid = bev_grids_np['cost']
        self.bev_grid_rss = bev_grids_np
        if VISUALIZE:
            self.visualize_bev(bev_grids_np['cost'], title='Cost BEV')
        return bev_grids_np

    def _cost_to_color(self, costs):
        """Map normalized costs to RGB colors using the traversability colormap."""
        import matplotlib.cm as cm
        normalized_costs = [max(0, min(cost, 1)) for cost in costs]
        custom_cmap = cm.get_cmap('RdYlGn_r')
        colors = custom_cmap(normalized_costs)[:, :3]
        return colors
 
    def visualize_bev(self, bev_image=None, title="Roughness BEV"):
        """Display one BEV cost grid for quick local debugging."""
        colored_image = np.ones((bev_image.shape[0], bev_image.shape[1], 3), dtype=np.float32)
        valid_mask = bev_image > 0
        if np.any(valid_mask):
            valid_values = bev_image[valid_mask]
            v_min, v_max = np.min(valid_values), np.max(valid_values)
            normalized_values = (valid_values - v_min) / (v_max - v_min) if v_max > v_min else np.zeros_like(valid_values)
            color_array = self._cost_to_color(normalized_values)
            for i in range(3):
                channel = colored_image[:, :, i]
                channel[valid_mask] = color_array[:, i]
                colored_image[:, :, i] = channel
        cv2.imshow(title, colored_image[:, :, ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_bev(self, output_dir, stamp, bev_grids):
        """Save raw BEV cost grids for one timestamp."""
        if bev_grids is None:
            bev_grids = self.bev_grid_rss
        for attr in ['cost']:
            bev_dir = os.path.join(output_dir, 'cost_mec', f'bev_{attr}')
            os.makedirs(bev_dir, exist_ok=True)
            np.save(os.path.join(bev_dir, f'{stamp}.npy'), bev_grids[attr])

    def save_bev_color(self, output_dir, stamp, bev_grids):
        """Save colorized BEV cost grids for one timestamp."""
        if bev_grids is None:
            bev_grids = self.bev_grid_rss
        for attr in [ 'cost']:
            bev_color_dir = os.path.join(output_dir, 'cost_mec', f'bev_{attr}_color')
            os.makedirs(bev_color_dir, exist_ok=True)
            output_path = os.path.join(bev_color_dir, f'{stamp}.png')
            bev_image = bev_grids[attr]
            colored_image = np.ones((bev_image.shape[0], bev_image.shape[1], 3), dtype=np.float32)
            valid_mask = bev_image > 0
            if np.any(valid_mask):
                valid_values = bev_image[valid_mask]
                v_min, v_max = np.min(valid_values), np.max(valid_values)
                normalized_values = (valid_values - v_min) / (v_max - v_min) if v_max > v_min else np.zeros_like(valid_values)
                color_array = self._cost_to_color(normalized_values)
                for i in range(3):
                    channel = colored_image[:, :, i]
                    channel[valid_mask] = color_array[:, i]
                    colored_image[:, :, i] = channel
            cv2.imwrite(output_path, (colored_image[:, :, ::-1] * 255).astype(np.uint8))


@dataclass
class LidarFrameMatch:
    img_stamp: int
    lidar_idx: int
    lidar_stamp: int
    odom_idx: int
    odom_stamp: int
    prev_lidar_stamp: int
    prev_odom_stamp: int


@dataclass
class CompensatedLidarFrame:
    match: LidarFrameMatch
    points: torch.Tensor
    odom: torch.Tensor
    prev_odom: torch.Tensor


def match_lidar_frame(runner, img_stamp):
    lidar_idx, lidar_stamp = runner.time_match(runner.lidar_stamps, img_stamp)
    odom_idx, odom_stamp = runner.time_match(runner.lidar_odom_stamps, img_stamp)
    if not lidar_idx or not lidar_stamp or not odom_idx or not odom_stamp:
        return None

    return LidarFrameMatch(
        img_stamp=img_stamp,
        lidar_idx=lidar_idx,
        lidar_stamp=lidar_stamp,
        odom_idx=odom_idx,
        odom_stamp=odom_stamp,
        prev_lidar_stamp=runner.lidar_stamps[max(0, lidar_idx - 1)],
        prev_odom_stamp=runner.lidar_odom_stamps[max(0, odom_idx - 1)],
    )


def compensate_lidar_frame(runner, lidar_sensor, match):
    cur_points = lidar_sensor.load_lidarbin(runner.lidar_files[match.lidar_stamp])
    cur_odom = runner.lidar_odom_files[match.odom_stamp]
    prev_odom = runner.lidar_odom_files[match.prev_odom_stamp]
    compensated_points = lidar_sensor.motion_compensation(cur_points, prev_odom, cur_odom)

    runner.lidar_data[match.lidar_stamp] = compensated_points
    runner.odom_data[match.lidar_stamp] = cur_odom

    return CompensatedLidarFrame(
        match=match,
        points=compensated_points,
        odom=cur_odom,
        prev_odom=prev_odom,
    )


def project_points_to_image_time(
    runner,
    lidar_sensor,
    frame,
    image_bgr,
    proj_mat=LM_AR0231_Front,
    scale_factor=SCALE_FACTOR,
    visualize=False,
):
    denom = max(1, frame.match.lidar_stamp - frame.match.prev_lidar_stamp)
    t = torch.from_numpy(np.array((frame.match.img_stamp - frame.match.lidar_stamp) / denom)).to(runner.device)
    img_pose, img_quat = lidar_sensor.inter_pose(
        frame.prev_odom[:3],
        frame.prev_odom[3:],
        frame.odom[:3],
        frame.odom[3:],
        t,
    )
    img_points = lidar_sensor.projection(frame.points, frame.odom[:3], frame.odom[3:], img_pose, img_quat)
    return projection.lidar_to_camera(
        runner,
        image_bgr,
        img_points,
        frame.points,
        proj_mat=proj_mat,
        visualize=visualize,
        scale_factor=scale_factor,
    )


def project_features_to_image_time(
    runner,
    lidar_sensor,
    frame,
    patch_features,
    feat_size,
    image_size,
    proj_mat=LM_AR0231_Front,
):
    denom = max(1, frame.match.lidar_stamp - frame.match.prev_lidar_stamp)
    t = torch.from_numpy(np.array((frame.match.img_stamp - frame.match.lidar_stamp) / denom)).to(runner.device)
    img_pose, img_quat = lidar_sensor.inter_pose(
        frame.prev_odom[:3],
        frame.prev_odom[3:],
        frame.odom[:3],
        frame.odom[3:],
        t,
    )
    img_points = lidar_sensor.projection(frame.points, frame.odom[:3], frame.odom[3:], img_pose, img_quat)
    feat_h, feat_w = feat_size
    scale_args = (feat_w, feat_h, image_size[0], image_size[1])
    return projection.lidar_to_features(
        runner,
        patch_features,
        img_points,
        frame.points,
        proj_mat=proj_mat,
        scale_args=scale_args,
    )


def accumulate_history(lidar_sensor, history_points, odom_history, cur_odom, voxel_size=VOXEL_SIZE):
    return lidar_sensor.projection_accumulation(history_points, odom_history, cur_odom, voxel_size=voxel_size)


def contact_region_mask(points_xyz, device, radius=RESOLUTION, contact_position=None):
    center = contact_position
    if center is None:
        center = torch.tensor([0.0, 0.0, -LIDAR_H], device=device)
    distances = torch.norm(points_xyz - center.unsqueeze(0), dim=1)
    return distances < radius


__all__ = [
    "Lidar",
    "LidarFrameMatch",
    "CompensatedLidarFrame",
    "match_lidar_frame",
    "compensate_lidar_frame",
    "project_points_to_image_time",
    "project_features_to_image_time",
    "accumulate_history",
    "contact_region_mask",
]
