"""Projection utilities between image, feature-map, LiDAR, and BEV spaces."""

import time

import cv2
import numpy as np
import torch
from torch_scatter import scatter_min

from common_struct import LM_AR0231_Front, SCALE_FACTOR


def visualize_projection(img_resized, imgx, imgy, final_indices, img_points):
    img_vis = img_resized.copy()
    if len(final_indices) > 0:
        u = imgx[final_indices].cpu().numpy().astype(np.int32)
        v = imgy[final_indices].cpu().numpy().astype(np.int32)
        depths = np.sqrt((img_points[:, 0] ** 2 + img_points[:, 1] ** 2).cpu().numpy())

        min_depth = depths.min() if len(depths) > 0 else 0
        max_depth = depths.max() if len(depths) > 0 else 1
        normalized_depths = (depths - min_depth) / (max_depth - min_depth) if max_depth > min_depth else np.zeros_like(depths)

        colors = np.zeros((len(normalized_depths), 3), dtype=np.uint8)
        for i, depth_ratio in enumerate(normalized_depths):
            if depth_ratio < 0.5:
                r, g, b = 0, int(255 * (depth_ratio * 2)), int(255 * (1 - depth_ratio * 2))
            else:
                r, g, b = int(255 * ((depth_ratio - 0.5) * 2)), int(255 * (1 - (depth_ratio - 0.5) * 2)), 0
            colors[i] = [b, g, r]

        for u_val, v_val, color in zip(u, v, colors):
            cv2.circle(img_vis, (u_val, v_val), 1, color.tolist(), -1)

    cv2.namedWindow("Projection", cv2.WINDOW_NORMAL)
    cv2.imshow("Projection", img_vis)
    cv2.waitKey(1)


def batch_extract_colors(device, img_resized, imgx, imgy, width, height):
    valid_mask = (imgx >= 0) & (imgx < width) & (imgy >= 0) & (imgy < height)
    valid_imgx = imgx[valid_mask]
    valid_imgy = imgy[valid_mask]

    if len(valid_imgx) == 0:
        return torch.zeros((len(imgx), 3), device=device)

    img_tensor = torch.from_numpy(img_resized).to(device).float()
    colors = img_tensor[valid_imgy, valid_imgx]
    rgb_colors = colors[:, [2, 1, 0]]

    result = torch.zeros((len(imgx), 3), device=device)
    result[valid_mask] = rgb_colors
    return result


def lidar_to_camera(
    runner,
    img,
    points,
    img_points,
    proj_mat=LM_AR0231_Front,
    visualize=False,
    scale_factor=SCALE_FACTOR,
):
    start = time.perf_counter()
    torch.cuda.synchronize()

    img_resized = cv2.resize(
        img,
        (int(img.shape[1] * scale_factor), int(img.shape[0] * scale_factor)),
        interpolation=cv2.INTER_LINEAR,
    )
    proj_mat_gpu = torch.from_numpy(proj_mat).float().to(runner.device)
    points_gpu = points
    img_points_gpu = img_points

    points_homo = torch.cat(
        [
            img_points_gpu * 1000.0,
            torch.ones(img_points_gpu.shape[0], 1, device=runner.device),
        ],
        dim=1,
    )
    projected = torch.matmul(proj_mat_gpu, points_homo.t()).t()
    imgx = (projected[:, 0] / projected[:, 2] * scale_factor).round().to(torch.int32)
    imgy = (projected[:, 1] / projected[:, 2] * scale_factor).round().to(torch.int32)

    height, width = img_resized.shape[:2]
    valid_mask = (
        (imgx >= 0) & (imgx < width) & (imgy >= 0) & (imgy < height) & (points_homo[:, 0] >= 0)
    )

    final_indices = None
    valid_indices = torch.where(valid_mask)[0]
    if len(valid_indices) > 0:
        pixel_keys = imgx[valid_indices] * width + imgy[valid_indices]
        depths = projected[:, 2].clone()[valid_indices]
        _, inverse_indices = torch.unique(pixel_keys, return_inverse=True)
        _, min_indices = scatter_min(depths, inverse_indices)
        final_indices = valid_indices[min_indices]

        imgx_final = imgx[final_indices]
        imgy_final = imgy[final_indices]
        rgb_colors_tensor = batch_extract_colors(runner.device, img_resized, imgx_final, imgy_final, width, height)

        color_points = torch.stack(
            [
                points_gpu[final_indices, 0],
                points_gpu[final_indices, 1],
                points_gpu[final_indices, 2],
                rgb_colors_tensor[:, 0],
                rgb_colors_tensor[:, 1],
                rgb_colors_tensor[:, 2],
            ],
            dim=1,
        )
    else:
        color_points = torch.empty((0, 8), device=runner.device)

    torch.cuda.synchronize()
    print("投影至图像计算耗时:%.6f s" % (time.perf_counter() - start))

    if visualize and final_indices is not None:
        visualize_projection(img_resized, imgx, imgy, final_indices, color_points)

    return color_points.float()


def lidar_to_features(
    runner,
    patch_features,
    points,
    img_points,
    proj_mat=LM_AR0231_Front,
    scale_args=None,
):
    start = time.perf_counter()
    torch.cuda.synchronize()

    feat_w, feat_h = scale_args[:2]
    scale_factor = (
        (scale_args[0] / scale_args[2], scale_args[1] / scale_args[3])
        if scale_args is not None
        else (SCALE_FACTOR, SCALE_FACTOR)
    )

    proj_mat_gpu = torch.from_numpy(proj_mat).float().to(runner.device)
    points_gpu = points
    img_points_gpu = img_points

    points_homo = torch.cat(
        [
            img_points_gpu * 1000.0,
            torch.ones(img_points_gpu.shape[0], 1, device=runner.device),
        ],
        dim=1,
    )
    projected = torch.matmul(proj_mat_gpu, points_homo.t()).t()
    imgx = (projected[:, 0] / projected[:, 2] * scale_factor[0]).round().to(torch.int32)
    imgy = (projected[:, 1] / projected[:, 2] * scale_factor[1]).round().to(torch.int32)

    valid_mask = (
        (imgx >= 0) & (imgx < feat_w) & (imgy >= 0) & (imgy < feat_h) & (points_homo[:, 0] >= 0)
    )
    valid_indices = torch.where(valid_mask)[0]
    if len(valid_indices) > 0:
        pixel_keys = imgx[valid_indices] * feat_h + imgy[valid_indices]
        depths = projected[:, 2].clone()[valid_indices]
        _, inverse_indices = torch.unique(pixel_keys, return_inverse=True)
        _, min_indices = scatter_min(depths, inverse_indices)
        final_indices = valid_indices[min_indices]

        imgx_final = imgx[final_indices]
        imgy_final = imgy[final_indices]
        feature_indices = imgy_final * feat_w + imgx_final
        point_features = patch_features[feature_indices]
        feature_points = torch.cat([points_gpu[final_indices, :3], point_features], dim=1)
    else:
        feature_points = torch.empty((0, 3 + patch_features.shape[1]), device=runner.device)

    torch.cuda.synchronize()
    print(f"点云投影至特征图计算耗时: {time.perf_counter() - start:.6f} s")
    print(f"带有特征的点云数量: {len(feature_points)}")
    return feature_points.float()
