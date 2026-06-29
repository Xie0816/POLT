"""Per-frame perception containers and feature extraction helpers."""

from dataclasses import dataclass

from PIL import Image
import torch

from common_struct import VOXEL_SIZE
from sensors.image import load_pil_image
from sensors.lidar import (
    CompensatedLidarFrame,
    LidarFrameMatch,
    accumulate_history,
    compensate_lidar_frame,
    match_lidar_frame,
    project_features_to_image_time,
)


@dataclass
class SyncedFrame:
    """Sensor payloads matched to one front-camera timestamp."""

    img_stamp: int
    lidar_idx: int
    lidar_stamp: int
    odom_idx: int
    odom_stamp: int
    proprio_idx: int
    proprio_stamp: int
    image: Image.Image
    compensated_points: torch.Tensor
    odom: torch.Tensor
    proprio: dict
    prev_odom: torch.Tensor


@dataclass
class FeatureFrame:
    """Projected DINO/VLAD features for the current LiDAR frame and history."""

    synced: SyncedFrame
    patch_features: torch.Tensor
    feat_h: int
    feat_w: int
    features_points: torch.Tensor
    accumulated_feats_points: torch.Tensor


def load_synced_frame(runner, lidar_sample, img_stamp):
    """Synchronize camera, LiDAR, odometry, and proprioception for one image."""
    match = match_lidar_frame(runner, img_stamp)
    if match is None:
        return None
    proprio_idx, proprio_stamp = runner.time_match(runner.proprio_stamps, img_stamp)
    if proprio_idx is None or proprio_stamp is None:
        return None

    frame = compensate_lidar_frame(runner, lidar_sample, match)
    image = load_pil_image(runner.img_files[img_stamp])
    cur_proprio = runner.proprio_files[str(proprio_stamp)]
    runner.proprio_data[match.lidar_stamp] = cur_proprio

    return SyncedFrame(
        img_stamp=img_stamp,
        lidar_idx=match.lidar_idx,
        lidar_stamp=match.lidar_stamp,
        odom_idx=match.odom_idx,
        odom_stamp=match.odom_stamp,
        proprio_idx=proprio_idx,
        proprio_stamp=proprio_stamp,
        image=image,
        compensated_points=frame.points,
        odom=frame.odom,
        proprio=cur_proprio,
        prev_odom=frame.prev_odom,
    )


def infer_patch_features(dino_sampler, image, use_vlad):
    """Run DINO inference and return either patch features or VLAD features."""
    if use_vlad:
        vlad_features, _, _, (feat_h, feat_w) = dino_sampler.model_infer(image, use_vlad=True)
        return vlad_features, feat_h, feat_w
    patch_features, _, (feat_h, feat_w) = dino_sampler.model_infer(image)
    return patch_features, feat_h, feat_w


def build_feature_frame(runner, lidar_sample, dino_sampler, synced_frame, use_vlad, voxel_size=VOXEL_SIZE):
    """Project image features onto compensated LiDAR and accumulate history."""
    patch_features, feat_h, feat_w = infer_patch_features(dino_sampler, synced_frame.image, use_vlad)
    frame = CompensatedLidarFrame(
        match=LidarFrameMatch(
            img_stamp=synced_frame.img_stamp,
            lidar_idx=synced_frame.lidar_idx,
            lidar_stamp=synced_frame.lidar_stamp,
            odom_idx=synced_frame.odom_idx,
            odom_stamp=synced_frame.odom_stamp,
            prev_lidar_stamp=runner.lidar_stamps[max(0, synced_frame.lidar_idx - 1)],
            prev_odom_stamp=runner.lidar_odom_stamps[max(0, synced_frame.odom_idx - 1)],
        ),
        points=synced_frame.compensated_points,
        odom=synced_frame.odom,
        prev_odom=synced_frame.prev_odom,
    )
    features_points = project_features_to_image_time(
        runner,
        lidar_sample,
        frame,
        patch_features,
        (feat_h, feat_w),
        synced_frame.image.size,
    )
    runner.points_xyzfeat[synced_frame.lidar_stamp] = features_points
    accumulated_feats_points = accumulate_history(
        lidar_sample,
        runner.points_xyzfeat,
        runner.odom_data,
        synced_frame.odom,
        voxel_size=voxel_size,
    )

    return FeatureFrame(
        synced=synced_frame,
        patch_features=patch_features,
        feat_h=feat_h,
        feat_w=feat_w,
        features_points=features_points,
        accumulated_feats_points=accumulated_feats_points,
    )
