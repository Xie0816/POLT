"""Lightweight timing utilities for profiling POLT frame processing."""

import time

import numpy as np
import torch


def _profile_now(self, enabled=True):
    if enabled and torch.cuda.is_available():
        torch.cuda.synchronize(self.device)
    return time.perf_counter()


def _profile_record(self, profile, name, start_time, enabled=True):
    if not enabled or start_time is None:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize(self.device)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    profile[name] = profile.get(name, 0.0) + elapsed_ms


def _print_timing_summary(self, timing_records, top_k=20):
    if not timing_records:
        return
    non_timing_keys = {"stamp", "accumulated_points"}
    keys = sorted({key for record in timing_records for key in record if key not in non_timing_keys})
    print("========== Per-frame timing summary ==========")
    total_values = [record.get("frame_total", 0.0) for record in timing_records if record.get("frame_total", 0.0) > 0.0]
    if total_values:
        avg_total = float(np.mean(total_values))
        print(f"frames={len(timing_records)}, avg_total={avg_total:.2f} ms, approx_fps={1000.0 / max(avg_total, 1e-6):.2f}")
    stats = []
    for key in keys:
        values = [record.get(key, 0.0) for record in timing_records]
        stats.append((float(np.mean(values)), float(np.max(values)), key))
    for mean_ms, max_ms, key in sorted(stats, reverse=True)[:top_k]:
        print(f"{key:>28s}: mean={mean_ms:8.2f} ms, max={max_ms:8.2f} ms")
    mean_by_key = {key: mean_ms for mean_ms, _, key in stats}
    print("========== FPS optimization hints ==========")
    if mean_by_key.get("open3d_visualization", 0.0) + mean_by_key.get("opencv_bev_visualization", 0.0) > 20.0:
        print("- visualization: VIS=False 可直接提高离线处理帧率；Open3D 点云/轨迹每帧重建通常很慢。")
    if mean_by_key.get("dino_vlad_infer", 0.0) > 20.0:
        print("- dino_vlad_infer: 可优先考虑降低输入分辨率、减少 VLAD 聚类/后处理或缓存图像特征。")
    if mean_by_key.get("feature_accumulation", 0.0) > 20.0:
        print("- feature_accumulation: 可减小 TIME_LEN/VOXEL_SIZE 历史点数量，或限制只累积规划需要的前方区域。")
    gpr_mean = mean_by_key.get("gpr_predict_mechanical", 0.0) + mean_by_key.get("gpr_predict_roughness", 0.0)
    if gpr_mean > 20.0:
        print("- GPR prediction: 当前双分支对全部累积点推理，优先做点云下采样、ROI 裁剪或合并批处理。")
    if mean_by_key.get("risk_map_build", 0.0) > 20.0:
        print("- risk_map_build: distance transform 和 1000x1000 栅格较重，可降低地图尺寸/分辨率；planner 已复用同一张 risk_map，避免重复建图。")
    if mean_by_key.get("planner_total", 0.0) > 20.0:
        print("- planner_total: MPPI 主要受 K*T 影响；需要更快可降低 samples_K 或进一步向量化 rollout/cost。")
    if mean_by_key.get("planner_trace_save", 0.0) > 10.0:
        print("- planner_trace_save: 压缩保存每帧轨迹会拖慢速度，测速或实时运行可设置 save_planner_traces=False。")
