"""Memory backends used by POLT online learning modes."""

import time

from model.memory.GPRMemoryForest import GPRMemoryForest
from polt_runtime import traversability


class HierarchicalGPRBackend:
    """Adapter for the dynamic hierarchical GPR memory."""

    def __init__(self, mem_buffer, vlad_processor):
        self.memory = GPRMemoryForest()
        if mem_buffer:
            self.memory._mf_init(mem_buffer, vlad=vlad_processor)

    def update(self, nearest_point_feat, cost):
        return self.memory._online_learning_update_with_gpr(nearest_point_feat, cost)

    def predict_point(self, nearest_point_feat):
        return self.memory.predict_cost(nearest_point_feat.cpu().numpy())

    def predict_batch(self, accumulated_feats_points, label):
        return traversability.predict_costs_for_points(self.memory, accumulated_feats_points, label)

    def nodes_num(self):
        return traversability.memory_nodes_num(self.memory)


class DataBufferGPRBackend:
    """Adapter for the fixed-size data buffer memory."""

    def __init__(self, mem_buffer, dino_sampler, vlad_clusters, buffer_size=200):
        from model.memory.DataBuffer_Cost_GPR import DataBufferCostGPR

        self.dino_sampler = dino_sampler
        self.memory = DataBufferCostGPR(
            {
                "buffer_size": buffer_size,
                "vlad_clusters": vlad_clusters,
                "train_kernel": True,
                "unknown_cost": 1.0,
                "costmap_cvar": 0.95,
            }
        )
        if mem_buffer:
            self.memory._initialize_buffer_with_prior_knowledge(mem_buffer, dino_sampler.vlad_processor)

    def _cluster_id(self, nearest_point_feat):
        if self.dino_sampler.vlad_processor is None:
            return None
        return self.dino_sampler.vlad_processor.get_nearest_cluster_id(nearest_point_feat)

    def update(self, nearest_point_feat, cost):
        feature_np = nearest_point_feat.detach().cpu().numpy()
        pred_info = self.memory.predict_cost(feature_np)
        cluster_id = self._cluster_id(feature_np)
        if cluster_id is not None:
            print(f"最近点特征对应的VLAD聚类ID: {cluster_id}")
        self.memory.update_train_buffer(feature_np.flatten(), cost, cluster_id)
        train_info = self.memory.update_gpr_model()
        return {
            "gpr_prediction": {"mu": pred_info["predicted_cost"], "var": pred_info.get("uncertainty", 0.0)},
            "train_info": train_info,
        }

    def predict_point(self, nearest_point_feat):
        feature_np = nearest_point_feat.detach().cpu().numpy()
        pred_info = self.memory.predict_cost(feature_np)
        return {"mu": pred_info["predicted_cost"], "var": pred_info.get("uncertainty", 0.0)}

    def predict_batch(self, accumulated_feats_points, label):
        if accumulated_feats_points is None or len(accumulated_feats_points) == 0:
            return []
        feat_vectors = accumulated_feats_points[:, 3:]
        start_time = time.perf_counter()
        gpr_results = self.memory.predict_cost_batch(feat_vectors, batch_size=50000)
        predicted_costs = []
        for i, gpr_result in enumerate(gpr_results):
            predicted_costs.append(
                {
                    "point_index": i,
                    "semantic_features": feat_vectors[i],
                    "predicted_cost": gpr_result["predicted_cost"],
                    "predicted_variance": gpr_result["uncertainty"] if gpr_result["uncertainty"] is not None else 0.0,
                }
            )
        inference_time = time.perf_counter() - start_time
        print(f"{label} 批量GPR推理完成: {len(predicted_costs)}个点, 耗时: {inference_time:.4f}秒")
        return predicted_costs

    def nodes_num(self):
        return int(self.memory.buffer_idx)


def resolve_memory_buffer_path(config, feedback_mode):
    """Pick the memory initialization path for a feedback branch."""
    if feedback_mode == "mechanical":
        return config.mechanical_memory_buffer_path or config.memory_buffer_path
    if feedback_mode == "roughness":
        return config.roughness_memory_buffer_path or config.memory_buffer_path
    raise ValueError(f"Unsupported feedback mode for memory path: {feedback_mode}")


def build_memory_backend(config, dino_sampler, feedback_mode):
    """Construct the configured memory implementation for one feedback branch."""
    mem_buffer = resolve_memory_buffer_path(config, feedback_mode)
    if config.memory_mode == "dynamic_memory":
        return HierarchicalGPRBackend(mem_buffer, dino_sampler.vlad_processor)
    if config.memory_mode in {"max_similiarity_out", "fixed_size_data_buffer"}:
        buffer_size = 300 if feedback_mode == "roughness" else 200
        return DataBufferGPRBackend(mem_buffer, dino_sampler, config.vlad_clusters, buffer_size=buffer_size)
    raise ValueError(f"Unsupported memory mode: {config.memory_mode}")


__all__ = [
    "HierarchicalGPRBackend",
    "DataBufferGPRBackend",
    "resolve_memory_buffer_path",
    "build_memory_backend",
]
