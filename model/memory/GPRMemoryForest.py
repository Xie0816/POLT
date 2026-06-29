"""
Dynamic GPR memory for POLT online learning.

The memory keeps a two-level structure: semantic categories contain
``SemanticCostNode`` entries, and a global GPR model is fitted over all cost
nodes. Online updates use GPR prediction and kernel similarity to decide
whether a new feedback sample should merge, create a conflict, or add a node.
"""

import numpy as np
import torch
import gpytorch
from scipy import stats
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import uuid
import math
import json
import os
import warnings
warnings.filterwarnings('ignore')
import time

@dataclass
class SemanticCostNode:
    """One memory node pairing semantic features with proprioceptive cost."""

    semantic_features: torch.Tensor  # Semantic feature vector, stored on GPU.
    proprio_cost: float  # Proprioceptive feedback cost.
    conflict_nums: int = 0  # Number of unresolved conflicts.
    conflict_buffer: list = field(default_factory=list)
    access_nums: int = 0  # Number of merge/access events.
    access_buffer: list = field(default_factory=list)

    def __post_init__(self):
        """Normalize buffer types and keep semantic features on CUDA."""
        # Ensure conflict_buffer is a list after JSON import.
        if not isinstance(self.conflict_buffer, list):
            try:
                self.conflict_buffer = list(self.conflict_buffer) if hasattr(self.conflict_buffer, '__iter__') else []
            except:
                self.conflict_buffer = []
        
        # Ensure access_buffer is a list after JSON import.
        if not isinstance(self.access_buffer, list):
            try:
                self.access_buffer = list(self.access_buffer) if hasattr(self.access_buffer, '__iter__') else []
            except:
                self.access_buffer = []
        
        # Store features as CUDA tensors for GPR prediction/update paths.
        if not isinstance(self.semantic_features, torch.Tensor):
            self.semantic_features = torch.from_numpy(self.semantic_features).float().cuda()
        elif not self.semantic_features.is_cuda:
            self.semantic_features = self.semantic_features.cuda()

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, feat_dim=32):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        ls_constraint = gpytorch.constraints.Interval(0.01, math.sqrt(feat_dim))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=1.5, lengthscale_constraint=ls_constraint)
        ) 
        # self.covar_module = gpytorch.kernels.ScaleKernel(
        #     gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=feat_dim, lengthscale_constraint=ls_constraint)
        # ) 
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class GPRMemoryForest:
    """Two-level dynamic memory backed by a global exact GPR model."""
    
    def __init__(self):
        """Initialize memory storage, GPR state, and decision thresholds."""
        # First level: semantic category. Second level: SemanticCostNode.
        self.hierarchical_memory: Dict[str, Dict[str, SemanticCostNode]] = {}
        
        # Global GPR fitted with all SemanticCostNode entries.
        self.global_gpr_model: Optional[ExactGPModel] = None
        # self.likelihood: Optional[gpytorch.likelihoods.GaussianLikelihood] = gpytorch.likelihoods.GaussianLikelihood().cuda()
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Interval(1e-2, 0.5))
        self.gpr_training_data: Dict[str, np.ndarray] = {}  # Cached training data.
        self.gp_params = None
        
        # Decision thresholds for cost error and feature similarity.
        self.mu_diff = 0.05
        self.sim = 0.9

        # Conflict/merge counters.
        self.conflict_times = 1

        self.merge_times = 1
        
        # Normalization statistics for GPR inputs and targets.
        self.in_mean: Optional[torch.Tensor] = None
        self.in_std: Optional[torch.Tensor] = None
        self.cost_mean: Optional[torch.Tensor] = None
        self.cost_std: Optional[torch.Tensor] = None
        # self.is_norm_locked = False
        
        self.update_count = 0

    def _mf_init(self, mem_buffer_dir, vlad = None):
        self.import_hierarchical_memory(mem_buffer_dir, vlad = vlad)
        self._update_global_gpr_model()

    def _add_semantic_category(self, 
                            meta_semantic_label: str) -> None:
        """Create an empty semantic category in hierarchical memory."""

        self.hierarchical_memory[meta_semantic_label] = {}
    
    def _add_semantic_cost_node(self,
                             semantic_category: str,
                             semantic_features: np.ndarray,
                             proprio_cost: float,
                             conflict_nums: int = 0,
                             conflict_buffer: Optional[List] = None,
                             access_nums: int = 0,
                             access_buffer: Optional[List] = None) -> str:
        """
        Add one semantic-cost node and return its node ID.

        ``semantic_features`` can be a numpy array, list, or torch tensor; it is
        normalized by ``SemanticCostNode.__post_init__``.
        """
        # Ensure the category exists before inserting the cost node.
        if semantic_category not in self.hierarchical_memory:
            self._add_semantic_category(semantic_category)
        
        # Normalize optional buffers from callers/imported JSON.
        if conflict_buffer is None:
            conflict_buffer = []
        if access_buffer is None:
            access_buffer = []
        
        # Convert to numpy first so dataclass initialization is consistent.
        if isinstance(semantic_features, list):
            semantic_features = np.array(semantic_features, dtype=np.float32)
        elif isinstance(semantic_features, torch.Tensor):
            semantic_features = semantic_features.cpu().numpy()
        

        cost_node_id = f"cost_{len(self.hierarchical_memory[semantic_category])}"
        cost_node = SemanticCostNode(
            semantic_features=semantic_features,
            proprio_cost=proprio_cost,
            conflict_nums=conflict_nums,
            conflict_buffer=conflict_buffer,
            access_nums= access_nums,
            access_buffer=access_buffer
        )
        
        # Register the node under its semantic category.
        self.hierarchical_memory[semantic_category][cost_node_id] = cost_node
        
        return cost_node_id
    
    def _update_global_gpr_model(self, max_iter=100): 
        """
        Refit the global GPR model from all current cost nodes.

        The model is trained in float32 with L-BFGS. ARD is intentionally
        disabled to reduce overfitting in high-dimensional visual descriptors.
        """
        # Collect features, costs, and node references from hierarchical memory.
        X_list = []
        y_list = []
        node_info = []
        
        for category_label, cost_nodes in self.hierarchical_memory.items():
            for cost_node_id, cost_node in cost_nodes.items():
                # Convert semantic features to numpy before building tensors.
                if isinstance(cost_node.semantic_features, torch.Tensor):
                    feat = cost_node.semantic_features.cpu().numpy()
                else:
                    feat = cost_node.semantic_features
                X_list.append(feat)
                y_list.append(cost_node.proprio_cost)
                node_info.append({
                    'category_label': category_label,
                    'cost_node_id': cost_node_id,
                    'cost_node': cost_node
                })
        
        if len(X_list) < 2:
            return {}
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        # Convert to float32 CUDA tensors for GPR training.
        X_tensor = torch.from_numpy(X).float().cuda()
        y_tensor = torch.from_numpy(y).float().cuda()
        
        # Compute normalization statistics.
        self.in_mean = torch.mean(X_tensor, dim=0)
        # self.in_std = torch.std(X_tensor, dim=0)
        # self.in_std[self.in_std < 1e-6] = 1.0
        self.in_std = torch.ones_like(self.in_mean)  # Keep feature scale unchanged.
        
        self.cost_mean = torch.mean(y_tensor)
        self.cost_std = torch.std(y_tensor)
        if self.cost_std < 1e-6:
            self.cost_std = torch.tensor(1.0).cuda()
                

        
        # Normalize inputs and targets for exact GPR.
        X_normalized = (X_tensor - self.in_mean) / self.in_std
        y_normalized = (y_tensor - self.cost_mean) / self.cost_std
        
        # Initialize model and likelihood.
        self.global_gpr_model = ExactGPModel(
            X_normalized, 
            y_normalized, 
            self.likelihood
        ).cuda()

        # if self.gp_params is not None:
        #     self.global_gpr_model.load_state_dict(self.gp_params)

        self.global_gpr_model.train()
        self.likelihood.train()
        
        # L-BFGS is substantially faster than Adam for this small exact-GPR fit.
        optimizer = torch.optim.LBFGS(
            self.global_gpr_model.parameters(), 
            lr=1, 
            max_iter=max_iter, 
            line_search_fn='strong_wolfe'
        )
        
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.global_gpr_model)
        
        print(f"开始 GPR 训练 (N={len(X)}, L-BFGS)...")
        
        loss_history = []
        
        # L-BFGS requires a closure that recomputes the objective.
        def closure():
            optimizer.zero_grad()
            output = self.global_gpr_model(X_normalized)
            loss = -mll(output, y_normalized)
            loss.backward()
            loss_history.append(loss.item())
            return loss

        start_train = time.perf_counter()

        # Run optimizer.
        optimizer.step(closure)
        
        # Recompute final loss for diagnostics.
        with torch.no_grad():
            final_output = self.global_gpr_model(X_normalized)
            final_loss = -mll(final_output, y_normalized)

        end_train = time.perf_counter()
        train_duration = (end_train - start_train) * 1000
             
        print(f"GPR 训练完成 - Final Loss: {final_loss.item():.4f}, 训练耗时: {train_duration:.2f} ms")

        # LBFGS defaults max_eval to approximately 1.25 * max_iter.
        actual_evals = len(loss_history)
        is_converged = actual_evals < (max_iter * 1.25)
        
        print(f"实际函数评估次数: {actual_evals}")
        if is_converged:
            print(f"结论: 模型已提前收敛！(满足 tolerance_change/grad 阈值)")
        else:
            print(f"结论: 未完全收敛，达到了最大迭代步数 ({max_iter})。")

        self.gp_params = self.global_gpr_model.state_dict()
        
        # Store normalized training data and node references for similarity lookup.
        self.gpr_training_data = {
            'X_normalized': X_normalized,
            'y_normalized': y_normalized,
            'node_info': node_info
        }

        train_info = {
            'loss':final_loss,
            'duration':train_duration,
            'epochs':actual_evals
        }

        return train_info
    
    def predict_cost(self, 
                            semantic_features: np.ndarray,
                            return_kernel_similarity: bool = False) -> Dict[str, Any]:
        """
        使用全局GPR预测代价，可选择返回通过核函数计算的与训练样本的相似度
        
        参数:
            semantic_features: 语义特征
            return_kernel_similarity: 是否返回通过核函数计算的相似度
            
        返回:
            预测结果字典，包含mu、var和可选的核函数相似度信息
        """
        if self.global_gpr_model is None or self.likelihood is None:
            result = {
                'mu': -1.0,
                'var': 1.0
            }
            if return_kernel_similarity:
                result['kernel_similarity_info'] = {
                    'training_samples': 0,
                    'kernel_similarities': [],
                    'top_similar_nodes': []
                }
            return result
        
        # Convert to a CUDA tensor with a batch dimension.
        X_tensor = torch.from_numpy(semantic_features).float().cuda().reshape(1, -1)
        
        try:
            # Normalize with the latest GPR statistics.
            X_normalized = (X_tensor - self.in_mean) / self.in_std
            
            # Switch to evaluation mode for prediction.
            self.global_gpr_model.eval()
            self.likelihood.eval()
            
            start_train = time.perf_counter()
            # Predict normalized cost mean and variance.
            with torch.no_grad():
                observed_pred = self.likelihood(self.global_gpr_model(X_normalized))
                mu_normalized = observed_pred.mean
                var_normalized = observed_pred.variance
                
                # Denormalize to the original cost scale.
                mu = (mu_normalized * self.cost_std + self.cost_mean).item()
                var = (var_normalized * (self.cost_std ** 2)).item()

            end_train = time.perf_counter()
            train_duration = (end_train - start_train) * 1000
            print(f"单次推理耗时: {train_duration:.2f} ms") #

            result = {
                'mu': mu,
                'var': var
            }
            
            # Optionally include kernel-similarity diagnostics.
            if return_kernel_similarity and 'node_info' in self.gpr_training_data:
                kernel_similarity_info = self._calculate_kernel_similarities(semantic_features)
                result['kernel_similarity_info'] = kernel_similarity_info
            
            return result
            
        except Exception as e:
            print(f"GPR预测失败: {e}")
            result = {
                'mu': 0.0,
                'var': 1.0
            }
            if return_kernel_similarity:
                result['kernel_similarity_info'] = {
                    'training_samples': 0,
                    'kernel_similarities': [],
                    'top_similar_nodes': []
                }
            return result

    def predict_cost_batch(self, 
                          semantic_features_batch: np.ndarray,
                          batch_size: int = 10000,
                          return_kernel_similarity: bool = False) -> List[Dict[str, Any]]:
        """
        Batch GPR cost prediction with optional kernel-similarity diagnostics.

        The implementation keeps tensors on GPU, chunks large inputs, and uses
        float32 only around GPyTorch Cholesky operations for numerical stability.
        """
        # Return defaults before the model has been fitted.
        if self.global_gpr_model is None or self.likelihood is None:
            default_result = {
                'mu': -1.0,
                'var': 1.0,
                'kernel_similarity_info': {
                    'training_samples': 0,
                    'kernel_similarities': [],
                    'top_similar_nodes': []
                } if return_kernel_similarity else None
            }
            return [default_result] * len(semantic_features_batch)
        
        results = []
        num_points = len(semantic_features_batch)
        
        # Convert all inputs once to avoid repeated CPU/GPU transfers.
        all_features_tensor = torch.from_numpy(semantic_features_batch).half().cuda()
        
        # Enable fast predictive variance in evaluation mode.
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            self.global_gpr_model.eval()
            self.likelihood.eval()
            
            for i in range(0, num_points, batch_size):
                end_idx = min(i + batch_size, num_points)
                
                # Slice the current batch chunk.
                batch_features_tensor = all_features_tensor[i:end_idx]
                
                try:
                    # Normalize features with the latest GPR statistics.
                    X_normalized = (batch_features_tensor - self.in_mean) / self.in_std
                    
                    # Use float32 for GPyTorch Cholesky compatibility.
                    X_normalized_float32 = X_normalized.float()
                    
                    gpr_model_float32 = self.global_gpr_model.float()
                    likelihood_float32 = self.likelihood.float()
                    
                    observed_pred = likelihood_float32(gpr_model_float32(X_normalized_float32))
                    mu_normalized = observed_pred.mean.half()
                    var_normalized = observed_pred.variance.half()
                    
                    # Restore half precision for the runtime model.
                    self.global_gpr_model = self.global_gpr_model.half()
                    self.likelihood = self.likelihood.half()
                    
                    # Denormalize on GPU.
                    mu_batch = (mu_normalized * self.cost_std + self.cost_mean)
                    var_batch = (var_normalized * (self.cost_std ** 2))
                    
                    # Transfer full chunks to CPU instead of calling .item().
                    mu_batch_np = mu_batch.cpu().numpy()
                    var_batch_np = var_batch.cpu().numpy()
                    
                    # ---  ---
                    batch_len = len(batch_features_tensor)
                    
                    for j in range(batch_len):
                        result = {
                            'mu': float(mu_batch_np[j]),
                            'var': float(var_batch_np[j])
                        }
                        
                        # Optionally include similarity diagnostics.
                        if return_kernel_similarity and 'node_info' in self.gpr_training_data:
                            # Compute only when explicitly requested.
                            kernel_similarity_info = self._calculate_kernel_similarities(
                                semantic_features_batch[i + j]
                            )
                            result['kernel_similarity_info'] = kernel_similarity_info
                        
                        results.append(result)
                        
                except Exception as e:
                    print(f"批量GPR预测失败: {e}")
                    # Append default results for the failed chunk.
                    batch_len = end_idx - i
                    for _ in range(batch_len):
                        result = {
                            'mu': 0.0,
                            'var': 1.0
                        }
                        if return_kernel_similarity:
                            result['kernel_similarity_info'] = {
                                'training_samples': 0,
                                'kernel_similarities': [],
                                'top_similar_nodes': []
                            }
                        results.append(result)
        
        return results
    
    def _calculate_kernel_similarities(self, 
                                     semantic_features: np.ndarray) -> Dict[str, Any]:
        """
        通过gpytorch核函数计算与训练样本的相似度
        
        参数:
            semantic_features: 语义特征
            
        返回:
            核函数相似度信息字典
        """
        if self.global_gpr_model is None or 'node_info' not in self.gpr_training_data:
            return {
                'training_samples': 0,
                'kernel_similarities': [],
                'top_similar_nodes': []
            }
        
        # 转换为PyTorch张量并标准化，确保设备一致性（GPU）
        X_pred = torch.from_numpy(semantic_features.reshape(1, -1)).float().cuda()
        X_pred_normalized = (X_pred - self.in_mean) / self.in_std
        
        kernel_similarities = []
        
        try:
            # 使用gpytorch核函数计算与每个训练样本的相似度
            for i, node_info in enumerate(self.gpr_training_data['node_info']):
                cost_node = node_info['cost_node']
                category_label = node_info['category_label']
                
                # 获取对应的标准化训练样本
                X_train_sample_normalized = self.gpr_training_data['X_normalized'][i:i+1]
                
                try:
                    # 使用gpytorch的核函数计算相似度
                    with torch.no_grad():
                        # kernel_value = self.global_gpr_model.covar_module(X_pred_normalized, X_train_sample_normalized)
                        kernel_value = self.global_gpr_model.covar_module.base_kernel(X_pred_normalized, X_train_sample_normalized)
                        kernel_similarity = float(kernel_value.evaluate()[0, 0])
                    
                    kernel_similarities.append({
                        'category_label': category_label,
                        'cost_node_id': node_info['cost_node_id'],
                        'cost_node': cost_node,
                        'kernel_similarity': kernel_similarity
                    })
                except Exception as e:
                    print(f"单个样本核函数相似度计算失败: {e}")
                    continue
            
            # 按核函数相似度降序排序
            kernel_similarities.sort(key=lambda x: x['kernel_similarity'], reverse=True)
            
            # 获取最相似的节点
            top_similar_nodes = kernel_similarities[0] if kernel_similarities else {}
            
            return {
                'training_samples': len(kernel_similarities),
                'kernel_similarities': kernel_similarities,
                'top_similar_nodes': top_similar_nodes
            }
            
        except Exception as e:
            print(f"核函数相似度计算失败: {e}")
            return {
                'training_samples': 0,
                'kernel_similarities': [],
                'top_similar_nodes': []
            }

    def _online_learning_update_with_gpr(self, semantic_features: torch.Tensor, proprio_cost: float) -> Dict[str, Any]:
        """
        在线学习更新：将新样本添加到GPR训练数据中
        
        参数:
            semantic_features: 语义特征 (torch.Tensor, 应在GPU上)
            proprio_cost: 本体感受代价
            
        返回:
            更新结果字典
        """
        # 确保输入是torch.Tensor并在GPU上
        if not isinstance(semantic_features, torch.Tensor):
            semantic_features = torch.from_numpy(semantic_features).float().cuda()
        elif not semantic_features.is_cuda:
            semantic_features = semantic_features.cuda()
        
        # Cold start: create initial nodes before GPR has enough support.
        total_nodes = sum(len(nodes) for nodes in self.hierarchical_memory.values())
        
        # Force increment when too few nodes exist or the GPR model is missing.
        if total_nodes < 2 or self.global_gpr_model is None:
            print(f'冷启动引导：当前节点数仅为 {total_nodes}，强制执行增量机制建立基础记忆。')
            result = {
                'cur_feats': semantic_features,
                'cur_cost': proprio_cost,
                'nearest_nodes': {},
                'kernel_similarity_info': {},
                'action_taken': None,
                'new_nodes_created': [],
                'conflict_detection': None,
                'mechanism_type': 'class_increment',
                'gpr_prediction': {'mu': 1.0, 'var': 1.0, 'diff_mu': 1.0}
            }
            return self._handle_increment({}, semantic_features, proprio_cost, result)
        
        # Query GPR prediction and kernel-nearest memory node.
        gpr_result = self.predict_cost(semantic_features.cpu().numpy(), True)
        mu = gpr_result['mu']
        var = gpr_result['var']
        kernel_similarity_info = gpr_result.get('kernel_similarity_info', {})
        best_match = kernel_similarity_info.get('top_similar_nodes', {})
        sim = best_match.get('kernel_similarity')
        
        # Use absolute cost error to avoid division instability near zero.
        diff_mu = abs(proprio_cost - mu)
        # diff_mu = abs(proprio_cost - mu) / abs(proprio_cost)

        result = {
            'cur_feats': semantic_features,
            'cur_cost': proprio_cost,
            'nearest_nodes': best_match,
            'kernel_similarity_info': kernel_similarity_info,
            'action_taken': None,
            'new_nodes_created': [],
            'conflict_detection': None,
            'mechanism_type': None,
            'gpr_prediction': {'mu': mu, 'var': var, 'diff_mu': diff_mu}
        }
        
        # Merge: prediction is accurate and the feature is similar to memory.
        if diff_mu <= self.mu_diff and sim >= self.sim:
            print(f'执行合并机制，预测均值为{mu:.3f}，误差为{diff_mu:.3f}，相似度为{sim:.3f}')
            return self._handle_merge(
                best_match, semantic_features, proprio_cost, result
            )
        # Conflict: feature is similar but observed cost disagrees.
        elif diff_mu > self.mu_diff and sim >= self.sim:
            print(f'执行冲突机制，预测均值为{mu:.3f}，误差为{diff_mu:.3f}，相似度为{sim:.3f}')
            return self._handle_conflict(
                best_match, semantic_features, proprio_cost, result
            )
        # Increment: feature is dissimilar and should create a new experience.
        elif diff_mu > self.mu_diff and sim < self.sim:
            print(f'执行增量机制，预测均值为{mu:.3f}，误差为{diff_mu:.3f}，相似度为{sim:.3f}')
            return self._handle_increment(
                best_match, semantic_features, proprio_cost, result
            )
        # Valid inference: no memory update is needed.
        else:
            print(f'推理有效，预测均值为{mu:.3f}，误差为{diff_mu:.3f}，相似度为{sim:.3f}')
            result.update({
                'action_taken': 'no_action_needed',
                'mechanism_type': 'valid'
            })
            return result

    def _handle_conflict(self, best_match: Dict[str, Any],
                        semantic_features: torch.Tensor,
                        proprio_cost: float,
                        result: Dict[str, Any]) -> Dict[str, Any]:
        """Handle the conflict mechanism for a similar feature with mismatched cost."""
        cost_node = best_match['cost_node']
        category_label = best_match['category_label']
        
        # Count conflict observations on this memory node.
        cost_node.conflict_nums += 1
        
        # Store conflict evidence for later node replacement/update.
        conflict_info = {
            'semantic_features': semantic_features.cpu().numpy(),
            'proprio_cost': proprio_cost,
            'predicted_cost': result['gpr_prediction']['mu'],
            'timestamp': datetime.now()
        }
        cost_node.conflict_buffer.append(conflict_info)
        
        # Resolve conflicts once enough evidence has accumulated.
        train_info = {}
        if cost_node.conflict_nums >= self.conflict_times:
            train_info = self._update_features_with_best_cost_proximity(cost_node)

        
        result.update({
            'train_info':{
                'loss':train_info.get('loss', 0),
                'duration':train_info.get('duration', 0),
                'epochs':train_info.get('epochs', 0)
            },
            'action_taken': 'conflict_detected',
            'conflict_detection': {
                'category_label': category_label,
                'cost_node_id': best_match['cost_node_id'],
                'conflict_nums': cost_node.conflict_nums,
                'conflict_buffer_size': len(cost_node.conflict_buffer),
                'feature_updated': cost_node.conflict_nums >= self.conflict_times
            },
            'mechanism_type': 'conflict'
        })
        
        return result
    
    def _handle_merge(self, best_match: Dict[str, Any],
                     semantic_features: torch.Tensor,
                     proprio_cost: float,
                     result: Dict[str, Any]) -> Dict[str, Any]:
        """Handle the merge mechanism for a consistent memory observation."""
        cost_node = best_match['cost_node']
        category_label = best_match['category_label']
        
        # Store access evidence for possible centroid refinement.
        access_info = {
            'semantic_features': semantic_features.cpu().numpy(),
            'proprio_cost': proprio_cost,
            'timestamp': datetime.now()
        }
        cost_node.access_buffer.append(access_info)
        
        # Count successful accesses for this memory node.
        cost_node.access_nums += 1
        
        # Refine the representative feature after enough accesses.
        train_info = {}
        if cost_node.access_nums >= self.merge_times:
            train_info = self._update_features_with_max_similarity(cost_node)

        
        result.update({
            'train_info':{
                'loss':train_info.get('loss', 0),
                'duration':train_info.get('duration', 0),
                'epochs':train_info.get('epochs', 0)
            },
            'action_taken': 'merged_existing_node',
            'merge_info': {
                'category_label': category_label,
                'cost_node_id': best_match['cost_node_id'],
                'access_nums': cost_node.access_nums,
                'access_buffer_size': len(cost_node.access_buffer)
            },
            'mechanism_type': 'merge'
        })
        
        return result
    
    def _handle_increment(self,
                         best_match: Dict[str, Any],
                         semantic_features: torch.Tensor,
                         proprio_cost: float,
                         result: Dict[str, Any]) -> Dict[str, Any]:
        """Handle the increment mechanism by adding a new cost node."""
        # Ensure semantic features are stored as CUDA tensors.
        if not isinstance(semantic_features, torch.Tensor):
            semantic_features = torch.from_numpy(semantic_features).float().cuda()
        elif not semantic_features.is_cuda:
            semantic_features = semantic_features.cuda()
        
        # Create a new semantic category when no similar node exists.
        if not best_match:
            # Generate a new category label.
            new_category_label = f"category_{len(self.hierarchical_memory)}"
            
            # Create both category and cost node.
            self._add_semantic_category(new_category_label)
            cost_node_id = self._add_semantic_cost_node(
                new_category_label, semantic_features.cpu().numpy(), proprio_cost
            )
            
            result.update({
                'action_taken': 'created_new_category_and_cost_node',
                'new_nodes_created': [f"{new_category_label}_category", f"{new_category_label}_{cost_node_id}"],
                'mechanism_type': 'class_increment'
            })
        else:
            # Reuse the nearest category and add a new cost node under it.
            category_label = best_match['category_label']
            cost_node_id = self._add_semantic_cost_node(
                category_label, semantic_features.cpu().numpy(), proprio_cost
            )
            
            result.update({
                'action_taken': 'created_new_cost_node',
                'new_nodes_created': [f"{category_label}_{cost_node_id}"],
                'mechanism_type': 'class_increment'
            })
        
        train_info = self._update_global_gpr_model()

        result.update({'train_info':{
                'loss':train_info.get('loss', 0),
                'duration':train_info.get('duration', 0),
                'epochs':train_info.get('epochs', 0)
            }})

        return result
    
    def _update_features_with_max_similarity(self, cost_node: SemanticCostNode):
        """
        Update the node representative to the highest-similarity access sample.

        The current node itself is included in the candidate pool, so the update
        keeps the original feature when it remains the best representative.
        """
        if not cost_node.access_buffer:
            return

        # Collect access-buffer features and costs.
        features = [info['semantic_features'] for info in cost_node.access_buffer]
        costs = [info['proprio_cost'] for info in cost_node.access_buffer]

        # Add the current node itself to the candidate pool.
        current_feat = cost_node.semantic_features
        if isinstance(current_feat, torch.Tensor):
            current_feat = current_feat.cpu().numpy()
        features.append(current_feat)
        costs.append(cost_node.proprio_cost)

        features_tensor = torch.from_numpy(np.array(features)).float().cuda()
        
        best_idx = 0
        avg_sim = 0.0

        # Compute the full kernel-similarity matrix.
        self.global_gpr_model.eval()
        with torch.no_grad():
            # Pairwise kernel similarity matrix: [N+1, N+1].
            covar_matrix = self.global_gpr_model.covar_module(features_tensor).evaluate()
            total_sims = covar_matrix.sum(dim=1)
            best_idx = torch.argmax(total_sims).item()
            avg_sim = total_sims[best_idx].item() / len(features)

        # Update node representative and clear the access buffer.
        cost_node.semantic_features = features[best_idx]
        cost_node.proprio_cost = costs[best_idx]
        
        # Report whether the original node or a new sample became the centroid.
        if best_idx == len(features) - 1:
            print(f"合并机制：原节点依然是质心，维持原特征。核相似度均值: {avg_sim:.3f}")
            train_info = {}
        else:
            print(f"合并机制：经验发生微调，使用新样本作为质心。核相似度均值: {avg_sim:.3f}")
            # Refit the global GPR model after representative update.
            train_info = self._update_global_gpr_model()
            
            
        cost_node.access_buffer = []
        cost_node.access_nums = 0

        return train_info

    def _update_features_with_best_cost_proximity(self, cost_node: SemanticCostNode):
        """
        Resolve conflicts by selecting the most self-consistent candidate.

        The method freezes current GPR hyperparameters, builds a pool from the
        conflicting node and buffered observations, then uses leave-one-out
        prediction error to decide whether a stable replacement exists.
        """
        if not cost_node.conflict_buffer or self.global_gpr_model is None:
            return

        # --- Step 1: 准备基础训练数据 (排除当前冲突节点) ---
        X_train_all = self.gpr_training_data['X_normalized']  # [M, Dim]
        y_train_all = self.gpr_training_data['y_normalized']  # [M]
        node_info_all = self.gpr_training_data['node_info']
        
        # 找到非冲突节点的索引
        current_feat_raw = cost_node.semantic_features
        if not isinstance(current_feat_raw, torch.Tensor):
            current_feat_raw = torch.from_numpy(current_feat_raw).float().cuda()
        current_x_norm = (current_feat_raw- self.in_mean) / self.in_std

        keep_indices = [idx for idx, x_train in enumerate(X_train_all) 
                       if (x_train != current_x_norm).any()]
        
        X_base = X_train_all[keep_indices]
        y_base = y_train_all[keep_indices]

        # --- Step 2: 构建候选池 (缓冲区数据 + 当前节点) ---
        buffer_feats = [info['semantic_features'] for info in cost_node.conflict_buffer]
        buffer_costs = [info['proprio_cost'] for info in cost_node.conflict_buffer]
        
        # 将当前节点的旧特征和旧代价也加进来参与竞争
        current_feat = cost_node.semantic_features
        if isinstance(current_feat, torch.Tensor):
            current_feat = current_feat.cpu().numpy()
        
        # buffer_feats.append(current_feat)
        # buffer_costs.append(cost_node.proprio_cost)
        
        pool_feats = np.array(buffer_feats)
        pool_costs = np.array(buffer_costs)
        N_pool = len(pool_feats) # 此时数量为 N_buffer + 1

        # 将池中数据标准化
        pool_feats_norm = (torch.from_numpy(pool_feats).float().cuda() - self.in_mean) / self.in_std
        pool_costs_norm = (torch.from_numpy(pool_costs).float().cuda() - self.cost_mean) / self.cost_std

        total_errors = []

        # --- Step 3: 留一法模型推理循环 ---
        # 预先获取全局模型的超参数状态
        global_state = self.global_gpr_model.state_dict()
        likelihood_state = self.likelihood.state_dict()
        
        for i in range(N_pool):
            # a. 构建临时训练集: 基础数据 + 池中第 i 个候选点
            X_temp = torch.cat([X_base, pool_feats_norm[i:i+1]], dim=0)
            y_temp = torch.cat([y_base, pool_costs_norm[i:i+1]], dim=0)
            
            # b. 【核心修复】创建临时的 Likelihood 实例，防止 train_inputs 追踪冲突
            temp_likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
            temp_likelihood.load_state_dict(likelihood_state)
            
            # c. 创建临时模型并加载超参
            temp_model = ExactGPModel(X_temp, y_temp, temp_likelihood).cuda()
            temp_model.load_state_dict(global_state)
            
            # d. 切换到评估模式，锁定超参
            temp_model.eval()
            temp_likelihood.eval()
            
            # # e. 准备测试数据 (池中除了点 i 之外的所有点 j)
            # test_indices = [idx for idx in range(N_pool) if idx != i]
            # X_test = pool_feats_norm[test_indices]
            # y_test_real = pool_costs_norm[test_indices]

            # e. 准备测试数据
            X_test = pool_feats_norm
            y_test_real = pool_costs_norm
            
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                # 获取后验分布并计算 MAE
                output = temp_model(X_test)
                observed_pred = temp_likelihood(output)
                mu_pred = observed_pred.mean
                
                mae = torch.mean(torch.abs(mu_pred - y_test_real)).item()
                total_errors.append(mae)

        # --- Step 4: 择优与噪声判定 ---
        best_idx = np.argmin(total_errors)
        min_error = total_errors[best_idx]
        
        NOISE_THRESHOLD_NORM = 0.15 
        
        if min_error > NOISE_THRESHOLD_NORM:
            print(f"冲突机制 [噪声滤除]：最小预测误差 MAE={min_error:.4f} > 阈值，判定为随机噪声干扰。")
            cost_node.conflict_buffer = []
            cost_node.conflict_nums = 0
            return {}

        else:
            old_cost = cost_node.proprio_cost
            best_sample_feat = pool_feats[best_idx]
            best_sample_cost = pool_costs[best_idx]
            
            cost_node.semantic_features = best_sample_feat
            cost_node.proprio_cost = float(best_sample_cost)
            
            print(f"冲突机制：识别到稳定漂移，选取最佳观测点更新 (MAE={min_error:.4f})")
            print(f"节点代价已由 {old_cost:.4f} 更新为新稳态值 {cost_node.proprio_cost:.4f}")

            # 更新全局GPR模型
            train_info = self._update_global_gpr_model()
        
        # 重置缓冲区
        cost_node.conflict_buffer = []
        cost_node.conflict_nums = 0

        return train_info

    def _semantic_cost_node_to_dict(self, cost_node: SemanticCostNode) -> Dict[str, Any]:
        """
        将SemanticCostNode转换为可序列化的字典
        
        参数:
            cost_node: 语义代价节点
            
        返回:
            可序列化的字典
        """
        # 处理语义特征转换为列表（支持torch.Tensor和np.ndarray）
        semantic_features = cost_node.semantic_features
        if isinstance(semantic_features, torch.Tensor):
            semantic_features = semantic_features.cpu().numpy().tolist()
        elif isinstance(semantic_features, np.ndarray):
            semantic_features = semantic_features.tolist()
        
        # 处理冲突缓存中的numpy数组
        conflict_buffer = []
        # 更安全地检查conflict_buffer是否为可迭代的实际列表
        try:
            actual_conflict_buffer = list(cost_node.conflict_buffer)
        except TypeError:
            actual_conflict_buffer = []
        
        for conflict_info in actual_conflict_buffer:
            processed_conflict = {}
            for key, value in conflict_info.items():
                if isinstance(value, np.ndarray):
                    processed_conflict[key] = value.tolist()
                elif isinstance(value, torch.Tensor):
                    processed_conflict[key] = value.cpu().numpy().tolist()
                elif isinstance(value, datetime):
                    processed_conflict[key] = value.isoformat()
                else:
                    processed_conflict[key] = value
            conflict_buffer.append(processed_conflict)
        
        # 处理访问缓存中的numpy数组
        access_buffer = []
        # 更安全地检查access_buffer是否为可迭代的实际列表
        try:
            actual_access_buffer = list(cost_node.access_buffer)
        except TypeError:
            actual_access_buffer = []
        
        for access_info in actual_access_buffer:
            processed_access = {}
            for key, value in access_info.items():
                if isinstance(value, np.ndarray):
                    processed_access[key] = value.tolist()
                elif isinstance(value, torch.Tensor):
                    processed_access[key] = value.cpu().numpy().tolist()
                elif isinstance(value, datetime):
                    processed_access[key] = value.isoformat()
                else:
                    processed_access[key] = value
            access_buffer.append(processed_access)
        
        return {
            'semantic_features': semantic_features,
            'proprio_cost': cost_node.proprio_cost,
            'conflict_nums': cost_node.conflict_nums,
            'conflict_buffer': conflict_buffer,
            'access_nums': cost_node.access_nums,
            'access_buffer': access_buffer
        }
    
    def export_hierarchical_memory(self, export_dir: str = "mem_buffer/meta_memory") -> Dict[str, Any]:
        """
        导出整个hierarchical_memory结构到JSON文件
        
        参数:
            export_dir: 导出目录路径
            
        返回:
            导出结果字典
        """
        # 确保目录存在
        os.makedirs(export_dir, exist_ok=True)
        
        # 准备导出数据
        exported_categories_nums = 0
        exported_cost_nodes_nums = 0
        export_hierarchical_memory = {}
        
        for category_label, cost_nodes_data in self.hierarchical_memory.items():
            export_hierarchical_memory[category_label] = {}
            exported_categories_nums += 1
            
            for cost_node_id, cost_node in cost_nodes_data.items():
                cost_node_name = f"{category_label}_{cost_node_id}.json"
                export_hierarchical_memory[category_label][cost_node_id] = cost_node_name
                cost_node_path = os.path.join(export_dir, cost_node_name)
                
                # 将SemanticCostNode转换为可序列化的字典
                export_cost_node = self._semantic_cost_node_to_dict(cost_node)
                
                with open(cost_node_path, 'w', encoding='utf-8') as f:
                    json.dump(export_cost_node, f, ensure_ascii=False, indent=2)
                exported_cost_nodes_nums += 1
        
        export_hierarchical_memory_path = os.path.join(export_dir, "hierarchical_memory.json")
        
        # 写入JSON文件
        with open(export_hierarchical_memory_path, 'w', encoding='utf-8') as f:
            json.dump(export_hierarchical_memory, f, indent=2, ensure_ascii=False)
        
        print(f"hierarchical_memory导出完成: {export_hierarchical_memory_path}")
        return {
            'success': True,
            'export_path': export_hierarchical_memory_path,
            'total_categories': exported_categories_nums,
            'total_cost_nodes': exported_cost_nodes_nums
        }
            
    def import_hierarchical_memory(self, import_dir: str = "mem_buffer/meta_memory", vlad = None) -> Dict[str, Any]:
        """
        从JSON文件导入整个hierarchical_memory结构
        
        参数:
            import_path: 导入文件路径
            
        返回:
            导入结果字典
        """

        json_path = os.path.join(import_dir, "hierarchical_memory.json")
        if not os.path.exists(json_path):
            print(f"导入文件不存在: {json_path}")
            return {'success': False, 'error': 'Import file does not exist'}
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            # 清空现有记忆
            self.hierarchical_memory.clear()
            
            # 导入数据
            imported_categories = 0
            imported_cost_nodes = 0
            
            for category_label, cost_nodes_data in import_data.items():
                self._add_semantic_category(category_label)
                
                for cost_node_id, cost_node_name in cost_nodes_data.items():
                    cost_node_path = os.path.join(import_dir, cost_node_name)
                    if not os.path.exists(cost_node_path):
                        print(f"导入文件不存在: {cost_node_path}")
                        continue
                    with open(cost_node_path, 'r', encoding='utf-8') as f:
                        cost_node_data = json.load(f)
                    
                    # 转换列表为numpy数组
                    semantic_features = np.array(cost_node_data.get("semantic_features", []))
                    if vlad is not None and semantic_features.shape[0] != vlad.num_clusters:

                        vlad_features = vlad.get_cluster_assignments(semantic_features.reshape(1,-1))

                        semantic_features = vlad_features.squeeze(0)
                        
                    proprio_cost = cost_node_data.get("proprio_cost", 0.0)
                    # conflict_nums = cost_node_data.get("conflict_nums", 0)
                    conflict_nums = 0
                    conflict_buffer = cost_node_data.get("conflict_buffer", [])
                    # access_nums = cost_node_data.get("access_nums", 0)
                    access_nums = 0
                    access_buffer = cost_node_data.get("access_buffer", [])
                    
                    self._add_semantic_cost_node(
                        semantic_category=category_label,
                        semantic_features=semantic_features,
                        proprio_cost=proprio_cost,
                        conflict_nums=conflict_nums,
                        conflict_buffer=conflict_buffer,
                        access_nums=access_nums,
                        access_buffer=access_buffer
                    )
                    imported_cost_nodes += 1
                
                imported_categories += 1
            
            # 更新全局GPR模型
            if imported_cost_nodes > 0:
                self._update_global_gpr_model()
            
            print(f"hierarchical_memory导入完成: {json_path}")
            return {
                'success': True,
                'imported_categories': imported_categories,
                'imported_cost_nodes': imported_cost_nodes
            }
            
        except Exception as e:
            print(f"hierarchical_memory导入失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _cost_to_color(self, costs):
        """
        将代价分数映射到颜色
        映射逻辑变换:
        - 输入范围: [-1, 1]
        - 区间 [-1, 0] 映射到原 RdYlGn 的 [0, 0.2] (深红 -> 红橙)
        - 区间 (0, 1] 映射到原 RdYlGn 的 (0.2, 1.0] (红橙 -> 绿)
        
        参数:
            costs: 代价分数列表 [N]
            
        返回:
            colors: RGB颜色 [N, 3] 范围 [0, 1]
        """
        import matplotlib.cm as cm
        import numpy as np
        
        # 确保输入是numpy数组
        costs = np.array(costs)
        
        # 截断范围到 [-1, 1]
        costs = np.clip(costs, -1.0, 1.0)
        
        # 初始化映射后的归一化值数组
        norm_costs = np.zeros_like(costs, dtype=np.float32)
        
        # 逻辑 1: 原来 0-0.2 的颜色对应现在的 -1-0
        # Math: y = 0.2 * (x + 1)
        mask_neg = (costs <= 0)
        norm_costs[mask_neg] = 0.3 * (costs[mask_neg] + 1.0)
        
        # 逻辑 2: 原来 0.2-1 的颜色对应现在的 0-1
        # Math: y = 0.8 * x + 0.2
        mask_pos = (costs > 0)
        norm_costs[mask_pos] = 0.7 * costs[mask_pos] + 0.3
        
        # 获取颜色映射
        custom_cmap = cm.get_cmap('RdYlGn')
        
        # 应用颜色映射
        colors = custom_cmap(norm_costs)[:, :3]  # 取RGB，忽略alpha通道
        
        return colors

    def visualize_memory_mechanisms(self, kernel_info: dict, image_size: tuple = (1000, 800)):
            """
            更新版：
            1. 右图坐标系由 (Error, Var) 切换为 (Error, Similarity)
            2. 固定右图纵轴范围为 [0, 1.0]，反映相似度区间
            3. 保持红星底层绘制逻辑
            """
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                from matplotlib.backends.backend_agg import FigureCanvasAgg
                import cv2
                import torch
                import numpy as np
            except ImportError:
                return None

            if kernel_info is None: return None

            # 1. 基础数据准备
            cur_cost = kernel_info['cur_cost']
            # 注意：这里我们关注 mu 误差和最高相似度 sim
            diff_mu = kernel_info['gpr_prediction']['diff_mu']
            mechanism = kernel_info.get('mechanism_type', 'Unknown')

            nodes_sim = kernel_info.get('kernel_similarity_info', []).get('kernel_similarities', [])
            best_match = kernel_info.get('nearest_nodes', {})
            best_sim = best_match.get('kernel_similarity', 0.0) # 获取最高相似度
            best_sim = min(0.99, best_sim)

            dpi = 150
            fig = plt.figure(figsize=(image_size[0]/dpi, image_size[1]/dpi), dpi=dpi)
            canvas = FigureCanvasAgg(fig)
            
            ax1 = fig.add_subplot(121, projection='polar') # 左图：雷达分布
            ax2 = fig.add_subplot(122)                     # 右图：相似度决策空间

            # --- A. 左图：雷达图 (保持红星在底层) ---
            categories = list(self.hierarchical_memory.keys())
            cat_to_angle = {cat: (i * 2 * np.pi / len(categories)) for i, cat in enumerate(categories)}

            star_color = self._cost_to_color([cur_cost])[0]
            # 先画中心红星 (zorder=2)
            ax1.scatter(0, 0, c=[star_color], s=350, marker='*', 
                        edgecolors='red', linewidth=1.5, zorder=2, label='Current Input')

            if nodes_sim:
                node_costs = [item['cost_node'].proprio_cost for item in nodes_sim]
                node_colors = self._cost_to_color(node_costs) 
                for i, item in enumerate(nodes_sim):
                    sim_val = item['kernel_similarity']
                    dist = 1.0 - sim_val
                    cat = item['category_label']
                    angle = cat_to_angle[cat] + np.random.uniform(-0.1, 0.1)
                    # 后画节点 (zorder=5)
                    ax1.scatter(angle, dist, c=[node_colors[i]], 
                                s=80, edgecolors='white', alpha=0.6, zorder=5)

            ax1.set_ylim(0, 1.0) 
            ax1.set_rticks([0.2, 0.5, 0.8])
            ax1.set_yticklabels(['Sim 0.8', 'Sim 0.5', 'Sim 0.2'], fontsize=8)
            ax1.set_title(f"Similarity Radar", fontsize=11, pad=15)

            # --- B. 右图：相似度决策空间 (基于 Sim 替换 Var) ---
            MAX_ERR = 0.5 
            # 相似度纵轴固定为 0-1.0
            
            # 绘制背景色块，注意纵轴逻辑反转：相似度高在上方
            # 1. Merge 区域：Error小 且 Sim高 (绿色)
            ax2.add_patch(plt.Rectangle((0, self.sim), self.mu_diff, 1.0 - self.sim, color='green', alpha=0.1))
            # 2. Conflict 区域：Error大 且 Sim高 (红色)
            ax2.add_patch(plt.Rectangle((self.mu_diff, self.sim), MAX_ERR, 1.0 - self.sim, color='red', alpha=0.1))
            # 3. Increment 区域：Sim低 (橙色)
            ax2.add_patch(plt.Rectangle((0, 0), MAX_ERR + self.mu_diff, self.sim, color='orange', alpha=0.1))

            # 当前决策星星：(Error, Similarity)
            ax2.scatter(diff_mu, best_sim, c=[star_color], s=150, marker='*', edgecolors='black', zorder=10)
            
            # 绘制阈值线
            ax2.axvline(x=self.mu_diff, color='black', linestyle='--', alpha=0.3)
            ax2.axhline(y=self.sim, color='black', linestyle='--', alpha=0.3)
            
            ax2.set_xlim(0, MAX_ERR)
            ax2.set_ylim(0, 1.0) # 相似度范围固定 0-1
            
            ax2.set_xlabel("GPR Prediction Error")
            ax2.set_ylabel("Max Kernel Similarity")
            ax2.set_title(f"Mechanism: {mechanism.upper()}", fontsize=11, fontweight='bold')

            # --- C. 底部信息栏 ---
            info_text = (f"Target Cost: {cur_cost:.2f}\n"
                        f"GPR Error: {diff_mu:.3f}\n"
                        f"Best Sim: {best_sim:.3f}\n"
                        f"Mechanism: {mechanism.upper()}")
            
            fig.text(0.05, 0.02, info_text, ha='left', va='bottom', fontsize=10, 
                    linespacing=1.5,
                    bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.1, ec="gray"))

            plt.tight_layout(rect=[0, 0.08, 1, 0.95])
            
            canvas.draw()
            image_bgr = cv2.cvtColor(np.asarray(canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
            plt.close(fig)
            return image_bgr
    
    def import_node(self, node_path, vlad = None):
        if not os.path.exists(node_path):
            print(f"导入文件不存在: {node_path}")
            return {'success': False, 'error': 'Import file does not exist'}
        
        try:
            with open(node_path, 'r', encoding='utf-8') as f:
                cost_node_data = json.load(f)
                    
                # 转换列表为numpy数组
                semantic_features = np.array(cost_node_data.get("semantic_features", []))
                if vlad is not None and semantic_features.shape[0] != vlad.num_clusters:

                    vlad_features = vlad.get_cluster_assignments(semantic_features.reshape(1,-1))

                    semantic_features = vlad_features.squeeze(0)
                    
                proprio_cost = cost_node_data.get("proprio_cost", 0.0)

                proprio_cost = 0.8
                
                mem_info = self._online_learning_update_with_gpr(semantic_features, proprio_cost)

                # proprio_cost = cost_node_data.get("proprio_cost", 0.0)
                # proprio_cost = 0.8
                # conflict_nums = 0
                # conflict_buffer = cost_node_data.get("conflict_buffer", [])
                # # access_nums = cost_node_data.get("access_nums", 0)
                # access_nums = 0
                # access_buffer = cost_node_data.get("access_buffer", [])
                
                # self._add_semantic_cost_node(
                #     semantic_category='quick_add',
                #     semantic_features=semantic_features,
                #     proprio_cost=proprio_cost,
                #     conflict_nums=conflict_nums,
                #     conflict_buffer=conflict_buffer,
                #     access_nums=access_nums,
                #     access_buffer=access_buffer
                # )

                return mem_info
            
        except Exception as e:
            print(f"hierarchical_memory导入失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
