"""Data-buffer GPR memory used by the ``max_similiarity_out`` mode.

This module keeps a fixed-size feature/cost buffer, updates it online, and
trains a lightweight GPR model for traversability cost prediction.
"""

import numpy as np
import torch
import gpytorch
from scipy import stats
from typing import Dict, Any, List, Optional
import warnings
import math
warnings.filterwarnings('ignore')
import time

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, feat_dim=32):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        ls_constraint = gpytorch.constraints.Interval(0.01, math.sqrt(feat_dim))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=0.5, lengthscale_constraint=ls_constraint)
        )
        # self.covar_module = gpytorch.kernels.ScaleKernel(
        #     gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=feat_dim, lengthscale_constraint=ls_constraint)
        # ) 


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class DataBufferCostGPR:
    """Fixed-size data-buffer memory with GPR cost prediction."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize buffer state and GPR configuration."""
        self.config = config or self._get_default_config()
        
        # GPR state is initialized lazily after the first valid feature arrives.
        self.gp_model: Optional[ExactGPModel] = None
        self.likelihood = self.likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Interval(1e-2, 0.5)).cuda()
        self.gp_params: Optional[Dict] = None
        
        self.buffer_size = self.config['buffer_size']
        
        # Buffer tensors are allocated after feature dimensionality is known.
        self.feature_dim = None
        self.train_in_buffer = None
        self.train_label_buffer = None
        self.train_buffer_classes = None
        self.time_of_insertion = None
        self.buffer_idx = 0
        self.buffer_full = False
        self.buffer_initialized = False
        
        # Running normalization statistics for feature and cost channels.
        self.in_mean = None
        self.in_std = None
        self.cost_mean = None
        self.cost_std = None
        
        # Protect imported prior samples from online replacement.
        self.num_initial_inserts = 0
        self.velocity_threshold = None
         
    def _initialize_buffer_with_prior_knowledge(self, import_dir: str = "data_buffer", vlad = None):
        """Warm-start the buffer from exported prior memory nodes."""
        import json
        import os

        json_path = os.path.join(import_dir, "hierarchical_memory.json")
        if not os.path.exists(json_path):
            print(f"先验知识文件不存在: {json_path}，使用默认初始化")
            return
        
        try:
            # Load the same prior-memory format used by the dynamic memory backend.
            with open(json_path, 'r', encoding='utf-8') as f:
                hierarchical_data = json.load(f)
            
            avoid_data_list = []
            avoid_features_list = []
            
            for category_label in hierarchical_data.keys():
                    
                cost_nodes_data = hierarchical_data[category_label]
                
                for cost_node_id, cost_node_filename in cost_nodes_data.items():
                    cost_node_path = os.path.join(import_dir, cost_node_filename)
                    if not os.path.exists(cost_node_path):
                        continue
                        
                    with open(cost_node_path, 'r', encoding='utf-8') as f:
                        cost_node_data = json.load(f)
                    
                    semantic_features = np.array(cost_node_data.get("semantic_features", []))
                    proprio_cost = cost_node_data.get("proprio_cost", 0)
                    
                    if len(semantic_features) > 0:
                        avoid_features_list.append(semantic_features)
                        avoid_data_list.append(proprio_cost)
            
            num_insert = len(avoid_data_list)
            self.num_initial_inserts = num_insert
            
            print(f"从{import_dir}加载了{len(avoid_data_list)}个样本，将插入{num_insert}个")
            
            # Convert prior features to the active representation before insertion.
            for i in range(num_insert):
                features = torch.from_numpy(avoid_features_list[i]).float().cuda()

                if vlad is not None and features.shape[0] != vlad.num_clusters:
                    encoded_features = vlad.get_cluster_assignments(features.reshape(1,-1))

                    features = encoded_features.squeeze(0)

                cost = avoid_data_list[i]
                
                if not self.buffer_initialized:
                    self._initialize_buffer(len(features))

                if len(features) != self.feature_dim:
                    print(f"警告：特征维度不匹配，期望{self.feature_dim}，实际{len(features)}，跳过该样本")
                    continue
                
                self.train_in_buffer[i] = features
                self.train_label_buffer[i] = torch.tensor([cost]).cuda()
                self.train_buffer_classes[i] = self.feature_dim
                self.time_of_insertion[i] = 0
            
            self.buffer_idx = num_insert
            print(f"缓冲区初始化完成，从目录导入了{num_insert}个避障先验样本")

            self.update_gpr_model()
            
        except Exception as e:
            print(f"从目录导入先验知识失败: {e}，使用默认初始化")

    def _initialize_buffer(self, feature_dim: int):
        """Allocate fixed-size CUDA tensors after feature dimension is known."""
        self.feature_dim = feature_dim
        
        print(f"初始化缓冲区，特征维度: {feature_dim}")
        
        self.train_in_buffer = torch.zeros((self.buffer_size, feature_dim)).cuda()
        self.train_label_buffer = torch.zeros((self.buffer_size, 1)).cuda()
        self.train_buffer_classes = np.zeros((self.buffer_size))
        self.time_of_insertion = np.zeros((self.buffer_size))
        
        # Feature-only statistics; cost statistics are refreshed during GPR update.
        self.in_mean = torch.zeros(feature_dim).cuda()
        self.in_std = torch.ones(feature_dim).cuda()
        
        self.buffer_initialized = True
    
    def update_train_buffer(self, features: np.ndarray, cost: float, terrain_class: int = None):
        """Insert one feature/cost sample into the data buffer."""
        features_tensor = torch.from_numpy(features).float().cuda()
        cost_tensor = torch.tensor([cost]).float().cuda()
        
        if not self.buffer_initialized:
            self._initialize_buffer(len(features))
        
        if len(features) != self.feature_dim:
            print(f"警告: 特征维度不匹配，期望 {self.feature_dim}，实际 {len(features)}")
            return
        
        input_sample = features_tensor
        
        # All-zero features indicate unknown/unprojected space and are ignored.
        if torch.count_nonzero(input_sample) == 0:
            print("在未知空间，不添加到缓冲区")
            return
        
        if terrain_class is None:
            terrain_class = 0
            print("terrain_class为None，使用默认值0")
        
        if self.buffer_full:
            self._smart_buffer_replacement(input_sample, cost_tensor, terrain_class)
        else:
            self._simple_buffer_append(input_sample, cost_tensor, terrain_class)
    
    def _simple_buffer_append(self, input_sample: torch.Tensor, cost: torch.Tensor, terrain_class: int):
        """Append a sample while the fixed-size buffer still has free slots."""
        self.train_in_buffer[self.buffer_idx] = input_sample
        self.train_label_buffer[self.buffer_idx] = cost
        self.train_buffer_classes[self.buffer_idx] = terrain_class
        self.buffer_idx = min(self.buffer_idx + 1, self.buffer_size)
        
        if self.buffer_idx == self.buffer_size:
            self.buffer_full = True
            print("缓冲区已满，将启用智能替换策略")
    
    def _smart_buffer_replacement(self, input_sample: torch.Tensor, cost: torch.Tensor, terrain_class: int):
        """
        Replace one sample in a full buffer while preserving seed samples.

        The policy targets over-represented terrain/cost regions so the buffer
        stays more balanced than plain FIFO replacement.
        """
        # Find the most common terrain class in the buffer.
        if len(self.train_buffer_classes) > 0:
            most_class = stats.mode(self.train_buffer_classes)[0]
        else:
            most_class = terrain_class
        
        # Analyze the cost distribution for that class.
        if np.sum(self.train_buffer_classes == most_class) > 0:
            # Restrict candidates to the dominant terrain class.
            class_mask = self.train_buffer_classes == most_class
            class_costs = self.train_in_buffer[class_mask, -1].cpu().numpy()
            
            # Select the densest cost bin as the replacement target.
            hist, bin_edges = np.histogram(class_costs, bins=5, range=(0, 2.0))
            most_bin = np.argmax(hist)
            
            sel_min = bin_edges[most_bin]
            sel_max = bin_edges[most_bin + 1]
            
            # Randomly replace one candidate in the dominant class/bin.
            candidate_indices = np.where(
                (self.train_buffer_classes == most_class) & 
                (self.train_in_buffer[:, -1].cpu().numpy() > sel_min) & 
                (self.train_in_buffer[:, -1].cpu().numpy() < sel_max)
            )[0]
            
            if len(candidate_indices) > 0:
                # Protect the manually seeded prior samples.
                valid_indices = [idx for idx in candidate_indices if idx >= self.num_initial_inserts]
                
                if len(valid_indices) > 0:
                    insert_idx = np.random.choice(valid_indices)
                    
                    self.train_in_buffer[insert_idx] = input_sample
                    self.train_label_buffer[insert_idx] = cost
                    self.train_buffer_classes[insert_idx] = terrain_class
                    
                    print(f"智能替换：在类别{most_class}的cost区间[{sel_min:.2f}, {sel_max:.2f}]中替换了样本{insert_idx}")
                    return
        
        # Fall back to FIFO when class/bin selection has no valid candidate.
        self._fifo_buffer_replacement(input_sample, cost, terrain_class)
    
    def _fifo_buffer_replacement(self, input_sample: torch.Tensor, cost: torch.Tensor, terrain_class: int):
        """Replace the oldest non-seed sample."""
        # Find the oldest inserted sample while excluding seed samples.
        valid_indices = np.where(self.time_of_insertion > 0)[0]
        if len(valid_indices) > 0:
            oldest_idx = np.argmin(self.time_of_insertion[valid_indices])
            insert_idx = valid_indices[oldest_idx]
        else:
            # If timestamps are unavailable, choose a non-seed slot randomly.
            insert_idx = np.random.randint(self.num_initial_inserts, self.buffer_size)
        
        self.train_in_buffer[insert_idx] = input_sample
        self.train_label_buffer[insert_idx] = cost
        self.train_buffer_classes[insert_idx] = terrain_class
        
        print(f"FIFO替换：替换了样本{insert_idx}")
    
    def update_gpr_model(self):
        """
        Refit the GPR model from the current data buffer.

        The caller controls when this function is invoked. In POLT presets it is
        typically called at a fixed interval or during the first initialization.
        """
        print('****************************************************')
        print(f'更新GPR模型，缓冲区索引: {self.buffer_idx}')
        
        if not self.buffer_full:
            train_data_in_buffer = self.train_in_buffer[:self.buffer_idx]
            train_cost_in_buffer = self.train_label_buffer[:self.buffer_idx, 0]
        else:
            train_data_in_buffer = self.train_in_buffer
            train_cost_in_buffer = self.train_label_buffer[:, 0]
        
        if train_data_in_buffer.shape[0] < 2:
            return {}
        
        # Normalize input features; current policy keeps unit variance per feature.
        train_data = train_data_in_buffer
        self.in_mean = torch.mean(train_data, dim=0)
        # self.in_std = torch.std(train_data, dim=0)
        # self.in_std[self.in_std == 0] = 1.0
        self.in_std = torch.ones_like(self.in_mean)
        train_data_normalized = (train_data - self.in_mean) / self.in_std
        
        cost = train_cost_in_buffer
        self.cost_mean = torch.mean(cost)
        self.cost_std = torch.std(cost)
        if self.cost_std < 1e-6:
            self.cost_std = torch.tensor(1.0).cuda()
            
        cost_normalized = (cost - self.cost_mean) / self.cost_std

        # Build a fresh exact GPR model from the current buffer state.
        self.gp_model = ExactGPModel(
            train_data_normalized, 
            cost_normalized,
            self.likelihood
        ).cuda()
        
        # if self.gp_params is not None:
        #     self.gp_model.load_state_dict(self.gp_params)

        
        # Train kernel hyperparameters if enabled.
        train_info = {}
        if self.config.get('train_kernel', True) and train_data_in_buffer.shape[0] >= 2:
            train_info = self._train_gpr_kernel(train_data_normalized, cost_normalized)

        print(f"GPR模型更新成功，训练样本数: {train_data_in_buffer.shape[0]}")

        return train_info

    def _train_gpr_kernel(self, train_data: torch.Tensor, cost: torch.Tensor, max_iterations: int = 100):
        """Train GPR kernel hyperparameters with L-BFGS."""
        optimizer = torch.optim.LBFGS(
        self.gp_model.parameters(), 
        lr=1, 
        max_iter=max_iterations, 
        line_search_fn='strong_wolfe'
        )

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp_model)
    
        print(f"开始 GPR 训练 (N={len(train_data)}, L-BFGS)...")

        loss_history = []
        # L-BFGS requires a closure to recompute the objective.
        def closure():
            optimizer.zero_grad()
            output = self.gp_model(train_data)
            loss = -mll(output, cost)
            loss.backward()
            loss_history.append(loss.item())
            return loss
        
        start_train = time.perf_counter()

        optimizer.step(closure)
        
        # Recompute the final marginal likelihood for diagnostics.
        with torch.no_grad():
            final_output = self.gp_model(train_data)
            final_loss = -mll(final_output, cost)
            
        end_train = time.perf_counter()
        train_duration = (end_train - start_train) * 1000

        # LBFGS defaults max_eval to 1.25 * max_iter.
        actual_evals = len(loss_history)
        is_converged = actual_evals < (max_iterations * 1.25)
        
        print(f"GPR 训练完成 - Final Loss: {final_loss.item():.4f}, 训练耗时: {train_duration:.2f} ms")
        print(f"实际函数评估次数: {actual_evals}")
        if is_converged:
            print(f"结论: 模型已提前收敛！(满足 tolerance_change/grad 阈值)")
        else:
            print(f"结论: 未完全收敛，达到了最大迭代步数 ({max_iterations})。")
        
        self.gp_params = self.gp_model.state_dict()

        train_info = {
            'loss':final_loss,
            'duration':train_duration,
            'epochs':actual_evals
        }

        return train_info
    
    def predict_cost(self, feature_vector: np.ndarray, return_uncertainty: bool = True) -> Dict[str, Any]:
        """
        Predict traversability cost for one feature vector.

        Args:
            feature_vector: Numpy array with shape ``(feature_dim,)`` or
                ``(1, feature_dim)``.
            return_uncertainty: Whether to include predictive variance.

        Returns:
            Dictionary with mean cost, uncertainty, CVaR-adjusted cost, and
            validity flags.
        """
        # Return an unknown-cost fallback before the model is initialized.
        if self.gp_model is None:
            return {
                'predicted_cost': self.config['unknown_cost'],
                'uncertainty': 1.0,
                'cvar_adjusted_cost': self.config['unknown_cost'],
                'is_valid': False,
                'is_unknown': False
            }

        # Ensure shape (1, D) and move to half-precision CUDA tensors.
        features_tensor = torch.from_numpy(feature_vector).float().cuda()
        if features_tensor.dim() == 1:
            features_tensor = features_tensor.unsqueeze(0)
        
        features_tensor = features_tensor.half()

        start_train = time.perf_counter()
        # Run exact GPR prediction without gradient tracking.
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            self.gp_model.eval()
            self.likelihood.eval()

            # All-zero features indicate unknown/unprojected space.
            if torch.count_nonzero(features_tensor) == 0:
                return {
                    'predicted_cost': self.config['unknown_cost'],
                    'uncertainty': 1.0 if return_uncertainty else None,
                    'cvar_adjusted_cost': self.config['unknown_cost'],
                    'is_valid': False,
                    'is_unknown': True
                }

            # Normalize with the statistics from the latest fitted buffer.
            normalized_features = (features_tensor - self.in_mean) / self.in_std

            # GPyTorch Cholesky operations are more stable in float32.
            gp_model_f32 = self.gp_model.float()
            likelihood_f32 = self.likelihood.float()
            
            observed_pred = likelihood_f32(gp_model_f32(normalized_features.float()))
            
            # Convert predictions back to the runtime half-precision state.
            mu = observed_pred.mean.half()
            var = observed_pred.variance.half()
            
        end_train = time.perf_counter()
        train_duration = (end_train - start_train) * 1000
        print(f"单次推理耗时: {train_duration:.2f} ms") #

        # Restore the model's half-precision runtime state.
        self.gp_model.half()
        self.likelihood.half()

        # Denormalize cost mean and variance.
        cost_pred = (mu * self.cost_std) + self.cost_mean
        cost_var = var * (self.cost_std ** 2)

        # Apply optional CVaR-style conservative cost inflation.
        cvar_alpha = self.config['costmap_cvar']
        if cvar_alpha > 0:
            phi = stats.norm.pdf(stats.norm.ppf(cvar_alpha))
            cvar_cost = cost_pred + (cost_var * phi) / (1.0 - cvar_alpha)
        else:
            cvar_cost = cost_pred

        return {
            'predicted_cost': float(cost_pred.cpu().item()),
            'uncertainty': float(cost_var.cpu().item()) if return_uncertainty else None,
            'cvar_adjusted_cost': float(cvar_cost.cpu().item()),
            'is_valid': True,
            'is_unknown': False
        }

    def predict_cost_batch(self, feature_batch: np.ndarray, batch_size: int = 10000, return_uncertainty: bool = True) -> List[Dict[str, Any]]:
        """
        Predict traversability cost for a batch of feature vectors.

        Batch inference keeps intermediate tensors on GPU and transfers results
        to CPU in chunks, which is substantially faster than per-point calls.
        """
        # Return fallback records before the model is fitted.
        if self.gp_model is None:
            default_result = {
                'predicted_cost': 1.0,
                'uncertainty': 1.0,
                'cvar_adjusted_cost': self.config.get('unknown_cost', 1.0),
                'is_valid': False
            }
            return [default_result] * len(feature_batch)
        
        results = []
        num_points = len(feature_batch)
        
        # Convert the whole batch once to avoid repeated CPU/GPU transfers.
        all_features_tensor = feature_batch.half().cuda()
        
        # Enable fast predictive variance and run in evaluation mode.
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            self.gp_model.eval()
            self.likelihood.eval()
            
            for i in range(0, num_points, batch_size):
                end_idx = min(i + batch_size, num_points)
                
                # Slice the current GPU chunk.
                batch_features_tensor = all_features_tensor[i:end_idx]
                
                # Detect unknown/unprojected points with vectorized tensor ops.
                is_unknown_mask = (torch.count_nonzero(batch_features_tensor, dim=1) == 0)
                
                # Normalize with statistics from the latest fitted buffer.
                feature_batch_normalized = (batch_features_tensor - self.in_mean) / self.in_std
                
                # Use float32 for GPyTorch Cholesky compatibility.
                feature_batch_float32 = feature_batch_normalized.float()
                
                gp_model_float32 = self.gp_model.float()
                likelihood_float32 = self.likelihood.float()
                
                observed_pred = likelihood_float32(gp_model_float32(feature_batch_float32))
                cost_pred_batch = observed_pred.mean.half()
                cost_var_batch = observed_pred.variance.half()
                
                # Restore half precision after the GPR call.
                self.gp_model = self.gp_model.half()
                self.likelihood = self.likelihood.half()

                # Denormalize on GPU.
                cost_pred_original = (cost_pred_batch * self.cost_std) + self.cost_mean
                cost_var_original = cost_var_batch * (self.cost_std ** 2)
                
                # Apply optional CVaR inflation on GPU.
                cvar_alpha = self.config['costmap_cvar']
                if cvar_alpha > 0:
                    phi = stats.norm.pdf(stats.norm.ppf(cvar_alpha))
                    cvar_adjusted_batch = cost_pred_original + (cost_var_original * phi) / (1.0 - cvar_alpha)
                else:
                    cvar_adjusted_batch = cost_pred_original
                
                # Transfer a full chunk to CPU once instead of calling .item().
                cost_pred_np = cost_pred_original.cpu().numpy()
                cost_var_np = cost_var_original.cpu().numpy()
                cvar_np = cvar_adjusted_batch.cpu().numpy()
                is_unknown_np = is_unknown_mask.cpu().numpy()
                
                # Assemble Python result dictionaries on CPU.
                batch_len = len(batch_features_tensor)
                unknown_cost = self.config['unknown_cost']
                
                for j in range(batch_len):
                    if is_unknown_np[j]:
                        results.append({
                            'predicted_cost': unknown_cost,
                            'uncertainty': 1.0 if return_uncertainty else None,
                            'cvar_adjusted_cost': unknown_cost,
                            'is_valid': False,
                            'is_unknown': True
                        })
                    else:
                        results.append({
                            'predicted_cost': float(cost_pred_np[j]),
                            'uncertainty': float(cost_var_np[j]) if return_uncertainty else None,
                            'cvar_adjusted_cost': float(cvar_np[j]),
                            'is_valid': True,
                            'is_unknown': False
                        })
        
        return results

    def export_buffer_to_hierarchical_json(self, export_dir: str = "mem_buffer/exported_from_buffer") -> Dict[str, Any]:
        """
        Export the data buffer using the hierarchical-memory JSON layout.

        The export creates a ``hierarchical_memory.json`` index and one node
        JSON file per buffered sample.
        """
        import json
        import os
        from datetime import datetime

        if not self.buffer_initialized or self.buffer_idx == 0:
            print("缓冲区为空，无需导出。")
            return {'success': False, 'error': 'Buffer is empty'}

        os.makedirs(export_dir, exist_ok=True)
        
        # Prepare the main index that maps category/node IDs to JSON files.
        export_index = {}
        total_samples = self.buffer_idx if not self.buffer_full else self.buffer_size
        exported_count = 0

        print(f"开始导出 {total_samples} 个样本到 {export_dir}...")

        # Write each buffer item as a standalone memory node.
        for i in range(total_samples):
            # Use terrain class as the coarse category label.
            terrain_class = int(self.train_buffer_classes[i])
            category_label = f"class_{terrain_class}"
            
            if category_label not in export_index:
                export_index[category_label] = {}

            # Build stable node IDs and file names.
            node_id = f"node_{i}"
            node_filename = f"{category_label}_{node_id}.json"
            export_index[category_label][node_id] = node_filename

            # Align exported fields with the SemanticCostNode format.
            semantic_features = self.train_in_buffer[i].cpu().numpy().tolist()
            proprio_cost = float(self.train_label_buffer[i].cpu().item())
            
            node_data = {
                'semantic_features': semantic_features,
                'proprio_cost': proprio_cost,
                'conflict_nums': 0,
                'conflict_buffer': [],
                'access_nums': 1,
                'access_buffer': [
                    {
                        'semantic_features': semantic_features,
                        'proprio_cost': proprio_cost,
                        'timestamp': datetime.now().isoformat()
                    }
                ]
            }

            # Write one node file.
            node_path = os.path.join(export_dir, node_filename)
            with open(node_path, 'w', encoding='utf-8') as f:
                json.dump(node_data, f, ensure_ascii=False, indent=2)
            
            exported_count += 1

        # Write the hierarchical_memory.json index file.
        index_path = os.path.join(export_dir, "hierarchical_memory.json")
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(export_index, f, indent=2, ensure_ascii=False)

        print(f"导出完成！共导出 {exported_count} 个节点。索引文件：{index_path}")
        return {
            'success': True,
            'export_path': index_path,
            'total_exported': exported_count
        }
