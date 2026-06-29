"""Temporal PINNs model for proprioception-based mechanical feedback."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import torch.nn.functional as F

def compute_dugoff_tire_forces(theta: torch.Tensor, omega_r: torch.Tensor, pitch: torch.Tensor, roll: torch.Tensor, vx: torch.Tensor, vy: torch.Tensor,
                              ax: torch.Tensor, ay: torch.Tensor, omega_fl: torch.Tensor, omega_fr: torch.Tensor,
                              omega_rl: torch.Tensor, omega_rr: torch.Tensor,
                              Cx: torch.Tensor, Cy: torch.Tensor, mu: torch.Tensor,
                              M: torch.Tensor, Iz: torch.Tensor, LF: torch.Tensor, LR: torch.Tensor,
                              BF: torch.Tensor, BR: torch.Tensor, h: torch.Tensor, R: torch.Tensor,
                              K_rollf: torch.Tensor, K_rollr: torch.Tensor, g: float = 9.8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute tire forces with the Dugoff tire model.

    Returns:
        Fx: Longitudinal tire forces with shape ``[batch_size, 4]`` ordered as
            ``[fl, fr, rl, rr]``.
        Fy: Lateral tire forces with shape ``[batch_size, 4]`` ordered as
            ``[fl, fr, rl, rr]``.
    """
    # 1. Compute tire vertical loads with pitch/roll effects.
    L = LF + LR  # Wheelbase.
    
    # Per-wheel vertical loads.
    Fz_fl = (M * g * LR * torch.cos(pitch)) / (2 * L) - (M * ax * h) / (2 * L) - (M * g * h * torch.sin(pitch)) / (2 * L) - (M * ay * LR * h) / (2 * L * BF) - (K_rollf * roll) / (2 * BF) 
    Fz_fr = (M * g * LR * torch.cos(pitch)) / (2 * L) - (M * ax * h) / (2 * L) - (M * g * h * torch.sin(pitch)) / (2 * L) + (M * ay * LR * h) / (2 * L * BF) + (K_rollf * roll) / (2 * BF)
    Fz_rl = (M * g * LF * torch.cos(pitch)) / (2 * L) + (M * ax * h) / (2 * L) + (M * g * h * torch.sin(pitch)) / (2 * L) - (M * ay * LF * h) / (2 * L * BR) - (K_rollr * roll) / (2 * BR)
    Fz_rr = (M * g * LF * torch.cos(pitch)) / (2 * L) + (M * ax * h) / (2 * L) + (M * g * h * torch.sin(pitch)) / (2 * L) + (M * ay * LF * h) / (2 * L * BR) + (K_rollr * roll) / (2 * BR)
    
    # Stack vertical loads: [batch_size, 4].
    Fz = torch.stack([Fz_fl, Fz_fr, Fz_rl, Fz_rr], dim=1)
    
    # 2. Compute wheel-center longitudinal velocities.
    u_fl = (vx - BF/2 * omega_r) * torch.cos(theta) + (vy + LF * omega_r) * torch.sin(theta)
    u_fr = (vx + BF/2 * omega_r) * torch.cos(theta) + (vy + LF * omega_r) * torch.sin(theta)
    u_rl = vx - BR/2 * omega_r
    u_rr = vx + BR/2 * omega_r
    
    # 3. Compute longitudinal slip ratio.
    # Stack wheel angular speeds: [batch_size, 4].
    omega = torch.stack([omega_fl, omega_fr, omega_rl, omega_rr], dim=1)
    # Stack wheel-center velocities: [batch_size, 4].
    u = torch.stack([u_fl, u_fr, u_rl, u_rr], dim=1)
    
    eps = 1e-8
    omega_R = omega * R
    s = torch.where(u < omega_R,
                   (omega_R - u) / (omega_R + eps), 
                   (omega_R - u) / (u + eps))       
    
    # 4. Compute tire slip angles.
    alpha_fl = theta - torch.atan2(vy + LF * omega_r, vx - BF/2 * omega_r + eps)
    alpha_fr = theta - torch.atan2(vy + LF * omega_r, vx + BF/2 * omega_r + eps)
    alpha_rl = -torch.atan2(vy - LR * omega_r, vx - BR/2 * omega_r + eps)
    alpha_rr = -torch.atan2(vy - LR * omega_r, vx + BR/2 * omega_r + eps)
    
    # Stack slip angles: [batch_size, 4].
    alpha = torch.stack([alpha_fl, alpha_fr, alpha_rl, alpha_rr], dim=1)
    
    # 5. Compute Dugoff tire forces using helper f(sigma).
    sigma = (mu.unsqueeze(1) * Fz * (1 + s)) / (2 * torch.sqrt(Cx**2 * s**2 + Cy**2 * torch.tan(alpha)**2 + eps))
    f_sigma = torch.where(sigma <= 1, (2 - sigma) * sigma, torch.tensor(1.0, device=sigma.device))
    
    # Longitudinal force: [batch_size, 4].
    Fx = Cx * s / (1 + s + eps) * f_sigma
    # Lateral force: [batch_size, 4].
    Fy = Cy * torch.tan(alpha) / (1 + torch.abs(alpha) + eps) * f_sigma
    
    return Fx, Fy

def physics_loss_mu(mu_pred: torch.Tensor, theta: torch.Tensor, omega_r: torch.Tensor, pitch: torch.Tensor, roll: torch.Tensor, vx: torch.Tensor, vy: torch.Tensor,
                   ax: torch.Tensor, ay: torch.Tensor, omega_fl: torch.Tensor, omega_fr: torch.Tensor,
                   omega_rl: torch.Tensor, omega_rr: torch.Tensor,
                   Cx: torch.Tensor, Cy: torch.Tensor,
                   M: torch.Tensor, Iz: torch.Tensor, LF: torch.Tensor, LR: torch.Tensor,
                   BF: torch.Tensor, BR: torch.Tensor, h: torch.Tensor, R: torch.Tensor,
                   K_rollf: torch.Tensor, K_rollr: torch.Tensor, g: float = 9.8) -> torch.Tensor:
    """
    Compute physics-consistency loss for the predicted adhesion coefficient.

    The loss compares measured accelerations with accelerations implied by the
    Dugoff tire forces and vehicle dynamics.
    """
    # Compute per-wheel forces from the predicted adhesion coefficient.
    # Fx/Fy shape: [batch_size, 4] ordered as [fl, fr, rl, rr].
    Fx, Fy = compute_dugoff_tire_forces(
        theta, omega_r, pitch, roll, vx, vy, ax, ay, omega_fl, omega_fr, omega_rl, omega_rr,
        Cx, Cy, mu_pred, M, Iz, LF, LR, BF, BR, h, R, K_rollf, K_rollr, g
    )
    
    # Vehicle dynamics equilibrium equations:
    # M * ax = (F_xfl + F_xfr) * cos(theta) - (F_yfl + F_yfr) * sin(theta) + F_xrl + F_xrr
    # M * ay = (F_xfl + F_xfr) * sin(theta) - (F_yfl + F_yfr) * cos(theta) + F_yrl + F_yrr
    
    # Extract front/rear longitudinal and lateral force sums.
    # Fx columns: front-left, front-right, rear-left, rear-right.
    
    # Fy columns use the same wheel ordering.
    
    # Front axle force sums.
    Fx_front = Fx[:, 0] + Fx[:, 1]  # F_xfl + F_xfr
    Fy_front = Fy[:, 0] + Fy[:, 1]  # F_yfl + F_yfr
    
    # Rear axle force sums.
    Fx_rear = Fx[:, 2] + Fx[:, 3]   # F_xrl + F_xrr
    Fy_rear = Fy[:, 2] + Fy[:, 3]   # F_yrl + F_yrr
    
    # Compute implied longitudinal acceleration.
    ax_calc = (Fx_front * torch.cos(theta) - Fy_front * torch.sin(theta) + Fx_rear - M * g * torch.sin(pitch)) / M 
    
    # Compute implied lateral acceleration.
    ay_calc = (Fx_front * torch.sin(theta) - Fy_front * torch.cos(theta) + Fy_rear - M * g * torch.sin(roll)) / M
    
    # Acceleration consistency loss.
    l1_loss = nn.L1Loss(reduction='mean')
    ax_error = l1_loss(ax_calc, ax)
    ay_error = l1_loss(ay_calc, ay)
    # Optional full loss: ax_error + ay_error.
    # phys_loss = ax_error + ay_error
    phys_loss = ax_error
    return phys_loss

class TemporalEncoderLayer(nn.Module):
    """
    Standard Transformer encoder layer used for temporal denoising.

    Structure: MultiheadAttention -> Add & Norm -> Feed Forward -> Add & Norm.
    """
    def __init__(self, feature_dim, nhead=4, dim_feedforward=128, dropout=0.1):
        super().__init__()
        # batch_first=True 让输入输出都是 [Batch, Time, Feature]
        self.self_attn = nn.MultiheadAttention(feature_dim, nhead, dropout=dropout, batch_first=True)
        
        # Feed Forward Network (FFN)
        self.linear1 = nn.Linear(feature_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, feature_dim)

        # Normalization Layers
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        
        # Dropouts
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, x):
        """
        x: [Batch, Time, Feature]
        """
        # 1. Self-Attention (无 Mask，双向可见)
        # attn_output 包含了所有时间步的上下文信息
        attn_output, _ = self.self_attn(x, x, x, need_weights=False)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # 2. Feed Forward
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        
        return x

class TemporalEncoder(nn.Module):
    """
    时序编码器主体
    """
    def __init__(self, feature_dim, n_layers=2, nhead=4, dim_feedforward=128):
        super().__init__()
        
        # 堆叠多层 Encoder Layer
        self.layers = nn.ModuleList([
            TemporalEncoderLayer(feature_dim, nhead=nhead, dim_feedforward=dim_feedforward) 
            for _ in range(n_layers)
        ])
        
        # 最后的修正量投影层
        self.correction_proj = nn.Linear(feature_dim, feature_dim)

        # *** 在这里直接调用初始化 ***
        self._init_weights()

    def _init_weights(self):
        """
        专门为 Transformer 结构设计的初始化
        """
        for p in self.parameters():
            if p.dim() > 1:
                # Xavier Uniform 初始化对 Transformer 通常比较稳
                nn.init.xavier_uniform_(p)
            else:
                # 偏置项初始化为 0
                nn.init.constant_(p, 0)

    def forward(self, x):
        # x [Batch, Time, Feature]
        current_raw_state = x[:, -1, :] 
        
        # 通过多层编码器
        for layer in self.layers:
            x = layer(x)
            
        # 聚合策略：因为没有 Mask，所有时间步都已经充分交互。
        # 取平均值 (Mean Pooling) 可以获得极其鲁棒的全局特征。
        global_features = x.mean(dim=1) 
        
        # 计算残差修正量
        delta_state = self.correction_proj(global_features)
        
        # 恢复残差: 原始状态 + 修正量
        denoised_state = current_raw_state + delta_state
        
        return denoised_state


class TemporalPINNs_Dugoff(nn.Module):
    """
    基于时序注意力机制的物理信息神经网络道路附着系数预测模型
    Encoder-Decoder结构：
    - Encoder: 时序输入 → 时序attention → 12个去噪特征
    - 物理损失: 使用去噪特征（denorm后）和可学习物理参数计算物理一致性损失
    - Decoder: 去噪特征 → MLP → μ预测
    """
    def __init__(self, feature_dim: int = 12, time_steps: int = 50, 
                 attention_dim: int = 64, decoder_hidden_dims: list = [128, 64, 32],
                 normalization_stats: Optional[dict] = None):
        super().__init__()
        
        # 时序参数
        self.feature_dim = feature_dim  # 特征维度（12个PINNs特征）
        self.time_steps = time_steps    # 时序长度
        self.attention_dim = attention_dim
        
        # 时序注意力Encoder（用于去噪）
        self.temporal_attention = TemporalEncoder(
            feature_dim=feature_dim, 
            n_layers=2, 
            nhead=4
        )

        # MLP Decoder：将去噪特征映射到μ预测
        decoder_layers = []
        input_dim = feature_dim  # 12个去噪特征
        
        for hidden_dim in decoder_hidden_dims:
            decoder_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh()
            ])
            input_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(input_dim, 1))  # 输出μ（道路附着系数）
        self.decoder = nn.Sequential(*decoder_layers)
        
        # 归一化参数（用于反归一化输入特征）
        self.normalization_stats = normalization_stats
        if normalization_stats is not None:
            # 将归一化参数转换为张量并注册为buffer（不参与梯度计算）
            self._register_normalization_buffers(normalization_stats)
        
        # 定义参数范围和归一化统计（与原始PINNs模型相同）
        self.param_ranges = {
            # Dugoff轮胎模型参数
            'Cx': {'min': 140000.0, 'max': 170000.0, 'init': 160000.0},    # 轮胎纵向刚度 (N)
            'Cy': {'min': 120000.0, 'max': 150000.0, 'init': 140000.0},     # 轮胎横向刚度 (N)
            
            # 车辆结构参数
            'M': {'min': 5000.0, 'max': 6000.0, 'init': 5500.0},       # 车辆质量 (kg)
            'Iz': {'min': 75000.0, 'max': 90000.0, 'init': 80000.0},      # 绕z轴转动惯量 (kg·m²)
            'LF': {'min': 1.5, 'max': 3, 'init': 2.0},               # 质心到前轴距离 (m)
            'LR': {'min': 1.5, 'max': 2.5, 'init': 2.0},               # 质心到后轴距离 (m)
            'BF': {'min': 1.8, 'max': 2.5, 'init': 2.3},               # 前轮距 (m)
            'BR': {'min': 1.8, 'max': 2.5, 'init': 2.3},               # 后轮距 (m)
            'h': {'min': 0.8, 'max': 1.5, 'init': 1.0},                # 质心高度 (m)
            'R': {'min': 0.4, 'max': 0.6, 'init': 0.5},                # 轮胎半径 (m)
            
            # 侧倾刚度参数
            'K_rollf': {'min': 500000.0, 'max': 650000.0, 'init': 580000.0},  # 前轴侧倾刚度 (N·m/rad)
            'K_rollr': {'min': 550000.0, 'max': 750000.0, 'init': 630000.0},  # 后轴侧倾刚度 (N·m/rad)
        }
        
        # 计算归一化统计：将参数缩放到[-1, 1]范围
        self.param_normalization_stats = {}
        for param_name, param_range in self.param_ranges.items():
            min_val = param_range['min']
            max_val = param_range['max']
            init_val = param_range['init']
            
            # 计算归一化统计：mean = (min+max)/2, std = (max-min)/2
            # 这样归一化后的参数值在[-1, 1]范围内
            mean_val = (min_val + max_val) / 2.0
            std_val = (max_val - min_val) / 2.0
            
            self.param_normalization_stats[param_name] = {
                'mean': mean_val,
                'std': std_val,
                'min': min_val,
                'max': max_val
            }
        
        # Dugoff轮胎模型参数（可学习）- 使用归一化参数（与原始PINNs模型相同）
        self.Cx_norm = nn.Parameter(self._init_normalized_param('Cx'))  # 归一化的轮胎纵向刚度
        self.Cy_norm = nn.Parameter(self._init_normalized_param('Cy'))  # 归一化的轮胎横向刚度
        
        # 车辆结构参数（可学习）- 使用归一化参数
        self.M_norm = nn.Parameter(self._init_normalized_param('M'))   # 归一化的车辆质量
        self.Iz_norm = nn.Parameter(self._init_normalized_param('Iz'))  # 归一化的绕z轴转动惯量
        self.LF_norm = nn.Parameter(self._init_normalized_param('LF'))  # 归一化的质心到前轴距离
        self.LR_norm = nn.Parameter(self._init_normalized_param('LR'))  # 归一化的质心到后轴距离
        self.BF_norm = nn.Parameter(self._init_normalized_param('BF'))  # 归一化的前轮距
        self.BR_norm = nn.Parameter(self._init_normalized_param('BR'))  # 归一化的后轮距
        self.h_norm = nn.Parameter(self._init_normalized_param('h'))   # 归一化的质心高度
        self.R_norm = nn.Parameter(self._init_normalized_param('R'))   # 归一化的轮胎半径
        
        # 侧倾刚度参数
        self.K_rollf_norm = nn.Parameter(self._init_normalized_param('K_rollf'))  # 归一化的前轴侧倾刚度
        self.K_rollr_norm = nn.Parameter(self._init_normalized_param('K_rollr'))  # 归一化的后轴侧倾刚度
        
        # 初始化网络权重
        self._initialize_weights()
    
    def _register_normalization_buffers(self, normalization_stats: dict):
        """注册归一化参数为buffer"""
        feature_names = [
            'front_wheel_angle', 'yaw_rate', 'pitch', 'roll', 
            'vx', 'vy', 'ax', 'ay', 
            'omega_fl', 'omega_fr', 'omega_rl', 'omega_rr'
        ]
        
        for i, name in enumerate(feature_names):
            if name in normalization_stats:
                stat_dict = normalization_stats[name]
                
                # 注册均值和标准差为buffer（不参与梯度计算）
                mean_val = stat_dict.get('mean', 0.0)
                std_val = stat_dict.get('std', 1.0)
                
                self.register_buffer(f'norm_mean_{i}', torch.tensor(mean_val, dtype=torch.float32))
                self.register_buffer(f'norm_std_{i}', torch.tensor(std_val, dtype=torch.float32))
    
    def _init_normalized_param(self, param_name: str) -> torch.Tensor:
        """
        根据param_ranges中的init值初始化归一化参数
        Args:
            param_name: 参数名称
        Returns:
            归一化的参数值（在[-1, 1]范围内）
        """
        if param_name not in self.param_ranges:
            return torch.tensor(0.0)
        
        param_range = self.param_ranges[param_name]
        init_val = param_range['init']
        
        if param_name in self.param_normalization_stats:
            mean_val = self.param_normalization_stats[param_name]['mean']
            std_val = self.param_normalization_stats[param_name]['std']
        else:
            # 如果没有归一化统计，使用默认计算
            min_val = param_range['min']
            max_val = param_range['max']
            mean_val = (min_val + max_val) / 2.0
            std_val = (max_val - min_val) / 2.0
        
        # Z-score归一化: z = (x - mean) / std
        norm_val = (init_val - mean_val) / std_val
        
        # 确保在[-1, 1]范围内
        norm_val = max(min(norm_val, 1.0), -1.0)
        
        return torch.tensor(norm_val, dtype=torch.float32)
    
    def _initialize_weights(self):
        """
        全局初始化网络权重 (Xavier/Glorot)
        能够处理 Linear, LayerNorm, MultiheadAttention
        """
        for m in self.modules():
            # 1. 处理线性层 (Linear)
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            # 2. 处理归一化层 (LayerNorm)
            # LayerNorm 的 weight 初始化为 1，bias 初始化为 0
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            
            # 3. 处理多头注意力 (MultiheadAttention)
            elif isinstance(m, nn.MultiheadAttention):
                # PyTorch 的 MultiheadAttention 参数是打包的 (in_proj_weight)
                # 对所有 weight 参数应用 Xavier
                if m.in_proj_weight is not None:
                    nn.init.xavier_uniform_(m.in_proj_weight)
                if m.out_proj.weight is not None:
                    nn.init.xavier_uniform_(m.out_proj.weight)
                
                # 偏置项归零
                if m.in_proj_bias is not None:
                    nn.init.constant_(m.in_proj_bias, 0)
                if m.out_proj.bias is not None:
                    nn.init.constant_(m.out_proj.bias, 0)
    
    def _denormalize_learned_parameters(self) -> dict:
        """
        将归一化的可学习参数值反归一化为原始物理单位
        Returns:
            dict: 包含反归一化后参数的字典
        """
        denorm_params = {}
        
        # 反归一化每个参数
        param_names = ['Cx', 'Cy', 'M', 'Iz', 'LF', 'LR', 'BF', 'BR', 'h', 'R', 'K_rollf', 'K_rollr']
        norm_params = {
            'Cx': self.Cx_norm,
            'Cy': self.Cy_norm,
            'M': self.M_norm,
            'Iz': self.Iz_norm,
            'LF': self.LF_norm,
            'LR': self.LR_norm,
            'BF': self.BF_norm,
            'BR': self.BR_norm,
            'h': self.h_norm,
            'R': self.R_norm,
            'K_rollf': self.K_rollf_norm,
            'K_rollr': self.K_rollr_norm
        }
        
        for param_name in param_names:
            if param_name in self.param_normalization_stats:
                norm_val = norm_params[param_name]
                mean_val = self.param_normalization_stats[param_name]['mean']
                std_val = self.param_normalization_stats[param_name]['std']
                
                # Z-score反归一化: x = z * std + mean
                denorm_val = norm_val * std_val + mean_val
                denorm_params[param_name] = denorm_val
        
        return denorm_params
    
    def _denormalize_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        将归一化的时序特征反归一化回原始物理单位
        Args:
            x: 归一化的时序输入特征 [batch_size, feature_dim, time_steps]
        Returns:
            denormalized_x: 反归一化的时序特征 [batch_size, feature_dim, time_steps]
        """
        if self.normalization_stats is None:
            return x
        
        batch_size, feature_dim, time_steps = x.shape
        denormalized_x = x.clone()
        feature_names = ['front_wheel_angle', 'yaw_rate', 'vx', 'vy', 'vz', 'ax', 'ay', 'az', 'omega_fl', 'omega_fr', 'omega_rl', 'omega_rr']
        
        for i, name in enumerate(feature_names):
            if hasattr(self, f'norm_mean_{i}') and hasattr(self, f'norm_std_{i}'):
                mean_val = getattr(self, f'norm_mean_{i}')
                std_val = getattr(self, f'norm_std_{i}')
                
                # Z-score反归一化: x_original = x_normalized * std + mean
                denormalized_x[:, i, :] = x[:, i, :] * std_val + mean_val
        
        return denormalized_x
    
    def _denormalize_parameters(self, norm_params: torch.Tensor) -> dict:
        """
        将归一化的参数值反归一化为原始物理单位
        Args:
            norm_params: 归一化的参数值 [batch_size, 12]
        Returns:
            dict: 包含反归一化后参数的字典
        """
        batch_size = norm_params.shape[0]
        denorm_params = {}
        
        # 参数名称顺序：Cx, Cy, M, Iz, LF, LR, BF, BR, h, R, K_rollf, K_rollr
        param_names = ['Cx', 'Cy', 'M', 'Iz', 'LF', 'LR', 'BF', 'BR', 'h', 'R', 'K_rollf', 'K_rollr']
        
        for i, param_name in enumerate(param_names):
            if param_name in self.param_normalization_stats:
                norm_val = norm_params[:, i]  # [batch_size]
                mean_val = self.param_normalization_stats[param_name]['mean']
                std_val = self.param_normalization_stats[param_name]['std']
                
                # Z-score反归一化: x = z * std + mean
                denorm_val = norm_val * std_val + mean_val
                denorm_params[param_name] = denorm_val
        
        return denorm_params
    
    def set_physics_params_grad(self, requires_grad: bool = True):
        """
        控制物理参数是否参与梯度计算（冻结/解冻）
        Args:
            requires_grad (bool): True表示解冻(学习), False表示冻结(固定)
        """
        # 定义所有需要控制的物理参数名称后缀或全名
        physics_param_names = [
            'Cx_norm', 'Cy_norm', 
            'M_norm', 'Iz_norm', 
            'LF_norm', 'LR_norm', 
            'BF_norm', 'BR_norm', 
            'h_norm', 'R_norm', 
            'K_rollf_norm', 'K_rollr_norm'
        ]
        
        status = "解冻 (Learnable)" if requires_grad else "冻结 (Frozen)"
        # print(f"\n[Model Status Change] 正在将物理参数设置为: {status}")
        
        counter = 0
        for name, param in self.named_parameters():
            # 检查参数名是否在物理参数列表中
            if name in physics_param_names:
                param.requires_grad = requires_grad
                counter += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: 输入时序张量 [batch_size, time_steps, feature_dim]，包含归一化的12个特征
        Returns:
            mu_pred: 预测的道路附着系数μ [batch_size, 1]
        """
        # 时序注意力提取去噪特征
        denoised_state = self.temporal_attention(x)  # [batch_size, feature_dim]
        
        # MLP Decoder预测μ（使用去噪特征）
        mu_pred = self.decoder(denoised_state)  # [batch_size, 1]

        mu_pred = torch.clamp(mu_pred, min=0.0, max=1.0)
        
        return mu_pred
    
    def total_loss(self, x: torch.Tensor, mu_true: Optional[torch.Tensor] = None,
                  physics_weight: float = 0.1, data_weight: float = 0.9, state_weight: float = 1) -> Tuple[torch.Tensor, dict]:
        """
        计算总损失：物理约束损失 + 数据损失（若有标签）
        Args:
            x: 输入时序数据 [batch_size, feature_dim, time_steps]，包含归一化的12个特征
            mu_true: 真实μ标签 [batch_size, 1]（可选）
            physics_weight: 物理损失权重
            data_weight: 数据损失权重
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的字典
        """
        # 前向传播获取预测的μ
        # 1. 前向传播获取 去噪状态 和 修正量
        denoised_features_norm = self.temporal_attention(x)
        mu_pred = self.decoder(denoised_features_norm)
        
        # 2. 计算状态保真损失 (Reconstruction Loss)
        # 强制去噪后的状态不能偏离原始观测值太远
        # 直接在归一化空间计算即可，因为x是归一化的
        current_raw_x = x[:, -1, :] 
        l1_loss = nn.L1Loss(reduction='mean')
        state_loss = l1_loss(denoised_features_norm, current_raw_x)
        
        # 3. 反归一化 (用于物理计算)
        # 注意：这里必须使用 denoised_features_norm (去噪后的) 传给物理模块
        # 如果这里传 raw_x，物理约束就无法优化状态估计模块
        denoised_features = self._denormalize_features(denoised_features_norm.unsqueeze(2)).squeeze(2)

        # 解析12维去噪特征（现在已经是原始物理单位）
        theta = denoised_features[:, 0]  # 前轮侧偏角 (rad)
        omega_r = denoised_features[:, 1]  # 横摆角速度 (rad/s)
        pitch = denoised_features[:, 2]  # 俯仰角 (rad)
        roll = denoised_features[:, 3]   # 侧倾角 (rad)
        vx = denoised_features[:, 4]  # 车辆纵向速度 (m/s)
        vy = denoised_features[:, 5]  # 车辆横向速度 (m/s)
        ax = denoised_features[:, 6]  # 车辆纵向加速度 (m/s²)
        ay = denoised_features[:, 7]  # 车辆横向加速度 (m/s²)
        omega_fl = denoised_features[:, 8]  # 左前轮角速度 (rad/s)
        omega_fr = denoised_features[:, 9]  # 右前轮角速度 (rad/s)
        omega_rl = denoised_features[:, 10]  # 左后轮角速度 (rad/s)
        omega_rr = denoised_features[:, 11]  # 右后轮角速度 (rad/s)
        
        # 获取反归一化后的可学习参数值
        denorm_params = self._denormalize_learned_parameters()
        
        # # 物理约束损失（使用physics_loss_mu函数，传入去噪特征和可学习参数）
        phys_loss = physics_loss_mu(
            mu_pred.squeeze(), theta, omega_r, pitch, roll, vx, vy, ax, ay,
            omega_fl, omega_fr, omega_rl, omega_rr,
            denorm_params['Cx'], denorm_params['Cy'], denorm_params['M'], denorm_params['Iz'],
            denorm_params['LF'], denorm_params['LR'], denorm_params['BF'], denorm_params['BR'],
            denorm_params['h'], denorm_params['R'], denorm_params['K_rollf'], denorm_params['K_rollr']
        )
        # # 物理约束损失（使用physics_loss_mu函数，传入去噪特征和可学习参数）
        # phys_loss = physics_loss_mu(
        #     mu_true.squeeze(), theta, omega_r, pitch, roll, vx, vy, ax, ay,
        #     omega_fl, omega_fr, omega_rl, omega_rr,
        #     denorm_params['Cx'], denorm_params['Cy'], denorm_params['M'], denorm_params['Iz'],
        #     denorm_params['LF'], denorm_params['LR'], denorm_params['BF'], denorm_params['BR'],
        #     denorm_params['h'], denorm_params['R'], denorm_params['K_rollf'], denorm_params['K_rollr']
        # )
        
        # 数据损失（如果有真实标签）
        if mu_true is not None:
            l1_loss = nn.L1Loss(reduction='mean')
            data_loss = l1_loss(mu_pred, mu_true)
        else:
            data_loss = torch.tensor(0.0, device=x.device)
        
        # 总损失
        total_loss = physics_weight * phys_loss + data_weight * data_loss + state_loss * state_weight
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'physics_loss': phys_loss.item(),
            'data_loss': data_loss.item() if mu_true is not None else 0.0,
            'state_loss':state_loss.item(),
            'phy_tensor': phys_loss, 'data_tensor': data_loss, 'recon_tensor':state_loss
        }
        
        return total_loss, loss_dict
    
    def clamp_parameters(self):
        """
        将归一化的可学习参数限制在[-1, 1]范围内
        在每次优化器步骤后调用此方法
        """
        with torch.no_grad():
            # 将所有归一化的可学习参数限制在[-1, 1]范围内
            # 这样在反归一化时，参数值会保持在param_ranges定义的合理范围内
            self.Cx_norm.data.clamp_(min=-1.0, max=1.0)    # 归一化的轮胎纵向刚度
            self.Cy_norm.data.clamp_(min=-1.0, max=1.0)    # 归一化的轮胎横向刚度
            
            # 车辆结构参数
            self.M_norm.data.clamp_(min=-1.0, max=1.0)     # 归一化的车辆质量
            self.Iz_norm.data.clamp_(min=-1.0, max=1.0)    # 归一化的绕z轴转动惯量
            self.LF_norm.data.clamp_(min=-1.0, max=1.0)    # 归一化的质心到前轴距离
            self.LR_norm.data.clamp_(min=-1.0, max=1.0)    # 归一化的质心到后轴距离
            self.BF_norm.data.clamp_(min=-1.0, max=1.0)    # 归一化的前轮距
            self.BR_norm.data.clamp_(min=-1.0, max=1.0)    # 归一化的后轮距
            self.h_norm.data.clamp_(min=-1.0, max=1.0)     # 归一化的质心高度
            self.R_norm.data.clamp_(min=-1.0, max=1.0)     # 归一化的轮胎半径
            
            # 侧倾刚度参数
            self.K_rollf_norm.data.clamp_(min=-1.0, max=1.0)  # 归一化的前轴侧倾刚度
            self.K_rollr_norm.data.clamp_(min=-1.0, max=1.0)  # 归一化的后轴侧倾刚度
    
    def get_parameters_dict(self) -> dict:
        """
        获取可学习参数的当前值（反归一化后的物理单位值）
        """
        # 获取反归一化后的可学习参数值
        denorm_params = self._denormalize_learned_parameters()
        
        # 转换为Python标量值
        result = {}
        for param_name, param_tensor in denorm_params.items():
            result[param_name] = param_tensor.item()
        
        # 添加归一化参数值（可选，用于调试）
        result['Cx_norm'] = self.Cx_norm.item()
        result['Cy_norm'] = self.Cy_norm.item()
        result['M_norm'] = self.M_norm.item()
        result['Iz_norm'] = self.Iz_norm.item()
        result['LF_norm'] = self.LF_norm.item()
        result['LR_norm'] = self.LR_norm.item()
        result['BF_norm'] = self.BF_norm.item()
        result['BR_norm'] = self.BR_norm.item()
        result['h_norm'] = self.h_norm.item()
        result['R_norm'] = self.R_norm.item()
        result['K_rollf_norm'] = self.K_rollf_norm.item()
        result['K_rollr_norm'] = self.K_rollr_norm.item()
        
        return result


if __name__ == "__main__":
    # 测试模型
    feature_dim = 12  # 12个PINNs特征
    time_steps = 50   # 50个时间步（1秒，50Hz采样率）
    
    # 创建模型
    model = TemporalPINNs_Dugoff(
        feature_dim=feature_dim,
        time_steps=time_steps,
        attention_dim=64,
        decoder_hidden_dims=[128, 64, 32],
        normalization_stats=None
    )
    
    print("时序PINNs模型结构:")
    print(model)
    
    # 测试前向传播
    batch_size = 4
    test_input = torch.randn(batch_size, feature_dim, time_steps)
    mu_pred = model(test_input)
    
    print(f"\n输入形状: {test_input.shape}")
    print(f"μ预测形状: {mu_pred.shape}")
    
    # 测试总损失计算
    mu_true = torch.randn(batch_size, 1)
    total_loss, loss_dict = model.total_loss(test_input, mu_true)
    
    print(f"\n总损失: {total_loss.item():.6f}")
    print(f"损失字典: {loss_dict}")
    
    # 测试参数获取
    params_dict = model.get_parameters_dict()
    print(f"\n物理参数值:")
    for param_name, param_value in params_dict.items():
        print(f"  {param_name}: {param_value:.6f}")
    
    # 打印参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数数量: {total_params:,}")
