"""Proprioceptive telemetry parsing and temporal feature construction."""

import torch
import torch.nn as nn
import numpy as np
import math
from collections import deque
import json
import os

from model.proprio.TemporalPINNs_Dugoff import TemporalPINNs_Dugoff 
from common_struct import *

class Prorio:
    """Reader and feature builder for proprioceptive telemetry."""

    def __init__(self, time_steps=50):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.time_steps = time_steps
        
        # Feature normalization statistics loaded with the trained model.
        self.norm_stats = {} 
        self.feature_names = [
            'front_wheel_angle', 'yaw_rate', 'pitch', 'roll', 
            'vx', 'vy', 'ax', 'ay', 
            'omega_fl', 'omega_fr', 'omega_rl', 'omega_rr'
        ]

        self.proprio_data = {}
        self.tss = []
        self.proprio_feats = {}

    def data_init(self, proprio_data):
        self.proprio_data = proprio_data
        self.tss = sorted(self.proprio_data.keys(), key=lambda x: int(x))
        for idx in range(len(self.tss)):
            ts = self.tss[idx]
            pinns_features = self._extract_pinns_features(idx)
            self.proprio_feats[ts] = pinns_features

    def compute_slip(self, id, R = 0.5):
        """Estimate scalar longitudinal slip from rear-wheel speed and local velocity."""
        frame_id = self.tss[id]
        frame_data = self.proprio_data[frame_id]
        chassis = frame_data.get('chassis', {})
        wheel_speed = chassis.get('rr_wheel_speed', 0.0)
        omega =  wheel_speed / 3.6  # km/h -> m/s

        v, a = self._extract_globalpose_data(id)
        vx = v[0]

        eps = 1e-8
        # Numerical guard for near-stationary frames.
        if abs(vx) < 0.01 and abs(omega) < 0.01:
            return 0.0
            
        s = (omega - vx) / (omega + eps) if vx < omega else (vx - omega) / (vx + eps)

        if vx < 0.1 or omega < 0.1:
            s = 0

        return s
    
    def compute_longitude_lateral_slip(self, id, R = 0.5, LF=2, BF=2, LR=2, BR=2):
        """Estimate per-wheel longitudinal slip and lateral slip angles."""
        frame_id = self.tss[id]
        frame_data = self.proprio_data[frame_id]
        globalpose = frame_data.get('globalpose', {})
        steer = frame_data.get('steer', {})
        chassis = frame_data.get('chassis', {})

        front_wheel_angle = steer.get('front_wheel_angle', 0.0)
        front_wheel_angle_rad = np.radians(front_wheel_angle)
        
        yaw_rate = globalpose.get('dev_azimuth', 0.0)  # degree/s
        yaw_rate_rad = np.radians(yaw_rate)

        v, a = self._extract_globalpose_data(id)
        vx, vy = v[0], v[1]

        R = 0.5  # Tire radius in meters.
        wheel_speeds_kmh = [
            chassis.get('lf_wheel_speed', 0.0),
            chassis.get('rf_wheel_speed', 0.0), 
            chassis.get('lr_wheel_speed', 0.0),
            chassis.get('rr_wheel_speed', 0.0)
        ]
        
        wheel_speeds_ms = [speed / 3.6 for speed in wheel_speeds_kmh]  # km/h -> m/s
        omega_fl = wheel_speeds_ms[0] / R if R > 0 else 0.0
        omega_fr = wheel_speeds_ms[1] / R if R > 0 else 0.0
        omega_rl = wheel_speeds_ms[2] / R if R > 0 else 0.0
        omega_rr = wheel_speeds_ms[3] / R if R > 0 else 0.0

        u_fl = (vx - BF/2 * yaw_rate_rad) * np.cos(front_wheel_angle_rad) + (vy + LF * yaw_rate_rad) * np.sin(front_wheel_angle_rad)
        u_fr = (vx + BF/2 * yaw_rate_rad) * np.cos(front_wheel_angle_rad) + (vy + LF * yaw_rate_rad) * np.sin(front_wheel_angle_rad)
        u_rl = vx - BR/2 * yaw_rate_rad
        u_rr = vx + BR/2 * yaw_rate_rad
        
        # Longitudinal slip for four wheels.
        omega = np.array([omega_fl, omega_fr, omega_rl, omega_rr])
        u = np.array([u_fl, u_fr, u_rl, u_rr])
        
        eps = 1e-8
        omega_R = omega * R
        s = np.where(u < omega_R,
                    (omega_R - u) / (omega_R + eps), 
                    (omega_R - u) / (u + eps))       
        
        # Lateral slip angles for front/rear and left/right wheels.
        alpha_fl = front_wheel_angle_rad - np.atan2(vy + LF * yaw_rate_rad, vx - BF/2 * yaw_rate_rad + eps)
        alpha_fr = front_wheel_angle_rad - np.atan2(vy + LF * yaw_rate_rad, vx + BF/2 * yaw_rate_rad + eps)
        alpha_rl = -np.atan2(vy - LR * yaw_rate_rad, vx - BR/2 * yaw_rate_rad + eps)
        alpha_rr = -np.atan2(vy - LR * yaw_rate_rad, vx + BR/2 * yaw_rate_rad + eps)
    
        alpha =[alpha_fl, alpha_fr, alpha_rl, alpha_rr]

        return s.tolist(),alpha

    def _extract_globalpose_data(self, id):
        """Extract global velocity/acceleration and rotate them into the vehicle frame."""
        prev_id = max(id - 5, 0)
        prev_frame_id = self.tss[prev_id]
        prev_frame_data = self.proprio_data.get(prev_frame_id, {})
        prev_globalpose = prev_frame_data.get('globalpose', {})
        
        next_id = min(id + 5, len(self.tss) - 5)
        next_frame_id = self.tss[next_id]
        next_frame_data = self.proprio_data.get(next_frame_id, {})
        next_globalpose = next_frame_data.get('globalpose', {})

        frame_id = self.tss[id]
        curr_frame_data = self.proprio_data[frame_id]
        curr_globalpose = curr_frame_data.get('globalpose', {})

        v_east_curr = curr_globalpose.get('vEast', 0.0)  # East velocity.
        v_north_curr = curr_globalpose.get('vNorth', 0.0)  # North velocity.
        v_up_curr = curr_globalpose.get('vUp', 0.0)  # Up velocity.
        
        # Use a centered finite difference for acceleration when neighboring frames exist.
        v_east_prev = prev_globalpose.get('vEast', 0.0)
        v_north_prev = prev_globalpose.get('vNorth', 0.0)
        v_up_prev = prev_globalpose.get('vUp', 0.0)
        
        v_east_next = next_globalpose.get('vEast', 0.0)
        v_north_next = next_globalpose.get('vNorth', 0.0)
        v_up_next = next_globalpose.get('vUp', 0.0)
        
        dt = (int(next_frame_id) - int(prev_frame_id)) / 1000  # s
        
        a_east = (v_east_next - v_east_prev) / dt
        a_north = (v_north_next - v_north_prev) / dt
        a_up = (v_up_next - v_up_prev) / dt

        # Convert current attitude to radians before frame rotation.
        azimuth = np.radians(curr_globalpose.get('azimuth', 0.0))  # Heading/yaw angle from East.
        pitch = np.radians(curr_globalpose.get('pitch', 0.0))  # Pitch angle.
        roll = np.radians(curr_globalpose.get('roll', 0.0))  # Roll angle.

        # Rotation matrix from global ENU-like axes to the vehicle body frame.
        cos_azimuth = np.cos(azimuth)
        sin_azimuth = np.sin(azimuth)
        cos_pitch = np.cos(pitch)
        sin_pitch = np.sin(pitch)
        cos_roll = np.cos(roll)
        sin_roll = np.sin(roll)

        # Base yaw rotation from vehicle frame to the ENU-like global frame.
        R_z = np.array([
            [cos_azimuth, -sin_azimuth, 0],  # Vehicle forward direction in ENU.
            [sin_azimuth, cos_azimuth, 0],   # Vehicle left direction in ENU.
            [0, 0, 1]                        # Vehicle up direction in ENU.
        ])
        
        # Apply pitch and roll rotations after heading.
        R_y = np.array([
            [cos_pitch, 0, sin_pitch],
            [0, 1, 0],
            [-sin_pitch, 0, cos_pitch]
        ])
        
        R_x = np.array([
            [1, 0, 0],
            [0, cos_roll, -sin_roll],
            [0, sin_roll, cos_roll]
        ])
        
        # Full rotation: heading, then pitch, then roll.
        R = R_z @ R_y @ R_x

        # Project global velocity into the vehicle frame: v_local = R^T @ v_global.
        v_global = np.array([v_east_curr, v_north_curr, v_up_curr])
        v_local = R.T @ v_global

        # Project global acceleration into the vehicle frame.
        a_global = np.array([a_east, a_north, a_up])
        a_local = R.T @ a_global

        # Return vehicle-frame velocity and acceleration.
        return v_local.tolist(), a_local.tolist()

    def _extract_pinns_features(self, id):
        """
        Build the 12-D temporal PINNs input vector for one proprio frame.

        The feature order follows the dynamics model: front wheel angle, yaw
        rate, pitch, roll, vehicle-frame velocity/acceleration, and four wheel
        angular velocities.
        """
        frame_id = self.tss[id]
        frame_data = self.proprio_data[frame_id]
        
        globalpose = frame_data.get('globalpose', {})
        chassis = frame_data.get('chassis', {})
        steer = frame_data.get('steer', {})
        imu = frame_data.get('imu', {})
        
        # 1. Front wheel angle.
        front_wheel_angle = steer.get('front_wheel_angle', 0.0)
        front_wheel_angle_rad = np.radians(front_wheel_angle)
        
        # 2. Yaw rate.
        yaw_rate = globalpose.get('dev_azimuth', 0.0)  # degree/s
        yaw_rate_rad = np.radians(yaw_rate)

        # 3. Pitch and roll.
        pitch = globalpose.get('pitch', 0.0)
        roll = globalpose.get('roll', 0.0)
        pitch = np.radians(pitch) 
        roll = np.radians(roll) 
        
        # 4. Vehicle-frame velocity and acceleration.
        v_vehicle, a_vehicle = self._extract_globalpose_data(id)  # [v_x, v_y, v_z], [a_x, a_y, a_z]
        vx = v_vehicle[0]  # Longitudinal velocity v_x.
        vy = v_vehicle[1]  # Lateral velocity v_y.
        vz = v_vehicle[2]  # Vertical velocity v_z, kept for debugging.
        ax = a_vehicle[0]  # Longitudinal acceleration a_x.
        ay = a_vehicle[1]  # Lateral acceleration a_y.
        az = a_vehicle[2]  # Vertical acceleration a_z, kept for debugging.
        
        # 5. Wheel angular velocities in rad/s.
        R = 0.5  # Tire radius in meters.
        wheel_speeds_kmh = [
            chassis.get('lf_wheel_speed', 0.0),
            chassis.get('rf_wheel_speed', 0.0), 
            chassis.get('lr_wheel_speed', 0.0),
            chassis.get('rr_wheel_speed', 0.0)
        ]
        
        # Convert wheel speeds from km/h to m/s, then to angular velocity.
        wheel_speeds_ms = [speed / 3.6 for speed in wheel_speeds_kmh]  # km/h -> m/s
        omega_fl = wheel_speeds_ms[0] / R if R > 0 else 0.0
        omega_fr = wheel_speeds_ms[1] / R if R > 0 else 0.0
        omega_rl = wheel_speeds_ms[2] / R if R > 0 else 0.0
        omega_rr = wheel_speeds_ms[3] / R if R > 0 else 0.0
        
        # Compose the 12-D PINNs input vector.
        pinns_features = np.array([
            front_wheel_angle_rad,
            yaw_rate_rad,
            pitch,
            roll,
            vx,
            vy,
            ax,
            ay,
            omega_fl,
            omega_fr,
            omega_rl,
            omega_rr
        ])
            
        return pinns_features

    def _prepare_temporal_feats(self, id):
        his_id = max(0, id - self.time_steps + 1)

        temporal_features_temp = []
        for i in range(his_id, id + 1):
            ts_i = self.tss[i]
            pinns_features = self.proprio_feats[ts_i]
            temporal_features_temp.append(pinns_features)
        temporal_features = np.array(temporal_features_temp)
        temporal_features = self.normalize_features(temporal_features)
        return temporal_features

    def normalize_features(self, features):
        """Apply Z-score normalization expected by the temporal PINNs model."""
        features = torch.tensor(features, dtype=torch.float32)
        if not self.norm_stats:
            # Without normalization the model input scale is invalid.
            print("[Warning] No normalization stats found! Returning raw features.")
            return features 

        normalized = features.clone()
        for i, name in enumerate(self.feature_names):
            mean = self.norm_stats[name]['mean']
            std = self.norm_stats[name]['std']
            # normalized[i,:] = (features[i,:] - mean) / std
            normalized[:,i] = (features[:,i] - mean) / std

        
        return normalized

    def model_init(self, weights_path):
        """
        Initialize the model structure and load weights.
        """
        normalization_json_path = os.path.join(weights_path, "temporal_normalization_stats_t50.json")
        
        if os.path.exists(normalization_json_path):
            with open(normalization_json_path, 'r') as f:
                self.norm_stats = json.load(f)
            print(f"Loaded normalization stats from {normalization_json_path}")
        else:
            print(f"[Error] Normalization stats file not found: {normalization_json_path}")

        checkpoint_path = os.path.join(weights_path, "final_model_t50.pth")
        
        # Pass loaded normalization statistics directly into the model.
        self.model = TemporalPINNs_Dugoff(
                time_steps=self.time_steps,
                decoder_hidden_dims=[128, 64, 32], 
                normalization_stats=self.norm_stats
            )
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            
        # Use strict loading because this runtime expects the released checkpoint schema.
        self.model.load_state_dict(checkpoint, strict=True)
        
        self.model.to(self.device)
        self.model.eval()
        print("本体感知模型导入成功！")

    def model_infer(self, features):
        """
        Args:
            features: numpy array [Time, 12]
        """
        
        features = features.unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(features)
            
        adhesion_coefficient = outputs.item()

        cost = 0.8 * (1 - adhesion_coefficient)
            
        return cost


Proprio = Prorio


def load_proprio_frame(proprio_sensor, proprio_files, proprio_stamp):
    return proprio_sensor.data_format(proprio_files[str(proprio_stamp)])


def should_trigger_proprio_feedback(proprio_sensor, proprio_idx, slip_limit=INFER_SLIP_LIMIT):
    return proprio_sensor.compute_slip(proprio_idx) > slip_limit


def build_proprio_history_point(device, adhesion_coefficient, color_mapper):
    color = color_mapper([adhesion_coefficient])[0]
    proprio_color = (color * 255).astype(np.uint8)
    proprio_pose = torch.tensor([0.0, 0.0, -LIDAR_H], device=device)
    proprio_rgb = torch.tensor(proprio_color, device=device)
    return torch.cat([proprio_pose, proprio_rgb])


__all__ = [
    "Prorio",
    "Proprio",
    "load_proprio_frame",
    "should_trigger_proprio_feedback",
    "build_proprio_history_point",
]
