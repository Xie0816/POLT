#!/usr/bin/env python3
"""
SALON roughness-feedback cost calculator.

This module ports the core cost computation from the original ROS node while
keeping it dependency-free for POLT runtime presets.
"""

import numpy as np
import scipy.signal
try:
    from scipy.integrate import simpson as simps
except ImportError:
    from scipy.integrate import simps


class Buffer:
    """Small rolling buffer used for fixed-duration signal windows."""

    def __init__(self, buffer_size, padded=False, pad_val=None):
        self.buffer_size = buffer_size
        if not padded:
            self._data = []
            self.data = np.array(self._data)
        else:
            assert pad_val is not None, "pad_val must be provided for padded buffers"
            self._data = [pad_val] * buffer_size
            self.data = np.array(self._data)

    def insert(self, data_point):
        self._data.append(data_point)
        if len(self._data) > self.buffer_size:
            self._data = self._data[1:]
        self.data = np.array(self._data)

    def get_data(self):
        return self.data

    def show(self):
        print(self.data)


def bandpower(data, sf, band, window_sec=None, relative=False):
    """
    Compute average signal power inside a frequency band.

    Adapted from https://raphaelvallat.com/bandpower.html.

    Args:
        data: One-dimensional time-domain signal.
        sf: Sampling frequency.
        band: Lower and upper frequency bounds.
        window_sec: Welch window length in seconds.
        relative: Return relative power if true, absolute power otherwise.

    Returns:
        Absolute or relative band power.
    """
    band = np.asarray(band)
    low, high = band

    # Select the Welch window length.
    if window_sec is not None:
        nperseg = window_sec * sf
    else:
        nperseg = None

    # Estimate the power spectral density with Welch's method.
    freqs, psd = scipy.signal.welch(data, sf, nperseg=nperseg)

    # Frequency resolution for numerical integration.
    freq_res = freqs[1] - freqs[0]

    # Select samples inside the requested frequency band.
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integrate the spectrum with Simpson's rule.
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simps(psd, dx=freq_res)
    return bp


class SalonCostCalculator:
    """Compute SALON-style roughness feedback from IMU and shock signals."""
    
    def __init__(self, params=None, stats=None):
        """Initialize cost parameters, normalization statistics, and buffers."""
        # Default parameters from the source implementation.
        self.params = params or {
            'IMU_min_freq': {'z': 2, 'y': 9, 'x': 0},
            'IMU_max_freq': {'z': 30, 'y': 13, 'x': 22},
            'IMU_mult': {'z': 1.0, 'y': 0.7, 'x': 0.6},
            'shock_min_freq': 0,
            'shock_max_freq': 46,
            'mean_mult': 0.1,
            'shock_mult': 0.5,
            # 'cutoff_factor': 0.6,
            # 'ALL_MAX': 0.09220535518703862,
            # 'ALL_MIN': 0.002474401263335543,
            # 'ALL_AVG': 0.019438000929697122,
            'cutoff_factor': 0.8,
            'ALL_MAX': 0.8,
            'ALL_MIN': 0.024,
            'ALL_AVG': 0.05
        }
        
        # Default normalization statistics from the source implementation.
        self.stats = stats or {
            'IMU_MIN': np.array([-0.5554102710448205, -0.6436653784476221, -0.5440625478513539, 
                                 -16.969271264076234, -20.00854387164116, -4.090186840295792]),
            'IMU_MAX': np.array([0.5882288794964552, 0.5955823929980397, 0.6560320965945721, 
                                 16.157508494853975, 37.19758415222168, 23.688631061911583]),
            'SHOCK_MIN': np.array([4.366000175476074, 4.552999973297119]),
            'SHOCK_MAX': np.array([6.920000076293945, 7.138999938964844])
        }
        
        # Configure fixed one-second signal windows.
        self.imu_freq = 100
        self.shock_freq = 50
        self.num_secs = 1
        self.buffer_size = int(self.num_secs * self.imu_freq)
        self.shock_buff_size = int(self.num_secs * self.shock_freq)
        
        # Initialize IMU buffers with normalized padding values.
        imu_pad_z = (9.81 - self.stats['IMU_MIN'][5]) / (self.stats['IMU_MAX'][5] - self.stats['IMU_MIN'][5])
        
        self.bufferZ = Buffer(self.buffer_size, padded=True, pad_val=imu_pad_z)
        self.bufferX = Buffer(self.buffer_size, padded=True, pad_val=0)
        self.bufferY = Buffer(self.buffer_size, padded=True, pad_val=0)
        self.bufferRoll = Buffer(self.buffer_size, padded=True, pad_val=0)
        self.bufferPitch = Buffer(self.buffer_size, padded=True, pad_val=0)
        
        # Shock buffers are kept at the original lower sampling rate.
        self.bufferL = Buffer(self.shock_buff_size, padded=True, pad_val=6)
        self.bufferR = Buffer(self.shock_buff_size, padded=True, pad_val=6)
        
        # Optional command/joystick buffer retained for compatibility.
        self.bufferJoy = Buffer(32, padded=True, pad_val=0)
        
        # Runtime cost state.
        self.shock_cost = 0
        self.joy_cost = 0
        self.diff_cost = 0
        self.velocity = 0
        self.vel_mismatch = 0.0
        self.desired_vel = None
        
    def update_imu(self, imu_data):
        """
        Update IMU buffers and return the current roughness cost.

        Args:
            imu_data: Array with angular velocity and linear acceleration
                ``[wx, wy, wz, ax, ay, az]``.
        """
        # Normalize incoming IMU channels using the source statistics.
        imu_norm = (imu_data - self.stats['IMU_MIN']) / (self.stats['IMU_MAX'] - self.stats['IMU_MIN'])
        imu_norm[0] -= 0.5  # Roll offset.
        imu_norm[1] -= 0.5  # Pitch offset.
        
        # Update rolling buffers for the frequency-domain roughness estimate.
        self.bufferZ.insert(imu_norm[5])      # Z acceleration.
        self.bufferRoll.insert(imu_norm[0])   # Roll rate.
        self.bufferPitch.insert(imu_norm[1])  # Pitch rate.
        self.bufferX.insert(imu_norm[3])      # X acceleration.
        self.bufferY.insert(imu_norm[4])      # Y acceleration.
        
        # Accumulate band power from vertical, longitudinal, and lateral axes.
        bp = bandpower(self.bufferZ.data, self.imu_freq,
                      band=[self.params['IMU_min_freq']['z'], self.params['IMU_max_freq']['z']],
                      window_sec=self.num_secs)
        bp *= self.params['IMU_mult']['z']
        
        xbp = bandpower(self.bufferX.data, self.imu_freq,
                       band=[self.params['IMU_min_freq']['x'], self.params['IMU_max_freq']['x']],
                       window_sec=self.num_secs)
        xbp *= self.params['IMU_mult']['x']
        bp += xbp
        
        ybp = bandpower(self.bufferY.data, self.imu_freq,
                       band=[self.params['IMU_min_freq']['y'], self.params['IMU_max_freq']['y']],
                       window_sec=self.num_secs)
        ybp *= self.params['IMU_mult']['y']
        bp += ybp
        
        # Add roll mean and shock terms used by the original SALON cost.
        MEAN = np.mean(np.abs(self.bufferRoll.data))
        bp += MEAN * self.params['mean_mult']
        
        sbp = self.shock_cost * self.params['shock_mult']
        bp += sbp
        
        # Normalize and clamp cost to [0, 1].
        bp = (bp - self.params['ALL_MIN']) / (self.params['ALL_MAX'] * self.params['cutoff_factor'] - self.params['ALL_MIN'])
        bp = np.clip(bp, 0, 1)
        cost = bp
        
        # Add a simple shock-jerk penalty.
        jerk = np.mean(np.abs(np.diff(self.bufferL.data[-30:])))
        cost += jerk
        
        return cost
    
    def update_shock(self, shock_data):
        """Update shock buffers from rear suspension readings."""
        # Normalize shock displacement/position channels.
        shock_norm = (shock_data - self.stats['SHOCK_MIN']) / (self.stats['SHOCK_MAX'] - self.stats['SHOCK_MIN'])
        
        # The current implementation uses the left rear shock channel.
        self.bufferL.insert(shock_norm[0])
        
        # Convert shock vibration energy into an additive cost term.
        costL = bandpower(self.bufferL.data, self.shock_freq,
                         band=[self.params['shock_min_freq'], self.params['shock_max_freq']],
                         window_sec=self.num_secs)
        
        self.shock_cost = costL
    
    def update_joy(self, joy_axis):
        """Update the optional joystick buffer and return a simple variance cost."""
        self.bufferJoy.insert(joy_axis)
        
        # Simplified command-variation cost retained for compatibility.
        cost = np.var(self.bufferJoy.data)
        self.joy_cost = self.joy_cost + (cost - self.joy_cost) * 0.05
    
    def update_terrain_mismatch(self, terrain_diff, max_vel=8.0, max_diff=1.0):
        """Update the optional terrain-mismatch penalty."""
        maxcost = max_vel * max_diff
        diff_cost = self.velocity * terrain_diff
        self.diff_cost = diff_cost / maxcost
    
    def update_velocity(self, velocity, desired_vel=None):
        """Update velocity state and track command/actual mismatch."""
        self.velocity = velocity
        self.desired_vel = desired_vel
        
        if desired_vel is not None:
            mismatch = np.abs(velocity - desired_vel)
        else:
            mismatch = 0.0
        
        self.vel_mismatch = self.vel_mismatch + (mismatch - self.vel_mismatch) * 0.5
    
    def calculate_cost(self, imu_data, shock_data=None, joy_axis=None, 
                      terrain_diff=None, velocity=None, desired_vel=None):
        """
        Compute the combined roughness cost for one synchronized sensor sample.

        Args:
            imu_data: IMU vector with shape ``(6,)``.
            shock_data: Optional rear shock vector with shape ``(2,)``.
            joy_axis: Optional joystick axis value.
            terrain_diff: Optional terrain mismatch signal.
            velocity: Optional current velocity.
            desired_vel: Optional desired velocity.
        """
        # Update auxiliary buffers before evaluating the IMU-driven main cost.
        if shock_data is not None:
            self.update_shock(shock_data)
        
        if joy_axis is not None:
            self.update_joy(joy_axis)
        
        if velocity is not None:
            self.update_velocity(velocity, desired_vel)
        
        if terrain_diff is not None:
            self.update_terrain_mismatch(terrain_diff)
        
        # The IMU update returns the main roughness feedback.
        cost = self.update_imu(imu_data)
        
        return cost


def calculate_simple_cost(imu_z_data, sampling_freq=100, freq_range=(2, 30)):
    """Compute a lightweight cost from vertical acceleration band power only."""
    return bandpower(imu_z_data, sampling_freq, band=freq_range)


def test_salon_cost():
    """Run a small local smoke test for the SALON calculator."""
    print("测试SALON代价计算器...")
    
    calculator = SalonCostCalculator()
    
    # Generate deterministic synthetic signals for local debugging.
    np.random.seed(42)
    
    # Synthetic six-channel IMU sample.
    imu_test = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 9.81])
    
    # Synthetic rear shock sample.
    shock_test = np.array([5.5, 5.6])
    
    cost = calculator.calculate_cost(
        imu_data=imu_test,
        shock_data=shock_test,
        joy_axis=0.5,
        terrain_diff=0.1,
        velocity=3.0,
        desired_vel=4.0
    )
    
    print(f"计算出的代价: {cost:.4f}")
    
    # Exercise the lightweight vertical-acceleration helper.
    imu_z_test = np.random.randn(100) + 9.81
    simple_cost = calculate_simple_cost(imu_z_test)
    print(f"简化代价: {simple_cost:.4f}")
    
    return cost


if __name__ == "__main__":
    test_salon_cost()
