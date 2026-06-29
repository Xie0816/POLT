"""Shared constants for POLT calibration, map geometry, and runtime limits."""

import numpy as np

# LiDAR parameters.
HORIZONTAL_ANGULAR_RESOLUTION = 0.2  # Horizontal angular resolution in degrees.

# Sensor timestamp matching.
TSS_GAP = 50  # Sensor timestamp tolerance in milliseconds.

# Point-cloud accumulation.
TIME_LEN = 100  # Number of frames kept in the accumulation window.
TIME_GAP = 1  # Frame stride for accumulation.
Z_max = 10  # Maximum retained point height.
Z_min = -10  # Minimum retained point height.
LIDAR_H = 2.2  # LiDAR mounting height in meters.
VOXEL_SIZE = 0.1
O3DVIS_CAMERA_PARAMS ={
    "front": [-1, 0, 0.5],
    "lookat": [2, 0, 0],
    "up": [0, 0, 1],
    "zoom": 0.1
}

# BEV map settings.
MAP_SIZE = 1000  # BEV map size in pixels.
RESOLUTION = 0.2  # BEV resolution in meters per pixel.

# Image settings.
SCALE_FACTOR = 1  # Image scaling factor used by projection helpers.


# LiDAR-to-front-camera projection matrix.
LM_AR0231_Front = np.array(
    [[1004.55, -1944.58, 93.9691, 48470.2],
[571.708, -30.8496, -1926.77, 7056.09],
[0.998301, 0.0125654, 0.0568946, 74.0718]]
)

# Proprioceptive data settings.
PROPRIO_WINDOWS = 50  # 50 time steps, approximately 1 second.
INFER_SLIP_LIMIT = 0.001  # Slip-ratio threshold for inference guards.
