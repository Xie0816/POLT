"""Dataset loading and timestamp synchronization helpers for POLT runtime."""

import json
import os

import numpy as np
import torch

from common_struct import TIME_LEN, TSS_GAP


def initialize_runtime_state(runner):
    """Create mutable runtime containers used across the frame loop."""
    runner.img_files = {}
    runner.lidar_files = {}
    runner.lidar_odom_files = {}
    runner.proprio_files = {}

    runner.img_stamps = []
    runner.lidar_stamps = []
    runner.lidar_odom_stamps = []
    runner.proprio_stamps = []

    runner.lidar_data = {}
    runner.odom_data = {}
    runner.proprio_data = {}

    runner.points_xyzrgb = {}
    runner.points_xyzfeat = {}
    runner.proprio_color_history = {}
    runner.log = {}


def free_cache(runner):
    """Drop old cached frame data once the rolling history window is full."""
    tss = sorted(runner.lidar_data.keys())
    if len(tss) >= TIME_LEN:
        del_ts = min(tss)
        if del_ts in runner.lidar_data:
            del runner.lidar_data[del_ts]
        if del_ts in runner.odom_data:
            del runner.odom_data[del_ts]
        if del_ts in runner.proprio_data:
            del runner.proprio_data[del_ts]
        if del_ts in runner.points_xyzfeat:
            del runner.points_xyzfeat[del_ts]
        if del_ts in runner.points_xyzrgb:
            del runner.points_xyzrgb[del_ts]
        if del_ts in runner.proprio_color_history:
            del runner.proprio_color_history[del_ts]

    torch.cuda.empty_cache()


def read_folder(folder_path):
    """Read a timestamp-named file folder into ``{timestamp: path}``."""
    folder_dic = {}
    for file_path in os.listdir(folder_path):
        file_name = file_path.split(".")[0]
        folder_dic[int(file_name)] = os.path.join(folder_path, file_path)
    return folder_dic


def read_file(runner, file_path):
    """Read supported metadata files from the dataset root."""
    if file_path.endswith("lidarodometry.txt"):
        txt = np.loadtxt(file_path)
        file_dic = {}
        for i in range(txt.shape[0]):
            local_time = int(txt[i, 0])
            file_dic[local_time] = torch.from_numpy(txt[i, 1:]).float().to(runner.device)
        return file_dic
    if file_path.endswith("proprio_infos.json"):
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    return {}


def data_loader(runner, path):
    """Load either a supported file or a timestamped directory."""
    if os.path.isfile(path):
        return read_file(runner, path)
    if os.path.isdir(path):
        return read_folder(path)
    return {}


def time_match(stamps, target_stamp, tss_gap=TSS_GAP):
    """Find the nearest timestamp within the allowed synchronization gap."""
    diff = [abs(stamp - target_stamp) for stamp in stamps]
    min_diff = min(diff)
    min_index = diff.index(min_diff)
    if min_diff <= tss_gap:
        return min_index, stamps[min_index]
    return None, None


def data_init(runner, root):
    """Validate and index the required POLT dataset streams under ``root``."""
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    images_path = os.path.join(root, "image_front")
    lidar_path = os.path.join(root, "lidar")
    lidarodometry_path = os.path.join(root, "lidarodometry.txt")
    proprio_path = os.path.join(root, "proprio_infos.json")

    required_paths = {
        "image_front": images_path,
        "lidar": lidar_path,
        "lidarodometry.txt": lidarodometry_path,
        "proprio_infos.json": proprio_path,
    }
    missing_paths = [f"{name}: {path}" for name, path in required_paths.items() if not os.path.exists(path)]
    if missing_paths:
        raise FileNotFoundError("Dataset is missing required inputs:\n" + "\n".join(missing_paths))

    runner.img_files = data_loader(runner, images_path)
    runner.img_stamps = sorted(runner.img_files.keys())

    runner.lidar_files = data_loader(runner, lidar_path)
    runner.lidar_stamps = sorted(runner.lidar_files.keys())

    runner.lidar_odom_files = data_loader(runner, lidarodometry_path)
    runner.lidar_odom_stamps = sorted(runner.lidar_odom_files.keys())

    runner.proprio_files = data_loader(runner, proprio_path)
    runner.proprio_stamps = sorted(map(int, runner.proprio_files.keys()))

    print(
        "Dataset loaded: "
        f"images={len(runner.img_stamps)}, "
        f"lidar={len(runner.lidar_stamps)}, "
        f"odom={len(runner.lidar_odom_stamps)}, "
        f"proprio={len(runner.proprio_stamps)}"
    )
    if not runner.img_stamps or not runner.lidar_stamps or not runner.lidar_odom_stamps or not runner.proprio_stamps:
        raise ValueError(f"Dataset contains empty required streams: {root}")
