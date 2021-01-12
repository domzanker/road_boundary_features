import h5py
import numpy as np
import cv2


def read_sample(path):
    with h5py.File(path, mode="r", swmr=True) as f:

        assert f["road_direction_map"].shape[-1] == 2
        assert f["inverse_distance_map"].shape[-1] == 1
        assert f["end_points_map"].shape[-1] == 1

        rgb = f["rgb"][()]  # uint8 -> float32
        h = np.nan_to_num(f["lidar_height"][()])
        height = h[None, :, :]  # float16 -> float32

        # float16->float32
        inverse_distance_map = np.nan_to_num(f["inverse_distance_map"][()])
        end_points_map = np.nan_to_num(f["end_points_map"][()])  # float16
        road_direction_map = np.nan_to_num(f["road_direction_map"][()])  # float16

    return rgb, height, inverse_distance_map, end_points_map, road_direction_map


def save_sample(rgb):
    pass
