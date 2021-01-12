import h5py
import torch
import cv2
import numpy as np

sample_file = "data/lyft-lvl5-20x20/train/scene_38_sample_1_data.h5"
with h5py.File(sample_file, mode="r", swmr=True) as f:

    assert f["road_direction_map"].shape[-1] == 2
    assert f["inverse_distance_map"].shape[-1] == 1
    assert f["end_points_map"].shape[-1] == 1

    rgb = torch.from_numpy(f["rgb"][()]).float() / 255  # uint8 -> float32
    h = np.nan_to_num(f["lidar_height"][()])
    height = torch.from_numpy(h[None, :, :]).float()  # float16 -> float32
    height_deriv = torch.from_numpy(
        cv2.Laplacian(h.astype(np.float32), cv2.CV_32F, ksize=3)
    ).float()
    height_deriv = height_deriv[None, :, :]

    # float16->float32
    inverse_distance_map = torch.from_numpy(
        np.nan_to_num(f["inverse_distance_map"][()])
    ).float()
    end_points_map = torch.from_numpy(
        np.nan_to_num(f["end_points_map"][()])
    ).float()  # float16 -> float32
    road_direction_map = torch.from_numpy(
        np.nan_to_num(f["road_direction_map"][()])
    ).float()  # float32

assert torch.isfinite(road_direction_map).all()
assert torch.isfinite(inverse_distance_map).all()
assert torch.isfinite(end_points_map).all()

assert torch.isfinite(rgb).all()
assert torch.isfinite(height).all()

print(rgb.shape)
