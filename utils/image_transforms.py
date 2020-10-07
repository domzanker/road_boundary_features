import cv2
import numpy as np
import torch


def angle_map(vector_map):
    vector_map = vector_map.numpy()
    angles_ = np.arctan2(vector_map[:, 0, :, :], vector_map[:, 1, :, :])
    angles_ = (angles_ + np.pi) / (2 * np.pi) * 255
    angles_ = angles_.astype(np.uint8)
    angles_ = np.transpose(angles_, (0, 2, 3, 1))
    batch = [
        cv2.applyColorMap(angles_[i, :, :, :], cv2.COLORMAP_JET)
        for i in range(angles_.shape[0])
    ]
    batch = np.stack(batch, axis=0)
    batch = np.transpose(batch, (0, 3, 1, 2))
    return torch.Tensor(batch)
