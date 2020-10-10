import numpy as np
import cv2
import torch
import torchvision


def angle_map(vector_map):
    vector_map = vector_map.numpy()
    angles_ = np.arctan2(vector_map[:, 0:1, :, :], vector_map[:, 1:2, :, :])
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


def to_tensorboard(img: torch.Tensor):
    img -= img.min()
    img = img / img.max()
    return img * 255


def apply_colormap(img, colormap=cv2.COLORMAP_JET):
    if img.ndimension() == 3:
        return cv2.applyColorMap(img, colormap)
    elif img.ndimension() == 4 and img.shape[1] == 1:
        return cv2.applyColorMap()


def _normalize(img):
    img -= img.min()
    img /= img.max()
    return img


if __name__ == "__main__":
    ba = torch.rand(10, 2, 500, 700)
    a = angle_map(ba)
