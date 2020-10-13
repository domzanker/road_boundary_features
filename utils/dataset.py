import pickle
from pathlib import Path
from typing import Callable, Tuple
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor
import torchvision.transforms as vision_transforms

from functools import partial

from time import sleep


class RoadBoundaryDataset(Dataset):
    def __init__(
        self,
        path: Path,
        transform=None,
        *,
        suffix=".pkl",
        image_size: Tuple[int, int] = (640, 360)
    ):
        path = Path(path)

        assert path.is_dir()

        self.path = path.resolve()
        self.index = []
        for entry in self.path.iterdir():
            if entry.suffix == suffix:
                self.index.append(entry)

        if transform is not None:
            self.transform = partial(self._preproc, **transform)
        else:
            self.transform = None

        self.image_size = image_size
        super().__init__()
        self.first

    def __len__(self):
        return len(self.index)

    def _preproc(
        self, x, mean=None, std=None, input_space="RGB", input_range=None, **kwargs
    ):

        if input_space == "BGR":
            x = x[..., ::-1].copy()

        if input_range is not None:
            if x.max() > 1 and input_range[1] == 1:
                x = x / x.max()

        if mean is None:
            mean = 0
        if std is None:
            std = 0

        x = vision_transforms.functional.normalize(x, mean, std)

        return x

    def _to_tensor(self, arr):
        tensor = torch.Tensor(arr)
        if tensor.ndimension() == 2:
            tensor = tensor.unsqueeze(0)
        else:
            tensor = tensor.permute(2, 0, 1)
        return tensor

    def __getitem__(self, indx: int):

        sample_file = self.index[indx]

        if not sample_file.is_file():
            sleep(0.01)

        with sample_file.open("rb") as f:
            complete_sample = pickle.load(f)
        """
        sample_collection = {
            "bev": bev,
            "rgb": self.rgb,
            "lidar_height": self.lidar[:, :, 2],
            "lidar_intensity": self.lidar[:, :, 1],
            "ground_truth": self.ground_truth,
            "road_direction_map": self.direction_map,
            "inverse_distance_map": self.distance_map,
        }
        """
        assert complete_sample["road_direction_map"].shape[-1] == 2
        assert complete_sample["inverse_distance_map"].shape[-1] == 1
        assert complete_sample["end_points_map"].shape[-1] == 1

        # HWC -> CHW
        rgb = complete_sample["rgb"].astype(np.uint8)
        rgb = to_tensor(rgb)  # range [0,1]

        height = self._to_tensor(complete_sample["lidar_height"].astype(np.float32))
        height = height - height.min()  # range [0, inf]

        end_points = self._to_tensor(
            complete_sample["end_points_map"].astype(np.float32)
        )

        direction_map = self._to_tensor(
            complete_sample["road_direction_map"].astype(np.float32)
        )
        distance_map = self._to_tensor(
            complete_sample["inverse_distance_map"].astype(np.float32)
        )
        distance_map = distance_map - distance_map.min()
        distance_map = distance_map / distance_map.max()  # range [0, 1]
        assert torch.isfinite(distance_map).all()

        assert end_points.shape[0] == 1
        assert direction_map.shape[0] == 2
        assert distance_map.shape[0] == 1

        # convert to torch tensors with CHW
        # targets_torch = torch.cat([distance_map, end_points, direction_map], 0)
        targets_torch = distance_map
        if self.image_size is not None:
            targets_torch = F.interpolate(
                targets_torch[None, :, :, :], size=self.image_size, mode="bicubic"
            ).squeeze(dim=0)

        if self.transform:
            rgb = self.transform(rgb)

        image_torch = torch.cat([rgb, height])
        if self.image_size is not None:
            image_torch = F.interpolate(
                image_torch[None, :, :, :], size=self.image_size, mode="bicubic"
            ).squeeze(dim=0)

        image_torch = vision_transforms.functional.normalize(
            image_torch, mean=(0, 0, 0, 0), std=(1, 1, 1, 1)
        )

        assert targets_torch.shape[0] == 4

        return (image_torch, targets_torch)
