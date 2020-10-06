import pickle
from pathlib import Path
from typing import Callable, Tuple
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
import torchvision.transforms as vision_transforms

from functools import partial


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

        self.path = path
        self.index = []
        for entry in self.path.iterdir():
            if entry.suffix == suffix:
                self.index.append(entry)
        self.transform = transform

        self.image_size = image_size
        super().__init__()

    def __len__(self):
        return len(self.index)

    def _preproc(x, mean=None, std=None, input_space="RGB", input_range=None, **kwargs):

        if input_space == "BGR":
            x = x[..., ::-1].copy()

        if input_range is not None:
            if x.max() > 1 and input_range[1] == 1:
                x = x / 255.0

        if mean is not None:
            mean = torch.Tensor(mean)
            x = x - mean

        if std is not None:
            std = torch.Tensor(std)
            x = x / std

        return x

    def __getitem__(self, indx: int):

        sample_file = self.index[indx]

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
        default_transforms = vision_transforms.Compose(
            [
                vision_transforms.ToPILImage(),
                vision_transforms.Resize(size=self.image_size),
                vision_transforms.ToTensor(),
            ]
        )

        assert complete_sample["road_direction_map"].shape[-1] == 2
        assert complete_sample["inverse_distance_map"].shape[-1] == 1
        assert complete_sample["end_points_map"].shape[-1] == 1

        rgb = default_transforms(complete_sample["rgb"].astype(np.uint8))
        height = default_transforms(complete_sample["lidar_height"].astype(np.uint8))

        end_points = default_transforms(
            complete_sample["end_points_map"].astype(np.uint8)
        )
        direction_map = default_transforms(
            complete_sample["road_direction_map"].astype(np.uint8)
        )
        distance_map = default_transforms(
            complete_sample["inverse_distance_map"].astype(np.uint8)
        )

        assert end_points.shape[0] == 1
        assert direction_map.shape[0] == 2
        assert distance_map.shape[0] == 1

        # convert to torch tensors with CHW
        targets_torch = torch.cat([distance_map, end_points, direction_map], 0)

        if self.transform:
            transformation = partial(self._preproc, self.transform)
            rgb = transformation(rgb)
            # targets_torch = self.transform(targets_torch)
        image_torch = torch.cat([rgb, height])

        assert targets_torch.shape[0] == 4

        return (image_torch, targets_torch)
