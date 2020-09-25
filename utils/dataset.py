import pickle
from pathlib import Path
from typing import Callable, Tuple
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
import torchvision.transforms as vision_transforms


class RoadBoundaryDataset(Dataset):
    def __init__(self, path: Path, transform=None, *, suffix=".pkl"):
        path = Path(path)

        assert path.is_dir()

        self.path = path
        self.index = []
        for entry in self.path.iterdir():
            if entry.suffix == suffix:
                self.index.append(entry)
        self.transform = transform
        super().__init__()

    def __len__(self):
        return len(self.index)

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
                vision_transforms.Resize(size=(1280, 720)),
                vision_transforms.ToTensor(),
            ]
        )

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

        # convert to torch tensors with CHW
        image_torch = torch.cat([rgb, height])
        targets_torch = torch.cat([distance_map, end_points, direction_map], 0)

        if self.transform:
            image_torch = self.transform(image_torch)
            targets_torch = self.transform(targets_torch)
        return (image_torch, targets_torch)
