import pickle
from pathlib import Path
from typing import Callable, Tuple
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


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

        images = (complete_sample["bev"]).astype(np.float32)
        # image as float [0,1]

        targets = np.concatenate(
            [
                complete_sample["inverse_distance_map"],
                complete_sample["end_points_map"],
                complete_sample["road_direction_map"],
            ],
            axis=-1,
        )

        # transform sample
        if self.transform:
            images = self.transform(images)
            targets = self.transform(targets)

        # sample = {"sample": images, "targets": targets}

        # convert to torch tensors with CHW
        image_torch = to_tensor(images)
        targets_torch = to_tensor(targets)
        return (image_torch, targets_torch)
