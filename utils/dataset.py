from pathlib import Path
from time import sleep
from typing import Optional, Tuple

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as vision_transforms
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


class RoadBoundaryDataset(Dataset):
    def __init__(
        self,
        path: Path,
        transform=None,
        *,
        suffix: str = ".h5",
        image_size: Optional[Tuple[int, int]] = None
    ):
        path = Path(path)

        assert path.is_dir()

        self.path = path.resolve()
        self.index = []
        for entry in self.path.iterdir():
            if entry.suffix == suffix:
                self.index.append(entry)

        if transform is not None:
            self.transform_params = transform
        else:
            self.transform_params = None

        self.image_size = image_size
        super().__init__()

    def __len__(self):
        return len(self.index)

    def __getitem__(self, indx: int):

        sample_file = self.index[indx]

        if not sample_file.is_file():
            sleep(0.01)

        with h5py.File(sample_file, mode="r", swmr=True) as f:
            assert f["road_direction_map"].shape[-1] == 2
            assert f["inverse_distance_map"].shape[-1] == 1
            assert f["end_points_map"].shape[-1] == 1

            rgb = torch.from_numpy(f["rgb"][()]).float() / 255  # uint8 -> float32
            height = torch.from_numpy(
                f["lidar_height"][()]
            ).float()  # float16 -> float32

            # float16->float32
            inverse_distance_map = torch.from_numpy(
                f["inverse_distance_map"][()]
            ).float()
            end_points_map = torch.from_numpy(
                f["end_points_map"][()]
            ).float()  # float16 -> float32
            road_direction_map = torch.from_numpy(
                f["road_direction_map"][()]
            ).float()  # float32

        assert torch.isfinite(road_direction_map).all()
        assert torch.isfinite(inverse_distance_map).all()
        assert torch.isfinite(end_points_map).all()

        assert torch.isfinite(rgb).all()
        assert torch.isfinite(height).all()

        rgb = rgb.permute(2, 0, 1)
        height = height[None, :, :]

        inverse_distance_map = inverse_distance_map.permute(2, 0, 1)
        end_points_map = end_points_map.permute(2, 0, 1)
        road_direction_map = road_direction_map.permute(2, 0, 1)

        assert end_points_map.shape[0] == 1
        assert road_direction_map.shape[0] == 2
        assert inverse_distance_map.shape[0] == 1

        # convert to torch tensors with CHW
        # targets_torch = torch.cat([distance_map, end_points, direction_map], 0)
        targets_torch = inverse_distance_map
        image_torch = torch.cat([rgb, height])
        if self.image_size is not None:
            targets_torch = F.interpolate(
                targets_torch[None, :, :, :], size=self.image_size
            ).squeeze(dim=0)

            image_torch = F.interpolate(
                image_torch[None, :, :, :], size=self.image_size
            ).squeeze(dim=0)

        if self.transform_params is not None:
            mean = self.transform_params["mean"]
            std = self.transform_params["std"]
            image_torch[:3, :, :] = F.normalize(
                image_torch[:3, :, :], mean=mean, std=std
            )

        return (image_torch, targets_torch)


class ImageDataset(RoadBoundaryDataset):
    def __init__(
        self,
        path: Path,
        transform=None,
        *,
        suffix: str = ".jpeg",
        image_size: Optional[Tuple[int, int]] = None
    ):
        super().__init__(
            path, transform=transform, suffix=suffix, image_size=image_size
        )
        self.index = []
        for scenes in path.iterdir():
            if scenes.is_dir():
                for samples in scenes.iterdir():
                    if (samples / "CAM_FRONT.png").is_file():
                        self.index.append(samples / "CAM_FRONT.png")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, indx: int):

        sample_file = self.index[indx]

        image = cv2.imread(sample_file)

        image_tensor = to_tensor(image)

        return image_tensor, image_tensor
