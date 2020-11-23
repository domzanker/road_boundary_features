from pathlib import Path
from time import sleep
from typing import Optional, Tuple, Union

import cv2
import h5py
import numpy as np
import torch
import random
import torch.nn.functional as F
import torchvision.transforms as vision_transforms
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


class RandomRotation(torch.nn.Module):
    def __init__(self, p: float = 0.5):
        super(RandomRotation, self).__init__()
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        direction = random.random()
        if direction < 0.5:
            x = torch.transpose(x, -2, -1)
        else:
            x = torch.transpose(x, -2, -1).flip(-1)
        return x

    def __repr__(self):
        return self.__class__.__name__ + "(p={})".format(self.p)


class RandomCenteredCrop(torch.nn.Module):
    def __init__(self, size: Union[int, Tuple[int, int]]):
        super(RandomCenteredCrop, self).__init__()
        self.target_size = (size, size) if isinstance(size, int) else size

    def forward(self, x):
        left = random.randint(0, x.shape[-1] - self.target_size[1])
        top = random.randint(0, x.shape[-2] - self.target_size[0])

        cropped = vision_transforms.functional_tensor.crop(
            x,
            top=top,
            left=left,
            height=self.target_size[0],
            width=self.target_size[1],
        )

        return cropped

    def __repr__(self):
        return self.__class__.__name__ + "(size={})".format(self.target_size)


class RoadBoundaryDataset(Dataset):
    def __init__(
        self,
        path: Path,
        transform=None,
        *,
        suffix: str = ".h5",
        image_size: Optional[Tuple[int, int]] = None,
        angle_bins: int = None,
        augmentation: float = None
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

        self.angle_bins = angle_bins

        self.crop = RandomCenteredCrop(448)

        if augmentation is not None:
            self.augmentation = vision_transforms.Compose(
                [
                    # vision_transforms.RandomCrop(448),
                    vision_transforms.RandomHorizontalFlip(p=augmentation),
                    vision_transforms.RandomVerticalFlip(p=augmentation),
                    RandomRotation(p=augmentation),
                ]
            )
        else:
            self.augmentation = None

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
            h = np.nan_to_num(f["lidar_height"][()])
            h = h[None, :, :]
            height = torch.from_numpy(h).float()  # float16 -> float32
            height_deriv = torch.from_numpy(
                cv2.Laplacian(h.astype(np.float32), cv2.CV_32F, ksize=3)
            ).float()

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

        rgb = rgb.permute(2, 0, 1)

        inverse_distance_map = inverse_distance_map.permute(2, 0, 1)
        end_points_map = end_points_map.permute(2, 0, 1)
        road_direction_map = road_direction_map.permute(2, 0, 1)

        assert end_points_map.shape[0] == 1
        assert road_direction_map.shape[0] == 2
        assert inverse_distance_map.shape[0] == 1

        if self.angle_bins is not None:
            road_direction_map = self._angle_bins(road_direction_map, self.angle_bins)

        # convert to torch tensors with CHW
        targets_torch = torch.cat(
            [
                inverse_distance_map,
                end_points_map,
                road_direction_map,
            ],
            0,
        )
        assert targets_torch.shape[0] == 4
        # targets_torch = inverse_distance_map
        image_torch = torch.cat([rgb, height, height_deriv])
        if self.image_size is not None:
            targets_torch = F.interpolate(
                targets_torch[None, :, :, :], size=self.image_size
            ).squeeze(dim=0)

            image_torch = F.interpolate(
                image_torch[None, :, :, :], size=self.image_size
            ).squeeze(dim=0)

        if self.transform_params is not None:
            # mean = self.transform_params["mean"]
            # std = self.transform_params["std"]
            """
            image_torch[:3, :, :] = F.normalize(
                image_torch[:3, :, :], mean=[0.0, 0.0, 0.0], std=[1, 1, 1]
            )
            """
        if self.augmentation is not None:
            cropped = self.crop(torch.cat([image_torch, targets_torch]))

            augmented = self.augmentation(cropped)
            image_torch = augmented[:5]
            targets_torch = augmented[5:]
        else:
            cropped = self.crop(torch.cat([image_torch, targets_torch]))
            image_torch = augmented[:5]
            targets_torch = augmented[5:]

        return (image_torch, targets_torch)

    def _angle_bins(self, vector_field, bins: int = 4):
        assert vector_field.shape[0] == 2
        angle_map = torch.atan2(vector_field[0], vector_field[1])
        # angle range rad -pi, pi
        bin_size = 2 * np.pi / bins
        for i in range(bins):
            lower_bin = -np.pi + i * bin_size
            upper_bin = -np.pi + (i + 1) * bin_size
            # set angle to bin angle
            angle_map[
                torch.logical_and(angle_map >= lower_bin, angle_map < upper_bin)
            ] = lower_bin
        vector_field[0] = torch.cos(angle_map)
        vector_field[1] = torch.sin(angle_map)
        return vector_field


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


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    path = "data/lyft-lvl5-15x10/valid"
    d = RoadBoundaryDataset(Path(path), image_size=(720, 490), augmentation=1)
    fig, ax = plt.subplots()

    sample = d.__getitem__(0)
    print(sample[0].shape)
    ax.imshow(sample[0][4])
    plt.show()
