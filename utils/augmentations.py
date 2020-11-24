from typing import Tuple, Union
from torchvision.transforms import functional as tv_func
from torchvision.transforms import functional_tensor as tv_func_t
from torch.nn import Module
import torch
import torchvision
import random


class RandomCenteredCrop(torch.nn.Module):
    def __init__(self, size: Union[int, Tuple[int, int]]):
        super(RandomCenteredCrop, self).__init__()
        self.target_size = (size, size) if isinstance(size, int) else size

    def forward(self, x):
        left = random.randint(0, x.shape[-1] - self.target_size[1])
        top = random.randint(0, x.shape[-2] - self.target_size[0])

        cropped = tv_func_t.crop(
            x,
            top=top,
            left=left,
            height=self.target_size[0],
            width=self.target_size[1],
        )

        return cropped

    def __repr__(self):
        return self.__class__.__name__ + "(size={})".format(self.target_size)


class RandomHoricontalFlip(Module):
    def __init__(self, p: float = 0.5):
        super(RandomHoricontalFlip, self).__init__()
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x

        flip = tv_func.hflip(x)
        # all x-components in the vector field are now wrong
        flip[-2] = -1 * flip[-2]
        return flip

    def __repr__(self):
        return self.__class__.__name__ + "(p={})".format(self.p)


class RandomVerticalFlip(Module):
    def __init__(self, p: float = 0.5):
        super(RandomVerticalFlip, self).__init__()
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x

        flip = tv_func.vflip(x)
        # all y-components in the vector field are now wrong
        flip[-1] = -1 * flip[-1]
        return flip

    def __repr__(self):
        return self.__class__.__name__ + "(p={})".format(self.p)


class RandomRotation(Module):
    def __init__(self, p: float = 0.5):
        super(RandomRotation, self).__init__()
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        # rotate 90 degrees clockwise
        if random.random() >= 0.5:
            return self._rotate_clock(x)
        # rotate 90 degrees counter-clockwise
        else:
            return self._rotate_counter_clock(x)

    def _rotate_counter_clock(self, x):
        r = torch.transpose(x, -2, -1)
        r = r.flip(-2)

        # rotate the vector field
        y_ = r[-1].clone()
        x_ = r[-2].clone()

        r[-2] = -y_
        r[-1] = x_
        return r

    def _rotate_clock(self, x):
        r = torch.transpose(x, -2, -1)
        r = r.flip(-1)

        # rotate the vector field
        y_ = r[-1].clone()
        x_ = r[-2].clone()

        r[-2] = y_
        r[-1] = -x_

        return r

    def __repr__(self):
        return self.__class__.__name__ + "(p={})".format(self.p)


if __name__ == "__main__":
    from dataset import RoadBoundaryDataset
    import matplotlib.pyplot as plt
    from image_transforms import angle_map

    d = RoadBoundaryDataset("data/lyft-lvl5-15x10/train")
    sample_data, sample_targets = d[1]
    # sample_targets[-2:] *= -1

    vector_field = torch.zeros([2, 100, 100])
    # up
    vector_field[1, :50, :50] = -1
    # down
    vector_field[1, :50, 50:] = 1
    # left
    vector_field[0, 50:, 50:] = -1
    # right
    vector_field[0, 50:, :50] = 1

    fig, ax = plt.subplots(3, 5)
    hflip = RandomHoricontalFlip(p=1)
    vflip = RandomVerticalFlip(p=1)
    rot = RandomRotation(1)

    ax[0][0].imshow(sample_data[:3].permute(1, 2, 0))
    ax[1][0].imshow(angle_map(sample_targets[None, -2:]).squeeze().permute(1, 2, 0))
    ax[2][0].set_title("plain")
    ax[2][0].imshow(angle_map(vector_field[None]).squeeze().permute(1, 2, 0))

    f = hflip.forward(torch.cat([sample_data, sample_targets]))
    ax[0][1].imshow(f[:3].permute(1, 2, 0))
    ax[1][1].imshow(angle_map(f[None, -2:]).squeeze().permute(1, 2, 0))
    ax[2][1].set_title("hflip")
    ax[2][1].imshow(
        angle_map(hflip.forward(vector_field)[None]).squeeze().permute(1, 2, 0)
    )

    f = vflip.forward(torch.cat([sample_data, sample_targets]))
    ax[0][2].imshow(f[:3].permute(1, 2, 0))
    ax[1][2].imshow(angle_map(f[None, -2:]).squeeze().permute(1, 2, 0))
    ax[2][2].set_title("vflip")
    ax[2][2].imshow(
        angle_map(vflip.forward(vector_field)[None]).squeeze().permute(1, 2, 0)
    )

    f = rot._rotate_clock(torch.cat([sample_data, sample_targets]))
    ax[0][3].imshow(f[:3].permute(1, 2, 0))
    ax[1][3].imshow(angle_map(f[None, -2:]).squeeze().permute(1, 2, 0))
    ax[2][3].set_title("clockwise")
    ax[2][3].imshow(
        angle_map(rot._rotate_clock(vector_field)[None]).squeeze().permute(1, 2, 0)
    )

    f = rot._rotate_counter_clock(torch.cat([sample_data, sample_targets]))
    ax[0][4].imshow(f[:3].permute(1, 2, 0))
    ax[1][4].imshow(angle_map(f[None, -2:]).squeeze().permute(1, 2, 0))
    ax[2][4].set_title("counter clockwise")
    ax[2][4].imshow(
        angle_map(rot._rotate_counter_clock(vector_field)[None])
        .squeeze()
        .permute(1, 2, 0)
    )
    plt.show()
