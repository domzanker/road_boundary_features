import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List, Dict, Any
from torch.nn.common_types import _size_2_t
import segmentation_models_pytorch as smp


def upsampling_modules(upsampling: str, *args, **kwargs):
    return nn.ModuleDict(
        {
            "bilinear": nn.UpsamplingBilinear2d(*args, **kwargs),
            "nearest": nn.UpsamplingNearest2d(*args, **kwargs),
            "transposed": nn.ConvTranspose2d(*args, **kwargs),
        }
    )[upsampling]


def activation_func(activation: str):
    return torch.nn.ModuleDict(
        {
            "relu": torch.nn.ReLU(),
            "sigmoid": torch.nn.Sigmoid(),
            "softmax": torch.nn.Softmax2d(),
            "lrelu": torch.nn.LeakyReLU(),
            "none": torch.nn.Identity(),
            "tanh": torch.nn.Tanh(),
            "hardtanh": torch.nn.Hardtanh(),
        }
    )[activation]


smp_models = {"linknet": smp.Linknet, "fpn": smp.FPN}


class SegmentationHead(nn.Module):
    def __init__(
        self,
        branches: Union[List[List[int]], List[Dict[str, Any]]],
        upsampling_mode: str = "nearest",
        scale_factor: float = 4.0,
        **kwargs
    ):

        super(SegmentationHead, self).__init__()

        self.branches = nn.ModuleList()
        if isinstance(branches, dict):
            branches = list(branches.values())
        for branch in branches:
            if isinstance(branch, list):
                branch = {
                    "in_channels": branch[0],
                    "out_channels": branch[1],
                    "kernel_sizes": branch[2],
                    "strides": branch[3],
                    "dilation": branch[5],
                    "batch_norm": True,
                    "end_activation": "relu",
                }
            in_channels = branch["in_channels"]
            out_channels = branch["out_channels"]

            if "depth" in branch.keys():
                depth = branch["keys"]
            else:
                depth = len(in_channels)

            # kernel_sizes = branch["kernel_size"]
            if not isinstance(branch["kernel_sizes"], list):
                kernel_sizes = [branch["kernel_sizes"] for _ in range(depth)]
            else:
                kernel_sizes = branch["kernel_sizes"]

            if not isinstance(branch["strides"], list):
                strides = [branch["strides"] for _ in range(depth)]
            else:
                strides = branch["strides"]

            if not isinstance(branch["dilation"], list):
                dilation = [branch["dilation"] for _ in range(depth)]
            else:
                dilation = branch["dilation"]

            batch_norm = branch["batch_norm"]
            end_activation = branch["end_activation"]

            if "upsampling_mode" not in branch:
                branch["upsampling_mode"] = upsampling_mode
            if "scale_factor" not in branch:
                branch["scale_factor"] = scale_factor

            self.branches.append(
                SegmentationBranch(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    depth=depth,
                    kernel_sizes=kernel_sizes,
                    strides=strides,
                    dilation=dilation,
                    end_activation=end_activation,
                    batch_norm=batch_norm,
                    upsampling_mode=branch["upsampling_mode"],
                    scale_factor=branch["scale_factor"],
                )
            )

    def forward(self, x: torch.Tensor):
        output = []
        for branch in self.branches:
            output.append(branch(x))
        return torch.cat(output, dim=1)


class SegmentationBranch(nn.Module):
    def __init__(
        self,
        depth: int,
        in_channels: List[int],
        out_channels: List[int],
        kernel_sizes: Union[int, Tuple[int, int], List[Tuple[int, int]]],
        strides: Union[int, Tuple[int, int], List[int], List[Tuple[int, int]]],
        dilation: Union[int, Tuple[int, int], List[int], List[Tuple[int, int]]] = 1,
        end_activation: str = "relu",
        batch_norm: bool = True,
        upsampling_mode: str = "nearest",
        scale_factor: float = 2.0,
        activation: str = "relu",
    ):
        super(SegmentationBranch, self).__init__()
        if not isinstance(kernel_sizes, list):
            kernel_sizes = [kernel_sizes for _ in range(depth)]
        if not isinstance(strides, list):
            strides = [strides for _ in range(depth)]
        if not isinstance(dilation, list):
            dilation = [dilation for _ in range(depth)]

        activations = [activation for _ in range(depth)]
        activations[-1] = end_activation

        batch_norm = [batch_norm for _ in range(depth)]
        batch_norm[-1] = False

        # TODO get out_channels

        if upsampling_mode == "transposed":
            self.upsampling = torch.nn.ConvTranspose2d(
                in_channels=in_channels[0],
                out_channels=in_channels[0],
                kernel_size=kernel_sizes[0],
                stride=scale_factor,
                padding=1,
                output_padding=1,
                dilation=1,
            )
        else:
            self.upsampling = nn.Upsample(
                scale_factor=scale_factor, mode=upsampling_mode
            )

        self.conv_blocks = nn.ModuleList(
            [
                ConvBlock(
                    in_channels=in_channels[i],
                    out_channels=out_channels[i],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    dilation=dilation[i],
                    activation=activations[i],
                    batch_norm=batch_norm[i],
                )
                for i in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor):
        x = self.upsampling(x)
        for block in self.conv_blocks:
            x = block(x)
        return x


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]],
        dilation: Union[int, Tuple[int, int]] = 1,
        activation: str = "relu",
        batch_norm: bool = True,
        dropout: float = 0.0,
        *args,
        **kwargs
    ):
        super(ConvBlock, self).__init__()
        if batch_norm:
            self.batch_norm = torch.nn.BatchNorm2d(out_channels)
        else:
            self.batch_norm = torch.nn.Identity()

        self.conv = Conv2dAuto(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            *args,
            **kwargs
        )
        self.activation = activation_func(activation)
        self.dropout = nn.Dropout2d(dropout, inplace=True)

    def forward(self, x: torch.Tensor):
        conv = self.conv(x)
        drop = self.dropout(conv)
        norm = self.batch_norm(drop)
        return self.activation(norm)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int = 3,
        kernel_size: Union[_size_2_t, List[_size_2_t]] = 3,
        stride: Union[_size_2_t, List[_size_2_t]] = 1,
        dilation: Union[_size_2_t, List[_size_2_t]] = 1,
        activation: str = "relu",
        batch_norm: bool = False,
        shortcut: str = "projection",
        *args,
        **kwargs
    ):
        super(ResidualBlock, self).__init__()
        self.in_channels, self.out_channels, self.activation = (
            in_channels,
            out_channels,
            activation,
        )

        in_channels = [in_channels]
        for i in range(1, depth):
            in_channels.append(out_channels)

        if not isinstance(kernel_size, list):
            kernel_size = [kernel_size for _ in range(depth)]
        if not isinstance(stride, list):
            stride = [stride for _ in range(depth)]
        if not isinstance(dilation, list):
            dilation = [dilation for _ in range(depth)]
        if not isinstance(out_channels, list):
            out_channels = [out_channels for _ in range(depth)]

        self.blocks = nn.Sequential(
            *[
                ConvBlock(
                    in_channels=in_channels[i],
                    out_channels=out_channels[i],
                    kernel_size=kernel_size[i],
                    stride=stride[i],
                    dilation=dilation[i],
                    *args,
                    **kwargs
                )
                for i in range(depth)
            ]
        )

        if batch_norm:
            self.normalize = nn.BatchNorm2d(out_channels[-1])
        else:
            self.normalize = nn.Identity()
        self.activate = activation_func(activation)
        if stride[0] != 1 or self.apply_skip_connection:
            if shortcut == "projection":
                self.shortcut = nn.Sequential(
                    ConvBlock(
                        in_channels=in_channels[0],
                        out_channels=out_channels[-1],
                        kernel_size=1,
                        stride=stride[0],
                        bias=False,
                        batch_norm=True,
                    ),
                )
            else:
                raise NotImplementedError
        else:
            self.shortcut = None

    def forward(self, x):

        residual = x
        if self.apply_skip_connection:
            residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = self.normalize(x)
        x = self.activate(x)
        return x

    @property
    def apply_skip_connection(self):
        # return self.shortcut is not None
        return self.in_channels != self.out_channels


class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(Conv2dAuto, self).__init__(*args, **kwargs)
        self.padding = (
            (self.kernel_size[0] + (self.kernel_size[0] - 1) * (self.dilation[0] - 1))
            // 2,
            (self.kernel_size[1] + (self.kernel_size[1] - 1) * (self.dilation[1] - 1))
            // 2,
        )


class Interpolate(nn.Module):
    def __init__(
        self,
        size=None,
        scale_factor=None,
        mode="nearest",
        align_corners=None,
        recompute_scale_factor=None,
    ):
        super(Interpolate, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor

    def forward(self, x):
        return torch.nn.functional.interpolate(
            x,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
            recompute_scale_factor=self.recompute_scale_factor,
        )


class FullConv(Conv2dAuto):
    def __init__(self):
        super(FullConv, self).__init__()
        raise NotImplementedError
