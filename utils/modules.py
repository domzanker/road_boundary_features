import torch
from typing import Optional, Tuple, Union, List, Dict, Any
from torch.nn.common_types import _size_2_t
import segmentation_models_pytorch as smp


def activation_func(activation: str):
    return torch.nn.ModuleDict(
        {
            "relu": torch.nn.ReLU(),
            "sigmoid": torch.nn.Sigmoid(),
            "softmax": torch.nn.Softmax2d(),
            "lrelu": torch.nn.LeakyReLU(),
            "none": torch.nn.Identity(),
            "tanh": torch.nn.Tanh(),
        }
    )[activation]


smp_models = {"linknet": smp.Linknet, "fpn": smp.FPN}


class FeatureExtrationNet(torch.nn.Module):
    def __init__(self, model_configs: Dict[str, Any]):
        super(FeatureExtrationNet, self).__init__()

        self.encoder_decoder = smp_models[model_configs["encoder_decoder_name"]](
            **model_configs["encoder_decoder"]
        )
        if model_configs["encoder_decoder"]["encoder_weights"]:
            self.preprocessing_params = smp.encoders.get_preprocessing_params(
                encoder_name=model_configs["encoder_decoder"]["encoder_name"],
                pretrained=model_configs["encoder_decoder"]["encoder_weights"],
            )
        else:
            self.preprocessing_params = None

        # self.head = SegmentationHead(**model_configs["head"])
        self.head = torch.nn.Identity()

    def forward(self, x):
        x = self.encoder_decoder(x)
        x = self.head(x)
        return x


class SegmentationHead(torch.nn.Module):
    def __init__(
        self,
        branches: Union[List[List[int]], List[Dict[str, Any]]],
        upsample_mode: str = "nearest",
        scale_factor: float = 4.0,
        **kwargs
    ):

        super(SegmentationHead, self).__init__()

        self.branches = torch.nn.ModuleList()
        for branch in branches:
            if isinstance(branch, list):
                branch = {
                    "in_channels": branch[0],
                    "out_channels": branch[1],
                    "kernel_sizes": branch[2],
                    "strides": branch[3],
                    "padding": branch[4],
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

            if not isinstance(branch["padding"], list):
                padding = [branch["padding"] for _ in range(depth)]
            else:
                padding = branch["padding"]

            batch_norm = branch["batch_norm"]
            end_activation = branch["end_activation"]

            self.branches.append(
                SegmentationBranch(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    depth=depth,
                    kernel_sizes=kernel_sizes,
                    strides=strides,
                    dilation=dilation,
                    padding=padding,
                    end_activation=end_activation,
                    batch_norm=batch_norm,
                    upsample_mode=upsample_mode,
                    scale_factor=scale_factor,
                )
            )

    def forward(self, x: torch.Tensor):
        output = []
        for branch in self.branches:
            output.append(branch(x))
        return torch.cat(output, dim=1)


class SegmentationBranch(torch.nn.Module):
    def __init__(
        self,
        depth: int,
        in_channels: List[int],
        out_channels: List[int],
        kernel_sizes: Union[int, Tuple[int, int], List[Tuple[int, int]]],
        strides: Union[int, Tuple[int, int], List[int], List[Tuple[int, int]]],
        dilation: Union[int, Tuple[int, int], List[int], List[Tuple[int, int]]] = 1,
        padding: Union[int, Tuple[int, int], List[int], List[Tuple[int, int]]] = 0,
        end_activation: str = "relu",
        batch_norm: bool = True,
        upsample_mode: str = "nearest",
        scale_factor: float = 2.0,
    ):
        super(SegmentationBranch, self).__init__()
        if not isinstance(kernel_sizes, list):
            kernel_sizes = [kernel_sizes for _ in range(depth)]
        if not isinstance(strides, list):
            strides = [strides for _ in range(depth)]
        if not isinstance(dilation, list):
            dilation = [dilation for _ in range(depth)]
        if not isinstance(padding, list):
            padding = [padding for _ in range(depth)]

        activations = ["relu" for _ in range(depth)]
        activations[-1] = end_activation

        # TODO get out_channels

        self.upsampling = torch.nn.Upsample(
            scale_factor=scale_factor, mode=upsample_mode
        )

        self.conv_blocks = torch.nn.ModuleList(
            [
                ConvBlock(
                    in_channels=in_channels[i],
                    out_channels=out_channels[i],
                    kernel_size=kernel_sizes[i],
                    padding=padding[i],
                    stride=strides[i],
                    dilation=dilation[i],
                    activation=activations[i],
                    batch_norm=batch_norm,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor):
        x = self.upsampling(x)
        for block in self.conv_blocks:
            x = block(x)
        return x


class ConvBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        activation: str = "relu",
        batch_norm: bool = True,
    ):
        super(ConvBlock, self).__init__()
        if batch_norm:
            self.batch_norm = torch.nn.BatchNorm2d(in_channels)
        else:
            self.batch_norm = None

        self.conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
        )
        self.activation = activation_func(activation)

    def forward(self, x: torch.Tensor):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        conv = self.conv(x)
        return self.activation(conv)


class ResidualBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int = 3,
        kernel_size: Union[_size_2_t, List[_size_2_t]] = 3,
        stride: Union[_size_2_t, List[_size_2_t]] = 1,
        dilation: Union[_size_2_t, List[_size_2_t]] = 1,
        activation: str = "sigmoid",
        batch_norm: bool = False,
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

        self.blocks = torch.nn.Sequential(
            *[
                Conv2dAuto(
                    in_channels=in_channels[i],
                    out_channels=out_channels[i],
                    kernel_size=kernel_size[i],
                    stride=stride[i],
                    dilation=dilation[i],
                )
                for i in range(depth)
            ]
        )

        if batch_norm:
            self.normalize = torch.nn.BatchNorm2d(self.out_channels[-1])
        else:
            self.normalize = torch.nn.Identity()
        self.activate = activation_func(activation)
        self.shortcut = torch.nn.Identity()

    def forward(self, x):

        residual = x
        x = self.blocks(x)
        if self.apply_skip_connection:
            residual = self.shortcut(x)
        x += residual
        x = self.normalize(x)
        x = self.activate(x)
        return x

    @property
    def apply_skip_connection(self):
        return self.in_channels != self.out_channels


class Conv2dAuto(torch.nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(Conv2dAuto, self).__init__(*args, **kwargs)
        self.padding = (
            (self.kernel_size[0] + (self.kernel_size[0] - 1) * (self.dilation[0] - 1))
            // 2,
            (self.kernel_size[1] + (self.kernel_size[1] - 1) * (self.dilation[1] - 1))
            // 2,
        )


class FullConv(Conv2dAuto):
    def __init__(self):
        super(FullConv, self).__init__()
        raise NotImplementedError
