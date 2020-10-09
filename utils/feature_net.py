import torch
from torch.nn import Module, Sequential
from utils.modules import SegmentationHead, ResidualBlock, Conv2dAuto, activation_func

from typing import Union, List, Tuple, Optional, Dict, Any
from torch.nn.common_types import _size_2_t


class FeatureNet(Module):
    def __init__(
        self,
        in_channels: int,
        encoder_prec: Dict[str, Any],
        encoder: Dict[str, Any],
        decoder: Dict[str, Any],
        head: Dict[str, Any],
        use_custom_encoder: bool = False,
        **kwargs,
    ):
        super(FeatureNet, self).__init__()

        self.encoder_prec = Sequential(
            Conv2dAuto(in_channels=in_channels, **encoder_prec),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        if use_custom_encoder:
            raise NotImplementedError
        else:
            self.encoder = Encoder(**encoder)
        self.decoder = Decoder(**decoder)
        self.head = SegmentationHead(**head)

    def forward(self, x):
        x = self.encoder_prec(x)
        # pass through encoder, saving skip connections
        x = self.encoder(x)
        x = self.decoder(*x)
        x = self.head(x)
        return x


class Decoder(Module):
    def __init__(
        self, decoder_depth: int, conv_per_block: int, blocks: List[Dict[str, Any]]
    ):
        super(Decoder, self).__init__()
        self.blocks = torch.nn.ModuleList(
            [
                DecoderBlock(**blocks[c], nmbr_convs=conv_per_block)
                for c in range(decoder_depth)
            ]
        )

    def forward(self, *features):
        # features = features[1:]
        features = features[::-1]

        x = features[0]
        skips = features[1:]

        for i, block in enumerate(self.blocks):
            skip = features[i] if i < len(skips) else None
            x = block(x, skip)
        return x


class DecoderBlock(Module):
    def __init__(
        self,
        in_channels: Union[int, List[int]],
        out_channels: Optional[List[int]] = None,
        nmbr_convs: int = 4,
        kernel_size: Union[_size_2_t, List[_size_2_t]] = 3,
        stride: Union[_size_2_t, List[_size_2_t]] = 1,
        dilation: Union[_size_2_t, List[_size_2_t]] = 0,
        activation: str = "relu",
        upsampling_mode: str = "nearest",
        upsampling_factor: int = 2,
        apply_instance_norm: bool = False,
    ):
        super(DecoderBlock, self).__init__()
        # every decoder block

        if not isinstance(in_channels, list):
            in_channels = [in_channels / (2 ** c) for c in range(nmbr_convs)]
        if out_channels is None:
            out_channels = in_channels[1:]
            out_channels.append(in_channels[-1] / 2)

        if not isinstance(kernel_size, list):
            kernel_size = [kernel_size for i in range(nmbr_convs)]
        if not isinstance(dilation, list):
            dilation = [dilation for i in range(nmbr_convs)]
        if not isinstance(stride, list):
            stride = [stride for i in range(nmbr_convs)]

        self.block = torch.nn.ModuleList(
            [
                Conv2dAuto(
                    in_channels=in_channels[i],
                    out_channels=out_channels[i],
                    kernel_size=kernel_size[i],
                    stride=stride[i],
                    dilation=dilation[i],
                )
                for i in range(nmbr_convs)
            ]
        )
        self.activate = activation_func(activation)
        self.upsample = torch.nn.Upsample(
            scale_factor=upsampling_factor, mode=upsampling_mode
        )
        if apply_instance_norm:
            self.instance_normalize = torch.nn.ModuleList(
                [
                    torch.nn.InstanceNorm2d(num_features=in_channels[i])
                    for i in range(nmbr_convs)
                ]
            )
        else:
            self.instance_normalize = torch.nn.ModuleList(
                [torch.nn.Identity() for _ in range(nmbr_convs)]
            )

    def forward(self, x, skip=None):
        x = self.upsample(x)
        for i, layer in enumerate(self.block):
            x = self.instance_normalize[i](x)
            x = layer(x)
            x = self.activate(x)
        if skip is not None:
            x = x + skip
        return x


class Encoder(Module):
    def __init__(
        self,
        encoder_depth: int,
        in_channels: Union[int, List[int]],
        out_channels: Optional[List[int]] = None,
        **kwargs,
    ):
        super(Encoder, self).__init__()
        if not isinstance(in_channels, list):
            # per default double the number of channels with every encoder block
            in_channels = [in_channels * 2 ** c for c in range(encoder_depth)]
        if out_channels is None:
            out_channels = in_channels[1:]
            out_channels.append(in_channels[-1] * 2)

        self.blocks = torch.nn.ModuleList(
            [
                EncoderBlock(
                    in_channels=in_channels[i],
                    out_channels=out_channels[i],
                    **kwargs,
                )
                for i in range(encoder_depth)
            ]
        )

    def forward(self, x):
        out = []
        for block in self.blocks:
            x = block(x)
            out.append(x)
        return out


class EncoderBlock(Module):
    """
    A EncoderBlock consists of 2 residual blocks
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        nmbr_res_blocks: int = 2,
        kernel_size: Union[_size_2_t, List[_size_2_t]] = 3,
        dilation: Union[_size_2_t, List[_size_2_t]] = 1,
        *args,
        **kwargs,
    ):
        super(EncoderBlock, self).__init__()
        if not isinstance(kernel_size, list):
            kernel_size = [kernel_size for i in range(1, nmbr_res_blocks)]
        if not isinstance(dilation, list):
            dilation = [dilation for i in range(1, nmbr_res_blocks)]

        self.head_block = ResidualBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=[2, 1, 1],
            kernel_size=kernel_size[0],
            dilation=dilation[0],
        )
        self.tail = Sequential(
            *[
                ResidualBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size[i],
                    dilation=dilation[i],
                    *args,
                    **kwargs,
                )
                for i in range(nmbr_res_blocks - 1)
            ]
        )

    def forward(self, x):
        x = self.head_block(x)
        x = self.tail(x)
        return x


if __name__ == "__main__":
    # e = Encoder(3, 3)
    pass
