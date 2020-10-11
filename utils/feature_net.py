from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.nn import Module, Sequential
from torch.nn.common_types import _size_2_t
from torchvision.utils import make_grid
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.modules import Conv2dAuto, ResidualBlock, SegmentationHead, activation_func
from utils.losses import loss_func

from utils.dataset import RoadBoundaryDataset
import pytorch_lightning as pl


class FeatureNet(pl.LightningModule):
    def __init__(
        self,
        configs: Dict[str, Any],
    ):
        super(FeatureNet, self).__init__()

        self.model_configs = configs["model"]
        self.train_configs = configs["train"]

        self.encoder_prec = Sequential(
            Conv2dAuto(
                in_channels=self.model_configs["input_channels"],
                **self.model_configs["encoder_prec"],
            ),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        if False:
            raise NotImplementedError
        else:
            self.encoder = Encoder(**self.model_configs["encoder"])
        self.decoder = Decoder(**self.model_configs["decoder"])
        self.head = SegmentationHead(**self.model_configs["head"])

        loss_args = (
            self.train_configs["loss-args"]
            if "loss-args" in self.train_configs.keys()
            else {}
        )
        self.loss = loss_func(self.train_configs["loss"], **loss_args)

        self.train_mse = pl.metrics.MeanSquaredError()
        self.train_dist_accuracy = pl.metrics.Accuracy()

        self.val_mse = pl.metrics.MeanSquaredError()
        self.val_dist_accuracy = pl.metrics.Accuracy()

        self.save_hyperparameters()

    def forward(self, x):

        x = self.encoder_prec(x)
        # pass through encoder, saving skip connections
        x = self.encoder(x)
        x = self.decoder(*x)
        x = self.head(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch

        prec = self.encoder_prec(x)
        encoding = self.encoder(prec)
        decoding = self.decoder(*encoding)
        segmentation = self.head(decoding)

        loss = self.loss(segmentation[0], y[:, 0:1, :, :])

        pred = segmentation[0]
        target = y[:, :1, :, :]
        self.log_dict(
            {
                "train_mse": self.train_mse(pred, target),
                "train_dist_accuracy": self.train_dist_accuracy(pred, target),
            },
            on_step=True,
            on_epoch=False,
        )
        # logging to tensorboard
        self.log("train_loss", loss)

        """
        # log out out
        y_ = y[:, 0:1, :, :].detach()
        y_ = y_ - y_.min()
        y_ = y_ / y_.max()
        tensorboard.add_images("train distance map", y_ * 255, dataformats="NCHW")

        pred = segmentation[0].detach()
        pred = pred - pred.min()
        pred = pred / pred.max()
        tensorboard.add_images("train distance pred", pred * 255)

        lid = x[:, 3:, :, :].detach()
        lid = lid - lid.min()
        lid = lid / lid.max()
        tensorboard.add_images("train input lidar", lid * 255, dataformats="NCHW")
        rgb = x[:, :3, :, :].detach()
        tensorboard.add_images("train input rgb", rgb, dataformats="NCHW")
        """

        return loss

    def validation_step(self, batch, batch_idx):

        x, y = batch

        prec = self.encoder_prec(x)
        encoding = self.encoder(prec)
        decoding = self.decoder(*encoding)
        segmentation = self.head(decoding)

        loss = self.loss(segmentation[0], y[:, 0:1, :, :])

        pred = segmentation[0]
        target = y[:, :1, :, :]
        self.log_dict(
            {
                "val_mse": self.val_mse(pred, target),
                "val_dist_accuracy": self.val_dist_accuracy(pred, target),
            },
            on_step=False,
            on_epoch=True,
        )

        return {"loss": loss, "y": y, "pred": segmentation[0].detach(), "x": x}

    def validation_epoch_end(self, outputs):
        tensorboard = self.logger.experiment
        y = torch.cat([t["y"] for t in outputs])
        x = torch.cat([t["x"] for t in outputs])
        pred = torch.cat([t["pred"] for t in outputs])

        # log out out
        y_ = y[:, 0:1, :, :].detach()
        y_ -= y_.min()
        y_ /= y_.max()
        tensorboard.add_images(
            "valid distance map",
            make_grid(y_[:64, :, :, :] * 255),
            dataformats="CHW",
            global_step=self.trainer.global_step,
        )

        pred = pred.detach()
        pred = pred - pred.min()
        pred /= pred.max()
        tensorboard.add_images(
            "valid distance pred",
            make_grid(pred[:64, :, :, :] * 255),
            dataformats="CHW",
            global_step=self.trainer.global_step,
        )

        lid = x[:, 3:, :, :].detach()
        lid -= lid.min()
        lid /= lid.max()
        tensorboard.add_images(
            "valid input lidar",
            make_grid((lid[:64, :, :, :] + 1) * 255),
            dataformats="CHW",
            global_step=self.trainer.global_step,
        )
        rgb = x[:, :3, :, :].detach()
        tensorboard.add_images(
            "valid input rgb",
            make_grid(rgb[:64, :, :, :] * 255),
            dataformats="CHW",
            global_step=self.trainer.global_step,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.train_configs["learning-rate"]
        )
        lr_scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimizer, mode="min", factor=self.train_configs["lr-decay"], patience=2
            ),
            "monitor": "val_loss",
            "name": "plateau_scheduler",
        }
        return [optimizer], [lr_scheduler]


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
        # [print(f.shape) for f in features]
        features = features[::-1]

        x = features[0]
        skips = features[1:]

        for i, block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
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
        upsampling: Optional[_size_2_t] = 2,
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
        if isinstance(upsampling, list):
            size = upsampling
            factor = None
        else:
            size = None
            factor = upsampling
        self.upsample = torch.nn.Upsample(
            size=size, scale_factor=factor, mode=upsampling_mode
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
