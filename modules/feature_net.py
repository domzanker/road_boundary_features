from typing import Any, Dict, List, Optional, Tuple, Union
import segmentation_models_pytorch as smp

import torch
from math import sqrt
from torch.nn import Module, Sequential
from torch.nn.common_types import _size_2_t
from torchvision.utils import make_grid
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.modules import Conv2dAuto, ResidualBlock, SegmentationHead, activation_func
from utils.losses import MultiFeaturesLoss

from utils.dataset import RoadBoundaryDataset
import pytorch_lightning as pl
from utils.image_transforms import angle_map, apply_colormap


class FeatureNet(pl.LightningModule):
    def __init__(self, configs: Dict[str, Any]):
        super(FeatureNet, self).__init__()

        self.model_configs = configs["model"]
        self.train_configs = configs["train"]

        self.learning_rate = self.train_configs["learning-rate"]

        if self.model_configs["use_custom"]:

            self.encoder_prec = Sequential(
                Conv2dAuto(
                    in_channels=self.model_configs["input_channels"],
                    **self.model_configs["encoder_prec"],
                ),
                torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
            self.encoder = Encoder(**self.model_configs["encoder"])
            self.decoder = Decoder(**self.model_configs["decoder"])
            self.head = SegmentationHead(**self.model_configs["head"])

        else:

            # TODO preprocessing_params
            self.encoder_prec = torch.nn.Identity()

            model = smp.Linknet(**self.model_configs["model"])
            self.encoder = model.encoder

            self.decoder = model.decoder
            self.head = Sequential(model.segmentation_head, activation_func("sigmoid"))

            self.preprocessing_params = smp.encoders.get_preprocessing_params(
                self.model_configs["model"]["encoder_name"],
                self.model_configs["model"]["encoder_weights"],
            )

        """
        loss_args = (
            self.train_configs["loss-args"]
            if "loss-args" in self.train_configs.keys()
            else {}
        )
        """
        self.loss = MultiFeaturesLoss(**self.train_configs["losses"])

        self.train_mse = pl.metrics.MeanSquaredError()
        self.train_dist_accuracy = pl.metrics.Accuracy()

        self.val_mse = pl.metrics.MeanSquaredError()
        self.val_dist_accuracy = pl.metrics.Accuracy()

        self.save_hyperparameters()

        if "input_size" in self.model_configs.keys():
            s = self.model_configs["input_size"]
            self.example_input_array = torch.ones([1, 4, s[0], s[1]])

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

        losses = self.loss(segmentation, y)

        # logging to tensorboard
        self.log("train_loss", losses["total_loss"])
        loss_dict = {
            "train_distance_loss": losses["distance_loss"].detach(),
            "train_end_loss": losses["end_loss"].detach(),
            "train_direction_loss": losses["direction_loss"].detach(),
        }
        self.log_dict(loss_dict)

        return losses["total_loss"]

    def validation_step(self, batch, batch_idx):
        x, y = batch
        prec = self.encoder_prec(x)
        encoding = self.encoder(prec)
        decoding = self.decoder(*encoding)
        segmentation = self.head(decoding)

        losses = self.loss(segmentation, y)

        loss_dict = {
            "train_distance_loss": losses["distance_loss"].detach(),
            "train_end_loss": losses["end_loss"].detach(),
            "train_direction_loss": losses["direction_loss"].detach(),
        }
        self.log_dict(loss_dict)

        pred = segmentation[:, :1, :, :].detach()
        tar = y[:, :1, :, :].detach()
        self.log_dict(
            {
                "val_loss": losses["total_loss"].item(),
                "val_mse": self.val_mse(pred, tar).item(),
                "val_dist_accuracy": self.val_dist_accuracy(pred, tar).item(),
            },
            on_step=False,
            on_epoch=True,
        )

        return {
            "loss": losses["total_loss"].detach(),
            "y": y.detach().cpu(),
            "pred": segmentation.detach().cpu(),
            "x": x.detach().cpu(),
        }

    def validation_epoch_end(self, outputs):

        tensorboard = self.logger[0].experiment
        comet = self.logger[1].experiment

        y = torch.cat([t["y"] for t in outputs]).detach().cpu()
        x = torch.cat([t["x"] for t in outputs]).detach().cpu()
        pred = torch.cat([t["pred"] for t in outputs]).detach().cpu()

        nmbr_images = self.train_configs["nmbr-logged-images"]
        nrows = 12

        # log out out
        dist = y[:, 0:1, :, :]
        dist_dbg = make_grid(apply_colormap(dist[:nmbr_images, :, :, :]), nrow=nrows)

        tensorboard.add_image(
            tag="distance map",
            img_tensor=dist_dbg,
            dataformats="CHW",
            global_step=self.trainer.global_step,
        )
        comet.log_image(
            dist_dbg,
            name="distance map",
            image_channels="first",
            step=self.trainer.global_step,
        )

        end = y[:, 1:2, :, :]
        end_dgb = make_grid(apply_colormap(end[:nmbr_images, :, :, :]), nrow=nrows)

        tensorboard.add_image(
            tag="end map",
            img_tensor=end_dgb,
            dataformats="CHW",
            global_step=self.trainer.global_step,
        )
        comet.log_image(
            end_dgb,
            name="end map",
            image_channels="first",
            step=self.trainer.global_step,
        )

        direc = y[:, 2:4, :, :]
        dir_dbg = make_grid(angle_map(direc[:nmbr_images, :, :, :]), nrow=nrows)

        tensorboard.add_image(
            tag="direction map",
            img_tensor=dir_dbg,
            dataformats="CHW",
            global_step=self.trainer.global_step,
        )
        comet.log_image(
            dir_dbg,
            name="direction map",
            image_channels="first",
            step=self.trainer.global_step,
        )

        pred = pred
        # log out out
        dist = pred[:, 0:1, :, :].detach()
        dist_pred = make_grid(apply_colormap(dist[:nmbr_images, :, :, :]), nrow=nrows)
        tensorboard.add_image(
            img_tensor=dist_pred,
            tag="distance pred",
            dataformats="CHW",
            global_step=self.trainer.global_step,
        )
        comet.log_image(
            dist_pred,
            name="distance pred",
            image_channels="first",
            step=self.trainer.global_step,
        )

        end = pred[:, 1:2, :, :].detach()
        end_pred = make_grid(apply_colormap(end[:nmbr_images, :, :, :]), nrow=nrows)
        tensorboard.add_image(
            img_tensor=end_pred,
            tag="end pred",
            dataformats="CHW",
            global_step=self.trainer.global_step,
        )
        comet.log_image(
            end_pred,
            name="end pred",
            image_channels="first",
            step=self.trainer.global_step,
        )

        direc = pred[:, 2:4, :, :].detach()
        dir_pred = make_grid(angle_map(direc[:nmbr_images, :, :, :]), nrow=nrows)
        tensorboard.add_image(
            img_tensor=dir_pred,
            tag="direction pred",
            dataformats="CHW",
            global_step=self.trainer.global_step,
        )
        comet.log_image(
            dir_pred,
            name="direction pred",
            image_channels="first",
            step=self.trainer.global_step,
        )

        lid = x[:, 3:, :, :]
        lid -= lid.min()
        lid /= lid.max()
        lid = make_grid(apply_colormap(lid[:nmbr_images, :, :, :]), nrow=nrows)
        tensorboard.add_image(
            tag="valid input lidar",
            img_tensor=lid,
            dataformats="CHW",
            global_step=self.trainer.global_step,
        )
        comet.log_image(
            lid,
            name="valid input lidar",
            image_channels="first",
            step=self.trainer.global_step,
        )

        rgb = make_grid(x[:nmbr_images, :3, :, :], nrow=nrows)
        tensorboard.add_image(
            tag="valid input rgb",
            img_tensor=rgb,
            dataformats="CHW",
            global_step=self.trainer.global_step,
        )
        comet.log_image(
            rgb,
            name="valid input rgb",
            image_channels="first",
            step=self.trainer.global_step,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimizer, mode="min", factor=self.train_configs["lr-decay"], patience=1
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
        skips = [None, *features[1:]]

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

        if skip is not None:
            x = x + skip

        x = self.upsample(x)
        for i, layer in enumerate(self.block):
            x = self.instance_normalize[i](x)
            x = layer(x)
            x = self.activate(x)
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

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.head_block = ResidualBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=[2, 1, 1],
            kernel_size=kernel_size[0],
            dilation=dilation[0],
            *args,
            **kwargs,
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


class AEDecoder(Decoder):
    def __init__(self, *args, **kwargs):
        super(AEDecoder, self).__init__(*args, **kwargs)

    def forward(self, *features):
        # features = features[1:]
        # [print(f.shape) for f in features]
        features = features[::-1]
        x = features[0]
        for i, block in enumerate(self.blocks):
            x = block(x)
        return x


class AEHead(Module):
    def __init__(
        self,
        scale: int = 2,
        kernel_size: int = 7,
        in_channels: int = 64,
        out_channels: int = 4,
    ):
        super(AEHead, self).__init__()
        self.head = Sequential(
            torch.nn.Upsample(scale_factor=scale),
            Conv2dAuto(
                kernel_size=kernel_size,
                in_channels=in_channels,
                out_channels=out_channels,
            ),
            activation_func("relu"),
        )

    def forward(self, x):
        return self.head(x)


class Interpolate(Module):
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


if __name__ == "__main__":
    # e = Encoder(3, 3)
    pass
