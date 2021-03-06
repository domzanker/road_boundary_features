from typing import Any, Dict, List, Optional, Tuple, Union
import segmentation_models_pytorch as smp

import torch
from math import sqrt
from torch.nn import Module, Sequential
from torch.nn.common_types import _size_2_t
from torchvision.utils import make_grid
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.modules import (
    ConvBlock,
    Conv2dAuto,
    ResidualBlock,
    SegmentationHead,
    activation_func,
)
from utils.losses import MultiTaskUncertaintyLoss

from utils.dataset import RoadBoundaryDataset
import pytorch_lightning as pl
from pytorch_lightning.metrics.metric import Metric
from utils.image_transforms import angle_map, apply_colormap


class FeatureNet(pl.LightningModule):
    def __init__(self, configs: Dict[str, Any]):
        super(FeatureNet, self).__init__()

        self.model_configs = configs["model"]
        self.train_configs = configs["train"]

        self.learning_rate = self.train_configs["learning-rate"]

        if self.model_configs["use_custom"]:

            self.encoder_prec = Sequential(
                torch.nn.BatchNorm2d(self.model_configs["input_channels"]),
                ConvBlock(
                    in_channels=self.model_configs["input_channels"],
                    **self.model_configs["encoder_prec"]["conv"],
                ),
                torch.nn.MaxPool2d(**self.model_configs["encoder_prec"]["pool"]),
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
        self.loss = MultiTaskUncertaintyLoss(**self.train_configs["losses"])

        self.train_mse = pl.metrics.MeanSquaredError()
        self.train_dist_mse = pl.metrics.MeanSquaredError()
        self.train_end_mse = pl.metrics.MeanSquaredError()
        self.train_dir_mse = MeanAbsAngleDeviation()

        self.val_dir_mse = MeanAbsAngleDeviation()
        self.val_dist_mse = pl.metrics.MeanSquaredError()
        self.val_end_mse = pl.metrics.MeanSquaredError()
        self.val__mse = pl.metrics.MeanSquaredError()

        self.save_hyperparameters()

        if "input_size" in self.model_configs.keys():
            s = self.model_configs["input_size"]
            self.example_input_array = torch.ones(
                [1, self.model_configs["input_channels"], s[0], s[1]]
            )

    def forward(self, x):

        x = self.encoder_prec(x)
        # pass through encoder, saving skip connections
        x = self.encoder(x)
        x = self.decoder(*x)
        x = self.head(x)
        return x

    def on_epoch_start(self):
        self.log("lr", self.learning_rate)

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
            "train_distance_variance": losses["loss_variance"][0].detach(),
            "train_end_variance": losses["loss_variance"][1].detach(),
            "train_direction_variance": losses["loss_variance"][2].detach(),
        }
        self.log_dict(loss_dict)

        prediction = segmentation.detach()
        targets = y.detach()
        self.log_dict(
            {
                "train_mse": self.train_mse(prediction, targets).item(),
                "train_dist_mse": self.train_dist_mse(
                    prediction[:, :1], targets[:, :1]
                ).item(),
                "train_end_mse": self.train_end_mse(
                    prediction[:, 1:2], targets[:, 1:2]
                ).item(),
                "train_dir_mse": self.train_dir_mse(
                    prediction[:, 2:4], targets[:, 2:4]
                ).item(),
            },
            on_step=True,
        )

        return losses["total_loss"]

    def validation_step(self, batch, batch_idx):
        x, y = batch
        prec = self.encoder_prec(x)
        encoding = self.encoder(prec)
        decoding = self.decoder(*encoding)
        segmentation = self.head(decoding)

        losses = self.loss(segmentation, y)

        loss_dict = {
            "val_distance_loss": losses["distance_loss"].detach(),
            "val_end_loss": losses["end_loss"].detach(),
            "val_direction_loss": losses["direction_loss"].detach(),
            "val_distance_variance": losses["loss_variance"][0].detach(),
            "val_end_variance": losses["loss_variance"][1].detach(),
            "val_direction_variance": losses["loss_variance"][2].detach(),
        }
        self.log_dict(loss_dict)
        self.log("val_loss", losses["total_loss"].item(), prog_bar=True, logger=True)
        self.log_dict(
            {
                "val_mse": self.val__mse(segmentation, y).item(),
                "val_dist_mse": self.val_dist_mse(segmentation[:, :1], y[:, :1]).item(),
                "val_end_mse": self.val_end_mse(segmentation[:, 1:2], y[:, 1:2]).item(),
                "val_dir_mad": self.val_dir_mse(segmentation[:, 2:4], y[:, 2:4]).item(),
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

        y = torch.cat([t["y"] for t in outputs]).detach().cpu()
        x = torch.cat([t["x"] for t in outputs]).detach().cpu()
        pred = torch.cat([t["pred"] for t in outputs]).detach().cpu()

        nmbr_images = self.train_configs["nmbr-logged-images"]
        nrows = 12

        # log out out
        dist = y[:, 0:1, :, :]
        dist_dbg = make_grid(apply_colormap(dist[:nmbr_images, :, :, :]), nrow=nrows)
        self._log_image(dist_dbg, "validation distance map")

        end = y[:, 1:2, :, :]
        end_dgb = make_grid(apply_colormap(end[:nmbr_images, :, :, :]), nrow=nrows)
        self._log_image(end_dgb, "validation end point map")

        direc = y[:, 2:4, :, :]
        dir_dbg = make_grid(angle_map(direc[:nmbr_images, :, :, :]), nrow=nrows)
        self._log_image(dir_dbg, "validation direction map")

        pred = pred
        dist = pred[:, 0:1, :, :].detach()
        dist_pred = make_grid(apply_colormap(dist[:nmbr_images, :, :, :]), nrow=nrows)
        self._log_image(dist_pred, "validation distance prediction")
        dist_comp = make_grid([dist_dbg, dist_pred], nrow=1)
        self._log_image(dist_comp, "validation distance comparison")

        end = pred[:, 1:2, :, :].detach()
        end_pred = make_grid(apply_colormap(end[:nmbr_images, :, :, :]), nrow=nrows)
        self._log_image(end_pred, "validation end point prediction")
        end_comp = make_grid([end_dgb, end_pred], nrow=1)
        self._log_image(end_comp, "validation end point comparison")

        direc = pred[:, 2:4, :, :].detach()
        dir_pred = make_grid(angle_map(direc[:nmbr_images, :, :, :]), nrow=nrows)
        self._log_image(dir_pred, "validation direction prediction")
        dir_comp = make_grid([dir_dbg, dir_pred], nrow=1)
        self._log_image(dir_comp, "validation direction comparison")

        lid = x[:, 3:4, :, :]
        lid -= lid.min()
        lid /= lid.max()
        lid = make_grid(apply_colormap(lid[:nmbr_images, :, :, :]), nrow=nrows)
        self._log_image(lid, "validation lidar input")

        lid_deriv = x[:, 4:, :, :]
        lid_deriv -= lid_deriv.min()
        lid_deriv /= lid_deriv.max()
        lid_deriv = make_grid(
            apply_colormap(lid_deriv[:nmbr_images, :, :, :]), nrow=nrows
        )
        self._log_image(lid_deriv, "validation lidar height deriv")

        rgb = make_grid(x[:nmbr_images, :3, :, :], nrow=nrows)
        self._log_image(rgb, "validation input rgb")

    def test_step(self, batch, batch_idx):
        x, y = batch
        prec = self.encoder_prec(x)
        encoding = self.encoder(prec)
        decoding = self.decoder(*encoding)
        segmentation = self.head(decoding)

        losses = self.loss(segmentation, y)
        return losses

    def _log_image(self, img, tag):
        self.logger[0].experiment.add_image(
            tag=tag,
            img_tensor=img,
            dataformats="CHW",
            global_step=self.trainer.global_step,
        )
        self.logger[1].experiment.log_image(
            img,
            name=tag,
            image_channels="first",
            step=self.trainer.global_step,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.train_configs["weight_decay"],
        )
        lr_scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.train_configs["lr-decay"],
                patience=self.train_configs["lr_decay_patience"],
            ),
            "monitor": "val_loss",
            "name": "plateau_scheduler",
        }
        return [optimizer], [lr_scheduler]

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        return items


class Decoder(Module):
    def __init__(
        self,
        decoder_depth: int,
        conv_per_block: int,
        stages: Union[List[Dict[str, Any]], Dict[str, Dict[str, Any]]],
        *args,
        **kwargs,
    ):
        super(Decoder, self).__init__()
        if isinstance(stages, dict):
            stages = list(stages.values())
        self.stages = torch.nn.ModuleList(
            [
                DecoderStage(**stages[c], nmbr_convs=conv_per_block)
                for c in range(decoder_depth)
            ]
        )

    def forward(self, *features):
        # features = features[1:]
        # [print(f.shape) for f in features]
        features = features[::-1]

        x = features[0]
        skips = [None, *features[1:]]

        for i, stage in enumerate(self.stages):
            skip = skips[i] if i < len(skips) else None
            x = stage(x, skip)
        return x


class DecoderStage(Module):
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
        upsample_indx: int = 0,
        upsampling: Optional[_size_2_t] = 2,
        apply_instance_norm: bool = False,
        *args,
        **kwargs,
    ):
        super(DecoderStage, self).__init__()
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

        self.stage = torch.nn.ModuleList(
            [
                ConvBlock(
                    in_channels=in_channels[i],
                    out_channels=out_channels[i],
                    kernel_size=kernel_size[i],
                    stride=stride[i],
                    dilation=dilation[i],
                    batch_norm=False,
                    activation=activation,
                    *args,
                    **kwargs,
                )
                for i in range(nmbr_convs)
            ]
        )
        if isinstance(upsampling, list):
            size = upsampling
            factor = None
        else:
            size = None
            factor = upsampling
        self.upsample_indx = upsample_indx
        if upsampling_mode == "transposed":
            self.upsample = torch.nn.ConvTranspose2d(
                in_channels=in_channels[upsample_indx],
                out_channels=in_channels[upsample_indx],
                kernel_size=kernel_size[upsample_indx],
                stride=factor,
                padding=1,
                output_padding=1,
                dilation=1,
            )
        else:
            self.upsample = torch.nn.Upsample(
                size=size,
                scale_factor=factor,
                mode=upsampling_mode,
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

        for i, layer in enumerate(self.stage):
            if i == self.upsample_indx:
                x = self.upsample(x)
            x = self.instance_normalize[i](x)
            x = layer(x)
        return x


class Encoder(Module):
    def __init__(
        self,
        encoder_depth: Optional[int] = None,
        in_channels: Optional[Union[int, List[int]]] = None,
        out_channels: Optional[List[int]] = None,
        **kwargs,
    ):
        super(Encoder, self).__init__()

        if "stages" in kwargs.keys():
            self.stages = torch.nn.ModuleList(
                [EncoderStage(**blocks) for key, blocks in kwargs["stages"].items()]
            )
        else:
            if encoder_depth is None or in_channels is None:
                raise AttributeError

            if not isinstance(in_channels, list):
                # per default double the number of channels with every encoder block
                in_channels = [in_channels * 2 ** c for c in range(encoder_depth)]
            if out_channels is None:
                out_channels = in_channels[1:]
                out_channels.append(in_channels[-1] * 2)
            self.blocks = torch.nn.ModuleList(
                [
                    EncoderStage(
                        in_channels=in_channels[i],
                        out_channels=out_channels[i],
                        **kwargs,
                    )
                    for i in range(encoder_depth)
                ]
            )

    def forward(self, x):
        out = []
        for stage in self.stages:
            x = stage(x)
            out.append(x)
        return out


class EncoderStage(Module):
    """
    A EncoderStage consists of 2 residual blocks
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        nmbr_res_blocks: int = 2,
        kernel_size: Union[_size_2_t, List[_size_2_t]] = 3,
        dilation: Union[_size_2_t, List[_size_2_t]] = 1,
        downsample_factor: int = 2,
        *args,
        **kwargs,
    ):
        super(EncoderStage, self).__init__()
        if not isinstance(kernel_size, list):
            kernel_size = [kernel_size for i in range(1, nmbr_res_blocks)]
        if not isinstance(dilation, list):
            dilation = [dilation for i in range(1, nmbr_res_blocks)]

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.head_stage = ResidualBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=[downsample_factor, 1, 1],
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
        x = self.head_stage(x)
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


class MeanAbsAngleDeviation(Metric):
    """
    Compute the mean angle deviation between two vector fields
    """

    def __init__(self):
        super().__init__()

        self.add_state(
            "sum_angle_deviation",
            default=torch.tensor(0, dtype=torch.float64),
            dist_reduce_fx="sum",
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self._check_same_shape(preds, target)

        angles_deviation = torch.abs(
            torch.acos(torch.nn.functional.cosine_similarity(preds, target, dim=-1))
        )

        self.sum_angle_deviation += angles_deviation.sum()

        self.total += target.numel()

    def compute(self):
        return self.sum_angle_deviation.float() / self.total


if __name__ == "__main__":
    # e = Encoder(3, 3)
    pass
