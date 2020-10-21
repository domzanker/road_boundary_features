import pytorch_lightning as pl
from modules.feature_net import FeatureNet, Encoder, Decoder
from typing import List, Union
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
from torch.nn import Module, ModuleList, Identity, ConvTranspose2d, Sequential
from torchvision.utils import make_grid
from utils.modules import activation_func, Conv2dAuto
from utils.losses import loss_func
from utils.image_transforms import apply_colormap


class AutoEncoder(FeatureNet):
    def __init__(self, configs):
        super(AutoEncoder, self).__init__(configs)

        self.encoder_prec = Identity()
        self.decoder = AEDecoder(self.encoder)
        self.encoder_prec = Conv2dAuto(
            in_channels=4,
            out_channels=self.model_configs["encoder"]["in_channels"][0],
            kernel_size=1,
        )

        self.head = ConvTranspose2d(
            in_channels=64,
            out_channels=4,
            kernel_size=1,
            padding=self.encoder_prec.padding,
        )
        self.loss = loss_func("mse")

    def training_step(self, batch, batch_idx):
        x, y = batch

        prec = self.encoder_prec(x)
        encoding = self.encoder(prec)
        pred = self.decoder(*encoding)
        pred = self.head(pred)

        loss = self.loss(pred, x)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        prec = self.encoder_prec(x)
        encoding = self.encoder(prec)
        pred = self.decoder(*encoding)
        pred = self.head(pred)

        loss = self.loss(pred, x)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
        )

        return {"loss": loss.detach(), "prediction": pred.detach(), "input": x.detach()}

    def validation_epoch_end(self, outputs):

        tensorboard = self.logger[0].experiment
        comet = self.logger[1].experiment

        prediction = torch.cat([t["prediction"] for t in outputs]).cpu()
        x = torch.cat([t["input"] for t in outputs]).cpu()

        nmbr_images = self.train_configs["nmbr-logged-images"]
        nrows = 12

        # log out out
        input_rgb = make_grid(x[:nmbr_images, :3, :, :], nrow=nrows)

        tensorboard.add_image(
            tag="input rgb",
            img_tensor=input_rgb,
            dataformats="CHW",
            global_step=self.trainer.global_step,
        )
        comet.log_image(
            input_rgb,
            name="input rgb",
            image_channels="first",
            step=self.trainer.global_step,
        )

        input_lidar = make_grid(apply_colormap(x[:nmbr_images, 3:, :, :]), nrow=nrows)

        tensorboard.add_image(
            tag="input lidar",
            img_tensor=input_lidar,
            dataformats="CHW",
            global_step=self.trainer.global_step,
        )
        comet.log_image(
            input_lidar,
            name="input lidar",
            image_channels="first",
            step=self.trainer.global_step,
        )

        pred_rgb = make_grid(prediction[:nmbr_images, :3, :, :], nrow=nrows)

        tensorboard.add_image(
            tag="prediction rgb",
            img_tensor=pred_rgb,
            dataformats="CHW",
            global_step=self.trainer.global_step,
        )
        comet.log_image(
            pred_rgb,
            name="prediction rgb",
            image_channels="first",
            step=self.trainer.global_step,
        )

        pred_lidar = make_grid(
            apply_colormap(prediction[:nmbr_images, 3:, :, :]), nrow=nrows
        )

        tensorboard.add_image(
            tag="pred_lidar lidar",
            img_tensor=pred_lidar,
            dataformats="CHW",
            global_step=self.trainer.global_step,
        )
        comet.log_image(
            pred_lidar,
            name="prediction lidar",
            image_channels="first",
            step=self.trainer.global_step,
        )


class AEDecoder(Module):
    def __init__(self, encoder: Encoder, *args, **kwargs):
        super(AEDecoder, self).__init__(*args, **kwargs)

        decoder_blocks = []
        for block in encoder.blocks[::-1]:
            out_padding = 1  # TODO
            conv = ConvTranspose2d(
                in_channels=block.out_channels,
                out_channels=block.in_channels,
                kernel_size=3,
                stride=2,
                dilation=block.head_block.blocks[0].dilation,
                padding=block.head_block.blocks[0].padding,
                output_padding=out_padding,
            )
            decoder_blocks.append(conv)
            decoder_blocks.append(activation_func("relu"))
        self.blocks = Sequential(*decoder_blocks)

    def forward(self, *features):
        features = features[::-1]
        x = features[0]
        x = self.blocks(x)
        return x
