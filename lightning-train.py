import os
import torch
import yaml
import argparse
import pytorch_lightning as pl
from pathlib import Path
from torch.utils.data.dataloader import DataLoader
from utils.dataset import RoadBoundaryDataset
from utils.feature_net import FeatureNet
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    GPUStatsMonitor,
    LearningRateMonitor,
)
from utils.yaml import Loader
import segmentation_models_pytorch as smp


def get_learning_rate_suggestion(model, data_loader):
    lr_trainer = pl.Trainer()

    lr_finder = lr_trainer.tuner.lr_find(model=model, train_dataloader=data_loader)

    suggestion = lr_finder.suggestion()
    plot = lr_finder.plot()

    return suggestion, plot


def train(opt):
    with Path(opt.configs).open("rb") as f:
        configs = yaml.load(f, Loader)

    if opt.batch_size != 0:
        configs["train"]["batch-size"] = opt.batch_size

    dist_backend = "ddp"
    if len(opt.gpu) == 1:
        if opt.gpu[0] == -1:
            opt.gpu = -1
        else:
            dist_backend = None

    if "input_size" in configs["model"]:
        configs["dataset"]["size"] = configs["model"]["input_size"]
    if not configs["model"]["use_custom"]:
        preprocessing_params = smp.encoders.get_preprocessing_params(
            configs["model"]["model"]["encoder_name"],
            configs["model"]["model"]["encoder_weights"],
        )
    else:
        preprocessing_params = None

    train_dataset = RoadBoundaryDataset(
        path=Path(configs["dataset"]["train-dataset"]),
        image_size=configs["dataset"]["size"],
        transform=preprocessing_params,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=configs["train"]["batch-size"],
        shuffle=True,
        num_workers=opt.cpu_workers,
        pin_memory=True,
    )

    val_dataset = RoadBoundaryDataset(
        path=Path(configs["dataset"]["valid-dataset"]),
        image_size=configs["dataset"]["size"],
        transform=preprocessing_params,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=configs["train"]["batch-size"],
        num_workers=3,
        pin_memory=True,
    )

    checkpoint_callback = ModelCheckpoint(
        filepath="data/checkpoints/" + opt.tag + "/{epoch}", period=1, verbose=True
    )
    gpustats = GPUStatsMonitor(temperature=True)
    lr_monitor = LearningRateMonitor()
    if opt.load_model or opt.resume_training:
        if opt.checkpoint is not None:
            checkpoint_file = opt.checkpoint
        else:
            raise NotImplementedError
        model = FeatureNet.load_from_checkpoint(checkpoint_file)
    else:
        model = FeatureNet(configs=configs, pretrain=opt.autoencoder)

    logger = TensorBoardLogger("data/tensorboard", opt.tag)

    if opt.find_lr and os.environ["LOCAL_RANK"] == 0:
        suggested_lr, fig = get_learning_rate_suggestion(model, train_loader)
        print("Using suggested learning rate of : ", suggested_lr)
        configs["train"]["learning-rate"] = suggested_lr
        logger.log_graph(fig)

    if opt.resume_training:
        trainer = pl.Trainer(
            gpus=opt.gpu,
            distributed_backend=dist_backend,
            accumulate_grad_batches=4,
            max_epochs=configs["train"]["epochs"],
            limit_val_batches=configs["train"]["validation-batches"],
            val_check_interval=configs["train"]["validation-interval"],
            logger=logger,
            log_every_n_steps=configs["train"]["logger-interval"],
            log_gpu_memory=True,
            checkpoint_callback=checkpoint_callback,
            resume_from_checkpoint=checkpoint_file,
            callbacks=[gpustats, lr_monitor],
            profiler=opt.profile,
        )
    else:
        trainer = pl.Trainer(
            gpus=opt.gpu,
            auto_select_gpus=True,
            distributed_backend=dist_backend,
            accumulate_grad_batches=2,
            max_epochs=configs["train"]["epochs"],
            limit_val_batches=configs["train"]["validation-batches"],
            val_check_interval=configs["train"]["validation-interval"],
            logger=logger,
            log_every_n_steps=configs["train"]["logger-interval"],
            log_gpu_memory=True,
            checkpoint_callback=checkpoint_callback,
            callbacks=[gpustats, lr_monitor],
            profiler=opt.profile,
            overfit_batches=100,
        )
    trainer.fit(model, train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cpu_workers", type=int, default=8, help="number of cpu threads for loading"
    )
    parser.add_argument("--gpu", type=int, default=0, nargs="+", help="gpu")
    parser.add_argument(
        "--batch_size", type=int, default=0, help="batch_size. this overrides configs"
    )
    parser.add_argument("--configs", type=str, default="params.yaml", help="")
    parser.add_argument("--tag", type=str, default="training", help="")
    parser.add_argument("--load_model", type=bool, default=False, help="")
    parser.add_argument("--resume_training", type=bool, default=False, help="")
    parser.add_argument("--checkpoint", type=str, default=None, help="")

    parser.add_argument("--profile", action="store_true", default=False, help="")
    parser.add_argument("--autoencoder", action="store_true", default=False, help="")
    parser.add_argument("--find_lr", action="store_true", default=False, help="")
    # FIXME resume training

    opt = parser.parse_args()

    train(opt)
