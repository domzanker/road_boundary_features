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


def train(opt):
    with Path(opt.configs).open("rb") as f:
        configs = yaml.safe_load(f)

    if opt.batch_size != 0:
        configs["train"]["batch-size"] = opt.batch_size

    if len(opt.gpu) == 1:
        dist_backend = None
    else:
        dist_backend = "ddp"
    if opt.gpu[0] == -1:
        opt.gpu = -1

    train_dataset = RoadBoundaryDataset(
        path=Path(configs["dataset"]["train-dataset"]),
        image_size=configs["dataset"]["size"],
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
        model = FeatureNet(configs=configs)

    logger = TensorBoardLogger("data/tensorboard", opt.tag)
    if opt.resume_training:
        trainer = pl.Trainer(
            gpus=opt.gpu,
            distributed_backend=dist_backend,
            max_epochs=configs["train"]["epochs"],
            limit_val_batches=configs["train"]["validation-batches"],
            val_check_interval=configs["train"]["validation-interval"],
            logger=logger,
            log_every_n_steps=configs["train"]["logger-interval"],
            log_gpu_memory=True,
            checkpoint_callback=checkpoint_callback,
            resume_from_checkpoint=checkpoint_file,
            callbacks=[gpustats, lr_monitor],
            fast_dev_run=opt.test_run,
        )
    else:
        trainer = pl.Trainer(
            gpus=opt.gpu,
            distributed_backend=dist_backend,
            max_epochs=configs["train"]["epochs"],
            limit_val_batches=configs["train"]["validation-batches"],
            val_check_interval=configs["train"]["validation-interval"],
            logger=logger,
            log_every_n_steps=configs["train"]["logger-interval"],
            log_gpu_memory=True,
            checkpoint_callback=checkpoint_callback,
            callbacks=[gpustats, lr_monitor],
            fast_dev_run=opt.test_run,
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

    parser.add_argument("--test_run", bool, default=False, help="")
    # FIXME resume training

    opt = parser.parse_args()

    if opt.test_run:
        opt.tag = "test"

    train(opt)
