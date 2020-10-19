import os
import torch
import yaml
import argparse
import pytorch_lightning as pl
from pathlib import Path
from torch.utils.data.dataloader import DataLoader

from utils.dataset import RoadBoundaryDataset, ImageDataset
from utils.yaml import Loader
from modules.feature_net import FeatureNet
from modules.autoencoder import AutoEncoder

from pytorch_lightning.loggers import TensorBoardLogger, CometLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    GPUStatsMonitor,
    LearningRateMonitor,
)
from pytorch_lightning.core.lightning import ModelSummary
import segmentation_models_pytorch as smp


"""
def get_learning_rate_suggestion(model, data_loader):
    lr_trainer = pl.Trainer(accumulate_grad_batches=2)

    lr_finder = lr_trainer.tuner.lr_find(model=model, train_dataloader=data_loader)

    suggestion = lr_finder.suggestion()
    plot = lr_finder.plot()

    lr_trainer.accelerator_backend.teardown()

    return suggestion, plot
"""

data_dict = {
    "road_boundary_dataset": RoadBoundaryDataset,
    "image_dataset": ImageDataset,
}


def train(opt):
    with Path(opt.configs).open("rb") as f:
        configs = yaml.load(f, Loader)

    if opt.batch_size != 0:
        configs["train"]["batch-size"] = opt.batch_size

    dist_backend = "ddp"
    if opt.gpu == 0:
        pass
    elif len(opt.gpu) == 1:
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

    dataset = data_dict[configs["dataset"]["name"]]

    train_dataset = dataset(
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

    val_dataset = dataset(
        path=Path(configs["dataset"]["valid-dataset"]),
        image_size=configs["dataset"]["size"],
        transform=preprocessing_params,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=configs["train"]["batch-size"],
        shuffle=True,
        num_workers=max(1, int(opt.cpu_workers // 4)),
        pin_memory=True,
    )

    checkpoint_callback = ModelCheckpoint(
        filepath="data/checkpoints/" + opt.tag + "/{epoch}", period=1, verbose=True
    )
    # gpustats = GPUStatsMonitor(temperature=True)
    lr_monitor = LearningRateMonitor()
    if opt.load_model or opt.resume_training:

        if opt.checkpoint is not None:
            checkpoint_file = opt.checkpoint
        else:
            raise NotImplementedError

        if opt.autoencoder:
            model = AutoEncoder.load_from_checkpoint(checkpoint_file)
        else:
            model = FeatureNet.load_from_checkpoint(checkpoint_file)
    else:
        if opt.autoencoder:
            model = AutoEncoder(configs=configs)
        else:
            model = FeatureNet(configs=configs)

    logger = TensorBoardLogger("data/tensorboard", opt.tag)
    comet_logger = CometLogger(
        save_dir="data/comet_ml",
        project_name="road-boundary-features",
        experiment_name=opt.tag,
    )

    """
    if opt.find_lr and os.environ.get("LOCAL_RANK", 0) == 0:
        suggested_lr, fig = get_learning_rate_suggestion(model, train_loader)
        print("Using suggested learning rate of : ", suggested_lr)
        configs["train"]["learning-rate"] = suggested_lr
        logger.log_graph(fig)
        logger.experiment.flush()
    """

    if opt.resume_training:
        trainer = pl.Trainer(
            gpus=opt.gpu,
            distributed_backend=dist_backend,
            accumulate_grad_batches=opt.accumulate_grad_batches,
            max_epochs=configs["train"]["epochs"],
            limit_val_batches=configs["train"]["validation-batches"],
            val_check_interval=configs["train"]["validation-interval"],
            logger=[logger, comet_logger],
            log_every_n_steps=configs["train"]["logger-interval"],
            log_gpu_memory=True,
            checkpoint_callback=checkpoint_callback,
            resume_from_checkpoint=checkpoint_file,
            # callbacks=[gpustats, lr_monitor],
            callbacks=[lr_monitor],
            profiler=opt.profile,
        )
    else:
        trainer = pl.Trainer(
            gpus=opt.gpu,
            auto_select_gpus=True,
            distributed_backend=dist_backend,
            accumulate_grad_batches=opt.accumulate_grad_batches,
            max_epochs=configs["train"]["epochs"],
            limit_val_batches=configs["train"]["validation-batches"],
            val_check_interval=configs["train"]["validation-interval"],
            logger=[logger, comet_logger],
            log_every_n_steps=configs["train"]["logger-interval"],
            log_gpu_memory=True,
            checkpoint_callback=checkpoint_callback,
            # callbacks=[gpustats, lr_monitor],
            callbacks=[lr_monitor],
            profiler=opt.profile,
            # overfit_batches=100,
        )

    comet_logger.experiment.set_model_graph(str(ModelSummary(model, mode="full")))
    trainer.fit(model, train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cpu_workers", type=int, default=8, help="number of cpu threads for loading"
    )
    parser.add_argument("--gpu", type=int, default=0, nargs="+", help="gpu")
    parser.add_argument(
        "--accumulate_grad_batches", type=int, default=2, help="accumulate_grad_batches"
    )
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
