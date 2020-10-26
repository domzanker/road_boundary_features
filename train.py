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


data_dict = {
    "road_boundary_dataset": RoadBoundaryDataset,
    "image_dataset": ImageDataset,
}


def clean_dir(dir):
    for file in os.scandir(dir):
        if file.is_dir():
            clean_dir(file)
        elif file.is_file():
            os.remove(file)


def find_gpu_configs(opt):
    if opt.gpu is not None:
        if opt.gpu[0] == -1:
            gpu = -1
        else:
            gpu = opt.gpu
        if len(gpu) > 1:
            backend = opt.distributed_backend
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu))
        else:
            backend = None

    else:
        backend = None
        gpu = opt.gpu

    return gpu, backend


def trainer_configs(opt, configs):
    gpu, distributed_backend = find_gpu_configs(opt)
    if (
        "accumulate_grad_batches" in configs["train"].keys()
        and opt.accumulate_grad_batches == 0
    ):
        acc = configs["train"]["accumulate_grad_batches"]
    else:
        acc = opt.accumulate_grad_batches

    tr_configs = {
        "gpus": gpu,
        "distributed_backend": distributed_backend,
        "accumulate_grad_batches": acc,
        "max_epochs": configs["train"]["epochs"],
        "limit_val_batches": configs["train"]["validation-batches"],
        "val_check_interval": configs["train"]["validation-interval"],
        "log_every_n_steps": configs["train"]["logger-interval"],
        "log_gpu_memory": True,
        "profiler": opt.profiler,
    }
    if "trainer_args" in configs["train"].keys():
        for key, value in configs["train"]["trainer_args"].items():
            tr_configs[key] = value
    return tr_configs


def train(opt):
    with Path(opt.configs).open("rb") as f:
        configs = yaml.load(f, Loader)

    if opt.batch_size != 0:
        configs["train"]["batch-size"] = opt.batch_size

    trainer_confs = trainer_configs(opt, configs)

    if opt.find_lr and trainer_confs["distributed_backend"] is not None:
        print("Learning rate finder is not implemented for distributed environment!")
        opt.find_lr = False

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
        num_workers=max(1, int(opt.cpu_workers // 4)),
        pin_memory=True,
    )

    # FIXME
    # gpustats = GPUStatsMonitor(temperature=True)
    lr_monitor = LearningRateMonitor()
    if opt.checkpoint is not None:
        checkpoint_callback = ModelCheckpoint(
            filepath="data/checkpoints/" + opt.name + "/{epoch}", period=1, verbose=True
        )

        checkpoint_file = opt.checkpoint

        if opt.autoencoder:
            model = AutoEncoder.load_from_checkpoint(checkpoint_file, strict=False)
        else:
            model = FeatureNet.load_from_checkpoint(checkpoint_file, strict=False)
    else:
        clean_dir("data/checkpoints/" + opt.name)
        if opt.autoencoder:
            model = AutoEncoder(configs=configs)
        else:
            model = FeatureNet(configs=configs)

    logger = TensorBoardLogger("data/tensorboard", opt.tag)
    clean_dir("data/comet_ml/")
    comet_logger = CometLogger(
        save_dir="data/comet_ml",
        project_name="road-boundary-features",
        experiment_name=opt.name,
        experiment_key=opt.comet,
    )

    if opt.resume_training:
        trainer = pl.Trainer(
            logger=[logger, comet_logger],
            checkpoint_callback=checkpoint_callback,
            resume_from_checkpoint=checkpoint_file,
            callbacks=[lr_monitor],
            **trainer_confs
        )
    else:
        trainer = pl.Trainer(
            logger=[logger, comet_logger],
            checkpoint_callback=checkpoint_callback,
            callbacks=[lr_monitor],
            **trainer_confs
        )

    comet_logger.experiment.set_model_graph(str(ModelSummary(model, mode="full")))

    if opt.find_lr:
        lr_finder = trainer.tuner.lr_find(model, train_loader)
        new_lr = lr_finder.suggestion()
        print("using learning rate suggestion %s" % new_lr)
        model.hparams.learning_rate = new_lr
        model.learning_rate = new_lr

        fig = lr_finder.plot(suggest=True)
        logger.experiment.add_figure(tag="learning rate finder", figure=fig)
        comet_logger.experiment.log_figure("learning rate finder", figure=fig)

    trainer.fit(model, train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cpu_workers", type=int, default=8, help="number of cpu threads for loading"
    )
    parser.add_argument("--gpu", default=None, type=int, nargs="+", help="gpu")
    parser.add_argument("--distributed_backend", default="ddp", help="gpu")
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=0,
        help="accumulate_grad_batches",
    )
    parser.add_argument(
        "--batch_size", type=int, default=0, help="batch_size. this overrides configs"
    )
    parser.add_argument("--configs", type=str, default="params.yaml", help="")
    parser.add_argument("--name", type=str, default="training", help="")
    parser.add_argument(
        "--tags", type=str, nargs="+", default="", help="Tags for comet logger"
    )
    parser.add_argument("--resume_training", type=bool, default=False, help="")
    parser.add_argument("--checkpoint", type=str, default=None, help="")
    parser.add_argument("--comet", type=str, default=None, help="")

    parser.add_argument("--use_encoder", action="store_true", default=False, help="")
    parser.add_argument("--profile", action="store_true", default=False, help="")
    parser.add_argument("--autoencoder", action="store_true", default=False, help="")
    parser.add_argument("--find_lr", action="store_true", default=False, help="")

    # FIXME resume training

    opt = parser.parse_args()

    train(opt)
