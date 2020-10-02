# from models import *
import os
import sys
import time
import datetime
import argparse
import yaml
from pathlib import Path

from ignite.engine import Events, Engine
from ignite.metrics import (
    Accuracy,
    RunningAverage,
    Average,
    MeanPairwiseDistance,
    MeanSquaredError,
)
import ignite.contrib.metrics.regression as ireg
from ignite.contrib.metrics import GpuInfo
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import ProgressBar

import torch

from utils.dataset import RoadBoundaryDataset
from utils.losses import CombinedLoss
from utils.modules import SegmentationHead
from utils.modules import defined_activations
import segmentation_models_pytorch as smp


def train(opt):
    configs_file = Path(opt.configs)
    with configs_file.open("rb") as f:
        configs = yaml.safe_load(f)

    # define device (if available)
    device = torch.device(("cuda:%s" % opt.gpu) if torch.cuda.is_available() else "cpu")

    # Initiate model
    model_configs = configs["model"]
    if not model_configs["use_custom_encoder"]:
        encoder = smp.Linknet(
            encoder_name=model_configs["encoder"],
            encoder_weights=model_configs["encoder_weights"],
            in_channels=4,
            encoder_depth=model_configs["encoder_depth"],
            classes=64,
            activation=model_configs["activation"],
            decoder_use_batchnorm=model_configs["decoder_use_batchnorm"],
        )

    """
    preprocessing_fn = smp.encoders.get_preprocessing_fn(
        encoder_name=model_configs["encoder"],
        pretrained=model_configs["encoder_weights"],
    )
    """
    segmentation_head = SegmentationHead(branch_definition=model_configs["head"])
    model = torch.nn.Sequential(encoder, segmentation_head)
    model.to(device)

    # Get dataloader
    train_dataset = RoadBoundaryDataset(
        path=Path(configs["dataset"]["train-dataset"]),
        image_size=configs["dataset"]["size"],
        # transform=preprocessing_fn,
    )
    valid_dataset = RoadBoundaryDataset(
        path=Path(configs["dataset"]["test-dataset"]),
        image_size=configs["dataset"]["size"],
        # transform=preprocessing_fn,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=configs["train"]["batch-size"],
        shuffle=True,
        num_workers=opt.cpu_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=1, shuffle=False, num_workers=2
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=configs["train"]["learning-rate"],
        weight_decay=configs["train"]["weight_decay"],
    )
    # criterion = CombinedLoss()

    def train_step(engine, batch):
        model.train()

        imgs, targets = batch
        imgs = imgs.to(device)

        targets = targets.to(device)
        dist_t = targets[:, 0:1, :, :]
        end_t = targets[:, 1:2, :, :]
        dir_t = targets[:, 2:4, :, :]
        targets = [dist_t, end_t, dir_t]

        model.zero_grad()

        predictions = model(imgs)

        # compute loss function
        distLoss = torch.nn.functional.mse_loss(predictions[0], targets[0])
        endLoss = torch.nn.functional.mse_loss(predictions[1], targets[1])
        dirLoss = torch.nn.functional.cosine_similarity(
            predictions[2], targets[2]
        ).sum()

        weight = 10
        combined_loss = dirLoss + weight * distLoss + weight * endLoss

        combined_loss.backward()

        optimizer.step()

        return combined_loss.item(), distLoss.item(), endLoss.item(), dirLoss.item()

    def valid_step(engine, batch):
        model.eval()

        imgs, targets = batch
        imgs = imgs.to(device)

        targets = targets.to(device)
        dist_t = targets[:, 0:1, :, :]
        end_t = targets[:, 1:2, :, :]
        dir_t = targets[:, 2:4, :, :]
        targets = [dist_t, end_t, dir_t]

        model.zero_grad()

        predictions = model(imgs)

        # compute loss function
        distLoss = torch.nn.functional.mse_loss(predictions[0], targets[0])
        endLoss = torch.nn.functional.mse_loss(predictions[1], targets[1])
        dirLoss = torch.nn.functional.cosine_similarity(
            predictions[2], targets[2]
        ).sum()

        weight = 10
        combined_loss = dirLoss + weight * distLoss + weight * endLoss

        kwargs = {
            "dist_pred": predictions[0],
            "end_pred": predictions[1],
            "dir_pred": predictions[2],
            "dist": dist_t,
            "end": end_t,
            "dir": dir_t,
            "loss": combined_loss.item(),
            "dist_loss": distLoss.item(),
            "end_loss": endLoss.item(),
            "dir_loss": dirLoss.item(),
        }
        progress_bar.log_message("rest")
        return torch.cat(predictions), torch.cat(targets), kwargs

    # define ignite objects
    trainer = Engine(train_step)
    train_evaluator = Engine(valid_step)
    valid_evaluator = Engine(valid_step)

    # evaluator = Engine(valid_step)
    # define progress bar
    progress_bar = ProgressBar()

    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, name="loss")
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, name="l_dist")
    RunningAverage(output_transform=lambda x: x[2]).attach(trainer, name="l_end")
    RunningAverage(output_transform=lambda x: x[3]).attach(trainer, name="l_dir")
    GpuInfo().attach(trainer, name="gpu")
    progress_bar.attach(
        trainer,
        metric_names=[
            "loss",
            "gpu:{} mem(%)".format(opt.gpu),
            "gpu:{} util(%)".format(opt.gpu),
        ],
    )

    MeanPairwiseDistance(p=4, output_transform=lambda x: [x[0], x[1]]).attach(
        train_evaluator, "mpd"
    )
    RunningAverage(output_transform=lambda x: x[2]["dist_loss"]).attach(
        train_evaluator, name="l_dist"
    )
    RunningAverage(output_transform=lambda x: x[2]["end_loss"]).attach(
        train_evaluator, name="l_end"
    )
    RunningAverage(output_transform=lambda x: x[2]["dir_loss"]).attach(
        train_evaluator, name="l_dir"
    )

    MeanPairwiseDistance(p=4, output_transform=lambda x: [x[0], x[1]]).attach(
        valid_evaluator, "mpd"
    )
    RunningAverage(output_transform=lambda x: x[2]["dist_loss"]).attach(
        valid_evaluator, name="l_dist"
    )
    RunningAverage(output_transform=lambda x: x[2]["end_loss"]).attach(
        valid_evaluator, name="l_end"
    )
    RunningAverage(output_transform=lambda x: x[2]["dir_loss"]).attach(
        valid_evaluator, name="l_dir"
    )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        train_evaluator.run(train_loader)
        metrics = train_evaluator.state.metrics
        progress_bar.log_message(
            "Trainings results - Epoch: {} Mean Pairwise Distance: {}  << distanceMap: {:.2f} endMap: {:.2f} directionMap: {:.2f}".format(
                engine.state.epoch,
                metrics["mpd"],
                metrics["dist_loss"],
                metrics["end_loss"],
                metrics["dir_loss"],
            )
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        valid_evaluator.run(val_loader)
        metrics = valid_evaluator.state.metrics
        progress_bar.log_message(
            "Trainings results - Epoch: {} Mean Pairwise Distance: {}  << distanceMap: {:.2f} endMap: {:.2f} directionMap: {:.2f}".format(
                engine.state.epoch,
                metrics["mpd"],
                metrics["dist_loss"],
                metrics["end_loss"],
                metrics["dir_loss"],
            )
        )

    checkpoint_handler = ModelCheckpoint(
        dirname="data/models",
        filename_prefix=("model_%s" % opt.tag),
        n_saved=2,
        save_as_state_dict=True,
        require_empty=False,
        create_dir=True,
    )
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED(every=configs["train"]["checkpoint-interval"]),
        checkpoint_handler,
        {"model": model},
    )

    trainer.run(train_loader, max_epochs=configs["train"]["epochs"], epoch_length=100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="number of epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="size of each image batch"
    )
    parser.add_argument(
        "--cpu_workers", type=int, default=4, help="number of cpu threads for loading"
    )
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--checkpoint_interval", type=int, default=10, help="")
    parser.add_argument("--configs", type=str, default="params.yaml", help="")

    parser.add_argument("--tag", type=str, default="training", help="")

    opt = parser.parse_args()

    train(opt)
