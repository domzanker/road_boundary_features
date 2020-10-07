# from models import *
import os
import glob
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
from ignite.handlers import Checkpoint, DiskSaver
from ignite.contrib.handlers import ProgressBar, tensorboard_logger, LRScheduler
from ignite.utils import setup_logger

import torch
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import make_grid

from utils.dataset import RoadBoundaryDataset
from utils.losses import CombinedLoss
from utils.modules import FeatureExtrationNet
from utils.modules import defined_activations
from utils.image_transforms import angle_map
import segmentation_models_pytorch as smp


def train(opt):
    configs_file = Path(opt.configs)
    with configs_file.open("rb") as f:
        configs = yaml.safe_load(f)

    # define device (if available)
    device = torch.device(("cuda:%s" % opt.gpu) if torch.cuda.is_available() else "cpu")

    # Initiate model
    model = FeatureExtrationNet(configs["model"])
    model.to(device)

    # Get dataloader
    train_dataset = RoadBoundaryDataset(
        path=Path(configs["dataset"]["train-dataset"]),
        image_size=configs["dataset"]["size"],
        transform=model.preprocessing_params,
    )
    valid_dataset = RoadBoundaryDataset(
        path=Path(configs["dataset"]["valid-dataset"]),
        image_size=configs["dataset"]["size"],
        transform=model.preprocessing_params,
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

    def train_step(engine, batch):
        model.train()

        imgs, targets = batch
        imgs = imgs.to(device)

        targets = targets.to(device)
        dist_t = targets[:, 0:1, :, :]
        # end_t = targets[:, 1:2, :, :]
        # dir_t = targets[:, 2:4, :, :]
        targets = [dist_t]  # , end_t, dir_t]

        model.zero_grad()

        predictions = model(imgs)

        # compute loss function
        distLoss = torch.nn.functional.mse_loss(
            predictions[0], targets[0], reduction="mean"
        )
        """
        endLoss = torch.nn.functional.mse_loss(
            predictions[1], targets[1], reduction="sum"
        )
        dirLoss = torch.nn.functional.cosine_similarity(
            predictions[2], targets[2]
        ).sum()

        # weight = 10
        # ombined_loss = dirLoss + weight * distLoss + weight * endLoss
        """

        distLoss.backward()

        optimizer.step()

        kwargs = {
            "predictions": [p.detach() for p in predictions],
            "ground_trouth": [t.detach() for t in targets],
            "input": imgs.detach(),
        }

        return (
            # combined_loss.item(),
            distLoss.item(),
            # endLoss.item(),
            # dirLoss.item(),
            kwargs,
        )

    def valid_step(engine, batch):
        model.eval()

        imgs, targets = batch
        imgs = imgs.to(device)

        targets = targets.to(device)
        dist_t = targets[:, 0:1, :, :]
        # end_t = targets[:, 1:2, :, :]
        # dir_t = targets[:, 2:4, :, :]
        targets = [dist_t]  # , end_t, dir_t]

        model.zero_grad()

        predictions = model(imgs)

        # compute loss function
        distLoss = torch.nn.functional.mse_loss(
            predictions[0], targets[0], reduction="sum"
        )

        """
        endLoss = torch.nn.functional.mse_loss(
            predictions[1], targets[1], reduction="sum"
        )
        dirLoss = torch.nn.functional.cosine_similarity(
            predictions[2], targets[2]
        ).sum()

        weight = 10
        combined_loss = dirLoss + weight * distLoss + weight * endLoss
        """
        combined_loss = distLoss

        kwargs = {
            "input": imgs.detach(),
            "dist_pred": predictions[0].detach(),
            # "end_pred": predictions[1].detach(),
            # "dir_pred": predictions[2].detach(),
            "dist": dist_t.detach(),
            # "end": end_t.detach(),
            # "dir": dir_t.detach(),
            "loss": combined_loss.item(),
            "dist_loss": distLoss.item(),
            # "end_loss": endLoss.item(),
            # "dir_loss": dirLoss.item(),
        }

        return (
            torch.cat(predictions, dim=1).detach(),
            torch.cat(targets, dim=1).detach(),
            kwargs,
        )

    # define ignite objects
    trainer = Engine(train_step)
    train_evaluator = Engine(valid_step)
    valid_evaluator = Engine(valid_step)

    # setup learning rate scheduler
    step_scheduler = StepLR(
        optimizer,
        step_size=configs["train"]["lr_decay_step_size"],
        gamma=configs["train"]["lr-decay"],
    )
    scheduler = LRScheduler(step_scheduler)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, scheduler)

    # evaluator = Engine(valid_step)
    # define progress bar
    progress_bar = ProgressBar(persist=True)
    valid_bar = ProgressBar()
    valid_bar.attach(train_evaluator)
    valid_bar.attach(valid_evaluator)

    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, name="loss")
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, name="l_dist")
    # RunningAverage(output_transform=lambda x: x[2]).attach(trainer, name="l_end")
    # RunningAverage(output_transform=lambda x: x[3]).attach(trainer, name="l_dir")
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
    """
    RunningAverage(output_transform=lambda x: x[2]["end_loss"]).attach(
        train_evaluator, name="l_end"
    )
    RunningAverage(output_transform=lambda x: x[2]["dir_loss"]).attach(
        train_evaluator, name="l_dir"
    )
    """

    MeanPairwiseDistance(p=4, output_transform=lambda x: [x[0], x[1]]).attach(
        valid_evaluator, "mpd"
    )
    RunningAverage(output_transform=lambda x: x[2]["dist_loss"]).attach(
        valid_evaluator, name="l_dist"
    )
    """
    RunningAverage(output_transform=lambda x: x[2]["end_loss"]).attach(
        valid_evaluator, name="l_end"
    )
    RunningAverage(output_transform=lambda x: x[2]["dir_loss"]).attach(
        valid_evaluator, name="l_dir"
    )
    """

    tb_logger = tensorboard_logger.TensorboardLogger(
        log_dir="data/tensorboard/dist_only/tb_logs_{}".format(opt.tag),
    )

    tb_logger.attach_output_handler(
        trainer,
        tag="training",
        event_name=Events.ITERATION_COMPLETED,
        metric_names="all",
    )
    tb_logger.attach_output_handler(
        valid_evaluator,
        tag="validation",
        event_name=Events.EPOCH_COMPLETED,
        metric_names="all",
        global_step_transform=tensorboard_logger.global_step_from_engine(trainer),
    )
    tb_logger.attach_opt_params_handler(
        trainer, event_name=Events.ITERATION_STARTED, optimizer=optimizer
    )

    @trainer.on(Events.ITERATION_COMPLETED(every=100))
    def log_tensorboard_images(engine):
        out = engine.state.output
        d = out[4]
        predictions = torch.cat(d["predictions"], dim=1)
        predictions = predictions.cpu()
        ground_trouth = torch.cat(d["ground_trouth"], dim=1)
        ground_trouth = ground_trouth.cpu()
        inp = d["input"].cpu()
        im_1 = make_grid(predictions[:, 0:1, :, :], normalize=True, scale_each=True)
        """
        im_2 = make_grid(predictions[:, 1:2, :, :], normalize=True, scale_each=True)
        angle_im = angle_map(predictions[:, 2:4, :, :])
        im_3 = make_grid(angle_im, normalize=True, scale_each=True)
        """

        t_1 = make_grid(ground_trouth[:, 0:1, :, :], normalize=True, scale_each=True)
        """
        t_2 = make_grid(ground_trouth[:, 1:2, :, :], normalize=True, scale_each=True)
        angle_im = angle_map(ground_trouth[:, 2:4, :, :])
        t_3 = make_grid(angle_im, normalize=True, scale_each=True)
        """

        rgb = make_grid(inp[:, :3, :, :], normalize=True, scale_each=True)

        glob_step = trainer.state.iteration

        tb_logger.writer.add_image("dist_pred", im_1, global_step=glob_step)
        tb_logger.writer.add_image("dist_gt", t_1, global_step=glob_step)
        """
        tb_logger.writer.add_image("end_pred", im_2, global_step=glob_step)
        tb_logger.writer.add_image("end_gt", t_2, global_step=glob_step)
        tb_logger.writer.add_image("dir_pred", im_3, global_step=glob_step)
        tb_logger.writer.add_image("dir_gt", t_3, global_step=glob_step)
        """
        tb_logger.writer.add_image("rgb", rgb, global_step=glob_step)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        train_evaluator.run(train_loader, epoch_length=150, max_epochs=1)
        metrics = train_evaluator.state.metrics
        progress_bar.log_message(
            "Trainings results - Epoch: {} Mean Pairwise Distance: {}  << distanceMap: {:.4f}".format(
                engine.state.epoch, metrics["mpd"], metrics["l_dist"]
            )
        )
        """
        progress_bar.log_message(
            "Trainings results - Epoch: {} Mean Pairwise Distance: {}  << distanceMap: {:.4f} endMap: {:.4f} directionMap: {:.4f}".format(
                engine.state.epoch,
                metrics["mpd"],
                metrics["l_dist"],
                metrics["l_end"],
                metrics["l_dir"],
            )
        )
        """

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        valid_evaluator.run(val_loader, epoch_length=150, max_epochs=1)
        metrics = valid_evaluator.state.metrics
        progress_bar.log_message(
            "Trainings results - Epoch: {} Mean Pairwise Distance: {}  << distanceMap: {:.4f}".format(
                engine.state.epoch, metrics["mpd"], metrics["l_dist"]
            )
        )
        """
        progress_bar.log_message(
            "Validation results - Epoch: {} Mean Pairwise Distance: {}  << distanceMap: {:.4f} endMap: {:.4f} directionMap: {:.4f}".format(
                engine.state.epoch,
                metrics["mpd"],
                metrics["l_dist"],
                metrics["l_end"],
                metrics["l_dir"],
            )
        )
        """

    to_save = {"model": model, "optimizer": optimizer, "trainer": trainer}
    checkpoint_handler = Checkpoint(
        # TODO: implementation
        to_save=to_save,
        save_handler=DiskSaver(
            "data/checkpoints", require_empty=False, create_dir=True
        ),
        filename_prefix=opt.tag,
        n_saved=5,
    )

    def load_checkpoint():

        if opt.resume:
            to_load = to_save
            # if valid checkpoint exists (right tag)
            if opt.checkpoint is None:
                checkpoint_path = sorted(
                    glob.glob(
                        "data/checkpoints/%s*%s" % (opt.tag, checkpoint_handler.ext)
                    ),
                    reverse=True,
                )
                if len(checkpoint_path) > 0:
                    checkpoint_path = checkpoint_path[0]
                else:
                    progress_bar.log_message(
                        "no checkpoint found. starting from scratch"
                    )
                    return

            else:
                checkpoint_path = opt.checkpoint
            checkpoint = torch.load(checkpoint_path)
            checkpoint_handler.load_objects(to_load, checkpoint)
            progress_bar.log_message(
                "resumed training from checkpoint: %s" % checkpoint_path
            )

    load_checkpoint()

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED(every=configs["train"]["checkpoint-interval"]),
        checkpoint_handler,
    )

    trainer.run(train_loader, max_epochs=configs["train"]["epochs"])
    # trainer.run(train_loader, max_epochs=configs["train"]["epochs"], epoch_length=1)
    tb_logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cpu_workers", type=int, default=8, help="number of cpu threads for loading"
    )
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--configs", type=str, default="params.yaml", help="")
    parser.add_argument("--tag", type=str, default="training", help="")
    parser.add_argument("--resume", type=bool, default=False, help="")
    parser.add_argument("--checkpoint", type=str, default=None, help="")
    # FIXME resume training

    opt = parser.parse_args()

    train(opt)
