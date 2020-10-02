# from models import *
import os
import sys
import time
import datetime
import argparse
import yaml
from pathlib import Path

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss

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

    outpath = Path("data/outputs/%s" % opt.tag)
    outpath.mkdir(parents=True, exist_ok=True)

    model_path = Path("data/models")
    model_path.mkdir(parents=True, exist_ok=True)

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

    preprocessing_fn = smp.encoders.get_preprocessing_fn(
        encoder_name=model_configs["encoder"],
        pretrained=model_configs["encoder_weights"],
    )
    segmentation_head = SegmentationHead(branch_definition=model_configs["head"])
    model = torch.nn.Sequential(preprocessing_fn, encoder, segmentation_head)
    model.to(device)

    # Get dataloader
    train_dataset = RoadBoundaryDataset(
        path=Path(configs["dataset"]["train-dataset"]), transform=preprocessing_fn
    )
    valid_dataset = RoadBoundaryDataset(
        path=Path(configs["dataset"]["test-dataset"]), transform=preprocessing_fn
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
    criterion = CombinedLoss()
    trainer = create_supervised_trainer(
        model, optimizer=optimizer, loss_fn=criterion, device=device
    )

    val_metrics = {"accuracy": Accuracy(), "loss": Loss(criterion)}
    evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)

    log_interval = 100

    @trainer.on((Events.ITERATION_COMPLETED(every=log_interval)))
    def log_training_loss(trainer):
        print(
            "Epoch[{}] Loss: {:.2f}]".format(trainer.sate.epoch, trainer.state.output)
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        print(
            "Training Results - Epoch: {} Avg accuracy: {:.2f} Avg loss: {:.2f}]]".format(
                trainer.state.epoch, metrics["accuracy"], metrics["loss"]
            )
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        print(
            "Validation Results - Epoch: {} Avg accuracy: {:.2f} Avg loss: {:.2f}]]".format(
                trainer.state.epoch, metrics["accuracy"], metrics["loss"]
            )
        )

        trainer.run(train_loader, max_epochs=100)


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
