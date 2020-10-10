import torch
import yaml
import argparse
import pytorch_lightning as pl
from pathlib import Path
from torch.utils.data.dataloader import DataLoader
from utils.dataset import RoadBoundaryDataset
from utils.feature_net import FeatureNet


def train(opt):
    with Path(opt.configs).open("rb") as f:
        configs = yaml.safe_load(f)

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

    # TODO define validation

    model = FeatureNet(configs=configs)

    trainer = pl.Trainer(gpus=1)
    trainer.fit(model, train_loader)


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
