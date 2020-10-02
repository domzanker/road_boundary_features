# from models import *
import os
import sys
import time
import datetime
import argparse
import yaml
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

from utils.dataset import RoadBoundaryDataset
from utils.losses import CombinedLoss
from utils.modules import SegmentationHead
from utils.modules import defined_activations
import segmentation_models_pytorch as smp


class Params:
    def __init__(self, config_dict: dict = None, file: Path = None):

        if file is not None:
            file = Path(file)
            assert file.suffix == ".yaml"
            with file.open(mode="rb") as f:
                config_dict = yaml.safe_load(f)

        for key, val in config_dict.items():
            if isinstance(val, dict):
                self.__dict__[key] = Params(config_dict=val)
            else:
                self.__dict__[key] = val

    def __str__(self):
        string, _ = self.__build_str__()
        return "\n".join(string)

    def __build_str__(self, string: str = [], parents: list = []):
        for key, val in self.__dict__.items():
            loc_parents = parents.copy()
            loc_parents.append(key)

            if isinstance(val, Params):
                string, loc_parents = val.__build_str__(
                    string=string, parents=loc_parents
                )
            else:
                s = ".".join(loc_parents)
                s += ": %s" % val
                string.append(s)
        return (string, parents)


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
    print(opt)

    configs_file = Path(opt.configs)
    with configs_file.open("rb") as f:
        configs = yaml.safe_load(f)

    # define device (if available)
    device = torch.device(("cuda:%s" % opt.gpu) if torch.cuda.is_available() else "cpu")

    outpath = Path("data/outputs/%s" % opt.tag)
    outpath.mkdir(parents=True, exist_ok=True)

    model_path = Path("data/models")
    model_path.mkdir(parents=True, exist_ok=True)

    tb_writer = SummaryWriter(str(outpath / "log"))

    # Get dataloader
    train_dataset = RoadBoundaryDataset(path=Path(configs["dataset"]["train-dataset"]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=configs["train"]["batch-size"],
        shuffle=True,
        num_workers=opt.cpu_workers,
        pin_memory=True,
    )

    valid_dataset = RoadBoundaryDataset(path=Path(configs["dataset"]["test-dataset"]))
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=1, shuffle=False, num_workers=2
    )

    # Initiate model
    model_configs = configs["model"]
    if not model_configs["use_custom_encoder"]:
        model = smp.Linknet(
            encoder_name=model_configs["encoder"],
            encoder_weights=model_configs["encoder_weights"],
            in_channels=4,
            encoder_depth=model_configs["encoder_depth"],
            classes=64,
            activation=model_configs["activation"],
            decoder_use_batchnorm=model_configs["decoder_use_batchnorm"],
        )
    model.to(device)

    preprocessing_fn = smp.encoders.get_preprocessing_fn(
        encoder_name=model_configs["encoder"],
        pretrained=model_configs["encoder_weights"],
    )

    segmentation_head = SegmentationHead(branch_definition=model_configs["head"])

    loss = CombinedLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=configs["train"]["learning-rate"],
        weight_decay=configs["train"]["weight_decay"],
    )

    num_epochs = configs["train"]["epochs"]
    for epoch in range(num_epochs):
        model.train()
        start_time = time.time()
        for batch_i, (imgs, targets) in enumerate(train_loader):

            batches_done = len(train_loader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            label_distance = targets[:, 0:1, :, :]
            label_end = targets[:, 1:2, :, :]
            label_direction = targets[:, 2:4, :, :]

            tb_writer.add_images("label_distance", label_distance, batches_done)
            # tb_writer.add_images("label_direction", label_direction, batches_done)
            tb_writer.add_images("label_end", label_end, batches_done)

            (feature_distance, feature_end, feature_direction) = model(imgs)

            # tb_writer.add_images("feature_direction", feature_direction, batches_done)
            tb_writer.add_images("feature_distance", feature_distance, batches_done)
            tb_writer.add_images("feature_end", feature_end, batches_done)

            # L(S,E,D) = l_det(S) + k1 * l_end(E) + k2 * l_dir(D)
            lo = loss(
                distance=(feature_distance, label_distance),
                end_points=(feature_end, label_end),
                direction=(feature_direction, label_direction),
            )

            tb_writer.add_scalar("loss", loss, batches_done)

            lo.backward()

            optimizer.step()
            optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n>>>> [Epoch %d/%d, Batch %d/%d] <<<< \n" % (
                epoch,
                num_epochs,
                batch_i,
                len(train_loader),
            )

            # Determine approximate time left for epoch
            epoch_batches_left = len(train_loader) - (batch_i + 1)
            time_left = datetime.timedelta(
                seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1)
            )
            log_str += f"\n>>>> ETA {time_left}"

            print(log_str)

            tb_writer.flush()

        if epoch % configs["train"]["eval-interval"] == 0 and epoch > 0:
            model.eval()
            print("\n>>>> Evaluating Model <<<<")
            with torch.no_grad():
                # FIXME
                for imgs, targets in valid_loader:
                    imgs = Variable(imgs.to(device))
                    targets = Variable(targets.to(device), requires_grad=False)

                    label_distance = targets[:, 0, :, :]
                    label_end = targets[:, 1, :, :]
                    label_direction = targets[:, 2:4, :, :]

                    (feature_distance, feature_end, feature_direction) = model(imgs)

                    # L(S,E,D) = l_det(S) + k1 * l_end(E) + k2 * l_dir(D)
                    lo = loss(
                        distance=(feature_distance, label_distance),
                        end_points=(feature_end, label_end),
                        direction=(feature_direction, label_direction),
                    )
                    print("Loss on test set: %s" % loss)

            # Evaluate the model on the validation set
            # TODO

        if epoch % configs["train"]["checkpoint-interval"] == 0:
            torch.save(
                model.state_dict(),
                str(model_path / ("model_%s_%d.pth" % (opt.tag, epoch))),
            )
