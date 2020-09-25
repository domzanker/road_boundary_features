# from models import *
import os
import sys
import time
import datetime
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

from utils.dataset import RoadBoundaryDataset
import segmentation_models_pytorch as smp

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
    parser.add_argument(
        "--train_dataset",
        type=str,
        default="/home/Zanker/data/nuscenes_train/data/",
        help="path to training dataset",
    )
    parser.add_argument(
        "--test_dataset",
        type=str,
        default="/home/Zanker/data/nuscenes_test/data/",
        help="path to training dataset",
    )
    parser.add_argument(
        "--evaluation_interval",
        type=int,
        default=1,
        help="number of epochs between evaluations",
    )
    parser.add_argument("--gpu", type=int, default=0, help="gpu")

    opt = parser.parse_args()
    print(opt)

    device = torch.device(("cuda:%s" % opt.gpu) if torch.cuda.is_available() else "cpu")
    os.makedirs("~/output", exist_ok=True)
    os.makedirs("~/training/checkpoints", exist_ok=True)

    # Get dataloader
    dataset = RoadBoundaryDataset(path=Path(opt.train_dataset))

    test_dataset = RoadBoundaryDataset(path=Path(opt.test_dataset))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=1
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.cpu_workers,
        pin_memory=True,
        # collate_fn=dataset.collate_fn,
    )
    # build dual loss
    # distance loss for endpoints and distance map
    mse_loss = torch.nn.MSELoss(reduction="none")
    # cosine similarity for direction_map
    cosine_similarity = torch.nn.CosineSimilarity(dim=1)

    # Initiate model
    model = smp.Linknet(in_channels=4, encoder_depth=3, classes=4, activation="sigmoid")

    optimizer = torch.optim.Adam(model.parameters())

    tb_writer = SummaryWriter("tensorboard/road_boundary")

    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        start_time = time.time()
        for batch_i, (imgs, targets) in enumerate(dataloader):

            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            label_distance = targets[:, 0, :, :]
            label_end = targets[:, 1, :, :]
            label_direction = targets[:, 2:4, :, :]

            tb_writer.add_image("label_distance", label_distance, batches_done)
            tb_writer.add_image("label_direction", label_direction, batches_done)
            tb_writer.add_image("label_end", label_end, batches_done)

            outputs = model(imgs)
            feature_distance = outputs[:, 0, :, :]
            feature_end = outputs[:, 1, :, :]
            feature_direction = outputs[:, 2:4, :, :]

            tb_writer.add_image("feature_direction", feature_direction, batches_done)
            tb_writer.add_image("feature_distance", feature_distance, batches_done)
            tb_writer.add_image("feature_end", feature_end, batches_done)

            # L(S,E,D) = l_det(S) + k1 * l_end(E) + k2 * l_dir(D)
            k1 = 10
            k2 = 10
            distance_loss = mse_loss(feature_distance, label_distance)
            end_loss = mse_loss(feature_end, label_end)
            direction_loss_0 = cosine_similarity(feature_direction, label_direction)

            loss = (
                direction_loss_0
                # + direction_loss_1[:, :, :]
                + k1 * distance_loss
                + k2 * end_loss
            ).sum()

            tb_writer.add_scalar("loss", loss, batches_done)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (
                epoch,
                num_epochs,
                batch_i,
                len(dataloader),
            )

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(
                seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1)
            )
            log_str += f"\n---- ETA {time_left}"

            print(log_str)

        if epoch % opt.evaluation_interval == 0:
            model.eval()
            print("\n---- Evaluating Model ----")
            with torch.no_grad():
                # FIXME
                data, target = test_loader[0]
                label_distance = targets[:, 0, :, :]
                label_end = targets[:, 1, :, :]
                label_direction = targets[:, 2:4, :, :]

                outputs = model(imgs)
                feature_distance = outputs[:, 0, :, :]
                feature_end = outputs[:, 1, :, :]
                feature_direction = outputs[:, 2:4, :, :]

                # L(S,E,D) = l_det(S) + k1 * l_end(E) + k2 * l_dir(D)
                k1 = 10
                k2 = 10
                distance_loss = mse_loss(feature_distance, label_distance)
                end_loss = mse_loss(feature_end, label_end)
                direction_loss_0 = cosine_similarity(feature_direction, label_direction)

                loss = (
                    direction_loss_0
                    # + direction_loss_1[:, :, :]
                    + k1 * distance_loss
                    + k2 * end_loss
                ).sum()
                print("Loss on test set: %s" % loss)

            # Evaluate the model on the validation set
            # TODO

        if epoch % opt.checkpoint_interval == 0:
            torch.save(model.state_dict(), "checkpoints/linknet_%d.pth" % epoch)
