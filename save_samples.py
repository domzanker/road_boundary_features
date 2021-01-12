import torch
import argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt

from pathlib import Path
from utils.dataset import RoadBoundaryDataset
from utils.image_transforms import apply_colormap, angle_map


def float2byte(array):
    array = array * 255
    array = array.astype(np.uint8)
    array = np.transpose(array, [1, 2, 0])
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    return array


def update_axis(sample_inx, ax):
    global handles
    sample_data, sample_targets = dataset[sample_inx]

    rgb = sample_data[:3, :, :].numpy()
    lidar_height = sample_data[3:4, :, :]

    distance_map = sample_targets[:1, :, :]
    distance_map = apply_colormap(distance_map)
    distance_map = distance_map.numpy()

    end_points = sample_targets[1:2, :, :]
    end_points = apply_colormap(end_points)
    end_points = end_points.numpy()

    direction_map = sample_targets[2:4, :, :]
    direction_map = apply_colormap(angle_map(direction_map[None, :, :, :]))
    direction_map = direction_map.numpy().squeeze()

    lidar_height -= lidar_height.min()
    lidar_height /= lidar_height.max()
    lidar_height = apply_colormap(lidar_height)
    lidar_height = lidar_height.numpy()

    # cv2.imwrite(str(args.outdir / "height.png"), float2byte(lidar_height))
    cv2.imwrite(str(args.outdir / "rgb.png"), float2byte(rgb))
    cv2.imwrite(str(args.outdir / "lidar.png"), float2byte(lidar_height))

    cv2.imwrite(str(args.outdir / "distance_map.png"), float2byte(distance_map))
    cv2.imwrite(str(args.outdir / "endpoints.png"), float2byte(end_points))
    cv2.imwrite(str(args.outdir / "direction_map.png"), float2byte(direction_map))

    handles[0][0].set_data(float2byte(rgb))
    handles[0][1].set_data(float2byte(lidar_height))

    handles[1][0].set_data(float2byte(distance_map))
    handles[1][1].set_data(float2byte(end_points))
    handles[1][2].set_data(float2byte(direction_map))
    plt.title(f"{dataset.index[sample_inx]}")
    plt.draw()


def find_next_index():
    global dataset, scene

    search = True
    scene += 1
    while search:
        try:
            sample_inx = dataset.index.index(
                dataset.path / f"Town03_scene_{scene}_sample_1_data.h5"
            )
            search = False
            return sample_inx
        except ValueError:
            scene += 1
            if scene >= len(dataset.index):
                return sample_inx


def on_click(event):
    global sample_inx, ax, scene

    if event.key in ["l", "right", "enter", "space"]:
        sample_inx = find_next_index()
        update_axis(sample_inx, ax)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--outdir", type=Path, default="/tmp/samples/")
    parser.add_argument("--sample", type=int, default=10)

    args = parser.parse_args()
    args.outdir.mkdir(exist_ok=True, parents=True)

    # 748, 498
    dataset = RoadBoundaryDataset(
        args.dataset, image_size=[498, 498], augmentation=None
    )

    fig, ax = plt.subplots(2, 3)

    scene = args.sample

    sample_inx = find_next_index()

    sample_data, sample_targets = dataset[sample_inx]

    rgb = sample_data[:3, :, :].numpy()
    lidar_height = sample_data[3:4, :, :]

    distance_map = sample_targets[:1, :, :]
    distance_map = apply_colormap(distance_map)
    distance_map = distance_map.numpy()

    end_points = sample_targets[1:2, :, :]
    end_points = apply_colormap(end_points)
    end_points = end_points.numpy()

    direction_map = sample_targets[2:4, :, :]
    direction_map = apply_colormap(angle_map(direction_map[None, :, :, :]))
    direction_map = direction_map.numpy().squeeze()

    lidar_height -= lidar_height.min()
    lidar_height /= lidar_height.max()
    lidar_height = apply_colormap(lidar_height)
    lidar_height = lidar_height.numpy()

    # cv2.imwrite(str(args.outdir / "height.png"), float2byte(lidar_height))
    cv2.imwrite(str(args.outdir / "rgb.png"), float2byte(rgb))
    cv2.imwrite(str(args.outdir / "lidar.png"), float2byte(lidar_height))

    cv2.imwrite(str(args.outdir / "distance_map.png"), float2byte(distance_map))
    cv2.imwrite(str(args.outdir / "endpoints.png"), float2byte(end_points))
    cv2.imwrite(str(args.outdir / "direction_map.png"), float2byte(direction_map))

    handles = []
    handles.append([])
    handles.append([])

    handles[0].append(ax[0][0].imshow(float2byte(rgb)))
    handles[0].append(ax[0][1].imshow(float2byte(lidar_height)))

    handles[1].append(ax[1][0].imshow(float2byte(distance_map)))
    handles[1].append(ax[1][1].imshow(float2byte(end_points)))
    handles[1].append(ax[1][2].imshow(float2byte(direction_map)))

    cid = fig.canvas.mpl_connect("key_press_event", on_click)
    plt.show()
