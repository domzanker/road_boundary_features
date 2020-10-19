import h5py
import pickle
import torch
import numpy as np
from pathlib import Path

from sys import argv
import time


def convert_directory(path="data/nuscenes-complete-15x10"):
    path = Path(path)

    for i, file in enumerate(path.iterdir(), 1):
        if file.suffix == ".pkl":
            print("File #{}".format(i), end="\r")
            with file.open(mode="rb") as f:
                d = pickle.load(f)

            outfile = file.with_suffix(".h5")
            with h5py.File(str(outfile), "w", swmr=True) as h:
                for key, value in d.items():

                    if key in ["bev", "lidar_intensity", "ground_truth"]:
                        continue
                    elif key == "rgb":
                        value = torch.from_numpy(value.astype(np.uint8))
                        assert torch.isfinite(value).all()
                    elif key == "road_direction_map":
                        value = value / (
                            np.abs(np.linalg.norm(value, axis=2, keepdims=True)) + 1e-12
                        )
                        value = torch.from_numpy(value.astype(np.float32))
                        assert torch.isfinite(value).all()
                    else:
                        value = value - value.min()
                        value = value / (value.max() + 1e-12)
                        value = torch.from_numpy(
                            np.nan_to_num(value.astype(np.float16))
                        )
                        assert torch.isfinite(value).all()

                    h.create_dataset(key, data=value)


def read_dir_h5(path):
    path = Path(path)
    i = 0
    for file in path.iterdir():
        if file.suffix == ".h5":
            h = h5py.File(file, mode="r", swmr=True)
            d = h["rgb"][()]  # noqa
            i = i + 1
            print(d.__repr__)
    print("Loaded %s files" % i)


def read_dir_pkl(path):
    path = Path(path)
    i = 0
    for file in path.iterdir():
        if file.suffix == ".pkl":
            with open(file, "rb") as f:
                h = pickle.load(f)
            d = h["rgb"]  # noqa
            i = i + 1
    print("Loaded %s files" % i)


def create_dummy_scene(dict):
    for i in range(50):
        with open(("data/dummy_data/sample_%s.pkl" % i), "wb+") as f:
            pickle.dump(dict, f)


if __name__ == "__main__":
    """
    with open("data/scene_849_sample_9_data.pkl", "rb") as f:
        d = pickle.load(f)
    create_dummy_scene(d)

    t = time.time()
    convert_directory("data/dummy_data")
    print(time.time() - t)

    t = time.time()
    read_dir_h5("data/dummy_data")
    print(time.time() - t)

    t = time.time()
    read_dir_pkl("data/dummy_data")
    print(time.time() - t)
    """

    p = argv[1]
    t = time.time()
    convert_directory(p)
    # read_dir_h5(p)
    print("finished after {}s".format(time.time() - t))
