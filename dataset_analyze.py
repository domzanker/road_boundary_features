import pickle
import numpy as np
import h5py
import bz2
import timeit
import torch
import cv2


def write_images():
    global new_copy
    for key, item in new_copy.items():
        cv2.imwrite(("data/%s.png" % key), item)


def write_pickle():
    global new_copy
    with open("data/new_sample_f32.pkl", "wb+") as f:
        pickle.dump(new_copy, f)


def write_bz2():
    global new_copy
    with bz2.BZ2File("data/new_sample_f32.pbz2", "w") as f:
        pickle.dump(new_copy, f)


def write_h5():
    global new_copy
    with h5py.File("data/new_sample_f32.h5", "w") as f:
        for key, value in new_copy.items():
            f.create_dataset(key, data=value)


def read_raw_pickle():
    with open("data/scene_849_sample_9_data.pkl", "rb") as f:
        d = pickle.load(f)
    t = d["rgb"]
    t = t.sum()
    return t


def read_pickle():
    with open("data/new_sample_f32.pkl", "rb") as f:
        d = pickle.load(f)
    t = d["rgb"]
    t = t.sum()
    return t


def read_bz2():
    with bz2.BZ2File("data/new_sample_f32.pbz2", "r") as f:
        d = pickle.load(f)
    t = d["rgb"]
    t = t.sum()
    return t


def read_h5():
    d = h5py.File("data/new_sample_f32.h5", "r")
    t = d["rgb"][()]
    t = torch.Tensor(t).sum()
    return t


if __name__ == "__main__":

    with open("data/scene_849_sample_9_data_float32.pkl", "rb") as f:
        d = pickle.load(f)

    new_copy = {}
    for key, value in d.items():
        if key != "bev" and key != "lidar_intensity" and key != "ground_truth":
            if key == "rgb":
                new_copy[key] = torch.from_numpy(value.astype(np.uint8))
            else:
                new_copy[key] = torch.from_numpy(value.astype(np.float16))
            if key == "road_direction_map":
                print(value.shape)
                v = value / (
                    np.abs(np.linalg.norm(value, axis=2, keepdims=True)) + 1e-12
                )
                print(
                    np.testing.assert_allclose(
                        np.linalg.norm(v, axis=2, keepdims=True), 1
                    )
                )
                print(value.min())
                print(value.max())

        with open(("data/%s" % key), "wb+") as f:
            pickle.dump({key: value}, f)

    print(timeit.timeit(write_pickle, number=5))
    print(timeit.timeit(write_bz2, number=5))
    print(timeit.timeit(write_h5, number=5))

    print(timeit.timeit(read_raw_pickle, number=5))
    print(timeit.timeit(read_pickle, number=5))
    print(timeit.timeit(read_bz2, number=5))
    print(timeit.timeit(read_h5, number=5))


"""
sample_collection = {
    "bev": bev,
    "rgb": self.rgb,
    "lidar_height": self.lidar[:, :, 2],
    "lidar_intensity": self.lidar[:, :, 1],
    "ground_truth": self.ground_truth,
    "road_direction_map": self.direction_map,
    "inverse_distance_map": self.distance_map,
}
"""
