import numpy as np
import pandas as pd
import argparse
import json
import comet_ml
import csv

import seaborn as sbn

import matplotlib.pyplot as plt

metrics = {
    "val_distance_loss": "distance",
    "val_end_loss": "endpoints",
    "val_direction_loss": "direction",
}


def load_metric(experiment, metric_name):
    global metrics
    metric_dicts = experiment.get_metrics(metric_name)

    epochs = []
    steps = []
    values = []
    for dict in metric_dicts:
        value = float(dict["metricValue"])
        epoch = int(dict["epoch"])
        if epoch in epochs:
            epoch = epoch + 0.5
        # epoch[-1] = 0.5(epoch[-1] + value)
        epochs.append(epoch)
        steps.append(int(dict["step"]))
        values.append(value)

    return pd.DataFrame.from_dict(
        {
            "experiment": experiment.name,
            "loss-type": metrics[metric_name],
            "step": steps,
            "epoch": epochs,
            "value": values,
        }
    )


def metric_dicts_to_dataframe(dicts):
    comb_dict = {}
    lengths = [len(d["epochs"]) for d in dicts]
    length = max(lengths)
    for metric in dicts:
        pass
        if len(metric["epochs"]) == length:
            comb_dict["epoch"] = metric["epochs"]
        comb_dict["value"] = metric["values"]
        comb_dict["metric"] = metric["name"]
        comb_dict["experiment"] = metric["experiment"]

    return pd.DataFrame.from_dict(comb_dict)


def load_csv_logs(csv_file):
    dataframes = []
    for metric_name, file in csv_file.items():  # zip(metrics, csv_file):
        epochs = []
        steps = []
        values = []
        epoch = 0.0
        with open(file) as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                epochs.append(epoch)
                steps.append(float(row[1]))
                values.append(float(row[2]))
                epoch += 0.5

        dataframes.append(
            pd.DataFrame.from_dict(
                {
                    "experiment": "3stages-lyft",
                    "loss-type": metric_name,
                    "step": steps,
                    "epoch": epochs,
                    "value": values,
                }
            )
        )
    return dataframes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--range", type=int, default=30)

    args = parser.parse_args()

    comet_api = comet_ml.api.API()

    base_exp = ("759df0806f10447b8291fddffa1903c7",)

    experiments = {
        "depth": {
            "key_list": [
                base_exp,
                "ec8c683bf0f04a5cb0d84dd504592c26",
                "0a8182a60b404624b97bc49b415eb6f0",
                # "4e43e36dac194858ab2d921c6f076a38",
                "38c50728e13d4c9fb9067c97fad8948b",
            ],
            "out": "/home/dominic/repos/master-thesis/Graphics/plots/depth_experiments.svg",
            "csv_files": {
                "endpoints": "data/thesis_trainings/3stages_lyft/run-3stages_lyft_version_2-tag-val_end_loss.csv",
                "distance": "data/thesis_trainings/3stages_lyft/run-3stages_lyft_version_2-tag-val_distance_loss.csv",
                "direction": "data/thesis_trainings/3stages_lyft/run-3stages_lyft_version_2-tag-val_direction_loss.csv",
            },
        },
        "upsampling": {
            "key_list": [base_exp, "023cd3038bbc434a89df13b9e6a73761"],
            "out": "/home/dominic/repos/master-thesis/Graphics/plots/upsampling_experiments.svg",
            "csv_files": None,
        },
        "loss": {
            "key_list": [
                base_exp,
                "38c50728e13d4c9fb9067c97fad8948b",  # 4stages lyft
                "9e8b8c73f51d4283bd70872912d9807a",
                "264f9dcf41164b8cbdd5a17d54d762d0",
            ],
            "out": "/home/dominic/repos/master-thesis/Graphics/plots/loss_experiments.svg",
            "csv_files": None,
        },
        "datasets": {
            "key_list": [
                base_exp,
                "38c50728e13d4c9fb9067c97fad8948b",
                "c66455d1fb4846f986a703479dfd3492",
            ],
            "out": "/home/dominic/repos/master-thesis/Graphics/plots/dataset_experiments.svg",
            "csv_files": None,
        },
    }

    # upsampling experiments

    # depth experiments
    # out = "/home/dominic/repos/master-thesis/Graphics/plots/depth_experiments.eps"

    config = experiments[args.experiment]
    dataframes = []
    # import tensorboard logs
    if config["csv_files"] is not None:
        dataframes = load_csv_logs(config["csv_files"])

    for key in config["key_list"]:
        experiment = comet_api.get_experiment(
            workspace="domzanker",
            project_name="road-boundary-features",
            experiment=key,
        )
        if experiment is None:
            continue

        metrics_collection = []
        for i, metric in enumerate(metrics):

            d = load_metric(experiment, metric)
            metrics_collection.append(d)
            dataframes.append(d)
            # create a DataFrame for every metric

    dataframe = pd.concat(dataframes)
    # filer datagrame for epochs <= 25
    dataframe = dataframe.sort_index()
    dataframe = dataframe[dataframe["epoch"] <= args.range]  # .truncate(after=60)
    # dataframe["value"] = dataframe["value"].ewm(span=5).mean()
    print(dataframe)
    # dataframe.rolling(3).mean()
    # now render
    # sbn.relplot(
    f = sbn.relplot(
        data=dataframe,
        kind="line",
        x="epoch",
        y="value",
        # col="name",
        hue="experiment",
        legend="brief",
        style="loss-type",
        linewidth=1.2,
        ci=None,
        aspect=1.5,
    )
    legend = f._legend
    legend.set_bbox_to_anchor([1, 0.5])  # coordinates of lower left of bbox
    legend._loc = 5
    plt.rcParams["svg.fonttype"] = "none"

    plt.savefig(config["out"], bbox_inches="tight", pad_inches=0.0)

    # get assets
    # plt.show()
