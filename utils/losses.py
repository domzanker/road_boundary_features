import torch

from typing import Optional, Callable, Union

regression_losses = torch.nn.ModuleDict(
    {
        "mse_mean": torch.nn.MSELoss(reduction="mean"),
        "mse_sum": torch.nn.MSELoss(reduction="sum"),
        "cosine": torch.nn.CosineSimilarity(),
        "bce": torch.nn.BCELoss(),
    }
)


class CombinedLoss:
    def __init__(
        self,
        regression_loss: Union[str, Callable] = torch.nn.MSELoss(reduction="none"),
        distance_loss: Union[str, Callable] = torch.nn.CosineSimilarity(dim=1),
        *,
        k1: float = 10,
        k2: float = 10
    ):
        self.regression_loss = regression_loss
        self.distance_loss = distance_loss
        self.k1 = k1
        self.k2 = k2

    def __call__(self, predictions, targets):
        distance, end_points, direction = predictions
        distance_t, end_points_t, direction_t = targets

        l_DirMap = self.distance_loss(direction, direction_t)
        l_DistMap = self.regression_loss(distance, distance_t)
        l_EndMap = self.regression_loss(end_points, end_points_t)
        loss = l_DirMap + self.k1 * l_DistMap + self.k2 * l_EndMap

        return (
            loss.sum(),
            {"l_DirMap": l_DirMap, "l_DistMap": l_DistMap, "l_EndMap": l_EndMap},
        )
