from torch.nn import ModuleDict, ModuleList, Module
from torch.nn import MSELoss, CrossEntropyLoss, BCELoss, NLLLoss, CosineSimilarity
import torch.nn
from typing import Optional, Callable, Union, List, Dict, Tuple
from math import pi


def loss_func(loss: str, reduction: str = "mean", **kwargs):
    return ModuleDict(
        {
            "mse": MSELoss(reduction=reduction),
            "cross_entropy": CrossEntropyLoss(reduction=reduction, **kwargs),
            "bce": BCELoss(reduction=reduction, **kwargs),
            "nll": NLLLoss(reduction=reduction, **kwargs),
            "cosine_similarity": CosineSimilarityLoss(reduction=reduction, **kwargs),
        }
    )[loss]


class CosineSimilarityLoss(Module):
    def __init__(self, reduction, *args, **kwargs):
        super(CosineSimilarityLoss, self).__init__()
        self.cosine_similarity = CosineSimilarity(dim=1)
        self.reduction = reduction
        self.pi = torch.acos(torch.zeros(1)).item() * 2

    def forward(self, x, y):

        # angular_distance = torch.acos(self.cosine_similarity(x, y)) / self.pi
        # dist = 1 - angular_distance
        dist = 1 - self.cosine_similarity(x, y)
        if self.reduction == "none":
            return dist
        elif self.reduction == "sum":
            return dist.sum()
        elif self.reduction == "mean":
            return dist.mean()
        else:
            raise NotImplementedError


class MultiFeaturesLoss(Module):
    def __init__(self, distance_loss, end_loss, direction_loss):
        super(MultiFeaturesLoss, self).__init__()

        self.factors = [
            distance_loss["factor"],
            end_loss["factor"],
            direction_loss["factor"],
        ]
        self.distance_loss = loss_func(distance_loss["loss"], **distance_loss["args"])
        self.end_loss = loss_func(end_loss["loss"], **end_loss["args"])
        self.direction_loss = loss_func(
            direction_loss["loss"], **direction_loss["args"]
        )

    def forward(self, x, y):
        distance_loss = self.distance_loss(x[:, :1, :, :], y[:, :1, :, :])
        end_loss = self.end_loss(x[:, 1:2, :, :], y[:, 1:2, :, :])
        direction_loss = self.direction_loss(x[:, 2:4, :, :], y[:, 2:4, :, :])

        total_loss = (
            self.factors[0] * distance_loss
            + self.factors[1] * end_loss
            + self.factors[2] * direction_loss
        )
        return {
            "total_loss": total_loss,
            "distance_loss": distance_loss,
            "end_loss": end_loss,
            "direction_loss": direction_loss,
        }


class MultiTaskUncertaintyLoss(MultiFeaturesLoss):
    def __init__(
        self,
        distance_loss,
        end_loss,
        direction_loss,
    ):
        super(MultiTaskUncertaintyLoss, self).__init__(
            distance_loss, end_loss, direction_loss
        )
        self.factors = torch.ones([3])

    def forward(self, x, y):
        distance_loss = self.distance_loss(x[:, :1, :, :], y[:, :1, :, :])
        end_loss = self.end_loss(x[:, 1:2, :, :], y[:, 1:2, :, :])
        direction_loss = self.direction_loss(x[:, 2:4, :, :], y[:, 2:4, :, :])

        total_loss = 0

        exp_facts = torch.exp(-self.factors)
        total_loss = (
            exp_facts[0] * distance_loss
            + self.factors[0]
            + exp_facts[1] * end_loss
            + self.factors[1]
            + exp_facts[2] * direction_loss
            + self.factors[2]
        )
        return {
            "total_loss": total_loss,
            "distance_loss": distance_loss.detach(),
            "end_loss": end_loss.detach(),
            "direction_loss": direction_loss.detach(),
            "factors": self.factors.detach(),
        }
