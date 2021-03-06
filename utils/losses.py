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
        self.cosine_similarity = CosineSimilarity(dim=-1)
        self.reduction = reduction

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


class MultiTaskUncertaintyLoss(Module):
    """
    implements https://arxiv.org/pdf/1805.06334.pdf
    """

    def __init__(
        self,
        distance_loss,
        end_loss,
        direction_loss,
    ):
        super(MultiTaskUncertaintyLoss, self).__init__()
        # factor = log(sigma)
        self.log_var = torch.nn.Parameter(torch.zeros(3), requires_grad=True)

        self.distance_loss = loss_func(distance_loss["loss"], **distance_loss["args"])
        self.end_loss = loss_func(end_loss["loss"], **end_loss["args"])
        self.direction_loss = loss_func(
            direction_loss["loss"], **direction_loss["args"]
        )

    def forward(self, x, y):
        distance_loss = self.distance_loss(x[:, :1, :, :], y[:, :1, :, :])
        end_loss = self.end_loss(x[:, 1:2, :, :], y[:, 1:2, :, :])
        direction_loss = self.direction_loss(x[:, 2:4, :, :], y[:, 2:4, :, :])

        # s = log(sigma**2)
        exp_fac = torch.exp(-self.log_var)
        regulization = torch.log(1 + torch.exp(self.log_var))
        total_loss = (
            exp_fac[0] * distance_loss
            + regulization[0]
            + exp_fac[1] * end_loss
            + regulization[1]
            + exp_fac[2] * direction_loss
            + regulization[2]
        )
        return {
            "total_loss": total_loss,
            "distance_loss": distance_loss.detach(),
            "end_loss": end_loss.detach(),
            "direction_loss": direction_loss.detach(),
            "loss_variance": torch.exp(self.log_var.detach()),
        }
