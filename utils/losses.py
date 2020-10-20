from torch.nn import ModuleDict, ModuleList, Module
from torch.nn import MSELoss, CrossEntropyLoss, BCELoss, NLLLoss
from typing import Optional, Callable, Union, List


def loss_func(loss: str, reduction: str = "mean", **kwargs):
    return ModuleDict(
        {
            "mse": MSELoss(reduction=reduction),
            "cross_entropy": CrossEntropyLoss(reduction=reduction, **kwargs),
            "bce": BCELoss(reduction=reduction, **kwargs),
            "nll": NLLLoss(reduction=reduction, **kwargs),
        }
    )[loss]


class MultiObjectiveLoss(Module):
    def __init__(self, losses: List[str], factors: List[float]):
        super(MultiObjectiveLoss, self).__init()

        self.factors = factors
        self.losses = ModuleList([loss_func(f) for f in losses])

    def forward(self, x):
        raise NotImplementedError
