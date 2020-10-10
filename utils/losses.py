from torch.nn import ModuleDict
from torch.nn import MSELoss, CrossEntropyLoss
from typing import Optional, Callable, Union


def loss_func(loss: str, reduction: str = "mean", **kwargs):
    return ModuleDict(
        {
            "mse": MSELoss(reduction=reduction),
            "cross_entropy": CrossEntropyLoss(reduction=reduction, **kwargs),
        }
    )[loss]
