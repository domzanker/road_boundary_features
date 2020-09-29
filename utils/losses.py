import torch


class CombinedLoss:
    def __init__(
        self,
        regression_loss=torch.nn.MSELoss(reduction="none"),
        distance_loss=torch.nn.CosineSimilarity(dim=1),
        *,
        k1=10,
        k2=10
    ):
        self.regression_loss = regression_loss
        self.distance_loss = distance_loss
        self.k1 = k1
        self.k2 = k2

    def __call__(self, distance, end_points, direction):
        loss = (
            self.distance_loss(direction[0], direction[1])
            + self.k1 * self.regression_loss(distance[0], distance[1])
            + self.k2 * self.regression_loss(end_points[0], end_points[1])
        )
        return loss.sum()
