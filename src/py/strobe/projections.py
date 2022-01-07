import torch

from .base import StrobeLayer


class Dropout(StrobeLayer):
    def __init__(
            self,
            dropout_probability: float,
            *shape: int,
            in_eval: bool = False):
        super(Dropout, self).__init__()
        self.p = dropout_probability
        self.shape = shape
        self.in_eval = in_eval

        self._mask = torch.ones(self.shape)

    def step(self):
        self._mask = torch.rand(self.shape) > self.p

    @property
    def mask(self):
        if self.training or self.in_eval:
            return self._mask
        else:
            return torch.ones(self.shape)

    def forward(self, x: torch.Tensor):
        return x * self.mask.to(x.device)


class Linear(torch.nn.Linear, StrobeLayer):
    def __init__(self, *shape: int, scale: float = 250.0):
        super(Linear, self).__init__(*shape, bias=False)
        self.scale = scale

    def forward(self, x: torch.Tensor):
        y = torch.nn.Linear.forward(self, x)
        return y


class SigmoidalWeights(StrobeLayer):
    def __init__(self, *shape: int, slope: float = 1.0, scale: float = 63.0):
        super(SigmoidalWeights, self).__init__()
        self.slope = slope
        self.scale = scale
        self._weight = torch.nn.Parameter(torch.empty((shape[1], shape[0])))
        self._weight.data.normal_()

    def forward(self, x: torch.Tensor):
        return torch.matmul(x, self.weight.T)

    @property
    def weight(self):
        return 2 * torch.sigmoid(self.slope * self._weight) - 1.0
