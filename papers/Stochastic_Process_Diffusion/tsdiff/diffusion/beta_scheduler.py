from typing import Callable
import torch
import torch.nn as nn
from torch import Tensor

def get_beta_scheduler(name: str) -> Callable:
    if name == 'linear':
        return BetaLinear

def get_loss_weighting(name: str) -> Callable:
    if name == 'exponential':
        return exponential_loss_weighting

class BetaLinear(nn.Module):
    """
    Linear scheduling for beta.
    Input t is always from interval [0, 1].

    Args:
        start: Lower bound (float)
        end: Upper bound (float)
    """
    def __init__(self, start: float, end: float):
        super().__init__()
        self.start = start
        self.end = end

    def forward(self, t: Tensor) -> Tensor:
        return self.start * (1 - t) + self.end * t

    def integral(self, t: Tensor) -> Tensor:
        return 0.5 * (self.end - self.start) * t.square() + self.start * t


def exponential_loss_weighting(beta_fn, i):
    return 1 - torch.exp(-beta_fn.integral(i))
