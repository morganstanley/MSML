from typing import List, Callable
from torchtyping import TensorType

import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: List[int],
        out_dim: int,
        activation: Callable = nn.ReLU(),
        final_activation: Callable = None,
    ):
        super().__init__()

        hidden_dims = hidden_dims[:]
        hidden_dims.append(out_dim)

        layers = [nn.Linear(in_dim, hidden_dims[0])]

        for i in range(len(hidden_dims) - 1):
            layers.append(activation)
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))

        if final_activation is not None:
            layers.append(final_activation)

        self.net = nn.Sequential(*layers)

    def forward(self, x: TensorType[..., 'in_dim']) -> TensorType[..., 'out_dim']:
        return self.net(x)
