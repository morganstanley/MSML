import math
from typing import Callable, Optional

import torch
from torch import Tensor, nn


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class FeedForward(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        out_dim: int,
        activation: Callable = nn.SiLU(),
        final_activation: Callable = None,
        time_embed_dim: int = 128,
    ):
        super().__init__()
        self.time_embed_dim = time_embed_dim

        hidden_dims = [hidden_dim] * num_layers
        hidden_dims.append(out_dim)

        self.t_module = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.input_enc = nn.Linear(in_dim, hidden_dims[0])

        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.append(activation)
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))

        if final_activation is not None:
            layers.append(final_activation)

        self.net = nn.Sequential(*layers)

    def forward(
        self,
        inputs: Tensor,
        time: Tensor,
        cond: Optional[Tensor] = None,
    ) -> Tensor:
        assert time.min() >= 0 and time.max() <= 1
        t_emb = timestep_embedding(time, self.time_embed_dim)
        t_out = self.t_module(t_emb)
        x = self.input_enc(inputs)
        return self.net(x + t_out)


