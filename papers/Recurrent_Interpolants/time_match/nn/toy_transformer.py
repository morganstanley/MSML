import math
from typing import Optional

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


class ToyTransformer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        out_dim: int,
    ):
        super().__init__()
        assert out_dim % in_dim == 0

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        self.t_module = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.x_module = nn.Linear(1, hidden_dim)

        self.x_pos_module = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

        self.pos = nn.Parameter(torch.randn(1, self.in_dim, self.hidden_dim))

        self.out_module = nn.Linear(hidden_dim, out_dim // in_dim)

    def time_embed(self, t: Tensor) -> Tensor:
        t = t.squeeze(-1)
        t_emb = timestep_embedding(t, self.hidden_dim)
        t_emb = self.t_module(t_emb)
        t_emb = t_emb.unsqueeze(1).repeat_interleave(self.in_dim, dim=1)
        return t_emb

    def forward(
        self,
        inputs: Tensor,
        time: Tensor,
        cond: Optional[Tensor] = None,
    ) -> Tensor:
        t = self.time_embed(time)

        pos = torch.arange(0, self.in_dim).long().to(inputs)
        pos = timestep_embedding(pos, self.hidden_dim, max_period=self.in_dim)
        pos = pos.unsqueeze(0).repeat_interleave(inputs.shape[0], dim=0)

        # pos = self.pos.repeat_interleave(inputs.shape[0], dim=0)

        x = inputs.unsqueeze(-1)
        x = self.x_module(x)

        x = self.x_pos_module(x + pos)

        out = self.transformer(x + t)
        out = self.out_module(x).flatten(1, 2)

        return out
