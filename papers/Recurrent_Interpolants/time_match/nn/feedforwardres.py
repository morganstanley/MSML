import math
from typing import Optional

import numpy as np
import torch
from torch import Tensor, nn


class ResNetFC(nn.Module):
    def __init__(self, data_dim, hidden_dim, num_res_blocks):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.map=nn.Linear(data_dim, hidden_dim)
        self.res_blocks = nn.ModuleList(
            [self.build_res_block() for _ in range(num_res_blocks)])

    def build_linear(self, in_features, out_features):
        linear = nn.Linear(in_features, out_features)
        return linear

    def build_res_block(self):
        hid = self.hidden_dim
        layers = []
        widths =[hid]*4
        for i in range(len(widths) - 1):
            layers.append(self.build_linear(widths[i], widths[i + 1]))
            layers.append(SiLU())
        return nn.Sequential(*layers)

    def forward(self, x):
        h=self.map(x)
        for res_block in self.res_blocks:
            h = (h + res_block(h)) / np.sqrt(2)
        return h

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

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

class FeedForwardResNet(torch.nn.Module):
    def __init__(
        self,
        in_dim=2,
        hidden_dim=256,
        time_embed_dim=128,
        num_layers=4,
        out_dim=2,
        zero_out_last_layer=True,
    ):
        super().__init__()

        self.time_embed_dim = time_embed_dim
        self.zero_out_last_layer = zero_out_last_layer
        hid = hidden_dim

        self.t_module = nn.Sequential(
            nn.Linear(self.time_embed_dim, hid),
            SiLU(),
            nn.Linear(hid, hid),
        )

        self.x_module = ResNetFC(in_dim, hidden_dim, num_res_blocks=num_layers)

        self.out_module = nn.Sequential(
            nn.Linear(hid,hid),
            SiLU(),
            nn.Linear(hid, out_dim),
        )
        if zero_out_last_layer:
            self.out_module[-1] = zero_module(self.out_module[-1])

    def forward(
        self,
        inputs: Tensor,
        time: Tensor,
        cond: Optional[Tensor] = None,
    ) -> Tensor:
        t = time.squeeze(-1)  # (B,)
        assert len(t.shape) == 1
        t_emb = timestep_embedding(t, self.time_embed_dim)
        t_out = self.t_module(t_emb)
        x_out = self.x_module(inputs)
        out   = self.out_module(x_out+t_out)

        return out
