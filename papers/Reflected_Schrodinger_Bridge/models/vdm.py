import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Iterable

from . import utils
from .layersv2 import *
from math import pi


class ImageFourierFeatures(nn.Module):
    """Fourier features used in VDMs. Meant for usage in image space"""
    def __init__(self, start=6, end=8):
        super().__init__()
        self.register_buffer("freqs", 2 ** torch.arange(start, end))

    def forward(self, x):
        freqs = (self.freqs * 2 * pi).repeat(x.shape[1])
        x_inp = x
        x = x.repeat_interleave(len(self.freqs), dim=1)
        
        x = freqs[None, :, None, None] * x
        return torch.cat([x_inp, x.sin(), x.cos()], dim=1)

    def extra_repr(self):
        return f"ImageFourierFeatures({self.freqs.detach().cpu().numpy()})"


def get_timestep_embedding(timesteps, embedding_dim, dtype=torch.float32):
    assert len(timesteps.shape) == 1
    timesteps *= 1000.

    half_dim = embedding_dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = (torch.arange(half_dim, dtype=dtype, device=timesteps.device) * -emb).exp()
    emb = timesteps.to(dtype)[:, None] * emb[None, :]
    emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1))
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


class ResNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim, dropout=0.1):
        super().__init__()
        self.conv1 = Conv2d(in_ch, out_ch, 3)
        self.conv2 = Conv2d(out_ch, out_ch, 3, init_weight=0)

        self.norm1 = nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6)
        self.norm2 = nn.GroupNorm(num_groups=min(out_ch // 4, 32), num_channels=out_ch, eps=1e-6)
        if in_ch != out_ch:
            self.skip = Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()
        
        self.cond_map = Linear(cond_dim, out_ch, bias=False, init_weight=0)

        self.dropout = dropout

    def forward(self, x, cond):
        h = x
        # activation for the last block
        h = F.silu(self.norm1(x))
        h = self.conv1(h)

        # add in conditioning
        h += self.cond_map(cond)[:, :, None, None]

        h = F.silu(self.norm2(h))
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h)
        x = h + self.skip(x)

        return x


class AttnBlock(nn.Module):
    """Self-attention residual block."""
    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(num_groups=min(channels // 4, 32), num_channels=channels, eps=1e-6)
        self.qkv = Conv2d(channels, 3 * channels, 1)
        self.proj_out = Conv2d(channels, channels, 1, init_weight=0)

    def forward(self, x):
        q, k, v = self.qkv(self.norm(x)).reshape(x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1).unbind(2)
        w = AttentionOp.apply(q, k)
        a = torch.einsum('nqk,nck->ncq', w, v)
        x = self.proj_out(a.reshape(*x.shape)).add_(x)

        return x


@utils.register_model(name='vdm')
class VDM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_blocks = config.model.num_blocks
        self.channels = channels = config.model.channels
        self.attention = config.model.attention
        dropout = config.model.dropout
        input_ch = config.data.num_channels

        self.sigma_min = config.sde.sigma_min
        self.sigma_max = config.sde.sigma_max
        self.scale_by_sigma = config.model.scale_by_sigma

        self.cond_map = nn.Sequential(
            Linear(channels, 4 * channels),
            nn.SiLU(),
            Linear(4 * channels, 4 * channels),
        )

        if config.model.image_fourier:
            self.image_fourier = ImageFourierFeatures(start=config.model.image_fourier_start, end=config.model.image_fourier_end)
            freqs = config.model.image_fourier_end - config.model.image_fourier_start
            fourier_channels = (2 * freqs + 1) * input_ch
        else:
            self.image_fourier = nn.Identity()
            fourier_channels = input_ch

        self.conv_in = Conv2d(fourier_channels, channels, 3)

        
        # "downsampling"
        enc = []
        for _ in range(self.num_blocks):
            enc.append(ResNetBlock(channels, channels, 4 * channels, dropout=dropout))
            if self.attention:
                enc.append(AttnBlock(channels))
        self.enc = nn.ModuleList(enc)

        # middle
        self.mid1 = ResNetBlock(channels, channels, 4 * channels, dropout=dropout)
        self.midattn = AttnBlock(channels)
        self.mid2 = ResNetBlock(channels, channels, 4 * channels, dropout=dropout)

        # "upsampling"
        dec = []
        for _ in range(self.num_blocks + 1):
            dec.append(ResNetBlock(2 * channels, channels, 4 * channels, dropout=dropout))
            if self.attention:
                dec.append(AttnBlock(channels))
        self.dec = nn.ModuleList(dec)

        #  output
        self.out = nn.Sequential(
            nn.GroupNorm(num_groups=min(channels // 4, 32), num_channels=channels, eps=1e-6),
            nn.SiLU(),
            Conv2d(channels, input_ch, 3, init_weight=0)
        )

    def forward(self, x, cond, class_labels=None):
        sigma_inp = cond
        t = (cond - self.sigma_min) / (self.sigma_max - self.sigma_min)
        temb = get_timestep_embedding(t, self.channels)
        cond = self.cond_map(temb)

        x = self.image_fourier(x)

        outputs = []

        x = self.conv_in(x)
        outputs.append(x)

        for i in range(self.num_blocks):
            if self.attention:
                x = self.enc[2 * i](x, cond)
                x = self.enc[2 * i + 1](x)
            else:
                x = self.enc[i](x, cond)
            outputs.append(x)
        
        x = self.mid1(x, cond)
        x = self.midattn(x)
        x = self.mid2(x, cond)

        for i in range(self.num_blocks + 1):   
            
            if self.attention:
                x = self.dec[2 * i](torch.cat((x, outputs.pop()), dim=1), cond)
                x = self.dec[2 * i + 1](x)
            else:
                x = self.dec[i](torch.cat((x, outputs.pop()), dim=1), cond)
        
        if len(outputs) > 0:
            raise ValueError("Something went wrong with the blocks")

        out = self.out(x)

        if self.scale_by_sigma:
            out = out / sigma_inp[:, None, None, None]
        
        return out
