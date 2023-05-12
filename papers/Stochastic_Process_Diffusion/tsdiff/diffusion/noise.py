from typing import Union
from torchtyping import TensorType

import numpy as np
import scipy.fftpack
from functools import lru_cache

import torch
import torch.nn as nn


class Normal(nn.Module):
    def __init__(self, dim: int, **kwargs):
        super().__init__()
        self.dim = dim

    def forward(self, *shape, **kwargs):
        return torch.randn(*shape, self.dim)

    def covariance(self, **kwargs):
        return torch.eye(self.dim)


class Wiener(nn.Module):
    """
    Wiener process / Brownian motion.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(
        self,
        t: Union[TensorType['seq_len'], TensorType[..., 'seq_len', 1]],
        **kwargs,
    ) -> Union[TensorType['seq_len'], TensorType[..., 'seq_len', 'dim']]:
        one_dimensional = len(t.shape) == 1

        if one_dimensional:
            t = t.unsqueeze(-1)
        t = t.repeat_interleave(self.dim, dim=-1)

        dt = torch.diff(t, dim=-2, prepend=torch.zeros_like(t[...,:1,:]).to(t))
        dw = torch.randn_like(dt) * dt.clamp(1e-5).sqrt()
        w = dw.cumsum(dim=-2)

        if one_dimensional and self.dim == 1:
            w = w.squeeze(-1)
        return w


class OrnsteinUhlenbeck(nn.Module):
    """
    Ornstein-Uhlenbeck process.

    Args:
        theta: Diffusion param, higher value = spikier (float)
    """
    def __init__(self, dim: int, theta: float = 0.5):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.wiener = Wiener(dim)

    def forward(
        self,
        *args,
        t: TensorType[..., 'seq_len', 1],
        **kwargs,
    ) -> TensorType[..., 'seq_len', 'dim']:

        delta = torch.diff(t, dim=-2, prepend=torch.zeros_like(t[...,:1,:]))
        coeff = torch.exp(-self.theta * delta)

        sample = []

        x = torch.randn(*t.shape[:-2], 1, self.dim).to(t)
        for i in range(coeff.shape[-2]):
            z = torch.randn(*t.shape[:-2], 1, self.dim).to(t)
            c = coeff[...,i,None,:]
            x = c * x + torch.sqrt(1 - c**2) * z
            sample.append(x)

        sample = torch.cat(sample, dim=-2)
        return sample

    def covariance(
        self,
        t: TensorType[..., 'seq_len', 1],
        diag_epsilon: float = 1e-4,
        **kwargs,
    ) -> TensorType[..., 'seq_len', 'seq_len']:
        t = t.squeeze(-1)
        diag = torch.eye(t.shape[-1]).to(t) * diag_epsilon
        cov = torch.exp(-(t.unsqueeze(-1) - t.unsqueeze(-2)).abs() * self.theta)
        return cov + diag

    def covariance_cholesky(self, t: TensorType[..., 'seq_len', 1]) -> TensorType[..., 'seq_len', 'seq_len']:
        return torch.linalg.cholesky(self.covariance(t))

    def covariance_inverse(self, t: TensorType[..., 'seq_len', 1]) -> TensorType[..., 'seq_len', 'seq_len']:
        return torch.linalg.inv(self.covariance(t))


class GaussianProcess(nn.Module):
    """
    Gaussian random field for one-dimensional (temporal) data.
    """
    def __init__(self, dim: int, sigma: float = 0.1):
        super().__init__()
        self.dim = dim
        self.sigma = sigma

    def forward(
        self,
        *args,
        t: TensorType[..., 'N', 1],
        **kwargs,
    ) -> TensorType[..., 'N', 'dim']:
        # If N is very large this could become slow
        # In that case, consider using sparse GP
        L = self.covariance_cholesky(t)
        e = torch.randn(*t.shape[:-1], self.dim).to(t)
        return L @ e

    def covariance(
        self,
        t: TensorType[..., 'N', 1],
        diag_epsilon: float = 1e-4,
        **kwargs,
    ) -> TensorType[..., 'N', 'N']:
        if t.shape[-1] != 1 or len(t.shape) < 2:
            t = t.unsqueeze(-1)
        distance = t - t.transpose(-1, -2)
        diag = torch.eye(t.shape[-2]).to(t) * diag_epsilon
        return torch.exp(-torch.square(distance / self.sigma)) + diag

    def covariance_cholesky(self, t: TensorType[..., 'N', 1]) -> TensorType[..., 'N', 'N']:
        return torch.linalg.cholesky(self.covariance(t))

    def covariance_inverse(self, t: TensorType[..., 'N', 1]) -> TensorType[..., 'N', 'N']:
        return torch.linalg.inv(self.covariance(t))
