from typing import Callable, Tuple, Optional, Union
from torchtyping import TensorType
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as td

from torchsde import sdeint
from torchdiffeq import odeint

from tsdiff.diffusion.noise import Normal, OrnsteinUhlenbeck, GaussianProcess


class ContinuousDiffusion(nn.Module):
    """
    Continuous diffusion using SDEs (https://arxiv.org/abs/2011.13456)

    Args:
        dim: Dimension of data
        beta_fn: Scheduler for noise levels
        t1: Final diffusion time
        noise_fn: Type of noise
        predict_gaussian_noise: Whether to approximate score with unit normal
        loss_weighting: Function returning loss weights given diffusion time
    """
    def __init__(
        self,
        dim: int,
        beta_fn: Callable,
        t1: float = 1.0,
        noise_fn: Callable = None,
        loss_weighting: Callable = None,
        is_time_series: bool = False,
        predict_gaussian_noise: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.t1 = t1
        self.predict_gaussian_noise = predict_gaussian_noise
        self.is_time_series = is_time_series

        self.beta_fn = beta_fn
        self.noise = noise_fn
        self.loss_weighting = partial(loss_weighting or (lambda beta, i: 1), beta_fn)

    def forward(
        self,
        x: TensorType[..., 'dim'],
        i: TensorType[..., 1],
        _return_all: Optional[bool] = False, # For internal use only
        **kwargs,
    ) -> Tuple[TensorType[..., 'dim'], TensorType[..., 'dim']]:

        noise_gaussian = torch.randn_like(x)

        if self.is_time_series:
            cov = self.noise.covariance(**kwargs)
            L = torch.linalg.cholesky(cov)
            noise = L @ noise_gaussian
        else:
            noise = noise_gaussian

        beta_int = self.beta_fn.integral(i)

        mean = x * torch.exp(-beta_int / 2)
        std = (1 - torch.exp(-beta_int)).clamp(1e-5).sqrt()

        y = mean + std * noise

        if _return_all:
            return y, noise, mean, std, cov if self.is_time_series else None

        if self.predict_gaussian_noise:
            return y, noise_gaussian
        else:
            return y, noise

    def get_loss(
        self,
        model: Callable,
        x: TensorType[..., 'dim'],
        **kwargs,
    ) -> TensorType[..., 1]:

        i = torch.rand(x.shape[0], *(1,) * len(x.shape[1:])).expand_as(x[...,:1]).to(x)
        i = i * self.t1

        x_noisy, noise = self.forward(x, i, **kwargs)

        pred_noise = model(x_noisy, i=i, **kwargs)
        loss = self.loss_weighting(i) * (pred_noise - noise)**2

        return loss

    def _get_score(self, model, x, i, L=None, **kwargs):
        """
        Returns score: ∇_xs log p(xs)
        """
        if isinstance(i, float):
            i = torch.Tensor([i]).to(x)
        if i.shape[:-1] != x.shape[:-1]:
            i = i.view(*(1,) * len(x.shape)).expand_as(x[...,:1])

        beta_int = self.beta_fn.integral(i)
        std = (1 - torch.exp(-beta_int)).clamp(1e-5).sqrt()

        noise = model(x, i=i, **kwargs)

        if L is not None:
            # We have to compute the score using -Sigma.inv() @ noise / std
            # assuming noise~N(0, Sigma).
            # If `predict_gaussian_noise=False`, compute (LL^T).inv()
            # Else, we can simplify (LL^T).inv() @ L @ noise
            # to (L^T).inv() @ noise, where noise~N(0, I).
            # So we anyways have to do (L^T).inv(), and sometimes L.inv()
            if not self.predict_gaussian_noise:
                noise = torch.linalg.solve_triangular(L, noise, upper=False)
            noise = torch.linalg.solve_triangular(L.transpose(-1, -2), noise, upper=True)

        score = -noise / std
        return score

    @torch.no_grad()
    def log_prob(
        self,
        model: Callable,
        x: Union[TensorType[..., 'dim'], TensorType[..., 'seq_len', 'dim']],
        num_samples: int = 1,
        **kwargs,
    ) -> TensorType[..., 1]:
        model.train() # Allows backprop through RNN
        self._e = torch.randn(num_samples, *x.shape).to(x)

        if self.is_time_series:
            cov = self.noise.covariance(**kwargs)
            L = torch.linalg.cholesky(cov)
        else:
            L = None

        def drift(i, state):
            y, _ = state
            with torch.set_grad_enabled(True):
                y = y.requires_grad_(True)
                score = self._get_score(model, y, i=i, L=L, **kwargs)
                if self.is_time_series:
                    # Have to include `cov` since g(t) = "scalar" * L @ dW
                    score = cov @ score
                dy = -0.5 * self.beta_fn(i) * (y + score)
                divergence = divergence_approx(dy, y, self._e, num_samples=num_samples)
            return dy, -divergence

        interval = torch.Tensor([0, self.t1]).to(x)

        # states = odeint(drift, (x, torch.zeros_like(x).to(x)), interval, rtol=1e-6, atol=1e-5)
        states = odeint(drift, (x, torch.zeros_like(x).to(x)), interval,
            method='rk4', options={'step_size': .01})
        y, div = states[0][-1], states[1][-1]

        if self.is_time_series:
            p0 = td.Independent(torch.distributions.MultivariateNormal(
                torch.zeros_like(y).transpose(-1, -2),
                cov.unsqueeze(-3).repeat_interleave(self.dim, dim=-3),
            ), 1)
            log_prob = p0.log_prob(y.transpose(-1, -2)) - div.sum([-1, -2])
            log_prob = log_prob / x.shape[-2]
        else:
            p0 = td.Independent(td.Normal(torch.zeros_like(y), torch.ones_like(y)), 1)
            log_prob = p0.log_prob(y) - div.sum(-1)

        return log_prob.unsqueeze(-1)

    @torch.no_grad()
    def sample(
        self,
        model: Callable,
        num_samples: int,
        device: str = None,
        use_ode: bool = True,
        **kwargs,
    ) -> TensorType['num_samples', 'dim']:
        if isinstance(num_samples, int):
            num_samples = (num_samples,)

        sampler = self.ode_sample if use_ode else self.sde_sample
        return sampler(model, num_samples, device, **kwargs)

    @torch.no_grad()
    def ode_sample(
        self,
        model: Callable,
        num_samples: int,
        device: str = None,
        **kwargs,
    ) -> TensorType['num_samples', 'dim']:
        if self.is_time_series:
            cov = self.noise.covariance(**kwargs)
            L = torch.linalg.cholesky(cov)
        else:
            L = None

        def drift(i, y):
            score = self._get_score(model, y, i=i, L=L, **kwargs)
            if self.is_time_series:
                # Have to include `cov` since g(t) = "scalar" * L @ dW
                score = cov @ score
            return -0.5 * self.beta_fn(i) * (y + score)

        x = self.noise(*num_samples, **kwargs).to(device)
        t = torch.Tensor([self.t1, 0]).to(device)
        y = odeint(drift, x, t, method='rk4', options={'step_size': .01})[1]
        # y = odeint(drift, x, t, rtol=1e-6, atol=1e-5)[1]

        return y

    @torch.no_grad()
    def sde_sample(
        self,
        model: Callable,
        num_samples: int,
        device: str = None,
        **kwargs,
    ) -> TensorType['num_samples', 'dim']:

        if self.is_time_series:
            cov = self.noise.covariance(**kwargs)
            L = torch.linalg.cholesky(cov)
        else:
            L = None

        is_time_series = self.is_time_series

        x = self.noise(*num_samples, **kwargs).to(device)
        shape = x.shape
        x = x.transpose(-2, -1).flatten(0, -2)

        class SDE(nn.Module):
            noise_type = 'general' if is_time_series else 'diagonal'
            sde_type = 'ito'

            def __init__(self, beta_fn, _get_score):
                super().__init__()
                self.beta_fn = beta_fn
                self._get_score = _get_score

            def f(self, i, inp):
                i = -i
                inp = inp.view(*shape) # Reshape back to original

                score = self._get_score(model, inp, i=i, L=L, **kwargs)
                if is_time_series:
                    score = cov @ score

                dx = self.beta_fn(i) * (0.5 * inp + score)

                if is_time_series:
                    return dx.transpose(-1, -2).flatten(0, -2)
                return dx.view(-1, shape[-1])

            def g(self, i, inp):
                i = -i
                beta = -self.beta_fn(i).sqrt()

                if is_time_series:
                    return (beta * L).repeat_interleave(shape[-1], dim=0)
                return beta.view(1, 1).repeat(np.prod(shape[:-1]), shape[-1]).to(device)

        sde = SDE(self.beta_fn, self._get_score)
        interval = torch.Tensor([-self.t1, 0]).to(device) # Time from -t1 to 0

        step_size = self.t1 / 100
        if not is_time_series:
            x = x.view(-1, shape[-1])
        else:
            x = x.view(-1, shape[-2])
        y = sdeint(sde, x, interval, dt=step_size)[-1]
        y = y.view(*shape)

        return y


class ContinuousGaussianDiffusion(ContinuousDiffusion):
    """ Continuous diffusion using Gaussian noise """
    def __init__(self, dim: int, beta_fn: Callable, predict_gaussian_noise=None, **kwargs):
        super().__init__(dim, beta_fn, noise_fn=Normal(dim), predict_gaussian_noise=True, **kwargs)


class ContinuousOUDiffusion(ContinuousDiffusion):
    """ Continuous diffusion using noise coming from an OU process """
    def __init__(self, dim: int, beta_fn: Callable, predict_gaussian_noise: bool = False, theta: float = 0.5, **kwargs):
        super().__init__(
            dim=dim,
            beta_fn=beta_fn,
            noise_fn=OrnsteinUhlenbeck(dim, theta=theta),
            predict_gaussian_noise=predict_gaussian_noise,
            is_time_series=True,
            **kwargs,
        )


class ContinuousGPDiffusion(ContinuousDiffusion):
    """ Continuous diffusion using noise coming from a Gaussian process """
    def __init__(self, dim: int, beta_fn: Callable, predict_gaussian_noise: bool = False, sigma: float = 0.1, **kwargs):
        super().__init__(
            dim=dim,
            beta_fn=beta_fn,
            noise_fn=GaussianProcess(dim, sigma=sigma),
            predict_gaussian_noise=predict_gaussian_noise,
            is_time_series=True,
            **kwargs,
        )


def divergence_approx(output, input, e, num_samples=1):
    out = 0
    for i in range(num_samples):
        out += torch.autograd.grad(output, input, e[i], create_graph=True)[0].detach() * e[i]
    return out / num_samples
