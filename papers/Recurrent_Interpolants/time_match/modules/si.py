from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torchsde import sdeint

from time_match.modules.base import (
    BaseEstimator,
    BaseForecastModule,
    BaseLightningModule,
)
from time_match.utils.gamma import Gamma, QuadGamma, SqrtGamma, TrigGamma
from time_match.utils.interpolant import EncDec, Linear, Trigonometric

EPS = 1e-4

def get_gamma(gamma):
    if gamma == 'trig':
        return TrigGamma()
    elif gamma == 'quad':
        return QuadGamma()
    elif gamma == 'sqrt':
        return SqrtGamma()
    elif gamma == 'zero':
        return Gamma()

def get_interpolant(interpolant, gamma):
    if interpolant == 'linear':
        return Linear(gamma)
    elif interpolant == 'trig':
        return Trigonometric(gamma)
    elif interpolant == 'encdec':
        return EncDec(gamma)

class SDE(nn.Module):
    noise_type = 'diagonal'
    sde_type = 'ito'

    def __init__(self, drift, epsilon, cond):
        super().__init__()
        self.epsilon = epsilon
        self.drift = drift
        self.cond = cond

    # Drift
    @torch.no_grad()
    def f(self, t, x_t):
        t = torch.ones_like(x_t[...,:1]) * t.clamp(0., 1.)
        drift = self.drift(x_t, t, cond=self.cond)
        return drift

    # Diffusion
    @torch.no_grad()
    def g(self, t, x_t):
        diff = torch.zeros_like(x_t) + np.sqrt(2 * self.epsilon)
        return diff

class SDEUncond(nn.Module):
    noise_type = 'diagonal'
    sde_type = 'ito'

    def __init__(self, drift, epsilon):
        super().__init__()
        self.epsilon = epsilon
        self.drift = drift

    # Drift
    @torch.no_grad()
    def f(self, t, x_t):
        t = torch.ones_like(x_t[...,:1]) * t.clamp(0., 1.)
        drift = self.drift(x_t, t)
        return drift

    # Diffusion
    @torch.no_grad()
    def g(self, t, x_t):
        diff = torch.zeros_like(x_t) + np.sqrt(2 * self.epsilon)
        return diff


@torch.no_grad()
def manual_solver(sde, x, num_steps):
    dt = 1. / num_steps

    t = torch.tensor(0.)
    for step in range(num_steps):
        drift = sde.f(t, x)
        diff = sde.g(t, x)
        x += drift * dt
        if step < num_steps - 1:
            x += diff * np.sqrt(dt) * torch.randn_like(x)
        t += dt

    return x


class SIModel(nn.Module):
    def __init__(
        self,
        start_noise: bool,
        epsilon: float,
        n_timestep: int,
        gamma: str,
        interpolant: str,
        importance_sampling: bool,
        velocity: nn.Module,
        score: nn.Module,
    ):
        """Model using both velocity and score functions."""
        super().__init__()
        self.start_noise = start_noise
        self.epsilon = epsilon
        self.n_timestep = n_timestep

        self.gamma = get_gamma(gamma)
        self.interpolant = get_interpolant(interpolant, self.gamma)
        self.importance_sampling = importance_sampling
        self.velocity = velocity
        self.score = score

    def sample(
        self,
        x: Tensor,
        cond: Optional[Tensor] = None,
        adaptive: bool = False,
        dt: float = 1e-3,
        epsilon: Optional[float] = None,
        **kwargs,
    ) -> Tensor:
        if self.start_noise or x is None:
            x = torch.randn_like(x)
        sample = self.sample_traj(
            x=x,
            cond=cond,
            adaptive=adaptive,
            traj_steps=1,
            epsilon=epsilon,
        )
        return sample[-1]

    def sample_traj(
        self,
        x: Tensor,
        cond: Optional[Tensor] = None,
        adaptive: bool = False,
        traj_steps: int = 10,
        epsilon: Optional[float] = None,
        **kwargs,
    ) -> Tensor:
        """
        traj_steps
            Sliced samples from the path. It is NOT the number of steps for SDE solver.
        """
        epsilon = self.epsilon if epsilon is None else epsilon
        if self.start_noise or x is None:
            x = torch.randn_like(x)
        # Note that torchsde.sdeint only handles (B,C) input.
        if len(x.shape)==2:
            bt, d, dim = x.shape[0], x.shape[-1], x.shape
        elif len(x.shape)==3:
            bt, d, dim = x.shape[0] * x.shape[1], x.shape[-1], x.shape

        if cond is None:  # Synthetic experiment doesn't need cond.
            drift = lambda y, t: self.velocity(y, t).reshape(bt, d) + \
                    epsilon * self.score(y, t).reshape(bt, d)
        else:
            drift = lambda y, t: self.velocity(y, t, cond=cond).reshape(bt, d) + \
                    epsilon * self.score(y, t, cond=cond).reshape(bt, d)
        sde = SDEUncond(drift, epsilon)
        sample_t = torch.linspace(0.0, 1.0, max(traj_steps, 2)).to(x)
        samples = sdeint(sde, x.reshape(bt, d), sample_t, adaptive=False, dt=1/self.n_timestep)
        sample_steps = 2 if traj_steps == 1 else traj_steps

        samples = samples.reshape((sample_steps,*dim))
        return samples


    def train_drift_gaussian(self, x_0, x_1, t, cond=None):
        """
        The interpolant is in Eq. 1.1, 2.1,
        """
        interpol = self.interpolant.interpolate(x_0, x_1, t)
        z = torch.randn_like(interpol)

        inter = interpol + self.interpolant.gamma(t) * z
        inter_ = interpol - self.interpolant.gamma(t) * z

        dt_interpol = self.interpolant.dt_interpolate(x_0, x_1, t)

        if cond is None:
            field = self.velocity(inter, t)
            field_ = self.velocity(inter_, t)
        else:
            field = self.velocity(inter, t, cond=cond)
            field_ = self.velocity(inter_, t, cond=cond)

        l = 0.5 * ((field - dt_interpol - self.interpolant.gamma_(t) * z) ** 2).sum(dim=-1)
        l += 0.5 * ((field_ - dt_interpol + self.interpolant.gamma_(t) * z) ** 2).sum(dim=-1)

        return l


    def train_score_gaussian(self, x_0, x_1, t, cond=None):
        """
        Paper eq. 2.15.
        """
        interpol = self.interpolant.interpolate(x_0, x_1, t)
        z = torch.randn_like(interpol)

        inter = interpol + self.interpolant.gamma(t) * z
        inter_ = interpol - self.interpolant.gamma(t) * z

        if cond is None:
            score = self.score(inter, t)
            score_ = self.score(inter_, t)
        else:
            score = self.score(inter, t, cond=cond)
            score_ = self.score(inter_, t, cond=cond)

        l = 0.5 * ((self.interpolant.gamma(t) * score + z) ** 2).sum(dim=-1)
        l += 0.5 * ((self.interpolant.gamma(t) * score_ - z) ** 2).sum(dim=-1)

        return l


    def get_loss(
        self,
        target: Tensor, # [..., dim]
        context: Optional[Tensor] = None, # [..., dim]
        cond: Optional[Tensor] = None, # [..., hidden_dim]
    ) -> Tensor: # [..., dim]
        if self.importance_sampling:
            return self.get_loss_importance_sampling(target, context, cond)
        else:
            return self.get_loss_uniform_sampling(target, context, cond)

    def get_loss_discrete_sampling(
        self,
        target: Tensor, # [..., dim]
        context: Optional[Tensor] = None, # [..., dim]
        cond: Optional[Tensor] = None, # [..., hidden_dim]
    ) -> Tensor: # [..., dim]
        if self.start_noise or context is None:
            context = torch.randn_like(target)
        sample_t = torch.linspace(0.001, 0.999, self.n_timestep).to(target)
        index = torch.randint(self.n_timestep, target[...,:1].shape).to(target.device)
        t = sample_t[index]
        loss = self.train_score_gaussian(context, target, t, cond) + \
               self.train_drift_gaussian(context, target, t, cond)
        return loss

    def get_loss_uniform_sampling(
        self,
        target: Tensor, # [..., dim]
        context: Optional[Tensor] = None, # [..., dim]
        cond: Optional[Tensor] = None, # [..., hidden_dim]
    ) -> Tensor: # [..., dim]
        if self.start_noise or context is None:
            context = torch.randn_like(target)
        t = torch.rand(target[...,:1].shape).to(target)
        loss = self.train_score_gaussian(context, target, t, cond) + \
               self.train_drift_gaussian(context, target, t, cond)

        return loss

    def get_loss_importance_sampling(
        self,
        target: Tensor, # [..., dim]
        context: Optional[Tensor] = None, # [..., dim]
        cond: Optional[Tensor] = None, # [..., hidden_dim]
    ) -> Tensor: # [..., dim]
        """Weighted loss using importance sampling. The proposal distribution is Beta."""
        if self.start_noise or context is None:
            context = torch.randn_like(target)

        t_sampler = torch.distributions.beta.Beta(torch.tensor([0.1]).to(target), torch.tensor([0.1]).to(target))
        # torch.distributions .sample() automatically extend the one more dimension.
        t = t_sampler.sample(sample_shape=target[...,:1].shape).squeeze(-1).to(target)
        # t = t * 0.98 + 0.01  # avoid time points near ends.
        sampler_pdf = torch.exp(t_sampler.log_prob(t))
        weights = 1 / sampler_pdf
        weights = weights / weights.sum()
        loss = self.train_score_gaussian(context, target, t, cond) + \
               self.train_drift_gaussian(context, target, t, cond)

        if len(target.shape) == 2:  # weights and loss are in [B,]
            loss = loss * weights
        elif len(target.shape) == 3: # In our setting of time series [B,T,C]., B,T is the actual batch axes.
            loss = loss * weights.squeeze(-1)  # [B,T,]
        return loss


class SIForecastModel(BaseForecastModule):
    def __init__(
        self,
        *args,
        start_noise: bool,
        epsilon: float,
        n_timestep: int,
        gamma: str,
        interpolant: str,
        importance_sampling: bool,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.si_model = SIModel(start_noise, epsilon, n_timestep, gamma, interpolant,
                importance_sampling, velocity=self.velocity_model, score=self.denoising_model)

    def sample(self, x: Tensor, cond: Optional[Tensor] = None) -> Tensor:
        return self.si_model.sample(x=x, cond=cond)

    def get_loss_values(
        self,
        context: Tensor,  #[B,T,K]
        target: Tensor,  #[B,T,K]. context and target must have same shape.
        cond: Tensor,  # [B,T,hidden_dim]. rnn_outputs[:, -prediction_length:]
        observed_values_mask: Tensor,
    ) -> Tensor:
        loss = self.si_model.get_loss(context=context, target=target, cond=cond)
        loss_values = loss * observed_values_mask
        return loss_values


class SILightningModule(BaseLightningModule):
    def __init__(
        self,
        model_kwargs,
        **kwargs,
    ) -> None:
        super().__init__(model_kwargs=model_kwargs, **kwargs)
        self.model = SIForecastModel(**model_kwargs)


class SIEstimator(BaseEstimator):
    def __init__(
        self,
        *args,
        start_noise: bool,
        epsilon: float,
        n_timestep: int,
        gamma: str,
        interpolant: str,
        importance_sampling: bool,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.lightning_module = SILightningModule
        self.additional_model_kwargs = {
            "start_noise": start_noise,
            "epsilon": epsilon,
            "n_timestep": n_timestep,
            "gamma": gamma,
            "interpolant": interpolant,
            "importance_sampling": importance_sampling,
        }
