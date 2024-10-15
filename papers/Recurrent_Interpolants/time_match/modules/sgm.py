from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from time_match.modules.base import (
    BaseEstimator,
    BaseForecastModule,
    BaseLightningModule,
)


class SGMModel(nn.Module):
    def __init__(
        self,
        linear_start: float,
        linear_end: float,
        n_timestep: int,
        net: nn.Module,
    ) -> None:
        super().__init__()
        self.n_timestep = n_timestep
        self.net = net

        self.beta_min = linear_start
        self.beta_max = linear_end

    def beta(self, t: Tensor) -> Tensor:
        return self.beta_min * (1 - t) + self.beta_max * t

    def beta_integral(self, t: Tensor) -> Tensor:
        return 0.5 * (self.beta_max - self.beta_min) * t**2 + self.beta_min * t

    def marginal(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        beta_int = self.beta_integral(t)
        mean = x * torch.exp(-0.5 * beta_int)
        var = (1 - torch.exp(-beta_int)).clamp_min(1e-5)
        std = torch.sqrt(var)
        return mean, std

    def q_sample(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        noise = torch.randn_like(x)
        mean, std = self.marginal(x, t)
        x_noisy = mean + std * noise
        return x_noisy, noise

    def sample_step(
        self,
        t: Tensor, # [...]
        x: Tensor, # [..., dim]
        cond: Tensor, # [..., hidden_dim]
        dt: float,
    ) -> Tensor: # [..., dim]
        beta = self.beta(t)
        _, std = self.marginal(x, t)
        score = -self.net(x, t, cond) / std
        dx = -0.5 * beta * (x + score)
        # Negative sign because integration t1->t0
        return x - dx * dt

    def sample(
        self,
        x: Tensor,
        cond: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        sample = self.sample_traj(
            x=x,
            cond=cond,
            traj_steps=1,
        )
        return sample[-1]

    def sample_traj(
        self,
        x: Tensor,
        cond: Optional[Tensor] = None,
        traj_steps: int = 10,
        **kwargs,
    ) -> Tensor:
        x = torch.randn_like(x)
        dt = 1 / self.n_timestep
        traj = []
        for diff_step in reversed(range(0, self.n_timestep)):
            step = (diff_step + 1) / self.n_timestep
            t = torch.ones_like(x[...,:1]) * step
            x = self.sample_step(t, x, cond, dt)
            traj.append(x)
        traj = torch.stack(traj, dim=0)
        ind = torch.linspace(0, self.n_timestep - 1, max(2, traj_steps)).round().long()
        return traj[ind]

    def get_loss(
        self,
        target: Tensor,
        cond: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        t = torch.rand_like(target[...,:1])
        xt, noise = self.q_sample(target, t)

        pred = self.net(
            inputs=xt,
            time=t,
            cond=cond,
        )

        loss = (pred - noise)**2
        return loss

class SGMForecastModel(BaseForecastModule):
    def __init__(
        self,
        *args,
        linear_start: float,
        linear_end: float,
        n_timestep: int,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sgm_model = SGMModel(linear_start, linear_end, n_timestep, self.denoising_model)

    def sample(self, x: Tensor, cond: Tensor) -> Tensor:
        return self.sgm_model.sample(x, cond)

    def get_loss_values(
        self,
        context: Tensor,
        target: Tensor,
        cond: Tensor,
        observed_values_mask: Tensor,
    ) -> Tensor:
        loss = self.sgm_model.get_loss(target=target, cond=cond)
        loss_values = loss.mean(-1) * observed_values_mask
        return loss_values


class SGMLightningModule(BaseLightningModule):
    def __init__(
        self,
        model_kwargs,
        **kwargs,
    ) -> None:
        super().__init__(model_kwargs=model_kwargs, **kwargs)
        self.model = SGMForecastModel(**model_kwargs)


class SGMEstimator(BaseEstimator):
    def __init__(
        self,
        *args,
        linear_start: float,
        linear_end: float,
        n_timestep: int,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.lightning_module = SGMLightningModule
        self.additional_model_kwargs = {
            "linear_start": linear_start,
            "linear_end": linear_end,
            "n_timestep": n_timestep,
        }
