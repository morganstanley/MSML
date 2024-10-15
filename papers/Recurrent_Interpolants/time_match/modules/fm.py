from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from time_match.modules.base import (
    BaseEstimator,
    BaseForecastModule,
    BaseLightningModule,
)


class FMModel(nn.Module):
    def __init__(
        self,
        sigma_min: float,
        n_timestep: int,
        net: nn.Module,
    ):
        super().__init__()
        self.n_timestep = n_timestep
        self.dt = 1. / n_timestep
        self.sigma_min = sigma_min
        self.net = net

    def psi_t(self, x, t):
        noise = torch.randn_like(x)
        xt = (1 - (1 - self.sigma_min) * t) * noise + t * x
        return xt, noise

    def sample_traj(
        self,
        x: Tensor, # [..., dim]
        cond: Optional[Tensor] = None, # [..., hidden_dim]
        traj_steps: int = 10,
        **kwargs,
    ) -> Tensor: # [traj_steps, ..., dim]
        x = torch.randn_like(x)
        dt = 1. / self.n_timestep
        t = torch.zeros_like(x[...,:1])
        traj = []
        for _ in range(self.n_timestep):
            x = x + self.net(x, t, cond=cond) * dt
            t = t + dt
            traj.append(x)
        traj = torch.stack(traj, dim=0)
        ind = torch.linspace(0, self.n_timestep - 1, max(2, traj_steps)).round().long()
        return traj[ind]

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

    def get_loss(
        self,
        target: Tensor,
        cond: Optional[Tensor] = None,
    ) -> Tensor:
        t = torch.rand_like(target[...,:1])
        xt, noise = self.psi_t(target, t)

        vt = self.net(
            inputs=xt,
            time=t,
            cond=cond,
        )

        y = (target - (1 - self.sigma_min) * noise)

        loss = (vt - y)**2
        return loss


class FMForecastingModel(BaseForecastModule):
    def __init__(
        self,
        *args,
        sigma_min: float,
        n_timestep: int,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.fm_model = FMModel(sigma_min, n_timestep, self.denoising_model)

    def sample(self, x: Tensor, cond: Tensor) -> Tensor:
        return self.fm_model.sample(x, cond)

    def get_loss_values(
        self,
        context: Tensor,
        target: Tensor,
        cond: Tensor,
        observed_values_mask: Tensor,
    ) -> Tensor:
        loss = self.fm_model.get_loss(target=target, cond=cond)
        loss_values = loss.mean(-1) * observed_values_mask
        return loss_values


class FMLightningModule(BaseLightningModule):
    def __init__(
        self,
        model_kwargs,
        **kwargs,
    ) -> None:
        super().__init__(model_kwargs=model_kwargs, **kwargs)
        self.model = FMForecastingModel(**model_kwargs)


class FMEstimator(BaseEstimator):
    def __init__(
        self,
        *args,
        sigma_min: float,
        n_timestep: int,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.lightning_module = FMLightningModule
        self.additional_model_kwargs = {
            "sigma_min": sigma_min,
            "n_timestep": n_timestep,
        }
