from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.normal import Normal

from time_match.modules.base import (
    BaseEstimator,
    BaseForecastModule,
    BaseLightningModule,
)


class BlendModel(nn.Module):
    def __init__(
        self,
        n_timestep: int,
        net: nn.Module,
    ):
        super().__init__()
        self.n_timestep = n_timestep
        self._net = net

    @staticmethod
    def squareplus(x: torch.Tensor) -> torch.Tensor:
        # From https://twitter.com/jon_barron/status/1387167648669048833
        return (x + torch.sqrt(torch.square(x) + 4.0)) / 2.0

    def net(
        self,
        inputs: Tensor, # [..., dim]
        time: Tensor, # [...]
        cond: Tensor, # [..., hidden_dim]
    ) -> tuple[Tensor, Tensor]:
        out = self._net(
            inputs=inputs,
            time=time,
            cond=cond,
        )
        mu, sigma = out.chunk(2, dim=-1)
        sigma = self.squareplus(sigma)
        return mu, sigma

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
    ) -> Tensor: # [traj_steps, ..., dim]
        x_alpha = x

        traj = []
        for t in range(self.n_timestep):
            alpha_start = t / self.n_timestep
            alpha_end = (t + 1) / self.n_timestep

            mu, sigma = self.net(
                inputs=x_alpha,
                time=torch.tensor([alpha_start]).to(x_alpha).expand_as(x_alpha[...,:1]),
                cond=cond,
            )

            d = Normal(loc=mu, scale=sigma).sample()
            x_alpha = x_alpha + (alpha_end - alpha_start) * d

            traj.append(x_alpha)
        traj = torch.stack(traj, dim=0)
        ind = torch.linspace(0, self.n_timestep - 1, max(2, traj_steps)).round().long()
        return traj[ind]

    def get_loss(
        self,
        target: Tensor,
        context: Optional[Tensor] = None,
        cond: Optional[Tensor] = None,
    ) -> Tensor:
        if context is None:
            context = torch.rand_like(target)

        alpha = torch.rand_like(target[...,:1])
        x_alpha = alpha * target + (1 - alpha) * context

        mu, sigma = self.net(
            inputs=x_alpha,
            time=alpha,
            cond=cond,
        )

        diff = target - context

        dist = Normal(loc=mu, scale=sigma)
        loss = -dist.log_prob(diff)
        return loss


class BlendForecastModel(BaseForecastModule):
    def __init__(
        self,
        input_size: int,
        n_timestep: int,
        **kwargs,
    ):
        target_dim = 2 * input_size
        super().__init__(input_size=input_size, target_dim=target_dim, **kwargs)
        self.n_timestep = n_timestep
        self.blend_model = BlendModel(self.n_timestep, self.denoising_model)

    def sample(self, x: Tensor, cond: Tensor) -> Tensor:
        return self.blend_model.sample(x, cond)

    def get_loss_values(
        self,
        context: Tensor,
        target: Tensor,
        cond: Tensor,
        observed_values_mask: Tensor,
    ) -> Tensor:
        loss = self.blend_model.get_loss(target=target, context=context, cond=cond)
        loss_values = loss.mean(-1) * observed_values_mask
        return loss_values


class BlendLightningModule(BaseLightningModule):
    def __init__(
        self,
        model_kwargs,
        **kwargs,
    ) -> None:
        super().__init__(model_kwargs=model_kwargs, **kwargs)
        self.model = BlendForecastModel(**model_kwargs)


class BlendEstimator(BaseEstimator):
    def __init__(
        self,
        *args,
        n_timestep: int,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.lightning_module = BlendLightningModule
        self.additional_model_kwargs = {
            "n_timestep": n_timestep,
        }
