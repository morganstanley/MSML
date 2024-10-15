from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from time_match.modules.base import (
    BaseEstimator,
    BaseForecastModule,
    BaseLightningModule,
)


class DDPMModel(nn.Module):
    def __init__(
        self,
        linear_start: float,
        linear_end: float,
        n_timestep: int,
        net: nn.Module,
        beta_schedule: str = 'linear',
    ) -> None:
        super().__init__()
        self.n_timestep = n_timestep
        self.net = net

        if beta_schedule == 'linear':
            self.betas = torch.linspace(linear_start, linear_end, n_timestep)
            self.alphas = torch.cumprod(1 - self.betas, dim=0)
        elif beta_schedule == 'cosine':
            # We don't need linear_start and linear_end here
            cosine_s = 0.008

            ts = torch.arange(n_timestep + 1)
            alphas_bar = torch.cos((ts / n_timestep + cosine_s) / (1 + cosine_s) * torch.pi / 2)**2
            alphas_bar = alphas_bar / alphas_bar.max()

            betas = 1 - alphas_bar[1:] / alphas_bar[:-1]
            self.betas = torch.clip(betas, 0.0001, 0.9999)
            self.alphas = torch.cumprod(1 - self.betas, dim=0)

    def q_sample(
        self,
        step: Tensor, # [..., 1]
        x: Tensor, # [..., dim]
    ) -> tuple[Tensor, Tensor]: # each [..., dim]
        noise = torch.randn_like(x)
        alpha = self.alphas.to(x)[step.long()]
        x_noisy = alpha.sqrt() * x + (1 - alpha).sqrt() * noise
        return x_noisy, noise

    def sample_step(
        self,
        step: int,
        x: Tensor, # [..., dim]
        cond: Tensor # [..., hidden_dim]
    ) -> Tensor: # [..., dim]
        alpha_prod_t = self.alphas[step]
        alpha_prod_t_prev = self.alphas[step - 1] if step > 0 else torch.tensor(1.0).to(x)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        t = torch.ones(x.shape[0]).to(x) * step / self.n_timestep
        pred_noise = self.net(x, t, cond)

        pred_original_sample = (x - beta_prod_t.sqrt() * pred_noise) / alpha_prod_t.sqrt()

        pred_original_sample_coeff = (alpha_prod_t_prev.sqrt() * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t.sqrt() * beta_prod_t_prev / beta_prod_t

        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * x

        if step > 0:
            z = torch.randn_like(x)
            variance = ((1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t).sqrt() * z
            pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample

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
        traj = []
        for diff_step in reversed(range(0, self.n_timestep)):
            x = self.sample_step(diff_step, x, cond)
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
        t = torch.randint_like(target[...,:1], 0, self.n_timestep).long()

        xt, noise = self.q_sample(step=t, x=target)

        pred = self.net(
            inputs=xt,
            time=t / self.n_timestep,
            cond=cond,
        )

        loss = (pred - noise)**2
        return loss


class DDPMForecastModel(BaseForecastModule):
    def __init__(
        self,
        *args,
        linear_start: float,
        linear_end: float,
        n_timestep: int,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ddpm_model = DDPMModel(linear_start, linear_end, n_timestep, self.denoising_model)

    def sample(self, x: Tensor, cond: Tensor) -> Tensor:
        return self.ddpm_model.sample(x, cond)

    def get_loss_values(
        self,
        context: Tensor,
        target: Tensor,
        cond: Tensor,
        observed_values_mask: Tensor,
    ) -> Tensor:
        loss = self.ddpm_model.get_loss(target=target, cond=cond)
        loss_values = loss.mean(-1) * observed_values_mask
        return loss_values


class DDPMLightningModule(BaseLightningModule):
    def __init__(
        self,
        model_kwargs,
        **kwargs,
    ) -> None:
        super().__init__(model_kwargs=model_kwargs, **kwargs)
        self.model = DDPMForecastModel(**model_kwargs)


class DDPMEstimator(BaseEstimator):
    def __init__(
        self,
        *args,
        linear_start: float,
        linear_end: float,
        n_timestep: int,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.lightning_module = DDPMLightningModule
        self.additional_model_kwargs = {
            "linear_start": linear_start,
            "linear_end": linear_end,
            "n_timestep": n_timestep,
        }
