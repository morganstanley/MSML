import time
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import Module

import policy
import sde
from loss import compute_sb_nll_alternate_train


def freeze_policy(policy):
    for p in policy.parameters():
        p.requires_grad = False
    policy.eval()
    return policy

def activate_policy(policy):
    for p in policy.parameters():
        p.requires_grad = True
    policy.train()
    return policy


class SchroedingerBridge(Module):
    def __init__(
        self,
        dim: int,
        # Diffusion time parameters
        t0: float,
        T: float,
        interval: int,
        beta_min: float,
        beta_max: float,
        beta_r: float,
        # Networks
        forward_net: Module,
        backward_net: Module,
    ):
        super().__init__()
        self.dim = dim
        self.t0 = t0
        self.T = T
        self.interval = interval
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.beta_r = beta_r

        self.q = torch.distributions.MultivariateNormal(torch.zeros(dim), torch.eye(dim))

        self.start_time=time.time()
        self.register_buffer('ts', torch.linspace(t0, T, interval))

        # build dynamics, forward (z_f) and backward (z_b) policies
        self.dyn = sde.build(t0, T, beta_min, beta_max, beta_r, interval, p=None, q=self.q)
        self.z_f = policy.build(net=forward_net, dyn=self.dyn, direction='forward')  # p -> q
        self.z_b = policy.build(net=backward_net, dyn=self.dyn, direction='backward') # q -> p

    @torch.no_grad()
    def sample(
        self,
        x: Tensor, # [B, 1, D]
        cond: Tensor, # [B, 1, H]
    ) -> Tensor: # [B, 1, D]
        A = self.z_f.net.A.detach().clone()
        D_mat = torch.eye(A.shape[0]).to(A) - 2 * A
        cov = sde.compute_variance(torch.Tensor([1.0]).to(A), D_mat, self.beta_min, self.beta_max, self.beta_r)[0][0]

        _, _, x = self.dyn.sample_traj(
            x=torch.randn_like(x),
            ts=self.ts,
            cond=cond,
            policy=self.z_b,
            save_traj=False,
            cov=cov,
        )
        return x

    @torch.no_grad()
    def sample_train_data(
        self,
        x: Tensor, # [batch, seq_len, dim]
        cond: Tensor, # [batch, seq_len, dim]
        policy: Any,
    ) -> Tuple[Tensor, Tensor]: # each [batch, steps, seq_len, dim]
        policy = freeze_policy(policy)
        train_xs, train_zs, _ = self.dyn.sample_traj(x=x, ts=self.ts, cond=cond, policy=policy)
        return train_xs.detach(), train_zs.detach()

    def get_sb_loss(
        self,
        x: Tensor, # [B, T, D]
        cond: Tensor, # [B, T, H]
        direction: str,
    ) -> Tensor: # [B, steps, T, D]
        B, T, D = x.shape
        x = x.reshape(-1, 1, D)
        cond = cond.reshape(-1, 1, cond.shape[-1])

        if direction == 'forward':
            policy_opt, policy_impt = self.z_f, self.z_b
        elif direction == 'backward':
            policy_opt, policy_impt = self.z_b, self.z_f

        policy_impt = freeze_policy(policy_impt)

        train_xs, train_zs = self.sample_train_data(x, cond, policy_impt)

        train_xs = train_xs.view(B, T, -1, D)
        train_zs = train_zs.view(B, T, -1, D)

        ind = np.random.randint(0, len(self.ts), B)
        train_ts = self.ts[ind]
        train_xs = train_xs[range(B),:,ind]
        train_zs = train_zs[range(B),:,ind]

        policy_impt = freeze_policy(policy_impt)
        policy_opt = activate_policy(policy_opt)

        train_ts = train_ts.repeat(T)

        train_xs.requires_grad_(True)

        train_xs = train_xs.flatten(0, 1).unsqueeze(1)
        train_zs = train_zs.flatten(0, 1).unsqueeze(1)

        loss = compute_sb_nll_alternate_train(
            dyn=self.dyn,
            ts=train_ts,
            xs=train_xs,
            cond=cond,
            zs_impt=train_zs,
            policy_opt=policy_opt,
        )
        loss = loss.reshape(B, T, D)
        return loss

    def get_backward_loss(
        self,
        x: Tensor, # [B, T, D]
        cond: Tensor, # [B, T, H]
    ) -> Tensor: # [B, steps, T, D]
        B, T, D = x.shape

        policy = activate_policy(self.z_b)

        A = self.z_f.net.A.detach().clone()

        compute_xs_label_fn = sde.get_xs_label_computer(
            beta_min=self.beta_min,
            beta_max=self.beta_max,
            beta_r=self.beta_r,
            ts=self.ts,
            A=A,
        )

        ind = torch.randint(self.interval, (x.shape[0],), device=x.device)
        ts = self.ts[ind].detach()

        xs, label = compute_xs_label_fn(
            x0=x,
            ind=ind,
        )

        predict = policy(
            x=xs.reshape(B * T, 1, D),
            t=ts.repeat(T),
            cond=cond.reshape(B * T, 1, -1),
        )
        predict = predict.reshape(B, T, D)
        loss = (label - predict)**2
        return loss


class Diffusion(Module):
    def __init__(
        self,
        dim: int,
        # Diffusion time parameters
        interval: int,
        beta_min: float,
        beta_max: float,
        # Network
        backward_net: Module,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.n_timestep = interval

        betas = torch.linspace(beta_min, beta_max, interval)
        alphas = torch.cumprod(1 - betas, dim=0)
        assert alphas.min() >= 0 and alphas.max() <= 1

        self.register_buffer("betas", torch.tensor(betas))
        self.register_buffer("alphas", torch.tensor(alphas))

        self.net = backward_net

    def q_sample(self, step, x):
        noise = torch.randn_like(x)
        alpha = self.alphas[step.long()].unsqueeze(-1)
        x_noisy = torch.sqrt(alpha) * x + torch.sqrt(1 - alpha) * noise
        return x_noisy, noise

    def sample_step(self, step, x, cond):
        alpha_prod_t = self.alphas[step]
        alpha_prod_t_prev = self.alphas[step - 1] if step > 0 else torch.tensor(1.0).to(x)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        t = torch.ones(x.shape[0]).to(x) * step
        pred_noise = self.net(x, t / self.n_timestep, cond)

        pred_original_sample = (x - beta_prod_t.sqrt() * pred_noise) / alpha_prod_t.sqrt()

        pred_original_sample_coeff = (alpha_prod_t_prev.sqrt() * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t.sqrt() * beta_prod_t_prev / beta_prod_t

        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * x

        if step > 0:
            z = torch.randn_like(x)
            variance = ((1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t).sqrt() * z
            pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample

    @torch.no_grad()
    def sample(
        self,
        x: Tensor, # [B, 1, D]
        cond: Tensor, # [B, 1, H]
    ) -> Tensor: # [B, 1, D]
        for diff_step in reversed(range(0, self.n_timestep)):
            x = self.sample_step(diff_step, x, cond)
        return x

    def get_backward_loss(
        self,
        x: Tensor, # [B, T, D]
        cond: Tensor, # [B, T, H]
        **kwargs,
    ) -> Tensor:
        B, T, D = x.shape

        timesteps = torch.randint(0, self.n_timestep, size=(B, T), device=x.device)
        xt, noise = self.q_sample(timesteps, x)

        pred = self.net(
            xt.reshape(B * T, 1, D),
            timesteps.reshape(B * T) / self.n_timestep,
            cond.reshape(B * T, 1, -1),
        )
        pred = pred.view(B, T, D)
        loss = (pred - noise)**2
        return loss
