from typing import Any, Callable, Tuple, Union
from torchtyping import TensorType

import torch
import torch.nn as nn
import torch.distributions as td

from tsdiff.diffusion.noise import Normal, OrnsteinUhlenbeck, GaussianProcess


class DiscreteDiffusion(nn.Module):
    """
    Discrete diffusion (https://arxiv.org/abs/2006.11239)

    Args:
        dim: Dimension of data
        num_steps: Number of diffusion steps
        beta_fn: Scheduler for noise levels
        noise_fn: Type of noise
        parallel_elbo: Whether to compute ELBO in parallel or not
    """
    def __init__(
        self,
        dim: int,
        num_steps: int,
        beta_fn: Callable,
        noise_fn: Callable,
        parallel_elbo: bool = False,
        is_time_series: bool = False,
        predict_gaussian_noise: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.num_steps = num_steps
        self.parallel_elbo = parallel_elbo
        self.is_time_series = is_time_series
        self.predict_gaussian_noise = predict_gaussian_noise

        self.betas = beta_fn(torch.linspace(0, 1, num_steps))
        self.alphas = torch.cumprod(1 - self.betas, dim=0)

        self.noise = noise_fn

    def forward(
        self,
        x: TensorType[..., 'dim'],
        i: TensorType[..., 1],
        **kwargs,
    ) -> Tuple[TensorType[..., 'dim'], TensorType[..., 'dim']]:

        noise_gaussian = torch.randn_like(x)

        if self.is_time_series:
            cov = self.noise.covariance(**kwargs)
            L = torch.linalg.cholesky(cov)
            noise = L @ noise_gaussian
        else:
            noise = noise_gaussian

        alpha = self.alphas[i.long()].to(x)
        y = torch.sqrt(alpha) * x + torch.sqrt(1 - alpha) * noise

        if self.predict_gaussian_noise:
            return y, noise_gaussian
        else:
            return y, noise

    def get_loss(
        self,
        model: Callable,
        x: TensorType[..., 'dim'],
        **kwargs,
    ) -> TensorType[..., 'dim']:

        i = torch.randint(0, self.num_steps, size=(x.shape[0],))
        i = i.view(-1, *(1,) * len(x.shape[1:])).expand_as(x[...,:1]).to(x)

        x_noisy, noise = self.forward(x, i, **kwargs)

        pred_noise = model(x_noisy, i=i, **kwargs)
        loss = (pred_noise - noise)**2

        return loss

    @torch.no_grad()
    def sample(
        self,
        model: Callable,
        num_samples: Union[int, Tuple],
        device: str = 'cpu',
        **kwargs,
    ) -> TensorType['*num_samples', 'dim']:
        if isinstance(num_samples, int):
            num_samples = (num_samples,)

        x = self.noise(*num_samples, **kwargs).to(device)

        if self.is_time_series and self.predict_gaussian_noise:
            cov = self.noise.covariance(**kwargs)
            L = torch.linalg.cholesky(cov)
        else:
            L = None

        for diff_step in reversed(range(0, self.num_steps)):
            alpha = self.alphas[diff_step]
            beta = self.betas[diff_step]

            # An alternative can be:
            # alpha_prev = self.alphas[diff_step - 1]
            # sigma = beta * (1 - alpha_prev) / (1 - alpha)
            sigma = beta

            if diff_step == 0:
                z = 0
            else:
                z = self.noise(*num_samples, **kwargs).to(device)

            i = torch.Tensor([diff_step]).expand_as(x[...,:1]).to(device)
            pred_noise = model(x, i=i, **kwargs)

            if L is not None:
                pred_noise = L @ pred_noise

            x = (x - beta * pred_noise / (1 - alpha).sqrt()) / (1 - beta).sqrt() + sigma.sqrt() * z

        return x

    @torch.no_grad()
    def log_prob(
        self,
        model: Callable,
        x: TensorType[..., 'dim'],
        num_samples: int = 1,
        **kwargs,
    ) -> TensorType[..., 1]:
        if self.is_time_series and self.predict_gaussian_noise:
            cov = self.noise.covariance(**kwargs)
            L = torch.linalg.cholesky(cov)
        else:
            L = None

        func = self._elbo_parallel if self.parallel_elbo else self._elbo_sequential
        return func(model, x, num_samples=num_samples, L=L, **kwargs)

    def _elbo_parallel(
        self,
        model: Callable,
        x: TensorType[..., 'dim'],
        L: TensorType[..., 'seq_len', 'seq_len'],
        num_samples: int = 1,
        **kwargs,
    ) -> TensorType[..., 1]:
        """
        Computes ELBO over all diffusion steps in parallel,
        then averages over `num_samples` runs.
        If diffusion `num_steps` large (and `num_samples` small)
        it will be heavy on the GPU memory.

        Args:
            model: Denoising diffusion model
            x: Clean input data
            num_samples: How many times to compute ELBO, final
                result is averaged over all ELBO samples
            **kwargs: Can be time, latent etc. depending on a model
        """
        elbo = 0

        i = expand_to_x(torch.arange(self.num_steps), x).expand(-1, *x[...,:1].shape).contiguous()
        alphas = expand_to_x(self.alphas, x)
        betas = expand_to_x(self.betas, x)

        xt, kwargs = expand_x_and_kwargs(x, kwargs, self.num_steps)

        for _ in range(num_samples):
            # Get diffused outputs
            xt, _ = self.forward(x, i, **kwargs) # [num_steps, ..., dim]

            # Output predicted noise
            epsilon = model(xt, i=i, **kwargs)

            if L is not None:
                epsilon = L @ epsilon

            # p(x_{t-1} | p_t)
            p_mu = get_p_mu(xt, betas, alphas, epsilon)
            px = td.Independent(td.Normal(p_mu[1:], betas[1:].sqrt()), 1)

            # p(x_0 | x_1)
            log_prob_x0_x1 = td.Independent(td.Normal(p_mu[0], betas[0].sqrt()), 1).log_prob(x)
            assert log_prob_x0_x1.shape == x.shape[:-1]

            # q(x_{t-1} | x_0, x_t), t > 1
            qx = get_qx(x.unsqueeze(0), xt[1:], alphas[1:], alphas[:-1], betas[1:])

            # KL[q(x_{t-1} | p_t) || p(x_{t-1} | p_t)]
            kl_q_p = td.kl_divergence(qx, px).sum(0)
            assert kl_q_p.shape == x.shape[:-1]

            # ELBO
            elbo_contribution = (log_prob_x0_x1 - kl_q_p) / num_samples
            elbo += elbo_contribution

        elbo = reduce_elbo(elbo, x)
        return elbo

    def _elbo_sequential(
        self,
        model: Callable,
        x: TensorType[..., 'dim'],
        L: TensorType[..., 'seq_len', 'seq_len'],
        num_samples: int = 1,
        **kwargs,
    ) -> TensorType[..., 1]:
        """
        Computes ELBO as a sum of diffusion steps - sequentially.

        Args:
            model: Denoising diffusion model
            x: Clean input data
            num_samples: How many times to compute ELBO, final
                result is averaged over all ELBO samples
            **kwargs: Can be time, latent etc. depending on a model
        """
        elbo = 0

        x, kwargs = expand_x_and_kwargs(x, kwargs, num_samples)

        for i in range(self.num_steps):
            # Prepare variables
            beta = self.betas[i].to(x)
            alpha = self.alphas[i].to(x)
            step = torch.Tensor([i]).expand_as(x[...,:1]).to(x)

            # Diffuse and predict noise
            xt, _ = self.forward(x, i=step, **kwargs)
            epsilon = model(xt, i=step, **kwargs)

            if L is not None:
                epsilon = L @ epsilon

            assert xt.shape == x.shape == epsilon.shape

            # p(x_{t-1} | p_t)
            p_mu = get_p_mu(xt, beta, alpha, epsilon)
            px = td.Independent(td.Normal(p_mu, beta.sqrt()), 1)

            if i == 0:
                elbo = elbo + px.log_prob(x).mean(0)
            else:
                prev_alpha = self.alphas[i - 1]

                # q(x_{t-1} | x_0, x_t), t > 1
                qx = get_qx(x, xt, alpha, prev_alpha, beta)

                # KL[q(x_{t-1} | p_t) || p(x_{t-1} | p_t)]
                kl = td.kl_divergence(qx, px).mean(0)
                elbo = elbo - kl

        elbo = reduce_elbo(elbo, x)
        return elbo


class GaussianDiffusion(DiscreteDiffusion):
    """ Discrete diffusion with Gaussian noise """
    def __init__(self, dim: int, num_steps: int, beta_fn: Callable, **kwargs):
        super().__init__(dim, num_steps, beta_fn, noise_fn=Normal(dim), **kwargs)


class OUDiffusion(DiscreteDiffusion):
    """ Discrete diffusion with noise coming from an OU process """
    def __init__(
        self,
        dim: int,
        num_steps: int,
        beta_fn: Callable,
        predict_gaussian_noise: bool,
        theta: float = 0.5,
        **kwargs,
    ):
        super().__init__(
            dim=dim,
            num_steps=num_steps,
            beta_fn=beta_fn,
            noise_fn=OrnsteinUhlenbeck(dim, theta=theta),
            is_time_series=True,
            predict_gaussian_noise=predict_gaussian_noise,
            **kwargs,
        )


class GPDiffusion(DiscreteDiffusion):
    """ Discrete diffusion with noise coming from a Gaussian process """
    def __init__(
        self,
        dim: int,
        num_steps: int,
        beta_fn: Callable,
        predict_gaussian_noise: bool,
        sigma: float = 0.1,
        **kwargs,
    ):
        super().__init__(
            dim=dim,
            num_steps=num_steps,
            beta_fn=beta_fn,
            noise_fn=GaussianProcess(dim, sigma=sigma),
            is_time_series=True,
            predict_gaussian_noise=predict_gaussian_noise,
            **kwargs,
        )


def expand_to_x(inputs, x):
    return inputs.view(-1, *(1,) * len(x.shape)).to(x)

def expand_x_and_kwargs(x, kwargs, N):
    # Expand dimensions
    x = x.unsqueeze(0).repeat_interleave(N, dim=0)

    # A hacky solution to repeat dimensions in all kwargs (latent, t, etc.)
    for key, value in kwargs.items():
        if torch.is_tensor(value):
            kwargs[key] = value.unsqueeze(0).repeat_interleave(N, dim=0)

    return x, kwargs

def reduce_elbo(
    elbo: TensorType['batch', Any],
    x: TensorType[Any],
) -> TensorType['batch', 1]:
    # Reduce ELBO over all but batch dimension: (B, ...) -> (B,)
    elbo = elbo.view(elbo.shape[0], -1).sum(1)

    if len(x.shape) > 2:
        elbo = elbo / x.shape[-2]

    return elbo.unsqueeze(1)

def get_p_mu(xt, beta, alpha, epsilon):
    mu = 1 / (1 - beta).sqrt() * (xt - beta / (1 - alpha).sqrt() * epsilon)
    return mu

def get_qx(x, xt, alpha, prev_alpha, beta):
    q_mu_1 = torch.sqrt(prev_alpha) * beta / (1 - alpha) * x
    q_mu_2 = torch.sqrt(1 - beta) * (1 - prev_alpha) / (1 - alpha) * xt
    q_mu = q_mu_1 + q_mu_2

    q_sigma = beta * (1 - prev_alpha) / (1 - alpha)

    qx = td.Independent(td.Normal(q_mu, q_sigma.expand_as(q_mu).sqrt()), 1)
    return qx
