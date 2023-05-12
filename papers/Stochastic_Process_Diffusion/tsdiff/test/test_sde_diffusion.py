import pytest
import numpy as np
import torch
import torchsde

from tsdiff.diffusion import (
    GaussianDiffusion,
    OUDiffusion,
    ContinuousGaussianDiffusion,
    ContinuousGPDiffusion,
    ContinuousOUDiffusion,
)
from tsdiff.diffusion.beta_scheduler import BetaLinear
from tsdiff.diffusion.noise import OrnsteinUhlenbeck

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def test_sde_diffusion_as_ddpm():
    num_steps = 1000
    step = 300
    N = 100_000
    beta_start, beta_end = 1e-4, 2 / num_steps * 10

    x = torch.randn(N, 1).square() + 1
    i = torch.Tensor([step]).repeat(N).unsqueeze(-1)

    beta_fn = BetaLinear(beta_start, beta_end)
    rescaled_beta_fn = BetaLinear(beta_start / num_steps, beta_end / num_steps)

    ddpm = GaussianDiffusion(dim=1, beta_fn=beta_fn, num_steps=num_steps, predict_gaussian_noise=False)
    sdediff = ContinuousGaussianDiffusion(dim=1, beta_fn=rescaled_beta_fn, predict_gaussian_noise=False)

    y1, _ = ddpm(x, i)
    y2, _ = sdediff(x, i)

    assert torch.allclose(y1.mean(), y2.mean(), atol=0.1)
    assert torch.allclose(y1.std(), y2.std(), atol=0.1)


def test_sde_ou_diffusion_as_ddpm():
    num_steps = 1000
    step = 300
    N = 100_000
    beta_start, beta_end = 1e-4, 2 / num_steps * 10

    x = torch.randn(N, 10, 1).square() + 1
    t = torch.linspace(0, 1, 10).view(1, -1, 1).repeat(N, 1, 1)
    i = torch.Tensor([step]).view(1, 1, 1).repeat(N, 10, 1)

    beta_fn = BetaLinear(beta_start, beta_end)
    rescaled_beta_fn = BetaLinear(beta_start / num_steps, beta_end / num_steps)

    ddpm = OUDiffusion(dim=1, beta_fn=beta_fn, num_steps=num_steps, predict_gaussian_noise=False)
    sdediff = ContinuousOUDiffusion(dim=1, beta_fn=rescaled_beta_fn, t1=num_steps, predict_gaussian_noise=False)

    y1, _ = ddpm(x, t=t, i=i)
    y2, _ = sdediff(x, t=t, i=i)

    assert torch.allclose(y1.mean(0), y2.mean(0), atol=0.1)
    assert torch.allclose(y1.std(0), y2.std(0), atol=0.1)


def test_sde_diffusion_score():
    torch.manual_seed(123)

    N = 10_000

    x = torch.randn(1).repeat(N).unsqueeze(-1).square() + 1
    i = torch.Tensor([0.3]).repeat(N).unsqueeze(-1)

    sdediff = ContinuousGaussianDiffusion(dim=1, beta_fn=BetaLinear(0, 10))

    y, noise, mean, std, _ = sdediff(x, i, _return_all=True, predict_gaussian_noise=False)
    score = -noise / std
    y = y.requires_grad_(True)

    mean = mean.unique()
    std = std.unique()

    assert len(mean) == 1 and len(std) == 1
    assert torch.allclose(y.mean(), mean, atol=0.05)
    assert torch.allclose(y.std(), std, atol=0.05)

    dist = torch.distributions.MultivariateNormal(mean, std.unsqueeze(-1).square())
    log_prob = dist.log_prob(y)

    true_score = torch.autograd.grad(log_prob.sum(), y)[0]

    assert torch.allclose(true_score, score, atol=1e-4)


@pytest.mark.parametrize('Diffusion', [ContinuousGPDiffusion, ContinuousOUDiffusion])
@pytest.mark.parametrize('predict_gaussian_noise', [True, False])
def test_sde_time_diffusion_score(Diffusion, predict_gaussian_noise):
    np.random.seed(132)
    torch.manual_seed(123)

    N = 10_000
    T = 10
    t1 = 1

    x = torch.randn(1, T, 1).repeat(N, 1, 1).square().to(device) + 1
    t = torch.linspace(0, 1, T).view(1, -1, 1).repeat(N, 1, 1).to(device)

    class SDE(torch.nn.Module):
        sde_type = 'ito'
        noise_type = 'general'

        def __init__(self, beta_fn, cov):
            super().__init__()
            self.beta_fn = beta_fn
            self.L = torch.linalg.cholesky(cov)

        def f(self, t, x):
            return -0.5 * self.beta_fn(t) * x

        def g(self, t, x):
            return self.beta_fn(t).sqrt() * self.L


    for ratio in [0.01, 0.3, 0.9]:
        i = torch.Tensor([ratio * t1]).view(1, 1, 1).repeat(N, T, 1).to(device)

        beta_fn = BetaLinear(0.1, 10)
        sdediff = Diffusion(
            dim=1,
            t1=t1,
            beta_fn=beta_fn,
            predict_gaussian_noise=predict_gaussian_noise,
        ).to(device)

        y, diff_noise, diff_mean, diff_std, diff_cov = sdediff(x, t=t, i=i, _return_all=True)
        L = torch.linalg.cholesky(diff_cov[0])

        y = y.requires_grad_(True)

        assert len(diff_mean.unique()) == 10
        assert torch.all(diff_cov[0] == diff_cov[1])

        diff_mean = diff_mean[0,:,0]
        diff_cov_tilde = (diff_cov * diff_std**2)[0]

        def model(*args, noise=None, **kwargs):
            # "Fake" model that perfectly predicts the noise
            # In case `predict_gaussian_noise=True`, undo covariance
            if predict_gaussian_noise:
                noise = torch.linalg.inv(L) @ noise
            return noise

        model_score = sdediff._get_score(model, x, i=i, t=t, L=L, noise=diff_noise)

        # Statistics from diffusion function vs. empirical covariance of diffused values
        empirical_diff_mean = y.mean([0, 2])
        empirical_diff_cov = torch.cov(y.squeeze(-1).T)
        assert torch.allclose(empirical_diff_mean, diff_mean, atol=0.05)
        assert torch.allclose(empirical_diff_cov, diff_cov_tilde, atol=0.05)

        # Empirical score vs. score from diffusion function
        diff_dist = torch.distributions.MultivariateNormal(diff_mean, diff_cov_tilde)
        log_prob = diff_dist.log_prob(y.squeeze(-1))
        empirical_diff_score = torch.autograd.grad(log_prob.sum(), y)[0]
        assert torch.allclose(empirical_diff_score, model_score, atol=0.05, rtol=0.05)

        # True theoretical covariance vs. calculated in diffusion function
        time_cov = sdediff.noise.covariance(t)
        assert torch.allclose(time_cov[0], diff_cov, atol=0.05)
        true_cov = (1 - torch.exp(-beta_fn.integral(i[0][0]))) * time_cov[0]
        assert torch.allclose(true_cov, diff_cov_tilde, atol=0.05)

        # SDE sequential computation vs. direct covariance computation
        sde = SDE(beta_fn=beta_fn, cov=time_cov)
        times = torch.Tensor([0, ratio * t1]).to(y)
        with torch.no_grad():
            true_y = torchsde.sdeint(sde, x.squeeze(-1), times, dt=5e-4)[-1]
        sde_cov = torch.cov(true_y.T)
        assert torch.allclose(true_cov, sde_cov, atol=0.05)
