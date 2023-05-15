import pytest
import torch
import torch.nn as nn
from tsdiff.diffusion import GaussianDiffusion, OUDiffusion, GPDiffusion
from tsdiff.diffusion.beta_scheduler import BetaLinear


@pytest.mark.parametrize('input_shape', [(1,), (1, 1), (10, 20, 4), (4, 3, 7, 2)])
def test_shapes(input_shape):
    num_steps = 1000
    x = torch.randn(*input_shape)
    i = torch.randint_like(x, 0, num_steps)

    diffusion = GaussianDiffusion(dim=input_shape[-1], beta_fn=BetaLinear(1e-4, 0.02), num_steps=num_steps)

    y, noise = diffusion(x, i)
    assert not torch.isnan(y).any() and not torch.isnan(noise).any()
    assert y.shape == noise.shape == x.shape


def test_alpha_vs_beta():
    torch.manual_seed(123)

    N = 100_000 # Number of samples
    max_steps = 1000
    steps = 300 # Number of steps after which we calculate statistics

    x = torch.randn(N, 1).square() + 1 # Sample from distribution that is not normal
    i = torch.Tensor([steps]).repeat(N).unsqueeze(-1)

    diffusion = GaussianDiffusion(dim=1, beta_fn=BetaLinear(1e-4, 0.02), num_steps=max_steps)

    y, _ = diffusion(x, i)

    y_ = x.clone()
    for j in range(steps):
        y_ = torch.sqrt(1 - diffusion.betas[j]) * y_ + torch.sqrt(diffusion.betas[j]) * torch.randn_like(y_)

    assert torch.allclose(y.mean(), y_.mean(), atol=0.1)
    assert torch.allclose(y.std(), y_.std(), atol=0.1)


@pytest.mark.parametrize('diffusion', [OUDiffusion, GPDiffusion])
@pytest.mark.parametrize('input_shape', [(1, 1), (1, 1, 1), (10, 20, 4), (4, 3, 7, 2)])
@pytest.mark.parametrize('predict_gaussian_noise', [True, False])
def test_time_diffusion_shapes(diffusion, input_shape, predict_gaussian_noise):
    num_steps = 1000
    x = torch.randn(*input_shape)
    t = torch.rand(*input_shape[:-1], 1)
    i = torch.randint_like(x, 0, num_steps)

    diffusion = diffusion(
        dim=input_shape[-1],
        beta_fn=BetaLinear(1e-4, 0.02),
        predict_gaussian_noise=predict_gaussian_noise,
        num_steps=num_steps,
    )

    y, noise = diffusion(x, i=i, t=t)
    assert not torch.isnan(y).any() and not torch.isnan(noise).any()
    assert y.shape == noise.shape == x.shape


@pytest.mark.parametrize('diffusion', [OUDiffusion, GPDiffusion])
@pytest.mark.parametrize('dim', [1, 3])
def test_time_diffusion_alpha_vs_beta(diffusion, dim):
    torch.manual_seed(123)

    N = 10_000 # Number of samples
    max_steps = 1000
    steps = 300 # Number of steps after which we calculate statistics

    x = torch.randn(N, 10, dim).square() + 1 # Sample from distribution that is not unit normal
    t = torch.linspace(0, 1, 10).view(1, -1, 1).repeat(N, 1, 1)
    i = torch.Tensor([steps]).view(1, 1, 1).repeat(N, 10, 1)

    diffusion = diffusion(
        dim=dim,
        beta_fn=BetaLinear(1e-4, 0.02),
        num_steps=max_steps,
        predict_gaussian_noise=True,
    )

    y, _ = diffusion(x, t=t, i=i)

    y_ = x.clone()
    for j in range(steps):
        y_ = torch.sqrt(1 - diffusion.betas[j]) * y_ + torch.sqrt(diffusion.betas[j]) * diffusion.noise(t=t)

    assert torch.allclose(y.mean(0), y_.mean(0), atol=0.1)
    assert torch.allclose(y.std(0), y_.std(0), atol=0.1)


@pytest.mark.parametrize('diffusion', [GaussianDiffusion, OUDiffusion, GPDiffusion])
@pytest.mark.parametrize('parallel_elbo', [True, False])
@pytest.mark.parametrize('predict_gaussian_noise', [True, False])
def test_latent_inputs(diffusion, parallel_elbo, predict_gaussian_noise):
    torch.manual_seed(123)

    max_steps = 100
    N, T, D, H = 32, 10, 4, 64

    x = torch.randn(N, T, D)
    t = torch.rand(N, T, 1).sort(dim=1)[0]
    latent = torch.randn(N, T, H)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(D + H + 2, H),
                nn.Tanh(),
                nn.Linear(H, D),
            )

        def forward(self, x, t, i, latent):
            x = torch.cat([x, t, i, latent], -1)
            return self.net(x)

    model = Model()
    diffusion = diffusion(
        dim=D,
        beta_fn=BetaLinear(1e-4, 2 / max_steps * 10),
        num_steps=max_steps,
        predict_gaussian_noise=predict_gaussian_noise,
        parallel_elbo=parallel_elbo,
    )

    samples = diffusion.sample(model, num_samples=(N, T), t=t, latent=latent)
    assert samples.shape == (N, T, D)
    assert not torch.any(torch.isnan(samples))

    elbo = diffusion.log_prob(model, x, t=t, latent=latent, num_samples=30)
    assert elbo.shape == (N, 1)
    assert not torch.any(torch.isnan(elbo))

    # Empirical result to catch bigger errors
    elbo = elbo.mean()
    assert elbo > -36 and elbo < -30

    loss = diffusion.get_loss(model, x, t=t, latent=latent)
    assert loss.shape == (N, T, D)
    assert not torch.any(torch.isnan(loss))
