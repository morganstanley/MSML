import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint
from torchsde import sdeint
from pathlib import Path

DATA_DIR = Path(__file__).parents[2] / 'data/synthetic'
DATA_DIR.mkdir(parents=True, exist_ok=True)

def generate_OU(N=10_000, mu=0.02, theta=0.1, sigma=0.4, regular=True):
    class OU(nn.Module):
        noise_type = 'diagonal'
        sde_type = 'ito'

        def __init__(self, mu, theta, sigma):
            super().__init__()
            self.mu = mu
            self.theta = theta
            self.sigma = sigma

        # Drift
        def f(self, t, y):
            return self.mu * t - self.theta * y

        # Diffusion
        def g(self, t, y):
            return self.sigma * torch.ones_like(y).to(y)

    f = OU(mu, theta, sigma)
    if regular:
        t = torch.linspace(0, 63, 64)
        x0 = torch.randn(N, 1)

        with torch.no_grad():
            x = sdeint(f, x0, t, dt=0.1).transpose(0, 1)

        t = t.view(1, -1, 1).expand_as(x[...,:1])

        np.savez(DATA_DIR / 'ou.npz', t=t.numpy(), x=x.numpy())
    else:
        np.savez(DATA_DIR / 'ou_irregular.npz', t=t.numpy(), x=x.numpy())


def generate_CIR(N=10_000, a=1, b=1.2, sigma=0.2, regular=True):
    class CIR(nn.Module):
        """ Cox-Ingersoll-Ross """
        noise_type = 'diagonal'
        sde_type = 'ito'

        def __init__(self, a, b, sigma):
            super().__init__()
            self.a = a
            self.b = b
            self.sigma = sigma

        def f(self, t, y):
            return self.a * (self.b - y)

        def g(self, t, y):
            return self.sigma * y.sqrt()

    f = CIR(a, b, sigma)
    if regular:
        t = torch.linspace(0, 63, 64)
        x0 = torch.randn(N, 1).abs()

        with torch.no_grad():
            x = sdeint(f, x0, t, dt=0.1).transpose(0, 1)

        t = t.view(1, -1, 1).expand_as(x[...,:1])

        np.savez(DATA_DIR / 'cir.npz', t=t.numpy(), x=x.numpy())
    else:
        np.savez(DATA_DIR / 'cir_irregular.npz', t=t.numpy(), x=x.numpy())


def generate_lorenz(N=10_000, rho=28, sigma=10, beta=2.667, regular=True):
    class Lorenz(nn.Module):
        def __init__(self, rho, sigma, beta):
            super().__init__()
            self.rho = rho
            self.sigma = sigma
            self.beta = beta

        def forward(self, t, inp):
            x, y, z = inp.chunk(3, dim=-1)
            dx = self.sigma * (y - x)
            dy = x * self.rho - y - x * z
            dz = x * y - self.beta * z
            d_inp = torch.cat([dx, dy, dz], -1)
            return d_inp

    f = Lorenz(rho, sigma, beta)
    if regular:
        t = torch.linspace(0, 2, 100)
        x0 = torch.randn(N, 3) * 10

        with torch.no_grad():
            x = odeint(f, x0, t, method='dopri5').transpose(0, 1)
        t = t.view(1, -1, 1).expand_as(x[...,:1])

        np.savez(DATA_DIR / 'lorenz.npz', t=t.numpy(), x=x.numpy())
    else:
        np.savez(DATA_DIR / 'lorenz_irregular.npz', t=t.numpy(), x=x.numpy())


def generate_sine(N=10_000, regular=True):
    a = torch.rand(N, 1, 5) + 3
    b = torch.rand(N, 1, 5) * 0.5
    c = torch.rand(N, 1, 5)

    if regular:
        t = torch.linspace(0, 10, 100).view(1, -1, 1).repeat(N, 1, 1)
        x = (c * torch.sin(a * t + b)).sum(-1, keepdim=True)

        np.savez(DATA_DIR / 'sine.npz', t=t.numpy(), x=x.numpy())
    else:
        np.savez(DATA_DIR / 'sine_irregular.npz', t=t.numpy(), x=x.numpy())


def generate_predator_prey(N=10_000, regular=True):
    class PredatorPrey(nn.Module):
        def forward(self, t, y):
            y1, y2 = y.chunk(2, dim=-1)
            dy = torch.cat([
                2/3 * y1 - 2/3 * y1 * y2,
                y1 * y2 - y2,
            ], -1)
            return dy

    f = PredatorPrey()
    if regular:
        t = torch.linspace(0, 10, 64)
        x0 = torch.rand(N, 2)

        with torch.no_grad():
            x = odeint(f, x0, t, method='dopri5').transpose(0, 1)

        t = t.view(1, -1, 1).expand_as(x[...,:1])

        np.savez(DATA_DIR / 'predator_prey.npz', t=t.numpy(), x=x.numpy())
    else:
        np.savez(DATA_DIR / 'predator_prey_irregular.npz', t=t.numpy(), x=x.numpy())

def generate_sink(N=10_000, regular=True):
    class Sink(nn.Module):
        def forward(self, t, y):
            A = torch.Tensor([[-4, 10], [-3, 2]]).to(y)
            return y @ A

    f = Sink()
    if regular:
        t = torch.linspace(0, 3, 64)
        x0 = torch.rand(N, 2)

        with torch.no_grad():
            x = odeint(f, x0, t, method='dopri5').transpose(0, 1)

        t = t.view(1, -1, 1).expand_as(x[...,:1])

        np.savez(DATA_DIR / 'sink.npz', t=t.numpy(), x=x.numpy())
    else:
        np.savez(DATA_DIR / 'sink_irregular.npz', t=t.numpy(), x=x.numpy())



if __name__ == '__main__':
    generate_OU()
    generate_CIR()
    generate_sine()
    generate_lorenz()
    generate_predator_prey()
    generate_sink()
