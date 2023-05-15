import pytest
import torch

from tsdiff.diffusion.beta_scheduler import BetaLinear


def test_linear():
    torch.manual_seed(123)

    f = BetaLinear(start=6, end=-3)

    t = torch.linspace(0, 1, 100).view(1, -1, 1).repeat(10, 1, 4)
    t = t.requires_grad_(True)

    beta = f(t)

    # Check boundaries
    assert (beta[:,0] == 6).all()
    assert (beta[:,-1] == -3).all()

    # Check integral
    beta_int = f.integral(t)

    beta_int_derivative = torch.autograd.grad(beta_int.sum(), t)[0]
    assert torch.allclose(beta, beta_int_derivative, atol=1e-6)
