"""Helper functions for the unit hypercube [0, 1]^D"""
import torch
from math import pi

def unsqueeze_as(x, y, back=True):
    """
    Unsqueeze x to have as many dimensions as y. For example, tensor shapes:

    x: (a, b, c), y: (a, b, c, d, e) -> output: (a, b, c, 1, 1)
    """
    if back:
        return x.view(*x.shape, *((1,) * (len(y.shape) - len(x.shape))))
    else:
        return x.view(*((1,) * (len(y.shape) - len(x.shape))), *x.shape)


def inside(x):
    """
    Checks if x is inside the unit hypercube, batchwise

    Args
    ----
    x (Tensor):
        input of shape [B, ...]
    
    Returns
    -------
    an output Tensor of shape [B] correpsonding to if each x[i] is in the cube
    """
    x = x.flatten(1)
    return torch.logical_and(x >= 0, x <= 1).all(dim=-1)


def reflect(x):
    """
    Performs reflections until x is inside the domain.

    Args
    ----
    x (Tensor):
        input of shape [B, ...]

    Returns
    -------
    an output Tensor with the same shape as x which is the "reflected"-inside version.
    """
    xm2 = x % 2  # NOTE: python modulo always returns same sign as divisor.
    xm2[xm2 > 1] = 2 - xm2[xm2 > 1]
    return xm2


def sample_hk(x, sigma):
    """
    Sample from heat kernel starting at point x with coefficient sigma.

    Args
    ----
    x (Tensor):
        input of shape [B, ...]. Corresponds to the pseudo-"mean" or "starting point".
    sigma (Tensor):
        input of shape [B]. Corresponds to the std dev of the underlying Gaussian
            or t^2/2 where t is the time of the heat equation PDE.
    Returns
    -------
    an output Tensor with the same shape as x corresponding to a random sample.
    """
    if not torch.is_tensor(sigma):
        sigma = sigma * torch.ones(x.shape[0]).to(x)
    samples_gauss = torch.randn_like(x) * unsqueeze_as(sigma, x) + x
    return reflect(samples_gauss)


def _score_hk_ef(x, x_orig, t, efs=20):
    """
    Computes the score of the heat kernel using eigenfunctions.

    Args
    ----
    x (Tensor):
        shape [B, ...]. Corresponds to the sampled point.
    x_orig (Tensor):
        shape [B, ...] same as x. Corresponds to the origin/pseudo-mean.
    t (Tensor):
        shape [B]. Time of the heat equation PDE.
    efs (int):
        number of eigenfunctions to compute with

    Returns
    -------
    an output tensor of the same shape as x corresponding to the score of the heat kernel.
    """
    eval_range = torch.arange(1, efs + 1).to(x)

    x_rescaled = pi * x.unsqueeze(0) * unsqueeze_as(eval_range, x.unsqueeze(0))
    x_orig_rescaled = pi * x_orig.unsqueeze(0) * unsqueeze_as(eval_range, x_orig.unsqueeze(0))

    x_sin = x_rescaled.sin()
    x_cos = x_rescaled.cos()
    x_orig_cos = x_orig_rescaled.cos()

    e_powers_denom = (-t.unsqueeze(0) * eval_range.unsqueeze(-1).pow(2) * (pi ** 2)).exp()
    e_powers_num = e_powers_denom * eval_range.unsqueeze(-1)

    num = - 2 * pi * (unsqueeze_as(e_powers_num, x_sin) * (x_sin * x_orig_cos)).sum(0)
    denom = 1 + 2 * (unsqueeze_as(e_powers_denom, x_sin) * (x_cos * x_orig_cos)).sum(0)

    return (num / (denom + 1e-12))
    

def _score_hk_refl(x, x_orig, t, refls=2):
    """
    Computes the score of the heat kernel using reflection.

    Args
    ----
    x (Tensor):
        shape [B, ...]. Corresponds to the sampled point.
    x_orig (Tensor):
        shape [B, ...] same as x. Corresponds to the origin/pseudo-mean.
    t (Tensor):
        shape [B]. Time of the heat flow PDE.
    refls (int):
        number of reflections to sum up.

    Returns
    -------
    an output tensor of the same shape as x corresponding to the score of the heat kernel.
    """
    refls = torch.arange(-2 * refls, 2 * refls + 1, 2).to(x)

    x_refl = torch.cat((
        unsqueeze_as(refls, x.unsqueeze(0)) + x.unsqueeze(0),
        unsqueeze_as(refls, x.unsqueeze(0)) - x.unsqueeze(0)
    ), dim=0)
    refl_sign = torch.cat((torch.ones_like(refls), -torch.ones_like(refls)), dim=0)

    x_minus = x_refl - x_orig.unsqueeze(0)
    fourt = (4 * unsqueeze_as(t.unsqueeze(0), x_minus))

    denom_coeff = - 2 * x_minus / fourt
    e_powers = (- x_minus.pow(2) / fourt).exp()

    num = (denom_coeff * e_powers * unsqueeze_as(refl_sign, e_powers)).sum(0)
    denom = e_powers.sum(0)

    return (num/ (denom + 1e-12))


def score_hk(x, x_orig, sigma, efs=20, refls=10, min_cutoff=1e-2):
    """
    Computes the score of the heat kernel using eigenfunctions.

    Args
    ----
    x (Tensor):
        shape [B, ...]. Corresponds to the sampled point.
    x_orig (Tensor):
        shape [B, ...] same as x. Corresponds to the origin/pseudo-mean.
    sigma (Tensor):
        shape [B]. Std dev of the underlying Guassian
    efs (int):
        see _score_hk_ef
    refls (int):
        see _score_hk_refl
    min_cutoff (float):
        value such that below computes with refls and above with efs
    
    Returns
    -------
    an output tensor of the same shape as x corresponding to the score of the heat kernel.
    """
    t = sigma ** 2 / 2
    if not torch.is_tensor(t):
        t = t * torch.ones(x.shape[0]).to(x)

    ef_cond = t > min_cutoff
    x_ef = x[ef_cond]
    x_orig_ef = x_orig[ef_cond]
    t_ef = t[ef_cond]

    refl_cond = torch.logical_not(ef_cond)
    x_refl = x[refl_cond]
    x_orig_refl = x_orig[refl_cond]
    t_refl = t[refl_cond]

    scores_ef = _score_hk_ef(x_ef, x_orig_ef, t_ef, efs=efs)
    scores_refl = _score_hk_refl(x_refl, x_orig_refl, t_refl, refls=refls)

    scores = torch.zeros_like(x)
    scores[ef_cond] = scores_ef
    scores[refl_cond] = scores_refl

    return scores
