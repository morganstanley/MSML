
import torch
from torch import Tensor


def sample_rademacher_like(y):
    return torch.randint(low=0, high=2, size=y.shape).to(y) * 2 - 1

def sample_gaussian_like(y):
    return torch.randn_like(y)


def compute_div_gz(dyn, ts, xs, policy, return_zs, cond=None, sample_type='gaussian'):
    zs = policy(xs, ts, cond=cond)

    g_ts = dyn.g(ts)
    g_ts = g_ts.view(-1, *[1]*(zs.ndim - 1))

    gzs = g_ts * zs

    if sample_type == 'gaussian':
        e = sample_gaussian_like(xs)
    elif sample_type == 'rademacher':
        e = sample_rademacher_like(xs)

    e_dzdx = torch.autograd.grad(gzs, xs, e, create_graph=True)[0]
    div_gz = e_dzdx * e

    assert div_gz.shape == xs.shape
    return [div_gz, zs] if return_zs else div_gz


def compute_sb_nll_alternate_train(dyn, ts, xs, zs_impt, policy_opt, cond=None) -> Tensor: # [B, T, D]
    """ Implementation of Eq (18,19) in our main paper.
    """
    assert xs.requires_grad
    assert not zs_impt.requires_grad

    with torch.enable_grad():
        div_gz, zs = compute_div_gz(dyn, ts, xs, policy_opt, cond=cond, return_zs=True)
        loss = zs * (0.5 * zs + zs_impt) + div_gz
    return loss
