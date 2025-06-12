from typing import Callable, Tuple
import numpy as np
import abc
from functools import partial
import torch
from torch import Tensor


def _assert_increasing(name, ts):
    assert (ts[1:] > ts[:-1]).all(), '{} must be strictly increasing'.format(name)

def build(t0, T, beta_min, beta_max, beta_r, interval, p, q):
    return VPSDE(t0, T, beta_min, beta_max, beta_r, interval, p, q)


class BaseSDE(metaclass=abc.ABCMeta):
    def __init__(self, t0, T, beta_min, beta_max, beta_r, interval, p, q):
        self.t0 = t0
        self.dt = T / interval
        self.p = p # data distribution
        self.q = q # prior distribution

        self.beta_min = beta_min
        self.beta_max = beta_max
        self.beta_r = beta_r

    @abc.abstractmethod
    def _f(self, x, t):
        raise NotImplementedError

    @abc.abstractmethod
    def _g(self, x, t):
        raise NotImplementedError

    def f(self, x, t, direction):
        sign = 1. if direction=='forward' else -1.
        return sign * self._f(x, t)

    def g(self, t):
        return self._g(t)

    def dw(self, x, dt=None):
        dt = self.dt if dt is None else dt
        return torch.randn_like(x)*np.sqrt(dt)

    def propagate(self, t, x, z, direction, f=None, dw=None, dt=None):
        g = self.g(t)
        f = self.f(x, t, direction) if f is None else f
        dt = self.dt if dt is None else dt
        dw = self.dw(x,dt) if dw is None else dw
        return x + (f + g*z)*dt + g*dw

    def propagate_ode(self, t, x, z, z_f, direction, f=None, dt=None):
        g = self.g(t)
        f = self.f(x, t, direction) if f is None else f
        dt = self.dt if dt is None else dt
        dsm_score = z + z_f # map fb-sde score to dsm score
        return x + (f - g * z_f + 0.5*g*dsm_score)*dt

    def sample_traj(self, x, ts, cond, policy, save_traj=True, policy_f=None, cov=None, **kwargs):
        # first we need to know whether we're doing forward or backward sampling
        direction = policy.direction
        # assert direction in ['forward', 'backward']
        assert direction == 'backward'

        if direction == 'backward' and cov is not None:
            dist = torch.distributions.MultivariateNormal(torch.zeros(x.shape[-1]).to(x), cov)
            x = dist.sample(x.shape[:-1]).to(x)

        # set up ts and init_distribution
        _assert_increasing('ts', ts)
        ts = ts if direction=='forward' else torch.flip(ts,dims=[0])

        xs = torch.empty((x.shape[0], len(ts), *x.shape[1:])).to(x) if save_traj else None
        zs = torch.empty_like(xs) if save_traj else None

        for idx, t in enumerate(ts):
            t_idx = idx if direction == 'forward' else len(ts)-idx-1

            f = self.f(x, t, direction)
            z = policy(x=x, t=t, cond=cond)

            z_f = policy_f(x, t) if policy_f != None else None

            dw = self.dw(x)

            if save_traj:
                xs[:,t_idx,...] = x
                zs[:,t_idx,...] = z

            if policy_f != None and direction == 'backward':
                x = self.propagate_ode(t, x, z, z_f, direction, f=f)
            else:
                x = self.propagate(t, x, z, direction, f=f, dw=dw)

        x_term = x

        res = [xs, zs, x_term]
        return res


class VPSDE(BaseSDE):
    def __init__(self, t0, T, beta_min, beta_max, beta_r, interval, p, q):
        super(VPSDE,self).__init__(t0, T, beta_min, beta_max, beta_r, interval, p, q)

    def _f(self, x, t):
        return compute_vp_drift_coef(t, self.beta_min, self.beta_max, self.beta_r) * x

    def _g(self, t):
        return compute_vp_diffusion(t, self.beta_min, self.beta_max, self.beta_r)

####################################################
##  Implementation of SDE analytic kernel         ##
##  Ref: https://arxiv.org/pdf/2011.13456v2.pdf,  ##
##       page 15-16, Eq (30,32,33)                ##
####################################################


""" Generalized Song, Yang linear schedule to non-linear """
def compute_vp_diffusion(t, b_min, b_max, b_r=1., T=1.):
    return torch.sqrt(b_min+(t/T)**b_r*(b_max-b_min))


def compute_vp_drift_coef(t, b_min, b_max, b_r=1.):
    g = compute_vp_diffusion(t, b_min, b_max, b_r)
    return -0.5 * g**2

def compute_vp_kernel_mean_scale(t, b_min, b_max, b_r=1.):
    return torch.exp(-0.5/(b_r+1)*t**(b_r+1)*(b_max-b_min)-0.5*t*b_min)

def compute_vp_kernel_mean_scale_diag_matrix(t, mat_b_min, mat_b_max, b_r=1.):
    mean_scale = torch.exp(-0.5/(b_r+1)*torch.einsum('t,ij->tij', t**(b_r+1), mat_b_max-mat_b_min)-0.5*torch.einsum('t,ij->tij', t, mat_b_min))
    mean_scale[:, 1, 0] = 0.
    mean_scale[:, 0, 1] = 0.
    return mean_scale

def compute_vp_kernel_mean_scale_matrix(t, D, b_min, b_max, b_r=1.):
    time_scalars = -0.5/(b_r+1)*t**(b_r+1)*(b_max-b_min)-0.5*t*b_min
    time_vary_D = torch.einsum('t,ij->tij', time_scalars, D)
    return torch.linalg.matrix_exp(time_vary_D)

def compute_variance(ts, D, b_min, b_max, b_r=1.):
    dim = D.shape[0]
    C_H_power = torch.block_diag(-D, D.t())
    C_H_power[:dim, dim:] = 2. * torch.eye(dim).to(ts)
    integrate_beta = 0.5 / (b_r + 1)*ts**(b_r + 1) * (b_max - b_min) + 0.5 * ts * b_min

    C_H_pair = torch.linalg.matrix_exp(torch.einsum('t,ij->tij', integrate_beta, C_H_power))
    Initial_Matrix = torch.cat((torch.zeros_like(torch.eye(dim)), torch.eye(dim)), dim=0).to(ts)

    C_H = torch.einsum('tij,jk->tik', C_H_pair, Initial_Matrix)
    C = C_H[:, : dim, :]
    H = C_H[:, dim: , :]
    Covariance = torch.einsum('tij,tjk->tik', C, torch.linalg.inv(H))
    L = torch.linalg.cholesky(Covariance)
    invL = torch.linalg.inv(L.mH)
    return Covariance, L, invL


def compute_vp_xs_label_matrix(
    x0: Tensor, # [B, T, D]
    sqrt_betas: Tensor, # [N]
    mean_scales: Tensor, # [N, D, D]
    D: Tensor, # [D, D]
    L: Tensor, # [N, D, D]
    invL: Tensor, # [N, D, D]
    ind: Tensor, # [B]
) -> Tuple[Tensor, Tensor]:
    noise = torch.randn_like(x0)
    mean_scale_t = mean_scales[ind]
    L_t = L[ind]
    invL_t = invL[ind]

    analytic_xs = noise @ L_t.transpose(1, 2) + x0 @ mean_scale_t.transpose(1, 2)
    part_label = -noise @ invL_t.transpose(1, 2)
    sqrt_beta_t = sqrt_betas[ind].view(-1, 1, 1)

    label = part_label * sqrt_beta_t
    I_minus_D = 0.5 * (torch.eye(x0.shape[-1]).to(x0) - D)
    label = label - sqrt_beta_t * (analytic_xs @ I_minus_D.T)

    return analytic_xs, label


def compute_vp_xs_label(x0, sqrt_betas, mean_scales, ind):
    noise = torch.randn_like(x0)
    mean_scale_t = mean_scales[ind].view(-1, 1, 1)
    std_t = torch.sqrt(1 - mean_scale_t**2)

    analytic_xs = std_t * noise + mean_scale_t * x0

    sqrt_beta_t = sqrt_betas[ind].view(-1, 1, 1)
    label = -noise / std_t * sqrt_beta_t

    return analytic_xs, label

def get_xs_label_computer(
    beta_min: float,
    beta_max: float,
    beta_r: float,
    ts: Tensor,
    A: Tensor,
    dsm_baseline: bool = True,
) -> Callable:
    sqrt_betas = compute_vp_diffusion(ts, beta_min, beta_max, beta_r)

    D = torch.eye(A.shape[0]).to(A) - 2 * A

    if dsm_baseline:
        mean_scales = compute_vp_kernel_mean_scale(ts, beta_min, beta_max, beta_r)
        fn = compute_vp_xs_label
        kwargs = dict(sqrt_betas=sqrt_betas, mean_scales=mean_scales)
    else:
        mean_scales = compute_vp_kernel_mean_scale_matrix(ts, D, beta_min, beta_max, beta_r)
        _, L, invL = compute_variance(ts, D, beta_min, beta_max, beta_r)
        fn = compute_vp_xs_label_matrix
        kwargs = dict(
            sqrt_betas=sqrt_betas,
            mean_scales=mean_scales,
            D=D,
            L=L,
            invL=invL,
        )

    return partial(fn, **kwargs)
