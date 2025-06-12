import numpy as np
import abc
from tqdm import tqdm
from functools import partial
import torch

import util
import loss
from ipdb import set_trace as debug


def _assert_increasing(name, ts):
    assert (ts[1:] > ts[:-1]).all(), '{} must be strictly increasing'.format(name)

def build(opt, p, q):
    print(util.magenta("build base sde..."))
    return VPSDE(opt, p, q)


class BaseSDE(metaclass=abc.ABCMeta):
    def __init__(self, opt, p, q):
        self.opt = opt
        self.dt = opt.T / opt.interval
        self.p = p # data distribution
        self.q = q # prior distribution

        self.b_min = opt.beta_min
        self.b_max = opt.beta_max
        self.b_r = opt.beta_r

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

    def sample_traj(self, ts, policy, corrector=None, apply_trick=True, save_traj=True, adaptive_prior=None, policy_f=None):

        # first we need to know whether we're doing forward or backward sampling
        opt = self.opt
        direction = policy.direction
        assert direction in ['forward', 'backward']

        # set up ts and init_distribution
        _assert_increasing('ts', ts)
        init_dist = self.p if direction=='forward' else self.q
        if direction == 'backward' and adaptive_prior != None:
            print('General prior for backward process.')
            init_dist = adaptive_prior
        ts = ts if direction=='forward' else torch.flip(ts,dims=[0])

        x = init_dist.sample() # [bs, x_dim]

        xs = torch.empty((x.shape[0], len(ts), *x.shape[1:])) if save_traj else None
        zs = torch.empty_like(xs) if save_traj else None

        _ts = tqdm(ts,desc=util.yellow("Propagating Dynamics..."))
        for idx, t in enumerate(_ts):
            _t=t if idx == ts.shape[0] - 1 else ts[idx+1]

            f = self.f(x, t, direction)
            z = policy(x, t)
            z_f = policy_f(x, t) if policy_f != None else None

            dw = self.dw(x)

            t_idx = idx if direction == 'forward' else len(ts)-idx-1
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
    def __init__(self, opt, p, q):
        super(VPSDE,self).__init__(opt, p, q)

    def _f(self, x, t):
        return compute_vp_drift_coef(t, self.b_min, self.b_max, self.b_r) * x

    def _g(self, t):
        return compute_vp_diffusion(t, self.b_min, self.b_max, self.b_r)

####################################################
##  Implementation of SDE analytic kernel         ##
##  Ref: https://arxiv.org/pdf/2011.13456v2.pdf,  ##
##       page 15-16, Eq (30,32,33)                ##
####################################################


""" Generalized Song, Yang's schedule by including a tuning parameter b_r """
def compute_vp_diffusion(t, b_min, b_max, b_r=1., T=1.):
    return torch.sqrt(b_min+(t/T)**b_r*(b_max-b_min))

def compute_vp_drift_coef(t, b_min, b_max, b_r=1.):
    g = compute_vp_diffusion(t, b_min, b_max, b_r)
    return -0.5 * g**2

def compute_vp_kernel_mean_scale(t, b_min, b_max, b_r=1.):
    return torch.exp(-0.5/(b_r+1)*t**(b_r+1)*(b_max-b_min)-0.5*t*b_min)

# approximate the integral of -0.5*[\beta D]_t in Eq.(13)
def compute_integral_beta_Dt(ts, beta_t, Dt):
    bD0 = Dt[0, :, :] * beta_t[0] * ts[0]
    bDt = torch.einsum('tij,t,t->tij', Dt[1:, :, :], beta_t[1:], torch.diff(ts))
    bDt = bD0 + torch.cumsum(bDt, dim=0)
    bDt = torch.cat((bD0.reshape(-1, *bD0.shape), bDt))
    return -0.5 * bDt


def compute_dyn_variance(int_beta_Dt, int_beta_t, dim, t_len):
    C_H_power_dyn = torch.zeros([t_len, dim*2, dim*2])
    C_H_power_dyn[:, :dim, :dim] = int_beta_Dt
    C_H_power_dyn[:, dim:, dim:] = -int_beta_Dt.mH
    C_H_power_dyn[:, :dim, dim:] = torch.einsum('t,ij->tij', int_beta_t, torch.eye(dim))

    C_H_pair = torch.linalg.matrix_exp(C_H_power_dyn)
    Initial_Matrix = torch.cat((torch.zeros_like(torch.eye(dim)), torch.eye(dim)), dim=0)
    # compute (Ct; Ht) in Eq.(14)
    C_H = torch.einsum('tij,jk->tik', C_H_pair, Initial_Matrix)
    C = C_H[:, : dim, :]
    H = C_H[:, dim: , :]
    Covariance = torch.einsum('tij,tjk->tik', C, torch.linalg.inv(H))
    L = torch.linalg.cholesky(Covariance)
    invL = torch.linalg.inv(L.mH)
    prior_covariance = Covariance[-1, :, :]
    return prior_covariance, L, invL

def cache_dynamics(opt, ts, At):
    dim, t_len = At[0].shape[0], ts.shape[0]
    Dt = torch.eye(dim).repeat(t_len, 1, 1) - 2 * At
    sqrt_betas = compute_vp_diffusion(ts, opt.beta_min, opt.beta_max, opt.beta_r)
    int_beta_Dt = compute_integral_beta_Dt(ts, sqrt_betas**2, Dt)

    int_beta_t = -2. * torch.log(compute_vp_kernel_mean_scale(ts, opt.beta_min, opt.beta_max, opt.beta_r))
    mean_scales = torch.linalg.matrix_exp(int_beta_Dt)
    prior_covariance, L, invL = compute_dyn_variance(int_beta_Dt, int_beta_t, dim, t_len)
    return prior_covariance, Dt, sqrt_betas, mean_scales, L, invL

def compute_vp_xs_label_matrix(opt, x0, sqrt_betas, mean_scales, A, L, invL, samp_t_idx):
    """ return xs.shape == [batch_x, batch_t, *x_dim]  """
    x_dim = opt.data_dim
    assert x_dim==list(x0.shape[1:])
    batch_x, batch_t = x0.shape[0], len(samp_t_idx)

    # p(x_t|x_0) = N(mean_scale * x_0, std_t^2)
    # x_t = mean_scale * x_0 + std_t * noise
    noise = torch.randn(batch_x, batch_t, *x_dim)
    mean_scale_t = mean_scales[samp_t_idx]
    L_t = L[samp_t_idx]
    invL_t = invL[samp_t_idx]
    A_t = A[samp_t_idx]
    # compute x_t in the last equation in page 4 left 
    analytic_xs = torch.einsum('tij,btj->bti', L_t, noise) + torch.einsum('tij,bj->bti', mean_scale_t, x0)
    # compute Eq.(15)
    part_label = - torch.einsum('tij,btj->bti', invL_t, noise)
    sqrt_beta_t = sqrt_betas[samp_t_idx].reshape(1,-1,*([1,]*len(x_dim))) # shape = [1,batch_t,1,1,1]
    # SB-FBSDE framework includes an additional scalar beta on the label
    label = part_label * sqrt_beta_t
    # change DSM score to SB-FBSDE score
    label -= (torch.einsum('tij,btj->bti', A_t, analytic_xs) * sqrt_beta_t)
    return analytic_xs, label

def get_xs_label_computer(opt, sqrt_betas, mean_scales, At, L, invL):
    if opt.forward_net.startswith('Linear'):
        fn = compute_vp_xs_label_matrix
        kwargs = dict(opt=opt, sqrt_betas=sqrt_betas, mean_scales=mean_scales, A=At, L=L, invL=invL)
    elif opt.forward_net == 'ImgLinear' and not opt.DSM_baseline:
        D = torch.ones_like(A) - 2 * A
        vec_beta_min = opt.beta_min * D
        vec_beta_max = opt.beta_max * D
        
        mean_scales = compute_vp_kernel_mean_scale_ImgVec(ts, vec_beta_min, vec_beta_max, opt.beta_r)
        fn = compute_vp_xs_label_ImgVec
        kwargs = dict(opt=opt, sqrt_betas=sqrt_betas, mean_scales=mean_scales, D=D)
    else:
        raise ValueError('Unknown forward network!')
    return partial(fn, **kwargs)
