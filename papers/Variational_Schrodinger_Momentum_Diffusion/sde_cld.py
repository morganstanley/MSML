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

def build(opt, p, q, v):
    print(util.magenta("build base sde..."))
    return VanillaSDE(opt, p, q, v)


class BaseSDE(metaclass=abc.ABCMeta):
    def __init__(self, opt, p, q, v):
        self.opt = opt
        self.dt = opt.T / opt.interval
        self.p = p # data distribution
        self.q = q # prior distribution
        self.v = v # velocity distribution

        self.b_min = opt.beta_min
        self.b_max = opt.beta_max
        self.b_r = opt.beta_r

    @abc.abstractmethod
    def _f(self, x, t):
        raise NotImplementedError

    @abc.abstractmethod
    def _g(self, x, t):
        raise NotImplementedError

    def f(self, a, t, direction):
        sign = 1. if direction == 'forward' else -1.
        return sign * self._f(a, t)

    def g(self, t):
        return self._g(t)

    def dw(self, a, dt=None):
        dt = self.dt if dt is None else dt
        x, v = torch.chunk(a, 2, dim=1)
        dw = torch.randn_like(v) * np.sqrt(dt)
        return torch.cat([torch.zeros_like(dw), dw], dim=-1)

    def propagate(self, t, a, z, direction, f=None, dw=None, dt=None):
        g = self.g(t)
        f = self.f(a, t, direction) if f is None else f
        dt = self.dt if dt is None else dt
        dw = self.dw(a, dt) if dw is None else dw
        z = torch.cat([torch.zeros_like(z,), z], dim=-1)
        return a + (f + g*z)*dt + g*dw

    def propagate_ode(self, t, a, z, z_f, direction, f=None, dt=None):
        g = self.g(t)
        f = self.f(a, t, direction) if f is None else f
        dt = self.dt if dt is None else dt
        z = torch.cat([torch.zeros_like(z,), z], dim=-1)
        z_f = torch.cat([torch.zeros_like(z_f,), z_f], dim=-1)
        dsm_score = z + z_f # map fb-sde score to dsm score
        return a + (f - g * z_f + 0.5 * g * dsm_score) * dt

    def sample_traj(self, ts, policy, corrector=None, apply_trick=True, save_traj=True, adaptive_prior=None, policy_f=None):
        # first we need to know whether we're doing forward or backward sampling
        opt = self.opt
        direction = policy.direction
        assert direction in ['forward', 'backward']

        # set up ts and init_distribution
        _assert_increasing('ts', ts)
        init_dist = self.p if direction == 'forward' else self.q
        if direction == 'backward' and adaptive_prior != None:
            print(util.cyan('Update prior for backward process.'))
            init_dist = adaptive_prior
        ts = ts if direction=='forward' else torch.flip(ts, dims=[0])

        x = init_dist.sample() # [bs, x_dim]
        v = self.v.sample()
        aug = torch.cat([x, v], dim=-1)
        augs = torch.empty((x.shape[0], len(ts), *aug.shape[1:])) if save_traj else None
        zs = torch.empty((x.shape[0], len(ts), *x.shape[1:])) if save_traj else None

        _ts = tqdm(ts, desc=util.yellow("Propagating Dynamics..."))
        for idx, t in enumerate(_ts):
            _t=t if idx == ts.shape[0] - 1 else ts[idx+1]

            f = self.f(aug, t, direction)
            z = policy(aug, t) # aug dim is 4 while z dim is 2
            z_f = policy_f(aug, t) if policy_f != None else None
            dw = self.dw(aug)

            t_idx = idx if direction == 'forward' else len(ts)-idx-1
            if save_traj:
                augs[:, t_idx, ...] = aug
                zs[:, t_idx, ...] = z

            if policy_f != None and direction == 'backward':
                aug = self.propagate_ode(t, aug, z, z_f, direction, f=f)
            else:
                aug = self.propagate(t, aug, z, direction, f=f, dw=dw)
        aug_term = aug
        res = [augs, zs, aug_term]
        return res

    def compute_nll(self, samp_bs, ts, z_f, z_b, covariance):
        assert z_f.direction == 'forward'
        assert z_b.direction == 'backward'
        opt = self.opt
        x = self.p.sample() # [bs, x_dim]
        v = self.v.sample()
        aug = torch.cat([x, v], dim=-1)
        delta_logp = 0
        e = loss.sample_e(opt, aug)
        for idx, t in enumerate(tqdm(ts,desc=util.yellow("Propagating Dynamics..."))):
            with torch.set_grad_enabled(True):
                aug.requires_grad_(True)
                g = self.g(t)
                
                f = self.f(aug, t, 'forward')
                z = z_f(aug, t)
                z2 = z_b(aug, t)
                z_aug = torch.cat([torch.zeros_like(z,), z], dim=-1)
                z2_aug = torch.cat([torch.zeros_like(z2,), z2], dim=-1)
                dx_dt = f + g * z_aug - 0.5 * g * (z_aug + z2_aug)
                divergence = divergence_approx(dx_dt, aug, e=e)
                dlogp_x_dt = - divergence.view(samp_bs, 1)

            del divergence, z2, g
            aug, dx_dt, dlogp_x_dt = aug.detach(), dx_dt.detach(), dlogp_x_dt.detach()

            z, f, direction = z.detach(), f.detach(), z_f.direction
            aug = self.propagate(t, aug, z, direction, f=f)
            if idx == 0: # skip t = t0 since we'll get its parametrized value later
                continue
            delta_logp = delta_logp + dlogp_x_dt*self.dt

        x_dim = np.prod(opt.data_dim)
        loc = torch.zeros(x_dim).to(opt.device)
        p_xT = torch.distributions.MultivariateNormal(loc=loc, covariance_matrix=covariance)
        log_px = p_xT.log_prob(x.reshape(samp_bs, -1)).to(x.device)
        logp_x = log_px - delta_logp.view(-1)
        logpx_per_dim = torch.sum(logp_x) / x.nelement() # averaged over batches
        bits_per_dim = -(logpx_per_dim - np.log(256)) / np.log(2)
        
        return bits_per_dim


def divergence_approx(f, y, e=None):
    e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
    e_dzdx_e = e_dzdx * e
    approx_tr_dzdx = e_dzdx_e.view(y.shape[0], -1).sum(dim=1)
    return approx_tr_dzdx

class VanillaSDE(BaseSDE):
    def __init__(self, opt, p, q, v):
        super(VanillaSDE,self).__init__(opt, p, q, v)

    def _f(self, a, t):
        damped_f = torch.kron(torch.Tensor([[0, -1], [1, self.opt.gamma]]), torch.eye(self.opt.data_dim[0]))
        scaled_damped = compute_drift_coef(t, self.b_min, self.b_max, self.b_r) * damped_f
        return torch.einsum('ij,bj->bi', scaled_damped, a)

    def _g(self, t):
        return compute_diffusion(t, self.b_min, self.b_max, self.b_r) * np.sqrt(self.opt.gamma)

####################################################
##  Implementation of SDE analytic kernel         ##
##  Ref: https://arxiv.org/pdf/2011.13456v2.pdf,  ##
##       page 15-16, Eq (30,32,33)                ##
####################################################


""" Generalized Song, Yang's schedule by including a tuning parameter b_r """
def compute_diffusion(t, b_min, b_max, b_r=1., T=1.):
    return torch.sqrt(b_min+(t/T)**b_r*(b_max-b_min))

def compute_drift_coef(t, b_min, b_max, b_r=1.):
    g = compute_diffusion(t, b_min, b_max, b_r)
    return -0.5 * g**2

def compute_kernel_mean_scale(t, b_min, b_max, b_r=1.):
    return torch.exp(-0.5/(b_r+1)*t**(b_r+1)*(b_max-b_min)-0.5*t*b_min)

# approximate the integral of -0.5*[\beta D]_t in Eq.(13)
def compute_integral_beta_Dt(ts, beta_t, Dt):
    bD0 = Dt[0, :, :] * beta_t[0] * ts[0]
    bDt = torch.einsum('tij,t,t->tij', Dt[1:, :, :], beta_t[1:], torch.diff(ts))
    bDt = bD0 + torch.cumsum(bDt, dim=0)
    bDt = torch.cat((bD0.reshape(-1, *bD0.shape), bDt))
    return -0.5 * bDt

def compute_dyn_variance(int_beta_Dt, int_beta_t, a_dim, t_len, gamma):
    # the top right entry of the matrix in (C; H) matrix computation. to detail later.......
    corner_mat = torch.kron(torch.Tensor([[0., 0.], [0., 1.]]), torch.eye(a_dim//2))
    C_H_power_dyn = torch.zeros([t_len, 2*a_dim, 2*a_dim])
    C_H_power_dyn[:, :a_dim, :a_dim] = int_beta_Dt
    C_H_power_dyn[:, a_dim:, a_dim:] = -int_beta_Dt.mH
    C_H_power_dyn[:, :a_dim, a_dim:] = gamma * torch.einsum('t,ij->tij', int_beta_t, corner_mat)
    C_H_pair = torch.linalg.matrix_exp(C_H_power_dyn)

    Initial_Matrix = torch.cat((torch.zeros_like(torch.eye(a_dim)), torch.eye(a_dim)), dim=0)
    # compute (Ct; Ht) in Eq.(14)
    C_H = torch.einsum('tij,jk->tik', C_H_pair, Initial_Matrix)
    C = C_H[:, : a_dim, :]
    H = C_H[:, a_dim: , :]
    Covariance = torch.einsum('tij,tjk->tik', C, torch.linalg.inv(H))
    L = torch.linalg.cholesky(Covariance)
    invL = torch.linalg.inv(L.mH)
    prior_covariance = Covariance[-1, :(a_dim//2), :(a_dim//2)]
    return prior_covariance, L, invL

def cache_dynamics(opt, ts, At):
    dim, t_len = opt.data_dim[0], ts.shape[0]
    #Dt = torch.eye(dim).repeat(t_len, 1, 1) - 2 * At # VSDM paper
    At = torch.cat([torch.zeros_like(At), At], dim=1)
    Dt = torch.kron(torch.Tensor([[0, -1], [1, opt.gamma]]), torch.eye(dim)).repeat(t_len, 1, 1) - 2 * opt.gamma * At
    sqrt_betas = compute_diffusion(ts, opt.beta_min, opt.beta_max, opt.beta_r)
    int_beta_Dt = compute_integral_beta_Dt(ts, sqrt_betas**2, Dt)

    int_beta_t = -2. * torch.log(compute_kernel_mean_scale(ts, opt.beta_min, opt.beta_max, opt.beta_r))
    mean_scales = torch.linalg.matrix_exp(int_beta_Dt)
    prior_covariance, L, invL = compute_dyn_variance(int_beta_Dt, int_beta_t, 2*dim, t_len, gamma=opt.gamma)
    return prior_covariance, Dt, sqrt_betas, mean_scales, L, invL

def compute_xs_label_matrix(opt, a0, sqrt_betas, mean_scales, A, L, invL, samp_t_idx):
    a_dim = [2 * opt.data_dim[0]]
    batch_x, batch_t = a0.shape[0], len(samp_t_idx)

    # p(x_t|x_0) = N(mean_scale * x_0, std_t^2)
    # x_t = mean_scale * x_0 + std_t * noise
    noise = torch.randn(batch_x, batch_t, *a_dim)
    mean_scale_t = mean_scales[samp_t_idx]
    L_t = L[samp_t_idx]
    invL_t = invL[samp_t_idx]
    A_t = A[samp_t_idx]

    analytic_as = torch.einsum('tij,btj->bti', L_t, noise) + torch.einsum('tij,bj->bti', mean_scale_t, a0)
    # compute Eq.(15)
    part_label = - torch.einsum('tij,btj->bti', invL_t, noise)
    sqrt_beta_t = sqrt_betas[samp_t_idx].reshape(1,-1,*([1,]*len(a_dim))) # shape = [1,batch_t,1,1,1]
    # SB-FBSDE framework includes an additional scalar beta on the label
    label = part_label * sqrt_beta_t
    x_label, v_label = torch.chunk(label, 2, dim=-1)
    # change DSM score to SB-FBSDE score
    v_label -= (torch.einsum('tij,btj->bti', A_t, analytic_as) * sqrt_beta_t)
    return analytic_as, v_label

def get_xs_label_computer(opt, sqrt_betas, mean_scales, At, L, invL):
    if opt.forward_net.startswith('Linear'):
        fn = compute_xs_label_matrix
        kwargs = dict(opt=opt, sqrt_betas=sqrt_betas, mean_scales=mean_scales, A=At, L=L, invL=invL)
    else:
        raise ValueError('Unknown forward network!')
    return partial(fn, **kwargs)
