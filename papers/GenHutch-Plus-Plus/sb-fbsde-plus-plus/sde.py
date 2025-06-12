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

    return {
        'vp': VPSDE,
        've': VESDE,
        'simple': SimpleSDE,
    }.get(opt.sde_type)(opt, p, q)


class BaseSDE(metaclass=abc.ABCMeta):
    def __init__(self, opt, p, q):
        self.opt = opt
        self.dt=opt.T/opt.interval
        self.p = p # data distribution
        self.q = q # prior distribution

    @abc.abstractmethod
    def _f(self, x, t):
        raise NotImplementedError

    @abc.abstractmethod
    def _g(self, x, t):
        raise NotImplementedError

    def f(self, x, t, direction):
        sign = 1. if direction=='forward' else -1.
        return sign * self._f(x,t)

    def g(self, t):
        return self._g(t)

    def dw(self, x, dt=None):
        dt = self.dt if dt is None else dt
        return torch.randn_like(x)*np.sqrt(dt)

    def propagate(self, t, x, z, direction, f=None, dw=None, dt=None):
        g = self.g(  t)
        f = self.f(x,t,direction) if f is None else f
        dt = self.dt if dt is None else dt
        dw = self.dw(x,dt) if dw is None else dw

        return x + (f + g*z)*dt + g*dw

    def propagate_x0_trick(self, x, policy, direction):
        """ propagate x0 by a tiny step """
        t0  = torch.Tensor([0])
        dt0 = self.opt.t0 - 0
        assert dt0 > 0
        z0  = policy(x,t0)
        return self.propagate(t0, x, z0, direction, dt=dt0)

    def denoise_step(self,opt,policy,policy2,x,t):
        """ currently deprecated function
        """
        if opt.sde_type=='ve':#VP's denosing step is just apply_trick2, this is only for VE.
            # z2 =policy2(x,t)
            zero=torch.zeros_like(t)
            z = policy(x,zero)
            g=self.g(zero)
            z=z
            x=x+z/g*self.sigma_min**2*self.opt.t0
            print('trick applied,sigma_min{}'.format(self.sigma_min))
        return x

    def sample_traj(self, ts, policy, corrector=None, apply_trick=True, save_traj=True):

        # first we need to know whether we're doing forward or backward sampling
        opt = self.opt
        direction = policy.direction
        assert direction in ['forward','backward']

        # set up ts and init_distribution
        _assert_increasing('ts', ts)
        init_dist = self.p if direction=='forward' else self.q
        ts = ts if direction=='forward' else torch.flip(ts,dims=[0])

        x = init_dist.sample() # [bs, x_dim]

        apply_trick1, apply_trick2, apply_trick3 = compute_tricks_condition(opt, apply_trick, direction)

        # [trick 1] propagate img (x0) by a tiny step
        if apply_trick1: x = self.propagate_x0_trick(x, policy, direction)

        xs = torch.empty((x.shape[0], len(ts), *x.shape[1:])) if save_traj else None
        zs = torch.empty_like(xs) if save_traj else None

        # don't use tqdm for fbsde since it'll resample every itr
        _ts = ts if opt.train_method=='joint' else tqdm(ts,desc=util.yellow("Propagating Dynamics..."))
        for idx, t in enumerate(_ts):
            _t=t if idx==ts.shape[0]-1 else ts[idx+1]

            f = self.f(x,t,direction)
            z =policy(x,t)
            dw = self.dw(x)

            t_idx = idx if direction=='forward' else len(ts)-idx-1
            if save_traj:
                xs[:,t_idx,...]=x
                zs[:,t_idx,...]=z

            # [trick 2] zero out dw
            if apply_trick2(t_idx=t_idx): dw = torch.zeros_like(dw)
            x = self.propagate(t, x, z, direction, f=f, dw=dw)

            if corrector is not None:
                denoise_xT = False # apply_trick3(t_idx=t_idx) # [trick 3] additional denoising step for xT
                x  = self.corrector_langevin_update(_t ,x, corrector, denoise_xT)

        x_term = x

        res = [xs, zs, x_term]
        return res

    def corrector_langevin_update(self, t, x, corrector, denoise_xT):
        opt = self.opt
        batch = x.shape[0]
        alpha_t = compute_alphas(t, opt.beta_min, opt.beta_max) if util.use_vp_sde(opt) else 1.
        g_t = self.g(t)
        for _ in range(opt.num_corrector):
            # here, z = g * score
            z =  corrector(x,t)

            # score-based model : eps_{SGM} = 2 * alpha * (snr * \norm{noise/score} )^2
            # schrodinger bridge: eps_{SB}  = 2 * alpha * (snr * \norm{noise/z} )^2
            #                               = g^{-2} * eps_{SGM}
            z_avg_norm = z.reshape(batch,-1).norm(dim=1).mean()
            eps_temp = 2 * alpha_t * (opt.snr / z_avg_norm )**2
            noise=torch.randn_like(z)
            noise_avg_norm = noise.reshape(batch,-1).norm(dim=1).mean()
            eps = eps_temp * (noise_avg_norm**2)

            # score-based model:  x <- x + eps_SGM * score + sqrt{2 * eps_SGM} * noise
            # schrodinger bridge: x <- x + g * eps_SB * z  + sqrt(2 * eps_SB) * g * noise
            #                     (so that drift and diffusion are of the same scale) 
            x = x + g_t*eps*z + g_t*torch.sqrt(2*eps)*noise

        if denoise_xT: x = x + g_t*z

        return x

    def compute_nll(self, samp_bs, ts, z_f, z_b):

        assert z_f.direction == 'forward'
        assert z_b.direction == 'backward'

        opt = self.opt

        x = self.p.sample() # [bs, x_dim]

        delta_logp = 0
        e = loss.sample_e(opt, x)

        for idx, t in enumerate(tqdm(ts,desc=util.yellow("Propagating Dynamics..."))):

            with torch.set_grad_enabled(True):
                x.requires_grad_(True)
                g = self.g(  t)
                f = self.f(x,t,'forward')
                z = z_f(x,t)
                z2 = z_b(x,t)

                dx_dt = f + g * z - 0.5 * g * (z + z2)
                divergence = divergence_approx(dx_dt, x, e=e)
                dlogp_x_dt = - divergence.view(samp_bs, 1)

            del divergence, z2, g
            x, dx_dt, dlogp_x_dt = x.detach(), dx_dt.detach(), dlogp_x_dt.detach()
            z, f, direction = z.detach(), f.detach(), z_f.direction
            x = self.propagate(t, x, z, direction, f=f)

            # ===== uncomment if using corrector =====
            # _t=t if idx==ts.shape[0]-1 else ts[idx+1]
            # x  = self.corrector_langevin_update(_t, x, z_f, z_b, False)
            # ========================================

            if idx == 0: # skip t = t0 since we'll get its parametrized value later
                continue
            delta_logp = delta_logp + dlogp_x_dt*self.dt

        x_dim = np.prod(opt.data_dim)
        loc = torch.zeros(x_dim).to(opt.device)
        covariance_matrix = opt.sigma_max**2*torch.eye(x_dim).to(opt.device)
        p_xT = torch.distributions.MultivariateNormal(loc=loc, covariance_matrix=covariance_matrix)
        log_px = p_xT.log_prob(x.reshape(samp_bs, -1)).to(x.device)

        logp_x = log_px - delta_logp.view(-1)
        logpx_per_dim = torch.sum(logp_x) / x.nelement() # averaged over batches
        bits_per_dim = -(logpx_per_dim - np.log(256)) / np.log(2)
        
        return bits_per_dim

def compute_tricks_condition(opt, apply_trick, direction):
    if not apply_trick:
        return False, lambda t_idx: False,  False

    # [trick 1] source: Song et al ICLR 2021 Appendix C
    # when: (i) image, (ii) p -> q, (iii) t0 > 0,
    # do:   propagate img (x0) by a tiny step.
    apply_trick1 = (util.is_image_dataset(opt) and direction == 'forward' and opt.t0 > 0)

    # [trick 2] Improved DDPM
    # when: (i) image, (ii) q -> p, (iii) vp, (iv) last sampling step
    # do:   zero out dw
    trick2_cond123 = (util.is_image_dataset(opt) and direction=='backward' and util.use_vp_sde(opt))
    def _apply_trick2(trick2_cond123, t_idx):
        return trick2_cond123 and t_idx==0
    apply_trick2 = partial(_apply_trick2, trick2_cond123=trick2_cond123)

    # [trick 3] NCSNv2, Alg 1
    # when: (i) image, (ii) q -> p, (iii) last sampling step
    # do:   additional denoising step
    trick3_cond12 = (util.is_image_dataset(opt) and direction=='backward')
    def _apply_trick3(trick3_cond12, t_idx):
        return trick3_cond12 and t_idx==0
    apply_trick3 = partial(_apply_trick3, trick3_cond12=trick3_cond12)

    return apply_trick1, apply_trick2, apply_trick3

def divergence_approx(f, y, e=None):
    e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
    e_dzdx_e = e_dzdx * e
    approx_tr_dzdx = e_dzdx_e.contiguous().view(y.shape[0], -1).sum(dim=1)
    return approx_tr_dzdx

class SimpleSDE(BaseSDE):
    def __init__(self, opt, p, q, var=1.0):
        super(SimpleSDE, self).__init__(opt, p, q)
        self.var = var

    def _f(self, x, t):
        return torch.zeros_like(x)

    def _g(self, t):
        return torch.Tensor([self.var])

class VPSDE(BaseSDE):
    def __init__(self, opt, p, q):
        super(VPSDE,self).__init__(opt, p, q)
        self.b_min=opt.beta_min
        self.b_max=opt.beta_max

    def _f(self, x, t):
        return compute_vp_drift_coef(t, self.b_min, self.b_max)*x

    def _g(self, t):
        return compute_vp_diffusion(t, self.b_min, self.b_max)

class VESDE(BaseSDE):
    def __init__(self, opt, p, q):
        super(VESDE,self).__init__(opt, p, q)
        self.s_min=opt.sigma_min
        self.s_max=opt.sigma_max

    def _f(self, x, t):
        return torch.zeros_like(x)

    def _g(self, t):
        return compute_ve_diffusion(t, self.s_min, self.s_max)

####################################################
##  Implementation of SDE analytic kernel         ##
##  Ref: https://arxiv.org/pdf/2011.13456v2.pdf,  ##
##       page 15-16, Eq (30,32,33)                ##
####################################################

def compute_sigmas(t, s_min, s_max):
    return s_min * (s_max/s_min)**t

def compute_ve_g_scale(s_min, s_max):
    return np.sqrt(2*np.log(s_max/s_min))

def compute_ve_diffusion(t, s_min, s_max):
    return compute_sigmas(t, s_min, s_max) * compute_ve_g_scale(s_min, s_max)

def compute_vp_diffusion(t, b_min, b_max):
    return torch.sqrt(b_min+t*(b_max-b_min))

def compute_vp_drift_coef(t, b_min, b_max):
    g = compute_vp_diffusion(t, b_min, b_max)
    return -0.5 * g**2

def compute_vp_kernel_mean_scale(t, b_min, b_max):
    return torch.exp(-0.25*t**2*(b_max-b_min)-0.5*t*b_min)

def compute_alphas(t, b_min, b_max):
    return compute_vp_kernel_mean_scale(t, b_min, b_max)**2

def compute_ve_xs_label(opt, x0, sigmas, samp_t_idx):
    """ return xs.shape == [batch_x, *x_dim]  """
    s_max = opt.sigma_max
    s_min = opt.sigma_min
    x_dim = opt.data_dim

    assert x_dim==list(x0.shape[1:])
    batch_x, batch_t = x0.shape[0], len(samp_t_idx)

    # p(x_t|x_0) = N(x_0, sigma_t^2)
    # x_t = x_0 + sigma_t * noise
    noise = torch.randn(batch_x, batch_t, *x_dim)
    sigma_t = sigmas[samp_t_idx].reshape(1,-1,*([1,]*len(x_dim))) # shape = [1,batch_t,1,1,1]
    analytic_xs = sigma_t * noise + x0[:,None,...]

    # score_of_p = -1/sigma_t^2 (x_t - x_0) = -noise/sigma_t
    # dx_t = g dw_t, where g = sigma_t * g_scaling
    # hence, g * score_of_p = - noise * g_scaling
    label = - noise * compute_ve_g_scale(s_min, s_max)

    return analytic_xs, label

def compute_vp_xs_label(opt, x0, sqrt_betas, mean_scales, samp_t_idx):
    """ return xs.shape == [batch_x, batch_t, *x_dim]  """

    x_dim = opt.data_dim

    assert x_dim==list(x0.shape[1:])
    batch_x, batch_t = x0.shape[0], len(samp_t_idx)

    # p(x_t|x_0) = N(mean_scale * x_0, std_t^2)
    # x_t = mean_scale * x_0 + std_t * noise
    noise = torch.randn(batch_x, batch_t, *x_dim)
    mean_scale_t = mean_scales[samp_t_idx].reshape(1,-1,*([1,]*len(x_dim))) # shape = [1,batch_t,1,1,1]
    std_t = torch.sqrt(1 - mean_scale_t**2)
    analytic_xs = std_t * noise + mean_scale_t * x0[:,None,...]

    # score_of_p = -1/std_t^2 (x_t - mean_scale_t * x_0) = -noise/std_t
    # hence, g * score_of_p = - noise / std_t * sqrt_beta_t
    sqrt_beta_t = sqrt_betas[samp_t_idx].reshape(1,-1,*([1,]*len(x_dim))) # shape = [1,batch_t,1,1,1]
    label = - noise / std_t * sqrt_beta_t

    return analytic_xs, label

def get_xs_label_computer(opt, ts):

    if opt.sde_type == 'vp':
        mean_scales = compute_vp_kernel_mean_scale(ts, opt.beta_min, opt.beta_max)
        sqrt_betas = compute_vp_diffusion(ts, opt.beta_min, opt.beta_max)
        fn = compute_vp_xs_label
        kwargs = dict(opt=opt, sqrt_betas=sqrt_betas, mean_scales=mean_scales)

    elif opt.sde_type == 've':
        sigmas = compute_sigmas(ts, opt.sigma_min, opt.sigma_max)
        fn = compute_ve_xs_label
        kwargs = dict(opt=opt, sigmas=sigmas)

    else:
        raise RuntimeError()

    return partial(fn, **kwargs)
