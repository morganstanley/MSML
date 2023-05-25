import numpy as np
import abc
from tqdm import tqdm
import sys
from functools import partial
import torch

import util
import loss
from ipdb import set_trace as debug
import runner

def _assert_increasing(name, ts):
    assert (ts[1:] > ts[:-1]).all(), '{} must be strictly increasing'.format(name)

def build(opt, p, q):
    print(util.magenta("build base sde..."))

    return {
        'vp': VPSDE,
        'vp_v2': VPSDEv2,
        've': VESDE,
        'simple': SimpleSDE,
    }.get(opt.sde_type)(opt, p, q)


class BaseSDE(metaclass=abc.ABCMeta):
    def __init__(self, opt, p, q):
        self.opt = opt
        # self.dt=opt.T/opt.interval  # large error when interval is small.
        # the timeline in Runner self.ts = torch.linspace(opt.t0, opt.T, opt.interval)
        self.dt = (opt.T - opt.t0)/ (opt.interval-1)
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
        if self.opt.sde_type == 'vp_v2' and direction == 'backward':
            return self.propagate_vp_v2_backward(t, x, z, dw, dt)
        else:
            return self.propagate_default(t, x, z, direction, f, dw, dt)

    def propagate_vp_v2_backward(self, t, x, z, dw=None, dt=None):
        f = - self.f_back(x,t)
        # Note dt is positive here. should be negative infinitesimal. so f and socre will be flipped.
        dt = self.dt if dt is None else dt
        # divide g here as the z is g*score, where g is from the forward process (label calculation).
        g = self.g(t)
        g_score_back = self.g_score_back(t)
        g_score = g_score_back / g
        # z is g*score, not score exactly, so skip one g here.
        g_back = self.g_back(t)
        dw = self.dw(x,dt) if dw is None else dw
        return x + (f + g_score*z)*dt + g_back*dw

    def propagate_default(self, t, x, z, direction, f=None, dw=None, dt=None):
        g = self.g(t)
        f = self.f(x,t,direction) if f is None else f
        dt = self.dt if dt is None else dt
        dw = self.dw(x,dt) if dw is None else dw

        return x + (f + g*z)*dt + g*dw
        # z is g*score, not score exactly, so skip one g here.
        # For backward process, dt should be neg, but we use +dt here, so f is turned into neg while
        # the score z is kept as positive.

    def propagate_x0_trick(self, x, policy, direction):
        """ propagate x0 by a tiny step """
        t0  = torch.Tensor([0]).to(x.device)
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

    def sample_traj(
            self,
            ts,
            policy,
            corrector=None,
            apply_trick=True,
            num_samples=None,
            save_traj=True):

        # first we need to know whether we're doing forward or backward sampling
        opt = self.opt
        direction = policy.direction
        assert direction in ['forward','backward']

        # set up ts and init_distribution
        _assert_increasing('ts', ts)
        init_dist = self.p if direction=='forward' else self.q
        ts = ts if direction=='forward' else torch.flip(ts,dims=[0])
        # For forward process, num_sampels is useless as it's fixed in data sampler, not prior sampler.
        x = init_dist.sample(num_samples=num_samples).to(opt.device) # [bs, x_dim]

        apply_trick1, apply_trick2, apply_trick3 = compute_tricks_condition(
                opt, apply_trick, direction)

        # [trick 1] propagate img (x0) by a tiny step
        if apply_trick1: x = self.propagate_x0_trick(x, policy, direction)
        if save_traj:
            xs = torch.empty((x.shape[0], len(ts), *x.shape[1:]), device=opt.device)
        else:
            xs = None
        zs = torch.empty_like(xs, device=opt.device) if save_traj else None

        # don't use tqdm for fbsde since it'll resample every itr
        if opt.train_method=='joint':
            _ts = ts
        else:
            _ts = tqdm(ts, ncols=80, file=sys.stdout, desc=util.yellow(f"{direction} SDE sampling..."))
        for idx, t in enumerate(_ts):
            _t=t if idx==ts.shape[0]-1 else ts[idx+1]

            f = self.f(x,t,direction)
            if getattr(opt, direction + '_net') in ['Transformerv2', 'Transformerv4', 'Transformerv5']:
                # Wrong unconditional setup as a reminder.
                # diff_input = torch.cat([x, torch.ones_like(x)], dim=1)  # (cond_obs, noisy_target)
                # x_input = (diff_input, torch.ones_like(x))  # (diff_input, cond_mask)
                diff_input = torch.cat([torch.zeros_like(x), x], dim=1)  # (cond_obs, noisy_target)
                x_input = (diff_input, torch.zeros_like(x))  # (diff_input, cond_mask)
            elif getattr(opt, direction + '_net') == 'Transformerv3':
                x_input = (x, torch.zeros_like(x))  # (diff_input, cond_mask)
            else:
                x_input = x
            z =policy(x_input,t)
            dw = self.dw(x)

            t_idx = idx if direction=='forward' else len(ts)-idx-1
            if save_traj:
                xs[:,t_idx,...]=x
                zs[:,t_idx,...]=z

            # [trick 2] zero out dw
            if apply_trick2(t_idx=t_idx): dw = torch.zeros_like(dw)
            x = self.propagate(t, x, z, direction, f=f, dw=dw)

            if corrector is not None:
                # [trick 3] additional denoising step for xT
                denoise_xT = False # apply_trick3(t_idx=t_idx)
                x  = self.corrector_langevin_update(_t, x, corrector, denoise_xT)

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


    def corrector_langevin_imputation_update(self, t, x, corrector, x_cond, cond_mask, denoise_xT):
        opt = self.opt
        batch = x.shape[0]
        alpha_t = compute_alphas(t, opt.beta_min, opt.beta_max) if util.use_vp_sde(opt) else 1.
        g_t = self.g(t)
        for _ in range(opt.num_corrector):
            # here, z = g * score
            z =  corrector(x, t, x_cond, cond_mask)

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


    def sample_traj_conditional(
            self,
            ts,
            x_cond,
            obs_mask,
            cond_mask,
            target_mask,
            policy,
            corrector=None,
            apply_trick=True,
            num_samples=None,
            save_traj=True,
            save_masked_x=True):

        # first we need to know whether we're doing forward or backward sampling
        opt = self.opt
        direction = policy.direction
        assert direction in ['forward','backward']

        num_batch,C,K,L = x_cond.shape
        x_cond = torch.repeat_interleave(x_cond, num_samples, dim=0)
        cond_mask = torch.repeat_interleave(cond_mask, num_samples, dim=0)
        target_mask = torch.repeat_interleave(target_mask, num_samples, dim=0)

        # set up ts and init_distribution
        _assert_increasing('ts', ts)
        assert direction == 'backward'
        init_dist = self.p if direction=='forward' else self.q
        ts = ts if direction=='forward' else torch.flip(ts,dims=[0])

        # For forward process, num_sampels is not useless.
        x = init_dist.sample(num_samples=num_batch*num_samples).to(opt.device) # [bs, x_dim]

        apply_trick1, apply_trick2, apply_trick3 = compute_tricks_condition(
                opt, apply_trick, direction)
        # [trick 1] propagate img (x0) by a tiny step
        if getattr(opt, direction + '_net') in ['Transformerv2', 'Transformerv4', 'Transformerv5']:
            cond_obs = cond_mask * x_cond
            noisy_target = (1-cond_mask) * x  # no big difference if using  target_mask * x
            total_input = torch.cat([cond_obs, noisy_target], dim=1)
            diff_input = (total_input, cond_mask)
        elif getattr(opt, direction + '_net') == 'Transformerv3':
            cond_obs = cond_mask * x_cond
            noisy_target = (1-cond_mask) * x
            total_input = cond_obs + noisy_target
            diff_input = (total_input, cond_mask)
        else:
            diff_input = x
        if apply_trick1: x = self.propagate_x0_trick(diff_input, policy, direction)

        xs = torch.empty((x.shape[0], len(ts), *x.shape[1:]), device=opt.device) if save_traj else None
        zs = torch.empty_like(xs, device=opt.device) if save_traj else None

        # don't use tqdm for fbsde since it'll resample every itr
        if opt.train_method=='joint':
            _ts = ts
        else:
            _ts = tqdm(ts, file=sys.stdout, ncols=80, desc=util.yellow("SDE sampling..."))
        for idx, t in enumerate(_ts):
            _t=t if idx==ts.shape[0]-1 else ts[idx+1]

            f = self.f(x, t, direction)
            if getattr(opt, direction + '_net') in ['Transformerv2', 'Transformerv4', 'Transformerv5']:
                cond_obs = cond_mask * x_cond
                noisy_target = (1-cond_mask) * x  # no big difference if using  target_mask * x
                total_input = torch.cat([cond_obs, noisy_target], dim=1)
                diff_input = (total_input, cond_mask)
            elif getattr(opt, direction + '_net') == 'Transformerv3':
                cond_obs = cond_mask * x_cond
                noisy_target = (1-cond_mask)
                total_input = cond_obs + noisy_target
                diff_input = (total_input, cond_mask)
            else:
                diff_input = x * (1-cond_mask) + x_cond * cond_mask
            z = policy(diff_input, t)
            dw = self.dw(x)

            t_idx = idx if direction=='forward' else len(ts)-idx-1
            if save_traj:
                xs[:,t_idx,...]=x * (1-cond_mask) + x_cond * cond_mask if save_masked_x else x
                zs[:,t_idx,...]=z

            # [trick 2] zero out dw
            if apply_trick2(t_idx=t_idx): dw = torch.zeros_like(dw)
            x = self.propagate(t, x, z, direction, f=f, dw=dw)

            if corrector is not None:
                # [trick 3] additional denoising step for xT
                denoise_xT = False # apply_trick3(t_idx=t_idx)
                x  = self.corrector_langevin_imputation_update(
                    _t, x, corrector, x_cond, cond_mask, denoise_xT)

        x_term = x

        res = [xs, zs, x_term]
        return res


    def sample_traj_imputation_forward(
            self,
            ts,
            policy,
            corrector=None,
            apply_trick=True,
            save_traj=True):
        """sample (opt.samp_bs, opt.T)

        Unconditional sampling so far.The forward process should condition on cond_mask and sample 
        the (1-cond_mask) or target entries.
        """
        opt = self.opt
        direction = policy.direction
        assert direction in ['forward']

        # set up ts and init_distribution
        _assert_increasing('ts', ts)
        init_dist = self.p  # sample from data, not prior.
        obs_data, obs_mask, gt_mask = init_dist.sample(return_all_mask=True)  # opt.samp_bs
        obs_data, obs_mask, gt_mask = obs_data.to(opt.device), obs_mask.to(opt.device), gt_mask.to(opt.device)
        cond_mask = runner.Runner.get_randmask(obs_mask)

        # if getattr(opt, direction + '_net') in ['Transformerv2', 'Transformerv3']:
        #     cond_mask = runner.Runner.get_randmask(obs_mask)
        # else:
        #     # Latter in loss calculation: target_mask = obs_mask - cond_mask.
        #     # So setting cond_mask = 0 means setting target_mask = obs_mask
        #     cond_mask = torch.zeros_like(obs_data)

        x = obs_data

        apply_trick1, apply_trick2, apply_trick3 = compute_tricks_condition(
                opt, apply_trick, direction)

        if opt.forward_net in ['Transformerv2', 'Transformerv4', 'Transformerv5']:
            cond_obs = cond_mask * obs_data
            noisy_target = (1-cond_mask) * x  # no big difference if using  target_mask * x
            total_input = torch.cat([cond_obs, noisy_target], dim=1)
            diff_input = (total_input, cond_mask)
        elif opt.forward_net == 'Transformerv3':
            cond_obs = cond_mask * obs_data
            noisy_target = (1-cond_mask) * x
            total_input = cond_obs + noisy_target
            diff_input = (total_input, cond_mask)
        else:
            # TODO: not sure about whether to use conditinal or unconditional forward sampling.
            # diff_input = x * (1-cond_mask) + obs_data * cond_mask  # unsure.
            diff_input = x

        # [trick 1] propagate img (x0) by a tiny step
        if apply_trick1: x = self.propagate_x0_trick(diff_input, policy, direction)

        xs = torch.empty((x.shape[0], len(ts), *x.shape[1:]), device=opt.device) if save_traj else None
        zs = torch.empty_like(xs, device=opt.device) if save_traj else None

        # don't use tqdm for fbsde since it'll resample every itr
        if opt.train_method=='joint':
            _ts = ts
        else:
            _ts = tqdm(ts, file=sys.stdout, ncols=80, desc=util.yellow("Forward SDE sampling..."))
        for idx, t in enumerate(_ts):
            _t=t if idx==ts.shape[0]-1 else ts[idx+1]

            f = self.f(x, t, direction)
            #
            if getattr(opt, direction + '_net') in ['Transformerv2', 'Transformerv4', 'Transformerv5']:
                cond_obs = cond_mask * obs_data
                noisy_target = (1-cond_mask) * x  # no big difference if using  target_mask * x
                total_input = torch.cat([cond_obs, noisy_target], dim=1)
                diff_input = (total_input, cond_mask)
            elif getattr(opt, direction + '_net') == 'Transformerv3':
                cond_obs = cond_mask * obs_data
                noisy_target = (1-cond_mask)
                total_input = cond_obs + noisy_target
                diff_input = (total_input, cond_mask)
            else:
                diff_input = x

            z = policy(diff_input, t)
            dw = self.dw(x)

            t_idx = idx if direction=='forward' else len(ts)-idx-1
            if save_traj:
                xs[:,t_idx,...]=x
                zs[:,t_idx,...]=z

            # [trick 2] zero out dw
            if apply_trick2(t_idx=t_idx): dw = torch.zeros_like(dw)
            x = self.propagate(t, x, z, direction, f=f, dw=dw)

            if corrector is not None:
                # [trick 3] additional denoising step for xT
                cond_obs = cond_mask * obs_data
                denoise_xT = False # apply_trick3(t_idx=t_idx)
                x  = self.corrector_langevin_imputation_update(
                    _t, x, corrector, cond_obs, cond_mask, denoise_xT)

        x_term = x

        res = [xs, zs, x_term, obs_data, obs_mask, cond_mask, gt_mask]
        return res


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
    approx_tr_dzdx = e_dzdx_e.view(y.shape[0], -1).sum(dim=1)
    return approx_tr_dzdx

class SimpleSDE(BaseSDE):
    def __init__(self, opt, p, q, var=1.0):
        super(SimpleSDE, self).__init__(opt, p, q)
        self.var = var

    def _f(self, x, t):
        return torch.zeros_like(x)

    def _g(self, t):
        return torch.Tensor([self.var]).to(t.device)

class VPSDE(BaseSDE):
    def __init__(self, opt, p, q):
        super(VPSDE,self).__init__(opt, p, q)
        self.b_min=opt.beta_min
        self.b_max=opt.beta_max

    def _f(self, x, t):
        return compute_vp_drift_coef(t, self.b_min, self.b_max)*x

    def _g(self, t):
        return compute_vp_diffusion(t, self.b_min, self.b_max)


class VPSDEv2(BaseSDE):
    def __init__(self, opt, p, q):
        super(VPSDEv2,self).__init__(opt, p, q)
        self.b_min=opt.beta_min
        self.b_max=opt.beta_max
        self.opt = opt

        # These alpha, beta are for discrete version as in DDPM.
        self.ts_ = torch.linspace(opt.t0, opt.T, opt.interval).to(opt.device)
        self.betas = (self.b_min + self.ts_ * (self.b_max - self.b_min)) * self.dt
        self.alphas = 1 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def _f(self, x, t):
        t_idx = ((t-self.opt.t0) / self.dt).long()
        # mean_scale = (-1 + 1 / torch.sqrt(1-self.betas[t_idx])) / self.dt
        mean_scale = (-1 + torch.sqrt(1-self.betas[t_idx])) / self.dt
        return mean_scale * x

    def f_back(self, x, t):
        t_idx = ((t-self.opt.t0) / self.dt).long()
        # 1-order Taylor approximation, working.
        # mean_scale = - 0.5 * self.betas[t_idx] / (torch.sqrt(1-self.betas[t_idx])**3) / self.dt
        mean_scale = (1 - 1 / torch.sqrt(1-self.betas[t_idx])) / self.dt
        return mean_scale * x

    def _g(self, t):
        return compute_vp_diffusion(t, self.b_min, self.b_max)

    def g_back(self, t):
        """The correction is only designed for scalar t, not for matrix t."""
        t_idx = ((t-self.opt.t0) / self.dt).long().to(self.opt.device)
        if t.dim() == 0 and t_idx > 0:
            g = torch.sqrt((1.0 - self.alpha_bar[t_idx-1]) / (1.0 - self.alpha_bar[t_idx]) *
                            self.betas[t_idx] / self.dt)
        else:
            g = torch.sqrt(self.betas[t_idx] / self.dt)
        return g

    def g_score_back(self, t):
        t_idx = ((t-self.opt.t0) / self.dt).long()
        g = self.betas[t_idx] / torch.sqrt(1-self.betas[t_idx])  # Working, but miss one term.
        g = g / self.dt
        return g


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

def compute_ve_diffusion(t, s_min, s_max):  # Backward diffusion
    return compute_sigmas(t, s_min, s_max) * compute_ve_g_scale(s_min, s_max)

def compute_vp_diffusion(t, b_min, b_max):  # Backward diffusion
    return torch.sqrt(b_min+t*(b_max-b_min))

def compute_vp_drift_coef(t, b_min, b_max):  # Backward drift.
    g = compute_vp_diffusion(t, b_min, b_max)
    return -0.5 * g**2

def compute_vp_kernel_mean_scale(t, b_min, b_max):  # Forward process.
    return torch.exp(-0.25*t**2*(b_max-b_min)-0.5*t*b_min)

def compute_vp_v2_kernel_mean_scale(t, dt, b_min, b_max):  # Forward process.
    # t is the full timeline. sqrt{\alpha_bar}
    betas = (b_min + t * (b_max - b_min)) * dt
    alphas = 1 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    mean_scale = torch.sqrt(alpha_bar)

    return mean_scale

def compute_alphas(t, b_min, b_max):
    return compute_vp_kernel_mean_scale(t, b_min, b_max)**2


def compute_ve_xs_label(opt, x0, sigmas, samp_t_idx, return_scale=False):
    if samp_t_idx.dim() == 1:
        return compute_ve_xs_label_tdim1(opt, x0, sigmas, samp_t_idx, return_scale)
    elif samp_t_idx.dim() == 2:
        return compute_ve_xs_label_tdim2(opt, x0, sigmas, samp_t_idx, return_scale)


def compute_ve_xs_label_tdim1(opt, x0, sigmas, samp_t_idx, return_scale):
    """ return xs.shape == [batch_x, *x_dim]
    samp_t_idx (batch_t) is the same for different batch_x samples.
    """
    s_max = opt.sigma_max
    s_min = opt.sigma_min
    x_dim = opt.data_dim

    assert x_dim==list(x0.shape[1:])
    batch_x, batch_t = x0.shape[0], len(samp_t_idx)

    # p(x_t|x_0) = N(x_0, sigma_t^2)
    # x_t = x_0 + sigma_t * noise
    noise = torch.randn(batch_x, batch_t, *x_dim, device=opt.device)
    sigma_t = sigmas[samp_t_idx].reshape(1,batch_t,*([1,]*len(x_dim))) # shape = [1,batch_t,1,1,1]
    analytic_xs = sigma_t * noise + x0[:,None,...]

    # score_of_p = -1/sigma_t^2 (x_t - x_0) = -noise/sigma_t
    # dx_t = g dw_t, where g = sigma_t * g_scaling
    # hence, g * score_of_p = - noise * g_scaling
    g = compute_ve_g_scale(s_min, s_max)
    label = - noise * g
    if return_scale:
        return analytic_xs, label, g
    else:
        return analytic_xs, label

def compute_ve_xs_label_tdim2(opt, x0, sigmas, samp_t_idx, return_scale):
    """ return xs.shape == [batch_x, *x_dim]  """
    s_max = opt.sigma_max
    s_min = opt.sigma_min
    x_dim = opt.data_dim

    assert x_dim == list(x0.shape[1:])
    assert x0.shape[0] == samp_t_idx.shape[0]
    batch_x, batch_t = samp_t_idx.shape

    # p(x_t|x_0) = N(x_0, sigma_t^2)
    # x_t = x_0 + sigma_t * noise
    noise = torch.randn(batch_x, batch_t, *x_dim, device=opt.device)
    # (batch_x,batch_t,1,1,1)
    sigma_t = sigmas[samp_t_idx].reshape(batch_x,batch_t,*([1,]*len(x_dim)))
    analytic_xs = sigma_t * noise + x0[:,None,...]

    # score_of_p = -1/sigma_t^2 (x_t - x_0) = -noise/sigma_t
    # dx_t = g dw_t, where g = sigma_t * g_scaling
    # hence, g * score_of_p = - noise * g_scaling
    g = compute_ve_g_scale(s_min, s_max)
    label = - noise * g
    if return_scale:
        return analytic_xs, label, g
    else:
        return analytic_xs, label


def compute_vp_xs_label(opt, x0, sqrt_betas, mean_scales, samp_t_idx, return_scale=False):
    if samp_t_idx.dim() == 1:
        return compute_vp_xs_label_tdim1(opt, x0, sqrt_betas, mean_scales, samp_t_idx, return_scale)
    elif samp_t_idx.dim() == 2:
        return compute_vp_xs_label_tdim2(opt, x0, sqrt_betas, mean_scales, samp_t_idx, return_scale)

def compute_vp_xs_label_tdim1(opt, x0, sqrt_betas, mean_scales, samp_t_idx, return_scale):
    """ return xs.shape == [batch_x, batch_t, *x_dim]
    Forward process p(xs|x0)
    """
    x_dim = opt.data_dim

    assert x_dim==list(x0.shape[1:])
    batch_x, batch_t = x0.shape[0], len(samp_t_idx)

    # p(x_t|x_0) = N(mean_scale * x_0, std_t^2)
    # x_t = mean_scale * x_0 + std_t * noise
    noise = torch.randn(batch_x, batch_t, *x_dim).to(opt.device)
    # shape = [1,batch_t,1,1,1]
    mean_scale_t = mean_scales[samp_t_idx].reshape(1,batch_t,*([1,]*len(x_dim)))
    std_t = torch.sqrt(1 - mean_scale_t**2)
    analytic_xs = std_t * noise + mean_scale_t * x0[:,None,...]

    # score_of_p = -1/sigma_t^2 (x_t - x_0) = -noise/sigma_t
    # g = sqrt_beta_t.
    # score_of_p = -1/std_t^2 (x_t - mean_scale_t * x_0) = -noise/std_t
    # hence, g * score_of_p = - noise / std_t * sqrt_beta_t  # Yu Chen. noise/std is not score??!!
    # shape = [1,batch_t,1,1,1]
    g_t = sqrt_betas[samp_t_idx].reshape(1,batch_t,*([1,]*len(x_dim)))
    noise_scale = 1 / std_t * g_t
    label = - noise * noise_scale
    if return_scale:
        return analytic_xs, label, noise_scale
    else:
        return analytic_xs, label


def compute_vp_xs_label_tdim2(opt, x0, sqrt_betas, mean_scales, samp_t_idx, return_scale):
    """ return xs.shape == [batch_x, batch_t, *x_dim]  """

    x_dim = opt.data_dim

    assert x_dim == list(x0.shape[1:])
    assert x0.shape[0] == samp_t_idx.shape[0]
    batch_x, batch_t = samp_t_idx.shape

    # p(x_t|x_0) = N(mean_scale * x_0, std_t^2)
    # x_t = mean_scale * x_0 + std_t * noise
    noise = torch.randn(batch_x, batch_t, *x_dim).to(opt.device)
    # shape = [batch_x,batch_t,1,1,1]
    mean_scale_t = mean_scales[samp_t_idx].reshape(batch_x,batch_t,*([1,]*len(x_dim)))
    std_t = torch.sqrt(1 - mean_scale_t**2)
    analytic_xs = std_t * noise + mean_scale_t * x0[:,None,...]

    # score_of_p = -1/sigma_t^2 (x_t - x_0) = -noise/sigma_t
    # score_of_p = -1/std_t^2 (x_t - mean_scale_t * x_0) = -noise/std_t
    # hence, g * score_of_p = - noise / std_t * sqrt_beta_t
    # shape = [batch_x,batch_t,1,1,1]
    g_t = sqrt_betas[samp_t_idx].reshape(batch_x,batch_t,*([1,]*len(x_dim)))
    noise_scale = 1 / std_t * g_t
    label = - noise * noise_scale
    if return_scale:
        return analytic_xs, label, noise_scale
    else:
        return analytic_xs, label


def get_xs_label_computer(opt, ts):
    # Forward process.
    if opt.sde_type == 'vp':
        mean_scales = compute_vp_kernel_mean_scale(ts, opt.beta_min, opt.beta_max)
        g = compute_vp_diffusion(ts, opt.beta_min, opt.beta_max)
        fn = compute_vp_xs_label
        kwargs = dict(opt=opt, sqrt_betas=g, mean_scales=mean_scales)

    elif opt.sde_type == 'vp_v2':
        dt = ts[1]-ts[0]
        # forward proces. mean_scales = sqrt{\alpha_bar}
        mean_scales = compute_vp_v2_kernel_mean_scale(ts, dt, opt.beta_min, opt.beta_max)
        # forward proces. sqrt_betas is g only used to scale the score to get z.
        g = compute_vp_diffusion(ts, opt.beta_min, opt.beta_max)
        fn = compute_vp_xs_label
        kwargs = dict(opt=opt, sqrt_betas=g, mean_scales=mean_scales)

    elif opt.sde_type == 've':
        sigmas = compute_sigmas(ts, opt.sigma_min, opt.sigma_max)
        fn = compute_ve_xs_label
        kwargs = dict(opt=opt, sigmas=sigmas)

    else:
        raise NotImplementedError('New SDE type.')

    return partial(fn, **kwargs)
