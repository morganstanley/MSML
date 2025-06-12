
import os, time, gc

import numpy as np

import torch
import torch.nn.functional as F
from torch.optim import SGD, RMSprop, Adagrad, AdamW, lr_scheduler, Adam
from torch.utils.tensorboard import SummaryWriter
from torch_ema import ExponentialMovingAverage

import policy
import sde_cld
import sde_ld
from loss import compute_sb_nll_alternate_train
import data
import util

from ipdb import set_trace as debug

def build_optimizer_ema_sched(opt, policy):
    direction = policy.direction

    optim_name = {
        'Adam': Adam,
        'AdamW': AdamW,
        'Adagrad': Adagrad,
        'RMSprop': RMSprop,
        'SGD': SGD,
    }.get(opt.optimizer)

    optim_dict = {
            "lr": opt.lr_f if direction=='forward' else opt.lr_b,
            'weight_decay': opt.l2_norm,
    }
    if opt.optimizer == 'SGD':
        optim_dict['momentum'] = 0.9
    
    optimizer = optim_name(policy.parameters(), **optim_dict)

    ema = ExponentialMovingAverage(policy.parameters(), decay=0.99) 
    cur_lr_gamma = opt.lr_gamma_f if direction=='forward' else opt.lr_gamma_b
    cur_lr_step = opt.lr_step_f if direction=='forward' else opt.lr_step_b
    if cur_lr_gamma < 1.0:
        sched = lr_scheduler.StepLR(optimizer, step_size=cur_lr_step, gamma=cur_lr_gamma)
    else:
        sched = None

    return optimizer, ema, sched

def freeze_policy(policy):
    for p in policy.parameters():
        p.requires_grad = False
    policy.eval()
    return policy

def activate_policy(policy):
    for p in policy.parameters():
        p.requires_grad = True
    
    for name, param in policy.net.named_parameters():
        if 'v_module' in name:
            param.requires_grad = False
    policy.train()
    return policy

class Runner():
    def __init__(self, opt):
        super(Runner, self).__init__()

        self.start_time=time.time()
        self.ts = torch.linspace(opt.t0, opt.T, opt.interval)
        self.opt = opt
        # build boundary distribution (p: target, q: prior)
        self.p, self.q, self.v = data.build_boundary_distribution(opt)

        # build dynamics, forward (z_f) and backward (z_b) policies
        if opt.diffusion == 'CLD':
            self.dyn = sde_cld.build(opt, self.p, self.q, self.v)
        else:
            self.dyn = sde_ld.build(opt, self.p, self.q)
        self.z_f = policy.build(opt, self.dyn, 'forward')  # p -> q
        self.z_b = policy.build(opt, self.dyn, 'backward') # q -> p
    
        self.optimizer_f, self.ema_f, self.sched_f = build_optimizer_ema_sched(opt, self.z_f)
        self.optimizer_b, self.ema_b, self.sched_b = build_optimizer_ema_sched(opt, self.z_b)

        self.adaptive_prior = opt.adaptive_prior

        if opt.log_tb: # tensorboard related things
            self.it_f = 0
            self.it_b = 0
            self.writer=SummaryWriter(
                log_dir=os.path.join('others/runs', opt.log_fn) if opt.log_fn is not None else None
            )

    def update_count(self, direction):
        if direction == 'forward':
            self.it_f += 1
            return self.it_f
        elif direction == 'backward':
            self.it_b += 1
            return self.it_b
        else:
            raise RuntimeError()

    def get_optimizer_ema_sched(self, z):
        if z == self.z_f:
            return self.optimizer_f, self.ema_f, self.sched_f
        elif z == self.z_b:
            return self.optimizer_b, self.ema_b, self.sched_b
        else:
            raise RuntimeError()

    @torch.no_grad()
    def sample_train_data(self, opt, policy_opt, policy_impt, reused_sampler):
        train_ts = self.ts

        # reuse or sample training xs and zs
        try:
            reused_traj = next(reused_sampler)
            train_xs, train_zs = reused_traj[:,0,...], reused_traj[:,1,...]
            print('generate train data from [{}]!'.format(util.green('reused samper')))
        except:
            _, ema, _ = self.get_optimizer_ema_sched(policy_opt)
            _, ema_impt, _ = self.get_optimizer_ema_sched(policy_impt)
            with ema.average_parameters(), ema_impt.average_parameters():
                policy_impt = freeze_policy(policy_impt)
                policy_opt  = freeze_policy(policy_opt)

                corrector = (lambda x,t: policy_impt(x,t) + policy_opt(x,t)) if opt.use_corrector else None
                xs, zs, _ = self.dyn.sample_traj(train_ts, policy_impt, corrector=corrector)
                train_xs = xs.detach().cpu(); del xs
                train_zs = zs.detach().cpu(); del zs
            print('generate train data from [{}]!'.format(util.red('sampling')))

        assert train_xs.shape[0] == opt.samp_bs
        assert train_xs.shape[1] == len(train_ts)
        assert train_xs.shape[:2] == train_zs.shape[:2]
        if opt.diffusion == 'CLD':
            assert train_xs.shape[2] == 2 * train_zs.shape[2]
        else:
            assert train_xs.shape[2] == train_zs.shape[2]
        gc.collect()

        return train_xs, train_zs, train_ts

    def sb_alternate_train_stage(self, opt, stage, epoch, direction, reused_sampler=None):
        policy_opt, policy_impt = {
            'forward':  [self.z_f, self.z_b], # train forwad,   sample from backward
            'backward': [self.z_b, self.z_f], # train backward, sample from forward
        }.get(direction)

        for ep in range(epoch):
            # prepare training data
            train_xs, train_zs, train_ts = self.sample_train_data(
                opt, policy_opt, policy_impt, reused_sampler
            )

            # train one epoch
            policy_impt = freeze_policy(policy_impt)
            policy_opt = activate_policy(policy_opt)
            self.sb_alternate_train_ep(
                opt, ep, stage, direction, train_xs, train_zs, train_ts, policy_opt, epoch
            )

    def sb_alternate_train_ep(
        self, opt, ep, stage, direction, train_xs, train_zs, train_ts, policy, num_epoch
    ):
        assert train_xs.shape[0] == opt.samp_bs
        assert train_zs.shape[0] == opt.samp_bs
        assert train_ts.shape[0] == opt.interval
        assert direction == policy.direction

        optimizer, ema, sched = self.get_optimizer_ema_sched(policy)

        for it in range(opt.num_itr):
            # -------- sample x_idx and t_idx \in [0, interval] --------
            samp_x_idx = torch.randint(opt.samp_bs,  (opt.train_bs_x,))
            samp_t_idx = torch.randint(opt.interval, (opt.train_bs_t,))
            if opt.use_arange_t: samp_t_idx = torch.arange(opt.interval)

            # -------- build sample --------
            ts = train_ts[samp_t_idx].detach()

            train_xs = train_xs.to(opt.device)
            train_zs = train_zs.to(opt.device)

            xs = train_xs[samp_x_idx][:, samp_t_idx, ...].to(opt.device)
            zs_impt = train_zs[samp_x_idx][:, samp_t_idx, ...].to(opt.device)
            optimizer.zero_grad()
            xs.requires_grad_(True) # we need this due to the later autograd.grad

            # -------- handle for batch_x and batch_t ---------
            # (batch, T, xdim) --> (batch*T, xdim)
            xs      = util.flatten_dim01(xs)
            zs_impt = util.flatten_dim01(zs_impt)
            ts = ts.repeat(opt.train_bs_x)
            assert xs.shape[0] == ts.shape[0]
            assert zs_impt.shape[0] == ts.shape[0]


            # -------- compute loss and backprop --------
            loss, zs = compute_sb_nll_alternate_train(
                opt, self.dyn, ts, xs, zs_impt, policy, return_z=True
            )

            assert not torch.isnan(loss)

            loss.backward()
            if opt.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), opt.grad_clip)
            optimizer.step()
            ema.update()
            
            if sched is not None: sched.step()

            # -------- logging --------
            zs = util.unflatten_dim01(zs, [len(samp_x_idx), len(samp_t_idx)])
            zs_impt = zs_impt.reshape(zs.shape)
            self.log_sb_alternate_train(
                opt, it, ep, stage, loss, zs, zs_impt, optimizer, direction, num_epoch
            )
        

    def dsm_train(self, opt, ep, stage, num_itr, batch_x, batch_t):
        policy = activate_policy(self.z_b)
        optimizer, ema, sched = self.optimizer_b, self.ema_b, self.sched_b

        with self.ema_f.average_parameters():
            At = self.z_f.net.At(self.ts).detach().clone()
            t_out = self.z_f.net.t_out.detach().clone()
            prior_covariance, Dt, sqrt_betas, mean_scales, L, invL = sde_ld.cache_dynamics(opt, self.ts, At)
            compute_xs_label = sde_ld.get_xs_label_computer(opt, sqrt_betas, mean_scales, At, L, invL)
            if stage % opt.snapshot_freq == 0:
                print(util.green(f'Sigma {self.z_f.net.Sigma.detach()}'))

            if opt.forward_net.startswith('Linear'):
                self.prior_covariance = prior_covariance
                print(util.green(f'Update prior covaraince\n {self.prior_covariance}'))
                if self.adaptive_prior:
                    print(util.yellow(f'Update adaptive prior'))
                    self.p, self.q = data.build_boundary_distribution(opt, self.prior_covariance)
                    self.dyn = sde_ld.build(opt, self.p, self.q)
            else:
                print(util.yellow(f'Maintain static prior'))

        p = data.build_data_sampler(opt, batch_x)

        for it in range(num_itr):

            x0 = p.sample()
            if x0.shape[0]!=batch_x:
                continue

            samp_t_idx = torch.randint(opt.interval, (batch_t,))
            ts = self.ts[samp_t_idx].detach()

            xs, label = compute_xs_label(x0=x0, samp_t_idx=samp_t_idx)

            # -------- handle for image ---------
            if util.is_image_dataset(opt):
                # (batch, T, xdim) --> (batch*T, xdim)
                xs = util.flatten_dim01(xs)
                ts = ts.repeat(batch_x)
                assert xs.shape[0] == ts.shape[0]

            optimizer.zero_grad()

            predict = policy(xs,ts)

            loss = F.mse_loss(label.reshape_as(predict), predict)
            loss.backward()

            if opt.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), opt.grad_clip)

            optimizer.step()
            ema.update()
            if sched is not None: sched.step()
            self.log_dsm_train(opt, it, ep, stage, loss, optimizer, num_itr)


    def cld_train(self, opt, ep, stage, num_itr, batch_x, batch_t):
        policy = activate_policy(self.z_b)
        optimizer, ema, sched = self.optimizer_b, self.ema_b, self.sched_b
        
        with self.ema_f.average_parameters():
            Axt = self.z_f.net.x_net.At(self.ts).detach().clone()
            Avt = self.z_f.net.v_net.At(self.ts).detach().clone()
            At = torch.cat([Axt, Avt], dim=-1)
            prior_covariance, Dt, sqrt_betas, mean_scales, L, invL = sde_cld.cache_dynamics(opt, self.ts, At)
            compute_xs_label = sde_cld.get_xs_label_computer(opt, sqrt_betas, mean_scales, At, L, invL)
            if stage % opt.snapshot_freq == 0:
                print(util.green(f'X Sigma {self.z_f.net.x_net.Sigma.detach()}'))
                print(util.green(f'V Sigma {self.z_f.net.v_net.Sigma.detach()}'))

            if opt.forward_net.startswith('Linear'):
                self.prior_covariance = prior_covariance
                print(util.green(f'Update prior covaraince\n {self.prior_covariance}'))
                if self.adaptive_prior:
                    print(util.yellow(f'Update adaptive prior'))
                    self.p, self.q = data.build_boundary_distribution(opt, self.prior_covariance)
                    self.dyn = sde_cld.build(opt, self.p, self.q)
            else:
                print(util.yellow(f'Maintain static prior'))
            
        p = data.build_data_sampler(opt, batch_x)
        v = data.build_prior_sampler(opt, batch_x)
        for it in range(num_itr):
            x0 = p.sample()
            v0 = v.sample()
            a0 = torch.cat([x0, v0], dim=-1)
            if a0.shape[0] != batch_x:
                continue

            samp_t_idx = torch.randint(opt.interval, (batch_t,))
            ts = self.ts[samp_t_idx].detach()

            augs, v_label = compute_xs_label(a0=a0, samp_t_idx=samp_t_idx)
            augs = util.flatten_dim01(augs)
            v_label = util.flatten_dim01(v_label)

            ts = ts.repeat(opt.train_bs_x_dsm)
            optimizer.zero_grad()
            predict = policy(augs, ts)
            loss = F.mse_loss(v_label.reshape_as(predict), predict)
            loss.backward()

            if opt.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), opt.grad_clip)

            optimizer.step()
            ema.update()
            if sched is not None: 
                sched.step()
            self.log_dsm_train(opt, it, ep, stage, loss, optimizer, num_itr)

    def backward_train_stage(self, opt, stage):
        for ep in range(opt.num_epoch):
            if opt.diffusion == 'CLD':
                self.cld_train(
                    opt, ep, stage, num_itr=opt.num_itr_dsm, batch_x=opt.train_bs_x_dsm, batch_t=opt.train_bs_t_dsm,
                )
            else:
                self.dsm_train(
                    opt, ep, stage, num_itr=opt.num_itr_dsm, batch_x=opt.train_bs_x_dsm, batch_t=opt.train_bs_t_dsm,
                )

    

    def sb_alternate_train(self, opt):
        for stage in range(opt.num_stage):
            forward_ep = backward_ep = opt.num_epoch

            self.backward_train_stage(opt, stage)

            # reuse evaluated trajectories for training forward policy
            n_reused_trajs = forward_ep * opt.samp_bs if opt.reuse_traj else 0
            reused_sampler = self.evaluate(opt, stage+1, n_reused_trajs=n_reused_trajs)

            if opt.baseline:
                print('The baseline computation is completed!')
                break

            # train forward policy (fix the forward policy in the last stages)
            self.sb_alternate_train_stage(
                opt, stage, forward_ep, 'forward', reused_sampler=reused_sampler
            )

        if opt.log_tb: self.writer.close()


    @torch.no_grad()
    def _generate_samples_and_reused_trajs(self, opt, batch, n_samples, n_trajs, plot_multi_steps=False):
        assert n_trajs <= n_samples

        ts = self.ts
        xTs = torch.empty((n_samples, *opt.data_dim), device='cpu')
        if n_trajs > 0:
            trajs = torch.empty((n_trajs, 2, len(ts), *opt.data_dim), device='cpu')
        else:
            trajs = None

        with self.ema_f.average_parameters(), self.ema_b.average_parameters():
            self.z_f = freeze_policy(self.z_f)
            self.z_b = freeze_policy(self.z_b)
            corrector = (lambda x,t: self.z_f(x,t) + self.z_b(x,t)) if opt.use_corrector else None

            it = 0
            while it < n_samples:
                # sample backward trajs; save traj if needed
                save_traj = (trajs is not None) and it < n_trajs

                if plot_multi_steps:
                    _xs, _zs, _x_T = self.dyn.sample_traj(
                        ts, self.z_b, corrector=corrector, save_traj=plot_multi_steps)
                else:
                    _xs, _zs, _x_T = self.dyn.sample_traj(
                        ts, self.z_b, corrector=corrector, save_traj=save_traj)

                # fill xTs (for FID usage) and trajs (for training log_q)
                xTs[it:it+batch,...] = _x_T.detach().cpu()[0:min(batch,xTs.shape[0]-it),...]
                if save_traj:
                    trajs[it:it+batch,0,...] = _xs.detach().cpu()[0:min(batch,trajs.shape[0]-it),...]
                    trajs[it:it+batch,1,...] = _zs.detach().cpu()[0:min(batch,trajs.shape[0]-it),...]

                it += batch
        if plot_multi_steps:
            print(util.yellow(f'Wei try to visualize multiple steps. Set plot_multi_steps=False can recover Chen version.'))
            xTs_list = []
            for my_rate in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
                pick_steps = int(my_rate * (opt.interval-1)) # 0 is data and -1 is Gaussian
                xTs_cur = xTs.clone()
                xTs_cur[:_xs.shape[0],...] = _xs.detach().cpu()[:_xs.shape[0], pick_steps, ...]
                xTs_list.append(xTs_cur)
            return xTs_list, trajs
        return xTs, trajs

    @torch.no_grad()
    def compute_NLL(self, opt, prior_covariance):
        num_NLL_sample = opt.samp_bs
        bpds=[]
        with self.ema_f.average_parameters(), self.ema_b.average_parameters():
            for _ in range(int(num_NLL_sample/opt.samp_bs)):
                bits_per_dim = self.dyn.compute_nll(opt.samp_bs, self.ts, self.z_f, self.z_b, prior_covariance)
                bpds.append(bits_per_dim.detach().cpu().numpy())

        print(util.yellow("=================== NLL={} ======================").format(np.array(bpds).mean()))

    @torch.no_grad()
    def evaluate(self, opt, stage, n_reused_trajs=0, metrics=None):
        if util.is_toy_dataset(opt):
            _, snapshot = util.evaluate_stage(opt, stage, metrics=None)
            if snapshot:
                real_data = None
                pred_data = None
                for idx, z in enumerate([self.z_f, self.z_b]):
                    z = freeze_policy(z)
                    if opt.forward_net.startswith('Linear'):
                        adaptive_prior = data.build_prior_sampler(self.opt, self.opt.samp_bs, Cov=self.prior_covariance)
                        if z.direction == 'backward':
                            print(util.green('Probability ODE for backward propagation!'))
                            z_f = freeze_policy(self.z_f)
                            if opt.diffusion == 'CLD':
                                augs_ode, _, _ = self.dyn.sample_traj(self.ts, z, save_traj=True, adaptive_prior=adaptive_prior, policy_f=z_f)
                                xs_ode, vs_ode = torch.chunk(augs_ode, 2, dim=2)
                            else:
                                xs_ode, _, _ = self.dyn.sample_traj(self.ts, z, save_traj=True, adaptive_prior=adaptive_prior, policy_f=z_f)
                            if opt.ode:
                                fn = "stage{}-{}-ode".format(stage, z.direction)
                                util.save_toy_npy_traj(
                                    opt, fn, xs_ode.detach().cpu().numpy(), n_snapshot=4, direction=z.direction
                                )
                        augs, _, _ = self.dyn.sample_traj(self.ts, z, save_traj=True, adaptive_prior=adaptive_prior)
                    else:
                        augs, _, _ = self.dyn.sample_traj(self.ts, z, save_traj=True)

                    if opt.diffusion == 'CLD':
                        xs, vs = torch.chunk(augs, 2, dim=2)
                    else:
                        xs = augs
                    fn = "stage{}-{}-sde".format(stage, z.direction)
                    sub_xs_plot = xs[:opt.samp_bs_plot, :, :]
                    util.save_toy_npy_traj(
                        opt, fn, sub_xs_plot.detach().cpu().numpy(), n_snapshot=4, direction=z.direction
                    )
                    if idx == 0:
                        real_data = xs.detach().cpu().numpy()[:, 0, :]
                    else:
                        pred_data = xs.detach().cpu().numpy()[:, 0, :]

                x_min, x_max = util.compute_axis_limits(real_data[:, 0])
                y_min, y_max = util.compute_axis_limits(real_data[:, 1])

                pmf_real = util.data_to_pmf(real_data, x_min, x_max, y_min, y_max)
                pmf_pred = util.data_to_pmf(pred_data, x_min, x_max, y_min, y_max)
                
                rmse = util.root_mean_squared_error(pmf_real, pmf_pred)
                print(util.yellow(f"=================== PMF distance={rmse:.2e} ======================"))
                


    def _print_train_itr(self, opt, it, ep, stage, loss, optimizer, num_itr, name):
        time_elapsed = util.get_time(time.time()-self.start_time)
        lr = optimizer.param_groups[0]['lr']
        print("[{0}] stage {1}/{2} | ep {3}/{4} | train_it {5}/{6} | lr:{7} | loss:{8} | time:{9}"
            .format(
                util.magenta(name),
                util.cyan("{}".format(1+stage)),
                opt.num_stage,
                util.cyan("{}".format(1+ep)),
                opt.num_epoch,
                util.cyan("{}".format(1+it+num_itr*ep)),
                num_itr*opt.num_epoch,
                util.yellow("{:.2e}".format(lr)),
                util.red("{:.4f}".format(loss.item())),
                util.green("{0}:{1:02d}:{2:05.2f}".format(*time_elapsed)),
        ))

    def log_dsm_train(self, opt, it, ep, stage, loss, optimizer, num_itr):
        if it % 100 != 0:
            return
        syntax = 'CLD backward' if opt.diffusion == 'CLD' else 'DSM backward'
        self._print_train_itr(opt, it, ep, stage, loss, optimizer, num_itr, name=syntax)
        if opt.log_tb:
            step = self.update_count('backward')
            self.log_tb(step, loss.detach(), 'loss', f'{opt.diffusion}_backward')

    def log_sb_alternate_train(self, opt, it, ep, stage, loss, zs, zs_impt, optimizer, direction, num_epoch):
        time_elapsed = util.get_time(time.time()-self.start_time)
        lr = optimizer.param_groups[0]['lr']
        if it % 100 != 0:
            return
        print("[{0}] stage {1}/{2} | ep {3}/{4} | train_it {5}/{6} | lr:{7} | loss:{8} | time:{9}"
            .format(
                util.magenta("SB {}".format(direction)),
                util.cyan("{}".format(1+stage)),
                opt.num_stage,
                util.cyan("{}".format(1+ep)),
                num_epoch,
                util.cyan("{}".format(1+it+opt.num_itr*ep)),
                opt.num_itr*num_epoch,
                util.yellow("{:.2e}".format(lr)),
                util.red("{:+.4f}".format(loss.item())),
                util.green("{0}:{1:02d}:{2:05.2f}".format(*time_elapsed)),
        ))
        if opt.log_tb:
            step = self.update_count(direction)
            neg_elbo = loss + util.compute_z_norm(zs_impt, self.dyn.dt)
            self.log_tb(step, loss.detach(), 'loss', 'SB_'+direction) # SB surrogate loss (see Eq 18 & 19 in the paper)
            self.log_tb(step, neg_elbo.detach(), 'neg_elbo', 'SB_'+direction) # negative ELBO (see Eq 16 in the paper)

    def log_tb(self, step, val, name, tag):
        self.writer.add_scalar(os.path.join(tag,name), val, global_step=step)

