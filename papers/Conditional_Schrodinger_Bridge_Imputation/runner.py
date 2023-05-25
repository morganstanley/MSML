
import os, time, gc

import numpy as np
import sys
from tqdm import tqdm
import datetime as dt

import torch
import torch.nn.functional as F
# from torch.optim import SGD, RMSprop, Adagrad, AdamW, lr_scheduler, Adam
from torch.utils.tensorboard import SummaryWriter
from torch_ema import ExponentialMovingAverage

import policy
import sde
from loss import (compute_sb_nll_alternate_train,
                  compute_sb_nll_joint_train,
                  compute_sb_nll_alternate_imputation_train)
import data
import util
from warmup_scheduler.scheduler import GradualWarmupScheduler
from ipdb import set_trace as debug

# dataloader issue. Solution found here https://github.com/pytorch/pytorch/issues/11201
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
print('set torch.multiprocessing.')

def build_optimizer_ema_sched(opt, policy):
    direction = policy.direction
    optim_name = {
        'Adam': torch.optim.Adam,
        'AdamW': torch.optim.AdamW,
        'Adagrad': torch.optim.Adagrad,
        'RMSprop': torch.optim.RMSprop,
        'SGD': torch.optim.SGD,
    }.get(opt.optimizer)

    warmup_multiplier = opt.warmup_multiplier if opt.warmup_lr_step > 0 else 1
    optim_dict = {
        "lr": opt.lr_f/warmup_multiplier if direction=='forward' else opt.lr_b/warmup_multiplier,
        'weight_decay':opt.l2_norm,
    }
    if opt.optimizer == 'SGD':
        optim_dict['momentum'] = 0.9

    optimizer = optim_name(policy.parameters(), **optim_dict)
    ema = ExponentialMovingAverage(policy.parameters(), decay=opt.ema_decay)  # 0.99

    # the LR decays to 10% after opt.lr_step epochs.
    after_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1**(1/opt.lr_step))
    if opt.warmup_lr_step > 0:
        scheduler = GradualWarmupScheduler(optimizer, multiplier=warmup_multiplier,
            total_epoch=opt.warmup_lr_step, after_scheduler=after_scheduler)
    else:
        scheduler = after_scheduler

    return optimizer, ema, scheduler

def freeze_policy(policy):
    for p in policy.parameters():
        p.requires_grad = False
    policy.eval()
    return policy

def activate_policy(policy):
    for p in policy.parameters():
        p.requires_grad = True
    policy.train()
    return policy

class Runner():
    def __init__(self,opt):
        super(Runner,self).__init__()

        self.start_time=time.time()
        # [t0, ..., T] * interval = [t0*interval,...,T*interval] will get the right end.
        self.ts = torch.linspace(opt.t0, opt.T, opt.interval).to(opt.device)

        # build boundary distribution (p: target, q: prior)
        self.p, self.q = data.build_boundary_distribution(opt)  # p: data  q: prior.

        # build dynamics, forward (z_f) and backward (z_b) policies
        self.dyn = sde.build(opt, self.p, self.q)
        self.z_f = policy.build(opt, self.dyn, 'forward')  # p -> q
        self.z_b = policy.build(opt, self.dyn, 'backward') # q -> p

        self.optimizer_f, self.ema_f, self.sched_f = build_optimizer_ema_sched(opt, self.z_f)
        self.optimizer_b, self.ema_b, self.sched_b = build_optimizer_ema_sched(opt, self.z_b)

        if opt.load:
            util.restore_checkpoint(opt, self, opt.load)
            self.reset_optimizer_ema_sched(opt)

        if opt.log_tb: # tensorboard related things
            self.it_f = 0
            self.it_b = 0
            self.writer=SummaryWriter(log_dir=opt.log_fn)

        if opt.train_method == 'evaluation':
            timestamp = dt.datetime.now().strftime("%m_%d_%Y_%H%M%S")
            util.save_opt(opt, opt.ckpt_path + f'/opt_eval_{timestamp}.yaml')
        else:
            util.save_opt(opt, opt.ckpt_path + '/opt.yaml')

    def reset_optimizer_ema_sched(self, opt):
        print(util.red('Reset optimizer, ema, and scheduler.'))
        self.optimizer_f, self.ema_f, self.sched_f = build_optimizer_ema_sched(opt, self.z_f)
        self.optimizer_b, self.ema_b, self.sched_b = build_optimizer_ema_sched(opt, self.z_b)

    def reset_ema(self, opt):
        print(util.red('Reset ema.'))
        self.ema_f = ExponentialMovingAverage(self.z_f.parameters(), decay=opt.ema_decay)
        self.ema_b = ExponentialMovingAverage(self.z_f.parameters(), decay=opt.ema_decay)

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

####################################################################################################
#                           Schrodinger bridge likehood-based training
####################################################################################################

    @torch.no_grad()
    def sample_train_data(self, opt, policy_opt, policy_impt, reused_sampler):
        """
        for imputation, this is only used for updating the forward policy, i.e. draw traing
        samples using the backward policy. policy_opt = forward, policy_impt = backward
        """
        train_ts = self.ts

        # The backward process uses unconditioanl sampling as walking from the prior doesn't
        # involve any missing values. For unconditional sampling, also refer to sde.sample_traj()
        if opt.use_corrector and opt.backward_net in ['Transformerv2', 'Transformerv4',
            'Transformerv5']:
            def corrector(x, t):
                total_input = torch.cat([torch.zeros_like(x), x], dim=1)
                diff_input = (total_input, torch.zeros_like(x))
                return policy_opt(x,t) + policy_impt(diff_input, t)
        elif opt.use_corrector and opt.backward_net in ['Transformerv3']:
            def corrector(x, t):
                diff_input = (x, torch.zeros_like(x))
                return policy_opt(x,t) + policy_impt(diff_input, t)
        elif opt.use_corrector:
            def corrector(x, t):
                return policy_opt(x,t) + policy_impt(x,t)
        else:
            corrector = None

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

                xs, zs, _ = self.dyn.sample_traj(train_ts, policy_impt, corrector=corrector)
                train_xs = xs.detach().cpu(); del xs
                train_zs = zs.detach().cpu(); del zs
            # print('generate train data from [{}]!'.format(util.red('sampling')))

        assert train_xs.shape[0] == opt.samp_bs
        assert train_xs.shape[1] == len(train_ts)
        assert train_xs.shape == train_zs.shape
        gc.collect()

        return train_xs, train_zs, train_ts


    @torch.no_grad()
    def sample_imputation_forward_train_data(self, opt, policy_opt, policy_impt, reused_sampler):
        """Also check sde.sample_traj for unconditional sampling."""
        train_ts = self.ts
        # policy_opt = backward, policy_impt = forward

        if opt.use_corrector and opt.backward_net in ['Transformerv2', 'Transformerv4',
            'Transformerv5']:
            def corrector(x, t, x_cond, cond_mask):
                cond_obs = cond_mask * x_cond
                noisy_target = (1-cond_mask) * x  # no big difference if using  target_mask * x
                total_input = torch.cat([cond_obs, noisy_target], dim=1)
                diff_input = (total_input, cond_mask)
                return policy_impt(x,t) + policy_opt(diff_input, t)
        elif opt.use_corrector and opt.backward_net in ['Transformerv3']:
            def corrector(x, t, x_cond, cond_mask):
                cond_obs = cond_mask * x_cond
                noisy_target = (1-cond_mask) * x  # no big difference if using  target_mask * x
                total_input = cond_obs + noisy_target
                diff_input = (total_input, cond_mask)
                return policy_impt(x,t) + policy_opt(diff_input, t)
        elif opt.use_corrector:
            def corrector(x, t, x_cond=None, cond_mask=None):
                return policy_impt(x,t) + policy_opt(x,t)
        else:
            corrector = None

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

                (xs, zs, x_term, obs_data, obs_mask, cond_mask, gt_mask
                    ) = self.dyn.sample_traj_imputation_forward(train_ts, policy_impt, corrector=corrector)
                train_xs = xs.detach().cpu(); del xs
                train_zs = zs.detach().cpu(); del zs
            # print('generate train data from [{}]!'.format(util.red('sampling')))

        assert train_xs.shape[0] == opt.samp_bs
        assert train_xs.shape[1] == len(train_ts)
        assert train_xs.shape == train_zs.shape
        gc.collect()

        return train_xs, train_zs, train_ts, obs_data, obs_mask, cond_mask, gt_mask


    def sb_alternate_train_stage(self, opt, stage, epoch, direction, reused_sampler=None):
        policy_opt, policy_impt = {
            'forward':  [self.z_f, self.z_b], # train forwad,   sample from backward
            'backward': [self.z_b, self.z_f], # train backward, sample from forward
        }.get(direction)

        for ep in range(epoch):
            # prepare training data
            train_xs, train_zs, train_ts = self.sample_train_data(
                opt, policy_opt, policy_impt, reused_sampler)

            # train one epoch
            policy_impt = freeze_policy(policy_impt)
            policy_opt = activate_policy(policy_opt)
            self.sb_alternate_train_ep(
                opt, ep, stage, direction, train_xs, train_zs, train_ts, policy_opt, epoch)


    def sb_alternate_imputation_train_stage(self, opt, stage, epoch, direction, reused_sampler=None):
        assert direction == 'backward'
        policy_opt, policy_impt = self.z_b, self.z_f

        for ep in range(epoch):
            # prepare training data
            (train_xs, train_zs, train_ts, obs_data, obs_mask, cond_mask, gt_mask
                ) = self.sample_imputation_forward_train_data(opt, policy_opt, policy_impt, reused_sampler)

            # train one epoch
            policy_impt = freeze_policy(policy_impt)
            policy_opt = activate_policy(policy_opt)
            if opt.train_method in ['alternate_backward_imputation', 'alternate_imputation']:
                self.sb_alternate_imputation_train_ep(
                    opt, ep, stage, direction, train_xs, train_zs, train_ts,
                    obs_data, obs_mask, cond_mask, gt_mask,
                    policy_opt, epoch)

            elif opt.train_method in ['alternate_backward_imputation_v2', 'alternate_imputation_v2']:
                self.sb_alternate_imputation_train_ep_v2(
                    opt, ep, stage, direction, train_xs, train_zs, train_ts,
                    obs_data, obs_mask, cond_mask, gt_mask,
                    policy_opt, epoch)


    def sb_alternate_train_ep(
        self, opt, ep, stage, direction, train_xs, train_zs, train_ts, policy, num_epoch):
        assert train_xs.shape[0] == opt.samp_bs
        assert train_zs.shape[0] == opt.samp_bs
        assert train_ts.shape[0] == opt.interval
        assert direction == policy.direction

        optimizer, ema, sched = self.get_optimizer_ema_sched(policy)

        for it in range(opt.num_itr):
            optimizer.zero_grad()

            # -------- sample x_idx and t_idx \in [0, interval] --------
            samp_x_idx = torch.randint(opt.samp_bs,  (opt.train_bs_x,))
            samp_t_idx = torch.randint(opt.interval, (opt.train_bs_t,))
            if opt.use_arange_t: samp_t_idx = torch.arange(opt.interval)

            # -------- build sample --------
            ts = train_ts[samp_t_idx].detach()
            ts = ts.repeat(opt.train_bs_x)
            zs_impt = train_zs[samp_x_idx][:, samp_t_idx, ...].to(opt.device)
            xs = train_xs[samp_x_idx][:, samp_t_idx, ...].to(opt.device)
            xs.requires_grad_(True) # we need this due to the later autograd.grad

            # -------- handle for batch_x and batch_t ---------
            # (batch, T, xdim) --> (batch*T, xdim)
            xs      = util.flatten_dim01(xs)
            zs_impt = util.flatten_dim01(zs_impt)
            assert xs.shape[0] == ts.shape[0]
            assert zs_impt.shape[0] == ts.shape[0]

            # -------- compute loss and backprop --------
            loss, zs = compute_sb_nll_alternate_train(
                opt, self.dyn, ts, xs, zs_impt, policy, return_z=True)
            assert not torch.isnan(loss)

            loss.backward()

            if opt.grad_clip is not None:
                torch.nn.utils.clip_grad_norm(policy.parameters(), opt.grad_clip)
            optimizer.step()
            ema.update()

            # -------- logging --------
            zs = util.unflatten_dim01(zs, [len(samp_x_idx), len(samp_t_idx)])
            zs_impt = zs_impt.reshape(zs.shape)
            self.log_sb_alternate_train(
                opt, it, ep, stage, loss, zs, zs_impt, optimizer, direction, num_epoch)

        # Update shceduler every epoch.
        if sched is not None: sched.step()


    def sb_alternate_imputation_train_ep(
        self, opt, ep, stage, direction, train_xs, train_zs, train_ts,
        obs_data, obs_mask, cond_mask, gt_mask,
        policy, num_epoch):
        assert train_xs.shape[0] == opt.samp_bs
        assert train_zs.shape[0] == opt.samp_bs
        assert train_ts.shape[0] == opt.interval
        assert direction == policy.direction

        optimizer, ema, sched = self.get_optimizer_ema_sched(policy)

        for it in range(opt.num_itr):
            optimizer.zero_grad()

            # -------- sample x_idx and t_idx \in [0, interval] --------
            samp_x_idx = torch.randint(opt.samp_bs,  (opt.train_bs_x,))
            samp_t_idx = torch.randint(opt.interval, (opt.train_bs_t,))
            if opt.use_arange_t: samp_t_idx = torch.arange(opt.interval)

            # -------- build sample --------
            ts = train_ts[samp_t_idx].detach()
            ts = ts.repeat(opt.train_bs_x)
            zs_impt = train_zs[samp_x_idx][:, samp_t_idx, ...].to(opt.device)
            xs = train_xs[samp_x_idx][:, samp_t_idx, ...].to(opt.device)
            xs.requires_grad_(True) # we need this due to the later autograd.grad

            # -------- handle for batch_x and batch_t ---------
            obs_data_ = obs_data[samp_x_idx].unsqueeze(1).repeat(1,opt.train_bs_t,1,1,1)
            obs_mask_ = obs_mask[samp_x_idx].unsqueeze(1).repeat(1,opt.train_bs_t,1,1,1)
            cond_mask_ = cond_mask[samp_x_idx].unsqueeze(1).repeat(1,opt.train_bs_t,1,1,1)
            gt_mask_ = gt_mask[samp_x_idx].unsqueeze(1).repeat(1,opt.train_bs_t,1,1,1)
            # (batch, T, xdim) --> (batch*T, xdim)
            xs      = util.flatten_dim01(xs)
            zs_impt = util.flatten_dim01(zs_impt)
            obs_data_ = util.flatten_dim01(obs_data_)
            obs_mask_ = util.flatten_dim01(obs_mask_)
            cond_mask_ = util.flatten_dim01(cond_mask_)
            gt_mask_ = util.flatten_dim01(cond_mask_)
            assert xs.shape[0] == ts.shape[0]
            assert zs_impt.shape[0] == ts.shape[0]

            # -------- compute loss and backprop --------
            loss, zs = compute_sb_nll_alternate_imputation_train(
                opt, self.dyn, ts, xs, zs_impt,
                obs_data_, obs_mask_, cond_mask_, gt_mask_,
                policy, return_z=True)
            assert not torch.isnan(loss)
            loss.backward()

            if opt.grad_clip is not None:
                torch.nn.utils.clip_grad_norm(policy.parameters(), opt.grad_clip)
            optimizer.step()
            ema.update()

            # -------- logging --------
            zs = util.unflatten_dim01(zs, [len(samp_x_idx), len(samp_t_idx)])
            zs_impt = zs_impt.reshape(zs.shape)
            self.log_sb_alternate_train(
                opt, it, ep, stage, loss, zs, zs_impt, optimizer, direction, num_epoch)

        # Update scheluer every epoch.
        if sched is not None: sched.step()


    def sb_alternate_imputation_train_ep_v2(
        self, opt, ep, stage, direction, train_xs, train_zs, train_ts,
        obs_data, obs_mask, cond_mask, gt_mask,
        policy, num_epoch):
        """sample opt.train_bs_x*opt.train_bs_t independently without repeat, i.e. for each (x,t) 
        draw x and t independently."""
        assert train_xs.shape[0] == opt.samp_bs
        assert train_zs.shape[0] == opt.samp_bs
        assert train_ts.shape[0] == opt.interval
        assert direction == policy.direction

        optimizer, ema, sched = self.get_optimizer_ema_sched(policy)

        for it in range(opt.num_itr):
            optimizer.zero_grad()

            # -------- sample x_idx and t_idx \in [0, interval] --------
            samp_x_idx = torch.randint(opt.samp_bs,  (opt.train_bs_x*opt.train_bs_t,))
            samp_t_idx = torch.randint(opt.interval, (opt.train_bs_x*opt.train_bs_t,))

            # -------- build sample --------
            ts = train_ts[samp_t_idx].detach()
            zs_impt = train_zs[samp_x_idx, samp_t_idx, ...].to(opt.device)
            xs = train_xs[samp_x_idx, samp_t_idx, ...].to(opt.device)
            xs.requires_grad_(True) # we need this due to the later autograd.grad

            # -------- handle for batch_x and batch_t ---------
            # (batch, T, xdim) --> (batch*T, xdim)
            obs_data_ = obs_data[samp_x_idx]
            obs_mask_ = obs_mask[samp_x_idx]
            cond_mask_ = cond_mask[samp_x_idx]
            gt_mask_ = gt_mask[samp_x_idx]

            assert xs.shape[0] == ts.shape[0]
            assert zs_impt.shape[0] == ts.shape[0]

            # -------- compute loss and backprop --------
            loss, zs = compute_sb_nll_alternate_imputation_train(
                opt, self.dyn, ts, xs, zs_impt,
                obs_data_, obs_mask_, cond_mask_, gt_mask_,
                policy, return_z=True)
            assert not torch.isnan(loss)
            loss.backward()

            if opt.grad_clip is not None:
                torch.nn.utils.clip_grad_norm(policy.parameters(), opt.grad_clip)
            optimizer.step()
            ema.update()

            # -------- logging --------
            zs = util.unflatten_dim01(zs, [opt.train_bs_x, opt.train_bs_t])
            zs_impt = zs_impt.reshape(zs.shape)
            self.log_sb_alternate_train(
                opt, it, ep, stage, loss, zs, zs_impt, optimizer, direction, num_epoch)

        # Update scheluer every epoch.
        if sched is not None: sched.step()


    def sb_alternate_imputation_train_backward(self, opt):
        for stage in range(opt.num_stage):
            if stage == 0 and opt.DSM_warmup:
                self.dsm_train_first_stage(opt)

                # DSM parameters training trace should be different from SB.
                self.reset_optimizer_ema_sched(opt)
                # Change back the lr for SB alternative.
                # for g in self.optimizer_b.param_groups:
                #     g['lr'] = opt.lr_b
            else:
                self.sb_alternate_imputation_train_stage(opt, stage, opt.num_epoch, 'backward')

            if (opt.reset_ema_stage is not None and
                stage % opt.reset_ema_stage == opt.reset_ema_stage-1):
                self.reset_ema(opt)

            self.evaluate(opt, stage)
            keys = ['z_b','optimizer_b','ema_b']
            util.save_checkpoint(opt, self, keys, stage, suffix='fb')
        if opt.log_tb: self.writer.close()


    def sb_alternate_train_backward(self, opt):
        for stage in range(opt.num_stage):
            if stage == 0 and opt.DSM_warmup:
                self.dsm_train_first_stage(opt)

                # DSM parameters training trace should be different from SB.
                self.reset_optimizer_ema_sched(opt)
            else:
                self.sb_alternate_train_stage(opt, stage, opt.num_epoch, 'backward')

            if (opt.reset_ema_stage is not None and
                stage % opt.reset_ema_stage == opt.reset_ema_stage-1):
                self.reset_ema(opt)

            self.evaluate(opt, stage)
            keys = ['z_b','optimizer_b','ema_b']
            util.save_checkpoint(opt, self, keys, stage, suffix='fb')
        if opt.log_tb: self.writer.close()


    def sb_alternate_imputation_train(self, opt):
        """Training the backward needs conditional sampling as the observed data includes missing
        values; training the forward doesn't need conditional sampling as walking from the prior
        does not have missing values.
        """
        for stage in range(opt.num_stage):
            forward_ep = backward_ep = opt.num_epoch

            if stage == 0 and opt.DSM_warmup:
                self.dsm_train_first_stage(opt)
                if opt.backward_warmup_epoch > 0:
                    self.sb_alternate_imputation_train_stage(opt, stage, opt.backward_warmup_epoch, 'backward')

                # DSM parameters training trace should be different from SB.
                self.reset_optimizer_ema_sched(opt)
                forward_ep *= 3
            else:
                self.sb_alternate_imputation_train_stage(opt, stage, backward_ep, 'backward')

            # Eval right after updating the backward.
            self.evaluate(opt, stage)

            # Train forward.
            self.sb_alternate_train_stage(opt, stage, forward_ep, 'forward', reused_sampler=0)

            if (opt.reset_ema_stage is not None and
                stage % opt.reset_ema_stage == opt.reset_ema_stage-1):
                self.reset_ema(opt)

            keys = ['z_f','optimizer_f','ema_f', 'z_b','optimizer_b','ema_b']
            util.save_checkpoint(opt, self, keys, stage, suffix='fb')

        if opt.log_tb: self.writer.close()


    def sb_alternate_train(self, opt):
        for stage in range(opt.num_stage):
            forward_ep = backward_ep = opt.num_epoch

            # train backward policy;
            # skip the trainining at first stage if checkpoint is loaded
            train_backward = not (stage == 0 and opt.load is not None)
            if train_backward:
                if stage == 0 and opt.DSM_warmup:
                    self.dsm_train_first_stage(opt)

                    # A heuristic that, since DSM training can be quite long, we bump up
                    # the epoch of its following forward policy training, so that forward
                    # training can converge; otherwise it may mislead backward policy.
                    forward_ep *= 3 # for CIFAR10, this bump ep from 5 to 15

                    self.reset_optimizer_ema_sched(opt)
                else:
                    self.sb_alternate_train_stage(opt, stage, backward_ep, 'backward')

            # evaluate backward policy;
            # reuse evaluated trajectories for training forward policy
            n_reused_trajs = forward_ep * opt.samp_bs if opt.reuse_traj else 0
            reused_sampler = self.evaluate(opt, stage, n_reused_trajs=n_reused_trajs)

            # train forward policy
            self.sb_alternate_train_stage(
                opt, stage, forward_ep, 'forward', reused_sampler=reused_sampler)

            keys = ['z_f','optimizer_f','ema_f','z_b','optimizer_b','ema_b']
            util.save_checkpoint(opt, self, keys, stage, suffix='fb')

        if opt.log_tb: self.writer.close()

####################################################################################################
#                                         DSM training
####################################################################################################

    def dsm_train(self, opt, num_itr, batch_x, batch_t, stage=1):
        """ Our own implementation of dsm, support both VE and VP """

        policy = activate_policy(self.z_b)
        optimizer, ema, sched = self.optimizer_b, self.ema_b, self.sched_b
        # [batch_x, batch_t, *x_dim]
        compute_xs_label = sde.get_xs_label_computer(opt, self.ts)
        p = data.build_data_sampler(opt, batch_x)

        avg_loss = 0
        for it in range(1, num_itr+1):
            optimizer.zero_grad()
            x0 = p.sample().to(opt.device)
            if x0.shape[0]!=batch_x:
                continue

            if opt.dsm_train_method == 'dsm':
                samp_t_idx = torch.randint(opt.interval, (batch_t,))
                ts = self.ts[samp_t_idx].detach()
                ts = ts.repeat(batch_x)
            elif opt.dsm_train_method == 'dsm_v2':
                # Non-repeated timeline for each sample.
                samp_t_idx = torch.randint(opt.interval, (batch_x, batch_t))
                ts = self.ts[samp_t_idx].detach()  # (batch_x*batch_t)
                ts = ts.reshape(batch_x*batch_t)
            else:
                raise NotImplementedError(f'Unknow DSM training method:{opt.dsm_train_method}')

            xs, label, label_scale = compute_xs_label(x0=x0, samp_t_idx=samp_t_idx, return_scale=True)
            xs = util.flatten_dim01(xs)  # (batch, T, xdim) --> (batch*T, xdim)
            x0 = x0.unsqueeze(1).repeat(1,batch_t,1,1,1)
            x0 = util.flatten_dim01(x0)
            assert xs.shape[0] == ts.shape[0]

            if opt.backward_net in ['Transformerv2', 'Transformerv4', 'Transformerv5']:
                # Transformerv2 has to add random mask. Putting empty obs_mask or cond_obs here
                # will have bad results. I keep the wrong code here as a reminder.
                # total_input = torch.cat([xs, torch.zeros_like(xs)], dim=1)  # (B,2,K,L)  x, noise_target
                # diff_input = (total_input, torch.ones_like(xs))  # (x, cond_mask)
                obs_mask = torch.ones_like(xs)
                cond_mask = self.get_randmask(obs_mask, miss_ratio=opt.rand_mask_miss_ratio,
                    rank=opt.rand_mask_rank)  # Apply random mask here.
                target_mask = obs_mask - cond_mask
                cond_obs = cond_mask * x0
                noisy_target = (1 - cond_mask) * xs
                total_input = torch.cat([cond_obs, noisy_target], dim=1)
                diff_input = (total_input, cond_mask)
                loss_mask = target_mask  # don't use target_mask for fixed cond_mask.

            elif opt.backward_net == 'Transformerv3':
                obs_mask = torch.ones_like(xs)
                cond_mask = self.get_randmask(obs_mask, miss_ratio=opt.rand_mask_miss_ratio,
                    rank=opt.rand_mask_rank)  # Apply random mask here.
                target_mask = obs_mask - cond_mask
                cond_obs = cond_mask * x0
                noisy_target = (1 - cond_mask) * xs
                total_input = cond_obs + noisy_target
                diff_input = (total_input, cond_mask)
                loss_mask = target_mask  # don't use target_mask for fixed cond_mask.

            else:
                diff_input = xs
                loss_mask = torch.ones_like(xs)

            predicted = policy(diff_input, ts)
            label = label.reshape_as(predicted)

            if opt.normalize_loss:
                label_scale = label_scale.reshape(batch_x*batch_t,1,1,1)
                residual = (label - predicted) * loss_mask / label_scale
            else:
                residual = (label - predicted) * loss_mask
            num_eval = loss_mask.sum()
            loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)

            loss.backward()
            if opt.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), opt.grad_clip)

            optimizer.step()
            ema.update()

            avg_loss += loss.item()
            self.log_dsm_train(opt, it, loss.item(), avg_loss/it, optimizer, num_itr)

        if sched is not None: sched.step()
        keys = ['optimizer_b','ema_b','z_b']
        util.save_checkpoint(opt, self, keys, stage, suffix='dsm')


    @classmethod
    def get_randmask(cls, obs_mask, miss_ratio=None, rank=None):
        """Random Mask.

        Args:
            miss_ratio: miss_ratio == sample_ratio, for those who will be set to -1 and will NOT
                be used as cond_mask.
        """
        if rank is None:
            rand_for_mask = torch.rand_like(obs_mask) * obs_mask
            rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
            for i in range(len(obs_mask)):
                if miss_ratio is None:
                    sample_ratio = np.random.rand()  # missing ratio
                else:
                    sample_ratio = miss_ratio
                num_observed = obs_mask[i].sum().item()
                num_masked = round(num_observed * sample_ratio)
                rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
            cond_mask = (rand_for_mask > 0).reshape(obs_mask.shape).float()
        else:
            B, C, K, L = obs_mask.shape
            cond_mask = torch.ones_like(obs_mask)
            for i in range(B):
                sample_ratio = np.random.rand()*0.9 + 0.1
                mask_L = torch.bernoulli(sample_ratio**0.6 * torch.rand(K, rank)**0.3)
                mask_R = torch.bernoulli(sample_ratio**0.6 * torch.rand(rank, L)**0.3)
                miss_mask = (mask_L @ mask_R).bool().to(obs_mask.device)
                cond_mask[i] = obs_mask[i] * miss_mask.unsqueeze(0)  # (1,K,L)
        return cond_mask


    def dsm_imputation_train(self, opt, num_itr, batch_x, batch_t, stage=1):
        """Training with observation mask."""
        policy = activate_policy(self.z_b)
        optimizer, ema, sched = self.optimizer_b, self.ema_b, self.sched_b
        # [batch_x, batch_t, *x_dim]
        compute_xs_label = sde.get_xs_label_computer(opt, self.ts)
        p = data.build_data_sampler(opt, batch_x)

        avg_loss = 0.0
        for it in range(1, num_itr+1):
            optimizer.zero_grad()

            x0, obs_mask, gt_mask = p.sample(return_all_mask=True)
            x0, obs_mask, gt_mask = x0.to(opt.device), obs_mask.to(opt.device), gt_mask.to(opt.device)
            if x0.shape[0]!=batch_x:
                continue

            if opt.dsm_train_method == 'dsm_imputation':
                # Repeated timeline for each sample.
                samp_t_idx = torch.randint(opt.interval, (batch_t,))
                ts = self.ts[samp_t_idx].detach()  # (batch_t)
                ts = ts.repeat(batch_x)  # (batch_x*batch_t)  e.g (1,3,2,  1,3,2,  1,3,2,  1,3,2)
            elif opt.dsm_train_method == 'dsm_imputation_v2':
                # Non-repeated timeline for each sample.
                samp_t_idx = torch.randint(opt.interval, (batch_x, batch_t))
                ts = self.ts[samp_t_idx].detach()  # (batch_x*batch_t)
                ts = ts.reshape(batch_x*batch_t)

            xs, label, label_scale = compute_xs_label(x0=x0, samp_t_idx=samp_t_idx, return_scale=True)
            xs = util.flatten_dim01(xs)  # (batch, T, xdim) --> (batch*T, xdim)
            x0 = x0.unsqueeze(1).repeat(1,batch_t,1,1,1)
            x0 = util.flatten_dim01(x0)
            obs_mask = obs_mask.unsqueeze(1).repeat(1,batch_t,1,1,1)
            obs_mask = util.flatten_dim01(obs_mask)
            assert xs.shape[0] == ts.shape[0]

            if opt.backward_net in ['Transformerv2', 'Transformerv4', 'Transformerv5']:
                cond_mask = self.get_randmask(obs_mask, miss_ratio=opt.rand_mask_miss_ratio,
                    rank=opt.rand_mask_rank)  # Apply random mask here.
                target_mask = obs_mask - cond_mask
                cond_obs = cond_mask * x0
                noisy_target = (1 - cond_mask) * xs
                total_input = torch.cat([cond_obs, noisy_target], dim=1)
                diff_input = (total_input, cond_mask)
                loss_mask = target_mask  # don't use target_mask for fixed cond_mask.

            elif opt.backward_net == 'Transformerv3':
                cond_mask = self.get_randmask(obs_mask, miss_ratio=opt.rand_mask_miss_ratio,
                    rank=opt.rand_mask_rank)  # Apply random mask here.
                target_mask = obs_mask - cond_mask
                cond_obs = cond_mask * x0
                noisy_target = (1 - cond_mask) * xs
                total_input = cond_obs + noisy_target
                diff_input = (total_input, cond_mask)
                loss_mask = target_mask  # don't use target_mask for fixed cond_mask.

            else:
                diff_input = xs
                loss_mask = obs_mask

            predicted = policy(diff_input, ts)
            label = label.reshape_as(predicted)

            if opt.normalize_loss:
                label_scale = label_scale.reshape(batch_x*batch_t,1,1,1)
                residual = (label - predicted) * loss_mask / label_scale
            else:
                residual = (label - predicted) * loss_mask
            num_eval = loss_mask.sum()
            loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
            loss.backward()
            # if opt.grad_clip is not None:
            #     torch.nn.utils.clip_grad_norm_(policy.parameters(), opt.grad_clip)
            optimizer.step()
            ema.update()

            avg_loss += loss.item()
            self.log_dsm_train(opt, it, loss.item(), avg_loss/it, optimizer, num_itr)

        if sched is not None: sched.step()
        keys = ['optimizer_b','ema_b','z_b']
        util.save_checkpoint(opt, self, keys, stage, suffix='dsm')


    def dsm_imputation_train_forward_verfication(self, opt, num_itr, batch_x, batch_t, stage=1):
        """Training with observation mask."""
        self.beta = np.linspace(opt.beta_min*self.dyn.dt, opt.beta_max*self.dyn.dt, opt.interval)
        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(np.expand_dims(self.alpha, axis=[1,2])).float()

        policy = activate_policy(self.z_b)
        optimizer, ema, sched = self.optimizer_b, self.ema_b, self.sched_b
        # [batch_x, batch_t, *x_dim]
        compute_xs_label = sde.get_xs_label_computer(opt, self.ts)
        p = data.build_data_sampler(opt, batch_x)

        avg_loss = 0.0
        for it in range(1, num_itr+1):
            optimizer.zero_grad()

            x0, obs_mask, gt_mask = p.sample(return_all_mask=True)
            x0, obs_mask, gt_mask = x0.to(opt.device), obs_mask.to(opt.device), gt_mask.to(opt.device)
            if x0.shape[0]!=batch_x:
                continue

            if opt.dsm_train_method == 'dsm_imputation':
                # Repeated timeline for each sample.
                samp_t_idx = torch.randint(opt.interval, (batch_t,))
                ts = self.ts[samp_t_idx].detach()  # (batch_t)
                ts = ts.repeat(batch_x)  # (batch_x*batch_t)  e.g (1,3,2,  1,3,2,  1,3,2,  1,3,2)
            elif opt.dsm_train_method in ['dsm_imputation_v2', 'dsm_imputation_forward_verfication']:
                # Non-repeated timeline for each sample.
                samp_t_idx = torch.randint(opt.interval, (batch_x, batch_t))
                ts = self.ts[samp_t_idx].detach()  # (batch_x*batch_t)
                ts = ts.reshape(batch_x*batch_t)

            # Equivalent continuous setting. Note that the difference is the noise should be corrected
            # with some scale. Check compute_vp_xs_label for label creation.
            xs, label, label_scale = compute_xs_label(x0=x0, samp_t_idx=samp_t_idx, return_scale=True)
            xs = util.flatten_dim01(xs)  # (batch, T, xdim) --> (batch*T, xdim)
            label = - label / label_scale

            x0 = x0.unsqueeze(1).repeat(1,batch_t,1,1,1)
            x0 = util.flatten_dim01(x0)
            obs_mask = obs_mask.unsqueeze(1).repeat(1,batch_t,1,1,1)
            obs_mask = util.flatten_dim01(obs_mask)
            assert xs.shape[0] == ts.shape[0]

            if opt.backward_net in ['Transformerv2', 'Transformerv4', 'Transformerv5']:
                cond_mask = self.get_randmask(obs_mask, miss_ratio=opt.rand_mask_miss_ratio,
                    rank=opt.rand_mask_rank)  # Apply random mask here.
                target_mask = obs_mask - cond_mask
                cond_obs = cond_mask * x0
                noisy_target = (1 - cond_mask) * xs
                total_input = torch.cat([cond_obs, noisy_target], dim=1)
                diff_input = (total_input, cond_mask)
                loss_mask = target_mask  # don't use target_mask for fixed cond_mask.

            elif opt.backward_net == 'Transformerv3':
                cond_mask = self.get_randmask(obs_mask, miss_ratio=opt.rand_mask_miss_ratio,
                    rank=opt.rand_mask_rank)  # Apply random mask here.
                target_mask = obs_mask - cond_mask
                cond_obs = cond_mask * x0
                noisy_target = (1 - cond_mask) * xs
                total_input = cond_obs + noisy_target
                diff_input = (total_input, cond_mask)
                loss_mask = target_mask  # don't use target_mask for fixed cond_mask.

            else:
                diff_input = xs
                loss_mask = obs_mask

            predicted = policy(diff_input, ts)
            label = label.reshape_as(predicted)

            if opt.normalize_loss:
                label_scale = label_scale.reshape(batch_x*batch_t,1,1,1)
                residual = (label - predicted) * loss_mask / label_scale
            else:
                residual = (label - predicted) * loss_mask
            num_eval = loss_mask.sum()
            loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
            loss.backward()
            optimizer.step()
            ema.update()
            # if sched is not None: sched.step()

            avg_loss += loss.item()
            self.log_dsm_train(opt, it, loss.item(), avg_loss/it, optimizer, num_itr)

        keys = ['optimizer_b','ema_b','z_b']
        util.save_checkpoint(opt, self, keys, stage, suffix='dsm')


    def sb_dsm_train(self, opt, evaluate=True):
        """Only the DSM pre-training. Part of sb_alternate_train."""
        assert self.z_f.zero_out_last_layer
        for g in self.optimizer_b.param_groups:
            g['lr'] = opt.lr_dsm

        for stage in range(opt.num_stage):
            if opt.dsm_train_method in ['dsm', 'dsm_v2']:
                self.dsm_train(opt, num_itr=opt.num_itr_dsm, batch_x=opt.train_bs_x_dsm,
                    batch_t=opt.train_bs_t_dsm, stage=stage)

            elif opt.dsm_train_method in ['dsm_imputation', 'dsm_imputation_v2']:
                self.dsm_imputation_train(opt, num_itr=opt.num_itr_dsm, batch_x=opt.train_bs_x_dsm,
                    batch_t=opt.train_bs_t_dsm, stage=stage)
            elif opt.dsm_train_method in ['dsm_imputation_forward_verfication']:
                self.dsm_imputation_train_forward_verfication(opt, num_itr=opt.num_itr_dsm,
                    batch_x=opt.train_bs_x_dsm, batch_t=opt.train_bs_t_dsm, stage=stage)
            else:
                raise NotImplementedError('New dsm train method')

            if evaluate:
                # No unconditional validation for imputation training.
                self.evaluate(opt, stage, n_reused_trajs=0)
        if opt.log_tb: self.writer.close()


    def dsm_train_first_stage(self, opt):
        assert opt.DSM_warmup and self.z_f.zero_out_last_layer
        # self.dsm_train(opt, num_itr=opt.num_itr_dsm, batch_x=opt.train_bs_x_dsm,
        #     batch_t=opt.train_bs_t_dsm)
        for g in self.optimizer_b.param_groups:
            g['lr'] = opt.lr_dsm

        if opt.dsm_train_method in ['dsm', 'dsm_v2']:
            self.dsm_train(opt, num_itr=opt.num_itr_dsm, batch_x=opt.train_bs_x_dsm,
                batch_t=opt.train_bs_t_dsm, stage=0)

        elif opt.dsm_train_method in ['dsm_imputation', 'dsm_imputation_v2']:
            self.dsm_imputation_train(opt, num_itr=opt.num_itr_dsm, batch_x=opt.train_bs_x_dsm,
                batch_t=opt.train_bs_t_dsm, stage=0)


    # Not actively used. Gradient graph over input too large.
    def sb_joint_train(self, opt):
        assert not util.is_image_dataset(opt)

        policy_f, policy_b = self.z_f, self.z_b
        policy_f = activate_policy(policy_f)
        policy_b = activate_policy(policy_b)
        optimizer_f, _, sched_f = self.get_optimizer_ema_sched(policy_f)
        optimizer_b, _, sched_b = self.get_optimizer_ema_sched(policy_b)

        ts      = self.ts
        batch_x = opt.samp_bs

        for it in range(opt.num_itr):

            optimizer_f.zero_grad()
            optimizer_b.zero_grad()

            xs_f, zs_f, x_term_f = self.dyn.sample_traj(ts, policy_f, save_traj=True)
            xs_f = util.flatten_dim01(xs_f)
            zs_f = util.flatten_dim01(zs_f)
            _ts = ts.repeat(batch_x)

            loss = compute_sb_nll_joint_train(
                opt, batch_x, self.dyn, _ts, xs_f, zs_f, x_term_f, policy_b
            )
            loss.backward()

            optimizer_f.step()
            optimizer_b.step()

            if sched_f is not None: sched_f.step()
            if sched_b is not None: sched_b.step()

            self.log_sb_joint_train(opt, it, loss, optimizer_f, opt.num_itr)

            # evaluate
            if (it+1)%opt.eval_itr==0:
                with torch.no_grad():
                    xs_b, _, _ = self.dyn.sample_traj(ts, policy_b, save_traj=True)
                util.save_toy_npy_traj(opt, 'train_it{}'.format(it+1), xs_b.detach().cpu().numpy())

####################################################################################################
#                                       Inference functions
####################################################################################################

    @torch.no_grad()
    def _generate_samples_and_reused_trajs(
            self,
            opt,
            batch=None,
            num_samples=None,
            n_trajs=0):
        """Only applies backward sampling here."""
        assert n_trajs <= num_samples
        batch = num_samples if batch is None else batch

        ts = self.ts
        xTs = torch.empty((num_samples, *opt.data_dim), device='cpu')
        if n_trajs > 0:
            trajs = torch.empty((n_trajs, 2, len(ts), *opt.data_dim), device='cpu')
        else:
            trajs = None

        with self.ema_f.average_parameters(), self.ema_b.average_parameters():
            self.z_f = freeze_policy(self.z_f)
            self.z_b = freeze_policy(self.z_b)
            corrector = (lambda x,t: self.z_f(x,t) + self.z_b(x,t)) if opt.use_corrector else None

            it = 0
            while it < num_samples:
                # sample backward trajs; save traj if needed
                save_traj = (trajs is not None) and it < n_trajs
                _xs, _zs, _x_T = self.dyn.sample_traj(
                    ts, self.z_b, corrector=corrector, num_samples=batch, save_traj=save_traj)

                # fill xTs (for FID usage) and trajs (for training log_q)
                xTs[it:it+batch,...] = _x_T.detach().cpu()[0:min(batch,xTs.shape[0]-it),...]
                if save_traj:
                    trajs[it:it+batch,0,...] = _xs.detach().cpu()[0:min(batch,trajs.shape[0]-it),...]
                    trajs[it:it+batch,1,...] = _zs.detach().cpu()[0:min(batch,trajs.shape[0]-it),...]
                it += batch

        return xTs, trajs


    @torch.no_grad()
    def _generate_samples_and_reused_trajs_conditional(
            self,
            opt,
            x_cond,
            obs_mask,
            cond_mask,
            target_mask,
            num_samples=None):
        """Conditional inference."""
        batch = None
        batch = num_samples if batch is None else batch

        ts = self.ts
        # xTs = torch.empty((num_samples, *opt.data_dim), device='cpu')

        with self.ema_f.average_parameters(), self.ema_b.average_parameters():
            self.z_f = freeze_policy(self.z_f)
            self.z_b = freeze_policy(self.z_b)
            if opt.use_corrector and opt.backward_net in ['Transformerv2', 'Transformerv4',
                'Transformerv5']:
                def corrector(x, t, x_cond, cond_mask):
                    cond_obs = cond_mask * x_cond
                    noisy_target = (1-cond_mask) * x  # no big difference if using  target_mask * x
                    total_input = torch.cat([cond_obs, noisy_target], dim=1)
                    diff_input = (total_input, cond_mask)
                    return self.z_f(x,t) + self.z_b(diff_input, t)
            elif opt.use_corrector and opt.backward_net in ['Transformerv3']:
                def corrector(x, t, x_cond, cond_mask):
                    cond_obs = cond_mask * x_cond
                    noisy_target = (1-cond_mask) * x  # no big difference if using  target_mask * x
                    total_input = cond_obs + noisy_target
                    diff_input = (total_input, cond_mask)
                    return self.z_f(x,t) + self.z_b(diff_input, t)
            elif opt.use_corrector:
                def corrector(x, t, x_cond=None, cond_mask=None):
                    return self.z_f(x,t) + self.z_b(x,t)
            else:
                corrector = None

            _xs, _zs, _x_T = self.dyn.sample_traj_conditional(
                ts, x_cond, obs_mask, cond_mask, target_mask,
                self.z_b, corrector=corrector, num_samples=num_samples, save_traj=False)

        return _x_T, None


    @torch.no_grad()
    def imputation_vpsde(
            self,
            opt,
            x_cond,
            obs_mask,
            cond_mask,
            target_mask,
            num_samples=None):
        """Conditional inference."""
        # assert opt.backward_net in ['Transformerv2', 'Transformerv3']
        ts_reverse = torch.flip(self.ts, dims=[0])  # Backward diffusion.
        K, L = opt.input_size
        B = x_cond.shape[0]

        policy = freeze_policy(self.z_b)
        imputed_samples = torch.zeros(B, num_samples, K, L).to(opt.device)

        for i in tqdm(range(num_samples), ncols=100, file=sys.stdout):
            current_sample = torch.randn_like(x_cond)

            for idx, t in enumerate(ts_reverse):
                t_idx = len(ts_reverse)-idx-1  # backward ids.
                if opt.backward_net in ['Transformerv2', 'Transformerv4', 'Transformerv5']:
                    # Only need conditional mask, no need obs_mask or target_mask.
                    cond_obs = cond_mask * x_cond
                    noisy_target = (1 - cond_mask) * current_sample  # not target_mask here.
                    total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
                    diff_input = (total_input, cond_mask)
                elif opt.backward_net == 'Transformerv3':
                    # Only need conditional mask, no need obs_mask or target_mask.
                    cond_obs = cond_mask * x_cond
                    noisy_target = (1 - cond_mask) * current_sample  # not target_mask here.
                    total_input = cond_obs + noisy_target  # (B,1,K,L)
                    diff_input = (total_input, cond_mask)
                else:
                    diff_input = cond_mask * x_cond + (1 - cond_mask) * current_sample

                z = policy(diff_input, t)
                dt = self.dyn.dt
                f = - self.dyn.f_back(current_sample, t)  # neg sign, as dt is pos not neg here.
                g = self.dyn.g(t)
                g_score_back = self.dyn.g_score_back(t)
                g_score = g_score_back / g
                current_sample = current_sample + (f + g_score * z)*dt
                if t_idx > 0:
                    dw = self.dyn.dw(current_sample)
                    g_back = self.dyn.g_back(t)
                    current_sample += g_back * dw

            imputed_samples[:, i] = current_sample.squeeze(1).detach()
        return imputed_samples


    @torch.no_grad()
    def imputation(
            self,
            opt,
            x_cond,
            obs_mask,
            cond_mask,
            target_mask,
            num_samples=None):
        """Conditional inference."""
        # assert opt.backward_net in ['Transformerv2', 'Transformerv3']
        ts_reverse = torch.flip(self.ts, dims=[0])  # Backward diffusion.
        K, L = opt.input_size
        B = x_cond.shape[0]

        if opt.use_corrector and opt.backward_net in ['Transformerv2', 'Transformerv4',
            'Transformerv5']:
            def corrector(x, t, x_cond, cond_mask):
                cond_obs = cond_mask * x_cond
                noisy_target = (1-cond_mask) * x  # no big difference if using  target_mask * x
                total_input = torch.cat([cond_obs, noisy_target], dim=1)
                diff_input = (total_input, cond_mask)
                return self.z_f(x,t) + self.z_b(diff_input, t)
        elif opt.use_corrector and opt.backward_net in ['Transformerv3']:
            def corrector(x, t, x_cond, cond_mask):
                cond_obs = cond_mask * x_cond
                noisy_target = (1-cond_mask) * x  # no big difference if using  target_mask * x
                total_input = cond_obs + noisy_target
                diff_input = (total_input, cond_mask)
                return self.z_f(x,t) + self.z_b(diff_input, t)
        elif opt.use_corrector:
            def corrector(x, t, x_cond=None, cond_mask=None):
                return self.z_f(x,t) + self.z_b(x,t)
        else:
            corrector = None

        policy = freeze_policy(self.z_b)
        imputed_samples = torch.zeros(B, num_samples, K, L).to(opt.device)

        for i in tqdm(range(num_samples), ncols=100, file=sys.stdout):
            current_sample = torch.randn_like(x_cond)

            for idx, t in enumerate(ts_reverse):
                t_idx = len(ts_reverse)-idx-1  # backward ids.
                if opt.backward_net in ['Transformerv2', 'Transformerv4', 'Transformerv5']:
                    # Only need conditional mask, no need obs_mask or target_mask.
                    cond_obs = cond_mask * x_cond
                    noisy_target = (1 - cond_mask) * current_sample  # not target_mask here.
                    total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
                    diff_input = (total_input, cond_mask)
                elif opt.backward_net == 'Transformerv3':
                    # Only need conditional mask, no need obs_mask or target_mask.
                    cond_obs = cond_mask * x_cond
                    noisy_target = (1 - cond_mask) * current_sample  # not target_mask here.
                    total_input = cond_obs + noisy_target  # (B,1,K,L)
                    diff_input = (total_input, cond_mask)
                else:
                    diff_input = cond_mask * x_cond + (1 - cond_mask) * current_sample

                z = policy(diff_input, t)
                g = self.dyn.g(t)
                f = self.dyn.f(current_sample,t,'backward')
                dt = self.dyn.dt
                dw = self.dyn.dw(current_sample,dt)
                current_sample = current_sample + (f + g * z)*dt
                if t > 0:
                    current_sample += g*dw

                # Apply corrector.
                if opt.use_corrector:
                    _t=t if idx==ts_reverse.shape[0]-1 else ts_reverse[idx+1]
                    current_sample = self.dyn.corrector_langevin_imputation_update(
                        _t, current_sample, corrector, x_cond, cond_mask, denoise_xT=False)

            imputed_samples[:, i] = current_sample.squeeze(1).detach()
        return imputed_samples


    @torch.no_grad()
    def imputation_backward_test(
            self,
            opt,
            x_cond,
            obs_mask,
            cond_mask,
            target_mask,
            num_samples=None):
        """The score function is the same as CSDI."""
        self.beta = np.linspace(0.0002, 0.4, opt.interval)
        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        # self.alpha_torch = torch.tensor(np.expand_dims(self.alpha, axis=(1,2,3))).float()
        self.alpha_torch = torch.tensor(self.alpha).float().unsqueeze(1).unsqueeze(1).to(opt.device)

        assert opt.backward_net in ['Transformerv1', 'Transformerv2', 'Transformerv3',
            'Transformerv4', 'Transformerv5']
        ts_reverse = torch.flip(self.ts, dims=[0])  # Backward diffusion.
        K, L = opt.input_size
        B = x_cond.shape[0]

        policy = freeze_policy(self.z_b)
        imputed_samples = torch.zeros(B, num_samples, K, L).to(opt.device)

        for i in tqdm(range(num_samples), ncols=100, file=sys.stdout):
            current_sample = torch.randn_like(x_cond)

            for idx, t in enumerate(ts_reverse):
                t_idx = len(ts_reverse)-idx-1  # backward ids.
                # score_scale = 1 / std_t * g_t
                # Only need conditional mask, no need obs_mask or target_mask.
                cond_obs = cond_mask * x_cond
                noisy_target = (1 - cond_mask) * current_sample  # not target_mask here.
                total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
                diff_input = (total_input, cond_mask)

                z = policy(diff_input, t)
                g = self.dyn.g(t)
                f = self.dyn.f(current_sample,t,'backward')
                dt = self.dyn.dt
                dw = self.dyn.dw(current_sample,dt)
                current_sample = current_sample + (f + g * z)*dt
                if t > 0:
                    current_sample += g*dw

            imputed_samples[:, i] = current_sample.squeeze(1).detach()
        return imputed_samples

    @torch.no_grad()
    def unconditional_sampling(
            self,
            opt,
            num_samples=None):
        """Conditional inference."""
        assert opt.backward_net in ['Transformerv2', 'Transformerv3', 'Transformerv4', 'Transformerv5']
        ts = torch.flip(self.ts, dims=[0])
        K, L = opt.input_size
        B = 1

        policy = freeze_policy(self.z_b)

        samples = torch.zeros(B, num_samples, K, L).to(opt.device)

        for i in tqdm(range(num_samples), ncols=100, file=sys.stdout):
            current_sample = torch.randn(1,1,K,L).to(opt.device)
            # for idx, t in tqdm(enumerate(ts), ncols=100, total=len(self.interval), file=sys.stdout,
            #         desc=util.yellow("SDE sampling...")):
            for idx, t in enumerate(ts):
                total_input = torch.cat([torch.zeros_like(current_sample), current_sample], dim=1)  # (B,2,K,L)
                diff_input = (total_input, torch.zeros_like(current_sample))
                score = policy(diff_input, t)

                g = self.dyn.g(t)
                f = self.dyn.f(current_sample,t,'backward')
                dt = self.dyn.dt
                dw = self.dyn.dw(current_sample,dt)
                current_sample += (f + g * score)*dt
                # current_sample += (f + g**2 * score)*dt

                if t > 0:
                    current_sample += g*dw

            samples[:, i] = current_sample.squeeze(1).detach()
        return samples


    @torch.no_grad()
    def compute_NLL(self, opt):
        num_NLL_sample = self.p.num_sample
        assert util.is_image_dataset(opt) and num_NLL_sample%opt.samp_bs==0
        bpds=[]
        with self.ema_f.average_parameters(), self.ema_b.average_parameters():
            for _ in range(int(num_NLL_sample/opt.samp_bs)):
                bits_per_dim = self.dyn.compute_nll(opt.samp_bs, self.ts, self.z_f, self.z_b)
                bpds.append(bits_per_dim.detach().cpu().numpy())

        print(util.yellow("=================== NLL={} ======================").format(np.array(bpds).mean()))

    @torch.no_grad()
    def evaluate_img_dataset(self, opt, stage, n_reused_trajs=0, metrics=None):
        assert util.is_image_dataset(opt)

        fid, snapshot, ckpt = util.evaluate_stage(opt, stage, metrics)

        if ckpt:
            keys = ['z_f','optimizer_f','ema_f','z_b','optimizer_b','ema_b']
            util.save_checkpoint(opt, self, keys, stage)

        # return if no evaluation effort needed in this stage
        if not (fid or snapshot): return

        # either fid or snapshot requires generating sample (meanwhile
        # we can collect trajectories and reuse them in later training)
        batch = opt.samp_bs
        n_reused_trajs = min(n_reused_trajs, opt.num_FID_sample)
        n_reused_trajs -= (n_reused_trajs % batch) # make sure n_reused_trajs is divisible by samp_bs
        xTs, trajs = self._generate_samples_and_reused_trajs(
            opt, batch, opt.num_FID_sample, n_reused_trajs,)

        if fid and util.exist_FID_ckpt(opt):
            FID = util.compute_fid(opt, xTs)
            print(util.yellow("===================FID={}===============================").format(FID))
            if opt.log_tb: self.log_tb(stage, FID, 'FID', 'eval')
        else:
            print(util.red("Does not exist FID ckpt, please compute FID manually."))

        # if snapshot:
        util.snapshot(opt, xTs, stage, 'backward', num_plots=batch)

        gc.collect()

        if trajs is not None:
            trajs = trajs.reshape(-1, batch, *trajs.shape[1:])
            return util.create_traj_sampler(trajs)

    @torch.no_grad()
    def evaluate(self, opt, stage, n_reused_trajs=0, metrics=None):
        if opt.problem_name in ['mnist','cifar10','celebA32','celebA64']:
            return self.evaluate_img_dataset(
                opt, stage, n_reused_trajs=n_reused_trajs, metrics=metrics)

        elif opt.problem_name in ['gmm','checkerboard', 'moon-to-spiral']:
            _, snapshot, ckpt = util.evaluate_stage(opt, stage, metrics=None)
            if snapshot:
                for z in [self.z_f, self.z_b]:
                    z = freeze_policy(z)
                    xs, _, _ = self.dyn.sample_traj(self.ts, z, save_traj=True)

                    fn = "stage{}-{}".format(stage, z.direction)
                    util.save_toy_npy_traj(
                        opt, fn, xs.detach().cpu().numpy(), n_snapshot=15, direction=z.direction
                    )
            if ckpt:
                keys = ['z_f','optimizer_f','ema_f','z_b','optimizer_b','ema_b']
                util.save_checkpoint(opt, self, keys, stage)

        elif opt.problem_name in ['sinusoid', 'pm25', 'physio',
            'exchange_rate_nips', 'solar_nips', 'electricity_nips',]:
            for z in [self.z_f, self.z_b]:
                z = freeze_policy(z)
                try:
                    _, _, x_term = self.dyn.sample_traj(self.ts, z, num_samples=opt.num_eval_sample,
                        save_traj=False)
                    x_term = x_term.permute(1,0,2,3)  # (1,samples,K,L)
                    fn = "stage{}-{}".format(stage, z.direction)
                    util.save_time_series_xT(opt, x_term.detach().cpu(),
                        direction=z.direction, fn=fn, show_figure=False)
                except:
                    print('Warning: sampling path failed.')
                    pass

            # Then do conditional test through imputation task.
            if opt.imputation_eval:
                if opt.full_eval_every_stage or (stage >= opt.num_stage - 2):
                    quick_eval = False  # Full eval at the end.
                else:
                    quick_eval = True
                run_validation = True
                if opt.problem_name in ['exchange_rate_nips', 'solar_nips', 'electricity_nips',]:
                    print('WARNING: temporally disable quick val for NIPS replication.')
                    run_validation = False

                self.imputation_eval(opt, stage=stage, quick_eval=quick_eval,
                                     run_validation=run_validation, output_dir=opt.ckpt_path)
        else:
            raise NotImplementedError('Evaluation: New problem. No evaluation.')


    @torch.no_grad()
    def imputation_eval(self, opt, stage=None, quick_eval=True, run_validation=True, output_dir=None):
        # Build test dataloader here.
        if opt.problem_name == 'pm25':
            assert opt.permute_batch == True
            import dataset_pm25
            train_loader, val_loader, test_loader, scaler, mean_scaler = dataset_pm25.get_dataloader(
                batch_size=42, eval_length=36, target_dim=36, validindex=0, device='cpu',
                return_dataset=False)

        elif opt.problem_name == 'physio':
            assert opt.permute_batch == True
            import dataset_physio
            train_loader, val_loader, test_loader = dataset_physio.get_dataloader(
                seed=1, nfold=opt.physio_nfold, missing_ratio=opt.dataset_missing_ratio, batch_size=64,
                eval_length=48, target_dim=35, device='cpu', return_dataset=False)
            scaler, mean_scaler = 1.0, 0.0

        elif opt.problem_name == 'sinusoid':
            assert opt.permute_batch == True
            import dataset_sinusoid
            train_loader, val_loader, test_loader = dataset_sinusoid.get_dataloader(
                opt.sinusoid_dataset_path, eval_length=50, batch_size=64, seed=1)
            scaler, mean_scaler = 1.0, 0.0

        elif opt.problem_name in ['exchange_rate_nips', 'solar_nips', 'electricity_nips',]:
            assert opt.permute_batch == True
            from dataset_nips import get_dataloader
            train_loader, val_loader, test_loader, scaler, mean_scaler = get_dataloader(
                opt.problem_name, batch_size=1, device='cpu', target_dim_range=opt.target_dim_range)
        else:
            raise NotImplementedError(f'New dataset {opt.problem_name}')

        # check the test set performance.
        if quick_eval:
            print(util.yellow('Quick check on test set.'))
        else:
            print(util.yellow('Complete check on test set.'))

        corrector_flag = '_corrector' if opt.use_corrector else ''
        num_samples = 5 if quick_eval else opt.num_eval_sample
        num_eval_batches = 3 if quick_eval else None  # if None then just iterates all test_dataset.
        output_dir = opt.ckpt_path if output_dir is None else output_dir
        samples_path = (output_dir + f'/samples{num_samples}_stage{stage}_' +
                       dt.datetime.now().strftime("%m_%d_%Y_%H%M%S") + f'test{corrector_flag}.pk')
        metrics_path = (output_dir + f'/metrics_stage{stage}_' +
                       dt.datetime.now().strftime("%m_%d_%Y_%H%M%S") + f'test{corrector_flag}.json')
        util.conditional_imputation_eval(opt.eval_impute_function, opt, test_loader, self,
            mean_scaler=mean_scaler, scaler=scaler, num_samples=num_samples,
            num_eval_batches=num_eval_batches, samples_path=samples_path, metrics_path=metrics_path)

        if not run_validation:
            return
        # check the validation set performance.
        print(util.yellow('Quick check on validation set.'))
        num_samples = 5  # quick check on val dataset.
        num_eval_batches = 3
        output_dir = opt.ckpt_path if output_dir is None else output_dir
        samples_path = (output_dir + f'/samples{num_samples}_stage{stage}_' +
                       dt.datetime.now().strftime("%m_%d_%Y_%H%M%S") + f'val{corrector_flag}.pk')
        metrics_path = (output_dir + f'/metrics_stage{stage}_' +
                       dt.datetime.now().strftime("%m_%d_%Y_%H%M%S") + f'val{corrector_flag}.json')
        util.conditional_imputation_eval(opt.eval_impute_function, opt, val_loader, self,
            mean_scaler=mean_scaler, scaler=scaler, num_samples=num_samples,
            num_eval_batches=num_eval_batches, samples_path=samples_path, metrics_path=metrics_path)



    def _print_train_itr(self, it, loss, avg_loss, optimizer, num_itr, name):
        time_elapsed = util.get_time(time.time()-self.start_time)
        lr = optimizer.param_groups[0]['lr']
        print("[{0}] train_it {1}/{2}\t| lr:{3} | loss:{4} | avg_loss:{5} | time:{6}"
            .format(
                util.cyan(name),
                util.yellow("{}".format(it)),
                num_itr,
                util.yellow("{:.3e}".format(lr)),
                util.yellow("{:.4f}".format(loss)),
                util.yellow("{:.4f}".format(avg_loss)),
                util.yellow("{0}:{1:02d}:{2:05.2f}".format(*time_elapsed)),
        ))

    def log_dsm_train(self, opt, it, loss, avg_loss, optimizer, num_itr):
        if it % 50 != 1:
            return
        self._print_train_itr(it, loss, avg_loss, optimizer, num_itr, name='DSM backward')
        if opt.log_tb:
            step = self.update_count('backward')
            self.log_tb(step, loss, 'loss', 'DSM_backward')

    def log_sb_joint_train(self, opt, it, loss, optimizer, num_itr):
        self._print_train_itr(it, loss, optimizer, num_itr, name='SB joint')
        if opt.log_tb:
            step = self.update_count('backward')
            self.log_tb(step, loss, 'loss', 'SB_joint')

    def log_sb_alternate_train(self, opt, it, ep, stage, loss, zs, zs_impt, optimizer, direction, num_epoch):
        if it % 50 != 0:
            return
        time_elapsed = util.get_time(time.time()-self.start_time)
        lr = optimizer.param_groups[0]['lr']
        # loss: # SB surrogate loss (see Eq 18 & 19 in the paper)
        # NELBO:  # negative ELBO (see Eq 16 in the paper)
        neg_elbo = loss + util.compute_z_norm(zs_impt, self.dyn.dt).detach()

        print("[{0}] stage {1}/{2} | ep {3}/{4} | iter {5}/{6} | lr:{7} | loss:{8} | nelbo:{9} | {10}"
            .format(
                util.cyan("SB {}".format(direction)),
                util.yellow("{}".format(1+stage)),
                opt.num_stage,
                util.yellow("{}".format(1+ep)),
                num_epoch,
                util.yellow("{}".format(1+it+opt.num_itr*ep)),
                opt.num_itr*num_epoch,
                util.yellow("{:.2e}".format(lr)),
                util.yellow("{:.4f}".format(loss)),
                util.yellow("{:.4f}".format(neg_elbo)),
                util.yellow("{0}:{1:02d}:{2:05.2f}".format(*time_elapsed)),
        ))
        if opt.log_tb:
            step = self.update_count(direction)
            # neg_elbo = loss + util.compute_z_norm(zs_impt, self.dyn.dt).detach()
            self.log_tb(step, loss, 'loss', 'SB_'+direction) # SB surrogate loss (see Eq 18 & 19 in the paper)
            self.log_tb(step, neg_elbo, 'neg_elbo', 'SB_'+direction) # negative ELBO (see Eq 16 in the paper)
            # if direction == 'forward':
            #     z_norm = util.compute_z_norm(zs, self.dyn.dt)
            #     self.log_tb(step, z_norm.detach(), 'z_norm', 'SB_forward')

    def log_tb(self, step, val, name, tag):
        self.writer.add_scalar(os.path.join(tag,name), val, global_step=step)

