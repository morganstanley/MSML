
import os, time, gc

import numpy as np

import torch
import torch.nn.functional as F
from torch.optim import SGD, RMSprop, Adagrad, AdamW, lr_scheduler, Adam
from torch.utils.tensorboard import SummaryWriter
from torch_ema import ExponentialMovingAverage

import policy
import sde
from loss import compute_sb_nll_alternate_train, compute_sb_nll_joint_train
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
            'weight_decay':opt.l2_norm,
    }
    if opt.optimizer == 'SGD':
        optim_dict['momentum'] = 0.9

    optimizer = optim_name(policy.parameters(), **optim_dict)
    ema = ExponentialMovingAverage(policy.parameters(), decay=0.99)
    if opt.lr_gamma < 1.0:
        sched = lr_scheduler.StepLR(optimizer, step_size=opt.lr_step, gamma=opt.lr_gamma)
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
    policy.train()
    return policy

class Runner():
    def __init__(self,opt):
        super(Runner,self).__init__()

        self.start_time=time.time()
        self.ts = torch.linspace(opt.t0, opt.T, opt.interval)

        # build boundary distribution (p: target, q: prior)
        self.p, self.q = data.build_boundary_distribution(opt)

        # build dynamics, forward (z_f) and backward (z_b) policies
        self.dyn = sde.build(opt, self.p, self.q)
        self.z_f = policy.build(opt, self.dyn, 'forward')  # p -> q
        self.z_b = policy.build(opt, self.dyn, 'backward') # q -> p

        self.optimizer_f, self.ema_f, self.sched_f = build_optimizer_ema_sched(opt, self.z_f)
        self.optimizer_b, self.ema_b, self.sched_b = build_optimizer_ema_sched(opt, self.z_b)

        if opt.load:
            util.restore_checkpoint(opt, self, opt.load)

        if opt.log_tb: # tensorboard related things
            self.it_f = 0
            self.it_b = 0
            self.writer=SummaryWriter(
                log_dir=os.path.join('runs', opt.log_fn) if opt.log_fn is not None else None
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
        if reused_sampler is not None:
            reused_traj = next(reused_sampler)
            train_xs, train_zs = reused_traj[:,0,...], reused_traj[:,1,...]
            print('generate train data from [{}]!'.format(util.green('reused samper')))
        else:
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
        assert train_xs.shape == train_zs.shape
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
            samp_x_idx = torch.randint(opt.samp_bs,  (opt.train_bs_x,)).cpu()
            samp_t_idx = torch.randint(opt.interval, (opt.train_bs_t,)).cpu()
            if opt.use_arange_t: samp_t_idx = torch.arange(opt.interval).cpu()

            # -------- build sample --------
            
            ts = train_ts[samp_t_idx].detach()
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
                torch.nn.utils.clip_grad_norm(policy.parameters(), opt.grad_clip)
            optimizer.step()
            ema.update()
            if sched is not None: sched.step()

            # -------- logging --------
            zs = util.unflatten_dim01(zs, [len(samp_x_idx), len(samp_t_idx)])
            zs_impt = zs_impt.reshape(zs.shape)
            self.log_sb_alternate_train(
                opt, it, ep, stage, loss, zs, zs_impt, optimizer, direction, num_epoch
            )

    def dsm_train(self, opt, num_itr, batch_x, batch_t):
        """ Our own implementation of dsm, support both VE and VP """

        policy = activate_policy(self.z_b)
        optimizer, ema, sched = self.optimizer_b, self.ema_b, self.sched_b

        compute_xs_label = sde.get_xs_label_computer(opt, self.ts)

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
                torch.nn.utils.clip_grad_norm(policy.parameters(), opt.grad_clip)

            optimizer.step()
            ema.update()
            if sched is not None: sched.step()

            self.log_dsm_train(opt, it, loss, optimizer, num_itr)

        keys = ['optimizer_b','ema_b','z_b']
        util.save_checkpoint(opt, self, keys, 1, dsm_train_it=num_itr)

    def dsm_train_first_stage(self, opt):
        assert opt.DSM_warmup and self.z_f.zero_out_last_layer
        self.dsm_train(
            opt, num_itr=opt.num_itr_dsm, batch_x=opt.train_bs_x_dsm, batch_t=opt.train_bs_t_dsm,
        )

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
                else:
                    self.sb_alternate_train_stage(opt, stage, backward_ep, 'backward')
            # evaluate backward policy;
            # reuse evaluated trajectories for training forward policy
            n_reused_trajs = forward_ep * opt.samp_bs if opt.reuse_traj else 0
            reused_sampler = self.evaluate(opt, stage+1, n_reused_trajs=n_reused_trajs)

            # train forward policy
            self.sb_alternate_train_stage(
                opt, stage, forward_ep, 'forward', reused_sampler=reused_sampler
            )

        if opt.log_tb: self.writer.close()

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

    @torch.no_grad()
    def _generate_samples_and_reused_trajs(self, opt, batch, n_samples, n_trajs):
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
            print(f'sample {n_samples} images...')
            while it < n_samples:
                # sample backward trajs; save traj if needed
                save_traj = (trajs is not None) and it < n_trajs
                _xs, _zs, _x_T = self.dyn.sample_traj(
                    ts, self.z_b, corrector=corrector, save_traj=save_traj)

                # fill xTs (for FID usage) and trajs (for training log_q)
                xTs[it:it+batch,...] = _x_T.detach().cpu()[0:min(batch,xTs.shape[0]-it),...]
                if save_traj:
                    trajs[it:it+batch,0,...] = _xs.detach().cpu()[0:min(batch,trajs.shape[0]-it),...]
                    trajs[it:it+batch,1,...] = _zs.detach().cpu()[0:min(batch,trajs.shape[0]-it),...]

                it += batch

        return xTs, trajs

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
        
        # self.compute_NLL(opt)
        # return if no evaluation effort needed in this stage
        if not (fid or snapshot): return

        # either fid or snapshot requires generating sample (meanwhile
        # we can collect trajectories and reuse them in later training)
        batch = opt.samp_bs
        n_reused_trajs = min(n_reused_trajs, opt.num_FID_sample)
        n_reused_trajs -= (n_reused_trajs % batch) # make sure n_reused_trajs is divisible by samp_bs
        xTs, trajs = self._generate_samples_and_reused_trajs(
            opt, batch, opt.num_FID_sample, n_reused_trajs,
        )

        if fid and util.exist_FID_ckpt(opt):
            FID = util.compute_fid(opt, xTs)
            print(util.yellow("===================FID={}===============================").format(FID))
            if opt.log_tb: self.log_tb(stage, FID, 'FID', 'eval')
        else:
            print(util.red("Does not exist FID ckpt, please compute FID manually."))

        if snapshot:
            util.snapshot(opt, xTs, stage, 'backward')

        gc.collect()

        if trajs is not None:
            trajs = trajs.reshape(-1, batch, *trajs.shape[1:])
            return util.create_traj_sampler(trajs)

    @torch.no_grad()
    def gen_img(self, opt, stage, n_reused_trajs=0):
        assert util.is_image_dataset(opt)

        # either fid or snapshot requires generating sample (meanwhile
        # we can collect trajectories and reuse them in later training)
        batch = opt.samp_bs
        n_reused_trajs = min(n_reused_trajs, opt.num_FID_sample)
        n_reused_trajs -= (n_reused_trajs % batch) # make sure n_reused_trajs is divisible by samp_bs
        xTs, trajs = self._generate_samples_and_reused_trajs(
            opt, batch, 500, n_reused_trajs,
        )

        util.snapshot(opt, xTs, stage, 'backward')

        gc.collect()


    @torch.no_grad()
    def evaluate(self, opt, stage, n_reused_trajs=0, metrics=None):
        if util.is_image_dataset(opt):
            return self.evaluate_img_dataset(
                opt, stage, n_reused_trajs=n_reused_trajs, metrics=metrics
            )
        elif util.is_toy_dataset(opt):
            _, snapshot, ckpt = util.evaluate_stage(opt, stage, metrics=None)
            if snapshot:
                for z in [self.z_f, self.z_b]:
                    z = freeze_policy(z)
                    xs, _, _ = self.dyn.sample_traj(self.ts, z, save_traj=True)

                    fn = "stage{}-{}".format(stage, z.direction)
                    util.save_toy_npy_traj(
                        opt, fn, xs.detach().cpu().numpy(), n_snapshot=5, direction=z.direction
                    )
            if ckpt:
                keys = ['z_f','optimizer_f','ema_f','z_b','optimizer_b','ema_b']
                util.save_checkpoint(opt, self, keys, stage)

    def _print_train_itr(self, it, loss, optimizer, num_itr, name):
        time_elapsed = util.get_time(time.time()-self.start_time)
        lr = optimizer.param_groups[0]['lr']
        print("[{0}] train_it {1}/{2} | lr:{3} | loss:{4} | time:{5}"
            .format(
                util.magenta(name),
                util.cyan("{}".format(1+it)),
                num_itr,
                util.yellow("{:.2e}".format(lr)),
                util.red("{:.4f}".format(loss.item())),
                util.green("{0}:{1:02d}:{2:05.2f}".format(*time_elapsed)),
        ))

    def log_dsm_train(self, opt, it, loss, optimizer, num_itr):
        self._print_train_itr(it, loss, optimizer, num_itr, name='DSM backward')
        if opt.log_tb:
            step = self.update_count('backward')
            self.log_tb(step, loss.detach(), 'loss', 'DSM_backward')

    def log_sb_joint_train(self, opt, it, loss, optimizer, num_itr):
        self._print_train_itr(it, loss, optimizer, num_itr, name='SB joint')
        if opt.log_tb:
            step = self.update_count('backward')
            self.log_tb(step, loss.detach(), 'loss', 'SB_joint')

    def log_sb_alternate_train(self, opt, it, ep, stage, loss, zs, zs_impt, optimizer, direction, num_epoch):
        time_elapsed = util.get_time(time.time()-self.start_time)
        lr = optimizer.param_groups[0]['lr']
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
            # if direction == 'forward':
            #     z_norm = util.compute_z_norm(zs, self.dyn.dt)
            #     self.log_tb(step, z_norm.detach(), 'z_norm', 'SB_forward')

    def log_tb(self, step, val, name, tag):
        self.writer.add_scalar(os.path.join(tag,name), val, global_step=step)

