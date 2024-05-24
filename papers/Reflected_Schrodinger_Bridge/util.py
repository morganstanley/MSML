import os,sys,re

import numpy as np
import pandas as pd
import shutil
import termcolor
import pathlib
from scipy import linalg
from PIL import Image
import matplotlib.pyplot as plt
import datetime as dt
from tqdm import tqdm
from ipdb import set_trace as debug

import torch
import torchvision
import torchvision.utils as tu
import torch.nn.functional as F
from torch.nn.functional import adaptive_avg_pool2d


# convert to colored strings
def red(content): return termcolor.colored(str(content),"red",attrs=["bold"])
def green(content): return termcolor.colored(str(content),"green",attrs=["bold"])
def blue(content): return termcolor.colored(str(content),"blue",attrs=["bold"])
def cyan(content): return termcolor.colored(str(content),"cyan",attrs=["bold"])
def yellow(content): return termcolor.colored(str(content),"yellow",attrs=["bold"])
def magenta(content): return termcolor.colored(str(content),"magenta",attrs=["bold"])

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_parameters_in_M(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6

def is_image_dataset(opt):
    return opt.problem_name in ['mnist','cifar10','celebA32','celebA64']

def is_toy_dataset(opt):
    return opt.problem_name in ['gmm','checkerboard', 'moon-to-spiral', 'gaussian-to-gaussian', 'moon', 'spiral']

def use_vp_sde(opt):
    return opt.sde_type == 'vp'

def evaluate_stage(opt, stage, metrics):
    """ Determine what metrics to evaluate for the current stage,
    if metrics is None, use the frequency in opt to decide it.
    """
    if metrics is not None:
        return [k in metrics for k in ['FID', 'snapshot', 'ckpt']]
    match = lambda freq: (freq>0 and stage%freq==0)
    return [match(opt.FID_freq), match(opt.snapshot_freq), match(opt.ckpt_freq)]

def get_time(sec):
    h = int(sec//3600)
    m = int((sec//60)%60)
    s = sec%60
    return h,m,s

def flatten_dim01(x):
    # (dim0, dim1, *dim2) --> (dim0x1, *dim2)
    return x.reshape(-1, *x.shape[2:])

def unflatten_dim01(x, dim01):
    # (dim0x1, *dim2) --> (dim0, dim1, *dim2)
    return x.reshape(*dim01, *x.shape[1:])

def compute_z_norm(zs, dt):
    # Given zs.shape = [batch, timesteps, *z_dim], return E[\int 0.5*norm(z)*dt],
    # where the norm is taken over z_dim, the integral is taken over timesteps,
    # and the expectation is taken over batch.
    zs = zs.reshape(*zs.shape[:2],-1)
    return 0.5 * zs.norm(dim=2).sum(dim=1).mean(dim=0) * dt

def create_traj_sampler(trajs):
    for traj in trajs:
        yield traj

def get_load_it(load_name):
    nums = re.findall('[0-9]+', load_name)
    assert len(nums)>0
    if 'stage' in load_name and 'dsm' in load_name:
        return int(nums[-2])
    return int(nums[-1])

def restore_checkpoint(opt, runner, load_name, subset_keys=None):
    assert load_name is not None
    print(green("#loading checkpoint {}...".format(load_name)))

    if 'checkpoint_16.pth' in load_name:
        # loading pre-trained NCSN++ from
        # https://drive.google.com/drive/folders/1sP4GwvrYiI-sDPTp7sKYzsxJLGVamVMZ
        assert opt.backward_net == 'ncsnpp'

        with torch.cuda.device(opt.gpu):
            checkpoint = torch.load(load_name)
            model_ckpt, ema_params_ckpt = checkpoint['model'], checkpoint['ema']['shadow_params']

            # load model
            res = {k.replace('module.', 'net.') : v for k, v in model_ckpt.items()}
            runner.z_b.load_state_dict(res) # Dont load key:sigmas.
            print(green('#successfully loaded all the modules'))

            # load ema
            assert type(runner.ema_b.shadow_params) == list
            runner.ema_b.shadow_params = [p.to(opt.device) for p in ema_params_ckpt]
            print(green('#loading form ema shadow parameter for polices'))

    elif 'checkpoint_19.pth' in load_name:
        # RVE NCSN++ bacward net ckpt.
        assert opt.backward_net == 'ncsnpp'
        assert opt.model_configs['ncsnpp'].model.convert_t_to_std
        with torch.cuda.device(opt.gpu):
            state_ckpt = torch.load(load_name)
            runner.z_b.net.load_state_dict(state_ckpt['model'])
            print(green('#successfully loaded model ckpt.'))
            # Copy current parameters to the shadow.
            # runner.ema_b.shadow_params = [p.to(opt.device) for p in runner.z_b.parameters()]
            # Previous ckpt didn't save Gaussian emb.
            ema_params_ckpt = state_ckpt['ema']['shadow_params']
            time_emb = [p.clone().detach().to(opt.device) for p in runner.z_b.parameters()][0]
            runner.ema_b.shadow_params = [time_emb] + [p.to(opt.device) for p in ema_params_ckpt]
            print(yellow('#copying model to ema.shadow parameter'))

    else:
        full_keys = ['z_f','optimizer_f','ema_f','z_b','optimizer_b','ema_b']

        with torch.cuda.device(opt.gpu):
            checkpoint = torch.load(load_name,map_location=opt.device)
            print('    state_dict keys:', checkpoint.keys())
            print('loading subset keys:', subset_keys)
            ckpt_keys=[*checkpoint.keys()]
            for k in ckpt_keys:
                if subset_keys and k not in subset_keys: continue
                print('loading state dict', k)
                getattr(runner,k).load_state_dict(checkpoint[k])

        if len(full_keys)!=len(ckpt_keys):
            value = { k for k in set(full_keys) - set(ckpt_keys) }
            print(green("#warning: does not load model for {}, check is it correct".format(value)))
        else:
            print(green('#successfully loaded all the modules'))

        # Stop copying ema, as we will reset it anyway.
        # Note: Copy the avergage parameter to policy. This seems to improve performance for
        # DSM warmup training (yet not sure whether it's true for SB in general)
        runner.ema_f.copy_to()
        runner.ema_b.copy_to()
        print(green('#loading form ema shadow parameter for polices'))
    print(magenta("#######summary of checkpoint##########"))


def save_opt(opt, opt_file):
    import yaml
    with open(opt_file, 'wt') as f:
        yaml.dump(vars(opt), f)
        print('save opt:' + opt_file)


def save_checkpoint(opt, runner, keys, stage_it, suffix=''):
    checkpoint = {}
    # fn = opt.ckpt_path + "/stage_{0}{1}.npz".format(
    #     stage_it, '_dsm{}'.format(dsm_train_it) if dsm_train_it is not None else '')
    if opt.overwrite_ckpt:
        fn = opt.ckpt_path + f"/stage_{suffix}.npz"
    else:
        fn = opt.ckpt_path + f"/stage_{stage_it}_{suffix}.npz"
    with torch.cuda.device(opt.gpu):
        for k in keys:
            checkpoint[k] = getattr(runner,k).state_dict()
        torch.save(checkpoint, fn)
    print("checkpoint saved: {}".format(fn))
    save_opt(opt, opt.ckpt_path + '/opt.yaml')


def compare_opts(opt_1, opt_2):
    """
    Args:
        opt_1: the base opt, which is from current model, or from ckpt folder.
        opt_2: the target opt, can be loaded from ckpt folder.
    """
    import yaml, argparse
    if isinstance(opt_1, str):
        if 'opt.yaml' not in opt_1:
            opt_1 = opt_1 + '/opt.yaml'  # Standard file name for configuration files.
        with open(opt_1, 'r') as f:
            opt_1 = yaml.unsafe_load(f)
    elif isinstance(opt_1, argparse.Namespace):
        opt_1 = vars(opt_1)

    if isinstance(opt_2, str):
        if 'opt.yaml' not in opt_2:
            opt_2 = opt_2 + '/opt.yaml'  # Standard file name for configuration files.
        with open(opt_2, 'r') as f:
            opt_2 = yaml.unsafe_load(f)
    elif isinstance(opt_2, argparse.Namespace):
        opt_2 = vars(opt_2)

    all_keys = set(list(opt_1.keys()) + list(opt_2.keys()))

    for key in all_keys:
        if key == 'model_configs':
            continue
        if key not in opt_1:
            print(f"{red('miss')} opt_1  {key}")
        elif key not in opt_2:
            print(f"{red('miss')} opt_2  {key}")
        else:  # key has to be in either of opt_1 or opt_2
            val_1 = opt_1[key]
            val_2 = opt_2[key]
            if val_1 != val_2:
                print(f'{yellow(key)}\topt_1: {val_1}\topt_2: {val_2}')


def save_toy_npy_traj_paper_v2(opt, fn, traj, n_snapshot=None, direction=None):
    #form of traj: [bs, interval, x_dim=2]
    import seaborn
    from scipy import stats

    lims = {
        'gmm': [-20, 20],
        'checkerboard': [-16, 16],
        'moon-to-spiral':[-20, 20],
        'gaussian-to-gaussian':[-20, 20],
        'moon':[-20, 20],
        'spiral':[-20, 20],
    }.get(opt.problem_name)

    num_row = 1
    num_col = np.ceil(n_snapshot/num_row).astype(int)
    total_steps = traj.shape[1]
    sample_steps = np.linspace(0, total_steps-1, n_snapshot).astype(int)
    # plt.style.use('dark_background')
    plt.style.use('default')
    fig, axes = plt.subplots(num_row, num_col, figsize=[num_col*2.5, num_row*2.5])
    plt.subplots_adjust(hspace=0.0, wspace=0.0)
    fig.patch.set_facecolor('lightgrey')
    axes = axes.reshape(-1)

    for ax, step in zip(axes, sample_steps):
        x, y = traj[:,step,0], traj[:,step,1]
        values = np.vstack([x, y])
        kernel = stats.gaussian_kde(values)(values)

        ax = fig.add_subplot(ax)
        # cmap = 'bwr' if direction == 'forward' else 'viridis'
        cmap = 'viridis'
        seaborn.scatterplot(x=x, y=y, s=8, c=kernel, vmin=-0.002, cmap=cmap, alpha=0.5)

        ax.set_xlim(*lims)
        ax.set_ylim(*lims)
        # ax.set_title('t = {:.2f}'.format((step+1)/total_steps*opt.T))
        # if direction == 'backward':
        #     ax.text(0.4, 0, f't = {(step+1)/total_steps*opt.T:.1f}', transform=ax.transAxes, fontsize=10)
        ax.axis('off')
    # fig.tight_layout()
    plt.axis('off')

    fn_fig = os.path.join('results', opt.dir, fn+'.pdf')
    plt.savefig(fn_fig, bbox_inches='tight')
    fn_fig = os.path.join('results', opt.dir, fn+'.png')
    plt.savefig(fn_fig, bbox_inches='tight', dpi=300)
    print('output fig', fn_fig)
    # plt.close()


def save_toy_npy_traj_paper_v3(opt, fn, traj, n_snapshot=None, direction=None):
    """mpl_scatter_density"""
    #form of traj: [bs, interval, x_dim=2]
    import seaborn
    import mpl_scatter_density
    import matplotlib.pyplot as plt

    lims = {
        'gmm': [-17, 17],
        'checkerboard': [-7, 7],
        'moon-to-spiral':[-20, 20],
        'gaussian-to-gaussian':[-20, 20],
        'moon':[-12, 12],
        'spiral':[-20, 20],
    }.get(opt.problem_name)

    num_row = 1
    num_col = np.ceil(n_snapshot/num_row).astype(int)
    total_steps = traj.shape[1]
    sample_steps = np.linspace(0, total_steps-1, n_snapshot).astype(int)
    # plt.style.use('dark_background')
    # fig, axes = plt.subplots(num_row, num_col, figsize=[num_col*2.5, num_row*2.5])
    # plt.subplots_adjust(hspace=0.0, wspace=0.0)
    # axes = axes.reshape(-1)
    fig = plt.figure(figsize=[num_col*2.5, num_row*2.5])
    plt.subplots_adjust(hspace=0.0, wspace=0.0)

    # for ax, step in zip(axs, sample_steps):
    for i in range(n_snapshot):
        step = sample_steps[i]
        ax = fig.add_subplot(1, n_snapshot, i+1, projection='scatter_density')

        x, y = traj[:,step,0], traj[:,step,1]
        # plt.plot(traj[:,step,0], traj[:,step,1], '.', ms=0.3, color=color)
        # seaborn.scatterplot(x=traj[:,step,0], y=traj[:,step,1], s=1.5, color=color, alpha=0.3)
        ax.scatter_density(x, y, )
        ax.set_xlim(*lims)
        ax.set_ylim(*lims)
        # ax.set_title('t = {:.2f}'.format((step+1)/total_steps*opt.T))
        # if direction == 'backward':
        #     ax.text(0.4, 0, f't = {(step+1)/total_steps*opt.T:.1f}', transform=ax.transAxes, fontsize=10)
        ax.axis('off')
    # fig.tight_layout()
    plt.axis('off')

    fn_fig = os.path.join('results', opt.dir, fn+'.pdf')
    plt.savefig(fn_fig, bbox_inches='tight')
    fn_fig = os.path.join('results', opt.dir, fn+'.png')
    plt.savefig(fn_fig, bbox_inches='tight', dpi=500)
    print('output fig', fn_fig)
    # plt.close()


def save_toy_npy_traj_paper(opt, fn, traj, n_snapshot=None, direction=None):
    #form of traj: [bs, interval, x_dim=2]

    lims = {
        'gmm': [-17, 17],
        'checkerboard': [-7, 7],
        'moon-to-spiral':[-20, 20],
        'gaussian-to-gaussian':[-20, 20],
        'moon':[-12, 12],
        'spiral':[-20, 20],
    }.get(opt.problem_name)

    num_row = 1
    num_col = np.ceil(n_snapshot/num_row).astype(int)
    total_steps = traj.shape[1]
    sample_steps = np.linspace(0, total_steps-1, n_snapshot).astype(int)
    fig, axs = plt.subplots(num_row, num_col, figsize=[num_col*2.5, num_row*2.5])
    plt.subplots_adjust(hspace=0.0, wspace=0.0)
    axs = axs.reshape(-1)

    color = 'salmon' if direction=='forward' else 'royalblue'
    for ax, step in zip(axs, sample_steps):
        ax.scatter(traj[:,step,0],traj[:,step,1], s=5, color=color)
        ax.set_xlim(*lims)
        ax.set_ylim(*lims)
        # ax.set_title('t = {:.2f}'.format((step+1)/total_steps*opt.T))
        if direction == 'backward':
            ax.text(0.4, 0, f't = {(step+1)/total_steps*opt.T:.1f}', transform=ax.transAxes, fontsize=10)
        ax.axis('off')
    # fig.tight_layout()

    fn_fig = os.path.join('results', opt.dir, fn+'.pdf')
    plt.savefig(fn_fig, bbox_inches='tight')
    fn_fig = os.path.join('results', opt.dir, fn+'.png')
    plt.savefig(fn_fig, bbox_inches='tight', dpi=500)
    print('output fig', fn_fig)
    # plt.close()


def save_toy_npy_traj(opt, fn, traj, n_snapshot=None, direction=None, num_row=3):
    #form of traj: [bs, interval, x_dim=2]

    lims = {
        'gmm': [-17, 17],
        'checkerboard': [-7, 7],
        'moon-to-spiral':[-0.5, 1.5],
        'gaussian-to-gaussian':[-20, 20],
        'moon':[-20, 20],
        # 'spiral':[-0.5, 1.5],
        'spiral':[-1, 2],
    }.get(opt.problem_name)

    if n_snapshot is None: # only store t=0
        plt.scatter(traj[:,0,0],traj[:,0,1], s=5)
        plt.xlim(*lims)
        plt.ylim(*lims)
    else:
        num_col = np.ceil(n_snapshot/num_row).astype(int)
        total_steps = traj.shape[1]
        sample_steps = np.linspace(0, total_steps-1, n_snapshot).astype(int)
        fig, axs = plt.subplots(num_row, num_col, figsize=[num_col*3, num_row*3])
        axs = axs.reshape(-1)

        color = 'salmon' if direction=='forward' else 'royalblue'
        for ax, step in zip(axs, sample_steps):
            ax.scatter(traj[:,step,0],traj[:,step,1], s=5, color=color)
            ax.set_xlim(*lims)
            ax.set_ylim(*lims)
            ax.set_title('time = {:.2f}'.format(step/(total_steps-1)*opt.T))
            ax.grid(True, linestyle=':')
        fig.tight_layout()

    if fn is not None:
        fn_npy = os.path.join('results', opt.dir, fn+'.npy')
        fn_fig = os.path.join('results', opt.dir, fn+'.png')
        plt.savefig(fn_fig)
        # np.save(fn_npy, traj)
        # plt.clf()
        print('output fig', fn_fig)
        plt.close()


def plot_inception_samples(opt, fn, traj, n_snapshot=10, direction=None, num_row=1):
    #form of traj: [bs, interval, x_dim=2]

    if n_snapshot is None: # only store t=0
        plt.scatter(traj[:,0,0],traj[:,0,1], s=5)
        # plt.xlim(*lims)
        # plt.ylim(*lims)
    else:
        num_col = np.ceil(n_snapshot/num_row).astype(int)
        total_steps = traj.shape[1]
        sample_steps = np.linspace(0, total_steps-1, n_snapshot).astype(int)
        fig, axs = plt.subplots(num_row, num_col, figsize=[num_col*3, num_row*3])
        axs = axs.reshape(-1)

        color = 'salmon' if direction=='forward' else 'royalblue'
        for ax, step in zip(axs, sample_steps):
            # hist = traj[:,step,:].sum(axis=0)
            max_ind = traj[:,step,:].argmax(axis=1)
            one_hots = F.one_hot(torch.from_numpy(max_ind), num_classes=1008)
            one_hot_hist = one_hots.sum(axis=0)
            ax.bar(range(1008), one_hot_hist)
            # ax.set_xlim(*lims)
            # ax.set_ylim(*lims)
            ax.set_title('time = {:.2f}'.format(step/(total_steps-1)*opt.T))
            ax.grid(True, linestyle=':')
        fig.tight_layout()

    if fn is not None:
        fn_npy = os.path.join('results', opt.dir, fn+'.npy')
        fn_fig = os.path.join('results', opt.dir, fn+'.png')
        plt.savefig(fn_fig)
        # np.save(fn_npy, traj)
        # plt.clf()
        print('output fig', fn_fig)
        plt.close()


def save_time_series_traj(
        opt,
        traj=None,
        n_snapshot=11,
        batch_id=0,
        fn=None,
        qlist=[0.05,0.5,0.95],
        show_figure=True):
    #form of traj: [bs, interval, x_dim]
    assert len(traj.shape) == 5  # (B,T,C,K,L)
    samples = traj
    B, T, C, K, L = samples.shape
    # samples = samples.cpu() * scaler + mean_scaler
    print(samples.shape)

    assert len(qlist) == 3
    quantiles_imp= []
    for q in qlist:
        output_samples = get_quantile(samples, q, dim=2)  # (B,T,K,L)
        quantiles_imp.append(output_samples)
    print('quantile samples shape', output_samples.shape)

    num_cols = K
    num_rows_block = np.ceil(K / num_cols).astype(int)
    num_rows = num_rows_block * n_snapshot
    sample_steps = np.linspace(0, T-1, n_snapshot).astype(int)

    gs_kw = dict(width_ratios=[1]*num_cols, height_ratios=[1]*num_rows)
    fig, axes = plt.subplots(figsize=(num_cols*2.2, num_rows*1.1), gridspec_kw=gs_kw,
        nrows=num_rows, ncols=num_cols)
    axes = axes.reshape(-1)
    plt.subplots_adjust(left=None, hspace=0, wspace=0.2)

    for step_id, step in enumerate(sample_steps):

        for k in range(K):
            ax_id = k + (num_cols*num_rows_block) * step_id
            ax = fig.add_subplot(axes[ax_id])
            # plt.text(0.1, 0.85, f'{step_id}  {k}  {ax_id}', transform=ax.transAxes, fontsize=8)

            color = 'g'  # 'g'  'tab:blue'
            ax.tick_params(labelbottom=False)
            plt.fill_between(range(0, L),
                quantiles_imp[0][batch_id,step,k,:],
                quantiles_imp[2][batch_id,step,k,:], color=color, alpha=0.3)
            plt.plot(range(0,L), quantiles_imp[1][batch_id,step,k,:], c=color,
                linestyle='solid', label='model')
            # if k == (num_rows-1)*num_cols:
            #     plt.ylabel('value')
            #     plt.xlabel('time')
            # if k >= (num_rows-1)*num_cols:
            #     ax.tick_params(labelbottom=True)
            if k == 0:
                total_steps = traj.shape[1]
                t = step/(total_steps-1)*opt.T
                plt.text(0.75, 0.85, f't={t:.2f}', transform=ax.transAxes, fontsize=8)
            plt.yticks(fontsize=7)
            # plt.ylim(-2, 2)

    if fn is not None:
        fn_npy = os.path.join('results', opt.dir, fn+'_traj.npy')
        fn_fig = os.path.join('results', opt.dir, fn+'_traj.png')
        plt.savefig(fn_fig)
        fn_fig = os.path.join('results', opt.dir, fn+'_traj.pdf')
        plt.savefig(fn_fig, bbox_inches='tight')
        np.save(fn_npy, traj.cpu())
        print('output fig', fn_fig)
    if show_figure:
        plt.show()
    else:
        plt.close()


def plot_time_series_traj_paper(
        opt,
        traj,
        all_target,
        all_evalpoint,
        all_observed,
        mean_scaler,
        scaler,
        n_snapshot=11,
        batch_id=0,
        qlist=[0.05,0.5,0.95],
        fn=None):
    #form of traj: [bs, interval, x_dim]
    assert len(traj.shape) == 5  # (B,T,C,K,L)
    samples = traj
    B, T, C, K, L = samples.shape
    print('samples.shape', samples.shape)
    print('all_target.shape', all_target.shape)

    samples = samples.cpu()
    all_target = all_target.cpu()
    all_evalpoint = all_evalpoint.cpu()
    all_observed = all_observed.cpu()
    all_given = all_observed - all_evalpoint
    mean_scaler = mean_scaler.cpu()
    scaler = scaler.cpu()

    samples = samples * scaler[None,None,None,:,None] + mean_scaler[None,None,None,:,None]  # (B,T,C,K,L)
    all_target = all_target * scaler[None,:,None] + mean_scaler[None,:,None]  #(B,K,L)

    # print('mean_scaler.shape', mean_scaler.shape)
    # print('scaler.shape', scaler.shape)
    # print('all_given.shape', all_given.shape)
    # print('all_target_np.shape', all_target_np.shape)

    nan_mask = torch.zeros_like(all_evalpoint)
    # nan_mask[all_evalpoint==0] = float('nan')
    nan_mask[all_given==1] = float('nan')
    print('nan_mask.shape', nan_mask.shape)
    nan_mask = nan_mask.unsqueeze(1)
    quantiles_imp= []
    for q in qlist:
        # output_samples = samples*(1-all_given) + all_target_np * all_given
        # output_samples = (get_quantile(samples, q, dim=2, tensor=True) * (1-all_given) +  # (B,T,K,L)
        #     all_target * all_given)
        sample_quantile = get_quantile(samples, q, dim=2, tensor=True)  # (B,T,K,L)
        output_samples = sample_quantile + nan_mask  # (B,T,K,L)
        quantiles_imp.append(output_samples)

    print('quantile samples shape', output_samples.shape)

    num_cols = K
    num_rows_block = np.ceil(K / num_cols).astype(int)
    num_rows = num_rows_block * n_snapshot
    sample_steps = np.linspace(0, T-1, n_snapshot).astype(int)

    gs_kw = dict(width_ratios=[1]*num_cols, height_ratios=[1]*num_rows)
    fig, axes = plt.subplots(figsize=(num_cols*2.2, num_rows*1.1), gridspec_kw=gs_kw,
        nrows=num_rows, ncols=num_cols)
    axes = axes.reshape(-1)
    plt.subplots_adjust(left=None, hspace=0, wspace=0.25)

    for step_id, step in enumerate(sample_steps):

        for k in range(K):
            ax_id = k + (num_cols*num_rows_block) * step_id
            ax = fig.add_subplot(axes[ax_id])
            # plt.text(0.1, 0.85, f'{step_id}  {k}  {ax_id}', transform=ax.transAxes, fontsize=8)

            # color = 'g'  # 'g'  'tab:blue'
            # ax.tick_params(labelbottom=False)
            # plt.fill_between(range(0, L),
            #     quantiles_imp[0][batch_id,step,k,:],
            #     quantiles_imp[2][batch_id,step,k,:], color=color, alpha=0.3)
            # plt.plot(range(0,L), quantiles_imp[1][batch_id,step,k,:], c=color,
            #     linestyle='solid', label='model')

            df = pd.DataFrame({"x":np.arange(0,L),
                               "val":all_target[batch_id,k,:],
                               "y":all_evalpoint[batch_id,k,:]})
            df = df[df.y != 0]
            df2 = pd.DataFrame({"x":np.arange(0,L),
                                "val":all_target[batch_id,k,:],
                                "y":all_given[batch_id,k,:]})
            df2 = df2[df2.y != 0]
            plt.plot(range(0, L), quantiles_imp[1][batch_id,step,k,:], c='g',
                linestyle='solid', label='model')
            plt.fill_between(range(0, L), quantiles_imp[0][batch_id,step,k,:],
                             quantiles_imp[2][batch_id,step,k,:], color='g', alpha=0.3)
            plt.plot(df.x,df.val, color='b', marker='.', ms=8, linestyle='None')
            plt.plot(df2.x,df2.val, color='k', marker='.', ms=4, linestyle='None')
            if k == (num_rows-1)*num_cols:
                plt.ylabel('value')
                plt.xlabel('time')
            if k >= (num_rows-1)*num_cols:
                ax.tick_params(labelbottom=True)
            # plt.ylim(plot_min - 0.05*plot_range, plot_max + 0.05*plot_range)
            ax.tick_params(labelsize=5)

            if k == 0:
                total_steps = traj.shape[1]
                t = step/(total_steps-1)*opt.T
                plt.text(0.35, 0.85, f't={t:.2f}', transform=ax.transAxes, fontsize=8)
            plt.yticks(fontsize=7)
            # plt.ylim(-2, 2)

    if fn is not None:
        fn_fig = os.path.join('results', opt.dir, fn+'_traj.png')
        plt.savefig(fn_fig)
        fn_fig = os.path.join('results', opt.dir, fn+'_traj.pdf')
        plt.savefig(fn_fig, bbox_inches='tight')
        print('output fig', fn_fig)


def plot_traj_demo(
        opt,
        traj=None,
        n_snapshot=11,
        batch_id=0,
        fn=None,
        qlist=[0.05,0.5,0.95],
        show_figure=True):
    #form of traj: [bs, interval, x_dim]
    assert len(traj.shape) == 5  # (B,T,C,K,L)
    samples = traj.cpu()
    B, T, C, K, L = samples.shape
    # samples = samples.cpu() * scaler + mean_scaler
    print(samples.shape)

    assert len(qlist) == 3
    quantiles_imp= []
    for q in qlist:
        output_samples = get_quantile(samples, q, dim=2)  # (B,T,K,L)
        quantiles_imp.append(output_samples)
    print('quantile samples shape', output_samples.shape)

    plot_features = [3, 7]
    num_plot_features = len(plot_features)
    num_cols = n_snapshot
    num_rows = num_plot_features
    sample_steps = np.linspace(0, T-1, n_snapshot).astype(int)

    gs_kw = dict(width_ratios=[1]*num_cols, height_ratios=[1]*num_rows)
    fig, axes = plt.subplots(figsize=(22, num_rows*1.1), gridspec_kw=gs_kw,
        nrows=num_rows, ncols=num_cols)
    axes = axes.reshape(-1)
    plt.subplots_adjust(left=None, hspace=0, wspace=0.2)

    for k_id, k in enumerate(plot_features):
        for step_id, step in enumerate(sample_steps):
            ax_id = step_id + num_cols * k_id
            ax = fig.add_subplot(axes[ax_id])

            ax.tick_params(labelbottom=False)
            if step_id == 0:
                plt.plot(range(0,L), samples[batch_id,0,111,k,:], '.-', c='g', ms=3)
            else:
                plt.fill_between(range(0, L),
                    quantiles_imp[0][batch_id,step,k,:],
                    quantiles_imp[2][batch_id,step,k,:], color='g', alpha=0.3)
                plt.plot(range(0,L), quantiles_imp[1][batch_id,step,k,:], c='g')
            # if k == (num_rows-1)*num_cols:
            #     plt.ylabel('value')
            #     plt.xlabel('time')
            # if k >= (num_rows-1)*num_cols:
            #     ax.tick_params(labelbottom=True)
            if k_id == 0:
                total_steps = traj.shape[1]
                t = step/(total_steps-1)*opt.T
                plt.text(0.75, 0.05, f't={t:.2f}', transform=ax.transAxes, fontsize=8)
            plt.yticks(fontsize=7)
            # plt.ylim(-2, 2)

    if fn is not None:
        plt.savefig(fn)
        print('output fig', fn)
    if show_figure:
        plt.show()
    else:
        plt.close()


def save_time_series_xT(opt, xT=None, direction=None, fn=None, show_figure=True):
    #form of traj: [bs, interval, x_dim]
    assert len(xT.shape) == 4  # (B,C,K,L)
    samples = xT  # (1,samples,K,L)
    num_batches, C, K, L = samples.shape
    # samples = samples.cpu() * scaler + mean_scaler

    qlist =[0.05, 0.5, 0.95]
    quantiles_imp= []
    for q in qlist:
        output_samples = get_quantile(samples, q, dim=1)
        quantiles_imp.append(output_samples)

    num_cols = 4
    num_rows = np.ceil(K / num_cols).astype(int)
    gs_kw = dict(width_ratios=[1]*num_cols, height_ratios=[1]*num_rows)
    fig, axes = plt.subplots(figsize=(20, num_rows*1.5), gridspec_kw=gs_kw,
        nrows=num_rows, ncols=num_cols)
    axes = axes.reshape(-1)
    plt.subplots_adjust(left=None, hspace=0, wspace=0.2)

    color = 'salmon' if direction=='forward' else 'g'
    batch_id = 0  # Only has one batch.
    for k in range(K):
        ax = fig.add_subplot(axes[k])
        ax.tick_params(labelbottom=False)
        plt.fill_between(range(0, L), quantiles_imp[0][batch_id,k,:],
                         quantiles_imp[2][batch_id,k,:], color=color, alpha=0.3)
        plt.plot(range(0,L), quantiles_imp[1][batch_id,k,:], c=color,
            linestyle='solid', label='model')
        if k == (num_rows-1)*num_cols:
            plt.ylabel('value')
            plt.xlabel('time')
        if k >= (num_rows-1)*num_cols:
            ax.tick_params(labelbottom=True)
        plt.yticks(fontsize=8)

    if fn is not None:
        fn_npy = os.path.join('results', opt.dir, fn+'_xT.npy')
        fn_fig = os.path.join('results', opt.dir, fn+'_xT.png')
        plt.savefig(fn_fig)
        fn_fig = os.path.join('results', opt.dir, fn+'_xT.pdf')
        plt.savefig(fn_fig, bbox_inches='tight')
        np.save(fn_npy, xT)
        print('output fig', fn_fig)
    if show_figure:
        plt.show()
    else:
        plt.close()


def get_FID_npz_path(opt):
    if opt.FID_ckpt is not None: return opt.FID_ckpt
    fid_npz_path = {
        # 'cifar10': 'checkpoint/cifar10_fid_stat_local.npz',
        # 'cifar10': 'data/cifar10/cifar10_fid/cifar10_fid_stat_local_50k.npz',  
        'cifar10': 'data/cifar10/fid_stats_cifar10_train.npz',  # seed from web directly.
        # 'cifar10': 'data/cifar10/cifar10_fid/cifar10_FID_test_sampls10000.npz',
        # 'imagenet': 'data/imagenet/imagenet_fid/fid_stats_imagenet_train.npz',
        'imagenet': 'data/imagenet/imagenet_fid/fid_stats_imagenet_valid.npz',
    }.get(opt.problem_name, None)
    print('fid_npz_path:', fid_npz_path)
    return fid_npz_path


def snapshot(opt, img, stage, direction, num_plots=None):

    t=-1 if direction=='forward' else 0
    if num_plots is None:
        n = 36 if opt.compute_FID else 24
    else:
        n = num_plots
    n = min(n, 2000)  # Max number of subplots.

    img = img[0:n,t,...] if len(img.shape)==5 else img[0:n,...]
    img=norm_data(opt, img) #Norm data to [0,1]

    fn = os.path.join(
        # opt.eval_path,
        opt.ckpt_path,
        direction,
        '{}stage{}.png'.format('sample_' if opt.compute_FID else '', stage)
    )
    # Number of images displayed in each row of the grid.
    torchvision.utils.save_image(img, fn, nrow=6 if n < 60 else 32)
    print('save images:', fn)

def save_generated_data(opt, x):
    timestamp = dt.datetime.now().strftime("_%m_%d_%Y_%H%M%S")
    generated_data_path = opt.generated_data_path + timestamp
    os.makedirs(generated_data_path, exist_ok=True)
    print('save data to', generated_data_path)
    x = norm_data(opt,x)
    x = torch.squeeze(x)
    for i in range(x.shape[0]):
        fn = os.path.join(generated_data_path, 'img{}.jpg'.format(i))
        torchvision.utils.save_image(x[i,...], fn)
    return generated_data_path

def compute_fid(opt, xTs):
    FID_path = get_FID_npz_path(opt)
    # save_generated_data(opt, xTs.to(opt.device))
    generated_data_path = save_generated_data(opt, xTs.to('cpu'))
    return get_fid(FID_path, generated_data_path)

def exist_FID_ckpt(opt):
    ckpt = get_FID_npz_path(opt)
    return ckpt is not None and os.path.exists(ckpt)

def norm_data(opt,x):
    if opt.problem_name=='mnist':
        x=x.repeat(1,3,1,1)
    _max=torch.max(torch.max(x,dim=-1)[0],dim=-1)[0][...,None,None]
    _min=torch.min(torch.min(x,dim=-1)[0],dim=-1)[0][...,None,None]
    x=(x-_min)/(_max-_min)
    return x

def check_duplication(opt):
    plt_dir='plots/'+opt.dir
    ckpt_dir='checkpoint/'+opt.group+'/'+opt.name
    runs_dir='runs/'+opt.log_fn
    plt_flag=os.path.isdir(plt_dir)
    ckpt_flag=os.path.isdir(ckpt_dir)
    run_flag=os.path.isdir(runs_dir)
    tot_flag= plt_flag or ckpt_flag or run_flag
    print([plt_flag,ckpt_flag,run_flag])
    if tot_flag:
        decision=input('Exist duplicated folder, do you want to overwrite it? [y/n]')

        if 'y' in decision:
            try:
                shutil.rmtree(plt_dir)
            except:
                pass
            try: 
                shutil.rmtree(ckpt_dir)
            except:
                pass
            try:
                shutil.rmtree(runs_dir)
            except:
                pass
        else:
            sys.exit()

######################################################################################
##                          Copy of FID computation utils                           ##
##  Ref: https://github.com/ermongroup/ncsnv2/blob/master/evaluation/fid_score.py,  ##
##       https://github.com/ermongroup/ncsnv2/blob/master/evaluation/inception.py,  ##
######################################################################################

def imread(filename):
    """
    Loads an image file into a (height, width, 3) uint8 ndarray.
    """
    return np.asarray(Image.open(filename), dtype=np.uint8)[..., :3]


def get_activations(files, model, batch_size=50, dims=2048,
                    cuda=False, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)

    pred_arr = np.empty((len(files), dims))

    for i in tqdm(range(0, len(files), batch_size)):
        if verbose:
            print('\rPropagating batch %d/%d' % (i + 1, batch_size),
                  end='', flush=True)
        start = i
        end = i + batch_size

        images = np.array([imread(str(f)).astype(np.float32)
                           for f in files[start:end]])

        # Reshape to (n_images, 3, height, width)
        images = images.transpose((0, 3, 1, 2))
        images /= 255

        batch = torch.from_numpy(images).type(torch.FloatTensor)
        if cuda:
            batch = batch.cuda()

        pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred_arr[start:end] = pred.cpu().data.numpy().reshape(pred.size(0), -1)

    if verbose:
        print(' done')

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(files, model, batch_size=50,
                                    dims=2048, cuda=False, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(files, model, batch_size, dims, cuda, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def _compute_statistics_of_path(path, model, batch_size, dims, cuda):
    if path.endswith('.npz'):
        f = np.load(path)
        m, s = f['mu'][:], f['sigma'][:]
        f.close()
    else:
        path = pathlib.Path(path)
        files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
        m, s = calculate_activation_statistics(files, model, batch_size,
                                               dims, cuda)

    return m, s

def calculate_fid_npz(path, root, name, batch_size=256, cuda=True, dims=2048):
    """Calculates the FID of two paths"""

    from models.InceptionNet.inception_net import InceptionV3

    if not os.path.exists(path):
        raise RuntimeError('Invalid path: %s' % path)
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx])
    if cuda:
        model.cuda()
    m1, s1 = _compute_statistics_of_path(path, model, batch_size,
                                         dims, cuda)

    if not os.path.exists(root):
        os.makedirs(root)
    np.savez(root+name, mu=m1, sigma=s1)

def calculate_fid_given_paths(paths, batch_size, cuda, dims):
    """Calculates the FID of two paths"""
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)

    from models.InceptionNet.inception_net import InceptionV3

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx])
    if cuda:
        model.cuda()
    m1, s1 = _compute_statistics_of_path(paths[0], model, batch_size,
                                         dims, cuda)

    m2, s2 = _compute_statistics_of_path(paths[1], model, batch_size,
                                         dims, cuda)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


def get_fid(path1, path2):
    fid_value = calculate_fid_given_paths([path1, path2], batch_size=256, cuda=True, dims=2048)
    return fid_value

def get_fid_stats_path(args, config, download=True):

    links = {
        'CIFAR10': 'http://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_cifar10_train.npz',
        'LSUN': 'http://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_lsun_train.npz'
    }
    if config.data.dataset == 'CIFAR10':
        path = os.path.join(args.exp, 'datasets', 'cifar10_fid.npz')
        if not os.path.exists(path):
            if not download:
                raise FileNotFoundError("no statistics file founded")
            else:
                import urllib
                urllib.request.urlretrieve(
                    links[config.data.dataset], path
                )
    elif config.data.dataset == 'CELEBA':
        path = os.path.join(args.exp, 'datasets', 'celeba_test_fid_stats.npz')
        if not os.path.exists(path):
            raise FileNotFoundError('no statistics file founded')

    return path


def generate_inception_probabilities(path, root, name, batch_size=256):
    """Calculates the FID of two paths"""
    from models.InceptionNet import inception_net
    from models.InceptionNet.inception_net import InceptionV3
    from torchvision import transforms
    import torch.nn.functional as F

    if not os.path.exists(path):
        raise RuntimeError('Invalid path: %s' % path)

    model  = inception_net.fid_inception_v3()
    model.cuda()

    path = pathlib.Path(path)
    files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
    # files = files[:2000]
    print('num files:', len(files))

    model.eval()
    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)

    pred_arr = np.zeros((len(files), 1008))
    for i in tqdm(range(0, len(files), batch_size)):
        start, end = i, i + batch_size
        images = np.array([imread(str(f)).astype(np.float32) for f in files[start:end]])       
        images = images.transpose((0, 3, 1, 2))  # Reshape to (n_images, 3, height, width)
        images /= 255
        batch = torch.from_numpy(images).type(torch.FloatTensor)
        batch = batch.cuda()
        batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
        # batch = 2 * batch - 1  # Scale from range (0, 1) to range (-1, 1)
        output = model(batch)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        pred_arr[start:end] = probabilities.detach().cpu().numpy()

    print('save file size:', pred_arr.shape)
    if not os.path.exists(root):
        os.makedirs(root)
    np.save(root+name, pred_arr)

