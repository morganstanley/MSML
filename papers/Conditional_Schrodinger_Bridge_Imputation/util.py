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

def is_image_dataset(opt):
    return opt.problem_name in ['mnist','cifar10','celebA32','celebA64']

def is_toy_dataset(opt):
    return opt.problem_name in ['gmm','checkerboard', 'moon-to-spiral']

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

def restore_checkpoint(opt, runner, load_name):
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

    else:
        full_keys = ['z_f','optimizer_f','ema_f','z_b','optimizer_b','ema_b']

        with torch.cuda.device(opt.gpu):
            checkpoint = torch.load(load_name,map_location=opt.device)
            print('state_dict keys:', checkpoint.keys())
            ckpt_keys=[*checkpoint.keys()]
            for k in ckpt_keys:
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


def save_toy_npy_traj(opt, fn, traj, n_snapshot=None, direction=None):
    #form of traj: [bs, interval, x_dim=2]
    fn_npy = os.path.join('results', opt.dir, fn+'.npy')
    fn_fig = os.path.join('results', opt.dir, fn+'.png')

    lims = {
        'gmm': [-17, 17],
        'checkerboard': [-7, 7],
        'moon-to-spiral':[-20, 20],
    }.get(opt.problem_name)

    if n_snapshot is None: # only store t=0
        plt.scatter(traj[:,0,0],traj[:,0,1], s=5)
        plt.xlim(*lims)
        plt.ylim(*lims)
    else:
        num_row = 3
        num_col = np.ceil(n_snapshot/num_row).astype(int)
        total_steps = traj.shape[1]
        sample_steps = np.linspace(0, total_steps-1, n_snapshot).astype(int)
        fig, axs = plt.subplots(num_row, num_col)
        axs = axs.reshape(-1)
        fig.set_size_inches(num_col*3, num_row*3)

        color = 'salmon' if direction=='forward' else 'royalblue'
        for ax, step in zip(axs, sample_steps):
            ax.scatter(traj[:,step,0],traj[:,step,1], s=5, color=color)
            ax.set_xlim(*lims)
            ax.set_ylim(*lims)
            ax.set_title('time = {:.2f}'.format(step/(total_steps-1)*opt.T))
        fig.tight_layout()

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

    nan_mask = torch.zeros_like(all_evalpoint)
    # nan_mask[all_evalpoint==0] = float('nan')
    nan_mask[all_given==1] = float('nan')
    print('nan_mask.shape', nan_mask.shape)
    nan_mask = nan_mask.unsqueeze(1)
    quantiles_imp= []
    for q in qlist:
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
            ax.tick_params(labelsize=5)

            if k == 0:
                total_steps = traj.shape[1]
                t = step/(total_steps-1)*opt.T
                plt.text(0.35, 0.85, f't={t:.2f}', transform=ax.transAxes, fontsize=8)
            plt.yticks(fontsize=7)

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
            if k_id == 0:
                total_steps = traj.shape[1]
                t = step/(total_steps-1)*opt.T
                plt.text(0.75, 0.05, f't={t:.2f}', transform=ax.transAxes, fontsize=8)
            plt.yticks(fontsize=7)

    if fn is not None:
        plt.savefig(fn)
        print('output fig', fn)
    if show_figure:
        plt.show()
    else:
        plt.close()


def plot_imputation_time_series_traj(
        opt,
        traj,
        obs_data,  # obs_data
        obs_mask,  # obs_mask
        target_mask,  # target_mask
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
    print('traj.shape (B,T,C,K,L)', samples.shape)

    samples = samples.cpu()
    all_target = obs_data.cpu().unsqueeze(1)
    all_evalpoint = target_mask.cpu().unsqueeze(1)
    all_observed = obs_mask.cpu().unsqueeze(1)
    all_given = all_observed - all_evalpoint

    assert len(qlist) == 3
    quantiles_imp= []
    nan_mask = torch.zeros_like(all_given)
    nan_mask[all_given==1] = float('nan')
    for q in qlist:
        output_samples = get_quantile(samples, q, dim=2, tensor=True) + nan_mask  # (B,T,K,L)
        quantiles_imp.append(output_samples)

    num_cols = 8
    num_rows_block = np.ceil(K / num_cols).astype(int)
    num_rows = num_rows_block * n_snapshot
    sample_steps = np.linspace(0, T-1, n_snapshot).astype(int)

    gs_kw = dict(width_ratios=[1]*num_cols, height_ratios=[1]*num_rows)
    fig, axes = plt.subplots(figsize=(22, num_rows*1.1), gridspec_kw=gs_kw,
        nrows=num_rows, ncols=num_cols)
    axes = axes.reshape(-1)
    plt.subplots_adjust(left=None, hspace=0, wspace=0.2)

    for step_id, step in enumerate(sample_steps):
        for k in range(K):
            ax_id = k + (num_cols*num_rows_block) * step_id
            ax = fig.add_subplot(axes[ax_id])

            ax.tick_params(labelbottom=False)
            plt.fill_between(range(0, L),
                quantiles_imp[0][batch_id,step,k,:],
                quantiles_imp[2][batch_id,step,k,:], color='g', alpha=0.3)
            plt.plot(range(0,L), quantiles_imp[1][batch_id,step,k,:], c='g',
                linestyle='solid', label='model')

            df = pd.DataFrame({"x":np.arange(0,L),
                               "val":all_target[batch_id,0,k,:],
                               "y":all_evalpoint[batch_id,0,k,:]})
            df = df[df.y != 0]
            df2 = pd.DataFrame({"x":np.arange(0,L),
                                "val":all_target[batch_id,0,k,:],
                                "y":all_given[batch_id,0,k,:]})
            df2 = df2[df2.y != 0]
            plt.plot(df2.x, df2.val, '.k', ms=2)

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
        np.save(fn_npy, traj)
        print('output fig', fn_fig)
    if show_figure:
        plt.show()
    else:
        plt.close()


def plot_imputation_time_series_traj_demo(
        opt,
        traj,
        obs_data,  # obs_data
        obs_mask,  # obs_mask
        target_mask,  # target_mask
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
    print('traj.shape (B,T,C,K,L)', samples.shape)

    samples = samples.cpu()
    all_target = obs_data.cpu().unsqueeze(1)
    all_evalpoint = target_mask.cpu().unsqueeze(1)
    all_observed = obs_mask.cpu().unsqueeze(1)
    all_given = all_observed - all_evalpoint

    assert len(qlist) == 3
    quantiles_imp= []

    nan_mask = torch.zeros_like(all_given)
    nan_mask[all_given==1] = float('nan')
    for q in qlist:
        output_samples = get_quantile(samples, q, dim=2, tensor=True) + nan_mask  # (B,T,K,L)
        quantiles_imp.append(output_samples)

    plot_features = [5, 6]
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
            plt.fill_between(range(0, L),
                quantiles_imp[0][batch_id,step,k,:],
                quantiles_imp[2][batch_id,step,k,:], color='g', alpha=0.3)
            plt.plot(range(0,L), quantiles_imp[1][batch_id,step,k,:], c='g',
                linestyle='solid', label='model')

            df = pd.DataFrame({"x":np.arange(0,L),
                               "val":all_target[batch_id,0,k,:],
                               "y":all_evalpoint[batch_id,0,k,:]})
            df = df[df.y != 0]
            df2 = pd.DataFrame({"x":np.arange(0,L),
                                "val":all_target[batch_id,0,k,:],
                                "y":all_given[batch_id,0,k,:]})
            df2 = df2[df2.y != 0]
            plt.plot(df2.x, df2.val, '.k', ms=2)

            if k_id == 0:
                total_steps = traj.shape[1]
                t = step/(total_steps-1)*opt.T
                # plt.text(0.75, 0.85, f't={t:.2f}', transform=ax.transAxes, fontsize=8)
                plt.text(0.75, 0.8, f't={t:.2f}', transform=ax.transAxes, fontsize=9)
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


def get_quantile(samples, q, dim=1, tensor=False):
    if tensor:
        return torch.quantile(samples, q, dim=dim).cpu()
    else:  # return numpy
        return torch.quantile(samples, q, dim=dim).cpu().numpy()


def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )


def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))


def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):
    """
    Args:
        target: B,1,L,K
        forecast: B, num_samples, L, K
    """
    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    # denom = calc_denominator(target, eval_points)
    denom = torch.sum(torch.abs(target * eval_points))
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)


def unconditional_imputation_eval(
        opt,
        test_queue,
        run,
        num_samples=128,
        mean_scaler=0.0,
        scaler=1.0,
        num_eval_batches=None,
        output_dir=None):
    mean_scaler, scaler = torch.tensor(mean_scaler).cpu(), torch.tensor(scaler).cpu()

    mse_total = 0
    mae_total = 0
    evalpoints_total = 0
    all_target = []
    all_observed_point = []
    all_observed_time = []
    all_evalpoint = []
    all_generated_samples = []

    torch.cuda.device(opt.gpu)
    # model.eval()
    for batch_id, batch in enumerate(test_queue):
        if num_eval_batches is not None and batch_id >= num_eval_batches:
            break
        if opt.permute_batch:
            observed_data = batch['observed_data'].float().permute(0,2,1).unsqueeze(1)  # (B,K,L)
            obs_mask = batch['observed_mask'].float().permute(0,2,1).unsqueeze(1)
            gt_mask = batch['gt_mask'].float().permute(0,2,1).unsqueeze(1)
            x = gt_mask * observed_data + (1.0 - gt_mask) * torch.randn_like(observed_data) * 0.1

        elif not opt.permute_batch:
            observed_data = batch['observed_data'].float()
            obs_mask = batch['observed_mask'].float()
            gt_mask = batch['gt_mask'].float()
            # x = gt_mask * observed_data + (1.0 - gt_mask) * torch.randn_like(observed_data) * 0.1
            x = (observed_data*gt_mask).unsqueeze(1)
            obs_mask = obs_mask.unsqueeze(1)
            gt_mask = gt_mask.unsqueeze(1)

        B, C, K, L = observed_data.shape
        observed_time = batch["timepoints"].float()
        cond_mask = gt_mask
        target_mask = obs_mask - cond_mask

        # Unconditionally draw samples.
        samples, _ = run._generate_samples_and_reused_trajs(opt, 1024, B*num_samples, 0,)
        samples = samples.cpu().reshape(B,num_samples,K,L).permute(0,1,3,2)  # (B,samples,L,K)
        print(samples.shape)

        c_target = observed_data.squeeze(1).permute(0,2,1)  # (B,L,K)
        eval_points = target_mask.squeeze(1).permute(0,2,1)
        observed_points = obs_mask.squeeze(1).permute(0,2,1)

        all_generated_samples.append(samples.float())
        all_target.append(c_target)
        all_evalpoint.append(eval_points)
        all_observed_point.append(observed_points)
        all_observed_time.append(observed_time)

    all_target = torch.cat(all_target, dim=0)
    all_evalpoint = torch.cat(all_evalpoint, dim=0)
    all_observed_point = torch.cat(all_observed_point, dim=0)
    all_observed_time = torch.cat(all_observed_time, dim=0)
    all_generated_samples = torch.cat(all_generated_samples, dim=0)

    samples_median = all_generated_samples.median(dim=1)
    evalpoints_total = all_evalpoint.sum().item()
    mse_list = torch.square((samples_median.values - all_target) * all_evalpoint) * (scaler ** 2)
    mae_list = torch.abs((samples_median.values - all_target) * all_evalpoint) * scaler
    CRPS = calc_quantile_CRPS(all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler)
    print('total number of eval points:', evalpoints_total)
    print("RMSE:", np.sqrt(mse_list.sum().item()/evalpoints_total))
    print("MAE:", mae_list.sum().item()/evalpoints_total)
    print("CRPS:", CRPS)

    if output_dir is not None:
        import pickle
        output_path = (output_dir + '/generated_' + str(num_samples) + '_samples_' +
                       dt.datetime.now().strftime("%m_%d_%Y_%H%M%S") + '.pk')
        with open(output_path, 'wb') as f:
            pickle.dump([all_generated_samples, all_target, all_evalpoint, all_observed_point,
                all_observed_time, scaler, mean_scaler], f)
        print('saved samples:', output_path)


def conditional_imputation_eval(
        impute_function,
        opt,
        test_queue,
        run,
        num_samples=128,
        mean_scaler=0.0,
        scaler=1.0,
        num_eval_batches=None,
        output_dir=None,
        samples_path=None,
        metrics_path=None):
    """
    impute_function:
        cond_traj: consistent with original sampling function `_generate_samples_and_reused_trajs`.
        imputation: simplified sampling function for non-vpsde.
        imputation_vpsde: simplified sampling function only for vpsde.
        csdi: for csdi model.

    """
    mean_scaler, scaler = torch.tensor(mean_scaler).cpu(), torch.tensor(scaler).cpu()

    mse_total = 0
    mae_total = 0
    evalpoints_total = 0
    all_target = []
    all_observed_point = []
    all_observed_time = []
    all_evalpoint = []
    all_generated_samples = []

    torch.cuda.device(opt.gpu)
    # model.eval()
    for batch_id, batch in enumerate(test_queue):
        if num_eval_batches is not None and batch_id >= num_eval_batches:
            break
        if opt.permute_batch:
            observed_data = batch['observed_data'].float().permute(0,2,1).unsqueeze(1)  # (B,K,L)
            obs_mask = batch['observed_mask'].float().permute(0,2,1).unsqueeze(1)
            gt_mask = batch['gt_mask'].float().permute(0,2,1).unsqueeze(1)

        elif not opt.permute_batch:
            observed_data = batch['observed_data'].float().unsqueeze(1)
            obs_mask = batch['observed_mask'].float().unsqueeze(1)
            gt_mask = batch['gt_mask'].float().unsqueeze(1)

        B, C, K, L = observed_data.shape
        observed_time = batch["timepoints"].float()
        cond_mask = gt_mask
        target_mask = obs_mask - cond_mask
        x_cond = observed_data * cond_mask

        if impute_function == 'cond_traj':
            samples, _ = run._generate_samples_and_reused_trajs_conditional(
                opt, x_cond.to(opt.device), obs_mask.to(opt.device), cond_mask.to(opt.device),
                target_mask.to(opt.device), num_samples=num_samples)

        elif impute_function == 'imputation':
            samples = run.imputation(
                opt, x_cond.to(opt.device), obs_mask.to(opt.device), cond_mask.to(opt.device),
                target_mask.to(opt.device), num_samples=num_samples)

        elif impute_function == 'imputation_vpsde':
            samples = run.imputation_vpsde(
                opt, x_cond.to(opt.device), obs_mask.to(opt.device), cond_mask.to(opt.device),
                target_mask.to(opt.device), num_samples=num_samples)

        elif impute_function == 'csdi':
            samples = run.imputation_csdi(
                opt, x_cond.to(opt.device), obs_mask.to(opt.device), cond_mask.to(opt.device),
                target_mask.to(opt.device), num_samples=num_samples)
            print('samples.shape', samples.shape)

        else:
            raise NotImplementedError('Imputation inference function not known.')

        samples = samples.cpu().reshape(B,num_samples,K,L).permute(0,1,3,2)  # (B,samples,L,K)
        c_target = observed_data.cpu().squeeze(1).permute(0,2,1)  # (B,L,K)
        eval_points = target_mask.cpu().squeeze(1).permute(0,2,1)
        observed_points = obs_mask.cpu().squeeze(1).permute(0,2,1)

        all_generated_samples.append(samples.float())
        all_target.append(c_target)
        all_evalpoint.append(eval_points)
        all_observed_point.append(observed_points)
        all_observed_time.append(observed_time)

    all_target = torch.cat(all_target, dim=0)
    all_evalpoint = torch.cat(all_evalpoint, dim=0)
    all_observed_point = torch.cat(all_observed_point, dim=0)
    all_observed_time = torch.cat(all_observed_time, dim=0)
    all_generated_samples = torch.cat(all_generated_samples, dim=0)

    samples_median = all_generated_samples.median(dim=1)
    evalpoints_total = all_evalpoint.sum().item()
    mse_list = torch.square((samples_median.values - all_target) * all_evalpoint) * (scaler ** 2)
    mae_list = torch.abs((samples_median.values - all_target) * all_evalpoint) * scaler
    CRPS = calc_quantile_CRPS(all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler)
    rmse = np.sqrt(mse_list.sum().item()/evalpoints_total)
    mae = mae_list.sum().item()/evalpoints_total
    print('total number of eval points:', evalpoints_total)
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("CRPS:", CRPS)

    corrector_flag = '_corrector' if opt.use_corrector else ''
    if samples_path is None and output_dir is not None:
        samples_path = (output_dir + '/generated_' + str(num_samples) + '_samples_' +
                       dt.datetime.now().strftime("%m_%d_%Y_%H%M%S") + f'{corrector_flag}.pk')
    if metrics_path is None and output_dir is not None:
        metrics_path = (output_dir + '/metrics_' +
                        dt.datetime.now().strftime("%m_%d_%Y_%H%M%S") + f'{corrector_flag}.json')

    if samples_path is not None:
        import pickle
        with open(samples_path, 'wb') as f:
            pickle.dump([all_generated_samples, all_target, all_evalpoint, all_observed_point,
                all_observed_time, scaler, mean_scaler], f)
            print('saved samples:', samples_path)

    if metrics_path is not None:
        import json
        metrics_dict = {'RMSE': rmse, 'MAE': mae, 'CRPS': CRPS, 'evalpoints_total': evalpoints_total,
            'samples_path': samples_path}
        with open(metrics_path, 'w') as fp:
            json.dump(metrics_dict, fp, indent=4)
            print('saved metrics:', metrics_path)


def plot_saved_imputation_samples(
        samples_path,
        dataind=12,
        num_cols=8,
        qlist=[0.1,0.5,0.9],
        back_transform=True,
        figure_path=None):
    """Visualize the imputation."""
    import pickle
    with open(samples_path, 'rb') as f:
        (samples, all_target, all_evalpoint, all_observed, all_observed_time,
            scaler, mean_scaler) = pickle.load( f)  # (B,L,K)

    num_batches, num_samples, L, K = samples.shape
    samples = samples.cpu()
    all_target = all_target.cpu()
    all_evalpoint = all_evalpoint.cpu()
    all_observed = all_observed.cpu()
    all_given = all_observed - all_evalpoint
    mean_scaler = mean_scaler.cpu()
    scaler = scaler.cpu()

    if back_transform and len(mean_scaler.shape)  == 3 and len(scaler.shape)  == 3:
        all_target = all_target * scaler + mean_scaler
        samples = samples * scaler.unsqueeze(1) + mean_scaler.unsqueeze(1)
    elif back_transform:
        all_target = all_target * scaler + mean_scaler
        samples = samples * scaler + mean_scaler
    else:
        # Do nothing here to present data in normalized space for debugging perspection.
        pass

    nan_mask = torch.zeros_like(all_evalpoint)
    # nan_mask[all_evalpoint==0] = float('nan')
    nan_mask[all_given==1] = float('nan')
    quantiles_imp= []
    for q in qlist:
        # output_samples = samples*(1-all_given) + all_target_np * all_given
        # output_samples = (get_quantile(samples, q, dim=2, tensor=True) * (1-all_given) +  # (B,T,K,L)
        #     all_target * all_given)
        output_samples = get_quantile(samples, q, dim=1, tensor=True) + nan_mask  # (B,T,K,L)
        quantiles_imp.append(output_samples)

    num_rows = np.ceil(K / num_cols).astype(int)
    gs_kw = dict(width_ratios=[1]*num_cols, height_ratios=[1]*num_rows)
    fig, axes = plt.subplots(figsize=(num_cols*3, num_rows*1.5), gridspec_kw=gs_kw,
        nrows=num_rows, ncols=num_cols)
    plt.rcParams["font.size"] = 10
    axes = axes.reshape(-1)
    plt.subplots_adjust(left=None, hspace=0.1, wspace=0.25)

    plot_min = quantiles_imp[1][dataind,:,:].min()
    plot_max = quantiles_imp[1][dataind,:,:].max()
    plot_range = plot_max - plot_min

    for k in range(K):
        ax = fig.add_subplot(axes[k])
        ax.tick_params(labelbottom=False)

        df = pd.DataFrame({"x":np.arange(0,L),
                           "val":all_target[dataind,:,k],
                           "y":all_evalpoint[dataind,:,k]})
        df = df[df.y != 0]
        df2 = pd.DataFrame({"x":np.arange(0,L),
                            "val":all_target[dataind,:,k],
                            "y":all_given[dataind,:,k]})
        df2 = df2[df2.y != 0]
        plt.plot(range(0, L), quantiles_imp[1][dataind,:,k], c='g',
            linestyle='solid', label='model')
        plt.fill_between(range(0, L), quantiles_imp[0][dataind,:,k],
                         quantiles_imp[2][dataind,:,k], color='g', alpha=0.3)
        plt.plot(df.x,df.val, color='b', marker='.', ms=10, linestyle='None')
        plt.plot(df2.x,df2.val, color='k', marker='.', ms=6, linestyle='None')
        if k == (num_rows-1)*num_cols:
            plt.ylabel('value')
            plt.xlabel('time')
        if k >= (num_rows-1)*num_cols:
            ax.tick_params(labelbottom=True)
        # plt.ylim(plot_min - 0.05*plot_range, plot_max + 0.05*plot_range)

    if figure_path is not None:
        # fig.tight_layout()
        plt.savefig(figure_path, bbox_inches='tight')
        print('save figure', figure_path)
    plt.show()


def plot_saved_imputation_samples_demo(
        samples_path,
        dataind=12,
        num_cols=8,
        qlist=[0.1,0.5,0.9],
        back_transform=True,
        figure_path=None):
    """Visualize the imputation."""
    import pickle
    with open(samples_path, 'rb') as f:
        (samples, all_target, all_evalpoint, all_observed, all_observed_time,
            scaler, mean_scaler) = pickle.load( f)  # (B,L,K)

    num_batches, num_samples, L, K = samples.shape
    samples = samples.cpu()
    all_target = all_target.cpu()
    all_evalpoint = all_evalpoint.cpu()
    all_observed = all_observed.cpu()
    all_given = all_observed - all_evalpoint
    mean_scaler = mean_scaler.cpu()
    scaler = scaler.cpu()

    if back_transform and len(mean_scaler.shape)  == 3 and len(scaler.shape)  == 3:
        all_target = all_target * scaler + mean_scaler
        samples = samples * scaler.unsqueeze(1) + mean_scaler.unsqueeze(1)
    elif back_transform:
        all_target = all_target * scaler + mean_scaler
        samples = samples * scaler + mean_scaler
    else:
        # Do nothing here to present data in normalized space for debugging perspection.
        pass

    # Only for sinuoid demo.
    all_evalpoint_full = torch.zeros(samples.shape[0],50,8)
    for j in range(8):
        mod = j % 4
        all_evalpoint_full[:,mod*10:mod*10 + 20,j]=1
    nan_mask = torch.zeros_like(all_evalpoint)
    nan_mask[all_evalpoint_full==0] = float('nan')
    quantiles_imp= []
    for q in qlist:
        output_samples = get_quantile(samples, q, dim=1, tensor=True) + nan_mask  # (B,T,K,L)
        quantiles_imp.append(output_samples)

    selected_feature = [0,1,2,3]
    num_rows = 1
    gs_kw = dict(width_ratios=[1]*num_cols, height_ratios=[1]*num_rows)
    fig, axes = plt.subplots(figsize=(11, 2), gridspec_kw=gs_kw,
        nrows=num_rows, ncols=num_cols)
    plt.rcParams["font.size"] = 10
    axes = axes.reshape(-1)
    plt.subplots_adjust(left=None, hspace=0, wspace=0.25)

    plot_min = quantiles_imp[1][dataind,:,:].min()
    plot_max = quantiles_imp[1][dataind,:,:].max()
    plot_range = plot_max - plot_min

    for k_id, k in enumerate(selected_feature):
        ax = fig.add_subplot(axes[k_id])
        ax.tick_params(labelbottom=False)

        df = pd.DataFrame({"x":np.arange(0,L),
                           "val":all_target[dataind,:,k],
                           "y":all_evalpoint[dataind,:,k]})
        df = df[df.y != 0]
        df2 = pd.DataFrame({"x":np.arange(0,L),
                            "val":all_target[dataind,:,k],
                            "y":all_given[dataind,:,k]})
        df2 = df2[df2.y != 0]
        plt.plot(range(0, L), quantiles_imp[1][dataind,:,k], c='g',
            linestyle='solid', label='model')
        plt.fill_between(range(0, L), quantiles_imp[0][dataind,:,k],
                         quantiles_imp[2][dataind,:,k], color='g', alpha=0.3)
        plt.plot(df.x,df.val, color='b', marker='.', ms=10, linestyle='None')
        plt.plot(df2.x,df2.val, color='k', marker='.', ms=6, linestyle='None')
        # plt.ylim(-1.3, 1.5)
        if k == (num_rows-1)*num_cols:
            plt.ylabel('value')
            plt.xlabel('time')
        if k >= (num_rows-1)*num_cols:
            ax.tick_params(labelbottom=True)
        # plt.ylim(plot_min - 0.05*plot_range, plot_max + 0.05*plot_range)
    if figure_path is not None:
        fig.tight_layout()
        plt.savefig(figure_path)
        print('figure_path', figure_path)
    plt.show()

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
        opt.eval_path,
        direction,
        '{}stage{}.png'.format('sample_' if opt.compute_FID else '', stage)
    )
    # Number of images displayed in each row of the grid.
    torchvision.utils.save_image(img, fn, nrow=6 if n < 60 else 32)
    print('save images:', fn)

def save_generated_data(opt, x):
    x = norm_data(opt,x)
    x = torch.squeeze(x)
    for i in range(x.shape[0]):
        fn = os.path.join(opt.generated_data_path, 'img{}.jpg'.format(i))
        tu.save_image(x[i,...], fn)

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
