# Copyright (c) 2024, Wei Deng. All rights reserved.
#
# This work is licensed under APACHE LICENSE, VERSION 2.0
# You should have received a copy of the license along with this
# work. If not, see https://www.apache.org/licenses/LICENSE-2.0.txt

"""
Our implementation of Forward-backward Schrodinger Bridge training on CIFAR10 dataset 
in ICML2024 paper "Variational Schrödinger Diffusion Models".
by Wei Deng, Weijian Luo, Yixin Tan, Marin Biloš, Yu Chen, Yuriy Nevmyvaka, Ricky T. Q. Chen
"""

"""Main training loop."""

import os
import time
import copy
import json
import pickle
import psutil
import numpy as np
import torch
import dnnlib
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc

import PIL.Image

def grid(array, ncols=8):
    array = np.pad(array, [(0,0),(1,1),(1,1),(0,0)], 'constant')
    nindex, height, width, intensity = array.shape
    ncols = min(nindex, ncols)
    nrows = (nindex+ncols-1)//ncols
    r = nrows*ncols - nindex # remainder
    # want result.shape = (height*nrows, width*ncols, intensity)
    arr = np.concatenate([array]+[np.zeros([1,height,width,intensity])]*r)
    result = (arr.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols, intensity))
    return np.pad(result, [(1,1),(1,1),(0,0)], 'constant')

#----------------------------------------------------------------------------

def sample_traj(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=100, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, forward=True,
):
    ## if forward: x_data --> x_noise; if backward: x_noise --> x_data

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps)])
    t_steps = torch.flip(t_steps, dims=[0]) if forward else t_steps

    all_xs = []
    all_sigmas = []
    all_dts = []

    # Main sampling loop.
    x_next = latents.to(torch.float64)

    # all_xs.append(x_next.to('cpu'))
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        dt = (t_next - t_cur)
        dx_dt = (x_cur - net(x_cur, t_cur, class_labels, augment_labels=None).to(torch.float64)) / t_cur

        dx_dt = -dx_dt if forward else dx_dt ## according to Schrodinger System, might use this to generate
        
        # x_next = x_cur + dx_dt*dt ## ODE
        x_next = x_cur + 2*dx_dt*dt + (2*t_cur*(t_next-t_cur).abs())**0.5*torch.randn_like(x_cur) ## SDE sampling

        all_xs.append(x_next)
        all_sigmas.append(t_next)
        all_dts.append((t_next - t_cur).abs())

    return all_xs, all_sigmas, all_dts


def training_loop(
    run_dir             = '.',      # Output directory.
    dataset_kwargs      = {},       # Options for training set.
    data_loader_kwargs  = {},       # Options for torch.utils.data.DataLoader.
    network_kwargs      = {},       # Options for model and preconditioning.
    loss_kwargs         = {},       # Options for loss function.
    optimizer_kwargs    = {},       # Options for optimizer.
    augment_kwargs      = None,     # Options for augmentation pipeline, None = disable.
    seed                = 0,        # Global random seed.
    batch_size          = 512,      # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU, None = no limit.
    total_kimg          = 200000,   # Training duration, measured in thousands of training images.
    ema_halflife_kimg   = 500,      # Half-life of the exponential moving average (EMA) of model weights.
    ema_rampup_ratio    = 0.05,     # EMA ramp-up coefficient, None = no rampup.
    lr_rampup_kimg      = 10000,    # Learning rate ramp-up duration.
    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    kimg_per_tick       = 50,       # Interval of progress prints.
    snapshot_ticks      = 50,       # How often to save network snapshots, None = disable.
    state_dump_ticks    = 500,      # How often to dump training state, None = disable.
    resume_pkl          = None,     # Start from the given network snapshot, None = random initialization.
    resume_state_dump   = None,     # Start from the given training state, None = reset training state.
    resume_kimg         = 0,        # Start from the given training progress.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    device              = torch.device('cuda'),
):
    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # Load dataset.
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # subclass of training.dataset.Dataset
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))

    # Construct network.
    dist.print0('Constructing network...')
    interface_kwargs = dict(img_resolution=dataset_obj.resolution, img_channels=dataset_obj.num_channels, label_dim=dataset_obj.label_dim)
    ## net is the backward network: forward sample, train backward net for sampling 
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs) # subclass of torch.nn.Module 
    net.train().requires_grad_(True).to(device) 

    ## fnet is the foward network: backward sample, train forward net for auxiliary 
    fnet = copy.deepcopy(net)

    if dist.get_rank() == 0:
        with torch.no_grad():
            images = torch.zeros([batch_gpu, net.img_channels, net.img_resolution, net.img_resolution], device=device)
            sigma = torch.ones([batch_gpu], device=device)
            labels = torch.zeros([batch_gpu, net.label_dim], device=device)
            misc.print_module_summary(net, [images, sigma, labels], max_nesting=2)

        del images

    # Setup optimizer.
    dist.print0('Setting up optimizer...')
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs) # training.loss.(VP|VE|EDM)Loss
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs) # subclass of torch.optim.Optimizer
    foptimizer = dnnlib.util.construct_class_by_name(params=fnet.parameters(), **optimizer_kwargs) # subclass of torch.optim.Optimizer
    augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs) if augment_kwargs is not None else None # training.augment.AugmentPipe
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device], broadcast_buffers=False)
    fddp = torch.nn.parallel.DistributedDataParallel(fnet, device_ids=[device], broadcast_buffers=False)
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    fema = copy.deepcopy(fnet).eval().requires_grad_(False)
    
    # Resume training from previous snapshot.
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier() # rank 0 goes first
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier() # other ranks follow
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=net, require_all=False)
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=ema, require_all=False)
        del data # conserve memory
        
    # Train.
    dist.print0(f'Training for {total_kimg} kimg...')
    dist.print0()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None

    ddp.eval().requires_grad_(False)
    fddp.eval().requires_grad_(False)


    iterations = -1
    phase = +1 ## +1 for forward sample, train backward; -1 for backward sample train forward

    images, labels = next(dataset_iterator)
    images = images.to(device).to(torch.float32) / 127.5 - 1
    labels = labels.to(device)

    while True:
        iterations += 1
        
        ## sample trajectories for training
        if iterations % 300 == 0: 
            phase *= -1

            if phase == 1: 
                images, labels = next(dataset_iterator)
                images = images.to(device).to(torch.float32) / 127.5 - 1
                labels = labels.to(device)
                images = images + 0.002*torch.randn_like(images)

                with torch.no_grad():
                    all_xs, all_sigmas, all_dts = sample_traj(fema, images, S_noise=0, forward=True, num_steps=200, rho=7)
                    all_xs = torch.stack(all_xs).to('cpu') ## trajectory too big, put on CPU
                    all_sigmas = torch.stack(all_sigmas).to(device)
                    all_dts = torch.stack(all_dts).to(device)

                vis_images = all_xs[80].to(device)
                vis_images = vis_images.detach().permute(0,2,3,1).clamp(-1,1)*0.5 + 0.5
                vis_images = (vis_images * 225.0).clip(0, 255).to(torch.uint8)
                vis_images = grid(torch.cat([vis_images],0).cpu()).squeeze()

                if dist.get_rank() ==0:
                    PIL.Image.fromarray(vis_images, 'RGB').save(os.path.join(run_dir, f'forw-samples-{cur_nimg//1000:06d}.png.png'))

            elif phase == -1:
                images = 80.0*torch.randn_like(images)
                labels = labels.to(device)

                ## backward sample, from noise to image
                with torch.no_grad():
                    all_xs, all_sigmas, all_dts = sample_traj(ema, images, S_noise=0, forward=False, num_steps=200, rho=7)
                    all_xs = torch.stack(all_xs).to('cpu') ## trajectory too big, put on CPU
                    all_sigmas = torch.stack(all_sigmas).to(device)
                    all_dts = torch.stack(all_dts).to(device)

                vis_images = all_xs[-1].to(device)
                vis_images = vis_images.detach().permute(0,2,3,1).clamp(-1,1)*0.5 + 0.5
                vis_images = (vis_images * 225.0).clip(0, 255).to(torch.uint8)
                vis_images = grid(torch.cat([vis_images],0).cpu()).squeeze()

                if dist.get_rank() ==0:
                    PIL.Image.fromarray(vis_images, 'RGB').save(os.path.join(run_dir, f'back-samples-{cur_nimg//1000:06d}.png.png'))

            else:
                raise ValueError('phase must be 1 or -1')

        if phase == 1:
            ## forward sample, traing backward network: ddp
            # Accumulate gradients.
            ddp.train().requires_grad_(True)
            optimizer.zero_grad(set_to_none=True)
            for round_idx in range(num_accumulation_rounds):
                with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):

                    batch_xs_idx = torch.randint(all_xs.shape[1], (images.shape[0], ))
                    batch_ts_idx = torch.randint(all_xs.shape[0], (images.shape[0], ))

                    batch_images = all_xs[batch_ts_idx, batch_xs_idx, :,:,:].to(device)
                    batch_sigmas = all_sigmas[batch_ts_idx]
                    batch_dts = all_dts[batch_ts_idx]/80.0

                    loss = loss_fn.forwnet_loss(optnet=ddp, imptnet=fema, xs=batch_images, ts=batch_sigmas, dts=batch_dts, labels=labels, augment_pipe=augment_pipe)
                    training_stats.report('Loss/backloss', loss)
                    loss.sum().mul(loss_scaling / batch_gpu_total).backward()
                    
                    # NOTE(Weijian): code for visualization, commented out
                    
                    # images = batch_images
                    # images = images.detach().permute(0,2,3,1).clamp(-1,1)*0.5 + 0.5
                    # images = (images * 225.0).clip(0, 255).to(torch.uint8)
                    # images = grid(torch.cat([images],0).cpu()).squeeze()

                    # if dist.get_rank() ==0:
                    #     PIL.Image.fromarray(images, 'RGB').save(os.path.join(run_dir, f'samples-{cur_nimg//1000:06d}.png'))

            ddp.eval().requires_grad_(False)

            # Update weights. 
            for param in net.parameters():
                if param.grad is not None:
                    torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)

            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)

            optimizer.step()

            # Update EMA.
            ema_halflife_nimg = ema_halflife_kimg * 1000
            if ema_rampup_ratio is not None:
                ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
            ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
            for p_ema, p_net in zip(ema.parameters(), net.parameters()):
                p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        elif phase == -1:
            fddp.train().requires_grad_(True)
            foptimizer.zero_grad(set_to_none=True)
            for round_idx in range(num_accumulation_rounds):
                with misc.ddp_sync(fddp, (round_idx == num_accumulation_rounds - 1)):

                    batch_xs_idx = torch.randint(all_xs.shape[1], (images.shape[0], ))
                    batch_ts_idx = torch.randint(all_xs.shape[0], (images.shape[0], ))

                    batch_images = all_xs[batch_ts_idx, batch_xs_idx, :,:,:].to(device)
                    batch_sigmas = all_sigmas[batch_ts_idx]
                    batch_dts = all_dts[batch_ts_idx]/80.0

                    loss = loss_fn.forwnet_loss(optnet=fddp, imptnet=ema, xs=batch_images, ts=batch_sigmas, dts=batch_dts, labels=labels, augment_pipe=augment_pipe)
                    training_stats.report('Loss/forwloss', loss)
                    loss.sum().mul(loss_scaling / batch_gpu_total).backward()

            fddp.eval().requires_grad_(False)

            # Update weights.
            for param in fnet.parameters():
                if param.grad is not None:
                    torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)

            torch.nn.utils.clip_grad_norm_(fnet.parameters(), 2.0)

            foptimizer.step()

            # Update EMA.
            ema_halflife_nimg = ema_halflife_kimg * 1000
            if ema_rampup_ratio is not None:
                ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
            ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
            for p_ema, p_net in zip(fema.parameters(), fnet.parameters()):
                p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        else:
            raise ValueError('phase must be 1 or -1')

        # Perform maintenance tasks once per tick.
        cur_nimg += batch_size
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(' '.join(fields))

        # Check for abort.
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0('Aborting...')

        # Save network snapshot.
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            # data = dict(ema=ema, loss_fn=loss_fn, augment_pipe=augment_pipe, dataset_kwargs=dict(dataset_kwargs))
            data = dict(ema=ema, fema=fema) ## save both forward and backward network
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value # conserve memory
            if dist.get_rank() == 0:
                with open(os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl'), 'wb') as f:
                    pickle.dump(data, f)
            del data # conserve memory

            pass 

        # Update logs.
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
            stats_jsonl.write(json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n')
            stats_jsonl.flush()
        dist.update_progress(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    dist.print0()
    dist.print0('Exiting...')

#----------------------------------------------------------------------------
