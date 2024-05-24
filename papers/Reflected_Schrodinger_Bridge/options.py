import numpy as np
import os
import argparse
import random
import torch
import torch.multiprocessing
import datetime as dt

import configs
import util

from ipdb import set_trace as debug

configs_dict = {
    'gmm':              configs.get_gmm_default_configs,
    'checkerboard':     configs.get_checkerboard_default_configs,
    'spiral':           configs.get_spiral_default_configs,
    'moon':             configs.get_moon_default_configs,
    'moon-to-spiral':   configs.get_moon_to_spiral_default_configs,
    'gaussian-to-gaussian': configs.get_gaussian_to_gaussian_default_configs,
    'cifar10':          configs.get_cifar10_default_configs,
    'celebA64':         configs.get_celebA64_default_configs,
    'celebA32':         configs.get_celebA32_default_configs,
    'mnist':            configs.get_mnist_default_configs,
    'sinusoid':         configs.get_sinusoid_default_configs,
    'sinusoid_large':   configs.get_sinusoid_large_default_configs,
    'pm25':             configs.get_pm25_default_configs,
    'physio':           configs.get_physio_default_configs,
    'inception':        configs.get_inception_default_configs,
}


def _get_default_parser():
    # --------------- basic ---------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem-name",   type=str)
    parser.add_argument("--target-dim-range", nargs='+', default=None, type=int)
    parser.add_argument("--seed",           type=int,   default=0)
    parser.add_argument("--gpu",            type=int,   default=0,        help="GPU device")
    parser.add_argument("--load",           type=str,   default=None,     help="load the checkpoints")
    parser.add_argument("--dir",            type=str,   default=None,     help="directory name to save the experiments under results/")
    parser.add_argument("--group",          type=str,   default='0',      help="father node of directionary for saving checkpoint")
    parser.add_argument("--name",           type=str,   default='debug',  help="son node of directionary for saving checkpoint")
    parser.add_argument("--log-fn",         type=str,   default=None,     help="name of tensorboard logging")
    parser.add_argument("--log-tb",         action="store_true",          help="logging with tensorboard")
    parser.add_argument("--cpu",            action="store_true",          help="use cpu device")
    parser.add_argument("--sinusoid_dataset_path", type=str, default="")
    parser.add_argument('--symbols', type=str, nargs='+', default=None,   help='list of symbols for etf selection.')
    parser.add_argument("--notes", type=str, default="")

    # --------------- SB model ---------------
    parser.add_argument("--t0",             type=float, default=1e-3,     help="time integral start time")
    parser.add_argument("--T",              type=float, default=1.,       help="time integral end time")
    parser.add_argument("--interval",       type=int,   default=100,      help="number of interval")
    parser.add_argument("--sde-type",       type=str,   default='ve', choices=['ve', 'rve', 'vp', 'rvp', 'vp_v2', 'vp_v3', 'simple'])
    parser.add_argument("--sigma-max",      type=float, default=50,       help="max diffusion for VESDE")
    parser.add_argument("--sigma-min",      type=float, default=0.01,     help="min diffusion for VESDE")
    parser.add_argument("--beta-max",       type=float, default=20,       help="max diffusion for VPSDE")
    parser.add_argument("--beta-min",       type=float, default=0.0001,      help="min diffusion for VPSDE")
    parser.add_argument("--simple-var",     type=float, default=1.0,      help="min diffusion for VPSDE")
    parser.add_argument("--prior-sigma",    type=float, default=None,)
    parser.add_argument("--uniform-prior",  action="store_true",          help="Cube uniform prior")
    parser.add_argument("--sampling-interval",  type=int,   default=100,   help="number of interval")

    parser.add_argument("--scale-by-g",     action="store_true", default=False          )
    parser.add_argument("--normalize-loss", action="store_true", default=False          )
    parser.add_argument("--forward-net",    type=str,   help="model class of forward nonlinear drift")
    parser.add_argument("--backward-net",   type=str,   help="model class of backward nonlinear drift")
    parser.add_argument("--output-layer",   type=str,   default='conv1d', help="Transformer related setting.")

    # --------------- SB training & sampling (corrector) ---------------
    parser.add_argument("--train-method",   type=str, default=None,       help="algorithm for training SB" )
    parser.add_argument("--dsm-train-method",type=str, default='dsm_v2',  help="algorithm for training SB" )
    parser.add_argument("--use-arange-t",   action="store_true",          help="[sb alternate train] use full timesteps for training")
    parser.add_argument("--reuse-traj",     action="store_true",          help="[sb alternate train] reuse the trajectory from sampling")
    parser.add_argument("--use-corrector",  action="store_true",          help="[sb alternate train] enable corrector during sampling")
    parser.add_argument("--use-backward-corrector",  action="store_true", help="[sb alternate train] enable corrector during sampling")
    parser.add_argument("--train-bs-x",     type=int,                     help="[sb alternate train] batch size for sampling data")
    parser.add_argument("--train-bs-t",     type=int,                     help="[sb alternate train] batch size for sampling timestep")
    parser.add_argument("--num-stage",      type=int,                     help="[sb alternate train] number of stage")
    parser.add_argument("--num-epoch",      type=int,                     help="[sb alternate train] number of training epoch in each stage")
    parser.add_argument("--num-corrector",  type=int, default=1,          help="[sb alternate train] number of corrector steps")
    parser.add_argument("--snr",            type=float,                   help="[sb alternate train] signal-to-noise ratio")
    parser.add_argument("--eval-itr",       type=int, default=200,        help="[sb joint train] frequency of evaluation")
    parser.add_argument("--samp-bs",        type=int,                     help="[sb train] batch size for all trajectory sampling purposes")
    parser.add_argument("--num-itr",        type=int,                     help="[sb train] number of training iterations (for each epoch)")
    parser.add_argument("--reset-ema-stage",type=int, default=None,       help="[sb train] number of training iterations (for each epoch)")

    parser.add_argument("--DSM-warmup",     action="store_true",  default=False, help="[dsm warmup train] enable dsm warmup at 1st stage")
    parser.add_argument("--backward-warmup-epoch", type=int, default=0,   help="[sb warmup train] epochs")
    parser.add_argument("--rand-mask-miss-ratio", type=float, default=None, help="[dsm warmup train] random mask missing ratio")
    parser.add_argument("--rand-mask-rank", type=int, default=None,       help="[dsm warmup train] low rank random mask")
    parser.add_argument("--train-bs-x-dsm", type=int,                     help="[dsm warmup train] batch size for sampling data")
    parser.add_argument("--train-bs-t-dsm", type=int,                     help="[dsm warmup train] batch size for sampling timestep")
    parser.add_argument("--num-itr-dsm",    type=int,                     help="[dsm warmup train] number of training iterations for DSM warmup")
    parser.add_argument("--dataset-missing-ratio", type=float, default=0.1, help="only used for physio")
    parser.add_argument("--physio-nfold",   type=int, default=0,          help="only used for physio dataset")
    parser.add_argument("--tba-features",   type=str, default=None,       help="only used for tba dataset")


    # --------------- optimizer and loss ---------------
    parser.add_argument("--lr",             type=float, default=5e-4,     help="learning rate")
    parser.add_argument("--lr-f",           type=float, default=None,     help="learning rate for forward network")
    parser.add_argument("--lr-b",           type=float, default=None,     help="learning rate for backward network")
    parser.add_argument("--lr-dsm",         type=float, default=1e-3,     help="learning rate for dsm training")
    parser.add_argument("--lr-gamma",       type=float, default=1.0,      help="learning rate decay ratio")
    parser.add_argument("--lr-step",        type=int,   default=300,      help="learning rate decay step size")
    parser.add_argument("--warmup-lr-step", type=int,   default=0,       help="learning rate warmup step size")
    parser.add_argument("--warmup-multiplier", type=int,   default=20,    help="warmup multiplier")
    parser.add_argument("--l2-norm",        type=float, default=0.0,      help="weight decay rate")
    parser.add_argument("--optimizer",      type=str,   default='AdamW',  help="optmizer")
    parser.add_argument("--grad-clip",      type=float, default=None,     help="clip the gradient")
    parser.add_argument("--noise-type",     type=str,   default='gaussian', choices=['gaussian','rademacher'], help='choose noise type to approximate Trace term')
    parser.add_argument("--num-hutchinson-samp", type=int, default=1,     help="Hutchinson estimator.")
    parser.add_argument("--ema-decay",      type=float, default=0.99,      )

    # ---------------- evaluation ----------------
    parser.add_argument("--FID-freq",       type=int,   default=0,        help="FID frequency w.r.t stages")
    parser.add_argument("--snapshot-freq",  type=int,   default=0,        help="snapshot frequency w.r.t stages")
    parser.add_argument("--ckpt-freq",      type=int,   default=0,        help="checkpoint saving frequency w.r.t stages")
    parser.add_argument("--FID-ckpt",       type=str,   default=None,     help="manually set ckpt path")
    parser.add_argument("--num-FID-sample", type=int,   default=10000,    help="number of sample for computing FID")
    parser.add_argument("--num-eval-sample",type=int,   default=100,       help="number of sample for evaluation")
    parser.add_argument("--full-eval-every-stage",  action="store_true", default=False, help="Run eval on full dataset")
    parser.add_argument("--compute-FID",    action="store_true",          help="flag: evaluate FID")
    parser.add_argument("--compute-NLL",    action="store_true",          help="flag: evaluate NLL")
    parser.add_argument("--permute-batch",  action="store_true", default=False, help="permute (B,C,L,K) to (B,C,K,L)")
    parser.add_argument("--imputation-eval",action="store_true", default=False, help="run imputation evaluation")
    parser.add_argument("--eval-impute-function", type=str,   default='imputation', )
    parser.add_argument("--ckpt-file",      type=str,   default=None,     help="loal ckpt file name (not full path)")
    parser.add_argument("--overwrite-ckpt", action="store_true",          help="Overwirte ckpt to save space")

    return parser


def set():
    parser = _get_default_parser()

    problem_name = parser.parse_args().problem_name
    default_config, model_configs = configs_dict.get(problem_name)()
    parser.set_defaults(**default_config)

    opt = parser.parse_args()

    # ========= seed & torch setup =========
    if opt.seed is not None:
        # https://github.com/pytorch/pytorch/issues/7068
        seed = opt.seed
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.deterministic = True

    # Weird error causes dataloader abnormal.
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # torch.autograd.set_detect_anomaly(True)

    # dataloader issue. Solution found here https://github.com/pytorch/pytorch/issues/11201
    # torch.multiprocessing.set_sharing_strategy('file_system')

    # ========= auto setup & path handle =========
    if opt.gpu is None or opt.cpu:
        opt.device = 'cpu'
    else:
        opt.device='cuda:'+str(opt.gpu)

    for net_name in model_configs.keys():
        if hasattr(model_configs[net_name], 'output_layer'):
            model_configs[net_name].output_layer = opt.output_layer
    opt.model_configs = model_configs
    # if opt.lr is not None:
        # opt.lr_f, opt.lr_b = opt.lr, opt.lr
    opt.lr_f = opt.lr if opt.lr_f is None else opt.lr_f
    opt.lr_b = opt.lr if opt.lr_b is None else opt.lr_b
    assert opt.interval >= opt.sampling_interval

    if opt.compute_NLL or opt.compute_FID:
        opt.DSM_warmup = False
        opt.train_method = None

    if opt.use_arange_t and opt.train_bs_t != opt.interval:
        print('[warning] reset opt.train_bs_t to {} since use_arange_t is enabled'.format(opt.interval))
        opt.train_bs_t = opt.interval

    timestamp = dt.datetime.now().strftime("%m_%d_%Y_%H%M%S")
    if opt.dir is None:
        if problem_name == 'physio':
            p_name = f'{problem_name}_nfold{opt.physio_nfold}_MR{opt.dataset_missing_ratio}'
        else:
            p_name = problem_name
        opt.dir = (f'{p_name}_{opt.forward_net}_{opt.backward_net}_{opt.sde_type}_' +
            f'{opt.train_method}_{opt.dsm_train_method}_{timestamp}_{opt.notes}')

    if not hasattr(opt, 'ckpt_path') or opt.ckpt_path is None:
        # opt.ckpt_path = os.path.join('checkpoint', opt.group, opt.name)
        opt.ckpt_path = os.path.join('results', opt.dir)
    os.makedirs(opt.ckpt_path, exist_ok=True)

    if opt.log_fn is None:
        opt.log_fn = opt.ckpt_path

    if opt.snapshot_freq and util.is_image_dataset(opt):
        # opt.eval_path = os.path.join('results', opt.dir)
        os.makedirs(os.path.join(opt.ckpt_path, 'forward'), exist_ok=True)
        os.makedirs(os.path.join(opt.ckpt_path, 'backward'), exist_ok=True)
    os.makedirs(os.path.join(opt.ckpt_path, 'backward'), exist_ok=True)

    if (opt.FID_freq and util.exist_FID_ckpt(opt)) or util.is_toy_dataset(opt):
        opt.generated_data_path = os.path.join(
            'results', opt.dir, 'backward', 'generated_data')
        os.makedirs(opt.generated_data_path, exist_ok=True)
    # util.check_duplication(opt)

    # ========= auto assert & (kind) warning =========
    # if opt.forward_net=='ncsnpp' or opt.backward_net=='ncsnpp':
    #     if model_configs['ncsnpp'].training.continuous==False:
    #         assert opt.interval==201

    # if opt.DSM_warmup:
    #     assert opt.train_method == 'alternate'

    # if opt.load is not None:
    #     assert not opt.DSM_warmup, 'Already load some models, no need to DSM-warm-up!'

    if opt.train_method is not None:
        if opt.num_FID_sample>10000:
            print(util.green("[warning] you are in the training phase, are you sure you want to have large number FID evaluation?"))
        if opt.snapshot_freq<1:
            print(util.green("[warning] you are in the training phase, are you sure you do not want to have snapshot?"))

    if not opt.reuse_traj:
        print(util.green("[warning] double check that you do not want to reuse FID evaluation trajectory for training!!!"))

    # ========= print options =========
    for o in vars(opt):
        print(util.green(o), ":", util.yellow(getattr(opt,o)), end=' ')
    print()

    return opt


####################################################################################################
#                                   args for jupyter notebook                                      #
####################################################################################################
def get_default_args(problem_name, return_model_configs=False):
    parser = _get_default_parser()

    default_config, model_configs = configs_dict.get(problem_name)()
    parser.set_defaults(**default_config)

    opt = parser.parse_args('')
    opt.model_configs = model_configs

    if return_model_configs:
        return opt, model_configs
    else:
        return opt


def post_process_args(opt, model_configs=None):
    # ========= seed & torch setup =========
    if opt.seed is not None:
        # https://github.com/pytorch/pytorch/issues/7068
        seed = opt.seed
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.deterministic = True

    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # torch.autograd.set_detect_anomaly(True)
    
    # ========= auto setup & path handle =========
    if opt.gpu is None:
        opt.device = 'cpu'
    else:
        opt.device='cuda:'+str(opt.gpu)

    for net_name in opt.model_configs.keys():
        if hasattr(opt.model_configs[net_name], 'output_layer'):
            opt.model_configs[net_name].output_layer = opt.output_layer
    opt.model_configs = model_configs

    # if opt.lr is not None:
    #     opt.lr_f, opt.lr_b = opt.lr, opt.lr
    opt.lr_f = opt.lr if opt.lr_f is None else opt.lr_f
    opt.lr_b = opt.lr if opt.lr_b is None else opt.lr_b

    if opt.compute_NLL or opt.compute_FID:
        opt.DSM_warmup = False
        opt.train_method = None

    if opt.use_arange_t and opt.train_bs_t != opt.interval:
        print('[warning] reset opt.train_bs_t to {} since use_arange_t is enabled'.format(opt.interval))
        opt.train_bs_t = opt.interval

    timestamp = dt.datetime.now().strftime("%m_%d_%Y_%H%M%S")
    if opt.dir is None:
        if problem_name == 'physio':
            p_name = f'{problem_name}_nfold{opt.physio_nfold}_MR{opt.dataset_missing_ratio}'
        else:
            p_name = problem_name
        opt.dir = (f'{p_name}_{opt.forward_net}_{opt.backward_net}_{opt.sde_type}_' +
            f'{opt.train_method}_{opt.dsm_train_method}_{timestamp}_{opt.notes}')

    if not hasattr(opt, 'ckpt_path') or opt.ckpt_path is None:
        # opt.ckpt_path = os.path.join('checkpoint', opt.group, opt.name)
        opt.ckpt_path = os.path.join('results', opt.dir)
    os.makedirs(opt.ckpt_path, exist_ok=True)

    if opt.snapshot_freq and util.is_image_dataset(opt):
        # opt.eval_path = os.path.join('results', opt.dir)
        os.makedirs(os.path.join(opt.ckpt_path, 'forward'), exist_ok=True)
        os.makedirs(os.path.join(opt.ckpt_path, 'backward'), exist_ok=True)
    os.makedirs(os.path.join(opt.ckpt_path, 'backward'), exist_ok=True)

    if (opt.FID_freq and util.exist_FID_ckpt(opt)) or util.is_toy_dataset(opt):
        opt.generated_data_path = os.path.join(
            'results', opt.dir, 'backward', 'generated_data'
        )
        os.makedirs(opt.generated_data_path, exist_ok=True)
    # util.check_duplication(opt)

    # ========= auto assert & (kind) warning =========
    # if opt.forward_net=='ncsnpp' or opt.backward_net=='ncsnpp':
    #     if model_configs['ncsnpp'].training.continuous==False:
    #         assert opt.interval==201

    # if opt.DSM_warmup:
    #     assert opt.train_method == 'alternate'

    # if opt.load is not None:
    #     assert not opt.DSM_warmup, 'Already load some models, no need to DSM-warm-up!'

    if opt.train_method is not None:
        if opt.num_FID_sample>10000:
            print(util.green("[warning] you are in the training phase, are you sure you want to have large number FID evaluation?"))
        if opt.snapshot_freq<1:
            print(util.green("[warning] you are in the training phase, are you sure you do not want to have snapshot?"))

    if not opt.reuse_traj:
        print(util.green("[warning] double check that you do not want to reuse FID evaluation trajectory for training!!!"))

    # ========= print options =========
    for o in vars(opt):
        print(util.green(o),":",util.yellow(getattr(opt,o)), end=' ')
    print()

    return opt

