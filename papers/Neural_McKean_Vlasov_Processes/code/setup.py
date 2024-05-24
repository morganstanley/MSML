import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import yaml
from scipy.io import loadmat

from MeanFieldMLP import *
from MLP import *
from Glow import *
from sim_process import *
from get_data import *
from utils import make_particle_label


def setup(yaml_filepath, device, seed, FK=False):
    with open(yaml_filepath, 'r') as f:
        cfg = yaml.load(f, yaml.SafeLoader)

    dataset_params = copy.deepcopy(cfg['dataset'])
    cfg['dataset']['seed'] = seed
    NF_params      = cfg['NF']
    MLP_params     = (cfg['MF'] if cfg['MF'] else cfg['MLP'])
    loader_params  = cfg['loader']
    optim_params   = cfg['optimizer']
    
    # load data
    if "extrapolate" in yaml_filepath:
        extrapolate = True
        grid_init  = False
        grid_space = 0
    else:
        extrapolate = False
        grid_space = (6 if 'opinion' in yaml_filepath else 0)
        grid_init = (True if 'opinion' in yaml_filepath else False)
    if "generative" in yaml_filepath:
        generative = True
    else:
        generative = False
        
    if "non_stoch_eval" in yaml_filepath:
        cfg['dataset']['non_stoch_eval'] = True
        dataset_params["non_stoch_eval"] = True
        
    dataset_params["grid_init"]  = grid_init
    dataset_params['fore_range'] = None
    dataset_params["grid_space"] = grid_space
    
    if "noise" in yaml_filepath:
        add_noise = True
        match = yaml_filepath.rfind("=")
        noise_level = float(yaml_filepath[match+1:match+3])
    else:
        add_noise = False
        noise_level = 0
        
   
    point_loader, train_path_loader, val_path_loader, test_path_loader, dt, train_split_t, irreg_t = load_data(cfg,
                                                                                                      extrapolate=extrapolate,
                                                                                                      generative=generative,
                                                                                                      add_noise = add_noise,
                                                                                                      noise_level=noise_level,
                                                                                                      FK=FK)
    if FK == False and val_path_loader is not None:
        x0 = np.concatenate([train_path_loader.dataset.tensors[0].detach().cpu().numpy()[:,0,:],
                        val_path_loader.dataset.tensors[0].detach().cpu().numpy()[:,0,:],
                        test_path_loader.dataset.tensors[0].detach().cpu().numpy()[:,0,:]],0)
    try:
        partition = dataset_params["partition"]
        hetero_class = max(make_particle_label(x0=x0, partition=dataset_params["partition"]))+1
    except KeyError:
        hetero_class = 1
    if "extrapolate" in yaml_filepath:
        dataset_params["init_var"] = [5,5]
    dataset_params['n_particles'] = cfg['other']['test_n_particles']
    
    if cfg['NF']:
        batch_size_points = cfg['loader']['batch_size_points']
        NF_param = cfg['NF']['net']
        NF_lr = cfg['NF']['lr']
        num_blocks = cfg['NF']['num_blocks']
        num_inputs = NF_param['num_inputs']
        
        nf_modules = []
        mask = torch.arange(0, NF_param['num_inputs']) % 2
        mask = mask.to(device).float()
        for _ in range(num_blocks):
            nf_modules += [
                BatchNormFlow(NF_param['num_inputs']),
                LUInvertibleMM(NF_param['num_inputs']),
                CouplingLayer(**NF_param, mask=mask)
            ]
            mask = 1 - mask

        NF = FlowSequential(*nf_modules).to(device)
        NF.num_inputs = num_inputs
        for module in NF.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.data.fill_(0)
                    
        optimizer_point = getattr(optim, optim_params['name'])(NF.parameters(), lr = NF_lr)
    else: 
        NF=optimizer_point=None

    driftMLP_param = MLP_params['net']
    
    driftMLP_param["num_cond_input"] = 0
    driftMLP_lr = MLP_params['lr']
    if "simulation" in yaml_filepath or "noise" in yaml_filepath or "extrap" in yaml_filepath:
        driftMLP_param["grad_sigma"] = False
        
        
    if cfg['MF']:
        if 'num_inputs' in driftMLP_param.keys():
            driftMLP_param['num_f_inputs']  = driftMLP_param['num_inputs']
            driftMLP_param['num_g_inputs']  = driftMLP_param['num_inputs']
            driftMLP_param['num_f_outputs'] = driftMLP_param['num_inputs']
            driftMLP_param['num_g_outputs'] = driftMLP_param['num_inputs']
        if "simulation_ablation" in yaml_filepath:
            cfg['optimizer']['n_epochs'] = 500
            driftMLP_param['g_num_hidden']  = 128
            driftMLP_param['f_num_hidden']  = 128
        if "EEG" in yaml_filepath or "Chemo" in yaml_filepath or "fitz" in yaml_filepath or "generative" in yaml_filepath:
            driftMLP_param['num_cond_inputs']  = 1
        else:
            driftMLP_param['num_cond_inputs']  = 0
            
        driftMLP_param['hetero_class'] = hetero_class
        driftMLP = MF(**driftMLP_param).to(device)
    elif cfg['MLP']:
        if "EEG" in yaml_filepath or "Chemo" in yaml_filepath or "fitz" in yaml_filepath or "generative" in yaml_filepath:
            driftMLP_param['num_cond_inputs']  = 1
        else:
            driftMLP_param['num_cond_inputs']  = 0
        driftMLP = MLP(**driftMLP_param).to(device)
    
    optimizer_path = getattr(optim, optim_params['name'])(driftMLP.parameters(), lr = driftMLP_lr, eps=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer_path, gamma=0.9998)

    driftMLP_param = 0
    if cfg['NF']:
        driftMLP_param = sum(p.numel() for p in NF.parameters())
    driftMLP_param += sum(p.numel() for p in driftMLP.parameters())
    print("Total Number of Parameters: {}".format(driftMLP_param))
    try:
        opt_experiment = cfg['other']['opt_experiment']
    except KeyError:
        opt_experiment = False
        pass
    
    try:
        brownian = cfg['other']['brownian']
    except KeyError:
        brownian = False
        pass
    
    try:
        bb_n = cfg['other']['bb_n']
        bb_m = cfg['other']['bb_m']
    except KeyError:
        bb_n = None
        bb_m = None
        if brownian:
            bb_n = 10
            bb_m = 5
        pass
    
    initialized = {
        'NF'               : NF,
        'driftMLP'         : driftMLP,
        'point_loader'     : point_loader, 
        'path_loader'      : train_path_loader,
        'validation_loader': val_path_loader,
        'dt'               : dt,
        'optimizer_point'  : optimizer_point, 
        'optimizer_path'   : optimizer_path,
        'data_param'       : dataset_params,
        'train_y_samp_size': cfg['other']['train_y_samp_size'],
        'test_y_samp_size' : cfg['other']['test_y_samp_size'],
        'brownian'         : brownian,
        'bb_n'             : bb_n,
        'bb_m'             : bb_m,
        'train_split_t'    : train_split_t,
        'n_epochs'         : cfg['optimizer']['n_epochs'],
        'loss_type'        : cfg['other']['loss_type'],
        'device'           : device,
        'extrapolate'      : extrapolate,
        'generative'       : generative,
        'scheduler'        : scheduler,
        'irreg_t'          : irreg_t,
    }
    print(dataset_params)
    return initialized, test_path_loader, cfg['head']
