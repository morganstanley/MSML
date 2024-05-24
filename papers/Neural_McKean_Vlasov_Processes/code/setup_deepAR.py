import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import yaml
from scipy.io import loadmat


from deepAR import *
from sim_process import *
from get_data import *
from transformer import *


def setup_deepAR(yaml_filepath, device, seed):
    with open(yaml_filepath, 'r') as f:
        cfg = yaml.load(f, yaml.SafeLoader)

    dataset_params = cfg['dataset']
    cfg['dataset']['seed'] = seed
    if cfg['deepAR'] is not None:
        deepAR_params  = cfg['deepAR']
    else:
        deepAR_params = cfg["transformer"]
    optim_params   = cfg['optimizer']
    
    deepAR_param = deepAR_params['net']
    deepAR_lr    = deepAR_params['lr']
    window_size  = deepAR_params['window_size']
    if cfg['deepAR'] is not None:
        deepAR = deepAR_net(**deepAR_param).to(device)
    else:
        deepAR = TransformerModel(**deepAR_param).to(device)
    
    optimizer_path = getattr(optim, optim_params['name'])(deepAR.parameters(), lr = deepAR_lr, eps=1e-8)
    dataset_params['n_particles'] = cfg['other']['test_n_particles']
    
    # load data
    train_path_loader, val_path_loader, test_path_loader, dt, train_split_t, xs_val, xs_test = load_data(cfg)
    
    deepAR_num_param = sum(p.numel() for p in deepAR.parameters())
    print("Total Number of Parameters: {}".format(deepAR_num_param))
    
    initialized = {
        'deepAR'           : deepAR,
        'path_loader'      : train_path_loader,
        'validation_loader': val_path_loader,
        'dt'               : dt,
        'optimizer_path'   : optimizer_path,
        'data_param'       : dataset_params,
        'train_y_samp_size': cfg['other']['train_y_samp_size'],
        'test_y_samp_size' : cfg['other']['test_y_samp_size'],
        'train_split_t'    : train_split_t,
        'n_epochs'         : cfg['optimizer']['n_epochs'],
        'loss_type'        : cfg['other']['loss_type'],
        'device'           : device,
        'window_size'      : window_size,
        'xs_val'           : xs_val
    }
    return initialized, test_path_loader, xs_test
