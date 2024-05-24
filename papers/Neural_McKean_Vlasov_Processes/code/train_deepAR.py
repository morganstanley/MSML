import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F

import yaml
import pickle
import copy

from tqdm.auto import tqdm

from train import format_directory, make_directory, get_parser
from plot import plot_particle, plot_drift
from sim_process import *
from deepAR import *
from utils import *
from setup_deepAR import *

import os

def train_deepAR(deepAR,
                 optimizer_path, 
                 path_loader, 
                 validation_loader,
                 dt,
                 data_param, 
                 train_y_samp_size,
                 test_y_samp_size,
                 train_split_t,
                 n_epochs, 
                 loss_type, 
                 device,
                 window_size,
                 xs_val,
                ):
    deepAR_best         = None
    deepAR_loss_list    = []
    MSE_list            = []
    MAE_list            = []
    CRPS_list           = []
    MSE_list_fore       = []
    MAE_list_fore       = []
    CRPS_list_fore      = []
    MSE_list_cond_fore  = []
    MAE_list_cond_fore  = []
    CRPS_list_cond_fore = []
    val_loss_list       = []
    
    pbar = tqdm(total=n_epochs)
    
    drift_MSE   = 0
    drift_MAE   = 0
    CRPS        = 0
    deepAR_loss = 0
    dt          = torch.tensor(dt).float().to(device)
    best_epoch  = 0
    
    for i in range(n_epochs):
        train_loss = 0
        path_loss_tot = 0

        deepAR.train()
        g_loss = nn.GaussianNLLLoss(reduction='mean')
        
        for batch_idx, data in enumerate(path_loader):
            hidden=None
            optimizer_path.zero_grad() 
            
            xs = data[0].float().to(device)
            xt = data[1].float().to(device)
            t  = None
            xt_hat, sig_hat, hidden = deepAR(xs, t, hidden)
            target = torch.cat([xs[:,-1,:].unsqueeze(1), xt.unsqueeze(1)], 1) if xt_hat.shape != xt.shape else xt
            loss = g_loss(input=xt_hat, target=target, var=(sig_hat**2))
            
            loss.backward()
            optimizer_path.step()
            train_loss += loss.item()
            
            pbar.set_description("Current Epoch-batch: {}-{} | deepAR Loss: {} | MSE: {} | CRPS: {}".format(
                i+1, 
                batch_idx+1,
                round(train_loss/(batch_idx + 1), 2),
                round(drift_MSE, 2),
                round(CRPS,2)
            ))
            
        deepAR_loss = train_loss/(batch_idx + 1)
            
        # Validate
        if (i+1)%50 == 0:
            deepAR.eval()
            
            val_out = validate_deepAR(deepAR, 
                                      data_param, 
                                      device,
                                      xs_val,
                                      validation_loader = validation_loader,
                                      test_y_samp_size=test_y_samp_size,
                                      train_split_t = train_split_t,
                                      window_size=window_size)
            
            
            drift_test, drift_MLP, drift_fore, val_loss, stats, stats_fore, stats_cond_fore = val_out
            
            drift_MSE = stats["MSE"]
            drift_MAE = stats["MAE"]
            if drift_test.shape[-1] == 1:
                CRPS  = stats["CRPS"]
            if (i+1)%int(n_epochs/5) == 0:
                plot_drift(dt.detach().cpu().numpy(), drift_test, drift_MLP, 
                           i, plot_savepath, data_param["simulation"], 
                           train_split_t=train_split_t, drift_fore = drift_fore, train=False)

                plot_particle(dt.detach().cpu().numpy(), drift_test, drift_MLP, 
                           i, plot_savepath, data_param["simulation"], train=False, 
                              plot_particles=5, train_split_t=train_split_t, drift_fore = drift_fore)
            
            pbar.set_description("Curr E-b: {}-{} | deepAR Loss: {} | Path Loss: {} | MSE: {} | MAE: {} | CRPS: {}".format(
                    i+1, 
                    batch_idx+1,
                    round(deepAR_loss, 2),
                    round(val_loss, 2),
                    round(drift_MSE, 2),
                    round(drift_MAE, 2),
                    round(CRPS, 2)
            ))
            if len(val_loss_list) > 0:
                if np.min(val_loss_list) > val_loss:
                        #plot_drift(dt.detach().cpu().numpy(), drift_test, drift_MLP, 
                        #       i, plot_savepath, data_param["simulation"],
                        #       train_split_t=train_split_t, drift_fore = drift_fore)

                        #plot_particle(dt.detach().cpu().numpy(), drift_test, drift_MLP, 
                        #           i, plot_savepath, data_param["simulation"], train=False, 
                        #              plot_particles=5, train_split_t=train_split_t, drift_fore = drift_fore)
                    best_epoch = i + 1
                    torch.save(deepAR.state_dict(), net_savepath + "/deepAR.pt")
            else:
                torch.save(deepAR.state_dict(), net_savepath + "/deepAR.pt")
                
            val_loss_list.append(val_loss)
            MSE_list.append(stats["MSE"])
            MAE_list.append(stats["MAE"])
            if drift_test.shape[-1] == 1:
                CRPS_list.append(stats["CRPS"])
            if train_split_t:
                MSE_list_fore.append(stats_fore["MSE"])
                MAE_list_fore.append(stats_fore["MAE"])
                if drift_test.shape[-1] == 1:
                    CRPS_list_fore.append(stats_fore["CRPS"])
                    
                MSE_list_cond_fore.append(stats_cond_fore["MSE"])
                MAE_list_cond_fore.append(stats_cond_fore["MAE"])
                if drift_test.shape[-1] == 1:
                    CRPS_list_cond_fore.append(stats_cond_fore["CRPS"])

        
        deepAR_loss_list.append(deepAR_loss)
        pbar.update(1)
    #torch.save(deepAR.state_dict(), net_savepath + "/deepAR.pt")
    stats = {"deepAR":deepAR_loss_list,
             "MSE":MSE_list, "MAE": MAE_list, 
             "val_loss": val_loss_list, "CRPS":CRPS_list, 'best_epoch':best_epoch}
    return stats, best_epoch

def gen_path_deepAR(deepAR, device, validation_loader, window_size, hidden=None):
    temp = validation_loader.dataset.tensors[0]
    path = torch.zeros(validation_loader.batch_size, int(temp.shape[0]/validation_loader.batch_size + 1), temp.shape[-1])
    start = 0
    end = start + window_size
    hidden=None
    for i, data in enumerate(validation_loader):
        if i == 0:
            xt = data[0].float().to(device)
            path[...,i,:] = xt[...,i,:]
        else:
            # Note here: We don't use the windowed data loader
            t = None
            x_hat, sig_hat, _ = deepAR(xt, t, hidden)
            x_gen = torch.normal(x_hat, sig_hat)
            # Instead we use the generated step to feed into RNN
            path[...,i,:] = x_gen
            xt = path[..., max(start,i-window_size+1):max(end,i+1),:].to(device)
            
    x_hat, sig_hat, hidden = deepAR(xt, t, hidden)
    x_gen = torch.normal(x_hat, sig_hat)
    # Instead we use the generated step to feed into RNN
    path[...,-1,:] = x_gen
    return path.detach().cpu().numpy()



def cond_fore_deepAR(deepAR, device, validation_loader, window_size, train_split_t, hidden=None):
    temp = validation_loader.dataset.tensors[0]
    path = torch.zeros(validation_loader.batch_size, 
                       int(temp.shape[0]/validation_loader.batch_size + 1) - (train_split_t-1), 
                       temp.shape[-1])
    start = 0
    end = start + window_size
    hidden=None
    for i, data in enumerate(validation_loader):
        # if we looped till train_split_t-1 or after
        if i >= train_split_t-1:
            if i == train_split_t-1:
                # at train_split_t-1, assign the first window value to path
                # Also clear everything after as we are only conditioning on one point.
                xt = data[0].float()
                path[...,i-(train_split_t-1),:] = xt[...,-1,:]
                xt_temp = torch.zeros(xt.shape)
                xt_temp[...,0,:] = xt[...,-1,:]
                xt = xt_temp.to(device)
            else:
                # We no longer need windowed data
                t = None
                x_hat, sig_hat, _ = deepAR(xt, t, hidden)
                x_gen = torch.normal(x_hat, sig_hat)
                # Instead we use the generated step to feed into deepAR
                path[...,i-(train_split_t-1),:] = x_gen
                xt = path[..., max(start,i-(train_split_t - 1)-window_size+1):max(end,i-(train_split_t - 1)+1),:].to(device)
        else: 
            continue
            
    x_hat, sig_hat, hidden = deepAR(xt, t, hidden)
    x_gen = torch.normal(x_hat, sig_hat)
    # Instead we use the generated step to feed into deepAR
    path[...,-1,:] = x_gen
    return path.detach().cpu().numpy()



def validate_deepAR(deepAR, 
                    data_param,
                    device,
                    xs_val,
                    validation_loader=None,
                    window_size=20,
                    train_split_t=None,
                    **params):
    test_y_samp_size = params.setdefault('test_y_samp_size', 128)
    
    # Todo: deepAR for simulation
    # There seems to have no way to directly compare the simulation result
    # Because we do not predict drift through deepAR
    if data_param['simulation']:
        raise NotImplementedError("Not comparable for simulation")
    
    else:
        # Path Generation
        gen_path  = gen_path_deepAR(deepAR = deepAR, 
                                    device = device, 
                                    validation_loader = validation_loader, 
                                    window_size=window_size)
        
        if train_split_t:
            gen_path_cond_fore = cond_fore_deepAR(deepAR = deepAR, 
                                                  device = device, 
                                                  validation_loader = validation_loader, 
                                                  window_size=window_size, 
                                                  train_split_t=train_split_t)
            
        val_loss = 0
        g_loss = nn.GaussianNLLLoss(reduction="mean")
        temp = validation_loader.dataset.tensors[0]
        path = torch.zeros(validation_loader.batch_size, int(train_split_t), temp.shape[-1])
        start = 0
        end = start + window_size
        hidden = None
        for i, data in enumerate(validation_loader):
            # xt are labels
            xs = data[0].float().to(device)
            xt = data[1].float().to(device)
            if i == train_split_t:
                break
            #if i == 0:
            #    xs = data[0].float().to(device)
            #    path[...,i,:] = xs[..., i,:]
            #else:
                # Similar Process to get generation
            t = None#data[2].float().cuda()
            xt_hat, sig_hat, hidden = deepAR(xs, t, hidden)
            val_loss += g_loss(input=xt_hat, target=xt, var=(sig_hat**2))
            # Instead we use the generated step to feed into RNN
            #path[...,i,:] = xt_hat
            #xs = path[..., max(start,i-window_size+1):max(end,i+1),:].to(device)
                
        stats = evaluation(test=xs_val, gen=gen_path)
        stats_fore = evaluation(test=xs_val[..., train_split_t-1:, :],
                                gen =gen_path[..., train_split_t-1:, :]) if train_split_t else None
        stats_cond_fore = evaluation(test=xs_val[..., train_split_t-1:, :],
                                     gen =gen_path_cond_fore) if train_split_t else None
        
        return xs_val, gen_path, gen_path_cond_fore, val_loss.item(), stats, stats_fore, stats_cond_fore
    
    
if __name__ == "__main__":
    import shutil

    args = get_parser().parse_args()

    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else: 
        device = torch.device(args.device)

    yaml_filepath = args.filename
    with open(yaml_filepath, 'r') as f:
        cfg = yaml.load(f, yaml.SafeLoader)
        
    experiment_folder = args.experiment
        

    all_stats = {'config':cfg, 'runs':[]}

    try:
        n_runs = cfg['n_runs']
    except KeyError:
        n_runs = 5
    try:
        n_tries = cfg['n_tries']
    except KeyError:
        n_tries = 1

    for run in range(0,n_runs):
        global savepath
        global plot_savepath
        global net_savepath
        savepath, plot_savepath, net_savepath = format_directory(experiment_folder, cfg, run)
        make_directory(savepath, plot_savepath, net_savepath)
        initialized, test_loader, xs_test = setup_deepAR(yaml_filepath, device, seed=run)
        all_stats, best_epoch = train_deepAR(**initialized)
        
        deepAR_best = initialized["deepAR"]
        deepAR_best.load_state_dict(torch.load(net_savepath + "/deepAR.pt"))
        deepAR_best.eval()
        
        test_stats = validate_deepAR(deepAR_best, initialized["data_param"], 
                              device, xs_test, test_loader, initialized["window_size"],
                              train_split_t=initialized["train_split_t"])
        test_stats = test_stats + (best_epoch,)
        
        with open(os.path.join(savepath,'test_stats.pkl'), 'wb') as f:
            pickle.dump(test_stats, f)
            f.close()
        with open(os.path.join(savepath,'saved_stats.pkl'), 'wb') as f:
            pickle.dump(all_stats, f)
            f.close()
    

                
        
    
    
    