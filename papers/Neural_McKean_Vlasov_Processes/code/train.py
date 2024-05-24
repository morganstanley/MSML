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

from MeanFieldMLP import *
from MLP import *
from Glow import *
from sim_process import *
from utils import *
from setup import *
from plot import *

"""
TO-DO:
Add Labeling to NF-train. 
"""


import os
torch.cuda.empty_cache()


def train(NF, driftMLP, 
          optimizer_point, 
          optimizer_path, 
          path_loader, 
          validation_loader,
          point_loader,
          dt,
          data_param, 
          train_y_samp_size,
          test_y_samp_size,
          train_split_t,
          n_epochs, 
          loss_type, 
          device,
          extrapolate,
          generative,
          scheduler,
          irreg_t,
          brownian,
          bb_n,
          bb_m,
         ):
    
    global savepath
    global plot_savepath
    global net_savepath
    
    NF_loss_list        = []
    MLP_loss_list       = []
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
    ed_list             = []
    
    pbar = tqdm(total=n_epochs)
    
    drift_MSE  = 0
    drift_MAE  = 0
    CRPS       = 0
    NF_loss    = 0
    MLP_loss   = 0
    dt         = torch.tensor(dt).float().to(device)
    best_epoch = 0
    if irreg_t is not None:
        irreg_t_set = set(irreg_t[:-1] + [0])

    for i in range(n_epochs):
        train_loss = 0
        path_loss_tot = 0

        driftMLP.train()
        if NF:
            NF.train()
            for batch_idx, data in enumerate(point_loader):
                x = data[0].float().to(device)
                t = data[1].float().to(device)
                    
                optimizer_point.zero_grad()
                loss = (-NF.log_probs(x, t.reshape(len(x),1))).mean()
                loss.backward()
                train_loss += loss.item()
                optimizer_point.step()
                
                pbar.set_description("Current Epoch-batch: {}-{} | NF Loss: {} | Path Loss: {} | MSE: {} | CRPS: {}".format(
                    i+1, 
                    batch_idx+1,
                    round(train_loss/(batch_idx + 1), 2),
                    round(MLP_loss,2),
                    round(drift_MSE, 2),
                    round(CRPS,2)
                ))
            NF_loss = train_loss/(batch_idx + 1)
            
        g_loss = nn.GaussianNLLLoss(reduction="mean")
        
        for batch_idx, data in enumerate(path_loader):
            path_loss = 0.0
            optimizer_path.zero_grad()
            x = data[0].float().to(device)
            t = data[1].float().to(device)
            
            try:
                label_x = (data[2].float().to(device) if driftMLP.label_x else None)
                label_y = (data[2].float().to(device) if driftMLP.label_y else None)
            except IndexError:
                pass

            irreg_t_count = -1
            for t_idx in range(t.shape[1]-1):
                # if irreg_t is None, do normal time propogation with step 1
                if irreg_t is None:
                    x_s = x[:, t_idx, :]
                    x_t = x[:, t_idx+1, :]
                    dt_train = dt
                elif irreg_t is not None and t_idx not in irreg_t_set:
                    # irreg_t is not None, but t_idx is not at one of the steps, continue.
                    continue
                elif irreg_t is not None and t_idx in irreg_t_set:
                    # In case when irregular time step and t_idx is at one of the steps
                    irreg_t_count += 1
                    x_s = x[:, t_idx, :]
                    x_t = x[:, irreg_t[irreg_t_count], :]
                    dt_train = t[0, irreg_t[irreg_t_count]] - t[0, t_idx]
                
                # Different y_t for different architectures
                y_t = None
                if NF is not None:
                    y_t = NF.sample(train_y_samp_size, 
                                    cond_inputs = t[0,t_idx].repeat(train_y_samp_size,1)).detach()
                elif driftMLP.W_0_hidden == 0 and NF is None:
                    y_t = x_s
                
                # Normal Training
                if brownian == False:
                    drift = driftMLP(x_s, y_t, t[0,t_idx], label_x, label_y)
                    sigma = driftMLP.sigma_forward(t[0,t_idx])
                    sigma = torch.clip(sigma, min=1e-3)
                    if loss_type == "girsanov":
                        path_loss += -(drift*(x_t-x_s)/(sigma**2) - \
                                      0.5*(drift**2)*dt_train/(sigma**2)).mean()
                    else:
                        path_loss += g_loss(target=x_t-x_s, input=drift*dt_train,
                                            var=sigma**2*dt.repeat(*(x_t.shape)))
                else:
                    # Brownian Bridge Interpolation Scheme
                    BB, BBtj = brownian_bridge_nd(x_s.detach().cpu(), x_t.detach().cpu(), t[0, t_idx].detach().cpu(), 
                                                  t[0, t_idx].detach().cpu()+dt_train.detach().cpu(), n=bb_n, m=bb_m, 
                                                  sigma=driftMLP.sigma_forward)
                    dt_train_bb = (BBtj[1] - BBtj[0]).to(device)
                    for BB_index in range(BB.shape[1]-1):
                        BB_x_s = BB[:, BB_index, :].to(device)
                        BB_x_t = BB[:, BB_index+1, :].to(device)
                        BB_t_t = BBtj[BB_index].to(device)

                        drift = driftMLP(BB_x_s, BB_x_s, BB_t_t, 
                                         label_x.repeat_interleave(bb_m,0) if label_x is not None else None,
                                         label_y.repeat_interleave(bb_m,0) if label_y is not None else None)
                        
                       
                        sigma = driftMLP.sigma_forward(BB_t_t)
                        sigma = torch.clip(sigma, min=1e-3)
                        if loss_type == "girsanov":
                            path_loss += -(drift*(BB_x_t-BB_x_s)/sigma**2 - \
                                          0.5*drift**2*dt_train_bb/sigma**2).mean()
                        else:
                            path_loss += g_loss(target=BB_x_t-BB_x_s, input=drift*dt_train_bb,
                                                var=sigma**2*dt_train_bb.repeat(*(BB_x_t.shape)))
                    
            path_loss.backward()
            optimizer_path.step() 
            path_loss_tot += path_loss.item()
            
            pbar.set_description("Curr E-b:{}-{} | NF Loss:{} | Path Loss:{} | MSE:{} | MAE:{} | CRPS:{} | sigma:{}".format(
                i+1, 
                batch_idx+1,
                round(NF_loss, 2),
                round(path_loss_tot/(batch_idx + 1), 2),
                round(drift_MSE, 2),
                round(drift_MAE, 2),
                round(CRPS, 2),
                np.round(sigma.detach().cpu().numpy(), 2)
            ))
            MLP_loss = path_loss_tot/(batch_idx + 1)
        scheduler.step()
        
        # Validation and Plotting
        if (i+1)%5 == 0:
            if NF:
                NF.eval()
            driftMLP.eval()
            train_particle_labels=None
            try:
                train_particle_labels = path_loader.dataset.tensors[2].detach()
            except IndexError:
                pass
            val_out = validate(driftMLP, 
                               NF, 
                               data_param, 
                               device,
                               loss_type,
                               val_path_loader = validation_loader,
                               test_y_samp_size=test_y_samp_size,
                               extrapolate=extrapolate,
                               generative=generative,
                               train_split_t = train_split_t,
                               train_paths = path_loader.dataset.tensors[0].detach(),
                               train_particle_labels=train_particle_labels)
            
            if data_param["simulation"]:
                gen_path_simu, xs_val, drift_test, drift_MLP, drift_fore, val_loss, stats, stats_fore, stats_cond_fore = val_out
                ed = stats["energy_distance"]
                ed_list.append(ed)
                if generative and (i+1)%int(n_epochs/5) == 0:
                    plot_gen_scatter(gen_path_simu,xs_val, i, plot_savepath, False)
                elif generative==False and (i+1)%int(n_epochs/5) == 0:
                    plot_particle(dt.detach().cpu().numpy(), xs_val, gen_path_simu,
                                  i, plot_savepath, data_param["simulation"], paths=True, train=False,
                                  plot_particles=xs_val.shape[0], train_split_t=train_split_t, 
                                  drift_fore = drift_fore, irreg_t=irreg_t)
            else:
                drift_test, drift_MLP, drift_fore, val_loss, stats, stats_fore, stats_cond_fore = val_out
            if generative == False:
                if (i+1)%int(n_epochs/5) == 0:
                    if drift_test.shape[-1] < 10:
                        plot_drift(dt.detach().cpu().numpy(), drift_test, drift_MLP, 
                                i, plot_savepath, data_param["simulation"], 
                                train_split_t=train_split_t, drift_fore = drift_fore, train=False)
                        plot_particle(dt.detach().cpu().numpy(), drift_test, drift_MLP, 
                                    i, plot_savepath, data_param["simulation"], train=False, 
                                    plot_particles=drift_test.shape[0], train_split_t=train_split_t, 
                                    drift_fore = drift_fore, irreg_t=irreg_t)
            
            drift_MSE = stats["MSE"]
            drift_MAE = stats["MAE"]
            if x_s.shape[-1] == 1:
                CRPS  = stats["CRPS"]
                
            pbar.set_description("Curr E-b: {}-{} | NF Loss: {} | Path Loss: {} | MSE: {} | MAE: {} | CRPS: {}".format(
                    i+1, 
                    batch_idx+1,
                    round(NF_loss, 2),
                    round(val_loss, 2),
                    round(drift_MSE, 2),
                    round(drift_MAE, 2),
                    round(CRPS, 2)
            ))
            if extrapolate == False:
                if len(val_loss_list) > 3:
                    # warmup to the 3rd evaluation then compare
                    if np.min(val_loss_list[3:]) > val_loss:
                        best_epoch = i + 1
                        if NF: torch.save(NF.state_dict(), net_savepath + "/NF.pt")
                        torch.save(driftMLP.state_dict(), net_savepath + "/driftMLP.pt")
                # save the 3rd evaluation
                elif len(val_loss_list) == 3:
                    best_epoch = i + 1
                    if NF: torch.save(NF.state_dict(), net_savepath + "/NF.pt")
                    torch.save(driftMLP.state_dict(), net_savepath + "/driftMLP.pt")
            else:
                if len(MSE_list) > 0:
                    if np.min(MSE_list) > drift_MSE:
                        best_epoch = i + 1
                        if NF: torch.save(NF.state_dict(), net_savepath + "/NF.pt")
                        torch.save(driftMLP.state_dict(), net_savepath + "/driftMLP.pt")
                else:
                    if NF: torch.save(NF.state_dict(), net_savepath + "/NF.pt")
                    torch.save(driftMLP.state_dict(), net_savepath + "/driftMLP.pt")
                
            val_loss_list.append(val_loss)
            MSE_list.append(stats["MSE"])
            MAE_list.append(stats["MAE"])
            if x_s.shape[-1] == 1:
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
        if NF:
            NF.train()
        driftMLP.train()
            
        
    stats = {"NF":NF_loss_list, "MLP":MLP_loss_list, 
             "MSE":MSE_list, "MAE": MAE_list, "ed":ed_list,
             "val_loss": val_loss_list, "CRPS":CRPS_list, 'best_epoch':best_epoch}
    return stats, best_epoch

def validate(driftMLP, 
             NF, 
             data_param,
             device, 
             loss_type,
             val_path_loader=None,
             extrapolate=False,
             generative=False,
             train_split_t=None,
             train_paths=None,
             train_particle_labels=None,
             **params):
    test_y_samp_size = params.setdefault('test_y_samp_size', 128)
    test_label_x = None
    test_label_y = None
    try:
        test_label_x    = val_path_loader.dataset.tensors[2]
        test_label_y    = val_path_loader.dataset.tensors[2]
    except Exception as e:
        test_label_x = None
        test_label_y = None
    driftMLP.eval(); 
    if NF:
        NF.eval()
    with torch.no_grad():     
        if data_param["simulation"]:
            driftMLP.eval()
            if generative:
                # Get test drift and predict drift
                test_path       = val_path_loader.dataset.tensors[0].detach().cpu().numpy()
                gen_path_simu   = np.zeros(test_path.shape)
                drift_MLP       = np.zeros(test_path.shape)
                drift_cond_fore = np.zeros(test_path.shape)
                drift_test      = np.zeros(test_path.shape)

                start = 0
                for val_idx, data in enumerate(val_path_loader):
                    # 3-d tensors
                    x   = data[0].float().to(device)
                    t   = data[1].float().to(device)
                    dt  = t[0,1] - t[0,0]
                    end = start + x.shape[0]
                    gen_path_simu[start:end] = generate_path(driftMLP=driftMLP,
                                                  NF=NF,
                                                  device = device,
                                                  label_x = None,
                                                  label_y = None,
                                                  x_init = x[:,0],
                                                  y_samps = train_paths,
                                                  num_samples = x.shape[0],
                                                  ts = t[0,:],
                                                  sigma = driftMLP.sigma,
                                                  n_vars = x.shape[-1]).detach().cpu().numpy()

                    start=end
                    ts_val = t[0,:].detach().cpu().numpy()
                
            elif extrapolate: 
                # Sample new paths to evaluate drift MSE
                xs_val, ts_val, _, _= sim_process_mckean(**data_param)
                ysamps_val = torch.from_numpy(xs_val)

                # Get test drift and predict drift
                drift_test = simu_drift(**data_param, y_samps=ysamps_val.cpu().numpy(), x_obs=xs_val, ts=ts_val)

                drift_MLP = MLP_drift(**data_param,
                                      driftMLP=driftMLP,
                                      NF=NF,
                                      y_samps=ysamps_val,
                                      x_obs=torch.from_numpy(xs_val),
                                      ts = torch.from_numpy(ts_val),
                                      label_x=None, 
                                      label_y=None,
                                      device=device
                                     ).detach().cpu().numpy()

                drift_cond_fore = np.zeros(xs_val.shape)

                gen_path_simu = generate_path(driftMLP=driftMLP,
                                              NF=NF,
                                              device = device,
                                              label_x = None,
                                              label_y = None,
                                              x_init = xs_val[:,0],
                                              num_samples = xs_val.shape[0],
                                              ts = ts_val,
                                              sigma = driftMLP.sigma,
                                              n_vars = xs_val.shape[-1]).detach().cpu().numpy()
                test_path = xs_val
            else: 
                # Get test drift and predict drift
                test_path       = val_path_loader.dataset.tensors[0].detach().cpu().numpy()
                test_label_x = None
                test_label_y = None
                try:
                    test_label_x    = val_path_loader.dataset.tensors[2]
                    test_label_y    = val_path_loader.dataset.tensors[2]
                except IndexError:
                        test_label_x = None
                        test_label_y = None
                gen_path_simu   = np.zeros(test_path.shape)
                drift_MLP       = np.zeros(test_path.shape)
                drift_cond_fore = np.zeros(test_path.shape)
                drift_test      = np.zeros(test_path.shape)

                start = 0
                for val_idx, data in enumerate(val_path_loader):
                    # 3-d tensors
                    x   = data[0].float().to(device)
                    t   = data[1].float().to(device)
                    dt  = t[0,1] - t[0,0]
                    end = start + x.shape[0]
                    
                    try:
                        if len(data[0].shape) > 3:
                            label_x = (data[2].float().squeeze(0).to(device) if driftMLP.label_x else None)
                            label_y = (data[2].float().squeeze(0).to(device) if driftMLP.label_y else None)
                        else: 
                            label_x = (data[2].float().to(device) if driftMLP.label_x else None)
                            label_y = (data[2].float().to(device) if driftMLP.label_y else None)
                    except IndexError:
                        pass
                    
                    drift_test[start:end] = simu_drift(**data_param, y_samps=train_paths.cpu().numpy(), 
                                                       x_obs=x.detach().cpu().numpy(), 
                                                       ts=t[0,:].detach().cpu().numpy())

                    drift_MLP[start:end] = MLP_drift(**data_param,
                                          driftMLP=driftMLP,
                                          NF=NF,
                                          y_samps=train_paths,
                                          x_obs=x,
                                          ts = t[0,:],
                                          label_x = label_x,
                                          label_y = train_particle_labels,
                                          device=device
                                         ).detach().cpu().numpy()

                    gen_path_simu[start:end] = generate_path(driftMLP=driftMLP,
                                                  NF=NF,
                                                  device = device,
                                                  label_x = label_x,
                                                  label_y = train_particle_labels,
                                                  x_init = x[:,0],
                                                  y_samps = train_paths,
                                                  num_samples = x.shape[0],
                                                  ts = t[0,:],
                                                  sigma = driftMLP.sigma,
                                                  n_vars = x.shape[-1]).detach().cpu().numpy()

                    start=end
                    ts_val = t[0,:].detach().cpu().numpy()

            val_loss = 0
            g_loss = nn.GaussianNLLLoss(reduction="mean")
            for val_idx, data in enumerate(val_path_loader):
                x = data[0].float().to(device)
                t = data[1].float().to(device)
                for t_idx in range(t.shape[1]-1):
                    # if irreg_t is None, do normal time propogation with step 1
                    x_s = x[:, t_idx, :]
                    x_t = x[:, t_idx+1, :]
                    y_t = None
                    if NF:
                        y_t = NF.sample(test_y_samp_size, 
                                        cond_inputs = t[0, t_idx].repeat(test_y_samp_size,1)).detach()
                    elif driftMLP.W_0_hidden == 0 and NF is None:
                        y_t = x_s
                
                    drift = driftMLP(x_s, y_t, t[0, t_idx].to(device),
                                    test_label_x.to(device)if test_label_x is not None else test_label_x, 
                                    test_label_y.to(device)if test_label_y is not None else test_label_y)
                    
                    sigma = driftMLP.sigma_forward(t[0, t_idx])
                    sigma = torch.clip(sigma, min=1e-3)
                    dt = t[0,1] - t[0,0]
                    if loss_type=="girsanov":
                        val_loss += -(drift*(x_t-x_s)/sigma.repeat(x_t.shape[0],1)**2 - \
                                    0.5*drift**2*dt/sigma.repeat(x_t.shape[0],1)**2).sum()
                    else:
                        val_loss += g_loss(target=x_t-x_s, input=drift*dt,
                                        var=sigma**2*dt.repeat(*(x_t.shape)))
            val_loss = val_loss/test_path.shape[0]
            
            if generative == False:
                stats = evaluation(test = drift_test, gen=drift_MLP)
                stats_fore = evaluation(test=drift_test[..., train_split_t-1:, :],
                                        gen= drift_MLP[..., train_split_t-1:, :]) if train_split_t else None
                stats_cond_fore = evaluation(test=drift_test[..., train_split_t-1:, :],
                                             gen= drift_cond_fore) if train_split_t else None
            else:
                drift_test = 0
                drift_MLP = 0
                drift_cond_fore = 0
                stats = evaluation(test = gen_path_simu, gen=test_path)
                stats_fore = {'MSE':0, 'MAE':0}
                stats_cond_fore = {'MSE':0, 'MAE':0}

            return (gen_path_simu, test_path, drift_test, drift_MLP, 
                    drift_cond_fore, val_loss.item(), stats, stats_fore, stats_cond_fore)

        else:
            test_path = val_path_loader.dataset.tensors[0].detach().cpu().numpy()
            gen_path  = np.zeros(test_path.shape)
            if train_split_t:
                gen_path_cond_fore = np.zeros((*test_path.shape[:-2], 
                                              test_path.shape[-2] - (train_split_t-1), 
                                              test_path.shape[-1]
                                             ))

            start = 0
            for val_idx, data in enumerate(val_path_loader):
                x  = data[0].float().to(device)
                t = data[1].float().to(device)
                dt = t[0,1] - t[0,0]
                end = start + x.shape[0]
                try:
                    label_x = (data[2].float().to(device) if driftMLP.label_x else None)
                    label_y = (data[2].float().to(device) if driftMLP.label_y else None)
                except IndexError:
                    pass
                gp  = generate_path(driftMLP=driftMLP,
                                    NF=NF,
                                    device = device,
                                    label_x = label_x,
                                    label_y = label_y,
                                    x_init = x[:,0],
                                    y_samps = x if x.shape[1] == len(t[0,:]) else None,
                                    num_samples = x.shape[0],
                                    ts = t[0,:],
                                    sigma = driftMLP.sigma,
                                    n_vars = x.shape[-1]).detach().cpu().numpy()
                
                gen_path[start:end] = gp 

                # Conditional Forecast given last observed data
                if train_split_t:
                    gen_path_cond_fore[start:end] = generate_path(driftMLP=driftMLP,
                                                                  NF=NF,
                                                                  device = device,
                                                                  label_x = label_x,
                                                                  label_y = label_y,
                                                                  x_init = x[:,train_split_t-1,:],
                                                                  y_samps = None,
                                                                  num_samples = x.shape[0],
                                                                  ts = t[0,train_split_t-1:],
                                                                  sigma = driftMLP.sigma,
                                                                  n_vars = x.shape[-1]).detach().cpu().numpy()

                start=end
                val_loss = 0
                g_loss = nn.GaussianNLLLoss(reduction="mean")
                for t_idx in range(train_split_t-1 if train_split_t else t.shape[1]-1):
                    x_s = x[:, t_idx, :]
                    x_t = x[:, t_idx+1, :]
                    y_t = None
                    if NF:
                        y_t = NF.sample(test_y_samp_size, 
                                        cond_inputs = t[0,t_idx].repeat(test_y_samp_size,1)).detach()
                    elif driftMLP.W_0_hidden == 0 and NF is None:
                        y_t = x_s
                    drift = driftMLP(x_s, y_t, t[0,t_idx], label_x, label_y)
                    sigma = driftMLP.sigma_forward(t[0,t_idx])
                    sigma = torch.clip(sigma, min=1e-3)
                    if loss_type=="girsanov":
                        val_loss += -(drift*(x_t-x_s)/sigma.repeat(x_t.shape[0],1)**2 - \
                                      0.5*drift**2*dt/sigma.repeat(x_t.shape[0],1)**2).sum()
                    else:
                        val_loss += g_loss(target=x_t-x_s, input=drift*dt,
                                        var=sigma**2*dt.repeat(*(x_t.shape)))
        val_loss = val_loss/gen_path.shape[0]
        gen_path = np.array(gen_path)
        gen_path_cond_fore = np.array(gen_path_cond_fore) if train_split_t else None
        
        stats =  evaluation(test=test_path[...,:train_split_t,:], 
                            gen=gen_path[...,:train_split_t,:]) if train_split_t else evaluation(test=test_path, gen=gen_path)
        
        stats_fore = evaluation(test=test_path[...,train_split_t:,:], 
                                gen=gen_path[...,train_split_t:,:]) if train_split_t else None
        
        stats_cond_fore = evaluation(test=test_path[...,train_split_t-1:,:], 
                                     gen=gen_path_cond_fore) if train_split_t else None
                    
        return test_path, gen_path, gen_path_cond_fore, val_loss.item(), stats, stats_fore, stats_cond_fore
     
    
def format_directory(experiment_folder, cfg, run):
    base_dir = "/scratch/hy190/MV-SDE"
    experiment_folder = str(experiment_folder)
    
    try: 
        brownian = cfg['other']['brownian']
        bb_n = cfg['other']['bb_m']
        bb_m = cfg['other']['bb_n']
        n_irreg = cfg['dataset']['n_irreg']
    except KeyError:
        bb_n = 0
        bb_m = 0
        brownian = None
        n_irreg = None
        
    try: 
        generative = cfg['dataset']['generative']
        n_samples = cfg['dataset']["n_samples"]
        n_bridge = cfg['dataset']["n_bridge"]
        n_points = cfg['dataset']["n_points"]
    except KeyError:
        generative = False
        n_samples = 0
        n_bridge = 0
        n_points = 0
    
    if cfg["NF"]:
        model_type = "NF_MLP"
        
    if cfg['MF']:
        if cfg["MF"]["net"]["W_0_hidden"] == 0 and cfg["NF"] is None and "W0_Xt" not in set(list(cfg["MF"]["net"].keys())):
            model_type = "MLP_Xt"
        elif cfg["MF"]["net"]["W_0_hidden"] != 0:
            model_type = "MLP_W0"
        elif "W0_Xt" in set(list(cfg["MF"]["net"].keys())):
            if cfg["MF"]["net"]["W0_Xt"]:
                model_type = "MLP_W0_Xt"
        act = cfg['MF']['net']['g_act']
        try: 
            flow = cfg['MF']['net']['W0_flow']
        except KeyError:
            flow = None
        try: 
            linear = cfg['MF']['net']['linear_drift']
        except KeyError:
            linear = None
        try: 
            KL_init = cfg['MF']['net']['KL_init']
        except KeyError:
            KL_init = None
        label_x = cfg['MF']['net']['label_x']
        label_y = cfg['MF']['net']['label_y'] 
        label_g = cfg['MF']['net']['label_g'] 
        label_f = cfg['MF']['net']['label_f']
        lr = cfg['MF']['lr']
        loss_type = cfg['other']['loss_type']
        W0_width = cfg['MF']['net']['W_0_hidden']
    elif cfg['MLP']:
        model_type = "MLP"
        act = cfg['MLP']['net']['act']
        flow = None
        linear = None
        KL_init = None
        label_x = None 
        label_y = None 
        label_g = None 
        label_f = None
        lr = cfg['MLP']['lr']
        loss_type = cfg['other']['loss_type']
        W0_width = None
    elif cfg["deepAR"]:
        model_type = cfg['deepAR']['net']['rnn']
        act = cfg['deepAR']['net']['act']
        label_x = None 
        label_y = None 
        label_g = None 
        label_f = None
        lr = cfg['deepAR']['lr']
        loss_type = cfg['other']['loss_type']
        W0_width = None
    elif cfg["transformer"]:
        model_type = 'transformer'
        act = "relu"
        label_x = None 
        label_y = None 
        label_g = None 
        label_f = None
        lr = cfg['transformer']['lr']
        loss_type = cfg['other']['loss_type']
        W0_width = None
        
    if generative and cfg['dataset']["simulation"]:
        savepath = '{}/{}_{}_{}_{}/{}_{}_results_{}/{}_{}_{}/run_generative_{}/'.format(
                    experiment_folder,
                    model_type,
                    flow,
                    W0_width,
                    loss_type,
                    act,
                    lr,
                    cfg['head'],
                    n_samples,
                    n_bridge,
                    n_points,
                    run)
        
    elif generative == False and cfg['dataset']["simulation"]:
        drift_h = '_'.join(cfg['dataset']['fcn_h'].strip('][').split(', '))
        if "ablation" in yaml_filepath:
            savepath = '{}/{}_{}_{}_{}/{}_{}_results_{}/{}_npaths={}_dt={}/run_{}/'.format(
                    experiment_folder,
                    model_type,
                    flow,
                    W0_width,
                    loss_type,
                    act,
                    lr,
                    cfg['head'],
                    drift_h,
                    cfg['dataset']['n_particles'],
                    (cfg['dataset']['tn'] - cfg['dataset']['t0'])/cfg['dataset']['n_points'],
                    run)
        elif "flow" in experiment_folder:
            savepath = '{}/{}_{}_{}/{}_{}_results_{}/{}_npaths={}_dt={}/run_{}/'.format(
                    experiment_folder,
                    model_type,
                    flow,
                    loss_type,
                    act,
                    lr,
                    cfg['head'],
                    drift_h,
                    cfg['dataset']['n_particles'],
                    (cfg['dataset']['tn'] - cfg['dataset']['t0'])/cfg['dataset']['n_points'],
                    run)
        elif "linear" in experiment_folder:
            if "KL" in experiment_folder:
                savepath = '{}/{}_{}_{}_{}_{}_{}/{}_{}_results_{}/{}_npaths={}_dt={}/run_{}/'.format(
                        experiment_folder,
                        model_type,
                        flow,
                        linear,
                        KL_init,
                        W0_width,
                        loss_type,
                        act,
                        lr,
                        cfg['head'],
                        drift_h,
                        cfg['dataset']['n_particles'],
                        (cfg['dataset']['tn'] - cfg['dataset']['t0'])/cfg['dataset']['n_points'],
                        run)
            else:
                savepath = '{}/{}_{}_{}_{}_{}/{}_{}_results_{}/{}_npaths={}_dt={}/run_{}/'.format(
                        experiment_folder,
                        model_type,
                        flow,
                        linear,
                        W0_width,
                        loss_type,
                        act,
                        lr,
                        cfg['head'],
                        drift_h,
                        cfg['dataset']['n_particles'],
                        (cfg['dataset']['tn'] - cfg['dataset']['t0'])/cfg['dataset']['n_points'],
                        run)
        else:
            if brownian:
                savepath = '{}/{}_{}_{}_{}_{}_{}/{}_{}_results_{}/{}_npaths={}_dt={}/run_{}/'.format(
                    experiment_folder,
                    model_type,
                    loss_type,
                    brownian,
                    bb_n,
                    bb_m,
                    n_irreg,
                    act,
                    lr,
                    cfg['head'],
                    drift_h,
                    cfg['dataset']['n_particles'],
                    (cfg['dataset']['tn'] - cfg['dataset']['t0'])/cfg['dataset']['n_points'],
                    run)
            else: 
                savepath = '{}/{}_{}/{}_{}_results_{}/{}_npaths={}_dt={}/run_{}/'.format(
                        experiment_folder,
                        model_type,
                        loss_type,
                        act,
                        lr,
                        cfg['head'],
                        drift_h,
                        cfg['dataset']['n_particles'],
                        (cfg['dataset']['tn'] - cfg['dataset']['t0'])/cfg['dataset']['n_points'],
                        run)
    else: 
        savepath = '{}/{}_{}/{}_{}_{}_results_{}_{}_{}_{}_{}/{}_{}_{}/run_{}/'.format(
                experiment_folder,
                model_type,
                loss_type,
                act,
                lr,
                cfg['head'],
                cfg['dataset']['dataset_type'],
                label_x,
                label_y,
                label_g,
                label_f,
                cfg["dataset"]['split_type'],
                cfg['dataset']['subset_time'],
                cfg["dataset"]['split_size'],
                run)
    plot_savepath = os.path.join(savepath,'plots/')
    net_savepath = os.path.join(savepath,'saved_nets/')
    return savepath, plot_savepath, os.path.join(base_dir,net_savepath)

def make_directory(savepath, plot_savepath, net_savepath):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    if not os.path.exists(plot_savepath):
        os.makedirs(plot_savepath)
    if not os.path.exists(net_savepath):
        os.makedirs(net_savepath)

def get_parser():
    """Get parser object."""
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-f",
        "--file",
        dest="filename",
        help="experiment definition file",
        metavar="FILE",
        required=True,
    )
    parser.add_argument(
        "-d",
        "--device",
        dest="device",
        help="experiment specified device",
        required=False,
    )
    parser.add_argument(
        "-e",
        "--experiment",
        dest="experiment",
        help="experiment specified directory",
        required=False,
    )
    return parser

if __name__ == '__main__':
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
        
    if "simulation" in yaml_filepath or "generative" in yaml_filepath:
        n_runs = 10
        
    try:
        n_tries = cfg['n_tries']
    except KeyError:
        n_tries = 1

    for run in range(n_runs):
        global savepath
        global plot_savepath
        global net_savepath
        savepath, plot_savepath, net_savepath = format_directory(experiment_folder, cfg, run)
        make_directory(savepath, plot_savepath, net_savepath)
        initialized, test_loader,head = setup(yaml_filepath, device, seed=run)
        all_stats, best_epoch = train(**initialized)
        
        NF_best       = initialized["NF"]
        driftMLP_best = initialized["driftMLP"]
        if NF_best: 
            NF_best.load_state_dict(torch.load(net_savepath + "/NF.pt"))
            NF_best.eval()
        driftMLP_best.load_state_dict(torch.load(net_savepath + "/driftMLP.pt"))
        driftMLP_best.eval()
        
        try:
            train_particle_labels = initialized['path_loader'].dataset.tensors[2].detach()
        except IndexError:
            train_particle_labels = None
            
        with torch.no_grad():
            test_stats = validate(driftMLP_best, 
                                NF_best, 
                                initialized["data_param"], 
                                device, 
                                initialized["loss_type"], 
                                test_loader, 
                                extrapolate=initialized["extrapolate"],
                                generative=initialized["generative"],
                                train_paths=initialized['validation_loader'].dataset.tensors[0].detach(),
                                train_particle_labels=train_particle_labels,
                                train_split_t=initialized["train_split_t"])
            test_stats = test_stats + (best_epoch,)
            with open(os.path.join(savepath,'test_stats.pkl'), 'wb') as f:
                pickle.dump(test_stats, f)
                f.close()
                
            
            if initialized["data_param"]["simulation"]:
                gen_path_simu, test_path, drift_test, drift_MLP, drift_fore, val_loss, stats, _, _,best_epoch = test_stats
            else:
                drift_test, drift_MLP, drift_fore, val_loss, stats, stats_fore, stats_cond_fore, best_epoch = test_stats
            ts = initialized['path_loader'].dataset.tensors[1].detach().cpu().numpy()[0]
            
            if initialized["data_param"]["simulation"] is True and initialized["generative"] is False:
                try:
                    train_particle_labels = initialized['path_loader'].dataset.tensors[2].detach()
                except IndexError:
                    train_particle_labels = None
                plot_particle(initialized["dt"], test_path, gen_path_simu,
                            best_epoch, savepath, initialized["data_param"]["simulation"], 
                            paths=True, train=False, test=True,
                            plot_particles=test_path.shape[0], train_split_t=initialized["train_split_t"],
                            drift_fore = drift_fore, irreg_t=initialized['irreg_t'],
                            head=head)
                
                plot_drift(initialized["dt"], drift_test, drift_MLP, 
                        best_epoch, savepath, initialized["data_param"]["simulation"], test=True,
                        train_split_t=initialized["train_split_t"], drift_fore = drift_fore, head=head)
            
                plot_particle(initialized["dt"], drift_test, drift_MLP, 
                            best_epoch, savepath, initialized["data_param"]["simulation"], train=False, test=True,
                            plot_particles=test_path.shape[0], train_split_t=initialized["train_split_t"],
                            drift_fore = drift_fore, irreg_t=initialized['irreg_t'],
                            head=head)
                if initialized['path_loader'].dataset.tensors[0].detach().cpu().numpy().shape[-1] == 2 and initialized["extrapolate"] == False:
                    plot_gradient(xs=initialized['path_loader'].dataset.tensors[0].detach().cpu().numpy(), driftMLP=driftMLP_best, ts=ts, device=initialized["device"],
                                plot_savepath = savepath, snap_time=test_path.shape[1]-1, data_params = initialized["data_param"], samples = gen_path_simu[:,-1,:],
                                plot_scale=2, test=True, head=head, truth=False, train_particle_labels=train_particle_labels)
                    plot_gradient(xs=initialized['path_loader'].dataset.tensors[0].detach().cpu().numpy(), driftMLP=driftMLP_best, ts=ts, device=initialized["device"],
                                plot_savepath = savepath, snap_time=test_path.shape[1]-1, data_params = initialized["data_param"],
                                plot_scale=2, test=True, truth=True, head=head)
                    
                    plot_gradient(xs=initialized['path_loader'].dataset.tensors[0].detach().cpu().numpy(), driftMLP=driftMLP_best, ts=ts, device=initialized["device"],
                                plot_savepath = savepath, snap_time=test_path.shape[1]-1, data_params = initialized["data_param"],samples = gen_path_simu[:,-1,:],
                                plot_scale=2, test=True, head=head, truth=False, train_particle_labels=train_particle_labels, plot_kde=False)
                    plot_gradient(xs=initialized['path_loader'].dataset.tensors[0].detach().cpu().numpy(), driftMLP=driftMLP_best, ts=ts, device=initialized["device"],
                                plot_savepath = savepath, snap_time=test_path.shape[1]-1, data_params = initialized["data_param"],
                                plot_scale=2, test=True, truth=True, head=head, plot_kde=False)
                    
            if initialized["generative"] is True:
                plot_gen_scatter(gen_path_simu=gen_path_simu, xs_val=test_path, epoch=-1, 
                                plot_savepath=savepath, train=False, test=False)
                print("-------------\n", stats["energy_distance"], "----------\n")
                
                xs = test_loader.dataset.tensors[0].detach().cpu().numpy()
            
                # plot final model scatters
                plot_ot_map(otmap = gen_path_simu,
                            x_init=gen_path_simu[:,0,:], x_terminal=gen_path_simu[:,-1,:], 
                            plot_savepath = savepath, plot_particles=50, 
                            n_samples=xs.shape[0], train=False, truth=False, average=False, grid=False, seed=run)
                
                plot_ot_map(otmap = xs,
                            x_init=xs[:,0,:], x_terminal=xs[:,-1,:], 
                            plot_savepath = savepath, plot_particles=50, 
                            n_samples=xs.shape[0], train=False, truth=True, average=False, grid=False, seed=run)
                
                if xs.shape[-1] < 3:
                    plot_gradient(xs=xs, driftMLP=driftMLP_best, ts=ts, device=initialized["device"],generative=True, samples=gen_path_simu[:,-1,:],
                                    plot_savepath = savepath, snap_time=xs.shape[1]-1, data_params = initialized["data_param"],
                                    plot_scale=3, test=True, head=head, truth=False, train_particle_labels=train_particle_labels, plot_kde=True)
                    
                    plot_gradient(xs=xs, driftMLP=driftMLP_best, ts=ts, device=initialized["device"],generative=True,
                                    plot_savepath = savepath, snap_time=xs.shape[1]-1, data_params = initialized["data_param"],
                                    plot_scale=3, test=True, head=head, truth=True, train_particle_labels=train_particle_labels, plot_kde=True)
        
        train_ts = initialized['path_loader'].dataset.tensors[1].detach().cpu().numpy()
        drift_train = initialized['path_loader'].dataset.tensors[0].detach().cpu().numpy()
        val_ts = test_loader.dataset.tensors[1].detach().cpu().numpy()
       
        with open(os.path.join(savepath,'saved_stats.pkl'), 'wb') as f:
            pickle.dump(all_stats, f)
            f.close()
        
        
        