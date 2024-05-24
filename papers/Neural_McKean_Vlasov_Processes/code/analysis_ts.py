import pickle
import numpy as np
import glob
import re
import matplotlib.pyplot as plt
import yaml
from train import format_directory
from MeanFieldMLP import *
from MLP import *
import warnings
import seaborn as sns
import pandas as pd
warnings.filterwarnings("ignore")

global_path = "Input Global Path"

def analyze_simulation(experiment, yaml_file, stats_key=4, generation=False, simulation=False, 
                       FK=False,runs=5, samples=False, subsample=False):
    result_path = []
    for run in range(runs):
        yaml_filepath = experiment + yaml_file
        with open(yaml_filepath, 'r') as f:
            cfg = yaml.load(f, yaml.SafeLoader)
        savepath, plot_savepath, net_savepath = format_directory(experiment, cfg, run)
        result_path.append(savepath)

    GIR = []
    MSE = []
    MAE = []
    CRPS = []
    ENERGY = []
    ENERGY_by_t = []
    elbo = []
    best_run = []
    sampled = []
    temp = []
        
    for run, result in enumerate(result_path):
        try:
            stats_val = open(result + 'saved_stats.pkl', 'rb')
        except:
            if FK == False:
                break
            else:
                pass
        if FK == False:
            stats_dict_val = pickle.load(stats_val)
        
        temp.append(np.array(stats_dict_val["val_loss"]))
        
        stats_test = open(result + 'test_stats.pkl', 'rb')
        stats_dict_test = pickle.load(stats_test)
        best_run.append(stats_dict_test[-1])
        if FK == True:
            best_run = []
            test_elbo = stats_dict_test[-1]
            elbo.append(test_elbo)
        else:
            if generation:
                GIR.append(stats_dict_test[5])
            else:
                MSE.append(stats_dict_test[stats_key]['MSE'])
                MAE.append(stats_dict_test[stats_key]['MAE'])
                try:
                    ENERGY.append(stats_dict_test[stats_key]['energy_distance'])
                    if subsample == False:
                        ENERGY_by_t.append(stats_dict_test[stats_key]['energy_distance_by_time'])
                    if subsample == True:
                        ENERGY_by_t.append(stats_dict_test[stats_key]['energy_distance_sub_sample_by_time'])
                except KeyError:
                    pass
                sampled.append(stats_dict_test[1].var())
            try:
                CRPS.append(stats_dict_test[stats_key]['CRPS'])
            except KeyError:
                CRPS.append(0)
                
    if FK == False:
        if stats_key == 4:
            if samples:
                return np.array(CRPS),np.array(MSE)/np.mean(sampled)
            return np.mean(CRPS), np.std(CRPS), np.mean(MSE)/np.mean(sampled), np.std(np.array(MSE)/np.mean(sampled))
        elif generation is True:
            if samples: 
                return np.array(GIR)
            return np.mean(GIR), np.std(GIR)
        elif stats_key == 5 and simulation is False:
            if samples:
                return np.array(CRPS),np.array(MSE)/np.mean(sampled)
            return np.mean(CRPS), np.std(CRPS), np.mean(MSE)/np.mean(sampled), np.std(np.array(MSE)/np.mean(sampled))
        elif stats_key == 6 and simulation is False:
            if samples:
                return np.array(CRPS),np.array(MSE)/np.mean(sampled)
            return np.mean(CRPS), np.std(CRPS),  np.mean(MSE)/np.mean(sampled), np.std(np.array(MSE)/np.mean(sampled))
        elif stats_key == 6 and simulation is True:
            if samples:
                return np.array(MSE)
            return np.mean(MSE), np.std(MSE)
    else:
        val_factor = cfg["other"]["test_n_particles"]/cfg["loader"]["batch_size_paths"]
        if samples: 
            return np.array(elbo)/val_factor
        return (np.array(elbo)/val_factor).mean(), (np.array(elbo)/val_factor).std()
    




######################################### Analysis on Synthetic Data (OU jump in analysis_generative) ###########################################   
    
datatype = ["kura", "fitz", "opinion","atlas",  "OU", "CIR"]
noise    = [1.0, 0.5, 0.1]
methods  = ["MLP", "W0", "NF","Xt"]
method_name = ["MLP", "IM", "ML","EM"]
#irregular_10_brownian_10_5_MLP_100_30
n_points    = [100]
n_particles = [30]
bb_n = [10]
bb_m = [5]
n_irreg = [10]
noise_mse = []
noise_sd  = []

nb_noise_mse = []
nb_noise_sd  = []

for da in datatype:
    d_noise_mse = []
    d_noise_sd  = []
    nb_d_noise_mse = []
    nb_d_noise_sd  = []
    for met in methods:
        d_noise_mse.append([])
        d_noise_sd.append([])
        
        nb_d_noise_mse.append([])
        nb_d_noise_sd.append([])
        for noi in noise:
            for npoi in n_points:
                for npar in n_particles:
                    for bn in bb_n:
                        for bm in bb_m: 
                            for nirreg in n_irreg:
                                mse, sd = analyze_simulation("{}/add_noise/noise_{}/noise={}/".format(global_path, da, noi),
                                                "irregular_{}_brownian_{}_{}_{}_{}_{}.yaml".format(nirreg, bn, bm, 
                                                                                                met, npoi, npar), 
                                            stats_key=6, simulation=True, plot=False, verbose=False, runs=10)
                                d_noise_mse[-1].append(mse)
                                d_noise_sd[-1].append(sd)
                    
            for npoi in n_points:
                for npar in n_particles:
                    for nirreg in n_irreg:
                        mse, sd = analyze_simulation("{}/add_noise/noise_{}/noise={}/".format(global_path, da, noi),
                                        "irregular_{}_{}_{}_{}.yaml".format(nirreg,met, npoi, npar), 
                                        stats_key=6, simulation=True, plot=False, verbose=False, runs=10)
                        nb_d_noise_mse[-1].append(mse)
                        nb_d_noise_sd[-1].append(sd)
                        
    noise_mse.append(d_noise_mse)
    noise_sd.append(d_noise_sd)
    nb_noise_mse.append(nb_d_noise_mse)
    nb_noise_sd.append(nb_d_noise_sd)
    
fig, ax = plt.subplots(figsize=(20,3), dpi=200, ncols=6)
datatype = ["Kuramoto", "Fitzhugh-Nagumo", "Opinion Dynamic", "Meanfield Atlas",  "OU", "Circles"]
for i, d in enumerate(datatype):
    for j, m in enumerate(methods):
        if j == 4:
            continue
        ax[i].plot(noise, noise_mse[i][j], marker='o')
        #ax[i].errorbar(noise, noise_mse[i][j], noise_sd[i][j], capsize=3)
    ax[i].invert_xaxis()
    ax[i].set_xticks(noise)
    #ax[i].legend(methods[0:4])
    ax[i].set_title(d, fontsize=18)
    ax[i].set_xlabel("Noise Level", fontsize=16)
    ax[0].set_ylabel("MSE", fontsize=16)
    ax[i].tick_params(axis='both', which='major', labelsize=18)
method_name = ["MLP", r"$W_0$", r"$\hat{P}_t$","Cylindrical"]
fig.legend(method_name, loc='lower center', bbox_to_anchor=(0.5,-0.20), ncol=len(method_name), prop={'size': 18},
           bbox_transform=fig.transFigure)
plt.tight_layout()    


######################################### Plotting on Synthetic Data (Appendix) ###########################################
from train import *
methods = ["MLP","Xt", "W0", "NF"]
path = "/scratch/hy190/MV-SDE/"
methods_label = ["MLP", "EM", "IM", "ML"]
exp = ["kura", "fitz", "opinion", "atlas", "OU", "CIR"]
exp_label = ["Kuramoto, " + r"$X_{1t}$", "Fitzhugh-Nagumo, " + r"$X_{1t}$", #"Fitzhugh-Nagumo, " + r"$X_{2t}$",
             "Opinion Dynamic, " + r"$X_{1t}$", "Meanfield Atlas, " + r"$X_{1t}$",
             "OU " + r"$X_{1t}$", "Circles " + r"$X_{1t}$"]
device="cuda:0"

fig, axes = plt.subplots(figsize=(2.5*6,1.8*5), ncols=6, nrows=5, dpi=200)

for idx, met in enumerate(methods):
    for i,e in enumerate(exp):
        result_path = []
        experiment = "{}/add_noise/noise_{}/noise=0.1/".format(global_path, e)
        yaml_file = "irregular_10_brownian_10_5_{}_100_30.yaml".format(met)
        yaml_filepath = experiment + yaml_file
        print(yaml_filepath)
        en_all = []
        for run in range(8,9):
            initialized, test_loader, head = setup(yaml_filepath, device, seed=run)
            driftMLP_best = initialized["driftMLP"]
            NF_best = initialized["NF"]
            
            with open(yaml_filepath, 'r') as f:
                cfg = yaml.load(f, yaml.SafeLoader)
            savepath, plot_savepath, net_savepath = format_directory(experiment, cfg, run)

            train_split_t = initialized["train_split_t"]
            test_stats = pickle.load(open(savepath + "test_stats.pkl", 'rb'))
            
            gen_path_simu, test_path, drift_test, drift_MLP, drift_fore, _, _, _, _,_ = test_stats
            test_path = test_loader.dataset.tensors[0].detach().cpu().numpy()
            dim = 0
            
            if "2" in exp_label[i]:
                dim=1
            ts = test_loader.dataset.tensors[1].detach().cpu().numpy()[0]
            
            ax = axes[0][i]
            
            colors = plt.cm.viridis(np.linspace(0,1,test_path.shape[0]))
            sort_particles = test_path[:,0,0].argsort()
            irreg_t = initialized['irreg_t']
            test_path = test_path[sort_particles]
            for particle in range(test_path.shape[0]):
                sns.lineplot(x=ts, y=test_path[particle,:,dim], ax=ax, color = colors[particle], alpha=0.15)
                sns.scatterplot(x=ts[irreg_t], y=test_path[particle,irreg_t,dim], ax=ax, color="grey", alpha=0.15)
            ax.set_title(exp_label[i], loc="left")
            
            ax = axes[idx+1][i]
            
            test_mean = drift_test.mean(0)[:,dim]
            gen_mean = drift_MLP.mean(0)[:,dim]
            fore_mean = drift_fore.mean(0)[:, dim]
            
            test_std = drift_test.std(0)[:,dim]
            gen_std = drift_MLP.std(0)[:,dim]
            fore_std = drift_fore.std(0)[:,dim]
            
            sns.lineplot(x=ts, y=test_mean, color="black", ax=ax)
            sns.lineplot(x=ts, y=gen_mean, color="red", ax=ax)
            if drift_fore is not None and train_split_t:
                sns.lineplot(x=ts[train_split_t-1:], y=fore_mean, color="green", ax=ax)
            ax.fill_between(ts, test_mean - test_std, test_mean + test_std, color='black', alpha=0.15)
            ax.fill_between(ts, gen_mean - gen_std, gen_mean + gen_std, color='red', alpha=0.15)
            axes[idx+1][0].set_title(methods_label[idx], loc="left")
    
            torch.cuda.empty_cache()
            
plt.tight_layout()
plt.savefig("{}/mean_samples.pdf".format(global_path), bbox_inches="tight")

    
######################################### Analysis on EEG, Crowd Trajectory, and Chemotaxi ########################################### 

"""
print_index = 0 means T_0 -> T_train generation (main text result)
print_index = 1 means T_0 -> T_forecast generation (Appendix forecast type I)
print_index = 2 means T_train - 1 -> T_forecast generation (Appendix forecast type II)
"""

CRPS_all = []
big_table = []
big_table_sd = []
print_index = 0

for subject_number in range(1,6):
    experiment = ["{}/EEG_by_electrode/subject{}/".format(global_path, subject_number),
                  "{}/EEG_deepAR/subject{}/".format(global_path, subject_number)]
    x_axis=[]
    CRPS = []
    CRPS_SD = []
    labels = []
    count = 1
    keys = [4,5,6]; gen_type = [r"$t \in [T_0, T_{train}]$", r"$ t\in [T_0, T_{eval}]$", 
                                r"$t \in [T_{train}, T_{eval}]$"]
    labels = ["MLP", r"$W_0$", r"$\hat{P}_t$", r"$X_t$",  "LSTM", "RNN", "GRU", "TR"]
    methods     = ["MLP","W0","NF","Xt"]#
    for met in methods:
        c_temp = []
        for k,g in zip(keys, gen_type):
            crps, mse = analyze_simulation(experiment[0], "{}.yaml".format(met), k, samples=True, runs=5)
            c_temp.append(crps)
        CRPS.append(c_temp)

    methods     =["lstm", "rnn", "gru", "transformer"]
    for met in methods:
        c_temp = []
        exp = experiment[1]
        for k,g in zip(keys, gen_type):
            crps, mse = analyze_simulation(exp, "{}.yaml".format(met), k, plot=False, samples=True,runs=5)
            
            c_temp.append(crps)
        CRPS.append(c_temp)
    CRPS_all.append(CRPS)
    
table = np.asarray(CRPS_all).mean(-1).mean(0)
big_table.append(table)
table_sd = np.asarray(CRPS_all).mean(-1).std(0)
big_table_sd.append(table_sd)
for i in range(8):
    print(np.round(table,3)[i,print_index], np.round(table_sd,3)[i,print_index])
print("\n")

CRPS_all = []
for subject_number in range(1,6):
    experiment = ["{}/post_icml/EEG_by_electrode_a/subject{}a/".format(global_path, subject_number),
                  "{}/post_icml/EEG_deepAR_a/subject{}/".format(global_path, subject_number)]
    x_axis=[]
    CRPS = []
    CRPS_SD = []
    labels = []
    count = 1
    keys = [4,5,6]; gen_type = [r"$t \in [T_0, T_{train}]$", r"$ t\in [T_0, T_{eval}]$", 
                                r"$t \in [T_{train}, T_{eval}]$"]
    labels = ["MLP", r"$W_0$", r"$\hat{P}_t$", r"$X_t$",  "LSTM", "RNN", "GRU", "TR"]
    methods     = ["MLP","W0","NF","Xt"]#
    for met in methods:
        c_temp = []
        for k,g in zip(keys, gen_type):
            crps, mse = analyze_simulation(experiment[0], "{}.yaml".format(met), k, samples=True, runs=5)
            c_temp.append(crps)
        CRPS.append(c_temp)

    methods     =["lstm", "rnn", "gru", "transformer"]
    for met in methods:
        c_temp = []
        for k,g in zip(keys, gen_type):
            crps, mse = analyze_simulation(experiment[1], "{}.yaml".format(met), k, plot=False, samples=True,runs=5)
            c_temp.append(crps)
        CRPS.append(c_temp)
    CRPS_all.append(CRPS)
    
table = np.asarray(CRPS_all).mean(-1).mean(0)
big_table.append(table)
table_sd = np.asarray(CRPS_all).mean(-1).std(0)
big_table_sd.append(table_sd)
for i in range(8):
    print(np.round(table,3)[i,print_index], np.round(table_sd,3)[i,print_index])
print("\n")
    
experiment = ["{}/Chemotaxi/Ccres/".format(global_path, subject_number),
              "{}/Chemotaxi_deepAR/Ccres/".format(global_path, subject_number)]
x_axis=[]
CRPS = []
CRPS_SD = []
labels = []
count = 1
keys = [4,5,6]; gen_type = [r"$t \in [T_0, T_{train}]$", r"$ t\in [T_0, T_{eval}]$", 
                            r"$t \in [T_{train}, T_{eval}]$"]
labels = ["MLP", r"$W_0$", r"$\hat{P}_t$", r"$X_t$",  "LSTM", "RNN", "GRU", "TR"]
methods     = ["MLP","W0","NF","Xt"]
for met in methods:
    c_temp = []
    for k,g in zip(keys, gen_type):
        crps, mse = analyze_simulation(experiment[0], "{}.yaml".format(met), k, samples=True, runs=5)
        c_temp.append(mse)
    CRPS.append(c_temp)
methods     =["lstm", "rnn", "gru", "transformer"]
for met in methods:
    c_temp = []
    exp = experiment[1]
    for k,g in zip(keys, gen_type):
        crps, mse = analyze_simulation(exp, "{}.yaml".format(met), k, plot=False, samples=True,runs=5)
        c_temp.append(mse)
    CRPS.append(c_temp)
    
table = np.asarray(CRPS).mean(-1)
big_table.append(table)
table_sd = np.asarray(CRPS).std(-1)
big_table_sd.append(table_sd)
for i in range(8):
    print(np.round(table,3)[i,print_index], np.round(table_sd,3)[i,print_index])
print("\n")

experiment = ["{}/crowd_traj/".format(global_path, subject_number),
              "{}/crowd_traj_deepAR/".format(global_path, subject_number)]
x_axis=[]
CRPS = []
CRPS_SD = []
labels = []
count = 1
keys = [4,5,6]; gen_type = [r"$t \in [T_0, T_{train}]$", r"$ t\in [T_0, T_{eval}]$", 
                            r"$t \in [T_{train}, T_{eval}]$"]
labels = ["MLP", r"$W_0$", r"$\hat{P}_t$", r"$X_t$",  "LSTM", "RNN", "GRU", "TR"]
methods     = ["MLP","W0","NF","Xt"]#
for met in methods:
    c_temp = []
    for k,g in zip(keys, gen_type):
        crps, mse = analyze_simulation(experiment[0], "{}.yaml".format(met), k, samples=True, runs=5)
        c_temp.append(mse)
    CRPS.append(c_temp)
methods     =["lstm", "rnn", "gru", "transformer"]
for met in methods:
    c_temp = []
    exp = experiment[1]
    for k,g in zip(keys, gen_type):
        crps, mse = analyze_simulation(exp, "{}.yaml".format(met), k, plot=False, samples=True,runs=5)
        c_temp.append(mse)
    CRPS.append(c_temp)
        #CRPS_SD.append(crps_sd)
    
table = np.asarray(CRPS).mean(-1)
big_table.append(table)
table_sd = np.asarray(CRPS).std(-1)
big_table_sd.append(table_sd)
for i in range(8):
    print(np.round(table,3)[i,print_index], np.round(table_sd,3)[i,print_index])
print("\n")


experiment = ["{}/Chemotaxi/Ecoli/".format(global_path, subject_number),
              "{}/Chemotaxi_deepAR/Ecoli/".format(global_path, subject_number)]
x_axis=[]
CRPS = []
CRPS_SD = []
labels = []
count = 1
keys = [4,5,6]; gen_type = [r"$t \in [T_0, T_{train}]$", r"$ t\in [T_0, T_{eval}]$", 
                            r"$t \in [T_{train}, T_{eval}]$"]
labels = ["MLP", r"$W_0$", r"$\hat{P}_t$", r"$X_t$",  "LSTM", "RNN", "GRU", "TR"]
methods     = ["MLP","W0","NF","Xt"]#
for met in methods:
    c_temp = []
    for k,g in zip(keys, gen_type):
        crps, mse = analyze_simulation(experiment[0], "{}.yaml".format(met), k, samples=True, runs=5)
        c_temp.append(mse)
    CRPS.append(c_temp)
methods     =["lstm", "rnn", "gru", "transformer"]
for met in methods:
    c_temp = []
    exp = experiment[1]
    for k,g in zip(keys, gen_type):
        crps, mse = analyze_simulation(exp, "{}.yaml".format(met), k, plot=False, samples=True,runs=5)
        c_temp.append(mse)
    CRPS.append(c_temp)
        #CRPS_SD.append(crps_sd)
    
table = np.asarray(CRPS).mean(-1)
big_table.append(table)
table_sd = np.asarray(CRPS).std(-1)
big_table_sd.append(table_sd)
for i in range(8):
    print(np.round(table,3)[i,print_index], np.round(table_sd,3)[i,print_index])
print("\n")


######################################### Plotting on EEG, crowd trajectory and Chemotaxi (Appendix) ###########################################

from train import *
from train_deepAR import *
from setup_deepAR import *

methods     = ["MLP", "Xt", "W0", "NF"]
methods_label     = ["MLP", "EM", "IM", "ML"]
device="cuda:0"
path = "/scratch/hy190/MV-SDE/"
fig, axes = plt.subplots(figsize=(12,1.8*8), ncols = 2, nrows=8, dpi=200)
for idx, met in enumerate(methods):
    result_path = []
    experiment = "{}/EEG_by_electrode/subject2/".format(global_path)
    yaml_file = "{}.yaml".format(met)
    yaml_filepath = experiment + yaml_file
    en_all = []
    for run in range(1,2):
        initialized, test_loader, head = setup(yaml_filepath, device, seed=run)
        
        with open(yaml_filepath, 'r') as f:
            cfg = yaml.load(f, yaml.SafeLoader)
        savepath, plot_savepath, net_savepath = format_directory(experiment, cfg, run)
        device="cpu"
        train_split_t = initialized["train_split_t"]
        val_out = pickle.load(open(savepath + "test_stats.pkl", 'rb'))
        drift_test, drift_MLP, drift_fore, val_loss, stats, stats_fore, stats_cond_fore, be = val_out
        
        dim = 0
        ts = test_loader.dataset.tensors[1].detach().cpu().numpy()[0]
        test_mean = drift_test.mean(0)[:,dim]
        gen_mean = drift_MLP.mean(0)[:,dim]
        fore_mean = drift_fore.mean(0)[:, dim]
        
        test_std = drift_test.std(0)[:,dim]
        gen_std = drift_MLP.std(0)[:,dim]
        fore_std = drift_fore.std(0)[:,dim]
        
        
        ax = axes[idx][0]
        sns.lineplot(x=ts, y=test_mean, color="black", ax=ax)
        sns.lineplot(x=ts, y=gen_mean, color="red", ax=ax)
        if drift_fore is not None and train_split_t:
            sns.lineplot(x=ts[train_split_t-1:], y=fore_mean, color="green", ax=ax)
        ax.fill_between(ts, test_mean - test_std, test_mean + test_std, color='black', alpha=0.15)
        ax.fill_between(ts, gen_mean - gen_std, gen_mean + gen_std, color='red', alpha=0.15)
        if train_split_t:
            ax.fill_between(ts[train_split_t-1:], fore_mean - fore_std, fore_mean + fore_std, color='green', alpha=0.15)
        
        if train_split_t is not None:
            plt.axvline(x=ts[train_split_t-1], linestyle="--", color="black")

        ax.set_ylabel("Bioactivity")
        ax.set_title(methods_label[idx], loc="left")
        ax.set_xlim([-5, 261])
        ax.set_ylim([-30, 30])
    
        torch.cuda.empty_cache()

methods     = ["lstm", "rnn", "gru", "transformer"]
methods_label     = ["LSTM", "RNN", "GRU", "Transformer"]
device="cuda:0"

for idx, met in enumerate(methods):
    result_path = []
    experiment = "{}/EEG_deepAR/subject2/".format(global_path)
    yaml_file = "{}.yaml".format(met)
    yaml_filepath = experiment + yaml_file
    en_all = []
    for run in range(1,2):
        with open(yaml_filepath, 'r') as f:
            cfg = yaml.load(f, yaml.SafeLoader)
        savepath, plot_savepath, net_savepath = format_directory(experiment, cfg, run)
        val_out = pickle.load(open(savepath + "test_stats.pkl", 'rb'))
        drift_test, drift_MLP, drift_fore, val_loss, stats, stats_fore, stats_cond_fore, be = val_out
        
        dim = 0
        ts = np.array(list(range(drift_test.shape[1])))
        test_mean = drift_test.mean(0)[:,dim]
        gen_mean = drift_MLP.mean(0)[:,dim]
        fore_mean = drift_fore.mean(0)[:, dim]
        
        test_std = drift_test.std(0)[:,dim]
        gen_std = drift_MLP.std(0)[:,dim]
        fore_std = drift_fore.std(0)[:,dim]
        
        ax = axes[idx+4][0]
        sns.lineplot(x=ts, y=test_mean, color="black", ax=ax)
        sns.lineplot(x=ts, y=gen_mean, color="red", ax=ax)
        if drift_fore is not None and train_split_t:
            sns.lineplot(x=ts[train_split_t-1:], y=fore_mean, color="green", ax=ax)
        ax.fill_between(ts, test_mean - test_std, test_mean + test_std, color='black', alpha=0.15)
        ax.fill_between(ts, gen_mean - gen_std, gen_mean + gen_std, color='red', alpha=0.15)
        if train_split_t:
            ax.fill_between(ts[train_split_t-1:], fore_mean - fore_std, fore_mean + fore_std, color='green', alpha=0.15)
        
        if train_split_t is not None:
            plt.axvline(x=ts[train_split_t-1], linestyle="--", color="black")

        ax.set_ylabel("Bioactivity")
        ax.set_title(methods_label[idx], loc="left")
        ax.set_xlim([-5, 261])
        ax.set_ylim([-30, 30])
    
        torch.cuda.empty_cache()

methods     = ["MLP", "Xt", "W0", "NF"]
methods_label     = ["MLP", "EM", "IM", "ML"]        

for idx, met in enumerate(methods):
    result_path = []
    experiment = "{}/EEG_by_electrode_a/subject1a/".format(global_path)
    yaml_file = "{}.yaml".format(met)
    yaml_filepath = experiment + yaml_file
    en_all = []
    for run in range(0,1):
        with open(yaml_filepath, 'r') as f:
            cfg = yaml.load(f, yaml.SafeLoader)
        savepath, plot_savepath, net_savepath = format_directory(experiment, cfg, run)
        
        val_out = pickle.load(open(savepath + "test_stats.pkl", 'rb'))
        drift_test, drift_MLP, drift_fore, val_loss, stats, stats_fore, stats_cond_fore, be = val_out
        
        dim = 0
        ts = test_loader.dataset.tensors[1].detach().cpu().numpy()[0]
        test_mean = drift_test.mean(0)[:,dim]
        gen_mean = drift_MLP.mean(0)[:,dim]
        fore_mean = drift_fore.mean(0)[:, dim]
        
        test_std = drift_test.std(0)[:,dim]
        gen_std = drift_MLP.std(0)[:,dim]
        fore_std = drift_fore.std(0)[:,dim]
        
        ax = axes[idx][1]
        sns.lineplot(x=ts, y=test_mean, color="black", ax=ax)
        sns.lineplot(x=ts, y=gen_mean, color="red", ax=ax)
        if drift_fore is not None and train_split_t:
            sns.lineplot(x=ts[train_split_t-1:], y=fore_mean, color="green", ax=ax)
        ax.fill_between(ts, test_mean - test_std, test_mean + test_std, color='black', alpha=0.15)
        ax.fill_between(ts, gen_mean - gen_std, gen_mean + gen_std, color='red', alpha=0.15)
        if train_split_t:
            ax.fill_between(ts[train_split_t-1:], fore_mean - fore_std, fore_mean + fore_std, color='green', alpha=0.15)
        
        if train_split_t is not None:
            plt.axvline(x=ts[train_split_t-1], linestyle="--", color="black")
    
        torch.cuda.empty_cache()
        
methods     = ["lstm", "rnn", "gru", "transformer"]
methods_label     = ["LSTM", "RNN", "GRU", "Transformer"]
device="cuda:0"

for idx, met in enumerate(methods):
    result_path = []
    experiment = "{}/EEG_deepAR_a/subject1/".format(global_path)
    yaml_file = "{}.yaml".format(met)
    yaml_filepath = experiment + yaml_file
    en_all = []
    for run in range(1,2):
        with open(yaml_filepath, 'r') as f:
            cfg = yaml.load(f, yaml.SafeLoader)
        savepath, plot_savepath, net_savepath = format_directory(experiment, cfg, run)
        val_out = pickle.load(open(savepath + "test_stats.pkl", 'rb'))
        drift_test, drift_MLP, drift_fore, val_loss, stats, stats_fore, stats_cond_fore, be = val_out
        
        dim = 0
        ts = np.array(list(range(drift_test.shape[1])))
        test_mean = drift_test.mean(0)[:,dim]
        gen_mean = drift_MLP.mean(0)[:,dim]
        fore_mean = drift_fore.mean(0)[:, dim]
        
        test_std = drift_test.std(0)[:,dim]
        gen_std = drift_MLP.std(0)[:,dim]
        fore_std = drift_fore.std(0)[:,dim]
        
        ax = axes[idx+4][1]
        sns.lineplot(x=ts, y=test_mean, color="black", ax=ax)
        sns.lineplot(x=ts, y=gen_mean, color="red", ax=ax)
        if drift_fore is not None and train_split_t:
            sns.lineplot(x=ts[train_split_t-1:], y=fore_mean, color="green", ax=ax)
        ax.fill_between(ts, test_mean - test_std, test_mean + test_std, color='black', alpha=0.15)
        ax.fill_between(ts, gen_mean - gen_std, gen_mean + gen_std, color='red', alpha=0.15)
        if train_split_t:
            ax.fill_between(ts[train_split_t-1:], fore_mean - fore_std, fore_mean + fore_std, color='green', alpha=0.15)
        
        if train_split_t is not None:
            plt.axvline(x=ts[train_split_t-1], linestyle="--", color="black")
    
        torch.cuda.empty_cache()
        
fig.legend(["True", "Generated", "Forecast"], 
           loc='lower center', bbox_to_anchor=(0.5,-0.04), 
           ncol=len(["True", "Generated", "Forecast"]), prop={'size': 13},
           bbox_transform=fig.transFigure)
plt.tight_layout()

plt.savefig("{}/EEG_mean_samples.pdf".format(global_path), bbox_inches="tight")


methods     = ["MLP", "Xt", "W0", "NF"]
methods_label     = ["MLP", "EM", "IM", "ML"]
device="cuda:0"

fig, axes = plt.subplots(figsize=(12,1.8*8), ncols = 2, nrows=8, dpi=200)
for idx, met in enumerate(methods):
    result_path = []
    experiment = "{}/Crowd_traj/".format(global_path)
    yaml_file = "{}.yaml".format(met)
    yaml_filepath = experiment + yaml_file
    en_all = []
    for run in range(1,2):
        initialized, test_loader, head = setup(yaml_filepath, device, seed=run)
        train_split_t = initialized["train_split_t"]
        with open(yaml_filepath, 'r') as f:
            cfg = yaml.load(f, yaml.SafeLoader)
        savepath, plot_savepath, net_savepath = format_directory(experiment, cfg, run)
        val_out = pickle.load(open(savepath + "test_stats.pkl", 'rb'))
        
        drift_test, drift_MLP, drift_fore, val_loss, stats, stats_fore, stats_cond_fore, be = val_out
 
        ylab = ["X", "Y"]
        for dim in range(drift_test.shape[-1]):
            ts = test_loader.dataset.tensors[1].detach().cpu().numpy()[0]
            test_mean = drift_test.mean(0)[:,dim]
            gen_mean = drift_MLP.mean(0)[:,dim]
            fore_mean = drift_fore.mean(0)[:, dim]
            
            test_std = drift_test.std(0)[:,dim]
            gen_std = drift_MLP.std(0)[:,dim]
            fore_std = drift_fore.std(0)[:,dim]
            
            
            ax = axes[idx][dim]
            sns.lineplot(x=ts, y=test_mean, color="black", ax=ax)
            sns.lineplot(x=ts, y=gen_mean, color="red", ax=ax)
            if drift_fore is not None and train_split_t:
                sns.lineplot(x=ts[train_split_t-1:], y=fore_mean, color="green", ax=ax)
            ax.fill_between(ts, test_mean - test_std, test_mean + test_std, color='black', alpha=0.15)
            ax.fill_between(ts, gen_mean - gen_std, gen_mean + gen_std, color='red', alpha=0.15)
            if train_split_t:
                ax.fill_between(ts[train_split_t-1:], fore_mean - fore_std, fore_mean + fore_std, color='green', alpha=0.15)
            
            if train_split_t is not None:
                plt.axvline(x=ts[train_split_t-1], linestyle="--", color="black")
                
            #plt.legend(, loc="lower left",ncol=3)
            #ax.set_xlabel("Time Steps")
            ax.set_ylabel(ylab[dim])
            axes[idx][0].set_title(methods_label[idx], loc="left")
    
        torch.cuda.empty_cache()
        
methods     = ["lstm", "rnn", "gru", "transformer"]
methods_label     = ["LSTM", "RNN", "GRU", "Transformer"]
device="cuda:0"

for idx, met in enumerate(methods):
    result_path = []
    experiment = "{}/crowd_traj_deepAR/".format(global_path)
    yaml_file = "{}.yaml".format(met)
    yaml_filepath = experiment + yaml_file
    en_all = []
    for run in range(1,2):
        initialized, test_loader, xs_val = setup_deepAR(yaml_filepath, device, seed=run)
        deepAR_best = initialized["deepAR"]
        train_split_t = initialized["train_split_t"]
        with open(yaml_filepath, 'r') as f:
            cfg = yaml.load(f, yaml.SafeLoader)
        savepath, plot_savepath, net_savepath = format_directory(experiment, cfg, run)
        val_out = pickle.load(open(savepath + "test_stats.pkl", 'rb'))
        
        drift_test, drift_MLP, drift_fore, val_loss, stats, stats_fore, stats_cond_fore, be = val_out
        
        for dim in range(drift_test.shape[-1]):
            ts = np.array(list(range(drift_test.shape[1])))
            test_mean = drift_test.mean(0)[:,dim]
            gen_mean = drift_MLP.mean(0)[:,dim]
            fore_mean = drift_fore.mean(0)[:, dim]
            
            test_std = drift_test.std(0)[:,dim]
            gen_std = drift_MLP.std(0)[:,dim]
            fore_std = drift_fore.std(0)[:,dim]
            
            
            ax = axes[4+idx][dim]
            sns.lineplot(x=ts, y=test_mean, color="black", ax=ax)
            sns.lineplot(x=ts, y=gen_mean, color="red", ax=ax)
            if drift_fore is not None and train_split_t:
                sns.lineplot(x=ts[train_split_t-1:], y=fore_mean, color="green", ax=ax)
            ax.fill_between(ts, test_mean - test_std, test_mean + test_std, color='black', alpha=0.15)
            ax.fill_between(ts, gen_mean - gen_std, gen_mean + gen_std, color='red', alpha=0.15)
            if train_split_t:
                ax.fill_between(ts[train_split_t-1:], fore_mean - fore_std, fore_mean + fore_std, color='green', alpha=0.15)
            
            if train_split_t is not None:
                ax.axvline(x=ts[train_split_t-1], linestyle="--", color="black")

            ylab = ["X", "Y"]
            ax.set_ylabel(ylab[dim])
            axes[4+idx][0].set_title(methods_label[idx], loc="left")
    
        torch.cuda.empty_cache()
        
fig.legend(["True", "Generated", "Forecast"], 
           loc='lower center', bbox_to_anchor=(0.5,-0.04), 
           ncol=len(["True", "Generated", "Forecast"]), prop={'size': 13},
           bbox_transform=fig.transFigure)
plt.tight_layout()

plt.savefig("{}/crowd_traj_mean_samples.pdf".format(global_path), bbox_inches="tight")


methods     = ["MLP", "Xt", "W0", "NF"]
methods_label     = ["MLP", "EM", "IM", "ML"]
device="cuda:0"

fig, axes = plt.subplots(figsize=(12,1.8*8), ncols = 3, nrows=8, dpi=200)
for idx, met in enumerate(methods):
    result_path = []
    experiment = "{}/Chemotaxi/Ccres/".format(global_path)
    yaml_file = "{}.yaml".format(met)
    yaml_filepath = experiment + yaml_file
    en_all = []
    for run in range(1,2):
        initialized, test_loader, head = setup(yaml_filepath, device, seed=run)
        train_split_t = initialized["train_split_t"]
        with open(yaml_filepath, 'r') as f:
            cfg = yaml.load(f, yaml.SafeLoader)
        savepath, plot_savepath, net_savepath = format_directory(experiment, cfg, run)
        val_out = pickle.load(open(savepath + "test_stats.pkl", 'rb'))
        
        drift_test, drift_MLP, drift_fore, val_loss, stats, stats_fore, stats_cond_fore, be = val_out
 
        ylab = ["X", "Y", "Z"]
        for dim in range(drift_test.shape[-1]):
            ts = test_loader.dataset.tensors[1].detach().cpu().numpy()[0]
            test_mean = drift_test.mean(0)[:,dim]
            gen_mean = drift_MLP.mean(0)[:,dim]
            fore_mean = drift_fore.mean(0)[:, dim]
            
            test_std = drift_test.std(0)[:,dim]
            gen_std = drift_MLP.std(0)[:,dim]
            fore_std = drift_fore.std(0)[:,dim]
            
            
            ax = axes[idx][dim]
            sns.lineplot(x=ts, y=test_mean, color="black", ax=ax)
            sns.lineplot(x=ts, y=gen_mean, color="red", ax=ax)
            if drift_fore is not None and train_split_t:
                sns.lineplot(x=ts[train_split_t-1:], y=fore_mean, color="green", ax=ax)
            ax.fill_between(ts, test_mean - test_std, test_mean + test_std, color='black', alpha=0.15)
            ax.fill_between(ts, gen_mean - gen_std, gen_mean + gen_std, color='red', alpha=0.15)
            if train_split_t:
                ax.fill_between(ts[train_split_t-1:], fore_mean - fore_std, fore_mean + fore_std, color='green', alpha=0.15)
            
            if train_split_t is not None:
                plt.axvline(x=ts[train_split_t-1], linestyle="--", color="black")
                
            #plt.legend(, loc="lower left",ncol=3)
            #ax.set_xlabel("Time Steps")
            ax.set_ylabel(ylab[dim])
            axes[idx][0].set_title(methods_label[idx], loc="left")
    
        torch.cuda.empty_cache()
        
methods     = ["lstm", "rnn", "gru", "transformer"]
methods_label     = ["LSTM", "RNN", "GRU", "Transformer"]
device="cuda:0"

for idx, met in enumerate(methods):
    result_path = []
    experiment = "{}/Chemotaxi_deepAR/Ccres/".format(global_path)
    yaml_file = "{}.yaml".format(met)
    yaml_filepath = experiment + yaml_file
    en_all = []
    for run in range(1,2):
        initialized, test_loader, xs_val = setup_deepAR(yaml_filepath, device, seed=run)
        deepAR_best = initialized["deepAR"]
        train_split_t = initialized["train_split_t"]
        with open(yaml_filepath, 'r') as f:
            cfg = yaml.load(f, yaml.SafeLoader)
        savepath, plot_savepath, net_savepath = format_directory(experiment, cfg, run)
        val_out = pickle.load(open(savepath + "test_stats.pkl", 'rb'))
        
        drift_test, drift_MLP, drift_fore, val_loss, stats, stats_fore, stats_cond_fore, be = val_out
        
        for dim in range(drift_test.shape[-1]):
            ts = np.array(list(range(drift_test.shape[1])))
            test_mean = drift_test.mean(0)[:,dim]
            gen_mean = drift_MLP.mean(0)[:,dim]
            fore_mean = drift_fore.mean(0)[:, dim]
            
            test_std = drift_test.std(0)[:,dim]
            gen_std = drift_MLP.std(0)[:,dim]
            fore_std = drift_fore.std(0)[:,dim]
            
            
            ax = axes[4+idx][dim]
            sns.lineplot(x=ts, y=test_mean, color="black", ax=ax)
            sns.lineplot(x=ts, y=gen_mean, color="red", ax=ax)
            if drift_fore is not None and train_split_t:
                sns.lineplot(x=ts[train_split_t-1:], y=fore_mean, color="green", ax=ax)
            ax.fill_between(ts, test_mean - test_std, test_mean + test_std, color='black', alpha=0.15)
            ax.fill_between(ts, gen_mean - gen_std, gen_mean + gen_std, color='red', alpha=0.15)
            if train_split_t:
                ax.fill_between(ts[train_split_t-1:], fore_mean - fore_std, fore_mean + fore_std, color='green', alpha=0.15)
            
            if train_split_t is not None:
                ax.axvline(x=ts[train_split_t-1], linestyle="--", color="black")

            ylab = ["X", "Y", "Z"]
            ax.set_ylabel(ylab[dim])
            axes[4+idx][0].set_title(methods_label[idx], loc="left")
    
        torch.cuda.empty_cache()
        
fig.legend(["True", "Generated", "Forecast"], 
           loc='lower center', bbox_to_anchor=(0.5,-0.04), 
           ncol=len(["True", "Generated", "Forecast"]), prop={'size': 13},
           bbox_transform=fig.transFigure)
plt.tight_layout()

plt.savefig("{}/Chemo_mean_samples.pdf".format(global_path), bbox_inches="tight")