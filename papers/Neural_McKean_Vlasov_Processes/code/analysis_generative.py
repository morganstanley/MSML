import pickle
import numpy as np
import matplotlib.pyplot as plt
import yaml
from train import format_directory
from MeanFieldMLP import *
from MLP import *
import warnings
import seaborn as sns
import pandas as pd
from utils import *
from setup import *
warnings.filterwarnings("ignore")

global_path = "Input Global Path"

def analyze_generative(experiment, yaml_file, runs=5, by_time=False, return_GIR = False, ELBO_plot=False):
    result_path = []
    for run in range(runs):
        yaml_filepath = experiment + yaml_file
        with open(yaml_filepath, 'r') as f:
            cfg = yaml.load(f, yaml.SafeLoader)
        savepath, plot_savepath, net_savepath = format_directory(experiment, cfg, run)
        result_path.append(savepath)
        
    ENERGY = []
    GIR = []
    ENERGY_by_t = []
    best_run = []
    temp = []
    
    for run, result in enumerate(result_path):
        stats_test = open(result + 'test_stats.pkl', 'rb')
        stats_dict_test = pickle.load(stats_test)
        best_run.append(stats_dict_test[-1])
        stats_test = evaluation(test = stats_dict_test[0], gen=stats_dict_test[1], rna_marginal=None)
        GIR.append(stats_dict_test[5])
        ENERGY.append(stats_test["energy_distance"])
        ENERGY_by_t.append(stats_test["energy_distance_by_time"])
        
    if ELBO_plot:
        return np.array(temp)
    if by_time and return_GIR == False:
        return np.array(ENERGY_by_t)
    elif by_time == False and return_GIR == False:
        return np.array(ENERGY)
    elif by_time == False and return_GIR == True:
        return np.array(GIR)
        
######################################### Analysis on Eight Gauss ###########################################

experiment = ["generative_100_10_bridge_eightgauss/d=2/", "generative_100_10_bridge_eightgauss/d=10/", 
              "generative_100_10_bridge_eightgauss/d=30/", 
              "generative_100_10_bridge_eightgauss/d=50/", "generative_100_10_bridge_eightgauss/d=100/"]
methods     =["MLP_generative_gir", "W0_generative_gir", "NF_generative_gir", "Xt_generative_gir"]
method_name = ["MLP", "IM", "ML", "EM"]
datatype = ["D=2", "D=10", "D=30", "D=50", "D=100"]
samples = dict()
x = [2, 10, 30, 50, 100]
for j, met in enumerate(methods):
    samples[method_name[j]] = {}
    for i, e in enumerate(experiment):
        sampled = analyze_generative("{}/{}".format(global_path,e), '{}.yaml'.format(met), by_time=False, runs=10, 
                                     return_GIR=True)
        samp_flat = sampled.flatten().tolist()
        samples[method_name[j]][datatype[i]] = samp_flat

fig, ax = plt.subplots(figsize=(7, 6), ncols=1, nrows=1, dpi=100)
for i,d in enumerate(method_name):
    df = pd.DataFrame(samples[method_name[i]]).values
    ax.errorbar(x=[2, 10, 30, 50, 100], y=df.mean(0)/x, yerr=(df/x).std(0), fmt='-o', linewidth=4, markersize=10)

ax.set_xticks([2, 10, 30, 50, 100])
ax.set_xticklabels([2, 10, 30, 50, 100])
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_xlabel("Dimension", fontsize=25)
ax.set_ylabel("ELBO", fontsize=25)
ax.legend(method_name, fontsize=25)
    
plt.tight_layout()
plt.savefig("generative_100_10_bridge_eightgauss/eightgauss_increase_d.pdf", bbox_inches="tight")

from plot import *
experiments = ["eightgauss"]
methods     = ["MLP", "Xt", "W0", "NF"]
methods_label     = ["MLP", "IM", "ML", "EM"]
indices = np.random.choice(list(range(0,100)), 10, True).reshape(5,2)
device="cuda:0"
for exp in experiments:
    fig, ax = plt.subplots(figsize=(3.3*5,3.0*5), dpi=200, nrows=5, ncols=5)
    for idx, met in enumerate(methods):
        result_path = []
        experiment = "generative_100_10_bridge_eightgauss/d=100/"
        yaml_file = "{}_generative_gir.yaml".format(met)
        yaml_filepath = experiment + yaml_file
        print(yaml_filepath)
        en_all = []
        for run in range(3,4):
            initialized, test_loader, head = setup(yaml_filepath, device, seed=run)
            driftMLP_best = initialized["driftMLP"]
            NF_best = initialized["NF"]
            
            with open(yaml_filepath, 'r') as f:
                cfg = yaml.load(f, yaml.SafeLoader)
            savepath, plot_savepath, net_savepath = format_directory(experiment, cfg, run)

            xs = test_loader.dataset.tensors[0].detach().cpu().numpy()
            test_stats = pickle.load(open(savepath + "test_stats.pkl", 'rb'))
            
            gen_path_simu, test_path, drift_test, drift_MLP, drift_fore, _, _, _, _,_ = test_stats
            ts = initialized['path_loader'].dataset.tensors[1].detach().cpu().numpy()[0]
            if exp != "3d-s":
                if exp == "pinwheel":
                    plot_scale=2.1
                elif exp == "swissroll":
                    plot_scale=2.3
                else:
                    plot_scale=3
                
                for plot_col, dimension in enumerate(indices):
                    ax[idx][plot_col] = plot_gradient(ax = ax[idx][plot_col], xs=xs, driftMLP=driftMLP_best, 
                                                ts=ts, device=initialized["device"],
                                generative=True, samples=gen_path_simu[:,-1,dimension],
                                                plot_savepath = savepath, snap_time=xs.shape[1]-1, 
                                                data_params = initialized["data_param"],
                                                plot_scale=plot_scale, test=True, head=head, truth=False, 
                                                train_particle_labels=None, plot_kde=True)
                    ax[idx][plot_col].set_title(methods_label[idx], loc="left")
                    ax[idx][plot_col].set_xlabel("Dimension " + str(dimension[0]))
                    ax[idx][plot_col].set_ylabel("Dimension " + str(dimension[1]))
            else:
                x, y, z = gen_path_simu[:,-1,:].T
                ax_temp = fig.add_subplot(1, 4, idx+1, projection='3d')
                ax_temp.scatter(x, y, z)
                
                ax_temp.view_init(azim=-60, elev=9)
                ax_temp.xaxis.set_major_locator(ticker.MultipleLocator(1))
                ax_temp.yaxis.set_major_locator(ticker.MultipleLocator(1))
                ax_temp.zaxis.set_major_locator(ticker.MultipleLocator(1))
                ax_temp.set_title(methods_label[idx], loc="left")
            
        
            torch.cuda.empty_cache()
    for plot_col, dimension in enumerate(indices):
        ax[-1][plot_col] = plot_gradient(ax = ax[-1][plot_col], xs=xs, driftMLP=driftMLP_best, 
                                            ts=ts, device=initialized["device"],
                            generative=True, samples=xs[:,-1,dimension],
                                            plot_savepath = savepath, snap_time=xs.shape[1]-1, 
                                            data_params = initialized["data_param"],
                                            plot_scale=plot_scale, test=True, head=head, truth=True, 
                                            train_particle_labels=None, plot_kde=True)
        ax[-1][plot_col].set_title("True", loc="left")
        ax[-1][plot_col].set_xlabel("Dimension " + str(dimension[0]))
        ax[-1][plot_col].set_ylabel("Dimension " + str(dimension[1]))
        
    plt.tight_layout()

    plt.savefig("generative_100_10_bridge_eightgauss/eightgauss_rd_dim.pdf", bbox_inches="tight")


    
######################################### Analysis on 5 Real Generative Data ###########################################
from sim_process import make_data_T
experiment = ["generative_100_30_bridge_realGen1_tanh/power/", 
              "generative_100_30_bridge_realGen1_tanh/miniboone/", 
              "generative_100_30_bridge_realGen1_tanh/hepmass/", 
              "generative_100_30_bridge_realGen1_tanh/gas/", 
              "generative_100_30_bridge_realGen1_tanh/cortex/"]

methods     =["MLP_generative_gir", "W0_generative_gir", "NF_generative_gir", "Xt_generative_gir"]
method_name = ["MLP", r"$X_t$", r"$W_0$", r"$\hat{P}_t$"]
datatype = ["Power", "Miniboone", "Hepmass", "Gas", "Cortex"]
samples = dict()
for i, e in enumerate(experiment):
    print(e)
    samples[datatype[i]] = {"method":[], "data":[]}
    for j, met in enumerate(methods):
        sampled = analyze_generative("{}/{}".format(global_path, e), '{}.yaml'.format(met), by_time=False, runs=10, return_GIR=False)
        samp_flat = sampled.flatten().tolist()
        samples[datatype[i]]["data"] += samp_flat
        samples[datatype[i]]["method"] += [method_name[j]]*len(samp_flat)
    
for i,d in enumerate(datatype):
    for j, m in enumerate(method_name):
        print(np.round(np.mean(np.array(samples[datatype[i]]['data']).reshape(4,10), 1)[j], 3), 
              "(" + str(np.round(np.std(np.array(samples[datatype[i]]['data']).reshape(4,10), 1)[j], 3)) + ")")
    print("")
    
    
######################################### Analysis on OU with Jumps ###########################################
experiment = ["OU_jump/1_jumps/", "OU_jump/2_jumps/", "OU_jump/4_jumps/"]
methods     =["MLP", "W0",
              "NF", "Xt"]
method_name = ["MLP", "IM", "ML", "EM"]
datatype = ["Jumps=1", "Jumps=2", "Jumps=4"]
samples = dict()
for j, met in enumerate(methods):
    samples[method_name[j]] = {}
    for i, e in enumerate(experiment):
        sampled = analyze_generative("{}".format(e), '{}.yaml'.format(met), by_time=False, runs=10, 
                                     return_GIR=False)
        samp_flat = sampled.flatten().tolist()
        samples[method_name[j]][datatype[i]] = samp_flat
        
        
fig, ax = plt.subplots(figsize=(7, 6), ncols=1, nrows=1, dpi=100)
import matplotlib as mpl

for i,d in enumerate(method_name):
    df = pd.DataFrame(samples[method_name[i]]).values
    ax.errorbar([1,2,4], df.mean(0), yerr=df.std(0), marker='o', linewidth=4, markersize=10)
ax.set_xticks([1,2,4])
ax.set_xticklabels([1,2,4])
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_xlabel("Number of Jumps", fontsize=25)
ax.set_ylabel("Energy Distance", fontsize=25)
ax.legend(method_name, fontsize=25)
    
plt.tight_layout()
plt.savefig("OU_jump/jump_result.pdf", bbox_inches="tight")

methods = ["MLP", "W0", "NF", "Xt"]
path = ""
methods_label = ["MLP", "IM", "ML", "EM", "True"]
exp = ["OU_jump/1_jumps/", "OU_jump/2_jumps/", "OU_jump/4_jumps/"]
exp_label = ["OU with 1 Jump", "OU with 2 Jump", "OU with 4 Jump"]
device="cuda:0"

fig, axes = plt.subplots(figsize=(2.5*3,1.8), ncols=3, nrows=1, dpi=200)

for i,e in enumerate(exp):
    ax = axes[i]
    for idx, met in enumerate(methods):
        result_path = []
        experiment = global_path + "{}".format(e)
        yaml_file = "{}.yaml".format(met)
        yaml_filepath = experiment + yaml_file
        print(yaml_filepath)
        en_all = []
        for run in range(6,7):
            with open(yaml_filepath, 'r') as f:
                cfg = yaml.load(f, yaml.SafeLoader)
            savepath, plot_savepath, net_savepath = format_directory(experiment, cfg, run)

            test_stats = pickle.load(open(savepath + "test_stats.pkl", 'rb'))
            
            gen_path_simu, test_path, drift_test, drift_MLP, drift_fore, _, _, _, _,_ = test_stats
            
            dim = 0
            ts = np.array(range(test_path.shape[1]))*5/test_path.shape[1]
            
            
            test_mean = test_path.mean(0)[:,dim]
            gen_mean = gen_path_simu.mean(0)[:,dim]
            
            sns.lineplot(x=ts, y=gen_mean, ax=ax)
    ax.set_title(exp_label[i], loc="left", fontsize=10)
    sns.lineplot(x=ts, y=test_mean, color="black", ax=ax)
    
fig.legend(methods_label, loc='lower center', bbox_to_anchor=(0.5,-0.13), ncol=len(methods_label), prop={'size': 9},
            bbox_transform=fig.transFigure)
plt.tight_layout()
plt.savefig("{}/jump_samples.pdf".format(global_path), bbox_inches="tight")

