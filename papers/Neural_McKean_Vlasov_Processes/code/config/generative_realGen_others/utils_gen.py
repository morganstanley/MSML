import os
import numpy as np
import dcor
import matplotlib.pyplot as plt


def format_directory(cfg, experiment, run):
    model_type =  cfg["model"]
    if model_type == "maf":
        savepath = os.path.join(str(experiment),"{}_{}_{}/run_{}".format(model_type, cfg["maf"]["num_blocks"], cfg["maf"]["num_hidden"], run))
    else:
        savepath = os.path.join(str(experiment),"{}/run_{}".format(model_type, run))
    net_savepath = "/scratch/hy190/MV-SDE/{}/saved_nets".format(savepath)
    return savepath, net_savepath

def make_directory(savepath, net_savepath):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    if not os.path.exists(net_savepath):
        os.makedirs(net_savepath)

def evaluate(test, gen):
    ed = np.sqrt(dcor.energy_distance(test, gen))
    return ed

def plot_samples(test, gen, savepath):
    plt.scatter(test[:,0], test[:,1], label="Test")
    plt.scatter(gen[:,0], gen[:,1], label="Generated")
    plt.legend()
    plt.savefig(savepath + "/first_2_dim_scatter.pdf", bbox_inches="tight")
    plt.close()