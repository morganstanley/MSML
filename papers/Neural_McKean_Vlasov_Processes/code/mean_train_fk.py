import torch
import torch.nn as nn

from torchvision.datasets import USPS
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from pytorch_lightning import Trainer, seed_everything

from mf_fk import FKModule
from setup import *
from MeanFieldMLP import MF
from train import format_directory, make_directory, get_parser


args = get_parser().parse_args()

if args.device is None:
    device = 'cuda:0'
else: 
    device = args.device

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
    
for run in range(n_runs):
    global savepath
    global plot_savepath
    global net_savepath
    
    seed_everything(run) 
    savepath, plot_savepath, net_savepath = format_directory(experiment_folder, cfg, run)
    make_directory(savepath, plot_savepath, net_savepath)
    initialized, test_loader,head = setup(yaml_filepath, device, seed=run, FK=True)
    trainer_params = {
        'accelerator': "gpu",
        'devices': [int(device[-1])],
        'check_val_every_n_epoch' : 20,
        'max_epochs' : initialized["n_epochs"],
        'checkpoint_callback' : False,
        'logger' : False
    }  

    mu = initialized["driftMLP"]
    if cfg["MLP"] is not None:
        lr = cfg["MLP"]["lr"]
    elif cfg["MF"] is not None:
        lr = cfg["MF"]["lr"]
        
    # boundary condition
    d = test_loader.dataset.tensors[0].shape[-1]
    mn = torch.distributions.multivariate_normal.MultivariateNormal(torch.ones(d).to(device), 0.5*torch.eye(d).to(device))
    g = lambda x: mn.log_prob(x).exp()

    fkm = FKModule(savepath, plot_savepath, net_savepath, 
                   test_loader.dataset.tensors[0], initialized["validation_loader"].dataset.tensors[0], 
                   device, d, mu, g, lr_mu=lr, noise='diag').to(device)

    trainer = Trainer(**trainer_params)    
    trainer.fit(fkm, initialized["path_loader"], initialized["validation_loader"])
    trainer.test(fkm, test_loader)
    trainer.save_checkpoint(net_savepath + "/best_model.ckpt")
    