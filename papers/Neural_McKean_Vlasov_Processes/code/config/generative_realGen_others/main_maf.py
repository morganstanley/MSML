import argparse
import copy
import math
import sys
import os

#import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm
from get_data import *
from utils_gen import *
import flows as fnn
import yaml
parser = argparse.ArgumentParser()

parser.add_argument("--f", help="yaml file path")
parser.add_argument("--d", default='cuda:0', help="device")
parser.add_argument("--e", help="experiment directory")
args = parser.parse_args()
yaml_filepath = args.f
device = args.d
experiment = args.e
    
with open(yaml_filepath, 'r') as f:
    cfg = yaml.load(f, yaml.SafeLoader)

print("here")
def setup_maf(cfg, device, seed):
    train_loader, val_loader, test_loader = load_data(cfg, seed)

    num_inputs = train_loader.dataset.tensors[0].shape[-1]
    num_hidden = cfg["maf"]["num_hidden"]
    num_blocks = cfg["maf"]["num_blocks"]

    act = 'tanh' if cfg['dataset']['dataset_type'] == 'ethylene_CO' else 'relu'

    modules = []
    for _ in range(num_blocks):
        modules += [
                fnn.MADE(num_inputs, num_hidden, 0, act=act),
                fnn.BatchNormFlow(num_inputs),
                fnn.Reverse(num_inputs)
                ]
    
    #modules = []
    #mask = torch.arange(0, num_inputs) % 2
    #mask = mask.to(device).float()
    #
    #for _ in range(num_blocks):
    #    modules += [
    #        fnn.CouplingLayer(
    #            num_inputs, num_hidden, mask, 0,
    #            s_act='tanh', t_act='relu'),
    #        fnn.BatchNormFlow(num_inputs)
    #    ]
    #    mask = 1 - mask

    model = fnn.FlowSequential(*modules)
    model.num_inputs = num_inputs
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.fill_(0)
                        
    optimizer = optim.AdamW(model.parameters(), lr=cfg["maf"]["lr"], weight_decay=1e-6)
    driftMLP_param = sum(p.numel() for p in model.parameters())
    print(driftMLP_param)
    return model.to(device), train_loader, val_loader, test_loader, optimizer, driftMLP_param

def validate_maf(epoch, model, loader, savepath=None):
    model.eval()
    val_loss = 0
    for batch_idx, data in enumerate(loader):
        data = data[0]
        data = data.to(device).float()
        with torch.no_grad():
            val_loss += -model.log_probs(data, None).sum().item()  # sum up batch loss
    if epoch >= 0:
        return val_loss / len(loader.dataset)
    elif epoch == -1:
        test = loader.dataset.tensors[0].detach().cpu().numpy()
        gen = model.sample(test.shape[0]).detach().cpu().numpy()
        plot_samples(test, gen, savepath=savepath)
        return evaluate(test, gen)


def train_maf_all_epoch(model, train_loader, val_loader, optimizer, n_epochs, net_savepath):
    best_validation_loss = float('inf')
    best_validation_epoch = 0
    best_model = model
    
    for epoch in range(n_epochs):
        
        print('\nEpoch: {}'.format(epoch))
        model.train()
        train_loss = 0
        pbar = tqdm(total=len(train_loader.dataset))
        for batch_idx, data in enumerate(train_loader):
            data = data[0]
            data = data.to(device).float()
            optimizer.zero_grad()
            loss = -model.log_probs(data, None).mean()
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            pbar.update(data.size(0))
            pbar.set_description('Train, Log likelihood in nats: {:.6f}'.format(-train_loss / (batch_idx + 1)))
        pbar.close()
        
        for module in model.modules():
            if isinstance(module, fnn.BatchNormFlow):
                module.momentum = 0
                
        with torch.no_grad():
            model(train_loader.dataset.tensors[0].to(data.device).float())
        for module in model.modules():
            if isinstance(module, fnn.BatchNormFlow):
                module.momentum = 1
                
        validation_loss = validate_maf(epoch, model, val_loader)
        
        if epoch - best_validation_epoch >= 30:
            break

        if validation_loss < best_validation_loss:
        # No Model Saving
            best_validation_epoch = epoch
            best_validation_loss = validation_loss
            best_model = copy.deepcopy(model)
            
            torch.save(best_model.state_dict(), net_savepath + "/{}.pt".format("maf"))

        print(
            'Best validation at epoch {}: Average Log Likelihood in nats: {:.4f}'.
            format(best_validation_epoch, -best_validation_loss))
        
n_runs = 1 #cfg['n_runs']
n_epochs = 1 #cfg['n_epochs']

for run in range(n_runs):
    model, train_loader, val_loader, test_loader, optimizer, driftMLP_param = setup_maf(cfg, device, run)
    
    savepath, net_savepath = format_directory(cfg, experiment, run)
    make_directory(savepath, net_savepath)
    train_maf_all_epoch(model, train_loader, val_loader, optimizer, n_epochs, net_savepath)
    model.load_state_dict(torch.load(net_savepath +"/{}.pt".format("maf")))
    
    model.eval()
    validation_loss = validate_maf(-1, model, test_loader, savepath)
    
    with open(os.path.join(savepath,'test_stats.pkl'), 'wb') as f:
        pickle.dump((validation_loss, driftMLP_param), f)
        f.close()
    
    
    
