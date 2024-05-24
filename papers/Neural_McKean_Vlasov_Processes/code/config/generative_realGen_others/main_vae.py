import argparse
import copy
import math
import sys
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm

from get_data import *
from utils_gen import *

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
    
    
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.FC_input = nn.Linear(input_dim, hidden_dim*2)
        self.FC_input2 = nn.Linear(hidden_dim*2, hidden_dim)
        self.FC_mean  = nn.Linear(hidden_dim, latent_dim)
        self.FC_var   = nn.Linear (hidden_dim, latent_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        h_       = self.LeakyReLU(self.FC_input(x))
        h_       = self.LeakyReLU(self.FC_input2(h_))
        mean     = self.FC_mean(h_)
        log_var  = self.FC_var(h_)
        return mean, log_var
    
    
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim*2)
        self.FC_output = nn.Linear(hidden_dim*2, output_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        h     = self.LeakyReLU(self.FC_hidden(x))
        h     = self.LeakyReLU(self.FC_hidden2(h))
        
        x_hat = torch.sigmoid(self.FC_output(h))
        return x_hat
        
class vae(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(vae, self).__init__()
        self.Encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.Decoder = Decoder(latent_dim, hidden_dim, input_dim)
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(mean.device)   
        z = mean + var*epsilon                       
        return z
        
    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        x_hat = self.Decoder(z)
        
        return x_hat, mean, log_var
        

def setup_vae(cfg, device, seed):
    train_loader, val_loader, test_loader = load_data(cfg, seed)

    input_dim = train_loader.dataset.tensors[0].shape[-1]
    hidden_dim = cfg["vae"]["num_hidden"]
    latent_dim = cfg["vae"]["latent_dim"]

    model = vae(input_dim = input_dim, hidden_dim = hidden_dim, latent_dim=latent_dim)
    optimizer = optim.AdamW(model.parameters(), lr=cfg["vae"]["lr"], weight_decay=1e-6)
    driftMLP_param = sum(p.numel() for p in model.parameters())
    print(driftMLP_param)
    return model.to(device), train_loader, val_loader, test_loader, optimizer, driftMLP_param

def validate_vae(epoch, model, loader, savepath=None):
    model.eval()
    val_loss = 0
    for batch_idx, data in enumerate(loader):
        data = data[0]
        data = data.to(device).float()
        with torch.no_grad():
            samples, m, v = model(data)
            val_loss += vae_loss(samples, data, m, v)
    if epoch >= 0:
        return val_loss / len(loader.dataset)
    elif epoch == -1:
        test = loader.dataset.tensors[0].to(device).float()
        gen,_,_ = model(test)
        plot_samples(test.detach().cpu().numpy(), gen.detach().cpu().numpy(), savepath=savepath)
        return evaluate(test.detach().cpu().numpy(), gen.detach().cpu().numpy())

def vae_loss(x_hat, x, mean, log_var):
    reproduction_loss = F.mse_loss(x_hat, x, reduction='sum')
    KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD

def train_vae_all_epoch(model, train_loader, val_loader, optimizer, n_epochs, net_savepath):
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
            
            x_hat, mean, logvar = model(data)
            loss = vae_loss(x_hat, data, mean, logvar)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            pbar.update(data.size(0))
            pbar.set_description('Train, Log likelihood in nats: {:.6f}'.format(-train_loss / (batch_idx + 1)))
        pbar.close()
                
        validation_loss = validate_vae(epoch, model, val_loader)
        
        if epoch - best_validation_epoch >= 30:
            break

        if validation_loss < best_validation_loss:
        # No Model Saving
            best_validation_epoch = epoch
            best_validation_loss = validation_loss
            best_model = copy.deepcopy(model)
            
            torch.save(best_model.state_dict(), net_savepath + "/{}.pt".format("vae"))

        print(
            'Best validation at epoch {}: Average Log Likelihood in nats: {:.4f}'.
            format(best_validation_epoch, -best_validation_loss))
        
n_runs = cfg['n_runs']
n_epochs = cfg['n_epochs']

for run in range(n_runs):
    model, train_loader, val_loader, test_loader, optimizer, driftMLP_param = setup_vae(cfg, device, run)
    
    savepath, net_savepath = format_directory(cfg, experiment, run)
    make_directory(savepath, net_savepath)
    train_vae_all_epoch(model, train_loader, val_loader, optimizer, n_epochs, net_savepath)
    model.load_state_dict(torch.load(net_savepath +"/{}.pt".format("vae")))
    
    model.eval()
    validation_loss = validate_vae(-1, model, test_loader, savepath)
    
    with open(os.path.join(savepath,'test_stats.pkl'), 'wb') as f:
        pickle.dump((validation_loss, driftMLP_param), f)
        f.close()
    
    
    
