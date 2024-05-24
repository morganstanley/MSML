import argparse
import os
import numpy as np
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
import yaml
from get_data import *
from utils_gen import *

import copy


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

class Generator(nn.Module):
    def __init__(self, latent_dim, output_size):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 64, normalize=False),
            *block(64, 128),
            *block(128, 256),
            nn.Linear(256, output_size),
            nn.Tanh()
        )

    def forward(self, z):
        samples = self.model(z)
        return samples
    
    def sample(self, num_samples, device):
        z = torch.randn(num_samples, self.latent_dim).to(device)
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )
    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity
    
def setup_wgan(cfg, device, seed):
    train_loader, val_loader, test_loader = load_data(cfg, seed)

    num_inputs = train_loader.dataset.tensors[0].shape[-1]

    generator = Generator(50, output_size = num_inputs)
    discriminator = Discriminator(input_size = num_inputs)
                        
    optimizer_G = optim.AdamW(generator.parameters(), lr=cfg["generator"]["lr"], weight_decay=1e-6)
    optimizer_D = optim.AdamW(discriminator.parameters(), lr=cfg["discriminator"]["lr"], weight_decay=1e-6)
    driftMLP_param = sum(p.numel() for p in generator.parameters())
    driftMLP_param += sum(p.numel() for p in discriminator.parameters())
    print(driftMLP_param)
    return generator.to(device), discriminator.to(device), train_loader, val_loader, test_loader, optimizer_G, optimizer_D, driftMLP_param

lambda_gp = 2

def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.from_numpy(np.random.random((real_samples.size(0), 1))).to(real_samples.device).float()
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(torch.FloatTensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False).to(real_samples.device).float()
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def validate_wgan(epoch, generator, discriminator, loader, savepath=None):
    generator.eval()
    discriminator.eval()
    val_loss = 0
    for batch_idx, data in enumerate(loader):
        # Generate a batch of images
        data = data[0].to(device).float()
        real_imgs = data
        fake_imgs = generator.sample(data.shape[0], data.device)

        real_validity = discriminator(real_imgs)
        fake_validity = discriminator(fake_imgs)

        gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)
        val_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
    if epoch >= 0:
        return val_loss / len(loader.dataset)
    elif epoch == -1:
        test = loader.dataset.tensors[0].detach().cpu().numpy()
        gen = generator.sample(test.shape[0], data.device).detach().cpu().numpy()
        plot_samples(test, gen, savepath=savepath)
        return evaluate(test, gen)


def train_wgan_all_epochs(n_epochs, n_critic, generator, discriminator, train_loader, val_loader, optimizer_G, optimizer_D):
    best_validation_loss = float('inf')
    best_validation_epoch = 0
    best_generator = generator
    best_discriminator = discriminator
    
    for epoch in range(n_epochs):
        for i, data in enumerate(train_loader):
            data = data[0].to(device).float()
            real_imgs = data
            optimizer_D.zero_grad()
            
            fake_imgs = generator.sample(data.shape[0], data.device)

            real_validity = discriminator(real_imgs)
            fake_validity = discriminator(fake_imgs)
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

            d_loss.backward()
            optimizer_D.step()
            optimizer_G.zero_grad()

            # Train the generator every n_critic steps
            if i % n_critic == 0:
                fake_imgs = generator.sample(data.shape[0], data.device)
                fake_validity = discriminator(fake_imgs)
                g_loss = -torch.mean(fake_validity)

                g_loss.backward()
                optimizer_G.step()
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, n_epochs, i, len(train_loader), d_loss.item(), g_loss.item())
                )
        validation_loss = validate_wgan(epoch, generator, discriminator, val_loader)
    
        if epoch - best_validation_epoch >= 30:
            break

        if validation_loss < best_validation_loss:
        # No Model Saving
            best_validation_epoch = epoch
            best_validation_loss = validation_loss
            best_generator = copy.deepcopy(generator)
            best_discriminator = copy.deepcopy(discriminator)
            
            torch.save(best_generator.state_dict(), net_savepath + "/{}.pt".format("generator"))
            torch.save(best_discriminator.state_dict(), net_savepath + "/{}.pt".format("discriminator"))
                
                
                
                
n_runs = cfg['n_runs']
n_epochs = cfg['n_epochs']
try:
    n_critic = cfg['n_critic']
except KeyError:
    n_critic = 5
    
    
for run in range(n_runs):
    generator, discriminator, train_loader, val_loader, test_loader, optimizer_G, optimizer_D, driftMLP_param = setup_wgan(cfg, device, run)
    
    savepath, net_savepath = format_directory(cfg, experiment, run)
    make_directory(savepath, net_savepath)
    train_wgan_all_epochs(n_epochs, n_critic, generator, discriminator, train_loader, val_loader, optimizer_G, optimizer_D)
    generator.load_state_dict(torch.load(net_savepath +"/{}.pt".format("generator")))
    discriminator.load_state_dict(torch.load(net_savepath +"/{}.pt".format("discriminator")))
    
    generator.eval()
    discriminator.eval()
    validation_loss = validate_wgan(-1, generator, discriminator, test_loader, savepath)
    
    with open(os.path.join(savepath,'test_stats.pkl'), 'wb') as f:
        pickle.dump((validation_loss, driftMLP_param), f)
        f.close()

                
    