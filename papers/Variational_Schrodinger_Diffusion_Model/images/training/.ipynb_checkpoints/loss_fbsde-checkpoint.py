# Copyright (c) 2024, Wei Deng. All rights reserved.
#
# This work is licensed under APACHE LICENSE, VERSION 2.0
# You should have received a copy of the license along with this
# work. If not, see https://www.apache.org/licenses/LICENSE-2.0.txt

"""
Loss functions used in ICM2024 paper "Variational Schrödinger Diffusion Models" when training Forward-backward Schrodinger Bridge. 
Created by Wei Deng, Weijian Luo, Yixin Tan, Marin Biloš, Yu Chen, Yuriy Nevmyvaka, Ricky T. Q. Chen
"""

import torch
from torch_utils import persistence

@persistence.persistent_class
class FBSDE_VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class FBSDE_EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss
    
    def forwnet_loss(self, optnet, imptnet, xs, ts, dts, labels=None, augment_pipe=None):

        y, augment_labels =  (xs, torch.zeros(xs.shape[0], 9).to(xs.device))

        ts = ts.view(-1,1,1,1)
        dts = dts.view(-1,1,1,1)

        g_ts = (2*ts)**0.5

        with torch.no_grad():
            zs_impt = (imptnet(y, ts, labels, augment_labels=augment_labels) - y) / ts**2 * g_ts

        y.requires_grad_(True)
        zs = (optnet(y, ts, labels, augment_labels=augment_labels) - y) / ts**2 * g_ts

        gzs = g_ts*zs

        e = sample_gaussian_like(y)
        e_dzdx = torch.autograd.grad(gzs, y, e, create_graph=True, retain_graph=True)[0]
        div_gz = e_dzdx * e

        weight = (ts ** 2 + self.sigma_data ** 2) / self.sigma_data ** 2

        loss = zs*(0.5*zs + zs_impt) + div_gz
        loss = dts*loss
        
        return loss

    def backnet_loss(self, net, xs, zs_impt, ts, labels=None, augment_pipe=None):

        y, augment_labels =  (xs, None)
        y.requires_grad_(True)
        zs = net(y, ts, labels, augment_labels=augment_labels)

        g_ts = 2*ts
        g_ts = g_ts[:,None,None,None]
        gzs = g_ts*zs

        e = sample_gaussian_like(y)
        e_dzdx = torch.autograd.grad(gzs, xs, e, create_graph=True)[0]
        div_gz = e_dzdx * e

        loss = zs*(0.5*zs + zs_impt) + div_gz
        loss = torch.sum(loss * dyn.dt) / xs.shape[0] / ts.shape[0]  # sum over x_dim and T, mean over batch
        return loss