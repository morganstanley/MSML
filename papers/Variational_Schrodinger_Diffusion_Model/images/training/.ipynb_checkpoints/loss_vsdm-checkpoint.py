# Copyright (c) 2024, Wei Deng. All rights reserved.
#
# This work is licensed under APACHE LICENSE, VERSION 2.0
# You should have received a copy of the license along with this
# work. If not, see https://www.apache.org/licenses/LICENSE-2.0.txt

"""
Loss functions used in ICM2024 paper "Variational Schrödinger Diffusion Models" when training variational schrodinger diffusion models. 
Created by Wei Deng, Weijian Luo, Yixin Tan, Marin Biloš, Yu Chen, Yuriy Nevmyvaka, Ricky T. Q. Chen
"""

import torch
from torch_utils import persistence

@persistence.persistent_class
class VSDM_EDMLoss:
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

    def vsdm_back_loss(self, net, fnet, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)

        n = torch.randn_like(y)*sigma
        w = fnet.model.diaglinear.weight.data**2
        mu = (-w*sigma).exp()*y
        L = 1/2**0.5*(1-(-2*w*sigma).exp()).sqrt()/(w*sigma)**0.5

        D_yn = net(mu + L*n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------

def sample_rademacher_like(y):
    return torch.randint(low=0, high=2, size=y.shape).to(y) * 2 - 1

def sample_gaussian_like(y):
    return torch.randn_like(y)
