import math

import torch
import torch.nn as nn
from models.utils import *
from sde_cld import compute_diffusion


class LinearSubPolicy(torch.nn.Module):
    def __init__(self, data_dim=2, hidden_dim=64, time_embed_dim=128, net_name='Linear'):
        super(LinearSubPolicy,self).__init__()

        self.data_dim = data_dim
        self.net_name = net_name
        self.Sigma = nn.Parameter(torch.zeros(data_dim))

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype

    @property
    def A(self):
        Sigma_mat = torch.diag(self.Sigma)
        return Sigma_mat 

    def At(self, t):
        self.t_out = torch.ones(t.shape[0])
        return torch.einsum('ij,t->tij', self.A, self.t_out)

    def forward(self, x, t, beta_min=0.1, beta_max=10., beta_r=1., interval=100.):
        # make sure t.shape = [T]
        if len(t.shape) == 0:
            t = t[None]

        out = torch.einsum('bij,bj->bi', self.At(t), x)
        out = compute_diffusion(t, b_min=beta_min, b_max=beta_max, b_r=beta_r, T=interval).unsqueeze(dim=1) * out
        return out


def damping_transform(x, gamma, damp_ratio=1.0): # =1: critical-damping; <1 means under-damping
    return 0.5 - math.sqrt(damp_ratio) * torch.sqrt(1. - 2 * gamma * x) / gamma

class LinearPolicy(torch.nn.Module):
    def __init__(self, data_dim=4, hidden_dim=64, time_embed_dim=128, net_name='Linear', gamma=0., damp_ratio=1.0):
        super(LinearPolicy,self).__init__()
        self.gamma = gamma
        self.damp_ratio = damp_ratio
        self.x_net = LinearSubPolicy(data_dim//2, hidden_dim, time_embed_dim, net_name)
        self.v_net = LinearSubPolicy(data_dim//2, hidden_dim, time_embed_dim, net_name)

    def forward(self, a, t, beta_min=0.1, beta_max=10., beta_r=1., interval=100., baseline=False):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        """

        # make sure t.shape = [T]
        if len(t.shape) == 0:
            t = t[None]
        x, v = torch.chunk(a, 2, dim=-1)
        if not baseline:
            self.v_net.Sigma.data = damping_transform(self.x_net.Sigma.data, self.gamma, self.damp_ratio)
        x_out = self.x_net(x, t, beta_min, beta_max, beta_r, interval)
        v_out = self.v_net(v, t, beta_min, beta_max, beta_r, interval)
        return x_out + v_out
