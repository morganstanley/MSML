import torch
import torch.nn as nn
from models.utils import *
from sde import compute_vp_diffusion


def create_orthogonal_layer(data_dim):
    linear_layer = nn.Linear(data_dim, data_dim, bias=False)
    linear_layer.weight.data = torch.eye(data_dim)
    return nn.utils.parametrizations.orthogonal(linear_layer) 

class LinearPolicy(torch.nn.Module):
    def __init__(self, data_dim=2, hidden_dim=64, time_embed_dim=128, net_name='Linear'):
        super(LinearPolicy,self).__init__()

        self.data_dim = data_dim
        self.net_name = net_name
        self.time_embed_dim = time_embed_dim

        self.Sigma = nn.Parameter(torch.zeros(data_dim))
        self.U = create_orthogonal_layer(data_dim)
        self.V = create_orthogonal_layer(data_dim)

        if 'static' not in self.net_name.lower():
            self.t_module = nn.Sequential(
                nn.Linear(self.time_embed_dim, hidden_dim),
                SiLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Linear(hidden_dim, 1, bias=False),
            )
            self.t_module[-1] = zero_module(self.t_module[-1])


    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype


    @property
    def A(self):
        Sigma_mat = torch.diag(self.Sigma)
        return self.U.weight @ Sigma_mat @ self.V.weight.T


    def At(self, t):
        if 'static' not in self.net_name.lower():
            t_emb = timestep_embedding(t, self.time_embed_dim)
            t_map = self.t_module(t_emb)
            self.t_out = torch.exp(t_map)[:, 0]
        else:
            self.t_out = torch.ones(t.shape[0])
        return torch.einsum('ij,t->tij', self.A, self.t_out)

    def forward(self, x, t, beta_min=0.1, beta_max=10., beta_r=1., interval=100.):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        """
        # make sure t.shape = [T]
        if len(t.shape) == 0:
            t = t[None]

        if 'static' not in self.net_name.lower():
            out = torch.einsum('bij,bj->bi', self.At(t), x)
        else:
            out = torch.einsum('ij,bj->bi', self.A, x)
        # include compute_vp_diffusion to adapt to Tianrong's framework
        out = compute_vp_diffusion(t, b_min=beta_min, b_max=beta_max, b_r=beta_r, T=interval).unsqueeze(dim=1) * out

        return out
