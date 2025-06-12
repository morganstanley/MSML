import torch
import torch.nn as nn
from models.utils import *

def build_toy(zero_out_last_layer):
    return ToyPolicy(zero_out_last_layer=zero_out_last_layer)

class ToyPolicy(torch.nn.Module):
    def __init__(self, data_dim=2, hidden_dim=256, time_embed_dim=128, zero_out_last_layer=False):
        super(ToyPolicy,self).__init__()

        self.time_embed_dim = time_embed_dim
        self.zero_out_last_layer = zero_out_last_layer
        hid = hidden_dim

        self.t_module = nn.Sequential(
            nn.Linear(self.time_embed_dim, hid),
            SiLU(),
            nn.Linear(hid, hid),
        )

        self.x_module = ResNet_FC(data_dim, hidden_dim, num_res_blocks=3)

        self.out_module = nn.Sequential(
            nn.Linear(hid,hid),
            SiLU(),
            nn.Linear(hid, data_dim),
        )
        if zero_out_last_layer:
            self.out_module[-1] = zero_module(self.out_module[-1])

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype

    def forward(self,x, t):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        """

        # make sure t.shape = [T]
        if len(t.shape)==0:
            t=t[None]

        t_emb = timestep_embedding(t, self.time_embed_dim)
        t_out = self.t_module(t_emb)
        x_out = self.x_module(x)
        out   = self.out_module(x_out+t_out)

        return out