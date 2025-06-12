import torch
import torch.nn as nn
from models.utils import *

def build_toy(data_dim, zero_out_last_layer,device, num_ResNet=2):
    return ToyPolicy(data_dim, zero_out_last_layer=zero_out_last_layer,device=device, num_ResNet=num_ResNet)

class ResNet_FCx(nn.Module):
    def __init__(self, data_dim, hidden_dim, num_res_blocks,device):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.map=nn.Linear(data_dim, hidden_dim)
        self.res_blocks = nn.ModuleList(
            [self.build_res_block() for _ in range(num_res_blocks)])
        self.const=torch.sqrt(torch.tensor([2]).to(device))
    def build_linear(self, in_features, out_features):
        linear = nn.Linear(in_features, out_features)
        return linear

    def build_res_block(self):
        hid = self.hidden_dim
        layers = []
        widths =[hid]*4
        for i in range(len(widths) - 1):
            layers.append(self.build_linear(widths[i], widths[i + 1]))
            layers.append(SiLU())
        return nn.Sequential(*layers)

    def forward(self, x):
        h=self.map(x)
        for res_block in self.res_blocks:
            h = (h + res_block(h)) / self.const
        return h

class ToyYtImpl(torch.nn.Module):
    def __init__(self, data_dim: int, time_embed_dim: int, hid: int, zero_out_last_layer: bool=False, device: str=None, num_ResNet=2):
        super().__init__()
        self.data_dim=data_dim
        self.t_module = nn.Sequential(
            nn.Linear(time_embed_dim, hid),
            SiLU(),
            nn.Linear(hid, hid),
            SiLU(),
            nn.Linear(hid, hid),
        )
    
        self.x_module = ResNet_FCx(int(data_dim/2), hid, num_res_blocks=num_ResNet,device=device)
        self.v_module = ResNet_FCx(int(data_dim/2), hid, num_res_blocks=num_ResNet,device=device)
        
        self.out_module = nn.Sequential(
            nn.Linear(hid*3 , hid),
            SiLU(),
            nn.Linear(hid , hid),
            SiLU(),
            nn.Linear(hid, int(data_dim/2)),
        )

        if zero_out_last_layer:
            self.out_module[-1] = zero_module(self.out_module[-1])

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        xx = x[:,0:int(self.data_dim/2)]
        vv = x[:,int(self.data_dim/2):]
        t_out = self.t_module(t_emb)
        x_out = self.x_module(xx)
        v_out = self.v_module(vv)
        out   = self.out_module(torch.cat([x_out, v_out, t_out], dim=-1))
        return out


class ToyPolicy(torch.nn.Module):
    def __init__(self, data_dim=4, hidden_dim=256, time_embed_dim=256, zero_out_last_layer=False,device=None, num_ResNet=2):
        super(ToyPolicy,self).__init__()
        self.time_embed_dim = time_embed_dim
        self.zero_out_last_layer = zero_out_last_layer
        hid = hidden_dim
        self.net = ToyYtImpl(data_dim, time_embed_dim, hid, zero_out_last_layer,device=device, num_ResNet=num_ResNet)
        #self.net = torch.jit.script(self.net)

    def forward(self, x, t):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        """

        # make sure t.shape = [T]
        if len(t.shape)==0:
            t = t[None]
        t_emb = timestep_embedding(t, self.time_embed_dim)

        out = self.net(x, t_emb)
        return out
