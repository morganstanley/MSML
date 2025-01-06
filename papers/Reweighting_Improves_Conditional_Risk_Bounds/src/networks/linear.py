import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear(torch.nn.Module):
    """
    a simple linear network
    """
    def __init__(self, params):
        
        super(Linear, self).__init__()
        
        self.input_dim = params['input_dim']
        self.output_dim = params['output_dim']
        
        self.use_softplus = params['use_softplus']
        self.softplus_gate = nn.Softplus(1)
        self.clip = params['clip']
        self.min_val = params.get('min_val', 1e-2)
        self.max_val = params.get('max_val', 1e2)
        
        self.ffnet = nn.Linear(self.input_dim, self.output_dim)
        
    def forward(self, x):
        
        out = self.ffnet(x)
        if self.use_softplus:
            out = torch.add(self.softplus_gate(out), float(self.min_val))
        if self.clip:
            out = torch.clamp(out, self.min_val, self.max_val)
        
        return out
