
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import MLP
        
class FCResNet(nn.Module):
    """
    a fully-connected network with two MLP and additional residual connections
    choice of architect: concat (the two MLP are concatenated side-by-side); stack (the two MLP are stacked)
    """
    def __init__(self,params):
    
        super(FCResNet, self).__init__()
        
        self.input_dim = params['input_dim']
        self.output_dim = params['output_dim']
        self.dropout_rate = params['dropout_rate']
    
        self.use_softplus = params['use_softplus']
        self.softplus_gate = nn.Softplus(1)
        
        self.clip = params['clip']
        self.min_val = params.get('min_val', 1e-2)
        self.max_val = params.get('max_val', 1e2)
        
        ff_params = params.copy()
        ff_params['output_dim'] = self.input_dim
        ff_params['use_softplus'] = False
        ff_params['clip'] = False
        
        self.fc1 = MLP(ff_params)
        self.fc2 = MLP(ff_params)
        
        self.architect_type = params['architect_type']
        if self.architect_type == 'stack':
            self.bn = nn.BatchNorm1d(self.input_dim)
            self.output_layer = nn.Linear(self.input_dim, self.output_dim)
        elif self.architect_type == 'concat':
            self.output_layer = nn.Linear(2 * self.input_dim, self.output_dim)
        else:
            raise ValueErorr('incorrect specification of architect_type; choose between [concat, stack]')
            
    def forward(self, x):
        
        if self.architect_type == 'stack':
            y = self.fc1(x)
            y += x
            y = self.bn(y)
            y = self.fc2(y)
            y += x
        elif self.architect_type == 'concat':
            y1 = self.fc1(x)
            y1 += x
            y2 = self.fc2(x)
            y2 += x
            y = torch.cat([y1,y2],dim=-1)
        
        out = self.output_layer(y)
        if self.use_softplus:
            out = torch.add(self.softplus_gate(out), float(self.min_val))
        if self.clip:
            out = torch.clamp(out, self.min_val, self.max_val)
        return out
