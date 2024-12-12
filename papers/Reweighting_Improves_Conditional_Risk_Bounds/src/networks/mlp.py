import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(torch.nn.Module):
    """
    a simple feed-forward network consisting of a sequence of linear -> activation -> dropout layers
    """
    def __init__(self, params):
        
        super(MLP, self).__init__()
        
        self.input_dim = params['input_dim']
        self.output_dim = params['output_dim']
        self.hidden_dims = params['hidden_dims']
        self.dropout_rate = params['dropout_rate']
        self.activation = params['activation']
        
        self.use_softplus = params['use_softplus']
        self.softplus_gate = nn.Softplus(1)
        self.clip = params['clip']
        self.min_val = params.get('min_val', 1e-2)
        self.max_val = params.get('max_val', 1e2)
        
        self.bn = nn.BatchNorm1d(self.input_dim)
        
        if self.activation == 'tanh':
            self.activation = nn.Tanh()
        elif self.activation == 'elu':
            self.activation = nn.ELU()
        elif self.activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError('currenctly not implemented')
        
        layers, input_dim = [], self.input_dim
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(self.activation)
            layers.append(nn.Dropout(self.dropout_rate))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, self.output_dim))
        self.ffnet = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.bn(x)
        out = self.ffnet(x)
        if self.use_softplus:
            out = torch.add(self.softplus_gate(out), float(self.min_val))
        if self.clip:
            out = torch.clamp(out, self.min_val, self.max_val)
        return out
