import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        dropout_rate=0.1,
        activation='tanh',
        use_softplus=False,
        clip = False,
        min_val=1e-5,
        max_val=100
    ):
        
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.min_val = min_val
        self.max_val = max_val
        self.use_softplus = use_softplus
        self.soft_plus_gate = nn.Softplus(1)
        self.clip = clip
        self.bn = nn.BatchNorm1d(input_dim)
        
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError('currenctly not implemented')
        
        self.fc_block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            self.activation,
            nn.Dropout(self.dropout_rate),
            
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Dropout(self.dropout_rate),
            
            nn.Linear(hidden_dim, hidden_dim),
            self.activation
        )
        self.ffn = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.bn(x)
        x = self.fc_block(x)
        out = self.ffn(x)
        if self.use_softplus:
            out = torch.add(self.soft_plus_gate(out), float(self.min_val))
        if self.clip:
            out = torch.clamp(out, self.min_val, self.max_val)
        return out
