import torch
import torch.nn as nn
import torch.nn.functional as F

class FCBlock(torch.nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        dropout_rate=0.1,
        activation='tanh',
    ):
        super(FCBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        
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
            self.activation,
            nn.Dropout(self.dropout_rate),
            
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self,x):
        output = self.fc_block(x)
        return output
        
class FCResNet(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        dropout_rate=0.1,
        activation='tanh',
        architect='concat',
        use_softplus=False,
        clip = False,
        min_val=1e-5,
        max_val=100
    ):
        
        super(FCResNet, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.min_val = min_val
        self.max_val = max_val
        
        self.use_softplus = use_softplus
        self.soft_plus_gate = nn.Softplus(1)
        self.clip = clip
        self.bn = nn.BatchNorm1d(input_dim)
        
        self.fc1 = FCBlock(input_dim,hidden_dim,input_dim,dropout_rate,activation)
        self.fc2 = FCBlock(input_dim,hidden_dim,input_dim,dropout_rate,activation)
        
        self.architect = architect
        if self.architect == 'stack':
            self.output_layer = nn.Linear(input_dim, output_dim)
        elif self.architect == 'concat':
            self.output_layer = nn.Linear(2*input_dim, output_dim)
        else:
            raise ValueErorr('incorrect specification of architect; choose between [concat, stack]')
            
    def forward(self, x):
        
        x = self.bn(x)
        if self.architect == 'stack':
            y = self.fc1(x)
            y += x
            y = self.fc2(y)
            y += x
        elif self.architect == 'concat':
            y1 = self.fc1(x)
            y1 += x
            y2 = self.fc2(x)
            y2 += x
            y = torch.cat([y1,y2],dim=-1)
        
        out = self.output_layer(y)
        if self.use_softplus:
            out = torch.add(self.soft_plus_gate(out), float(self.min_val))
        if self.clip:
            out = torch.clamp(out, self.min_val, self.max_val)
        return out
