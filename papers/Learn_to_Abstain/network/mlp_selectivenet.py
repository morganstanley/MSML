import torch
from torch import nn
import torch.nn.functional as F

class MLPSelectiveNet(nn.Module):
    def __init__(self, 
        num_layers: int = None, 
        input_dim:  int = None, 
        hidden_dim: int = None, 
        num_class:  int = None, 
        **kwargs):
        '''
            num_layers: number of layers in the neural nc_networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''

        super(MLPSelectiveNet, self).__init__()

        num_layers = num_layers if num_layers is not None else 4
        hidden_dim = hidden_dim if hidden_dim is not None else 128
        self.num_layers = num_layers
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        if num_layers < 2:
            raise ValueError("number of layers should be higher than 2!")
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()
            
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            
            self.cls_encoder = nn.Linear(hidden_dim, num_class)
            self.sel_encoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), 
                nn.BatchNorm1d(hidden_dim), 
                nn.Linear(hidden_dim, 1)
            )
            self.sigmoid = torch.nn.Sigmoid()
            self.aux_encoder = nn.Linear(hidden_dim, num_class)
                
            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):

        h = x
        for layer in range(self.num_layers - 1):
            h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
        cls_out = self.cls_encoder(h)
        sel_out = self.sigmoid(self.sel_encoder(h))
        aux_out = self.aux_encoder(h)
        
        return cls_out, sel_out, aux_out
    
    

if __name__ == '__main__':
    
    testnet = MLP(num_layers=3, input_dim=5, hidden_dim=5, num_class=3)