import torch
from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, 
        num_layers:int = None, 
        input_dim: int = None, 
        hidden_dim:int = None, 
        num_class: int = None, 
        **kwargs):
        '''
            num_layers: number of layers in the neural nc_networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''

        super(MLP, self).__init__()

        self.linear_or_not = True  # default is linear model
        num_layers = num_layers if num_layers is not None else 4
        hidden_dim = hidden_dim if hidden_dim is not None else 128
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.fc3 = nn.Linear(input_dim, num_class)
            self.linear = self.fc3
        else:
            self.fc3 = nn.Linear(hidden_dim, num_class)
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(self.fc3)

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x, *args, **kwargs):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)
    
    

if __name__ == '__main__':
    
    testnet = MLP(num_layers=3, input_dim=5, hidden_dim=5, num_class=3)