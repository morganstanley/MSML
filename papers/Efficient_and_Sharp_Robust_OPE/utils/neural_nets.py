import torch
import torch.nn as nn

class FFNet(nn.Module):
    def __init__(self, input_dim, output_dim, layer_sizes=None, dropout_rate=None):
        super().__init__()
        layers = []
        if layer_sizes is None:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            prev_size = input_dim
            for size in layer_sizes:
                layers.append(nn.Linear(prev_size, size))
                layers.append(nn.GELU())
                if dropout_rate is not None:
                    layers.append(nn.Dropout(dropout_rate))
                prev_size = size
            layers.append(nn.Linear(prev_size, output_dim))
        self.ff_net = nn.Sequential(*layers)

    def forward(self, input):
        return self.ff_net(input)

