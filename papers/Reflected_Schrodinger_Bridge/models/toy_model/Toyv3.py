import torch
import torch.nn as nn
from models.utils import *

def build_toyv3(config, zero_out_last_layer=True):
    return Toyv3(config, zero_out_last_layer=zero_out_last_layer)


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""  
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed 
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.dense(x)[..., None, None]


class Toyv3(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(self, config, zero_out_last_layer=True):
        """Simple network."""
        super().__init__()
        channels = config.channels
        embed_dim = config.embed_dim
        self.interval = config.interval
        self.zero_out_last_layer = zero_out_last_layer
        self.sizes_on_scales = self.input_sizes_on_scales(config.input_size, 3)
        print('sizes_on_scales', self.sizes_on_scales)
        # Gaussian random feature embedding layer for time
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
             nn.Linear(embed_dim, embed_dim))
        # Encoding layers where the resolution decreases
        self.conv1 = nn.Conv2d(1, channels[0], 3, stride=1, bias=False, padding=1)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False, padding=1)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False, padding=1)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])

        # Decoding layers where the resolution increases
        self.tconv3 = nn.ConvTranspose2d(channels[2], channels[1], 3, stride=2, bias=False, output_padding=1)
        self.tdense3 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
        self.tconv2 = nn.ConvTranspose2d(channels[1] + channels[1], channels[0], 3, stride=2, bias=False, output_padding=1)
        self.tdense2 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
        self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 1, 3, stride=1)

        if zero_out_last_layer:
            self.tconv1 = zero_module(self.tconv1)

        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)
        # self.marginal_prob_std = marginal_prob_std

    def input_sizes_on_scales(self, input_size, top_scale=None):
        """
        Returns:
            List of input_sizes on different scales.
        e.g. input(18, 18)
        [(18, 18), (9, 9), (5, 5), (3, 3), (1, 1), (0, 0),....]
        e.g.input(32, 36)
        [(32, 36), (16, 18), (8, 9), (4, 5), (2, 3), (1, 2), (0, 0),....]
        """
        outputs = [(0,0)] * 30  # Assume no more than 30 levels.
        def scale_size(input_size, scale_ind=0, outputs=outputs):
            outputs[scale_ind] = input_size
            if input_size[0] <= 1 or input_size[1] <= 1:
                return
            down_size = (np.ceil(input_size[0]/2).astype(int), np.ceil(input_size[1]/2).astype(int))
            scale_size(down_size, scale_ind+1, outputs)
        scale_size(input_size, 0, outputs)
        if top_scale:
            outputs = outputs[:top_scale]
        return outputs


    def forward(self, x, time_steps=None): 
        # Obtain the Gaussian random feature embedding for t   
        if time_steps is None:
            t = torch.zeros(x.shape[0], device=x.device) / self.interval
            embed = self.act(self.embed(t))
            embed = torch.zeros_like(embed)
        else:
            t = time_steps / self.interval
            embed = self.act(self.embed(t))

        #--------------- Sampling DOWN ----------------
        h1 = self.conv1(x)
        ## Incorporate information from t
        h1 += self.dense1(embed)
        h1 = self.gnorm1(h1)  # Group normalization
        h1 = self.act(h1)
        # print('h1', h1.shape)

        h2 = self.conv2(h1)
        h2 += self.dense2(embed)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)
        # print('h2', h2.shape)

        h3 = self.conv3(h2)
        h3 += self.dense3(embed)
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)
        # print('h3', h3.shape)

        #--------------- Sampling UP ----------------
        h = self.tconv3(h3)
        h = F.interpolate(h, size=(self.sizes_on_scales[-2][0], self.sizes_on_scales[-2][1]))
        # print('h3t', h.shape)
        h += self.tdense3(embed)
        h = self.tgnorm3(h)
        h = self.act(h)
        h = self.tconv2(torch.cat([h, h2], dim=1))
        h = F.interpolate(h, size=(self.sizes_on_scales[-3][0], self.sizes_on_scales[-3][1]))
        # print('h2t', h.shape)
        h += self.tdense2(embed)
        h = self.tgnorm2(h)
        h = self.act(h)
        h = self.tconv1(torch.cat([h, h1], dim=1))
        h = F.interpolate(h, size=(self.sizes_on_scales[-3][0], self.sizes_on_scales[-3][1]))
        # print('h1t', h.shape)

        # Normalize output
        # h = h / self.marginal_prob_std(t)[:, None, None, None]
        return h

