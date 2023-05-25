import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.utils import *


def get_torch_trans_original(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(d_model=channels, nhead=heads, dim_feedforward=64,
        batch_first=True, norm_first=True, dropout=0.0)
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


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


class diff_CSDI(nn.Module):
    def __init__(self, config, inputdim, zero_out_last_layer=True):
        super().__init__()
        self.channels = channels = config["channels"]
        self.diff_channels = config["diff_channels"]
        embed_dim = config["diffusion_embedding_dim"]
        self.interval = config["num_steps"]
        self.zero_out_last_layer = zero_out_last_layer
        self.sizes_on_scales = self.input_sizes_on_scales(config["input_size"], 4)
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
        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False, padding=1)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

        # Decoding layers where the resolution increases
        self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, bias=False, output_padding=1)
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
        self.tconv3 = nn.ConvTranspose2d(channels[2] + channels[2], channels[1], 3, stride=2, bias=False, output_padding=1)
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
        self.tconv2 = nn.ConvTranspose2d(channels[1] + channels[1], channels[0], 3, stride=2, bias=False, output_padding=1)
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
        self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 1, 3, stride=1)
        # self.tconv1 = nn.ConvTranspose2d(channels[0], 1, 3, stride=1)

        side_dim = config["side_dim"]
        self.cond_projection = nn.Conv2d(side_dim, channels[0], 3, padding=1)
        nn.init.kaiming_normal_(self.cond_projection.weight)

        self.reslayer = ResidualBlockv0(
                    side_dim=config["side_dim"],
                    channels=channels[0],
                    diffusion_embedding_dim=embed_dim,
                    nheads=config["nheads"],
                )

        # original output projection crushes divergence.
        self.output_projection1 = Conv1d_with_init(self.channels[0], self.channels[0], 1)
        self.output_projection2 = Conv1d_with_init(self.channels[0], 1, 1)
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


    def forward(self, x, time_steps=None, window_embedding=None): 
        # Obtain the Gaussian random feature embedding for t   
        B, _, K, L = x.shape
        t = time_steps / self.interval
        embed = self.act(self.embed(t))  # diffusion time embedding

        # print('window_embedding', window_embedding.shape)
        cond_emb = self.cond_projection(window_embedding)  # (B,channel,K*L)
        # print('cond_emb', cond_emb.shape)

        #--------------- Sampling DOWN ----------------
        h1 = self.conv1(x)
        ## Incorporate information from t
        h1 += self.dense1(embed) + cond_emb
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

        h4 = self.conv4(h3)
        h4 += self.dense4(embed)
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)
        # print('h4', h4.shape)

        #--------------- Sampling UP ----------------
        h = self.tconv4(h4)
        h = F.interpolate(h, size=(self.sizes_on_scales[-2][0], self.sizes_on_scales[-2][1]))
        # print('h4t', h.shape)
        ## Skip connection from the encoding path
        h += self.dense5(embed)
        h = self.tgnorm4(h)
        h = self.act(h)
        h = self.tconv3(torch.cat([h, h3], dim=1))
        h = F.interpolate(h, size=(self.sizes_on_scales[-3][0], self.sizes_on_scales[-3][1]))
        # print('h3t', h.shape)
        h += self.dense6(embed)
        h = self.tgnorm3(h)
        h = self.act(h)
        h = self.tconv2(torch.cat([h, h2], dim=1))
        h = F.interpolate(h, size=(self.sizes_on_scales[-4][0], self.sizes_on_scales[-4][1]))
        # print('h2t', h.shape)
        h += self.dense7(embed) + cond_emb
        h = self.tgnorm2(h)
        h = self.act(h)

        # print('hhhh', h.shape, cond_emb.shape, embed.shape)
        h,_ = self.reslayer(h, window_embedding, embed)
        # print('h reslayer out', h.shape)

        # this output projection works.
        # h = self.tconv1(torch.cat([h, h1], dim=1))
        # h = F.interpolate(h, size=(self.sizes_on_scales[-4][0], self.sizes_on_scales[-4][1]))
        # print('h1t', h.shape)

        h = h.reshape(B, self.channels[0], K * L)
        h = self.output_projection1(h)  # (B,channel,K*L)
        h = F.relu(h)
        h = self.output_projection2(h)  # (B,1,K*L)
        # x = x.reshape(B, K, L)
        h = h.reshape(B, 1, K, L)

        return h


class ResidualBlockv0(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.time_layer = get_torch_trans_original(heads=nheads, layers=1, channels=channels)
        self.feature_layer = get_torch_trans_original(heads=nheads, layers=1, channels=channels)

    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)  # (L,BK,C), (BK,C,L)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)  # (K,BL,C), (BL,C,K)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x, cond_info, diffusion_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)  # (B,channel,K*L)
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)
        y = y + cond_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip


# CRUSH the alternative training.
# The problem is that use batch_first consistent with the dim in forward_time and forward_feature.
class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)  # (L,BK,C), (BK,C,L)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)  # (K,BL,C), (BL,C,K)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x, cond_info, diffusion_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)  # (B,channel,K*L)
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)
        y = y + cond_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip


# Working
class ResidualBlock_conv(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads):
        super().__init__()
        self.channels = channels
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 1 * channels, 1)
        # self.mid_projection = Conv1d_with_init(channels, 1 * channels, 1)
        # self.output_projection = Conv1d_with_init(channels, 1 * channels, 1)
        self.output_projection = nn.Conv2d(channels, 1 * channels, 3, padding=1)

        # self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        # self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

        self.layer1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.layer2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B*K, channel, L)
        y = self.time_layer(y.permute(0, 2, 1)).permute(0, 2, 1)  # (BK,L,C)  (BK,C,L)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B*L, channel, K)
        y = self.feature_layer(y.permute(0, 2, 1)).permute(0, 2, 1)  # (BL,K,C), (BL,C,K)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x, cond_info, diffusion_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape
        # x = x.reshape(B, channel, K * L)

        # diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        # print('diffusion_emb', diffusion_emb.shape)
        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1).unsqueeze(-1)  # (B,channel,1)
        # print('diffusion_emb', diffusion_emb.shape, x.shape)
        y = x + diffusion_emb

        # y = self.forward_time(y, base_shape) # (B,channel,K*L)
        # y = self.forward_feature(y, base_shape)  # (B,channel,K*L)
        y = self.layer1(y)
        y = torch.relu(y)
        y = self.layer2(y)
        y = torch.relu(y)

        y = self.output_projection(y)

        return y, None


# Working
class ResidualBlock_time_trans(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads):
        super().__init__()
        self.channels = channels
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 1 * channels, 1)
        # self.mid_projection = Conv1d_with_init(channels, 1 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 1 * channels, 1)

        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B*K, channel, L)
        y = self.time_layer(y.permute(0, 2, 1)).permute(0, 2, 1)  # (BK,L,C)  (BK,C,L)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward(self, x, cond_info, diffusion_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape) # (B,channel,K*L)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_emb = self.cond_projection(cond_info)  # (B,1*channel,K*L)
        y = y + cond_emb

        y = self.output_projection(y)
        y = y.reshape(base_shape)

        return y, None


# Working
class ResidualBlock_feature_trans(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads):
        super().__init__()
        self.channels = channels
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 1 * channels, 1)
        # self.mid_projection = Conv1d_with_init(channels, 1 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 1 * channels, 1)

        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B*L, channel, K)
        y = self.feature_layer(y.permute(0, 2, 1)).permute(0, 2, 1)  # (BL,K,C), (BL,C,K)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x, cond_info, diffusion_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb

        y = self.forward_feature(y, base_shape)  # (B,channel,K*L)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_emb = self.cond_projection(cond_info)  # (B,1*channel,K*L)
        y = y + cond_emb

        y = self.output_projection(y)
        y = y.reshape(base_shape)

        return y, None


# Working
class ResidualBlock_time_feature_trans(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads):
        super().__init__()
        self.channels = channels
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 1 * channels, 1)
        # self.mid_projection = Conv1d_with_init(channels, 1 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 1 * channels, 1)
        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

    def forward_time(self, y, base_shape):
        """batch_first = True"""
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B*K, channel, L)
        y = self.time_layer(y.permute(0, 2, 1)).permute(0, 2, 1)  # (BK,L,C)  (BK,C,L)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape):
        """batch_first = True"""
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B*L, channel, K)
        y = self.feature_layer(y.permute(0, 2, 1)).permute(0, 2, 1)  # (BL,K,C), (BL,C,K)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x, cond_info, diffusion_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_emb = self.cond_projection(cond_info)  # (B,1*channel,K*L)
        y = y + cond_emb

        y = self.output_projection(y)
        y = y.reshape(base_shape)

        return y, None


# crush
class ResidualBlockv2(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 1 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 1 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 1 * channels, 1)

        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

    def forward_time(self, y, base_shape):
        """The setting is the same as original code, where batch_first = False"""
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)  # (L,BK,C), (BK,C,L)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape):
        """The setting is the same as original code, where batch_first = False"""
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)  # (K,BL,C), (BL,C,K)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x, cond_info, diffusion_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)  # (B,channel,K*L)
        y = self.mid_projection(y)  # (B,channel,K*L)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)
        y = y + cond_info

        y = self.output_projection(y)
        y = y.reshape(base_shape)
        return y, None


# working
class ResidualBlockv3(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 1 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 1 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 1 * channels, 1)

        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

    def forward_time(self, y, base_shape):
        """batch_first = True"""
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B*K, channel, L)
        y = self.time_layer(y.permute(0, 2, 1)).permute(0, 2, 1)  # (BK,L,C)  (BK,C,L)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape):
        """batch_first = True"""
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B*L, channel, K)
        y = self.feature_layer(y.permute(0, 2, 1)).permute(0, 2, 1)  # (BL,K,C), (BL,C,K)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x, cond_info, diffusion_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_emb = self.cond_projection(cond_info)  # (B,1*channel,K*L)
        y = y + cond_emb

        y = self.output_projection(y)
        y = y.reshape(base_shape)
        return y, None


# working
class ResidualBlockv4(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 1 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 1 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 1 * channels, 1)

        self.time_layer = get_torch_trans_original(heads=nheads, layers=1, channels=channels)
        self.feature_layer = get_torch_trans_original(heads=nheads, layers=1, channels=channels)

    def forward_time(self, y, base_shape):
        """batch_first = False"""
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)  # (L,BK,C), (BK,C,L)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape):
        """batch_first = False"""
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)  # (K,BL,C), (BL,C,K)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x, cond_info, diffusion_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_emb = self.cond_projection(cond_info)  # (B,1*channel,K*L)
        y = y + cond_emb

        y = self.output_projection(y)
        y = y.reshape(base_shape)
        return y, None

