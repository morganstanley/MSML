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


class diff_CSDI(nn.Module):
    def __init__(self, config, inputdim, zero_out_last_layer=True):
        super().__init__()
        self.channels = config["channels"]
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"])
        self.output_layer = config["output_layer"]

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.residual_layers = nn.ModuleList(
            # [
            #     ResidualBlock(
            #         side_dim=config["side_dim"],
            #         channels=self.channels,
            #         diffusion_embedding_dim=config["diffusion_embedding_dim"],
            #         nheads=config["nheads"],
            #     )
            # ]
            # +
            [
                ResidualConvBlock(
                    side_dim=config["side_dim"],
                    feature_embed_dim=config["featureemb"]+1,  # with cond_mask so +1.
                    channels=self.channels,
                    L=config["input_size"][1],
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                )
                for _ in range(config["layers"])
            ]
        )

        if self.output_layer == 'conv1d':
            # The output projection causes divergence calulation issue.
            self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
            self.output_projection = Conv1d_with_init(self.channels, 1, 1)
        elif self.output_layer == 'conv2d':
            self.output_projection = nn.Conv2d(self.channels, 1, 1, stride=1)

        if zero_out_last_layer:
            # nn.init.zeros_(self.output_projection.weight)
            self.output_projection = zero_module(self.output_projection)


    def forward(self, x, diffusion_step=None, window_embedding=None, feature_embed=None):
        B, inputdim, K, L = x.shape
        diffusion_step = diffusion_step.long()

        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)

        diffusion_emb = self.diffusion_embedding(diffusion_step).to(x.device)

        skip = []
        for l, layer in enumerate(self.residual_layers):
            if l < 0:
                x, skip_connection = layer(x, window_embedding, diffusion_emb)
            else:
                x, skip_connection = layer(x, feature_embed, diffusion_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))

        if self.output_layer == 'conv1d':
            # The output projection causes divergence calulation issue.
            x = x.reshape(B, self.channels, K * L)
            x = self.output_projection1(x)  # (B,channel,K*L)
            # x = F.relu(x)
            x = F.leaky_relu(x, negative_slope=0.1)
            x = self.output_projection(x)  # (B,1,K*L)
            x = x.reshape(B, 1, K, L)

        elif self.output_layer == 'conv2d':
            x = self.output_projection(x)

        return x

class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.time_layer= get_torch_trans(heads=nheads, layers=1, channels=channels)
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


class Conv1dResLayer(nn.Module):
    """Mimicking the ResNet bottleneck building blcok (He et.al. 2015)."""
    def __init__(
            self,
            Cin,
            Cmid,
            L):
        super().__init__()
        self.norm1 = nn.LayerNorm([Cin, L], eps=1e-5)
        self.conv1 = nn.Conv1d(Cin, Cmid, kernel_size=3, padding=1, bias=True)
        self.act1 = SiLU()
        self.norm2 = nn.LayerNorm([Cin, L], eps=1e-5)
        self.conv2 = nn.Conv1d(Cmid, Cmid, kernel_size=5, padding=2, bias=True)
        self.act2 = SiLU()
        self.conv3 = nn.Conv1d(Cmid, Cin, kernel_size=3, padding=1, bias=True)


    def forward(self, x):
        """x (X,C,L)"""
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.act1(y)
        y = self.norm2(y)
        y = self.conv2(y)
        y = self.act2(y)
        y = self.conv3(y)
        y = (y + x) / 1.414213
        return y

class Conv1dBlock(nn.Module):
    """Mimicking the ResNet bottleneck building blcok (He et.al. 2015)."""
    def __init__(
            self,
            Cin,
            Cmid,
            L,
            num_blocks=2):
        super().__init__()
        net = nn.ModuleList([Conv1dResLayer(Cin, Cmid, L) for _ in range(num_blocks)])
        self.f = nn.Sequential(*net)

    def forward(self, x):
        """x (X,C,L)"""
        x = self.f(x)
        return x

class ResidualConvBlock(nn.Module):
    def __init__(self, side_dim, feature_embed_dim, channels, L, diffusion_embedding_dim, nheads):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        # self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.cond_projection = Conv1d_with_init(feature_embed_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.time_layer = Conv1dBlock(channels, channels, L, num_blocks=3)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

    def forward_time(self, y, base_shape):
        B, C, K, L = base_shape
        y = y.reshape(B, C, K, L).permute(0, 2, 1, 3).reshape(B*K, C, L)  # (B*K,C,L)
        y = self.time_layer(y)  # (B*K,C,L)
        y = y.reshape(B, K, C, L).permute(0, 2, 1, 3)  # (B,C,K,L)
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

    def forward(self, x, feature_embed, diffusion_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)  # (B,channel,K*L)
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        _, cond_dim, _, _ = feature_embed.shape
        feature_embed = feature_embed.reshape(B, cond_dim, K * L)
        feature_embed = self.cond_projection(feature_embed)  # (B,2*channel,K*L)
        y = y + feature_embed

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip

