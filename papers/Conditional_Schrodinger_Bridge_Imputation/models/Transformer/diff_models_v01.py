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
        self.channels = config["diff_channels"]
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"])

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                )
                for _ in range(config["layers"])
            ]
        )

        self.reslayer = ResidualBlock_conv(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                )

        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        self.conv2d_output_projection = nn.Conv2d(self.channels, 1, 1, stride=1)
        # nn.init.zeros_(self.output_projection2.weight)
        if zero_out_last_layer:
            # self.output_projection1 = zero_module(self.output_projection1)
            self.output_projection2 = zero_module(self.output_projection2)


    def forward(self, x, diffusion_step=None, window_embedding=None):
        B, inputdim, K, L = x.shape
        diffusion_step = diffusion_step.long()

        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)

        diffusion_emb = self.diffusion_embedding(diffusion_step).to(x.device)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, window_embedding, diffusion_emb)
            skip.append(skip_connection)

        # x, _ = self.reslayer(x, window_embedding, diffusion_emb)
        # x, _ = self.reslayer(x, window_embedding, diffusion_emb)
        # x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))

        # original output projection crushes divergence.
        # x = x.reshape(B, self.channels, K * L)
        # x = self.output_projection1(x)  # (B,channel,K*L)
        # x = F.relu(x)
        # x = self.output_projection2(x)  # (B,1,K*L)
        # # x = x.reshape(B, K, L)
        # x = x.reshape(B, 1, K, L)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = self.conv2d_output_projection(x)

        return x


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

