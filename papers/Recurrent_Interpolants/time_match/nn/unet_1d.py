import math

import torch
import torch.nn.functional as F
from torch import nn


# embed time from unit interval to sin cos positional embedding
class TimeEmbedding(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Linear(2, hidden_size, bias=False)

    def forward(self, time):
        time = time * 2 * math.pi
        time = torch.stack([torch.sin(time), torch.cos(time)], dim=-1)
        time = self.embedding(time)
        return time


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size, residual_channels, dilation):
        super().__init__()
        self.dilated_conv = nn.Conv1d(
            residual_channels,
            2 * residual_channels,
            3,
            padding=dilation,
            dilation=dilation,
            padding_mode="circular",
        )
        self.diffusion_projection = nn.Linear(hidden_size, residual_channels)
        self.conditioner_projection = nn.Conv1d(
            1, 2 * residual_channels, 1, padding=2, padding_mode="circular"
        )
        self.output_projection = nn.Conv1d(residual_channels, 2 * residual_channels, 1)

        nn.init.kaiming_normal_(self.conditioner_projection.weight)
        nn.init.kaiming_normal_(self.output_projection.weight)

    def forward(self, x, conditioner, diffusion_step):
        diffusion_step = self.diffusion_projection(diffusion_step).squeeze(1).unsqueeze(-1)
        conditioner = self.conditioner_projection(conditioner)

        y = x + diffusion_step
        y = self.dilated_conv(y) + conditioner

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        y = F.leaky_relu(y, 0.4)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / math.sqrt(2.0), skip


class CondUpsampler(nn.Module):
    def __init__(self, hidden_size, target_dim):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, target_dim // 2)
        self.linear2 = nn.Linear(target_dim // 2, target_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = F.leaky_relu(x, 0.4)
        x = self.linear2(x)
        x = F.leaky_relu(x, 0.4)
        return x


class UNet1DConditionModel(nn.Module):
    def __init__(
        self,
        target_dim,
        hidden_size,
        time_emb_dim=16,
        residual_layers=8,
        residual_channels=8,
        dilation_cycle_length=2,
        residual_hidden=64,
    ):
        super().__init__()
        self.input_projection = nn.Conv1d(
            1, residual_channels, 1, padding=2, padding_mode="circular"
        )
        self.diffusion_embedding = TimeEmbedding(residual_hidden)

        self.cond_upsampler = CondUpsampler(
            target_dim=target_dim, hidden_size=hidden_size
        )
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    residual_channels=residual_channels,
                    dilation=2 ** (i % dilation_cycle_length),
                    hidden_size=residual_hidden,
                )
                for i in range(residual_layers)
            ]
        )
        self.skip_projection = nn.Conv1d(residual_channels, residual_channels, 3)
        self.output_projection = nn.Conv1d(residual_channels, 1, 3)

        nn.init.kaiming_normal_(self.input_projection.weight)
        nn.init.kaiming_normal_(self.skip_projection.weight)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, inputs, time, cond):
        # inputs: [B*T, 1, F]
        # time: [B*T,1 ]
        # cond: [B*T, 1, H]
        if len(inputs.shape) == 2: # synthetic data case.
            inputs = inputs.unsqueeze(1) # (B,2) --> (B,1,2)
        batch_size, time_len, dim = inputs.shape
        inputs = inputs.reshape(batch_size * time_len, 1, dim)
        time = time.reshape(batch_size * time_len, 1)
        cond = cond.reshape(batch_size * time_len, 1, -1)

        x = self.input_projection(inputs)  # [B, C, T]
        x = F.leaky_relu(x, 0.4)

        diffusion_step = self.diffusion_embedding(time)

        cond_up = self.cond_upsampler(cond)
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_up, diffusion_step)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.leaky_relu(x, 0.4)
        x = self.output_projection(x)
        x = x.reshape(batch_size, time_len, dim)
        return x
