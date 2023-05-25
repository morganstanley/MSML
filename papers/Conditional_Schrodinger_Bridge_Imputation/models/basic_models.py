
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    """norm_first makes a big difference in calculating the gradient."""
    def __init__(self, d_model=128, seq_len=20, norm_first=False):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=1, dim_feedforward=32,
            activation="gelu", batch_first=True, norm_first=norm_first)
        self.pe = self.time_embedding(torch.arange(seq_len), d_model)
        self.net = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.projection = nn.Linear(d_model, 1)

    def time_embedding(self, pos, d_model):
        pe = torch.zeros(len(pos), d_model)
        position = pos.unsqueeze(1)
        div_term = 1 / torch.pow(5000.0, torch.arange(0, d_model, 2) / d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):
        y = x.unsqueeze(2)
        y = y + self.pe.to(y.device)
        y = self.net(y)
        y = self.projection(y).squeeze(2)
        loss = torch.sum(torch.square(x-y)) / x.shape[0] / x.shape[1]
        return loss, y


class MLP(nn.Module):  # Identity works.
    def __init__(self, seq_len=20):
        super().__init__()
        self.layer1 = nn.Linear(seq_len, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, seq_len)
        
    def forward(self, x):
        y = self.layer1(x)
        y = torch.sigmoid(y)
        y = self.layer2(y)
        y = torch.sigmoid(y)
        y = self.layer3(y)
        loss = torch.sum(torch.square(x-y)) / x.shape[0] / x.shape[1]
        return loss, y


class TransAttn(nn.Module):  # Identity works.
    def __init__(self, d_model=128, seq_len=20, dim_feedforward=32):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, 2, dropout=0.0, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.pe = self.time_embedding(torch.arange(seq_len), d_model)
        self.projection = nn.Linear(d_model, 1)

    def time_embedding(self, pos, d_model):
        pe = torch.zeros(len(pos), d_model)
        position = pos.unsqueeze(1)
        div_term = 1 / torch.pow(5000.0, torch.arange(0, d_model, 2) / d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def _sa_block(self, x):
        x = self.self_attn(x, x, x)[0]
        return x

    def _ff_block(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x

    def forward(self, x):
        y = x.unsqueeze(2)
        y = y + self.pe.to(y.device)
        y = y + self._sa_block(y)
        y = y + self._ff_block(y)
        y = self.projection(y).squeeze(2)
        loss = torch.sum(torch.square(x-y)) / x.shape[0] / x.shape[1]
        return loss, y


class TransAttnv2(nn.Module):  # Identity works.
    def __init__(self, d_model=128, seq_len=20, dim_feedforward=32):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, 2, dropout=0.0, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.modules.normalization.LayerNorm(d_model, eps=1e-05, elementwise_affine=True)
        self.norm2 = nn.modules.normalization.LayerNorm(d_model, eps=1e-05, elementwise_affine=True)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

        self.pe = self.time_embedding(torch.arange(seq_len), d_model)
        self.projection = nn.Linear(d_model, 1)

    def time_embedding(self, pos, d_model):
        pe = torch.zeros(len(pos), d_model)
        position = pos.unsqueeze(1)
        div_term = 1 / torch.pow(5000.0, torch.arange(0, d_model, 2) / d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def _sa_block(self, x):
        x = self.self_attn(x, x, x)[0]
        x = self.dropout1(x)
        return x

    def _ff_block(self, x):
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x

    def forward(self, x):
        y = x.unsqueeze(2)
        y = y + self.pe.to(y.device)
#         y = y + self._sa_block(y)
#         y = y + self._ff_block(y)
        # y = y + self._sa_block(self.norm1(y))
        # y = y + self._ff_block(self.norm2(y))
        y = self.norm1(y + self._sa_block(y))
        y = self.norm2(y + self._ff_block(y))
        y = self.projection(y).squeeze(2)
        loss = torch.sum(torch.square(x-y)) / x.shape[0] / x.shape[1]
        return loss, y


from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm

class TransAttnv3(nn.Module):  # Identity works.
    def __init__(self, d_model=128, seq_len=20, dim_feedforward=32,
            dropout=0.1, activation=F.relu, layer_norm_eps=1e-5,
            batch_first=True, norm_first=False,):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead=2, dropout=dropout, batch_first=batch_first)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.pe = self.time_embedding(torch.arange(seq_len), d_model)
        self.projection = nn.Linear(d_model, 1)

    def time_embedding(self, pos, d_model):
        pe = torch.zeros(len(pos), d_model)
        position = pos.unsqueeze(1)
        div_term = 1 / torch.pow(5000.0, torch.arange(0, d_model, 2) / d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def _sa_block(self, x):
        x = self.self_attn(x, x, x)[0]
        x = self.dropout1(x)
        return x

    def _ff_block(self, x):
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x

    def forward(self, x):
        y = x.unsqueeze(2)
        y = y + self.pe.to(y.device)
#         y = y + self._sa_block(y)
#         y = y + self._ff_block(y)
        y = y + self._sa_block(self.norm1(y))
        y = y + self._ff_block(self.norm2(y))
        y = self.projection(y).squeeze(2)
        loss = torch.sum(torch.square(x-y)) / x.shape[0] / x.shape[1]
        return loss, y


