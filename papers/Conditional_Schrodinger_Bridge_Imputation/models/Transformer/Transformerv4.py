
import torch
from models.utils import *

from ipdb import set_trace as debug
from models.Transformer.diff_models_v4 import diff_CSDI


def build_transformerv4(config, interval, zero_out_last_layer):
    input_size = config.input_size

    config_diff = {}
    config_diff["num_steps"] = interval + 1
    config_diff["layers"] = config.layers
    config_diff["nheads"] = config.nheads
    config_diff["channels"] = config.channels
    config_diff["diffusion_embedding_dim"] = config.diffusion_embedding_dim
    config_diff["timeemb"] = config.timeemb
    config_diff["featureemb"] = config.featureemb
    config_diff["side_dim"] = config_diff["timeemb"] + config_diff["featureemb"] + 1
    if not hasattr(config, 'output_layer'):
        # Default as the original code. Issue with calculating divergence w.r.t. the input.
        config_diff["output_layer"] = 'conv1d'
    else:
        config_diff["output_layer"] = config.output_layer

    return Transformerv4(
        input_size=input_size,
        config_diff=config_diff,
        zero_out_last_layer=zero_out_last_layer,
    )

class Transformerv4(nn.Module):
    """
    """
    def __init__(
        self,
        input_size,
        config_diff,
        zero_out_last_layer=False):
        super().__init__()
        self.input_size = input_size  # (K,L)
        config_diff["input_size"] = input_size
        self.target_dim = input_size[0]
        self.zero_out_last_layer = zero_out_last_layer
        self.emb_time_dim = config_diff["timeemb"]
        self.emb_feature_dim = config_diff["featureemb"]
        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim + 1
        self.inputdim = 2

        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim)
        self.diffmodel = diff_CSDI(config_diff, inputdim=self.inputdim,
            zero_out_last_layer=zero_out_last_layer)


    def time_embedding(self, pos, d_model, device):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model, device=device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(10000.0, torch.arange(0, d_model, 2) / d_model).to(device)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_window_embedding(self, batch_size, observed_tp=None, cond_mask=None, device=None):
        B, K, L = batch_size, self.input_size[0], self.input_size[1]
        if observed_tp is None:
            observed_tp = torch.arange(L).unsqueeze(0).repeat(batch_size,1).to(device)
        time_embed = self.time_embedding(observed_tp, self.emb_time_dim, device)  # (B,L,Temb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)

        feature_embed = self.embed_layer(torch.arange(self.target_dim).to(device))  # (K,Kemb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        window_embedding = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,KTemb)
        window_embedding = window_embedding.permute(0, 3, 2, 1)  # (B,KTemb,K,L)
        window_embedding = torch.cat([window_embedding, cond_mask], dim=1)  # (B,KTemb+1,K,L)

        feature_embed = feature_embed.permute(0, 3, 2, 1)  # (B,Kemb,K,L)
        feature_embed = torch.cat([feature_embed, cond_mask], dim=1)  # (B,Kemb+1,K,L)

        return window_embedding, feature_embed


    def forward(self, input_x, timesteps):
        x, cond_mask = input_x
        window_embedding, feature_embed = self.get_window_embedding(
            batch_size=x.shape[0], cond_mask=cond_mask, device=x.device) # (B,emb,K,L)
        score = self.diffmodel(x, timesteps, window_embedding=window_embedding,
            feature_embed=feature_embed,)  # (B,1,K,L)

        return score

