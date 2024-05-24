import torch
import torch.nn as nn

from models.abstract_critic import AbstractCritic
from utils.neural_nets import FFNet


class FeedForwardCritic(AbstractCritic):
    def __init__(self, s_dim, num_a, config):
        super().__init__(s_dim=s_dim, num_a=num_a)
        s_embed_size = config["s_embed_dim"]
        a_embed_size = config["a_embed_dim"]
        sa_embed_size = s_embed_size + a_embed_size

        self.s_embed_net = FFNet(
            input_dim=s_dim,
            output_dim=s_embed_size,
            layer_sizes=config["s_embed_layers"],
            dropout_rate=config.get("s_embed_do", 0.05)
        )
        self.a_embed_net = nn.Embedding(
            num_embeddings=num_a,
            embedding_dim=a_embed_size,
        )
        self.q_critic = FFNet(
            input_dim=sa_embed_size,
            output_dim=1,
            layer_sizes=config["critic_layers"],
            dropout_rate=config.get("critic_do", 0.05)
        )
        self.eta_critic = FFNet(
            input_dim=sa_embed_size,
            output_dim=1,
            layer_sizes=config["critic_layers"],
            dropout_rate=config.get("critic_do", 0.05)
        )
        self.w_critic = FFNet(
            input_dim=s_embed_size,
            output_dim=1,
            layer_sizes=config["critic_layers"],
            dropout_rate=config.get("critic_do", 0.05)
        )
        self.eval()

    def forward(self, s, a, calc_q=False, calc_eta=False, calc_w=False):
        s_embed = self.s_embed_net(s)
        if calc_q or calc_eta:
            a_embed = self.a_embed_net(a)
            sa_concat = torch.cat([s_embed, a_embed], dim=1)

        if calc_q:
            f_q = self.q_critic(sa_concat)
        else:
            f_q = None

        if calc_eta:
            f_eta = self.eta_critic(sa_concat)
        else:
            f_eta = None

        if calc_w:
            f_w = self.w_critic(s_embed)
        else:
            f_w = None

        return f_q, f_eta, f_w

    def get_q(self, s, a):
        q, _, _ = self(s, a, calc_q=True)
        return q

    def get_eta(self, s, a):
        _,  eta, _ = self(s, a, calc_eta=True)
        return eta

    def get_w(self, s):
        _,  _, w = self(s, a=None, calc_w=True)
        return w

    def get_all(self, s, a):
        q, eta, w = self(s, a, calc_q=True, calc_eta=True, calc_w=True)
        return q, eta, w
