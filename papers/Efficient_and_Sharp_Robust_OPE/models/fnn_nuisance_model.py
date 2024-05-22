import os

import torch
import torch.nn as nn

from models.abstract_nuisance_model import AbstractNuisanceModel
from utils.neural_nets import FFNet


class FFNBetaModule(nn.Module):
    def __init__(self, s_dim, num_a, gamma, config, device=None):
        super().__init__()
        self.s_dim = s_dim
        self.num_a = num_a

        self.beta_net = FFNet(
            input_dim=s_dim,
            output_dim=num_a,
            layer_sizes=config["beta_layers"],
            dropout_rate=config.get("beta_do", 0.05)
        )
        # self.pos_head = nn.Softplus(beta=0.2)
        self.pos_head = lambda x_: x_.abs()

    def forward(self, s, a):
        beta_all = self.pos_head(self.beta_net(s))
        beta = beta_all.gather(dim=-1, index=a.unsqueeze(-1))
        return beta


class FFNuisanceModule(nn.Module):
    def __init__(self, s_dim, num_a, gamma, config, device=None):
        super().__init__()
        self.s_dim = s_dim
        self.num_a = num_a

        self.q_net = FFNet(
            input_dim=s_dim,
            output_dim=num_a,
            layer_sizes=config["q_layers"],
            dropout_rate=config.get("q_do", 0.05)
        )
        self.eta_net = FFNet(
            input_dim=s_dim,
            output_dim=num_a,
            layer_sizes=config["eta_layers"],
            dropout_rate=config.get("eta_do", 0.05)
        )
        self.w_net = FFNet(
            input_dim=s_dim,
            output_dim=1,
            layer_sizes=config["w_layers"],
            dropout_rate=config.get("w_do", 0.05)
        )
        # self.pos_head = nn.Softplus(beta=0.2)
        self.pos_head = lambda x_: x_.abs()

    def forward(self, s, a=None, ss=None, pi_ss=None, calc_q=False,
                calc_v=False, calc_eta=False, calc_w=False):

        if calc_q:
            assert a is not None
            q_all = self.pos_head(self.q_net(s))
            q = q_all.gather(dim=-1, index=a.unsqueeze(-1))
        else:
            q = None

        if calc_v:
            v_all = self.pos_head(self.q_net(ss))
            v = v_all.gather(dim=-1, index=pi_ss.unsqueeze(-1))
        else:
            v = None

        if calc_eta:
            a_probs = torch.softmax(self.eta_net(s), dim=-1)
            eta = a_probs.gather(dim=-1, index=a.unsqueeze(-1)) ** -1
        else:
            eta = None

        if calc_w:
            w = self.pos_head(self.w_net(s))
        else:
            w = None

        return q, v, eta, w


class FeedForwardNuisanceModel(AbstractNuisanceModel):
    def __init__(self, s_dim, num_a, gamma, config, device=None):
        super().__init__(s_dim, num_a)
        self.gamma = gamma
        self.config = config
        self.device = device
        self.reset_networks()

    def reset_networks(self):
        self.net = FFNuisanceModule(
            s_dim=self.s_dim, num_a=self.num_a, config=self.config,
            gamma=self.gamma, device=self.device,
        )
        self.beta_net = FFNBetaModule(
            s_dim=self.s_dim, num_a=self.num_a, config=self.config,
            gamma=self.gamma, device=self.device,
        )   
        if self.device is not None:
            self.net.to(self.device)
            self.beta_net.to(self.device)
        self.net.eval()
        self.beta_net.eval()

    def to(self, device):
        if device is not None:
            self.net.to(device)
            self.beta_net.to(device)
            self.device = device

    def get_q(self, s, a):
        q, _, _, _ = self.net(s, a, calc_q=True)
        return q

    def get_q_v_beta(self, s, a, ss, pi_ss):
        q, v, _, _ = self.net(
            s, a, ss, pi_ss,
            calc_q=True, calc_v=True, 
        )
        beta = self.beta_net(s, a)
        return q, v, beta

    def get_v_beta(self, s, a, ss, pi_ss):
        _, v, _, _ = self.net(
            s, a, ss, pi_ss,
            calc_v=True, 
        )
        beta = self.beta_net(s, a)
        return v, beta

    def get_eta(self, s, a):
        _, _, eta, _ = self.net(s, a, calc_eta=True)
        return eta

    def get_w(self, s):
        _, _, _, w = self.net(s, calc_w=True)
        return w

    def get_beta(self, s, a):
        return self.beta_net(s, a)

    def get_all(self, s, a, ss, pi_ss):
        beta = self.beta_net(s, a)
        q, v, eta, w = self.net(
            s, a, ss, pi_ss, calc_q=True, calc_v=True, 
            calc_eta=True, calc_w=True,
        )
        return q, v, beta, eta, w

    def get_init_kwargs(self):
        return {
            "s_dim": self.s_dim, "num_a": self.num_a,
            "gamma": self.gamma, "config": self.config,
        }

    def get_parameters(self):
        return self.net.parameters()

    def get_beta_parameters(self):
        return self.beta_net.parameters()

    def get_model_state(self):
        return self.net.state_dict()

    def get_beta_state(self):
        return self.beta_net.state_dict()

    def set_model_state(self, state_dict):
        self.net.load_state_dict(state_dict)

    def set_beta_state(self, state_dict):
        self.beta_net.load_state_dict(state_dict)

    def train(self):
        self.net.train()
        self.beta_net.train()

    def eval(self):
        self.net.eval()
        self.beta_net.eval()