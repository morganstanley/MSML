from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class AbstractCritic(ABC, nn.Module):
    def __init__(self, s_dim, num_a):
        super().__init__()
        self.s_dim = s_dim
        self.num_a = num_a

    @abstractmethod
    def get_q(self, s, a):
        pass

    @abstractmethod
    def get_eta(self, s, a):
        pass

    @abstractmethod
    def get_w(self, s):
        pass

    @abstractmethod
    def get_all(self, s, a):
        pass