import torch
import torch.nn as nn
import math

EPS = 1e-4


class Gamma(nn.Module):
    def __init__(self):
        super(Gamma, self).__init__()
        pass

    def gamma(self, t):
        return torch.zeros_like(t)

    def gamma_(self, t):
        return torch.zeros_like(t)

    def gamma_gamma(self, t):
        return self.gamma(t) * self.gamma_(t)


class SqrtGamma(Gamma):
    def __init__(self, coeff=2.):
        super(SqrtGamma, self).__init__()
        self.coeff = coeff

    def gamma(self, t):
        return torch.sqrt(self.coeff * t * (1. - t))

    def gamma_(self, t):
        return self.gamma_gamma(t) / (self.gamma(t) + EPS)

    def gamma_gamma(self, t):
        return 0.5 * self.coeff * (1. - 2 * t)


class QuadGamma(Gamma):
    def __init__(self):
        super(QuadGamma, self).__init__()
        pass

    def gamma(self, t):
        return t * (1-t)

    def gamma_(self, t):
        return 1. - 2 * t


class TrigGamma(Gamma):
    def __init__(self):
        super(TrigGamma, self).__init__()
        pass

    def gamma(self, t):
        return torch.sin(math.pi * t) ** 2

    def gamma_(self, t):
        return 2 * math.pi * torch.sin(math.pi * t) * torch.cos(math.pi * t)
