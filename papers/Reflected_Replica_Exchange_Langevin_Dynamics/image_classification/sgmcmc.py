"""
Created on May 12, 2024
@author: Haoyang Zheng & Wei Deng
Code for Constrained Exploration via Reflected Replica Exchange Stochastic Gradient Langevin Dynamics. ICML 2024
"""

import sys, copy
import numpy as np
import torch
import random
from torch.autograd import Variable


def reflect(x, bound=4.0):
    reflection = torch.where(torch.abs(x) <= bound, x, torch.sign(x) * bound * 2 - x)
    return reflection


class Sampler:
    def __init__(self, net, criterion,
                 momentum=0.9, lr=0.1, wdecay=5e-4, T=0.05, total=50000,
                 domain=False, bound=1.0, regularization=None):
        self.net = net
        self.eta = lr
        self.momentum = momentum
        self.T = T
        self.wdecay = wdecay
        self.V = 0.1
        self.velocity = []
        self.criterion = criterion
        self.total = total

        self.beta = 0.5 * self.V * self.eta
        self.alpha = 1 - self.momentum

        if self.beta > self.alpha:
            sys.exit('Momentum is too large')

        self.sigma = np.sqrt(2.0 * self.eta * (self.alpha - self.beta))
        self.scale = self.sigma * np.sqrt(self.T)

        if domain is True:
            self.domain = True
            self.bound = bound
        else:
            self.domain = False

        for param in net.parameters():
            p = torch.zeros_like(param.data)
            self.velocity.append(p)

        if regularization is None:
            self.regularization = 0.0
        else:
            self.regularization = regularization

    def set_T(self, factor=1):
        self.T /= factor
        self.scale = self.sigma * np.sqrt(self.T)

    def set_eta(self, eta):
        self.eta = eta
        self.beta = 0.5 * self.V * self.eta
        self.sigma = np.sqrt(2.0 * self.eta * (self.alpha - self.beta))
        self.scale = self.sigma * np.sqrt(self.T)

    def backprop(self, x, y):
        self.net.zero_grad()
        """ convert mean loss to sum losses """
        loss = self.criterion(self.net(x), y) * self.total

        if self.regularization != 0:
            l2_loss = torch.mean(torch.tensor([param.pow(2.0).mean() for param in self.net.parameters()]))
            loss = loss + 0.5 * self.regularization * l2_loss

        loss.backward()
        return loss

    def step(self, x, y):
        loss = self.backprop(x, y)
        for i, param in enumerate(self.net.parameters()):

            proposal = torch.cuda.FloatTensor(param.data.size()).normal_().mul(self.scale)
            grads = param.grad.data
            if self.wdecay != 0:
                grads.add_(self.wdecay, param.data)
            self.velocity[i].mul_(self.momentum).add_(-self.eta, grads).add_(proposal)
            param.data.add_(self.velocity[i])

            if self.domain is True:
                param.data = reflect(param.data, self.bound)

        return loss.data.item()
