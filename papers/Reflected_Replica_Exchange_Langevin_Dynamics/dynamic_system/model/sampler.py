import autograd.numpy as np
from autograd import grad
from autograd.numpy import log, sqrt, sin, cos, exp, pi, prod
from autograd.numpy.random import normal, uniform
from scipy.spatial.distance import pdist, squareform
import sys


class SGLD:
    def __init__(self, dim=None, xinit=None, batch_size=128, lr=0.1, T=1.0, decay_lr=100.,
                 myHelp=None, flags=None):
        self.dim = dim
        self.lr = lr
        self.T = T
        self.decay_lr = decay_lr
        self.batch_size = batch_size

        # initialization for SGLD
        if xinit is not None:
            self.x = np.array(xinit)
        else:
            self.x = np.zeros(dim).reshape(1, -1)

        if myHelp is not None:
            self.myHelper = myHelp
        else:
            self.myHelper = None

    def reflection(self, beta):
        if self.myHelper.inside_domain():
            return beta
        else:
            reflected_points = self.myHelper.get_reflection(beta)
            """ when reflection fails in extreme cases (disappear with a small learning rate) """
            return reflected_points

    def stochastic_grad(self, Theta, dxdt, para):
        return np.dot(np.transpose(Theta), np.dot(Theta, para) - dxdt) / self.batch_size

    def stochastic_f(self, Theta, dxdt, para):
        return np.sqrt(np.mean((np.dot(Theta, para)-dxdt) ** 2))

    def update(self, x, y):
        proposal = self.x - self.lr * self.stochastic_grad(x, y, self.x) + sqrt(2. * self.lr * self.T) * normal(
            size=self.dim)
        if self.myHelper is not None:
            self.x = self.reflection(proposal)
        else:
            self.x = proposal


class cyclicSGLD(SGLD):
    def __init__(self, dim=None, xinit=None,
                 batch_size=128, lr=0.1, T=1.0, M=4, total_epoch=4e5, myHelp=None):
        super(cyclicSGLD, self).__init__(dim=dim, xinit=xinit, batch_size=batch_size, lr=lr, T=T, myHelp=myHelp)

        self.M = M
        self.total_epoch = total_epoch
        self.lr0 = lr

    def adjust_learning_rate(self, iters):
        cos_inner = np.pi * (iters % (self.total_epoch * self.batch_size // self.M))
        cos_inner /= self.total_epoch * self.batch_size // self.M
        cos_out = np.cos(cos_inner) + 1
        lr = cos_out * self.lr0 / 2
        return np.max([lr, 1e-7])

    def update(self, x, y, iters):
        self.lr = self.adjust_learning_rate(iters)
        proposal = self.x - self.lr * self.stochastic_grad(x, y, self.x) + \
                   sqrt(2. * self.lr * self.T) * normal(size=self.dim)
        if self.myHelper is not None:
            self.x = self.reflection(proposal)
        else:
            self.x = proposal


class reSGLD(SGLD):
    def __init__(self, dim=None, xinit=None, lr=0.1, decay_lr=100., batch_size=1024,
                 myHelp=None, momentum=0.9, wdecay=5e-4, total=50000, n_chain=2,
                 flags=None):
        super(reSGLD, self).__init__(dim=dim, xinit=xinit, batch_size=batch_size, lr=lr, myHelp=myHelp, flags=flags)

        self.batch_size = batch_size
        self.n_chain = n_chain
        self.x_sub = []
        self.lr = []
        for i in range(n_chain):
            self.x_sub.append(np.array(xinit))
            # self.lr.append(lr)
            self.lr.append(lr * 10 ** i)

        self.T = []
        if flags is None:
            self.T.append(1)
            self.T.append(10)
            self.hat_var = 10
            self.threshold = 3e-4
        else:
            try:
                self.T.append(flags.T_low)
                self.T.append(flags.T_high)
                self.hat_var = flags.hat_var
                self.threshold = flags.threshold
            except:
                self.T.append(1)
                self.T.append(10)
                self.hat_var = 10
                self.threshold = 1e-3

        self.total = total
        self.swap_total = 0
        self.no_swap_total = 0

    def set_T(self, factor=1):
        self.T /= factor
        self.scale = self.sigma * np.sqrt(self.T)

    def correction(self, x, y):
        loss_diff_correct = self.stochastic_f(x, y, self.x_sub[1]) - self.stochastic_f(x, y, self.x_sub[0]) - (
                1 / self.T[1] - 1 / self.T[0]) * self.hat_var
        return np.exp((1 / self.T[1] - 1 / self.T[0]) * loss_diff_correct)

    def update(self, x, y):

        for i in range(self.n_chain):
            proposal = self.x_sub[i] - self.lr[i] * self.stochastic_grad(x, y, self.x_sub[i]) \
                       + sqrt(2. * self.lr[i] * self.T[i]) * normal(size=self.dim)
            if self.myHelper is not None:
                self.x_sub[i] = self.reflection(proposal)
            else:
                self.x_sub[i] = proposal

        """ Swap """
        integrand_corrected = min(1, self.correction(x, y))
        if self.threshold <= integrand_corrected:
            sub = self.x_sub[1]
            self.x_sub[1] = self.x_sub[0]
            self.x_sub[0] = sub
            self.swap_total += 1
        else:
            self.no_swap_total += 1

        self.x = self.x_sub[0]
