import autograd.numpy as np
from autograd import grad
from autograd.numpy import log, sqrt, sin, cos, exp, pi, prod
from autograd.numpy.random import normal, uniform
from scipy.spatial.distance import pdist, squareform
import sys

            
class SGLD:
    def __init__(self, f=None, dim=None, xinit=None, lr=0.1, T=1.0, decay_lr=None, decay_after=5000,
                 l2=None, myHelp=None):
        self.f = f
        self.dim = dim
        self.lr = lr
        self.T = T
        self.decay_lr = decay_lr

        # initialization for SGLD
        if xinit is not None:
            self.x = np.array(xinit)
        else:
            self.x = np.zeros(dim).reshape(1, -1)

        if myHelp is not None:
            self.myHelper = myHelp
        else:
            self.myHelper = None

        if l2 is None:
            self.l2 = 0.0
        else:
            self.l2 = l2

    def reflection(self, prev_beta, beta):
        if self.myHelper.inside_domain(beta):
            return beta
        else:
            reflected_points, boundary = self.myHelper.get_reflection(prev_beta, beta)
            """ when reflection fails in extreme cases (disappear with a small learning rate) """
            if not self.myHelper.inside_domain(reflected_points):
                return boundary
            return reflected_points.reshape(-1)
    
    def update_lr(self):
        self.lr *= self.decay_lr
        
    def stochastic_grad(self, beta):
        # l2_grad = 2 * self.l2 * np.array(beta)
        l2_grad = 0
        return grad(self.f)(beta) + 0.05 * normal(size=self.dim) + l2_grad

    def stochastic_f(self, beta):
        # l2_reg_term = self.l2 * np.sum(np.square(beta))
        l2_reg_term = 0
        return self.f(beta.tolist()) + 0.05 * normal(size=1) + l2_reg_term

    def update(self):
        # if self.decay_lr is not None:
        #     self.update_lr()
        
        proposal = self.x - self.lr * self.stochastic_grad(self.x) + sqrt(2. * self.lr * self.T) * normal(size=self.dim)
        if self.myHelper is not None:
            self.x = self.reflection(self.x, proposal)
        else:
            self.x = proposal


class cyclicSGLD(SGLD):
    def __init__(self, f=None, dim=None, xinit=None, lr=0.1, T=1.0, M=20, total_epoch=4e5, l2=None, myHelp=None):
        super(cyclicSGLD, self).__init__(f=f, dim=dim, xinit=xinit, lr=lr, T=T, l2=l2, myHelp=myHelp)
        
        # self.f = f
        # self.dim = dim
        # self.T = T
        self.M = M
        self.total_epoch = total_epoch
        self.lr0 = lr
        # self.lr = lr

        # initialization for SGLD
        # self.x = np.array(xinit)

        # if myHelp is not None:
        #     self.myHelper = myHelp
        # else:
        #     self.myHelper = None
            
        # if l2 is None:
        #     self.l2 = 0.0
        # else:
        #     self.l2 = l2

    def adjust_learning_rate(self, iters):
        cos_inner = np.pi * (iters % (self.total_epoch // self.M))
        cos_inner /= self.total_epoch // self.M
        cos_out = np.cos(cos_inner) + 1
        lr = 0.5 * cos_out * self.lr0
        return np.max([lr, 1e-4])

    def update(self, iters):
        self.lr = self.adjust_learning_rate(iters)
        proposal = self.x - self.lr * self.stochastic_grad(self.x) + sqrt(2. * self.lr * self.T) * normal(size=self.dim)
        if self.myHelper is not None:
            self.x = self.reflection(self.x, proposal)
        else:
            self.x = proposal


class reSGLD(SGLD):
    def __init__(self, f=None, dim=None, xinit=None, lr=0.1, lr_gap=2.0, T=[1.0, 10.0], decay_lr=None,
                 myHelp=None, momentum=0.9, wdecay=5e-4, total=50000, n_chain=2,
                 l2=None, flags=None):

        super(reSGLD, self).__init__(f=f, dim=dim, xinit=xinit, lr=lr, decay_lr=decay_lr, l2=l2, myHelp=myHelp)
        
        # self.f = f
        # self.dim = dim
        self.decay_lr = decay_lr

        # initialization for SGLD
        # self.x = np.array(xinit)
        self.n_chain = n_chain
        self.x_sub = []
        self.lr = []
        for i in range(n_chain):
            self.x_sub.append(np.array(xinit))
            # self.lr.append(lr)
            self.lr.append(lr * lr_gap ** i)

        self.momentum = momentum
        self.T = []
        if flags is None:
            self.T.append(1)
            self.T.append(10)
            self.hat_var = 7
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
                self.hat_var = 7
                self.threshold = 3e-4
                print("flags failed")

        self.wdecay = wdecay
        self.V = 0.1
        self.velocity = []
        # self.criterion = criterion
        self.total = total

        self.beta = 0.5 * self.V * lr
        self.alpha = 1 - self.momentum

        if self.beta > self.alpha:
            sys.exit('Momentum is too large')


        self.swap_total = 0
        self.no_swap_total = 0
            

    def update_lr(self):
        for i in range(self.n_chain):
            self.lr[i] *= self.decay_lr
        
    def set_T(self, factor=1):
        self.T /= factor
        self.scale = self.sigma * np.sqrt(self.T)

    def correction(self):
        loss_diff_correct = self.stochastic_f(self.x_sub[1]) - self.stochastic_f(self.x_sub[0]) - (
                1 / self.T[1] - 1 / self.T[0]) * self.hat_var
        return np.exp((1 / self.T[1] - 1 / self.T[0]) * loss_diff_correct)

    def update(self):

        # if self.decay_lr is not None:
        #     self.update_lr()
        
        for i in range(self.n_chain):
            # self.x_sub[i] = self.x_sub[i] - self.lr[i] * self.stochastic_grad(self.x_sub[i]) \
            proposal = self.x_sub[i] - self.lr[i] * self.stochastic_grad(self.x_sub[i]) \
                       + sqrt(2. * self.lr[i] * self.T[i]) * normal(size=self.dim)
            if self.myHelper is not None:
                self.x_sub[i] = self.reflection(self.x_sub[i], proposal)
            else:
                self.x_sub[i] = proposal

        """ Swap (quasi-buddle sort included) """
        integrand_corrected = min(1, self.correction())
        if self.threshold <= integrand_corrected:
            sub = self.x_sub[1]
            self.x_sub[1] = self.x_sub[0]
            self.x_sub[0] = sub
            self.swap_total += 1
        else:
            self.no_swap_total += 1

        self.x = self.x_sub[0]

class Sampler:
    def __init__(self, f=None, dim=None, boundary=None, xinit=None, partition=None, lr=0.1, T=1.0, zeta=1, mu=0.1,
                 decay_lr=100., parts=100, helper=None):

        self.f = f

        self.dim = dim
        self.lr = lr
        self.T = T
        self.partition = partition
        self.boundary = boundary
        self.xinit = np.array(xinit)
        self.zeta = zeta
        self.decay_lr = decay_lr
        self.parts = parts

        # baseline SGLD
        self.sgld_beta = self.xinit

        # cyclic SGLD
        self.cycsgld_beta = self.xinit
        self.r_remainder = 0

        # reflected replica exchange SGLD
        self.resgld_beta_high = self.xinit
        self.resgld_beta_low = self.xinit
        self.threshold = 3e-4
        self.swaps = 0
        self.frozen = 0
        self.frozen_threshold = 1
        
        self.myHelper = helper

    def reflection(self, prev_beta, beta):
        if self.myHelper.inside_domain(beta):
            return beta
        else:
            reflected_points, boundary = self.myHelper.get_reflection(prev_beta, beta)
            """ when reflection fails in extreme cases (disappear with a small learning rate) """
            if not self.myHelper.inside_domain(reflected_points):
                return boundary
            return reflected_points.reshape(-1)

    def in_domain(self, beta):
        return sum(map(lambda i: beta[i] < self.boundary[0] or beta[i] > self.boundary[1], range(self.dim))) == 0

    def stochastic_grad(self, beta):
        return grad(self.f)(beta) + 0.05 * normal(size=self.dim)

    def stochastic_f(self, beta):
        return self.f(beta.tolist()) + 0.05 * normal(size=1)

    def sgld_step(self):
        proposal = self.sgld_beta - 1.0 * self.lr * self.stochastic_grad(self.sgld_beta) + sqrt(
            2 * self.lr * self.T) * normal(size=self.dim)

        if self.myHelper is not None:
            self.sgld_beta = self.reflection(self.sgld_beta, proposal)
        else:
            self.sgld_beta = proposal

    def cycsgld_step(self, iters=1, cycles=5, total=5e5):
        sub_total = total / cycles
        self.r_remainder = (iters % sub_total) * 1.0 / sub_total
        cyc_lr = max(2.0 * self.lr * (cos(pi * self.r_remainder) + 1), 0.5 * self.lr)
        # cyc_lr = 5.0 * self.lr * (cos(pi * self.r_remainder) + 1)
        proposal = (self.cycsgld_beta - cyc_lr * self.stochastic_grad(self.cycsgld_beta) +
                    sqrt(2 * cyc_lr * self.T) * normal(size=self.dim))

        if self.myHelper is not None:
            self.cycsgld_beta = self.reflection(self.cycsgld_beta, proposal)
        else:
            self.cycsgld_beta = proposal

    def resgld_step(self, lr_multiply=5.0, T_multiply=10.0, var=9.0):
        proposal_low = (self.resgld_beta_low - self.lr * self.stochastic_grad(self.resgld_beta_low) +
                        sqrt(2 * self.lr * self.T) * normal(size=self.dim))

        proposal_high = (self.resgld_beta_high - lr_multiply * self.lr * self.stochastic_grad(self.resgld_beta_high) +
                         sqrt(2 * self.lr * lr_multiply * self.T * T_multiply) * normal(size=self.dim))

        if self.myHelper is not None:
            self.resgld_beta_high = self.reflection(self.resgld_beta_high, proposal_high)
            self.resgld_beta_low = self.reflection(self.resgld_beta_low, proposal_low)
        else:
            self.resgld_beta_high = proposal_high
            self.resgld_beta_low = proposal_low

        loss_diff_correct = self.stochastic_f(self.resgld_beta_high) - self.stochastic_f(self.resgld_beta_low) - (
                1 / (self.T * T_multiply) - 1 / self.T) * var
        correction = np.exp((1 / (self.T * T_multiply) - 1 / self.T) * loss_diff_correct)
        integrand_corrected = min(1, correction)

        if self.threshold <= integrand_corrected:
            sub = self.resgld_beta_high
            self.resgld_beta_high = self.resgld_beta_low
            self.resgld_beta_low = sub
            self.swaps += 1

