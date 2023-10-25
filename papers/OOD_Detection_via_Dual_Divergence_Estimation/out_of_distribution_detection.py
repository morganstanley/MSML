import math
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import scipy.special as ss

import torch
from torch import nn, optim
from torch.nn import functional as F


class KLDivDualFuncRepresenter(nn.Module):

    def __init__(
            self,
            input_size,
            num_layers=3,
            hidden_size=64,
            std=0.15,
            dropout=0.0,
    ):
        super().__init__()

        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.std = std
        self.dropout = dropout

        self.net = nn.Sequential()
        for j in range(self.num_layers):
            curr_layer = nn.Linear(hidden_size if j > 0 else input_size, hidden_size if j < (self.num_layers -1) else 1)
            nn.init.normal_(curr_layer.weight, std=self.std)
            nn.init.constant_(curr_layer.bias, 0)
            self.net.append(curr_layer)
            curr_layer = None

    def forward(self, x):

        assert x.ndim == 2
        
        for j in range(self.num_layers):
            x = self.net[j](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if j < (self.num_layers-1):
                x = F.relu(x)

        return x


def compute_donsker_varadhan_representation_of_kl_div(ft, c, is_optimization_else_estimation=False, is_max_approx=True):

    assert ft.ndim <= 2
    if ft.ndim == 2:
        assert ft.shape[1] == 1

    assert c.dtype == np.bool

    if c.sum() in [0, c.size]:
        if is_optimization_else_estimation:
            return torch.tensor(-math.inf, device=ft.device)
        else:
            return -math.inf

    ft_p = ft[c]
    ft_q = ft[~c]

    if is_optimization_else_estimation:
        if is_max_approx:
            kl_div = torch.mean(ft_p) - torch.max(ft_q) + torch.log(torch.tensor(ft_q.shape[0]))
        else:
            kl_div = torch.mean(ft_p) - torch.logsumexp(ft_q, dim=0) + torch.log(torch.tensor(ft_q.shape[0]))
    else:
        assert isinstance(ft_p, (np.ndarray, np.generic))
        assert isinstance(ft_q, (np.ndarray, np.generic))

        if is_max_approx:
            kl_div = ft_p.mean() - np.max(ft_q) + math.log(ft_q.size)
        else:
            kl_div = ft_p.mean() - ss.logsumexp(ft_q) + math.log(ft_q.size)

    return kl_div


def detect_ood_per_dual_func_of_kl_div_of_test_data_wrt_trn_data(f_train, f_test, is_smooth_max):

    # it is assumed that f comes from estimating KL-D of test data w.r.t. training data in its dual form.
    # more of sampling from bins where present episode lands, no optimization

    assert (f_train.ndim == 1) and (f_test.ndim == 1)
    assert (not np.all(np.isnan(f_train))) and (not np.all(np.isnan(f_test)))

    if is_smooth_max:
        max_f_train = ss.logsumexp(f_train)
    else:
        max_f_train = f_train.max()

    ood_idx = np.where(f_test > max_f_train)[0]

    return ood_idx


class DistributionalDriftEstimator:

    def __init__(
            self, seed=0, debug=False,
            hidden_size=64,
            std=0.1,
            dropout=0.0,
            num_hidden_layers=3,
            num_iter=1000,
            lr=1e-4,
            batch_size=10000,
            is_smooth_max=False,
    ):
        self.seed = seed
        self.hidden_size = hidden_size
        self.std = std
        self.dropout = dropout
        self.num_hidden_layers = num_hidden_layers
        self.num_iter = num_iter
        self.lr = lr
        self.batch_size = batch_size
        self.is_smooth_max = is_smooth_max

        self.sampler = np.random.RandomState(seed=self.seed)
        self.data_sampler = np.random.RandomState(seed=self.seed)
        self.debug = debug

    def __compute_kl_div_dual_func_for_eval__(self, model, X):

        if not torch.is_tensor(X):
            X = torch.FloatTensor(X.copy())
        
        training = model.training

        if training:
            model.eval()

        f = model(X).detach().cpu().numpy().flatten()

        if training:
            model.train()

        return f

    def __get_model__(self, input_shape):

        torch.manual_seed(self.seed)
        assert len(input_shape) == 2

        return KLDivDualFuncRepresenter(
            input_size=input_shape[-1],
            hidden_size=self.hidden_size,
            std=self.std,
            num_layers=self.num_hidden_layers,
            dropout=self.dropout,
        )

    def ood_detection_from_dual_divergence_estimation(self, X_train, X_test,
            return_kld=False,
            device='cpu',
    ):

        assert X_train.ndim in [2, 3, 4], X_train.ndim
        assert X_test.ndim == X_train.ndim
        assert X_train.shape[1:] == X_test.shape[1:]

        X_train = torch.FloatTensor(X_train).to(device)
        X_test = torch.FloatTensor(X_test).to(device)
        assert X_train.shape[1:] == X_test.shape[1:]

        model = self.__get_model__(input_shape=X_train.shape).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        batch_sampler = npr.RandomState(seed=self.seed)
        batch_loss = np.zeros(self.num_iter)
        batch_ood_frac = np.zeros(self.num_iter, dtype=np.float)

        iters_ood_frac = {}
        iters_kld = {}

        model.train()

        for i in range(self.num_iter):

            if self.debug:
                print('.', end='')

            sel_train_idx = batch_sampler.choice(X_train.shape[0], size=min(self.batch_size, X_train.shape[0]))
            sel_test_idx = batch_sampler.choice(X_test.shape[0], size=min(self.batch_size, X_test.shape[0]))

            xt = torch.vstack((X_test[sel_test_idx], X_train[sel_train_idx]))
            ct = np.concatenate((np.ones(sel_test_idx.size, dtype=np.bool), np.zeros(sel_train_idx.size, dtype=np.bool)))

            f_t = model(xt)

            # maximizing Donsker Varadhan representation of KL-D, i.e. estimation of KL-D
            loss = - compute_donsker_varadhan_representation_of_kl_div(
                ft=f_t,
                c=ct,
                is_optimization_else_estimation=True,
                is_max_approx=False,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss[i] = loss.item()

            f_t = f_t.detach().cpu().numpy().flatten()
            batch_ood_frac[i] = detect_ood_per_dual_func_of_kl_div_of_test_data_wrt_trn_data(
                f_train=f_t[sel_test_idx.size:], f_test=f_t[:sel_test_idx.size],
                is_smooth_max=self.is_smooth_max,
            ).size/float(sel_test_idx.size)

        if self.debug:
            plt.close()
            plt.plot(-batch_loss, '.', color='black', label='KL(test||train)')
            # plt.yscale('symlog')
            plt.legend()
            plt.show()

            plt.close()
            plt.plot(batch_ood_frac, '.', color='black', label='OOD')
            plt.ylabel('Frac. OOD')
            # plt.yscale('symlog')
            plt.legend()
            plt.show()

        model.eval()
        f_train = self.__compute_kl_div_dual_func_for_eval__(model=model, X=X_train)
        f_test = self.__compute_kl_div_dual_func_for_eval__(model=model, X=X_test)

        ood_test_idx = detect_ood_per_dual_func_of_kl_div_of_test_data_wrt_trn_data(
            f_train=f_train, f_test=f_test,
            is_smooth_max=self.is_smooth_max,
        )

        frac_ood = ood_test_idx.size/float(X_test.shape[0])

        if self.debug:

            plt.close()
            plt.plot(f_train, np.zeros(f_train.size), '.', color='navy', label='Train')
            f_test_ood = f_test[f_test > f_train.max()]
            plt.plot(f_test_ood, np.ones(f_test_ood.size), '.', color='crimson', label='Test-OOD')
            f_test_in_dist = f_test[f_test <= f_train.max()]
            plt.plot(f_test_in_dist, np.ones(f_test_in_dist.size), '.', color='cyan', label='Test-OOD')
            plt.title(f'Frac. of OOD={round(frac_ood, 2)}')
            plt.legend()
            plt.show()

        if return_kld:
            return frac_ood, iters_ood_frac, iters_kld
        else:
            return frac_ood, iters_ood_frac
