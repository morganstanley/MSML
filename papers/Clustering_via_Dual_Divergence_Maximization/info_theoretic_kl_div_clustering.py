import time
import math
import numpy as np
import numpy.random as npr

import matplotlib.pyplot as plt
import seaborn as sns

import scipy.special as scipy_special

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMMultiLayer(nn.Module):

    def __init__(
            self, input_size,
            hidden_size=32,
            std=0.1,
            dropout=0.0,
            bidirectional=False,
            num_layers=3,
    ):

        super(LSTMMultiLayer, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.std = std
        self.num_layers = num_layers
        
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=self.num_layers,
            bias=True,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.normal_(param, std=std)
            else:
                raise AssertionError(name)

        if bidirectional:
            self.out = nn.Linear(hidden_size*2, 1, bias=True)
        else:
            self.out = nn.Linear(hidden_size, 1, bias=True)

        nn.init.normal_(self.out.weight, std=std)
        nn.init.constant_(self.out.bias, 0)

    def forward(self, input):

        assert input.ndim == 3
        assert input.shape[-1] == self.input_size

        out, _ = self.lstm(input)
        out = self.out(out[:, -1, :])

        assert out.ndim == 2
        assert out.shape[1] == 1
        
        return out


class Transformer(nn.Module):

    def __init__(self,
        input_size,
        num_steps,
        hidden_size=32,
        num_layers=3,
        dropout=0.0,
        nhead=10,
        std= 0.1,
    ):

        super().__init__()
                
        self.input_size = input_size
        self.num_steps = num_steps
        self.hidden_size = hidden_size
        self.num_layers_per_encoder = num_layers
        self.nhead = nhead
        self.std = std
        self.dropout = dropout

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=self.nhead,
            dim_feedforward=hidden_size,
            batch_first=True,
            dropout=self.dropout,
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers_per_encoder)
        for name, param in self.transformer_encoder.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.normal_(param, std=self.std)
            else:
                raise AssertionError(name)

        self.hidden = nn.Linear(input_size, input_size, bias=True)
        nn.init.normal_(self.hidden.weight, std=self.std)
        nn.init.constant_(self.hidden.bias, 0)

        self.out = nn.Linear(
            int(input_size*num_steps),
            1, bias=True,
        )
        nn.init.normal_(self.out.weight, std=self.std)
        nn.init.constant_(self.out.bias, 0)

    def forward(self, input):

        device = input.device

        assert input.ndim == 3
        assert input.shape[-1] == self.input_size
        assert input.shape[1] == self.num_steps
        n = input.shape[0]

        input = self.transformer_encoder(input)

        hidden = torch.zeros((n, input.shape[1], self.input_size), device=device)
        for j in range(self.num_steps):
            hidden[:, j, :] = torch.squeeze(F.relu(self.hidden(input[:, [j], :])), dim=1)

        hidden = hidden.view(n, -1)

        out = self.out(hidden)
                
        assert out.ndim == 2
        assert out.shape[1] == 1
        
        return out


class FeedForwardNN(nn.Module):

    def __init__(
            self,
            input_size,
            num_layers=3,
            hidden_size=128,
            std=0.1,
            dropout=0.0,
    ):
        super().__init__()

        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.std = std
        self.dropout = dropout

        assert self.num_layers > 1

        self.net = nn.Sequential()
        for j in range(self.num_layers):
            curr_layer = nn.Linear(hidden_size if j > 0 else input_size, hidden_size if j < (self.num_layers -1) else 1)
            nn.init.normal_(curr_layer.weight, std=std)
            nn.init.constant_(curr_layer.bias, 0)
            self.net.append(curr_layer)

    def forward(self, x):

        assert x.dim() == 2, x.shape

        for j in range(self.num_layers):
            x = self.net[j](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if j < (self.num_layers-1):
                x = F.relu(x)

        if x.dtype == torch.float:
            x = x.double()

        return x


def get_model(input_shape, parameters, nn='transformer', seed=0):

    assert len(input_shape) in [2, 3], input_shape

    torch.manual_seed(seed)

    if nn == 'transformer':
        assert len(input_shape) == 3
        return Transformer(
            hidden_size=parameters['hidden_size'],
            num_layers=parameters['num_layers'],
            dropout=parameters['dropout'],
            nhead=parameters['nhead'] if 'nhead' in parameters else 10,
            input_size=input_shape[-1],
            num_steps=input_shape[1],
        )
    elif nn == 'lstm':
        assert len(input_shape) == 3
        return LSTMMultiLayer(
            input_size=input_shape[-1],
            hidden_size=parameters['hidden_size'],
            std=parameters['std'],
            num_layers=parameters['num_layers'],
            dropout=parameters['dropout'],
        )
    elif nn == 'fnn':
        assert len(input_shape) == 2, input_shape
        return FeedForwardNN(
            input_size=input_shape[-1],
            hidden_size=parameters['hidden_size'],
            std=parameters['std'],
            num_layers=parameters['num_layers'],
            dropout=parameters['dropout'],
        )
    else:
        raise NotImplemented


class NeuralKLDivClustering:
    # todo: add algo for greedy cuts

    def __init__(
            self, seed=0, debug=False, min_frac_of_cluster_size=0.1
    ):

        self.seed = seed
        self.debug = debug
        # for regularizing the optimization of clusters so as to avoid outliers being considered as individual clusters
        self.min_frac_of_cluster_size = min_frac_of_cluster_size
        
    def __compute_donsker_varadhan_representation_of_kl_div__(self, ft_p, ft_q, is_optimization_else_estimation=False, is_max_approx=False):

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
                kl_div = ft_p.mean() - scipy_special.logsumexp(ft_q) + math.log(ft_q.size)

        return kl_div

    def compute_donsker_varadhan_representation_of_kl_div(self, ft, c, is_optimization_else_estimation=False, is_max_approx=False, device='cpu'):

        assert c.dtype == np.bool
        assert c.ndim == 1, c.shape

        if c.sum() in [0, c.size]:
            if is_optimization_else_estimation:
                return torch.tensor(-math.inf, device=device)
            else:
                return -math.inf

        return self.__compute_donsker_varadhan_representation_of_kl_div__(
            ft_p=ft[c],
            ft_q=ft[~c],
            is_optimization_else_estimation=is_optimization_else_estimation,
            is_max_approx=is_max_approx,
        )

    def optimize_cluster_labels_via_cut_point_in_the_dual_space(self, f_pq, f_qp, is_max_approx=True, debug=True):

        # it is highly recommended to use max approximation for estimating the divergence function.

        # all computations are done in numpy here.
        # no need for backpropagation, so no tensors

        # f_pq and f_qp are dual function representations (1-D) of data points.
        # for the optimization of cut point in the dual space so the respective cluster labels, raw features are not required.
        # cluster labels are estimated afresh, without the knowledge of cluster labels from the past iteration.

        assert f_pq.ndim == 1
        assert f_qp.ndim == 1

        assert not np.all(np.isnan(f_pq))
        assert not np.all(np.isnan(f_qp))

        def compute_kl_div_for_cluster_opt(z):

            assert z.dtype == np.bool
            if z.sum() in [0, z.size]:
                return -math.inf

            num_ones = z.sum()
            num_zero = z.size - num_ones

            if is_max_approx:
                kl_pq = f_pq[z].mean() - f_pq[~z].max() + math.log(num_zero)
                kl_qp = f_qp[~z].mean() - f_qp[z].max() + math.log(num_ones)
            else:
                kl_pq = f_pq[z].mean() - scipy_special.logsumexp(f_pq[~z]) + math.log(num_zero)
                kl_qp = f_qp[~z].mean() - scipy_special.logsumexp(f_qp[z]) + math.log(num_ones)
            
            kl_div = kl_pq + kl_qp
            
            return kl_div

        n = f_pq.size
        assert f_qp.size == n

        # this is used for regularizing the optimization of clusters in the dual space, so as not to choose anomalies/outliers as individual clusters.
        min_cluster_size = int(self.min_frac_of_cluster_size*n)

        if debug:
            print(f'min_cluster_size={min_cluster_size}')

        # sorting the input data points as per their dual function values
        sort_idx_pq = f_pq.argsort()
        kl_div_for_cluster_configs_pq = np.zeros(n)
        # optimizing cut point in the dual space
        for j in range(n):
            if (j < min_cluster_size) or (j >= (n-min_cluster_size)):
                kl_div_for_cluster_configs_pq[j] = -math.inf
            else:
                z = np.zeros(n, dtype=np.bool)
                # mean is computed on ones for f_pq, which should be on higher values to keep KL-D positive
                z[sort_idx_pq[j:]] = True
                kl_div_for_cluster_configs_pq[j] = compute_kl_div_for_cluster_opt(z=z)

        # cut point with maximal KL-divergence
        cut_in_sort_idx_pq = kl_div_for_cluster_configs_pq.argmax()
        kl_div_opt_pq = kl_div_for_cluster_configs_pq[cut_in_sort_idx_pq]
        # cluster labels as per the cut point
        labels_opt_pq = np.zeros(n, dtype=np.bool)
        labels_opt_pq[sort_idx_pq[cut_in_sort_idx_pq:]] = True

        if debug:
            plt.close()
            plt.plot(f_pq[sort_idx_pq], kl_div_for_cluster_configs_pq, 'b.')
            plt.xlabel(f'f_pq')
            plt.ylabel('KL-D')
            plt.show()

        cut_in_sort_idx_pq, sort_idx_pq = (None,)*2

        # todo: refactor code to avoid redundancies
        # same as above for the opposite direction of divergence function
        sort_idx_qp = f_qp.argsort()
        kl_div_for_cluster_configs_qp = np.zeros(n)
        for j in range(n):
            if (j < min_cluster_size) or (j >= (n-min_cluster_size)):
                kl_div_for_cluster_configs_qp[j] = -math.inf
            else:
                z = np.zeros(n, dtype=np.bool)
                # ~z is used for KL q -> p
                # mean is computed on zeros for f_qp, which should be on higher values to keep KL-D positive
                z[sort_idx_qp[:j]] = True
                kl_div_for_cluster_configs_qp[j] = compute_kl_div_for_cluster_opt(z=z)
        cut_in_sort_idx_qp = kl_div_for_cluster_configs_qp.argmax()
        kl_div_opt_qp = kl_div_for_cluster_configs_qp[cut_in_sort_idx_qp]
        labels_opt_qp = np.zeros(n, dtype=np.bool)
        labels_opt_qp[sort_idx_qp[:cut_in_sort_idx_qp]] = True

        if debug:
            plt.close()
            plt.plot(f_qp[sort_idx_qp], kl_div_for_cluster_configs_qp, 'r.')
            plt.xlabel(f'f_qp')
            plt.ylabel('KL-D')
            plt.show()

        cut_in_sort_idx_qp, sort_idx_qp = (None,)*2

        if debug and (npr.random() < 0.1):

            plt.close()
            plt.scatter(
                x=f_pq,
                y=f_qp,
                c=labels_opt_pq,
            )
            plt.xlabel(r'$f_{p \to q}$', fontsize=16)
            plt.ylabel(r'$f_{q \to p}$', fontsize=16)
            plt.legend()
            plt.title(f'KLpq={kl_div_opt_pq}')
            plt.show()

            plt.close()
            plt.scatter(
                x=f_pq,
                y=f_qp,
                c=labels_opt_qp,
            )
            plt.xlabel(r'$f_{p \to q}$', fontsize=16)
            plt.ylabel(r'$f_{q \to p}$', fontsize=16)
            plt.legend()
            plt.title(f'KLqp={kl_div_opt_qp}')
            plt.show()

        return labels_opt_pq if (kl_div_opt_pq > kl_div_opt_qp) else labels_opt_qp

    def optimize_multiple_clusters_maximize_kl_div_greedy(
            self,
            X_org,
            # total number of iterations for optimizing clusters
            # same parameter value is used across all the datasets.
            # higher value can be used if higher number of clusters to be optimized
            num_iter=100,
            # strongly recommended to use max approximation when estimating KL-Divergence in the dual form
            # need not to tune it, same parameter value is used across all the datasets
            is_max_approx=True,
            # for optimization purpose, high learning rate is preferred when estimating KL-divergence for a given input (or adapted) configuration of clusters
            # need not to tune it, same parameter value is used across most of the datasets
            lr=1e-1,
            # for initialization of neural nets when estimating KL-divergence, same value has been used in other estimators such as in ITENE (conditional MI based estimator of Transfer Entropy)
            # need not to tune it, same parameter value is used across all the datasets
            std=0.1,
            # a few number of layers suffice in the most cases
            num_layers=3,
            # light archtectures are preferred, one could also use different number of units for different layers for further fine tuning
            hidden_size=32,
            # In every iteration, clusters are updated and correspondingly the divergence estimator is adapted in every iteration.
            # if the probability is lower than 1.0 (0, 1), more than one iterations are used to adapt the divergence estimator as per the adapted configuration of clusters.
            cluster_update_prob=1.0,
            # one can manual select the number of clusters as per the pairwise divergence between clusters or other metrics.
            num_clusters=6,
            # first few iterations are only for optimizing the divergence estimator for input cluster labels
            # need not to tune it, same parameter value is used across all the datasets
            num_pure_weight_updates=10,
            # by default, learning rate is fixed across iterations
            # need not to tune it, same parameter value is used across all the datasets
            lr_decay=1.0,
            # no need for dropout as such
            # need not to tune it, same parameter value is used across all the datasets
            dropout=0.0,
            # transformer is the most preferred choice of architecture as tested across various datasets
            nn='transformer',
            # default value of 10 is used across all the datasets
            nhead=10,
            # default is False
            debug=False,
            # device to use for computation (GPU or CPU)
            device='cpu',
    ):
        # this code is generically applicable for clustering of data in other modalities as well
        # maximize w.r.t. neural parameters as well as data itself, moving data points across clusters

        # processing timeseries as a flat vector or as a sequence of timesteps with each one of many observations
        assert X_org.ndim in [2, 3], X_org.ndim

        # number of timeseries to be clustered
        n = X_org.shape[0]

        # initializing as a single cluster with label
        # clusters are added greedily with labels incremented (0, 1, 2, 3, ...)
        c = np.zeros(n, dtype=np.int)
        # random sampler for selecting two clusters for estimating divergence between the two
        cluster_sampler = npr.RandomState(seed=self.seed)
        # random sampler of probability (0, 1)
        unif_prob_sampler = npr.RandomState(seed=self.seed)
        # random sampler for new cluster initialization
        cluster_update_sampler = npr.RandomState(seed=self.seed)

        # numpy to Tensor on GPU
        X_org = torch.FloatTensor(X_org).to(device)

        # greedily add clusters
        while np.unique(c).size < num_clusters:

            # divergence estimators for both directions
            # note: only two divergence estimators are used regardless of the number of clusters
            # furthermore, deep learning facilitates learning divergence estimators generically for different possible configuration of clusters, rather than a single one as one would expect in theory

            model_pq = get_model(
                input_shape = X_org.shape,
                parameters = {
                    'hidden_size': hidden_size,
                    'std': std,
                    'num_layers': num_layers,
                    'dropout': dropout,
                    'nhead': nhead,
                },
                nn=nn,
            ).to(device)
            optimizer_pq = torch.optim.Adam(model_pq.parameters(), lr=lr)

            model_qp = get_model(
                input_shape=X_org.shape,
                parameters = {
                    'hidden_size': hidden_size,
                    'std': std,
                    'num_layers': num_layers,
                    'dropout': dropout,
                    'nhead': nhead,
                },
                nn=nn,
            ).to(device)
            optimizer_qp = torch.optim.Adam(model_qp.parameters(), lr=lr)

            # learning rate decay
            lr *= lr_decay
            if self.debug:
                print(f'lr={lr}')

            # choose cluster for splitting into two
            # todo: compute entropy of clusters if cluster size is small
            unique_c, counts_c = np.unique(c, return_counts=True)
            assert unique_c.min() == 0
            assert unique_c.max() == (unique_c.size - 1)
            cluster_for_split = unique_c[counts_c.argmax()]
            counts_c = None
            sel_cluster_idx = np.where(c == cluster_for_split)[0]
            # randomly initialize
            new_cluster_label = unique_c.size
            c[sel_cluster_idx[cluster_sampler.choice([False, True], sel_cluster_idx.size)]] = new_cluster_label
            unique_c = np.unique(c)

            model_pq.train()
            model_qp.train()

            # loss function as negative of KL-divergence
            batch_loss = np.zeros(num_iter, dtype=np.float)
            # number of iteration to optimize the current cluster
            for i in range(num_iter):

                print('.', end='')

                if unif_prob_sampler.rand() < 0.5:
                    # choose the new cluster as one of the two clusters for estimating pairwise divergence between the two clusters
                    # the other cluster is selected randomly
                    sel_cluster_p = new_cluster_label
                    while sel_cluster_p == new_cluster_label:
                        sel_cluster_p = cluster_sampler.choice(unique_c, size=1, replace=False)
                    assert new_cluster_label > sel_cluster_p
                    sel_clusters_pq = [sel_cluster_p, new_cluster_label]
                    sel_cluster_p = None
                else:
                    # randomly select any two cluster for estimating pairwise divergence between the two clusters
                    sel_clusters_pq = cluster_sampler.choice(unique_c, size=2, replace=False)
                    sel_clusters_pq.sort()

                assert sel_clusters_pq[1] > sel_clusters_pq[0]

                # data indices corresponding to the two clusters
                p_idx = np.where(c == sel_clusters_pq[0])[0]
                q_idx = np.where(c == sel_clusters_pq[1])[0]
                sel_clusters_pq = None
                pq_idx = np.concatenate((p_idx, q_idx))
                # labels for two clusters for the purpose of estimating divergence
                z = np.concatenate((np.zeros(p_idx.size, dtype=np.bool), np.ones(q_idx.size, dtype=np.bool)))
                p_idx, q_idx = (None,)*2
                # select tensor for the data from the two clusters
                Xt = X_org[pq_idx, :]
                pq_idx = None

                # estimate dual divergence function
                fpq = model_pq(Xt)
                fqp = model_qp(Xt)

                # estimate loss as the negative of divergence function in both directions
                loss_pq = - self.compute_donsker_varadhan_representation_of_kl_div(
                    ft=fpq,
                    c=z,
                    is_optimization_else_estimation=True,
                    is_max_approx=is_max_approx,
                    device=device,
                )

                loss_qp = - self.compute_donsker_varadhan_representation_of_kl_div(
                    ft=fqp,
                    c=(~z),
                    is_optimization_else_estimation=True,
                    is_max_approx=is_max_approx,
                    device=device,
                )

                # maximize divergence to optimize the neural estimator
                optimizer_pq.zero_grad()
                loss_pq.backward()
                optimizer_pq.step()

                optimizer_qp.zero_grad()
                loss_qp.backward()
                optimizer_qp.step()

                batch_loss[i] = loss_pq.item() + loss_qp.item()
                
                Xt, z, fpq, fqp, loss_pq, loss_qp = (None,)*6

                if (i >= num_pure_weight_updates) and (cluster_update_sampler.random() < cluster_update_prob):

                    # update cluster labels

                    if debug:
                        print(f'lr={lr}')

                    start_time = time.time()

                    if debug:
                        plt.close()
                        plt.plot(batch_loss[:i+1], 'kx')
                        plt.xlabel('Number of Iterations')
                        plt.ylabel('Loss')
                        plt.show()

                    # data points from the cluster which is chosen for split to create new cluster from within it
                    Xt = X_org[sel_cluster_idx, :]

                    model_pq.eval()
                    model_qp.eval()
                    # estimate divergence function
                    fpq = model_pq(Xt).detach().cpu().numpy().flatten()
                    fqp = model_qp(Xt).detach().cpu().numpy().flatten()
                    Xt = None
                    model_pq.train()
                    model_qp.train()

                    assert not (np.all(np.isnan(fpq)) or np.all(np.isnan(fqp)))

                    # binary clustering optimization
                    z = self.optimize_cluster_labels_via_cut_point_in_the_dual_space(
                        f_pq=fpq,
                        f_qp=fqp,
                        is_max_approx=is_max_approx,
                        debug=debug,
                    )

                    assert z.dtype == np.bool
                    # update the cluster labels into global configuration of cluster labels
                    c[sel_cluster_idx[z]] = new_cluster_label
                    c[sel_cluster_idx[~z]] = cluster_for_split
                    z = None

                    if debug:
                        print(np.unique(c, return_counts=True))
                        print(f'{round(time.time()-start_time, 2)}s to update the labels')
            
            print('Cluster Sizes', np.unique(c, return_counts=True)[1])

        return c

    def estimate_kl_div_btw_clusters(self, X, c,
            num_iter=1000,
            is_max_approx=True,
            # it is typical to use 1e-4 value as learning rate for transformers. For other architectures, higher value (1e-3) can be used as it is a standard practice in the literature.
            lr=1e-4,
             # for initialization of neural nets when estimating KL-divergence, same value has been used in other estimators such as in ITENE (conditional MI based estimator of Transfer Entropy)
             # need not to tune it, same parameter value is used across all the datasets
            std=0.1,
            # even a single layer suffices when estimating divergence between clusters for evaluation purposes
            # in some scenarios, two layers are used.
            num_layers=1,
            # a light weight architecture with only a few units is preferred
            hidden_size=32,
            # in some scenarios, dropout value of up to 0.2 may be used, though not necessary
            dropout=0.0,
            # transformers are more robust when it comes to estimating divergence between clusters for evaluation purposes.
            nn='transformer',
            debug=False,
            stop_on_zero_self_kl=True,
            # choose cpu or GPU. multiple GPUs are not supported as such.
            device='cpu',
            #  standard value of 10 heads are used in transformers across all the datasets
            nhead=10,
    ):

        # this is only for the evaluation purpose, estimating divergence between any given cluster configuration of any method
        # note: what we care about is relative estimate of divergence between clusters rather than absolute values. Since clusters can have the underlying distributions with high divergence w.r.t. each other (as desired in this optimization of clusters as per our approach), the measure of divergence can goto infinity if we don't stop the optimization. There may not be any convegence of the estimator.
        # The criterion we use for stopping is either the maximum number of iterations or when divergence of a cluster w.r.t. itself approaches zero (originally being negative).

        assert X.ndim in [2, 3], X.ndim

        assert c.ndim == 1
        assert X.shape[0] == c.size

        unique_c, sizes_c = np.unique(c, return_counts=True)
        sort_idx = sizes_c.argsort()
        unique_c = unique_c[sort_idx]
        sizes_c = sizes_c[sort_idx]
        sort_idx = None

        # numpy to tensor on GPU
        X = torch.FloatTensor(X).to(device)

        # initializing divergence estimators for both directions
        # note: only two divergence estimators are used regardless of the number of clusters
        # furthermore, deep learning facilitates learning divergence estimators generically for different possible configuration of clusters, rather than a single one as one would expect in theory

        model_pq = get_model(
            input_shape=X.shape,
            parameters = {
                'hidden_size': hidden_size,
                'std': std,
                'num_layers': num_layers,
                'dropout': dropout,
                'nhead': nhead,
            },
            nn=nn,
        ).to(device)

        optimizer_pq = torch.optim.Adam(
            model_pq.parameters(),
            lr=lr,
        )

        model_qp = get_model(
            input_shape=X.shape,
            parameters = {
                'hidden_size': hidden_size,
                'std': std,
                'num_layers': num_layers,
                'dropout': dropout,
                'nhead': nhead,
            },
            nn=nn,
        ).to(device)
        
        optimizer_qp = torch.optim.Adam(
            model_qp.parameters(),
            lr=lr,
        )

        batch_loss_values = []

        cluster_sampler = npr.RandomState(seed=self.seed)

        # for recording KL-divergence between clusters at different stages (number of iterations) of learning the estimators
        kl_div_map = {
            'unique_c': unique_c,
            'sizes_c': sizes_c,
        }

        for i in range(num_iter):

            if (i % 10) == 0:
                print('.', end='')

            model_pq.train()
            model_qp.train()

            # choose two clusters randomly for estimate divergence between the two
            sel_clusters_pq = cluster_sampler.choice(unique_c, size=2, replace=False)
            sel_clusters_pq.sort()

            assert sel_clusters_pq[1] > sel_clusters_pq[0]

            # indices of data from the respective clusters
            p_idx = np.where(c == sel_clusters_pq[0])[0]
            q_idx = np.where(c == sel_clusters_pq[1])[0]
            pq_idx = np.concatenate((p_idx, q_idx))

            # binary labels for the two clusters for which divergence is estimated in the present iteration
            c_t = np.concatenate((np.zeros(p_idx.size, dtype=np.bool), np.ones(q_idx.size, dtype=np.bool)))

            # select data points from the two clusters
            X_t = X[pq_idx, :]
            X_t = X_t

            # estimate the dual functions
            fpq_t = model_pq(X_t)
            fqp_t = model_qp(X_t)

            # estimate loss as negtive of divergence between the two clusters
            loss_pq = -self.compute_donsker_varadhan_representation_of_kl_div(
                ft=fpq_t,
                c=c_t,
                is_optimization_else_estimation=True,
                is_max_approx=is_max_approx,
            )

            loss_qp = -self.compute_donsker_varadhan_representation_of_kl_div(
                ft=fqp_t,
                c=(~c_t),
                is_optimization_else_estimation=True,
                is_max_approx=is_max_approx,
            )

            # backpropagate neural weights for maximizing the estimate of divergence
            optimizer_pq.zero_grad()
            loss_pq.backward()
            optimizer_pq.step()

            optimizer_qp.zero_grad()
            loss_qp.backward()
            optimizer_qp.step()

            loss = loss_pq.item()+loss_qp.item()

            batch_loss_values.append(loss)

            X_t, c_t = (None,)*2

            model_pq.eval()
            model_qp.eval()

            # recording exact divergence between every pair of clusters
            # this could be avoided or be computed more efficiently
            fpq = model_pq(X).detach().cpu().numpy().flatten()
            fqp = model_qp(X).detach().cpu().numpy().flatten()

            kl_div = np.zeros((unique_c.size, unique_c.size), dtype=np.float)
            for j, p in enumerate(unique_c):
                p_idx = np.where(c == p)[0]

                for k, q in enumerate(unique_c):

                    q_idx = np.where(c == q)[0]

                    if p > q:
                        pq_idx = np.concatenate((q_idx, p_idx))
                        c_t = np.concatenate((np.zeros(q_idx.size, dtype=np.bool), np.ones(p_idx.size, dtype=np.bool)))
                    else:
                        pq_idx = np.concatenate((p_idx, q_idx))
                        c_t = np.concatenate((np.zeros(p_idx.size, dtype=np.bool), np.ones(q_idx.size, dtype=np.bool)))

                    kl_pq = self.compute_donsker_varadhan_representation_of_kl_div(
                        ft=fpq[pq_idx],
                        c=c_t,
                        is_optimization_else_estimation=False,
                        is_max_approx=is_max_approx,
                    )

                    kl_qp = self.compute_donsker_varadhan_representation_of_kl_div(
                        ft=fqp[pq_idx],
                        c=(~c_t),
                        is_optimization_else_estimation=False,
                        is_max_approx=is_max_approx,
                    )

                    kl_div[j, k] = kl_pq + kl_qp

            is_stop = False
            self_cluster_kl = ((kl_div[np.eye(kl_div.shape[0], dtype=np.bool)]*sizes_c).sum()/sizes_c.sum())

            # once divergence of a cluster w.r.t. itself reduces while maximizing the estimate of divergence across clusters, we can stop optimizing the estimator
            if stop_on_zero_self_kl and (self_cluster_kl <= 0.0) and ((kl_div[np.eye(kl_div.shape[0], dtype=np.bool)] <= 0.0).mean() > 0.5):
                is_stop = True
            self_cluster_kl = None

            kl_div_map[i] = kl_div.tolist()

            if is_stop:
                break

        if debug:
            plt.close()
            plt.plot(batch_loss_values, 'kx')
            plt.xlabel('Number of Iterations')
            plt.ylabel('Loss')
            plt.show()
        
        plt.close()
        ax = sns.heatmap(
            kl_div,
            vmin=max(0.0, kl_div.min()),
            annot=True,
            fmt=".0f",
            cmap="YlGnBu",
        )
        ax.set_xticklabels(sizes_c)
        ax.set_yticklabels(sizes_c)
        ax.set_xlabel('Clusters in the order of sizes')
        ax.set_ylabel('Clusters in the order of sizes')
        ax.set_title(f'KL-Divergence between clusters')
        plt.show()

        return kl_div_map, sizes_c
