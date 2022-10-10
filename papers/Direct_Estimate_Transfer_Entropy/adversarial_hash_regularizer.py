import math
import numpy as np
import numpy.random as npr
import scipy.special as scipy_special


class AdversarialHashRegularizer:

    def __init__(self,
                 C,
                 log_base_for_k=4.0,
                 max_val_for_k=8,
                 alpha=0.1,
                 seed=0,
                 num_hash_funcs=None,
                 debug=False,
    ):

        self.debug = debug

        if num_hash_funcs is None:
            num_hash_funcs = self.tune_num_hash_func(C=C)

        if num_hash_funcs < C.shape[1]:
            C = C[:, :num_hash_funcs]

        self.num_data = C.shape[0]
        self.digamma_of_num_data = scipy_special.digamma(self.num_data)
        self.k_for_knn_of_all_data = min(max(int(math.log(self.num_data, log_base_for_k)), 1), max_val_for_k)
        self.digamma_of_k_for_knn_of_all_data = scipy_special.digamma(self.k_for_knn_of_all_data)

        # alpha if Dirichlet for sampling new data within a hashcode
        self.alpha = alpha

        _, cluster_ids_for_adv_reg, self.cluster_sizes = np.unique(C, axis=0, return_inverse=True, return_counts=True)
        if self.debug:
            print(f'cluster sizes: {self.cluster_sizes.mean()}+-{self.cluster_sizes.std()} from {self.cluster_sizes.min()} to {np.median(self.cluster_sizes)} to {self.cluster_sizes.max()}')

        sel_clusters_logical_idx = (self.cluster_sizes > 1)
        self.unique_cluster_ids = np.where(sel_clusters_logical_idx)[0]
        self.cluster_sizes = self.cluster_sizes[sel_clusters_logical_idx]
        sel_clusters_logical_idx = None
        self.num_clusters = self.cluster_sizes.size
        self.k_for_knn_in_cluster = np.minimum(np.maximum((np.log(self.cluster_sizes)/np.log(log_base_for_k)).astype(np.int), 1), max_val_for_k)
        self.digamma_of_k_for_knn_in_cluster = scipy_special.digamma(self.k_for_knn_in_cluster)
        self.digamma_of_cluster_sizes = scipy_special.digamma(self.cluster_sizes)

        self.multinomial_sampler = npr.RandomState(seed=seed)

        self.data_idx_in_cluster = np.empty(self.num_clusters, dtype=np.object)
        for curr_idx, curr_cluster_id in enumerate(self.unique_cluster_ids):
            self.data_idx_in_cluster[curr_idx] = np.where(cluster_ids_for_adv_reg == curr_cluster_id)[0]
        self.unique_cluster_ids = None
        cluster_ids_for_adv_reg = None

    def tune_num_hash_func(self, C):
        num_hash_func = C.shape[1]
        for curr_num_hash_func in range(num_hash_func, 0, -1):
            _, self.cluster_sizes = np.unique(C[:, :curr_num_hash_func], axis=0, return_counts=True)
            if self.cluster_sizes.mean() > 1:
                return curr_num_hash_func
        raise AssertionError

    def compute_entropy_1d_with_histograms(self, x, num_bins=100, x_range=None):

        if x.std() == 0.0:
            return 0

        if x_range is not None:
            x_range = (min(x.min(), x_range[0]), max(x.max(), x_range[1]))

        # print(x, x_range, num_bins)

        # equal sized bins
        pde_hist, bins = np.histogram(x, bins=num_bins, density=True, range=x_range)
        assert pde_hist.size == num_bins

        pde_hist_sum = pde_hist.sum()
        assert pde_hist_sum > 0
        pde_hist /= pde_hist_sum

        # zero density bins do not contribute to entropy
        pde_hist = pde_hist[(pde_hist > 0)]
        entropy = -(pde_hist*np.log(pde_hist)).sum()

        return entropy

    def compute_entropy_1d_with_histograms_cond_hashcodes(self, x, num_bins=100, x_range=None):

        if x.std() == 0.0:
            return 0

        if x_range is not None:
            x_range = (min(x.min(), x_range[0]), max(x.max(), x_range[1]))
        else:
            x_range = (x.min(), x.max())

        H_cu = np.zeros(self.num_clusters, dtype=np.float)

        for curr_cluster_id in range(self.num_clusters):

            H_cu[curr_cluster_id] = self.compute_entropy_1d_with_histograms(
                x[self.data_idx_in_cluster[curr_cluster_id]],
                num_bins=num_bins,
                x_range=x_range,
            )

        entropy = (H_cu*self.cluster_sizes).sum()/float(self.cluster_sizes.sum())

        return entropy

    def generate_data_in_hash_bin(self, events, num_samples):

        alpha = self.alpha*np.ones(events.shape[0])
        # convex combination, ideally we should take events to be vertices of the convex hull
        weights = self.multinomial_sampler.dirichlet(alpha, size=num_samples)
        generated_data = weights.dot(events)

        return generated_data

    def generated_data_from_hashcodes(self, events, multiple=1):

        assert events.ndim == 2

        new_events = np.zeros(
            (self.cluster_sizes.sum()*multiple, events.shape[1]),
            dtype=events.dtype
        )

        curr_idx = 0
        for curr_cluster_id in range(self.num_clusters):
            assert self.cluster_sizes[curr_cluster_id] > 1
            n = self.cluster_sizes[curr_cluster_id]*multiple
            new_events[curr_idx:curr_idx+n, :] = self.generate_data_in_hash_bin(
                events=events[self.data_idx_in_cluster[curr_cluster_id], :],
                num_samples=n,
            )
            curr_idx += n

        assert curr_idx == new_events.shape[0], (curr_idx, new_events.shape)

        return new_events
