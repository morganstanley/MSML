import numpy as np
import numpy.random as npr
import scipy.sparse as scipy_sparse


class HashDataAugment:

    def __init__(self,
                 C,
                 alpha=0.1,
                 seed=0,
                 num_hash_funcs=None,
                 debug=False,
    ):
        self.debug = debug

        if num_hash_funcs < C.shape[1]:
            C = C[:, :num_hash_funcs]

        self.num_data = C.shape[0]

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

        self.multinomial_sampler = npr.RandomState(seed=seed)

        self.data_idx_in_cluster = np.empty(self.num_clusters, dtype=np.object)
        for curr_idx, curr_cluster_id in enumerate(self.unique_cluster_ids):
            self.data_idx_in_cluster[curr_idx] = np.where(cluster_ids_for_adv_reg == curr_cluster_id)[0]
        self.unique_cluster_ids = None
        cluster_ids_for_adv_reg = None

    def generate_data_in_hash_bin(self, events, labels, num_samples):

        alpha = self.alpha*np.ones(events.shape[0])
        # convex combination, ideally we should take events to be vertices of the convex hull
        weights = self.multinomial_sampler.dirichlet(alpha, size=num_samples)

        generated_data = weights.dot(events)
        generated_labels = weights.dot(labels)
        assert generated_labels.ndim == 1
        assert generated_labels.dtype == np.float

        if labels.dtype == np.bool:
            generated_labels_binary = np.zeros(generated_labels.size, dtype=np.bool)
            generated_labels_binary[generated_labels > 0.5] = True

            return generated_data, generated_labels_binary
        else:
            return generated_data, generated_labels

    def generate_data_from_hashcodes(self, events, labels, multiple=1.0):

        # todo: may be operate in sparse arrays rather converting to dense arrays
        if scipy_sparse.issparse(events):
            assert isinstance(events, scipy_sparse.csr_matrix)
            is_sparse = True
            events = events.toarray()
        else:
            is_sparse = False

        print('.', end='')

        assert events.ndim == 2
        assert labels.ndim == 1

        new_events = np.zeros(
            (
                int(self.cluster_sizes.sum()*multiple), events.shape[1]
            ),
            dtype=events.dtype,
        )

        new_labels = np.zeros(
            int(self.cluster_sizes.sum()*multiple), dtype=labels.dtype,
        )

        curr_idx = 0
        for curr_cluster_id in range(self.num_clusters):

            assert self.cluster_sizes[curr_cluster_id] > 1

            n = int(self.cluster_sizes[curr_cluster_id]*multiple)

            new_events[curr_idx:curr_idx+n, :], new_labels[curr_idx:curr_idx+n] = self.generate_data_in_hash_bin(
                events=events[self.data_idx_in_cluster[curr_cluster_id], :],
                labels=labels[self.data_idx_in_cluster[curr_cluster_id]],
                num_samples=n,
            )
            curr_idx += n

        assert curr_idx == new_events.shape[0], (curr_idx, new_events.shape)

        if is_sparse:
            new_events = scipy_sparse.csr_matrix(new_events)

        assert new_events.shape[1] == events.shape[1]
        assert labels.dtype == new_labels.dtype

        return new_events, new_labels
