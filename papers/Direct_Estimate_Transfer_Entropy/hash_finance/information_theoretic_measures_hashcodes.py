import time
import math
import numpy as np
import numpy.random as npr
import scipy.spatial.distance as ssd
import scipy.special as scipy_special


class InformationTheoreticMeasuresHashcodes:

    def __init__(self, seed=0, max_noise=1e-20):
        self.noise_sampler = npr.RandomState(seed=seed)
        self.max_noise = max_noise

    def compute_marginal_entropy(self, hash_vector):

        # start_time = time.time()
        assert hash_vector.ndim == 1
        pos_ratio = hash_vector.mean()
        if (pos_ratio == 0.0) or (pos_ratio == 1.0):
            entropy = 0.0
        else:
            entropy = -(pos_ratio * np.log(pos_ratio)) - ((1.0 - pos_ratio) * np.log((1.0 - pos_ratio)))
        # print 'H(x): {}'.format(time.time() - start_time)

        return entropy

    def compute_marginal_entropy_non_binary(self, x):

        # start_time = time.time()
        assert len(x.shape) == 1

        entropy = 0.0
        unique_x = np.unique(x)
        for curr_val in unique_x:
            idx = np.where(x == curr_val)[0]
            fraction = float(idx.size)/x.size
            if fraction > 0.0:
                entropy += -(fraction * np.log(fraction))
        # print 'H(x): {}'.format(time.time() - start_time)

        return entropy

    def compute_conditional_entropy_x_cond_z(self, x, z):

        # start_time = time.time()
        assert x.shape == z.shape
        Pz1 = z.mean()
        Pz0 = 1.0 - Pz1

        if Pz0 == 0.0:
            Hx_cond_z1 = self.compute_marginal_entropy(x[z == 1])
            # print 'Hx_cond_z1', Hx_cond_z1
            Hx_cond_z = Pz1 * Hx_cond_z1
        elif Pz0 == 1.0:
            Hx_cond_z0 = self.compute_marginal_entropy(x[z == 0])
            # print 'Hx_cond_z0', Hx_cond_z0
            Hx_cond_z = Pz0 * Hx_cond_z0
        else:
            Hx_cond_z0 = self.compute_marginal_entropy(x[z == 0])
            # print 'Hx_cond_z0', Hx_cond_z0
            Hx_cond_z1 = self.compute_marginal_entropy(x[z == 1])
            # print 'Hx_cond_z1', Hx_cond_z1
            Hx_cond_z = Pz0*Hx_cond_z0 + Pz1*Hx_cond_z1
        # print 'H(x|z): {}'.format(time.time() - start_time)

        return Hx_cond_z

    def compute_conditional_entropy_x_cond_z_non_binary(self, x, z):

        # start_time = time.time()
        assert x.shape == z.shape
        assert len(x.shape) == 1
        num_data = x.size

        Hx_cond_z = 0.0
        for curr_z_val in np.unique(z):
            curr_idx = np.where(z == curr_z_val)[0]
            curr_prob = float(curr_idx.size)/num_data
            if curr_prob > 0.0:
                curr_Hx_cond_z = self.compute_marginal_entropy_non_binary(x[curr_idx])
                Hx_cond_z += curr_prob * curr_Hx_cond_z

        # print 'H(x|z): {}'.format(time.time() - start_time)

        return Hx_cond_z

    def compute_kl_divergence_of_density_from_hashcodes_as_bins_wrt_uniform_dist(self, C, z=None):

        start_time = time.time()

        if z is not None:
            assert C.shape[0] == z.size

            if C.ndim == 1:
                C = np.hstack((C[:, None], z[:, None]))
            elif C.ndim == 2:
                C = np.hstack((C, z[:, None]))
            else:
                raise AssertionError
            z = None

        density_from_clusters = np.unique(C, axis=0, return_counts=True)[1].astype(np.float)
        density_from_clusters /= density_from_clusters.sum()

        uniform_density = np.ones(density_from_clusters.size, dtype=np.float)/density_from_clusters.size

        kl_div = np.log(uniform_density/density_from_clusters).mean()

        print(round(time.time()-start_time, 2), end=',')

        return kl_div

    def compute_kNN_entropy_cond_hashcodes(self, X, C, log_base_for_k=4.0, max_val_for_k=8):

        assert X.shape[0] == C.shape[0]

        num_dim = X.shape[1]

        log_c_d = self.compute_log_of_unit_ball_volume(num_dim=num_dim)

        Cu, Cu_ids, Cu_sizes = np.unique(C, axis=0, return_inverse=True, return_counts=True)
        num_hash_bins = Cu.shape[0]

        sel_cu_ids = np.where(Cu_sizes > 1)[0]

        H_cu = np.zeros(num_hash_bins, dtype=np.float)
        centriods = np.zeros((num_hash_bins, X.shape[1]))

        for curr_cu_id in range(num_hash_bins):
            data_idx_from_curr_bin = np.where(Cu_ids == curr_cu_id)[0]
            X_from_curr_bin = X[data_idx_from_curr_bin, :]
            centriods[curr_cu_id, :] = X_from_curr_bin.mean(0)

            if curr_cu_id in sel_cu_ids:
                D = ssd.squareform(
                    ssd.pdist(
                        X=X_from_curr_bin,
                        metric='euclidean',
                    )
                )
                H_cu[curr_cu_id] = self.compute_entropy_knn(
                    distances=D,
                    k=min(max(int(math.log(data_idx_from_curr_bin.size, log_base_for_k)), 1), max_val_for_k),
                    is_constant=True,
                    d=num_dim,
                    log_c_d=log_c_d,
                    digamma_k=None,
                )

        entropy = (H_cu*Cu_sizes).sum()/float(Cu_sizes.sum())

        D_centriods = ssd.squareform(
            ssd.pdist(
                X=centriods,
                metric='euclidean',
            )
        )
        entropy_centriods = self.compute_entropy_knn(
            distances=D_centriods,
            k=min(max(int(math.log(num_hash_bins, log_base_for_k)), 1), max_val_for_k),
            is_constant=True,
            d=num_dim,
            log_c_d=log_c_d,
            digamma_k=None,
        )

        return entropy, entropy_centriods

    def compute_kNN_entropy_all_data(self, X, log_base_for_k=4.0, max_val_for_k=8):

        num_data, num_dim = X.shape

        log_c_d = self.compute_log_of_unit_ball_volume(num_dim=num_dim)

        D = ssd.squareform(
            ssd.pdist(
                X=X,
                metric='euclidean',
            )
        )

        k = min(max(int(math.log(num_data, log_base_for_k)), 1), max_val_for_k)

        print(num_data, k)

        entropy = self.compute_entropy_knn(
            distances=D,
            k=k,
            is_constant=True,
            d=num_dim,
            log_c_d=log_c_d,
            digamma_k=None,
        )

        return entropy

    def compute_entropy_of_density_per_hashcodes_as_bins(self, C, z=None):

        if z is not None:
            assert C.shape[0] == z.size

            if C.ndim == 1:
                C = np.hstack((C[:, None], z[:, None]))
            elif C.ndim == 2:
                C = np.hstack((C, z[:, None]))
            else:
                raise AssertionError
            z = None

        density_from_clusters = np.unique(C, axis=0, return_counts=True)[1].astype(np.float)
        density_from_clusters /= density_from_clusters.sum()

        entropy = -np.log(density_from_clusters).mean()

        return entropy

    def compute_entropy_of_hashcodes_per_chain_rule(self, C):

        marginal_cond_entropies = self.compute_marginal_and_conditional_entropies_of_hashcodes_per_chain_rule(C=C)
        entropy = marginal_cond_entropies.sum()

        return entropy

    def compute_marginal_and_conditional_entropies_of_hashcodes_per_chain_rule(self, C):

        density_from_clusters = np.unique(C, axis=0, return_counts=True)[1].astype(np.float)
        density_from_clusters /= density_from_clusters.sum()
        num_hash_funcs = C.shape[1]
        marginal_cond_entropies = np.zeros(num_hash_funcs)
        for curr_hash_func in range(num_hash_funcs):
            if curr_hash_func == 0:
                marginal_cond_entropies[curr_hash_func] = self.compute_marginal_entropy(
                    hash_vector=C[:, curr_hash_func]
                )
            else:
                _, cluster_ids = np.unique(C[:, :curr_hash_func], axis=0, return_inverse=True)
                marginal_cond_entropies[curr_hash_func] = self.compute_conditional_entropy_z_cond_C(
                    cluster_ids=cluster_ids,
                    # C=C[:, :curr_hash_func],
                    z=C[:, curr_hash_func],
                )

        return marginal_cond_entropies

    def compute_KL_div_between_partitions_cond_clusters(self, cluster_ids, z, distances, max_num_samples_in_cluster_for_kl_div_compute, k=1):

        # start_time = time.time()

        unique_clusters, cluster_sizes = np.unique(cluster_ids, return_counts=True)

        if cluster_sizes.min() > max_num_samples_in_cluster_for_kl_div_compute:
            return 0
        elif cluster_sizes.max() <= (2*k+3):
            return 0

        sel_clusters_bool = ((cluster_sizes > (2*k+3)) & (cluster_sizes <= max_num_samples_in_cluster_for_kl_div_compute))
        cluster_ids_sel = unique_clusters[sel_clusters_bool]
        if cluster_ids_sel.size == 0:
            return 0

        sel_cluster_sizes = cluster_sizes[sel_clusters_bool]

        kl_clusters = np.zeros(cluster_ids_sel.size)
        # is_computed = np.zeros(cluster_ids_sel.size, dtype=np.bool)
        for curr_cluster_idx, curr_cluster_id in enumerate(cluster_ids_sel):
            data_idx_in_cluster = np.where(cluster_ids == curr_cluster_id)[0]
            split_ratio = z[data_idx_in_cluster].mean()
            if 0.25 < split_ratio < 0.75:
                # is_computed[curr_cluster_idx] = True
                kl_clusters[curr_cluster_idx] = self.compute_KL_div_between_partitions_within_cluster(
                    z=z[data_idx_in_cluster],
                    distances=distances[data_idx_in_cluster, :][:, data_idx_in_cluster].toarray(),
                    k=k,
                )
                # print(
                #     round(split_ratio, 2),
                #     round(kl_clusters[curr_cluster_idx], 2)
                # )

        kl = float((kl_clusters*sel_cluster_sizes).sum())/sel_cluster_sizes.sum()

        # if is_computed.sum() > 0:
        #     kl = float((kl_clusters[is_computed]*sel_cluster_sizes[is_computed]).sum())/sel_cluster_sizes[is_computed].sum()
        # else:
        #     kl = 0.0

        # print(f'{round(time.time()-start_time, 1)}', end=',')

        return kl

    def compute_KL_div_between_partitions_within_cluster(self, z, distances, k=1):

        # # assuming that all the data points are unique
        # assert np.count_nonzero(distances) == np.prod(distances.shape)

        assert distances.shape[0] == distances.shape[1]
        assert z.size == distances.shape[1]

        if z.size <= (2*k+3):
            return 0

        z0_idx = np.where(z == 0)[0]
        z1_idx = np.where(z == 1)[0]

        if z0_idx.size <= k:
            return 0
        elif z1_idx.size <= k:
            return 0

        distances_z0_z0 = distances[z0_idx, :][:, z0_idx]
        dist_of_knn_of_z0_in_z0 = np.clip(
            a=np.partition(
                a=distances_z0_z0 + self.max_noise*self.noise_sampler.uniform(size=distances_z0_z0.shape),
                kth=k,
                axis=1,
            )[:, k],
            a_min=1.0e-100,
            a_max=None,
        )
        distances_z0_z0 = None

        distances_z0_z1 = distances[z0_idx, :][:, z1_idx]
        dist_of_knn_of_z0_in_z1 = np.clip(
            a=np.partition(
                a=distances_z0_z1 + self.max_noise*self.noise_sampler.uniform(size=distances_z0_z1.shape),
                kth=k-1,
                axis=1,
            )[:, k-1],
            a_min=1.0e-100,
            a_max=None,
        )
        distances_z0_z1 = None

        kl_z0_z1 = np.log(dist_of_knn_of_z0_in_z1/dist_of_knn_of_z0_in_z0).mean()
        dist_of_knn_of_z0_in_z1, dist_of_knn_of_z0_in_z0 = (None,)*2

        distances_z1_z1 = distances[z1_idx, :][:, z1_idx]
        dist_of_knn_of_z1_in_z1 = np.clip(
            a=np.partition(
                a=distances_z1_z1 + self.max_noise*self.noise_sampler.uniform(size=distances_z1_z1.shape),
                kth=k,
                axis=1,
            )[:, k],
            a_min=1.0e-100,
            a_max=None,
        )
        distances_z1_z1 = None

        distances_z1_z0 = distances[z1_idx, :][:, z0_idx]
        dist_of_knn_of_z1_in_z0 = np.clip(
            a=np.partition(
                a=distances_z1_z0 + self.max_noise*self.noise_sampler.uniform(size=distances_z1_z0.shape),
                kth=k-1,
                axis=1,
            )[:, k-1],
            a_min=1.0e-100,
            a_max=None,
        )
        distances_z1_z0 = None
        kl_z1_z0 = np.log(dist_of_knn_of_z1_in_z0/dist_of_knn_of_z1_in_z1).mean()
        dist_of_knn_of_z1_in_z0, dist_of_knn_of_z1_in_z1 = (None,)*2

        return kl_z0_z1+kl_z1_z0

    def compute_log_of_unit_ball_volume(self, num_dim):
        log_c_d = (num_dim/2)*math.log(math.pi) - scipy_special.loggamma(1+(num_dim/2)) + num_dim*math.log(2)
        return log_c_d

    def compute_entropy_knn(self, distances, d=None, is_constant=False, log_c_d=None, k=1, digamma_k=None):

        # print(k, end=',')

        assert distances.shape[0] == distances.shape[1]
        # print(f'distances.shape={distances.shape}')

        num_data = distances.shape[0]

        if num_data < (k+1):
            return 0

        dist_of_knn = np.clip(
            a=2.0*np.partition(
                a=distances + self.max_noise*self.noise_sampler.uniform(size=distances.shape),
                kth=k,
                axis=1,
            )[:, k],
            a_min=1.0e-100,
            a_max=None,
        )

        entropy = np.log(dist_of_knn).mean()

        if is_constant:
            assert d is not None

            if digamma_k is None:
                digamma_k = scipy_special.digamma(k)

            if log_c_d is None:
                log_c_d = self.compute_log_of_unit_ball_volume(num_dim=d)

            entropy = (d*entropy) + (-digamma_k + scipy_special.digamma(num_data) + log_c_d)

        return entropy

    def compute_conditional_entropy_z_cond_C(self, cluster_ids, z):

        unique_clusters, cluster_sizes = np.unique(cluster_ids, return_counts=True)
        Hz_per_C = np.zeros(unique_clusters.size)
        for curr_cluster_idx, curr_cluster_id in enumerate(unique_clusters):
            data_idx_in_cluster = np.where(cluster_ids == curr_cluster_id)[0]
            Hz_per_C[curr_cluster_idx] = self.compute_marginal_entropy(
                    hash_vector=z[data_idx_in_cluster],
            )
        Hz_c = float((Hz_per_C*cluster_sizes).sum())/cluster_sizes.sum()
        Hz_per_C = None

        return Hz_c

    def count_elements_in_clusters(self, C):
        Cu, Cu_ids, n_per_C = np.unique(C, axis=0, return_counts=True)
        return Cu, n_per_C

