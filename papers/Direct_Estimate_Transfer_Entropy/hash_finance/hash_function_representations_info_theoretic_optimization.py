import time
import pickle
import os
import numpy as np
import numpy.random as npr
import importlib
import itertools

import scipy.sparse as scipy_sparse
import scipy.special as scipy_special
import scipy.spatial.distance as ssd

from . import information_theoretic_measures_hashcodes as itmh
importlib.reload(itmh)

from . import random_lr_hash as rlr
importlib.reload(rlr)

from . import divide_conquer_optimize_split
importlib.reload(divide_conquer_optimize_split)

from . import parallel_computing as pc
importlib.reload(pc)

from . import parallel_computing_wrapper as pcw
importlib.reload(pcw)


class EventDistances:

    def __init__(self, events, max_num_samples_in_cluster_for_intra_distance_compute):
        # fill pairwise distances only for selected data points
        self.events = events
        self.num_data = events.shape[0]
        self.distances = scipy_sparse.lil_matrix((self.num_data, self.num_data), dtype=np.float)
        self.bool_idx_of_events_with_distances_not_computed = np.ones(self.num_data, dtype=np.bool)
        self.max_num_samples_in_cluster_for_intra_distance_compute = max_num_samples_in_cluster_for_intra_distance_compute

    def add_intra_cluster_distances(self, cluster_ids, sel_clusters_ids=None):

        start_time = time.time()

        assert cluster_ids.size == self.num_data

        if sel_clusters_ids is None:
            unique_cluster_ids, cluster_sizes = np.unique(cluster_ids, return_counts=True)
            sel_clusters_ids = unique_cluster_ids[cluster_sizes <= self.max_num_samples_in_cluster_for_intra_distance_compute]

        assert sel_clusters_ids is not None

        if len(sel_clusters_ids) == 0:
            return
        else:
            for curr_cluster_id in sel_clusters_ids:
                # computing nearest neighbors for only small clusters
                # todo: try other metrics later or tune it, should upon the nature of data, sparsity in data etc.
                data_idx_in_cluster = np.where(cluster_ids == curr_cluster_id)[0]
                # assuming that cluster ids are computed in a certain fashion which come from hashcodes,
                # such that future cluster ids are only emerging from split of clusters from previous iterations
                num_data_idx_in_cluster_with_distances_not_precomputed = self.bool_idx_of_events_with_distances_not_computed[data_idx_in_cluster].sum()
                if num_data_idx_in_cluster_with_distances_not_precomputed == 0:
                    continue
                else:
                    assert num_data_idx_in_cluster_with_distances_not_precomputed == data_idx_in_cluster.size
                    self.distances[data_idx_in_cluster[:,None], data_idx_in_cluster] = ssd.squareform(
                        ssd.pdist(
                            X=self.events[data_idx_in_cluster, :],
                            metric='euclidean',
                        )
                    )
                    self.bool_idx_of_events_with_distances_not_computed[data_idx_in_cluster] = False

        print(f'Added intra cluster distances in {round(time.time()-start_time, 2)}s, number of distances added so far, {self.distances.getnnz()}.')


class HashFunctionRepresentations:

    # Implemented for R1NN currently as that is the most efficient hashing
    # when optimizing the split of a subset (artificial labels) to construct a hash function.

    def __init__(
         self,
            hash_func='RLR',
            num_cores=1,
            seed_val=0,
            is_optimize_split=True,
            min_set_size_to_split=8,
            max_set_size_for_info_cluster_optimal=32,
            is_optimize_hash_functions_nonredundancy=True,
            nonredundancy_score_beta=1.0,
            is_data_entropy_inside_cluster=True,
            max_num_samples_in_cluster_for_entropy_compute=1000,
            max_num_samples_in_cluster_for_kl_div_compute=100,
            max_num_samples_in_sel_cluster_for_kl_div_compute=300,
            k_for_entropy_estimate=2,
            # seems redundant term w.r.t. Hz_C
            is_data_entropy_from_hashcodes_as_bins=False,
            is_equal_split=True,
            tune_hp_hash_func=False,
            dir_path=None,
            model_name=None,
            debug=False,
            C=1.0,
            penalty='l1',
    ):

        self.debug = debug

        self.hash_func = hash_func

        self.num_cores = num_cores

        self.itmh_obj = itmh.InformationTheoreticMeasuresHashcodes()

        self.npr_sample_alpha_obj = npr.RandomState(seed=seed_val)
        self.npr_sample_nbhec_obj = npr.RandomState(seed=seed_val)
        self.npr_sample_subset_obj = npr.RandomState(seed=seed_val)
        self.npr_sample_hash_bits_obj = npr.RandomState(seed=seed_val)
        self.npr_sample_hash_bits_infer_cluster_obj = npr.RandomState(seed=seed_val)
        self.npr_sample_prob_deletion_old_obj = npr.RandomState(seed=seed_val)
        self.npr_sample_combinations_obj = npr.RandomState(seed=seed_val)
        self.npr_sample_data_obj = npr.RandomState(seed=seed_val)
        self.npr_sample_unlabeled_data_obj = npr.RandomState(seed=seed_val)
        self.npr_sample_cluster_obj = npr.RandomState(seed=seed_val)
        self.npr_sample_split_obj = npr.RandomState(seed=seed_val)
        self.npr_sample_sel_compute_obj = npr.RandomState(seed=seed_val)
        self.npr_bits_subset_size_reduction = npr.RandomState(seed=seed_val)

        self.is_rnd_sample_data_for_opt_kernel_par = True

        self.is_optimize_split = is_optimize_split

        self.min_set_size_to_split = min_set_size_to_split
        self.max_set_size_for_info_cluster_optimal = max_set_size_for_info_cluster_optimal

        self.is_optimize_hash_functions_nonredundancy = is_optimize_hash_functions_nonredundancy
        self.nonredundancy_score_beta = nonredundancy_score_beta

        self.is_equal_split = is_equal_split

        self.is_data_entropy_inside_cluster = is_data_entropy_inside_cluster

        self.max_num_samples_in_cluster_for_entropy_compute = max_num_samples_in_cluster_for_entropy_compute
        self.max_num_samples_in_cluster_for_kl_div_compute = max_num_samples_in_cluster_for_kl_div_compute
        self.max_num_samples_in_sel_cluster_for_kl_div_compute = max_num_samples_in_sel_cluster_for_kl_div_compute

        self.k_for_entropy_estimate = k_for_entropy_estimate

        self.is_data_entropy_from_hashcodes_as_bins = is_data_entropy_from_hashcodes_as_bins

        assert dir_path is not None
        assert model_name is not None
        self.dir_path = dir_path + '/' + model_name

        self.tune_hp_hash_func = tune_hp_hash_func

        self.C = C
        self.penalty = penalty

        if self.hash_func == 'RLR':
            self.hash_func_obj = rlr.RandomLRHash(
                tune_hp=self.tune_hp_hash_func,
                C=self.C,
                penalty=self.penalty,
            )
        else:
            raise NotImplemented

    def compute_score_for_reference_subset_partition(
            self,
            C,
            z,
            sample_counts=None,
            distances=None,
            cluster_ids=None,
            data_idx_from_sel_cluster=None,
    ):

        if sample_counts is not None:

            # implementation is complicated for this case
            assert not self.is_data_entropy_inside_cluster
            assert distances is None

            assert sample_counts.dtype == np.int

            total_sample_counts = sample_counts.sum()
            if self.debug:
                print('total_sample_counts', total_sample_counts)

            if C is not None:
                if self.debug:
                    print('................')
                    print('C.shape', C.shape)

                if len(C.shape) == 1:
                    org_num_bits = 1
                else:
                    assert len(C.shape) == 2
                    org_num_bits = C.shape[1]
                C = np.repeat(C, sample_counts, axis=0)

                if self.debug:
                    print('C.shape', C.shape)

                assert C.shape[0] == total_sample_counts
                if len(C.shape) == 1:
                    assert org_num_bits == 1
                else:
                    assert C.shape[1] == org_num_bits

            assert z is not None
            # print 'z.shape', z.shape
            z = np.repeat(z, sample_counts, axis=0)
            # print 'z.shape', z.shape
            assert z.ndim == 1
            assert z.size == total_sample_counts

            if cluster_ids is not None:
                cluster_ids = np.repeat(cluster_ids, sample_counts, axis=0)
                assert cluster_ids.ndim == 1
                assert cluster_ids.size == total_sample_counts

            total_sample_counts, sample_counts = None, None

        Hz = self.itmh_obj.compute_marginal_entropy(z)

        # computing H(z|C)
        if (C is not None) and (self.is_optimize_hash_functions_nonredundancy):
            assert cluster_ids is not None
            Hz_cond_C = self.itmh_obj.compute_conditional_entropy_z_cond_C(
                cluster_ids=cluster_ids,
                z=z,
            )

            if self.debug:
                print(f'Hz_cond_C: {round(Hz_cond_C, 2)}')
        else:
            Hz_cond_C = 0.0

        if self.is_data_entropy_inside_cluster and (C is not None) and (C.ndim == 2):
            assert sample_counts is None
            assert distances is not None
            assert cluster_ids is not None

            KL_div_in_all_clusters = self.itmh_obj.compute_KL_div_between_partitions_cond_clusters(
                cluster_ids=cluster_ids, z=z,
                distances=distances,
                k=self.k_for_entropy_estimate,
                max_num_samples_in_cluster_for_kl_div_compute=self.max_num_samples_in_cluster_for_kl_div_compute,
            )

            if (data_idx_from_sel_cluster is not None) and (data_idx_from_sel_cluster.size <= self.max_num_samples_in_sel_cluster_for_kl_div_compute):
                # print(f'data_idx_from_sel_cluster={data_idx_from_sel_cluster.size}')
                kl_div_in_cluster = self.itmh_obj.compute_KL_div_between_partitions_within_cluster(
                    z=z[data_idx_from_sel_cluster],
                    distances=distances[data_idx_from_sel_cluster, :][:, data_idx_from_sel_cluster],
                    k=self.k_for_entropy_estimate,
                )
            else:
                kl_div_in_cluster = 0.0
        else:
            KL_div_in_all_clusters = 0.0
            kl_div_in_cluster = 0.0

        if self.is_data_entropy_from_hashcodes_as_bins and (C is not None):
            kl_div_hash_dist_wrt_unif = self.itmh_obj.compute_kl_divergence_of_density_from_hashcodes_as_bins_wrt_uniform_dist(
                C=C, z=z,
            )
        else:
            kl_div_hash_dist_wrt_unif = 0.0

        curr_score = Hz + self.nonredundancy_score_beta*Hz_cond_C + kl_div_in_cluster + KL_div_in_all_clusters - kl_div_hash_dist_wrt_unif

        # if self.debug:
        score_str = f'Hz: {round(Hz, 2)},'
        if self.is_optimize_hash_functions_nonredundancy:
            score_str += f' Hz_C: {round(Hz_cond_C, 2)},'
        if self.is_data_entropy_from_hashcodes_as_bins:
            score_str += f' KL_div_hash_prob_unif: {round(kl_div_hash_dist_wrt_unif, 2)},'
        if self.is_data_entropy_inside_cluster:
            score_str += f' KL_sel_cluster: {round(kl_div_in_cluster, 2)},'
            score_str += f' KL_C: {round(KL_div_in_all_clusters, 2)},'
        score_str += f' score: {round(curr_score, 2)}'
        print(score_str)

        # total_time_score_compute = time.time() - start_time_score_compute

        return curr_score

    def compare_array_wrt_array_of_arrays(self, array_for_comparison, arrays):
        assert array_for_comparison.dtype == np.int
        assert arrays.dtype == np.object
        num_arrays_for_comparison = arrays.size
        for curr_idx in range(num_arrays_for_comparison):
            if np.array_equal(arrays[curr_idx], array_for_comparison):
                return True
        return False

    def get_all_combinations(self, subset_size, superset=None):

        start_time = time.time()

        if superset is None:
            superset = np.arange(subset_size, dtype=np.int)

        alpha = int(subset_size / 2)

        if self.is_equal_split:
            combination_sizes = [alpha]
        else:
            combination_sizes = np.arange(1, alpha+1, dtype=np.int)

        num_combinations = 0
        for curr_combination_size in combination_sizes:
            num_combinations += int(scipy_special.comb(subset_size, curr_combination_size))

        subsets1 = np.empty(num_combinations, dtype=np.object)
        subsets2 = np.empty(num_combinations, dtype=np.object)

        idx = -1
        none_idx = []
        for curr_combination_size in combination_sizes:
            for curr_subset in itertools.combinations(superset, curr_combination_size):
                idx += 1

                curr_subset = np.array(list(curr_subset))
                curr_subset_complement = np.setdiff1d(superset, curr_subset)

                if not self.compare_array_wrt_array_of_arrays(curr_subset, subsets2):
                    subsets1[idx] = curr_subset
                    subsets2[idx] = curr_subset_complement
                else:
                    none_idx.append(idx)
                    subsets1[idx] = None
                    subsets2[idx] = None

        if none_idx:
            none_idx = np.array(none_idx)
            all_idx = np.arange(num_combinations, dtype=np.int)
            not_none_idx = np.setdiff1d(all_idx, none_idx)
            del none_idx, all_idx
            subsets1 = subsets1[not_none_idx]
            subsets2 = subsets2[not_none_idx]
            del not_none_idx

        if self.debug:
            print('Time to compute combinations', time.time() - start_time)

        return subsets1, subsets2

    def compute_scores_for_combinations_wrapper(
            self,
            z_combinations,
            C,
            sample_counts=None,
            distances=None,
            cluster_ids=None,
            data_idx_from_sel_cluster=None,
    ):
        assert z_combinations.ndim == 1
        num_combinations = z_combinations.size
        scores = np.zeros(num_combinations)
        for curr_combination_idx in range(num_combinations):
            scores[curr_combination_idx] = self.compute_score_for_reference_subset_partition(
                z=z_combinations[curr_combination_idx],
                C=C,
                sample_counts=sample_counts,
                distances=distances,
                cluster_ids=cluster_ids,
                data_idx_from_sel_cluster=data_idx_from_sel_cluster
            )
        return scores

    def select_high_entropy_cluster(self,
            C, subset_size,
            sample_weights=None,
            is_return_cluster_data_indices=False,
            distances_obj=None,
    ):

        _, cluster_ids, n_per_C = np.unique(C, axis=0, return_inverse=True, return_counts=True)
        # unique_ids = np.arange(n_per_C.size, dtype=np.int)
        max_n_per_C = n_per_C.max()

        if (distances_obj is not None) and (max_n_per_C < self.max_num_samples_in_cluster_for_entropy_compute):
            start_time_high_entropy_cluster = time.time()
            num_cluster_choices = min(10, max(int(0.1*n_per_C.size), 3))
            sel_clusters_ids = np.argpartition(n_per_C, kth=-num_cluster_choices)[-num_cluster_choices:]
            distances_obj.add_intra_cluster_distances(
                cluster_ids=cluster_ids,
                sel_clusters_ids=sel_clusters_ids,
            )
            data_entropy_in_clusters = np.zeros(sel_clusters_ids.size, dtype=np.float)
            for curr_cluster_idx, curr_cluster_id in enumerate(sel_clusters_ids):
                data_idx_in_curr_cluster = np.where(cluster_ids == curr_cluster_id)[0]
                assert data_idx_in_curr_cluster.size > 0
                data_entropy_in_clusters[curr_cluster_idx] = self.itmh_obj.compute_entropy_knn(
                    distances=distances_obj.distances[data_idx_in_curr_cluster, :][:, data_idx_in_curr_cluster].toarray(),
                    k=self.k_for_entropy_estimate,
                )
                data_idx_in_curr_cluster = None
            print(f'data_entropy_in_clusters={data_entropy_in_clusters}, cluster_sizes={n_per_C[sel_clusters_ids]}')
            max_entropy_cluster_id = sel_clusters_ids[data_entropy_in_clusters.argmax()]
            print(f'Selected high entropy cluster in {round(time.time()-start_time_high_entropy_cluster, 2)}s.')
        else:
            max_entropy_cluster_id = np.where(n_per_C == max_n_per_C)[0][0]

        print(f'n_per_C size:{n_per_C.size} max:{n_per_C.max()} min:{n_per_C.min()} mean:{round(n_per_C.mean(), 0)} std:{round(n_per_C.std(), 0)}')

        data_idx_from_cluster = np.where(cluster_ids == max_entropy_cluster_id)[0]

        if data_idx_from_cluster.size <= self.max_num_samples_in_sel_cluster_for_kl_div_compute:
            distances_obj.add_intra_cluster_distances(
                cluster_ids=cluster_ids,
                sel_clusters_ids=[max_entropy_cluster_id],
            )

        if data_idx_from_cluster.size <= subset_size:
            subset_idx = np.copy(data_idx_from_cluster)
            subset_size = data_idx_from_cluster.size
        else:
            assert subset_size >= 2
            if sample_weights is not None:
                sample_weights__data_idx_from_cluster = sample_weights[data_idx_from_cluster]
                if self.debug:
                    print('sample_weights__data_idx_from_cluster', sample_weights__data_idx_from_cluster)

                sample_weights__data_idx_from_cluster =\
                    self.get_sample_weights_normalized_from_counts(sample_weights__data_idx_from_cluster, None)
                if self.debug:
                    print(sample_weights__data_idx_from_cluster)
            else:
                sample_weights__data_idx_from_cluster = None

            assert (sample_weights__data_idx_from_cluster is None) \
                   or np.all(sample_weights__data_idx_from_cluster > 0.0)

            subset_idx = self.npr_sample_subset_obj.choice(
                data_idx_from_cluster,
                subset_size,
                replace=False,
                p=sample_weights__data_idx_from_cluster,
            )

        if is_return_cluster_data_indices:
            return subset_idx, subset_size, data_idx_from_cluster
        else:
            return subset_idx, subset_size

    def compute_scores_for_combinations(
            self,
            z_combinations,
            C,
            sample_counts=None,
            distances=None,
            cluster_ids=None,
            data_idx_from_sel_cluster=None,
    ):
        assert z_combinations.ndim == 1
        num_combinations = z_combinations.size
        num_cores = min(num_combinations, self.num_cores)

        if num_cores == 1:
            scores = self.compute_scores_for_combinations_wrapper(
                z_combinations=z_combinations,
                C=C,
                sample_counts=sample_counts,
                distances=distances,
                cluster_ids=cluster_ids,
                data_idx_from_sel_cluster=data_idx_from_sel_cluster,
            )
        else:
            scores = np.zeros(num_combinations)

            print('num_combinations', num_cores)
            idx_range_parallel = pc.uniform_distribute_tasks_across_cores(num_combinations, num_cores)

            args_tuples_map = {}
            for currCore in range(num_cores):
                args_tuples_map[currCore] = (
                    z_combinations[idx_range_parallel[currCore]],
                    C,
                    sample_counts,
                    distances,
                    cluster_ids,
                    data_idx_from_sel_cluster,
                )

            pcw_obj = pcw.ParallelComputingWrapper(num_cores=num_cores)
            results_map = pcw_obj.process_method_parallel(
                method=self.compute_scores_for_combinations_wrapper,
                args_tuples_map=args_tuples_map,
            )

            for curr_core in range(num_cores):
                curr_result = results_map[curr_core]
                scores[idx_range_parallel[curr_core]] = curr_result

        mean_score = scores.mean()
        min_score = scores.min()
        max_score = scores.max()

        print(f'scores {scores.size}'
              f' min:{round(scores.min(), 2)}'
              f' max:{round(scores.max(), 2)}'
              f' mean:{round(scores.mean(), 2)}'
              f' std:{round(scores.std(), 2)}'
        )

        if self.debug:
            print('scores', scores)
            print('scores.min()', min_score)
            print('scores.max()', max_score)
            print('scores.mean()', mean_score)
            print('scores.std()', scores.std())

        return scores

    def evaluate_hash_func_scores(self,
          events,
          curr_subset_events,
          C,
          num_combinations,
          subsets1_arr, subsets2_arr,
          sample_counts=None,
          distances=None,
          cluster_ids=None,
          data_idx_from_sel_cluster=None,
    ):

        assert subsets1_arr.size == num_combinations
        assert subsets2_arr.size == num_combinations

        z_combinations = np.empty(num_combinations, dtype=np.object)
        for curr_combination_idx in range(num_combinations):
            z_combinations[curr_combination_idx] = self.hash_func_obj.compute_hashcode_bit(
                events=events,
                events_ref_set=curr_subset_events,
                subset1=subsets1_arr[curr_combination_idx],
                subset2=subsets2_arr[curr_combination_idx],
            )

        scores = self.compute_scores_for_combinations(
            z_combinations=z_combinations,
            C=C,
            sample_counts=sample_counts,
            distances=distances,
            cluster_ids=cluster_ids,
            data_idx_from_sel_cluster=data_idx_from_sel_cluster,
        )

        mean_score = scores.mean()
        min_score = scores.min()
        max_score = scores.max()

        print(f'scores {scores.size}'
              f' min:{round(scores.min(), 2)}'
              f' max:{round(scores.max(), 2)}'
              f' mean:{round(scores.mean(), 2)}'
              f' std:{round(scores.std(), 2)}'
              )

        if self.debug:
            print('scores', scores)
            print('scores.min()', min_score)
            print('scores.max()', max_score)
            print('scores.mean()', mean_score)
            print('scores.std()', scores.std())

        return scores

    def get_sample_weights_normalized_from_counts(self, path_tuples_all_counts, idx=None):

        if path_tuples_all_counts is not None:
            if idx is not None:
                sample_weights = path_tuples_all_counts[idx]
                epsilon = 1.0e-30
                if sample_weights.dtype != np.float:
                    assert sample_weights.dtype == np.int
                    sample_weights = ((sample_weights.astype(np.float)+epsilon) / float(sample_weights.sum()))
                else:
                    sample_weights = (sample_weights+epsilon) / sample_weights.sum()
            else:
                if path_tuples_all_counts.dtype != np.float:
                    assert path_tuples_all_counts.dtype == np.int
                    sample_weights = path_tuples_all_counts.astype(np.float) / float(path_tuples_all_counts.sum())
                else:
                    sample_weights = path_tuples_all_counts / path_tuples_all_counts.sum()
        else:
            sample_weights = None

        return sample_weights

    def add_intra_cluster_distances(self, events, C, distances):
        # todo: avoid recomputations of distances
        start_time = time.time()
        Cu, Cu_idx_in_original_arr, n_per_C = np.unique(C, axis=0, return_inverse=True, return_counts=True)
        num_clusters = Cu.shape[0]
        assert num_clusters == n_per_C.size
        assert Cu_idx_in_original_arr.size == C.shape[0]
        for curr_cluster_idx in range(num_clusters):
            # computing nearest neighbors for only small clusters
            # todo: try other metrics later or tune it, should upon the nature of data, sparsity in data etc.
            if 1 < n_per_C[curr_cluster_idx] <= self.max_num_samples_in_cluster_for_entropy_compute:
                data_idx_in_cluster = np.where(Cu_idx_in_original_arr == curr_cluster_idx)[0]
                distances[data_idx_in_cluster[:,None], data_idx_in_cluster] = ssd.squareform(
                    ssd.pdist(
                        X=events[data_idx_in_cluster, :],
                        metric='euclidean',
                    )
                )
        print(f'Added intra cluster distances in {round(time.time()-start_time, 2)}s, number of distances added so far, {distances.getnnz()}.')

    def print_events(self, events):
        print('.......................................')
        for _ in range(events.shape[0]):
            print(''.join([str(__) for __ in events[_, :]]))

    def learn_hashcodes_funcs(self,
          events,
          num_hash_functions,
          alpha,
          events_counts_org=None,
    ):
        # todo: use pandas dataframe to save hashcodes for some of the efficient computations,
        #  where groupby upon hashcodes can be used to compute stats for new bit under optimization

        print(events.shape)

        if events_counts_org is not None:
            assert events_counts_org.dtype == np.int
            assert events_counts_org.size == events.shape[0]
            if self.debug:
                print('path_tuples_all_counts_org', events_counts_org)
            assert np.all(events_counts_org >= 1)

        print('alpha', alpha)

        scores_opt_hash_func = np.zeros(num_hash_functions)

        C = None
        cluster_ids = None

        num_data = events.shape[0]

        curr_set_idx = -1
        num_hash_computations = 0

        events_references_objs = np.empty(num_hash_functions, dtype=np.object)
        opt_subset1_objs = np.empty(num_hash_functions, dtype=np.object)
        opt_subset2_objs = np.empty(num_hash_functions, dtype=np.object)

        if self.is_data_entropy_inside_cluster:
            distances_obj = EventDistances(
                events=events,
                max_num_samples_in_cluster_for_intra_distance_compute=self.max_num_samples_in_cluster_for_kl_div_compute,
            )
        else:
            distances_obj = None

        while curr_set_idx < (num_hash_functions-1):

            start_time_hash_opt_greedy = time.time()

            num_hash_computations += 1
            curr_set_idx += 1

            print('.'*70)
            print('curr_set_idx', curr_set_idx)

            print('alpha: {}'.format(alpha))

            cluster = None
            # todo: select subset from a selected cluster
            if curr_set_idx == 0:
                curr_subset12 = self.npr_sample_subset_obj.choice(
                    num_data,
                    alpha,
                    replace=False,
                    p=self.get_sample_weights_normalized_from_counts(
                        events_counts_org
                    )
                )
                subset_size = curr_subset12.size
                data_idx_from_sel_cluster = None
            else:
                curr_subset12, subset_size, data_idx_from_sel_cluster = self.select_high_entropy_cluster(
                        C,
                        subset_size=alpha,
                        sample_weights=self.get_sample_weights_normalized_from_counts(
                            events_counts_org),
                        is_return_cluster_data_indices=True,
                        distances_obj=distances_obj,
                )

            if self.debug:
                print('data_idx_from_sel_cluster', data_idx_from_sel_cluster)

            if self.debug:
                print('curr_subset12', subset_size)

            if events_counts_org is not None:
                events_counts = np.copy(events_counts_org)
            else:
                events_counts = None

            curr_subset_events_references = events[curr_subset12, :]

            if self.is_optimize_split:
                dcos_obj = divide_conquer_optimize_split.DivideConquerOptimizeSplit(
                    subset_size,
                    hash_func=self.hash_func,
                    min_set_size_to_split=self.min_set_size_to_split,
                    max_set_size_for_info_cluster_optimal=self.max_set_size_for_info_cluster_optimal,
                )
                opt_subset1, opt_subset2, max_score = dcos_obj.optimize(
                    events,
                    curr_subset_events_references,
                    C,
                    subset_size,
                    info_theoretic_opt_obj=self,
                    sample_counts=events_counts,
                    distances=distances_obj.distances,
                    cluster_ids=cluster_ids,
                    data_idx_from_sel_cluster=data_idx_from_sel_cluster,
                )
                dcos_obj = None

                scores_opt_hash_func[curr_set_idx] = max_score
                print('scores_opt_hash_func', scores_opt_hash_func[:curr_set_idx+1])
            else:
                opt_subset1 = self.npr_sample_split_obj.choice(subset_size, int(subset_size/2), replace=False)
                opt_subset2 = np.setdiff1d(np.arange(subset_size, dtype=np.int), opt_subset1)

            if self.debug:
                print(opt_subset1)
                print(opt_subset2)
                print(curr_subset_events_references.shape)
                self.print_events(curr_subset_events_references[opt_subset1])
                self.print_events(curr_subset_events_references[opt_subset2])

            events_references_objs[curr_set_idx] = curr_subset_events_references
            opt_subset1_objs[curr_set_idx] = opt_subset1
            opt_subset2_objs[curr_set_idx] = opt_subset2

            # computing hash vector
            c = self.hash_func_obj.compute_hashcode_bit(
                events=events,
                events_ref_set=curr_subset_events_references,
                subset1=opt_subset1,
                subset2=opt_subset2,
            )

            score_recomputed_for_opt_subsets = self.compute_score_for_reference_subset_partition(
                C=C,
                z=c,
                sample_counts=events_counts,
                distances=distances_obj.distances,
                cluster_ids=cluster_ids,
                data_idx_from_sel_cluster=data_idx_from_sel_cluster,
            )

            if self.debug:
                print(c.shape)

            if self.debug:
                print('score_recomputed_for_opt_subsets', score_recomputed_for_opt_subsets)

            scores_opt_hash_func[curr_set_idx] = score_recomputed_for_opt_subsets

            print('scores_opt_hash_func', scores_opt_hash_func[:curr_set_idx+1])

            if curr_set_idx == 0:
                assert C is None
                C = c
            else:
                if curr_set_idx == 1:
                    C = C.reshape(C.size, 1)

                c = c.reshape(c.size, 1)
                C = np.hstack((C, c))

                if self.debug:
                    print('C.shape', C.shape)

            _, cluster_ids = np.unique(C, axis=0, return_inverse=True)

            if self.is_data_entropy_inside_cluster:
                distances_obj.add_intra_cluster_distances(
                    cluster_ids=cluster_ids,
                )

            print(f'{curr_set_idx}th Greedy hash function optimized in {round(time.time()-start_time_hash_opt_greedy, 2)}s.')

        self.events_references_objs = events_references_objs
        self.opt_subset1_objs = opt_subset1_objs
        self.opt_subset2_objs = opt_subset2_objs

        self.save_hash_model(
            events=events,
            hashcodes=C,
        )

        return C

    def compute_hashcodes(self, events):

        if self.debug:
            print(
                self.events_references_objs.shape,
                self.opt_subset1_objs.shape,
                self.opt_subset2_objs.shape,
            )

        num_hash_functions = self.events_references_objs.size
        assert num_hash_functions == self.opt_subset1_objs.size
        assert num_hash_functions == self.opt_subset2_objs.size

        C = np.zeros((events.shape[0], num_hash_functions), dtype=np.bool)
        for curr_hash_func_idx in range(num_hash_functions):
            C[:, curr_hash_func_idx] = self.hash_func_obj.compute_hashcode_bit(
                events=events,
                events_ref_set=self.events_references_objs[curr_hash_func_idx],
                subset1=self.opt_subset1_objs[curr_hash_func_idx],
                subset2=self.opt_subset2_objs[curr_hash_func_idx],
            )

        return C

    def save_hash_model(self, events=None, hashcodes=None):

        print('dumping the object ....')
        start_time = time.time()

        assert self.dir_path is not None

        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)

        np.save(self.dir_path+'/events_references_objs', self.events_references_objs, allow_pickle=True)

        np.save(self.dir_path+'/opt_subset1_objs', self.opt_subset1_objs, allow_pickle=True)

        np.save(self.dir_path+'/opt_subset2_objs', self.opt_subset2_objs, allow_pickle=True)

        with open(self.dir_path+'/hash_func_obj.pickle', 'wb') as h:
            pickle.dump(self.hash_func_obj, h)

        np.save(self.dir_path+'/events', events)
        np.save(self.dir_path+'/hashcodes', hashcodes)

        print(f'Dumped in {round(time.time()-start_time, 2)}s.')

    def load_hash_model(self, load_train_events_and_hashcodes=False):

        if self.debug:
            print('loading the object ....')
        start_time = time.time()

        assert self.dir_path is not None

        self.events_references_objs = np.load(self.dir_path+'/events_references_objs.npy', allow_pickle=True)
        self.opt_subset1_objs = np.load(self.dir_path+'/opt_subset1_objs.npy', allow_pickle=True)
        self.opt_subset2_objs = np.load(self.dir_path+'/opt_subset2_objs.npy', allow_pickle=True)

        with open(self.dir_path+'/hash_func_obj.pickle', 'rb') as h:
            self.hash_func_obj = pickle.load(h)

        if self.debug:
            print(f'loaded in {round(time.time()-start_time, 2)}s.')

        if load_train_events_and_hashcodes:
            events = np.load(self.dir_path+'/events.npy')
            hashcodes = np.load(self.dir_path+'/hashcodes.npy')
            return events, hashcodes


