import numpy as np
import time
import math
import importlib

import scipy.stats as ss
import scipy.spatial.distance as ssd

from . import information_theoretic_measures_hashcodes as itmh
importlib.reload(itmh)


class DivideConquerOptimizeSplit:
    # this is relevant for optimizing splits of large sets

    def __init__(self,
                 alpha,
                 hash_func,
                 min_set_size_to_split=4,
                 max_set_size_for_info_cluster_optimal=32,
                 debug=False,
    ):
        self.alpha = alpha
        self.hash_func = hash_func
        self.min_set_size_to_split = min_set_size_to_split
        self.max_set_size_for_info_cluster_optimal = max_set_size_for_info_cluster_optimal
        self.debug = debug
        self.itmh_obj = itmh.InformationTheoreticMeasuresHashcodes()

    def random_divide(self, set):

        num_divisions = int(math.ceil(float(set.size)/self.min_set_size_to_split))

        if self.debug:
            print('num_divisions', num_divisions)

        divisions = [[] for d1 in range(num_divisions)]
        for curr_division in range(num_divisions):
            divisions[curr_division] = np.arange(curr_division, set.size, num_divisions, dtype=np.int)

        if self.debug:
           print('divisions', divisions)

        return divisions

    def elements_in_division(self, division, splits):

        elements = np.array([], dtype=np.int)

        for curr_idx in division:
            assert isinstance(splits[curr_idx], np.ndarray)
            elements = np.concatenate((elements, splits[curr_idx]))

        return elements

    def expand_combinations_per_elements_in_splits(self,
                                                   subsets_arr,
                                                   splits):

        assert subsets_arr.dtype == np.object

        for curr_idx in range(subsets_arr.size):
            curr_subset = subsets_arr[curr_idx]
            elements_in_subset = self.elements_in_division(curr_subset, splits=splits)
            subsets_arr[curr_idx] = elements_in_subset

        return subsets_arr

    def map_combinations_via_dict(self, subsets_arr, dict_element_in_division_to_index):

        assert subsets_arr.dtype == np.object

        new_subsets_arr = -10000*np.ones(subsets_arr.shape, dtype=np.object)

        for curr_idx in range(subsets_arr.size):
            curr_subset = subsets_arr[curr_idx]
            curr_subset = self.map_elements_via_dict(
                elements=curr_subset,
                dict_element_in_division_to_index=dict_element_in_division_to_index,
            )
            new_subsets_arr[curr_idx] = curr_subset

        return new_subsets_arr

    def map_elements_via_dict(self, elements, dict_element_in_division_to_index):

        assert elements.dtype == np.int
        new_elements = -1000*np.ones(elements.shape, dtype=elements.dtype)

        for curr_idx in range(elements.size):
            curr_element = elements[curr_idx]
            new_elements[curr_idx] = dict_element_in_division_to_index[curr_element]

        return new_elements

    def inv_array_indices(self, elements_in_division):

        if elements_in_division.size > 100:
            print('warning: this implementation may be slow for large arrays')
            # raise AssertionError, 'this implementation may be slow for large arrays'

        dict_element_in_division_to_index = {}
        for curr_idx in range(elements_in_division.size):
            curr_element = elements_in_division[curr_idx]
            dict_element_in_division_to_index[curr_element] = curr_idx

        return dict_element_in_division_to_index

    def get_combinations(self,
                         curr_division,
                         splits,
                         info_theoretic_opt_obj,
    ):

        subsets1_arr, subsets2_arr = info_theoretic_opt_obj.get_all_combinations(
            subset_size=curr_division.size, superset=curr_division
        )

        subsets1_arr = self.expand_combinations_per_elements_in_splits(
            subsets1_arr,
            splits,
        )
        subsets2_arr = self.expand_combinations_per_elements_in_splits(
            subsets2_arr,
            splits,
        )

        return subsets1_arr, subsets2_arr

    def evaluate_kl_div_scores(self,
            events,
            num_combinations,
            subsets1_arr,
            subsets2_arr,
            k_for_kl_div_estimate,
    ):
        start_time = time.time()

        num_data = events.shape[0]
        distances = ssd.squareform(
            ssd.pdist(
                X=events,
                metric='euclidean',
            )
        )
        events = None

        # parallelize it

        scores = np.zeros(num_combinations, dtype=np.float)
        z = np.zeros(num_data, dtype=np.bool)
        for curr_combination_idx in range(num_combinations):
            subset1 = subsets1_arr[curr_combination_idx]
            subset2 = subsets2_arr[curr_combination_idx]
            assert z.size == (subset1.size+subset2.size)
            z[subset1] = False
            z[subset2] = True
            scores[curr_combination_idx] = self.itmh_obj.compute_KL_div_between_partitions_within_cluster(
                distances=distances,
                z=z,
                k=k_for_kl_div_estimate,
            )

        print(
            f'KL-D scores {scores.size}'
              f' min:{round(scores.min(), 2)}'
              f' max:{round(scores.max(), 2)}'
              f' mean:{round(scores.mean(), 2)}'
              f' std:{round(scores.std(), 2)}'
              f' {round(time.time()-start_time, 2)}s'
        )

        return scores

    def optimize(self,
                 events,
                 curr_subset_events,
                 C,
                 subset_size,
                 info_theoretic_opt_obj,
                 sample_counts=None,
                 distances=None,
                 cluster_ids=None,
                 data_idx_from_sel_cluster=None,
    ):
        assert subset_size == curr_subset_events.shape[0]

        print(f'subset_size={subset_size}')
        splits = np.arange(subset_size, dtype=np.int).reshape((subset_size, 1))

        while len(splits) > 2:

            if self.debug:
                print('*****************************************')
                print('splits', splits)

            num_splits = len(splits)

            set = np.arange(num_splits, dtype=np.int)
            if self.debug:
                print('set', set)

            divisions = self.random_divide(set)
            if self.debug:
                print('divisions', divisions)

            new_splits = []

            for curr_division in divisions:

                if self.debug:
                    print('.............................')
                    print('curr_division', curr_division)

                # --------------------------------------------------------------------------------
                start_time_preprocessing = time.time()

                elements_in_division = self.elements_in_division(curr_division, splits=splits)
                # if self.debug:
                print('elements_in_division', elements_in_division.size)

                subsets1_arr, subsets2_arr = self.get_combinations(
                    curr_division,
                    splits,
                    info_theoretic_opt_obj,
                )
                num_combinations = subsets1_arr.size
                assert num_combinations == subsets2_arr.size

                start_time_inverse_index = time.time()

                dict_element_in_division_to_index = self.inv_array_indices(elements_in_division)
                # print('dict_element_in_division_to_index', dict_element_in_division_to_index)

                subsets1_arr_adjusted_per_kernel_matrix = self.map_combinations_via_dict(
                    subsets_arr=subsets1_arr,
                    dict_element_in_division_to_index=dict_element_in_division_to_index,
                )
                subsets2_arr_adjusted_per_kernel_matrix = self.map_combinations_via_dict(
                    subsets_arr=subsets2_arr,
                    dict_element_in_division_to_index=dict_element_in_division_to_index,
                )

                if self.debug:
                    print(f'Inverse indices in {round(time.time()-start_time_inverse_index, 3)}s.')

                if self.debug:
                    print(time.time() - start_time_preprocessing)

                if elements_in_division.size > self.max_set_size_for_info_cluster_optimal:

                    scores = info_theoretic_opt_obj.evaluate_hash_func_scores(
                        events,
                        curr_subset_events[elements_in_division, :],
                        C,
                        num_combinations,
                        subsets1_arr_adjusted_per_kernel_matrix,
                        subsets2_arr_adjusted_per_kernel_matrix,
                        sample_counts=sample_counts,
                        distances=distances,
                        cluster_ids=cluster_ids,
                        data_idx_from_sel_cluster=data_idx_from_sel_cluster,
                    )

                    scores_kl = self.evaluate_kl_div_scores(
                        curr_subset_events[elements_in_division, :],
                        num_combinations,
                        subsets1_arr_adjusted_per_kernel_matrix,
                        subsets2_arr_adjusted_per_kernel_matrix,
                        k_for_kl_div_estimate=info_theoretic_opt_obj.k_for_entropy_estimate,
                    )

                    print(ss.spearmanr(scores, scores_kl))

                    assert scores.shape == scores_kl.shape
                    scores = scores + scores_kl
                    scores_kl = None
                else:
                    scores = self.evaluate_kl_div_scores(
                        curr_subset_events[elements_in_division, :],
                        num_combinations,
                        subsets1_arr_adjusted_per_kernel_matrix,
                        subsets2_arr_adjusted_per_kernel_matrix,
                        k_for_kl_div_estimate=info_theoretic_opt_obj.k_for_entropy_estimate,
                    )

                opt_idx = scores.argmax()
                if self.debug:
                    print('opt_idx', opt_idx)

                max_score = scores[opt_idx]

                opt_subset1 = subsets1_arr[opt_idx]
                if self.debug:
                    print('opt_subset1', opt_subset1)
                new_splits.append(np.copy(opt_subset1))

                opt_subset2 = subsets2_arr[opt_idx]
                if self.debug:
                    print('opt_subset2', opt_subset2)
                new_splits.append(np.copy(opt_subset2))

                print(f'opt split: {opt_subset1.size}, {opt_subset2.size}')

            new_splits = np.array(new_splits)
            splits = new_splits
            del new_splits

        if self.debug:
            print('splits', splits)
        assert isinstance(splits, np.ndarray)

        if self.debug:
            print('splits.shape', splits.shape)
        assert splits.shape[0] == 2
        assert (splits[0].size + splits[1].size) == subset_size

        return splits[0], splits[1], max_score
