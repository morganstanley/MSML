import time
import numpy as np
import matplotlib.pyplot as plt

import importlib
from hash_finance import hash_function_representations_info_theoretic_optimization as hf_ito
importlib.reload(hf_ito)

from hash_finance import information_theoretic_measures_hashcodes as itmh
importlib.reload(itmh)


class HashcodeToInt:

    def __init__(self):
        pass

    def map_hashcode_to_int(self, hashcodes):
        assert hashcodes.dtype == np.bool
        # columns represent different bits
        num_bits = hashcodes.shape[1]
        hashcode_ints = np.zeros(hashcodes.shape[0], dtype=np.int)
        for j in range(num_bits):
            hashcode_ints += (2**(num_bits-j-1))*hashcodes[:, j]
        return hashcode_ints


class HashcodeModel:

    def __init__(
            self, debug=False,
            dir_path_for_saving_hash_models = '/v/campus/ny/cs/aiml_build/sahigarg/saved_hash_models',
            C=1,
            # C=10,
            penalty='l1',
    ):
        self.dir_path_for_saving_hash_models = dir_path_for_saving_hash_models
        self.debug = debug
        self.eps = 1e-5
        self.model_name = 'all_tickers'
        self.C = C
        self.penalty = penalty

    def load_pretrained_hashcodes_and_events(self, model_name=None):
        hashcodes_obj = hf_ito.HashFunctionRepresentations(
            dir_path=self.dir_path_for_saving_hash_models,
            model_name=self.model_name if model_name is None else model_name,
        )
        events, hashcodes = hashcodes_obj.load_hash_model(load_train_events_and_hashcodes=True)
        return events, hashcodes

    def visualize_quality_of_hashcodes(self, events, hashcodes_of_events):

        start_time = time.time()

        num_bits = hashcodes_of_events.shape[1]
        num_bits_arr = (np.arange(0.1, 1.01, 0.1)*num_bits).astype(np.int)

        # expensive to compute
        entropies_knn = np.zeros(num_bits_arr.size, np.float)
        entropies_knn_hash_bins = np.zeros(num_bits_arr.size, np.float)

        KL_div_loss = np.zeros(num_bits_arr.size, np.float)
        max_bin_size = np.zeros(num_bits_arr.size, np.float)
        min_bin_size = np.zeros(num_bits_arr.size, np.float)
        mean_bin_size = np.zeros(num_bits_arr.size, np.float)
        median_bin_size = np.zeros(num_bits_arr.size, np.float)

        for curr_idx, curr_num_bits in enumerate(num_bits_arr):

            print('.', end='')

            curr_hashcodes = hashcodes_of_events[:, :curr_num_bits]

            itmh_obj = itmh.InformationTheoreticMeasuresHashcodes()
            entropies_knn[curr_idx], entropies_knn_hash_bins[curr_idx] = itmh_obj.compute_kNN_entropy_cond_hashcodes(
                X=events,
                C=curr_hashcodes,
            )

            KL_div_loss[curr_idx] = itmh_obj.compute_kl_divergence_of_density_from_hashcodes_as_bins_wrt_uniform_dist(
                C=curr_hashcodes,
            )

            _, bin_counts_from_curr_sel = np.unique(
                curr_hashcodes,
                axis=0,
                return_counts=True,
            )
            max_bin_size[curr_idx] = bin_counts_from_curr_sel.max()
            min_bin_size[curr_idx] = bin_counts_from_curr_sel.min()
            mean_bin_size[curr_idx] = bin_counts_from_curr_sel.mean()
            median_bin_size[curr_idx] = np.median(bin_counts_from_curr_sel)

            bin_counts_from_curr_sel, curr_hashcodes = None, None

        fontsize=14

        plt.close()
        plt.rcParams["font.family"] = "Times New Roman"
        plt.plot(num_bits_arr, entropies_knn, 'o-', color='navy', label='Hx_c')
        plt.plot(num_bits_arr, entropies_knn_hash_bins, 's-', color='brown', label='Hc')
        plt.xlabel('Number of Selected Hash functions', fontsize=fontsize)
        plt.ylabel('kNN-Entropy of Density per Hash Bins', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=fontsize)
        plt.show()

        plt.close()
        plt.rcParams["font.family"] = "Times New Roman"
        plt.plot(num_bits_arr, KL_div_loss, 'x-', color='k')
        plt.xlabel('Number of Selected Hash functions', fontsize=fontsize)
        plt.ylabel('KL-D of Hash-Density w.r.t. Uniform Density', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.show()

        plt.close()
        plt.rcParams["font.family"] = "Times New Roman"
        plt.plot(num_bits_arr, mean_bin_size, 's-', color='black', label='Mean')
        plt.plot(num_bits_arr, median_bin_size, '+-', color='slategray', label='Median')
        plt.plot(num_bits_arr, min_bin_size, 'o-', color='brown', label='Min')
        plt.plot(num_bits_arr, max_bin_size, 'x-', color='navy', label='Max')
        plt.xlabel('Number of Selected Hash functions', fontsize=fontsize)
        plt.ylabel('Hash Bins Size', fontsize=fontsize)
        plt.yscale('log')
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=fontsize)
        plt.show()

        print(f'Time to compute visuals of hashcodes was {time.time()-start_time}s.')

    def compute_hashcodes(self,
                          features,
                          is_optimize=True,
                          num_cores=1,
                          num_hash_functions=20,
                          alpha=128,
                          max_num_samples_in_cluster_for_entropy_compute=1000,
                          max_num_samples_in_sel_cluster_for_kl_div_compute=300,
                          max_num_samples_in_cluster_for_kl_div_compute=100,
                          min_set_size_to_split=8,
                          max_set_size_for_info_cluster_optimal=32,
                          is_equal_split=True,
                          visualize_quality_of_hashcodes=False,
                          model_name=None,
    ):
        # print(features.shape, features.mean())
        # print(event_features)

        if is_optimize:
            hashcodes_obj = hf_ito.HashFunctionRepresentations(
                num_cores=num_cores,
                is_optimize_split=True,
                min_set_size_to_split=min_set_size_to_split,
                max_set_size_for_info_cluster_optimal=max_set_size_for_info_cluster_optimal,
                is_optimize_hash_functions_nonredundancy=True,
                nonredundancy_score_beta=1.0,
                is_data_entropy_inside_cluster=True,
                max_num_samples_in_cluster_for_entropy_compute=max_num_samples_in_cluster_for_entropy_compute,
                max_num_samples_in_sel_cluster_for_kl_div_compute=max_num_samples_in_sel_cluster_for_kl_div_compute,
                max_num_samples_in_cluster_for_kl_div_compute=max_num_samples_in_cluster_for_kl_div_compute,
                k_for_entropy_estimate=2,
                # seems redundant term w.r.t. Hz_C
                is_data_entropy_from_hashcodes_as_bins=False,
                is_equal_split=is_equal_split,
                tune_hp_hash_func=False,
                dir_path=self.dir_path_for_saving_hash_models,
                model_name=self.model_name if model_name is None else model_name,
                C=self.C,
                penalty=self.penalty,
            )
            events_hashcodes = hashcodes_obj.learn_hashcodes_funcs(
                events=features,
                num_hash_functions=num_hash_functions,
                alpha=alpha,
            )
            if visualize_quality_of_hashcodes:
                self.visualize_quality_of_hashcodes(
                    events=features,
                    hashcodes_of_events=events_hashcodes,
                )
        else:
            hashcodes_obj = hf_ito.HashFunctionRepresentations(
                dir_path=self.dir_path_for_saving_hash_models,
                model_name=self.model_name if model_name is None else model_name,
            )
            hashcodes_obj.load_hash_model()
            events_hashcodes = hashcodes_obj.compute_hashcodes(events=features)

        return events_hashcodes
