import importlib
from . import hash_function_representations_info_theoretic_optimization as hf_ito
importlib.reload(hf_ito)


class HashEvents:

    def __init__(
            self,
            hash_func,
            num_cores=1,
            alpha=32,
            min_set_size_to_split=8,
            dir_path=None,
            model_name=None,
            seed_val=0,
    ):
        self.hash_func = hash_func
        self.num_cores = num_cores
        self.alpha = alpha
        self.min_set_size_to_split = min_set_size_to_split
        self.dir_path = dir_path
        self.model_name = model_name
        self.hashcodes_obj = hf_ito.HashFunctionRepresentations(
            hash_func=self.hash_func,
            num_cores=self.num_cores,
            seed_val=seed_val,
            is_optimize_split=True,
            min_set_size_to_split=self.min_set_size_to_split,
            is_optimize_hash_functions_nonredundancy=True,
            nonredundancy_score_beta=1.0,
            is_data_entropy_inside_cluster=True,
            max_num_samples_in_cluster_for_entropy_compute=100,
            k_for_entropy_estimate=2,
            # seems redundant term w.r.t. Hz_C
            is_data_entropy_from_hashcodes_as_bins=False,
            is_equal_split=True,
            tune_hp_hash_func=False,
            dir_path=self.dir_path,
            model_name=self.model_name,
        )

    def hashcodes_for_events(
            self,
            events,
            num_hash_bits=10,
    ):
        hashcodes = self.hashcodes_obj.learn_hashcodes_funcs(
            events=events,
            num_hash_functions=num_hash_bits,
            alpha=self.alpha,
        )
        assert hashcodes.shape[1] == num_hash_bits

        return hashcodes
