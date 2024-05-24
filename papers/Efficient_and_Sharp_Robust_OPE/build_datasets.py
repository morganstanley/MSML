import json

from environments.toy_env import ToyEnv
# from experiment_config import config
from policies.generic_policies import EpsilonSmoothPolicy
from policies.toy_env_policies import ThresholdPolicy
from utils.offline_dataset import OfflineRLDataset

def main(config_path="experiment_config.json"):
    with open(config_path) as f:
        config = json.load(f)
    num_rep_range = config["num_rep_range"]
    s_threshold = config["s_threshold"]
    num_sample = config["num_sample"]
    pi_b_threshold = config["pi_b_threshold"]
    pi_b_epsilon = config["pi_b_epsilon"]

    for i in num_rep_range:
        print(f"building datasets for rep {i}")

        env = ToyEnv(s_init=s_threshold, adversarial=False)
        pi_e = ThresholdPolicy(env, s_threshold=s_threshold)
        pi_e_name = config["pi_e_name"]

        base_dataset_path_train = config["base_dataset_path_train"]
        base_dataset_path_test = config["base_dataset_path_test"]

        ## build datasets and save them

        pi_base = ThresholdPolicy(env, s_threshold=pi_b_threshold)
        pi_b = EpsilonSmoothPolicy(env, pi_base=pi_base, epsilon=pi_b_epsilon)

        dataset = OfflineRLDataset()
        burn_in = config["burn_in"]
        thin = config["thin"]
        dataset.sample_new_trajectory(env=env, pi=pi_b, burn_in=burn_in,
                                    num_sample=num_sample, thin=thin)

        test_dataset = OfflineRLDataset()
        test_dataset.sample_new_trajectory(env=env, pi=pi_b, burn_in=burn_in,
                                        num_sample=num_sample, thin=thin)

        dataset.apply_eval_policy(pi_e_name, pi_e)
        test_dataset.apply_eval_policy(pi_e_name, pi_e)

        dataset_path_train = "_".join([base_dataset_path_train, str(i)])
        dataset_path_test = "_".join([base_dataset_path_test, str(i)])

        dataset.save_dataset(dataset_path_train)
        test_dataset.save_dataset(dataset_path_test)


if __name__ == "__main__":
    main()
