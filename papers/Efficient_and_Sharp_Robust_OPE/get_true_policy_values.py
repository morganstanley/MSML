import json
import pandas

from environments.toy_env import ToyEnv
from policies.toy_env_policies import ThresholdPolicy
from utils.policy_evaluation import evaluate_policy


def main(config):
    s_threshold = config["s_threshold"]
    gamma = config["gamma"]
    results = []
    for adversarial_lambda in config["adversarial_lambda_values"]:
        env_eval = ToyEnv(s_init=s_threshold, adversarial=True,
                            adversarial_lambda=adversarial_lambda)
        pi_e = ThresholdPolicy(env_eval, s_threshold=s_threshold)
        pi_e_val = evaluate_policy(env_eval, pi_e, gamma, min_prec=5e-5)
        print(f"lambda: {adversarial_lambda}, true v(pi_e): {pi_e_val}")
        row = {"lambda": adversarial_lambda, "true_policy_value": pi_e_val}
        results.append(row)
    results_path = "results_true_policy_value.csv"
    df = pandas.DataFrame(results)
    df.to_csv(results_path, index=False)


if __name__ == "__main__":
    with open("experiment_config.json") as f:
        config = json.load(f)
    main(config)