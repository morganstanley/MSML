from multiprocessing import Process, Queue
import json
import os
import pandas
from tqdm import tqdm
from itertools import product
import argparse

from environments.toy_env import ToyEnv
from utils.policy_evaluation import evaluate_policy
from policies.generic_policies import EpsilonSmoothPolicy
from policies.toy_env_policies import ThresholdPolicy
from utils.offline_dataset import OfflineRLDataset
from models.fnn_nuisance_model import FeedForwardNuisanceModel
from models.fnn_critic import FeedForwardCritic
from learners.iterative_sieve_critic import IterativeSieveLearner
from learners.robust_fqi_learner import RobustFQILearner


def main(config):
    results_path = config["results_path"]
    num_rep_range = config["num_rep_range"]
    num_restart_range = config["num_restart_range"]
    lambda_range = config["adversarial_lambda_values"]
    job_queue = Queue()
    num_jobs = 0
    job_iter = product(num_rep_range, num_restart_range, lambda_range)

    for rep_i, restart_i, lmbda in job_iter:
        job = {"rep_i": rep_i, "adversarial_lambda": lmbda,
               "restart_i": restart_i}
        job_queue.put(job)
        num_jobs += 1

    procs = []
    results_queue = Queue()
    if "devices" in config:
        devices = config["devices"]
    else:
        devices = [None]
    for i in range(config["num_workers"]):
        device = devices[i % len(devices)]
        p_args = (job_queue, results_queue, config, device)
        p = Process(target=run_jobs_loop, args=p_args)
        procs.append(p)
        job_queue.put("STOP")
        p.start()

    all_results = []
    print("running experiments:")
    for _ in tqdm(range(num_jobs)):
        next_result = results_queue.get()
        all_results.extend(next_result)
        df = pandas.DataFrame(all_results)
        df.to_csv(results_path, index=False)
    for p in procs:
        p.join()

def run_jobs_loop(job_queue, results_queue, config, device):
    for job_kwargs in iter(job_queue.get, "STOP"):
        results = single_run(config=config, device=device, **job_kwargs)
        results_queue.put(results)

def single_run(config, rep_i, restart_i, adversarial_lambda, device=None):


    env = ToyEnv(s_init=config["s_threshold"], adversarial=False)
    s_dim = env.get_s_dim()
    num_a = env.get_num_a()
    gamma = config["gamma"]
    s_threshold = config["s_threshold"]
    batch_size = config["batch_size"]

    pi_e = ThresholdPolicy(env, s_threshold=s_threshold)
    pi_e_name = config["pi_e_name"]

    model_config = config["model_config"]
    model = FeedForwardNuisanceModel(s_dim=s_dim, num_a=num_a, gamma=gamma,
                                     config=model_config, device=device)
    critic_class = FeedForwardCritic
    critic_config = config["critic_config"]
    critic_kwargs = {
        "s_dim": s_dim,
        "num_a": num_a,
        "config": critic_config
    }

    base_dataset_path_train = config["base_dataset_path_train"]
    base_dataset_path_test = config["base_dataset_path_test"]
    dataset_path_train = "_".join([base_dataset_path_train, str(rep_i)])
    dataset_path_test = "_".join([base_dataset_path_test, str(rep_i)])
    train_dataset = OfflineRLDataset.load_dataset(dataset_path_train)
    test_dataset = OfflineRLDataset.load_dataset(dataset_path_test)
    if device is not None:
        train_dataset.to(device)
        test_dataset.to(device)

    # first train q/beta
    q_learner = RobustFQILearner(
        nuisance_model=model, gamma=gamma, use_dual_cvar=True,
        adversarial_lambda=adversarial_lambda,
    )
    s_init, a_init = env.get_s_a_init(pi_e)
    if device is not None:
        s_init = s_init.to(device)
        a_init = a_init.to(device)

    dl_test = test_dataset.get_batch_loader(batch_size=batch_size)
    evaluate_pv_kwargs = {
        "s_init": s_init, "a_init": a_init,
        "dl_test": dl_test, "pi_e_name": pi_e_name,
    }
    q_learner_kwargs = config["q_learner_kwargs"]
    q_learner.train(
        dataset=train_dataset, pi_e_name=pi_e_name, verbose=False,
        device=device, evaluate_pv_kwargs=evaluate_pv_kwargs,
        **q_learner_kwargs,
    )

    # second train eta
    eta_learner = IterativeSieveLearner(
        nuisance_model=model, gamma=gamma, use_dual_cvar=True,
        adversarial_lambda=adversarial_lambda,
        train_q_beta=False, train_eta=True, train_w=False, debug_beta=False,
    )
    eta_learner_kwargs = config["eta_learner_kwargs"]
    eta_learner.train(
        dataset=train_dataset, pi_e_name=pi_e_name, verbose=False,
        device=device, init_basis_func=env.bias_basis_func,
        num_init_basis=1, evaluate_pv_kwargs=evaluate_pv_kwargs,
        critic_class=critic_class, s_init=s_init,
        critic_kwargs=critic_kwargs, **eta_learner_kwargs,
    )

    # third train w
    w_learner = IterativeSieveLearner(
        nuisance_model=model, gamma=gamma, use_dual_cvar=True,
        adversarial_lambda=adversarial_lambda,
        train_q_beta=False, train_eta=False, train_w=True, debug_beta=False,
    )
    w_learner_kwargs = config["w_learner_kwargs"]
    w_learner.train(
        dataset=train_dataset, pi_e_name=pi_e_name, verbose=False,
        device=device, init_basis_func=env.bias_basis_func,
        num_init_basis=1, evaluate_pv_kwargs=evaluate_pv_kwargs,
        critic_class=critic_class, s_init=s_init,
        critic_kwargs=critic_kwargs, **w_learner_kwargs,
    )

    model_path_base = config["base_model_path"]
    model_name = "model"
    model_name += f"_lambda={adversarial_lambda}"
    model_name += f"_rep={rep_i}"
    model_path = os.path.join(model_path_base, model_name)
    model.save_model(model_path)

    ## evaluate model using 3 policy value estimators

    q_pv = model.estimate_policy_val_q(
        s_init=s_init, a_init=a_init, gamma=gamma
    )
    w_pv = model.estimate_policy_val_w(
        dl=dl_test, pi_e_name=pi_e_name,
    )
    w_pv_norm = model.estimate_policy_val_w(
        dl=dl_test, pi_e_name=pi_e_name, normalize=True,
    )
    dr_pv = model.estimate_policy_val_dr(
        s_init=s_init, a_init=a_init, pi_e_name=pi_e_name, dl=dl_test,
        adversarial_lambda=adversarial_lambda, gamma=gamma, dual_cvar=True,
    )
    dr_pv_norm = model.estimate_policy_val_dr(
        s_init=s_init, a_init=a_init, pi_e_name=pi_e_name, dl=dl_test,
        adversarial_lambda=adversarial_lambda, gamma=gamma, dual_cvar=True,
        normalize=True,
    )
    pv_results = {
        "q": q_pv, "w": w_pv, "w_norm": w_pv_norm,
        "dr": dr_pv, "dr_norm": dr_pv_norm, 
    }
    results = []
    for key, val in pv_results.items():
        row = {
            "rep_i": rep_i, "restart_i": restart_i, 
            "lambda": adversarial_lambda,
            "est_policy_value": val, "estimator": key,
        }
        results.append(row)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment_config.json")
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
    main(config=config)
