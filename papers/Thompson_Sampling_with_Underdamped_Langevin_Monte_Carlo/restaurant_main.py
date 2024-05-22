import random
import numpy as np
from tqdm import tqdm
from model.flags import get_flags
from model.langevin_res import over_damped_stochastic, under_damped_stochastic
from model.utils import LoadData

def run_func(args):
    # Hyperparameters
    dim = args.dim
    gamma = args.gamma
    batch = args.batch_size
    n_arms = 5

    n_rounds = args.n_round
    step_size = args.step_size
    n_iterations = args.n_iter
    prior_mean = np.zeros(dim)
    prior_variance = np.eye(dim)
    arm_covariances = [np.eye(dim) for _ in range(n_arms)]  # Known covariance for all arms

    # Simulated true means
    true_means = np.random.randn(n_arms, dim) * 5
    idx = np.argsort(-np.linalg.norm(true_means, axis=1))
    true_means = true_means[idx].copy()

    # Load data
    loader = LoadData('./data')

    # some metrics
    counts = np.zeros(n_arms)  # number of times to play arm
    sum_rewards = np.zeros((n_arms, dim))
    choose_arm_logs = []
    regret_total = 0
    regret_logs = [regret_total]
    observation = []
    current_position = []
    current_velocity = []
    for i in range(n_arms):
        observation.append([])
        current_position.append(prior_mean)
        current_velocity.append(np.zeros(prior_mean.shape))

    if args.sampler == 'overdamp':
        pbar = tqdm(range(n_rounds), dynamic_ncols=True, smoothing=0.1, desc='Overdamped TS')
    elif args.sampler == 'underdamp':
        pbar = tqdm(range(n_rounds), dynamic_ncols=True, smoothing=0.1, desc='Underdamped TS')
    else:
        raise 'Unknown sampler'

    for e in pbar:
        sampled_means = []

        for arm in range(n_arms):
            if counts[arm] == 0:
                if len(observation[arm]) == 0:
                    sampled_means.append(
                        np.random.multivariate_normal(prior_mean, prior_variance))  # No observation, sample from prior
                else:
                    if args.sampler == 'overdamp':
                        sampled_mean = over_damped_stochastic(
                            observation[arm], step_size, n_iterations, prior_mean, np.linalg.inv(prior_variance),
                            current_position[arm], batch_size=batch, likelihood_var=loader.reward_std[arm])
                    elif args.sampler == 'underdamp':
                        sampled_mean, vel = under_damped_stochastic(
                            observation[arm], step_size, n_iterations, prior_mean, np.linalg.inv(prior_variance),
                            [current_position[arm], current_velocity[arm]],
                            gamma=gamma, batch_size=batch, likelihood_var=loader.reward_std[arm])
                        current_velocity[arm] = vel

                    sampled_means.append(sampled_mean)
                    current_position[arm] = sampled_mean

            else:
                # # Sample from posterior using Langevin dynamics
                obs = np.mean(np.random.choice(loader.reward_sample[arm], size=args.reward_size))
                observation[arm].append(obs)

                if args.sampler == 'overdamp':
                    sampled_mean = over_damped_stochastic(
                        observation[arm], step_size, n_iterations, prior_mean, np.linalg.inv(prior_variance),
                        current_position[arm], batch_size=batch, likelihood_var=loader.reward_std[arm])
                elif args.sampler == 'underdamp':
                    sampled_mean, vel = under_damped_stochastic(
                        observation[arm], step_size, n_iterations, prior_mean, np.linalg.inv(prior_variance),
                        [current_position[arm], current_velocity[arm]],
                        gamma=gamma, batch_size=batch, likelihood_var=loader.reward_std[arm])
                    current_velocity[arm] = vel
                sampled_means.append(sampled_mean)
                current_position[arm] = sampled_mean

        chosen_arm = np.argmax([np.linalg.norm(mean) for mean in sampled_means])  # Select arm with highest norm
        choose_arm_logs.append(chosen_arm)

        reward = np.random.multivariate_normal(true_means[chosen_arm], arm_covariances[chosen_arm])
        if chosen_arm == 0:
            regret = 0
        else:
            optimal_reward = np.random.multivariate_normal(true_means[0], arm_covariances[0])
            regret = np.linalg.norm(optimal_reward) - np.linalg.norm(reward)
        regret_total += regret
        regret_logs.append(regret_total)

        # Update counts and observed rewards
        counts[chosen_arm] += 1
        sum_rewards[chosen_arm] += reward

        pbar.set_postfix({
            'Reward': '{0:1.4e}'.format(np.linalg.norm(reward)),
            'Regret': '{0:1.4e}'.format(regret_total)})

    return counts, regret_logs, choose_arm_logs


if __name__ == "__main__":
    # Get flags
    flags = get_flags()

    flags.sampler = 'overdamp'
    count1, logs1, arm_logs1 = run_func(flags)

    flags.sampler = 'underdamp'
    count2, logs2, arm_logs2 = run_func(flags)

