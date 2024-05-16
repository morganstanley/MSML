import numpy as np
import random


def over_damped(observations, step_size, n_iterations, prior_mean, prior_variance_inv, current,
                batch_size=10, prior_type=None):
    x_mean = current.copy()

    for _ in range(n_iterations):
        n = len(observations)
        if batch_size < n:
            sub_observe = random.sample(observations, k=batch_size)
            try:
                if prior_type is not None:
                    gradient = n * (np.mean(sub_observe, axis=0) - x_mean)
                else:
                    gradient = -prior_variance_inv.dot(x_mean - prior_mean) + n * (
                            np.mean(sub_observe, axis=0) - x_mean)
            except ValueError:
                print("value error")
        elif n == 0:
            gradient = -prior_variance_inv.dot(x_mean - prior_mean)
        else:
            if prior_type is not None:
                gradient = n * (np.mean(observations, axis=0) - x_mean)
            else:
                gradient = -prior_variance_inv.dot(x_mean - prior_mean) + n * (np.mean(observations, axis=0) - x_mean)
                
        x_mean = x_mean - step_size * gradient + np.sqrt(2 * step_size) * np.random.randn(x_mean.shape[0])

    return x_mean


def under_damped(observations, step_size, n_iterations, prior_mean, prior_variance_inv, current,
                 gamma=2, L=1, batch_size=10, prior_type=None):

    x_mean = current[0].copy()
    v_mean = current[1].copy()
    dim = prior_mean.shape[0]
    t = step_size

    for _ in range(n_iterations):
        n = len(observations)
        if batch_size < n:
            sub_observe = random.sample(observations, k=batch_size)
            try:
                if prior_type is not None:
                    gradient = n * (np.mean(sub_observe, axis=0) - x_mean)
                else:
                    gradient = -prior_variance_inv.dot(x_mean - prior_mean) + n * (
                            np.mean(sub_observe, axis=0) - x_mean)
            except ValueError:
                print("value error")
        elif n == 0:
            gradient = -prior_variance_inv.dot(x_mean - prior_mean)
        else:
            if prior_type is not None:
                gradient = n * (np.mean(observations, axis=0) - x_mean)
            else:
                gradient = -prior_variance_inv.dot(x_mean - prior_mean) + n * (np.mean(observations, axis=0) - x_mean)

        x_sub = x_mean + (1 - np.exp(-gamma * t)) * v_mean / gamma - (
                t - (1 - np.exp(-gamma * t)) / gamma) * gradient * (gamma * L)
        v_sub = v_mean * np.exp(-gamma * t) - (1 - np.exp(-gamma * t)) / (gamma * L) * gradient

        # Diagonal variances computed separately, to speed up the computation
        var_1 = 2 * (t - np.exp(-2 * gamma * t) / (2 * gamma) - 3 / (2 * gamma) +
                     2 * np.exp(-gamma * t) / gamma) / (gamma * L) 
        var_2 = (1 - np.exp(-2 * gamma * t)) / L 

        # Generating samples using independent Gaussian draws
        x_mean = np.random.normal(loc=x_sub, scale=np.sqrt(var_1), size=dim)
        v_mean = np.random.normal(loc=v_sub, scale=np.sqrt(var_2), size=dim)

    return x_mean, v_mean
