import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_sample_path(sol, figsize=(18, 2)):
    if torch.is_tensor(sol):
        sol = sol.detach().cpu()
    fig, axs = plt.subplots(ncols=sol.shape[0], figsize=figsize)

    for i in range(sol.shape[0]):
        axs[i].scatter(sol[i, :, 0], sol[i, :, 1], c='k', s=0.8)
        # axs[i].axis('off')

    return fig, axs


def plot_displacement(sol, figsize=(6, 6)):
    if torch.is_tensor(sol):
        sol = sol.detach().cpu()
    # sol = sol[:, :1000, :]
    fig, ax = plt.subplots(figsize=figsize)
    for i in range(sol.shape[1]):
        ax.plot(sol[:, i, 0], sol[:, i, 1], c='olive', alpha=0.1)
    ax.scatter(sol[0, :, 0], sol[0, :, 1], s=5, c='black', alpha=1.,linewidths=0.5)
    ax.scatter(sol[-1, :, 0], sol[-1, :, 1], s=5, c='blue', alpha=1., linewidths=0.5)
    ax.axis('off')
    return fig, ax


