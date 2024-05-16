from matplotlib import pyplot as plt
from datetime import datetime
import numpy as np
import os


class LoadData:
    def __init__(self, path='./data/'):

        self.reward_sample = []
        self.reward_mean = []
        self.reward_std = []
        dir_list = os.listdir(path)
        dir_list = sorted(dir_list)
        for dirs in dir_list:
            reward = np.loadtxt(os.path.join(path, dirs), dtype=float)
            self.reward_sample.append(reward)
            self.reward_mean.append(np.mean(reward))
            self.reward_std.append(np.std(reward))


def plot_figs(regret_logs, regret_logs_2, args):
    fig, ax = plt.subplots(dpi=150)
    ax.plot(np.hstack(regret_logs), linewidth=3, label='Overdamped')
    ax.plot(np.hstack(regret_logs_2), linewidth=3, label='Underdamped')
    ax.set_xlabel('Rounds', fontsize=15)
    ax.set_ylabel('Total Expected Regrets', fontsize=15)
    # ax.set_title("Steps v.s. KSD", fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    ax.legend(fontsize=15)
    plt.tight_layout()
    now = datetime.now()
    time = now.strftime("%Y_%m_%d_%H_%M_%S")
    str_seed = '_seed_' + str(int(args.seed))
    str_dim = '_dim_' + str(int(args.dim))
    str_arm = '_arm_' + str(int(args.n_arm))
    str_round = '_round_' + str(int(args.n_round))
    str_iter = '_iter_' + str(int(args.n_iter))
    str_batch = '_batch_' + str(int(args.batch_size))
    fig_info = time + str_seed + str_dim + str_arm + str_round + str_iter + str_batch
    plt.savefig('./logs/figs/regret_' + fig_info + '.pdf')
    plt.show()
