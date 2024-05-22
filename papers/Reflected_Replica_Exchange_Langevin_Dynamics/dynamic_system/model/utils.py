import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np


def scatter_plot_3d(data, show=True, cmap='RdYlBu',
                    labels=('x', 'y', 'z'),
                    figsize=(9, 7)):
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    cm = plt.cm.get_cmap(cmap)

    fig = plt.figure(figsize=figsize, dpi=300)
    ax = fig.add_subplot(111, projection='3d')

    t = range(len(data))
    cbar = ax.scatter(x, y, z, c=t, marker='.', cmap=cm)
    # cbar = plt.colorbar(cbar)
    # cbar.set_label('Time (t)')

    # ax.set_xlabel()
    # ax.set_ylabel()
    # ax.set_zlabel()

    plt.tight_layout()

    now = datetime.now()
    time = now.strftime("%Y_%m_%d_%H_%M_%S")
    plt.savefig('./figs/lorenz_' + time + '.jpg')
    # plt.savefig('./figs/lorenz_' + time + '.pdf')

    if show:
        plt.show()


def plot_trace(samples, n_bins=100):
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)  # Trace plot
    plt.plot(samples, color='blue')
    plt.title('Trace and Density Plot for Parameter 1')
    plt.ylabel('Parameter 1 Value')
    ax = plt.subplot(2, 1, 2)  # Histogram
    plt.hist(samples, bins=n_bins, alpha=0.6, color='blue', density=True)
    kde = sns.kdeplot(samples, color='black')
    plt.ylabel('Density')
    plt.xlabel('Parameter 1 Value')
    plt.grid(True)
    plt.tight_layout()
    # my_kde = kdeplot(data=df, x='x', hue='type', ax=ax)
    lines = kde.get_lines()

    for line in lines:
        x, y = line.get_data()
        print('Posterior mode:', x[np.argmax(y)])
        ax.axvline(x[np.argmax(y)], ls='--', linewidth=2, color='black')

    plt.show()


class Helper:
    def __init__(self, myClass=None, max_radius=1, grid_radius=1e-2, grid_curve=1e-3):  # finer grid is much slower
        # self.myClass = myClass(radius=max_radius)
        # self.cached_points_list = []
        # self.grid_radius = grid_radius
        # self.max_radius = max_radius
        self.correct_beta = np.array([[-10, 28, 0], [10, -1, 0], [0, 0, -8/3], [0, 0, 1], [0, -1, 0]])

    def inside_domain(self):
        return False

    def get_reflection(self, beta_current):

        sigma = (beta_current[1, 0] - beta_current[0, 0]) / 2
        rho = beta_current[0, 1]
        beta = -1 * beta_current[2, 2]

        if sigma < 0:
            sigma = -1 * sigma
        if beta < 0:
            beta = -1 * beta

        if sigma <= beta + 1:
            sub = sigma
            if sigma >= 1:
                sigma = beta + 1
                beta = sub - 1
            elif beta >= sigma + 1:
                sigma = beta + 1
                beta = sub
            else:
                sigma = 2 - sub
                beta = beta  # reflect along sigma=1 and then reflect along sigma=beta+1

        if rho < 1:
            rho = -1 * rho + 2
        elif rho == 1:
            rho += 0.01
        beta_current = self.correct_beta * 1
        beta_current[0, 0] = -1 * sigma
        beta_current[1, 0] = 1 * sigma
        beta_current[0, 1] = 1 * rho
        beta_current[2, 2] = -1 * beta

        return beta_current

