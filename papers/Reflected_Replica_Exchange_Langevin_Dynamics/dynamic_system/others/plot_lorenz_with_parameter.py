import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
from pymc3.ode import DifferentialEquation
from scipy.integrate import odeint

from scipy.integrate import ode
# from scipy.optimize import curve_fit

plt.style.use("seaborn-darkgrid")


def scatter_plot_3d(data, show=True, cmap='RdYlBu',
                    labels=('x', 'y', 'z'),
                    figsize=(9, 7)):
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    cm = plt.cm.get_cmap(cmap)

    fig = plt.figure(figsize=figsize, dpi=500)
    ax = fig.add_subplot(111, projection='3d', facecolor='white')
    ax.set_facecolor('white')

    # ax.grid(False)
    plt.axis('off')
    plt.grid(b=None)

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
    plt.savefig('../logs/figs/lorenz_' + time + '.jpg')
    # plt.savefig('./figs/lorenz_' + time + '.pdf')

    if show:
        plt.show()

def dX_dt(t, state, par):
    """ Return the growth rate of fox and rabbit populations. """
    sigma, rho, beta = par
    return np.array([sigma * (state[1] - state[0]),
                     state[0] * (rho - state[2]) - state[1],
                     state[0] * state[1] - beta * state[2]])

# getting the data
# samples = np.load('../lorenz/lorenz_resgld_sample.npy')

samples = np.load('../lorenz/lorenz_cycsgld_sample.npy')
# samples = np.load('../lorenz/lorenz_sgld_sample.npy')

kde = sns.kdeplot((samples[1, 0] - samples[0, 0]) / 2, color='black')
lines = kde.get_lines()

for line in lines:
    x, y = line.get_data()
    sigma = x[np.argmax(y)]
    print('Posterior mode:', sigma)
    # ax.axvline(x[np.argmax(y)], ls='--', linewidth=2, color='black')

kde = sns.kdeplot(samples[0, 1] * 1, color='black')
lines = kde.get_lines()
for line in lines:
    x, y = line.get_data()
    rho = x[np.argmax(y)]
    print('Posterior mode:', rho)

kde = sns.kdeplot(samples[2, 2] * -1, color='black')
lines = kde.get_lines()
for line in lines:
    x, y = line.get_data()
    beta = x[np.argmax(y)]
    print('Posterior mode:', beta)


rho = 28
sigma = 10
beta = 8 / 3

t = np.linspace(0, 100, 100000)  # time
X0 = np.array([1, 1, 1])  # initials conditions: 10 rabbits and 5 foxes
r = ode(dX_dt).set_integrator('dopri5')
r.set_initial_value(X0, t[0])
r.set_f_params((sigma, rho, beta))
X = np.zeros((len(X0), len(t)))
X[:, 0] = X0
for i, _t in enumerate(t):
    if i == 0:
        continue
    r.integrate(_t)
    X[:, i] = r.y

x, y, z = X[0].reshape(-1, 1), X[1].reshape(-1, 1), X[2].reshape(-1, 1)

scatter_plot_3d(np.hstack((x, y, z))[::1])

