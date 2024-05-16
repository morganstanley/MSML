# getting necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.colors import ListedColormap
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})


# getting the data
# samples = np.load('../lorenz/lorenz_ref_resgld_sample_20240104.npy')
samples = np.load('./lorenz_resgld_sample.npy')
# samples = np.load('../lorenz/lorenz_cycsgld_sample.npy')
# samples = np.load('../lorenz/lorenz_sgld_sample.npy')
sigma = ((samples[1, 0] - samples[0, 0]) / 2 - 10) / 10
rho = (samples[0, 1] - 28) / 28
beta = (samples[2, 2] + 8 / 3) * 3 / 8

# sigma = np.random.normal(loc=0.0, scale=0.15, size=10000)
# rho = np.random.normal(loc=0.02, scale=0.10, size=10000)
# beta = np.random.normal(loc=0.05, scale=0.12, size=10000)

kde = sns.kdeplot((samples[1, 0] - samples[0, 0]) / 2, color='black')
lines = kde.get_lines()

for line in lines:
    x, y = line.get_data()
    sigma_mean = x[np.argmax(y)]
    print('Posterior mode:', sigma_mean)
    # ax.axvline(x[np.argmax(y)], ls='--', linewidth=2, color='black')

kde = sns.kdeplot(samples[0, 1] * 1, color='black')
lines = kde.get_lines()
for line in lines:
    x, y = line.get_data()
    rho_mean = x[np.argmax(y)]
    print('Posterior mode:', rho_mean)

kde = sns.kdeplot(samples[2, 2] * -1, color='black')
lines = kde.get_lines()
for line in lines:
    x, y = line.get_data()
    beta_mean = x[np.argmax(y)]
    print('Posterior mode:', beta_mean)

sample_list = []
sample_data = []
sample_mean = []
for i in range(sigma.size):
    sample_list.append('sigma')
    sample_data.append(sigma[i])
    sample_mean.append(sigma_mean)

for i in range(rho.size):
    sample_list.append('rho')
    sample_data.append(rho[i])
    sample_mean.append(rho_mean)

for i in range(beta.size):
    sample_list.append('beta')
    sample_data.append(beta[i])
    sample_mean.append(beta_mean)

sample_list = pd.Series(sample_list, index=np.arange(len(sample_list)))
sample_data = pd.Series(sample_data, index=np.arange(len(sample_data)))
sample_mean = pd.Series(sample_mean, index=np.arange(len(sample_mean)))

temps = {'parameter': sample_list, 'mean_parameter': sample_mean, 'Mean_lorenz': sample_data}
temps = pd.DataFrame.from_dict(temps)


# cmaps = ListedColormap(['blue', 'red', 'green'])
# we generate a color palette with Seaborn.color_palette()
pal = sns.color_palette(['limegreen', 'royalblue', 'red'])
# pal = sns.color_palette(['#A6BAAF', '#A7BEC6', '#E7ADAC'])

sample_list = pd.Series(sample_list, index=np.arange(len(sample_list)))
sample_data = pd.Series(sample_data, index=np.arange(len(sample_data)))
sample_mean = pd.Series(sample_mean, index=np.arange(len(sample_mean)))

temps = {'parameter': sample_list, 'mean_parameter': sample_mean, 'Mean_lorenz': sample_data}
temps = pd.DataFrame.from_dict(temps)

# we define a dictionnary with months that we'll use later
month_dict = {1: '$\\hat\\sigma=$'+str(round(sigma_mean, 2)),
              2: '$\\hat\\rho=$'+str(round(rho_mean, 2)),
              3: '$\\hat\\beta=$'+str(round(1*beta_mean, 2))
              }
# month_dict = {1: '$\\hat\\sigma=$'+str(round(10.0, 2)),
#               2: '$\\hat\\rho=$'+str(round(28.0, 2)),
#               3: '$\\hat\\beta=$'+str(round(1*8/3, 2))}

# in the sns.FacetGrid class, the 'hue' argument is the one that is the one that will be represented by colors with 'palette'
g = sns.FacetGrid(temps, row='parameter', hue='mean_parameter',
                  aspect=2.5, height=1, palette=pal, gridspec_kws={"hspace": 0.0})

# then we add the densities kdeplots for each month
g.map(sns.kdeplot, 'Mean_lorenz',
      bw_adjust=1, clip_on=False,
      fill=True, alpha=.8, linewidth=0.0)
g.map(sns.kdeplot, 'Mean_lorenz',
      bw_adjust=1, clip_on=False,
      color="k", lw=1)

# we use matplotlib.Figure.subplots_adjust() function to get the subplots to overlap
# g.fig.subplots_adjust(hspace=-0.3)
# # here we add a white line that represents the contour of each kdeplot
# g.map(sns.kdeplot, 'Mean_lorenz',
#       bw_adjust=1, clip_on=False,
#       color="w", lw=2.5)

# here we add a horizontal line for each plot
g.map(plt.axhline, y=0.0, lw=1., color="k", clip_on=False)


# we loop over the FacetGrid figure axes (g.axes.flat) and add the month as text with the right color
# notice how ax.lines[-1].get_color() enables you to access the last line's color in each matplotlib.Axes
for i, ax in enumerate(g.axes.flat):
    ax.text(-.9, 6.5, month_dict[i + 1], fontsize=12,
            color=ax.lines[-1].get_color())
    # ax.text(-1.1, 0.5, month_dict[i + 1], fontsize=15,
    #         color=ax.lines[-1].get_color())
    ax.axvline(0, ls='--', linewidth=.5, color='black')

# eventually we remove axes titles, yticks and spines
g.set_titles("")
# g.set(xticks=[], yticks=[], xlabel=None, ylabel=None)
g.set(xticks=np.arange(-.5, .6, 0.5), yticks=[], xlabel=None, ylabel=None, xlim=(-1, 1))
g.despine(bottom=True, left=True)
# g.set_xticks(range(-1, 1))

# plt.setp(ax.get_xticklabels(), fontsize=10, fontweight='bold')
# # plt.xlabel('Temperature in degree Celsius', fontweight='bold', fontsize=15)
# # g.fig.suptitle('Daily average temperature in Seattle per month',
# #                ha='right',
# #                fontsize=20,
# #                fontweight=20)

plt.tight_layout()
# g.map(sns.kdeplot, 'Mean_lorenz')
now = datetime.now()
time = now.strftime("%Y_%m_%d_%H_%M_%S")
plt.savefig('../logs/figs/uq_parameter_' + time + '.pdf')
plt.show()
