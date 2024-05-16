import os

from tqdm import tqdm
import numpy as np
from model.utils import kl_divergence
from datetime import datetime
import matplotlib.pyplot as plt
import imageio
import seaborn as sns


class Event:
    def __init__(self, optimizer, path, grid, ground_trhth, warm_up=1000,
                 if_record=False, if_gifs=False, bound_help=None, flags=None, seed=None):

        self.sampler = optimizer
        self.path = path
        self.warm = warm_up
        if bound_help is not None:
            self.bound_help = bound_help
        else:
            self.bound_help = None

        self.grid_x = grid[0]
        self.grid_y = grid[1]
        self.ground_truth = ground_trhth
        self.num_points = ground_trhth.shape[0]
        if flags is not None:
            self.save_after = flags.save_after
            self.plot_after = flags.plot_after
            self.metric_after = flags.metric_after
            self.epoch = flags.n_epoch
        else:
            self.save_after = 10
            self.metric_after = 100
            self.plot_after = 1e3
            self.epoch = 1e5
        self.sampler_name = flags.optimizer
        if if_record:
            self.record = True
            self.x_record = np.array([self.sampler.x])
            self.kl = []
        else:
            self.record = False
        self.if_gifs = if_gifs
        if flags.if_save_metrics:
            self.if_save_metrics = True
            self.domain_type = flags.domain_type
            self.num_points = flags.num_points
        else:
            self.if_save_metrics = False

    def update_onestep(self, iters):
        try:
            self.sampler.update()
        except TypeError:
            self.sampler.update(iters)

    def update(self):

        my_images3 = []

        sgld_x = np.array([self.sampler.x])

        pbar = tqdm(range(int(self.epoch)), dynamic_ncols=True, smoothing=0.1, desc='Optimizer: '+self.sampler_name)
        for iters in pbar:
            self.update_onestep(iters)
            # self.sampler.x = proposal
            if self.record:
                self.x_record = np.vstack((self.x_record, self.sampler.x))
            if iters > self.warm:
                if iters % self.save_after == 0:
                    sgld_x = np.vstack((sgld_x, self.sampler.x))
                if self.if_gifs:
                    if iters % self.plot_after == 0:
                        image3 = self.plot(iters, sgld_x)
                        my_images3.append(image3)
                        plt.close('all')

                if iters % self.metric_after == 0:
                    self.kl.append(kl_divergence(self.ground_truth, sgld_x, num_grid=self.num_points)[0])

            if self.sampler_name == 'resgld':
                pbar.set_postfix({
                    'learn_rate': ' ({0:1.2e}'.format(self.sampler.lr[0])+', {0:1.2e}) '.format(self.sampler.lr[1]),
                    'swap rate': '{0:1.4f}'.format(
                        self.sampler.swap_total / (self.sampler.swap_total+self.sampler.no_swap_total) * 100)})
            elif self.sampler_name == 'cyclic_sgld':
                pbar.set_postfix({
                    'learn_rate': '{0:1.4e}'.format(self.sampler.lr),
                    'sample': ' ({0:1.2f}'.format(self.sampler.x[0])+', {0:1.2f}) '.format(self.sampler.x[1])})
            else:
                pbar.set_postfix({
                    'learn_rate': '{0:1.4e}'.format(self.sampler.lr),
                    'sample': '({0:1.2f}'.format(self.sampler.x[0])+', {0:1.2f}) '.format(self.sampler.x[1])})

                    

        if self.if_save_metrics:
            self.save_metrics(sgld_x)

        if self.if_gifs:
            self.plot_gifs(my_images3)

    def plot(self, iters, sgld_x):

        lower, upper = -2.5, 2.5

        fig3 = plt.figure(figsize=(4, 4))
        plt.yticks([-2, -1, 0, 1, 2])
        plt.scatter(sgld_x[:, 0], sgld_x[:, 1], marker='.', s=3, color='k', label="Iteration=" + str(iters), zorder=10)
        plt.contour(self.grid_x, self.grid_y, self.ground_truth, 100, cmap="Blues", zorder=1)
        plt.legend(loc="upper left", prop={'size': 10})
        plt.xlim([lower, upper])
        plt.ylim([lower, upper])
        plt.tight_layout()
        # plt.show()
        fig3.canvas.draw()
        image3 = np.frombuffer(
            fig3.canvas.tostring_rgb(), dtype='uint8').reshape(fig3.canvas.get_width_height()[::-1] + (3,))
        # my_images3.append(image3)

        return image3

    def plot_prob_empirical(self, samples):
        plt.figure(figsize=(8, 6))
        plt.imshow(kl_divergence(self.ground_truth, samples, num_grid=self.num_points)[2], extent=(-2.5, 2.5, -2.5, 2.5),
                   origin='lower', cmap='hot', interpolation='nearest')
        plt.colorbar(label='Density')
        plt.title('Heat Map of Empirical Density')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.show()

    def plot_prob_true(self):
        plt.figure(figsize=(8, 6))
        plt.imshow(self.ground_truth, extent=(-2.5, 2.5, -2.5, 2.5),
                   origin='lower', cmap='Blues', interpolation='nearest')
        # plt.colorbar(label='Density')
        # plt.title('Heat Map of Empirical Density')
        # plt.xlabel('X-axis')
        # plt.ylabel('Y-axis')
        plt.show()

    def plot_trace(self, samples):
        # Plot for the first parameter
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)  # Trace plot
        plt.plot(samples[:, 0], color='blue')
        plt.title('Trace and Density Plot for Parameter 1')
        plt.ylabel('Parameter 1 Value')

        plt.subplot(2, 1, 2)  # Histogram
        plt.hist(samples[:, 0], bins=200, alpha=0.6, color='blue', density=True)
        sns.kdeplot(samples[:, 0], color='black')
        plt.ylabel('Density')
        plt.xlabel('Parameter 1 Value')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        # Plot for the second parameter
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)  # Trace plot
        plt.plot(samples[:, 1], color='green')
        plt.title('Trace and Density Plot for Parameter 2')
        plt.ylabel('Parameter 2 Value')

        plt.subplot(2, 1, 2)  # Histogram
        plt.hist(samples[:, 1], bins=200, alpha=0.6, color='green', density=True)
        sns.kdeplot(samples[:, 1], color='black')
        plt.ylabel('Density')
        plt.xlabel('Parameter 2 Value')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def plot_density(self, sgld_x, num_points=200):
        lower, upper = self.grid_x[0, 0], self.grid_x[-1, -1]
        num_point = int(num_points)
        np.linspace(self.grid_x[0, 0], self.grid_x[-1, -1], num_point)
        axis_x = np.linspace(lower, upper, num_point)
        axis_y = np.linspace(lower, upper, num_point)
        axis_X, axis_Y = np.meshgrid(axis_x, axis_y)

        fig = plt.figure(figsize=(4, 4))
        plt.yticks([-2, -1, 0, 1, 2])
        density = kl_divergence(self.ground_truth, sgld_x, num_grid=num_point)[2]

        plt.imshow(density, extent=(lower, upper, lower, upper),
                   origin='lower', cmap='Blues', interpolation='nearest')
        # plt.contour(axis_X, axis_Y, density*1, 50, cmap="Blues", zorder=1)
        # plt.imshow(density, extent=(lower, upper, lower, upper),
        #                       cmap='Blues', interpolation='nearest')
        # plt.legend(loc="upper left", prop={'size': 10})
        plt.xlim([lower, upper])
        plt.ylim([lower, upper])
        plt.tight_layout()

        now = datetime.now()
        time = now.strftime("%Y_%m_%d_%H_%M_%S")
        # plt.savefig(self.path + self.sampler_name + '_contour_' + time + '.pdf')
        plt.show()

        # fig = plt.figure(figsize=(4, 4))
        # plt.hist2d(sgld_x[:, 0], sgld_x[:, 1], bins=(100, 100), cmap='Blues', extent=(lower, upper, lower, upper))
        # plt.xlim([lower, upper])
        # plt.ylim([lower, upper])
        # plt.show()


        # plt.figure(figsize=(8, 6))
        # density = kl_divergence(self.ground_truth, sgld_x, num_grid=self.num_points)[2]
        # plt.imshow(density, extent=(-2.5, 2.5, -2.5, 2.5),
        #            cmap='Blues', interpolation='nearest')
        # plt.colorbar(label='Density')
        # plt.title('Heat Map of Empirical Density')
        # plt.xlabel('X-axis')
        # plt.ylabel('Y-axis')
        # plt.show()

    def plot_gifs(self, images):

        now = datetime.now()
        time = now.strftime("%Y_%m_%d_%H_%M_%S")

        if self.sampler_name == 'resgld':
            imageio.mimsave(
                self.path + self.sampler_name + '_contour_' + str(self.sampler.lr[0]) + "_" + time + '.gif',
                images, fps=24, duration=0.5, loop=0)
        else:
            imageio.mimsave(
                self.path + self.sampler_name + '_contour_' + str(self.sampler.lr) + "_" + time + '.gif',
                images, fps=24, duration=0.5, loop=0)

    def save_metrics(self, samples):

        dict_ = {'sample': samples, 'kl_div': self.kl}

        now = datetime.now()
        time = now.strftime("%Y_%m_%d_%H_%M_%S")
        path = './logs/metrics/' + time + '_' + self.domain_type + '_' + self.sampler_name + '_' + str(self.num_points)
        os.mkdir(path)

        np.save(path + '/dict_metrics.npy', dict_)
        print("Save metrics: " + time + '_' + self.domain_type + '_' + self.sampler_name + '_' + str(self.num_points))
