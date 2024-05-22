import autograd.numpy as np
from autograd import grad
from autograd.numpy import log, sqrt, sin, cos, exp, pi, prod
from autograd.numpy.random import normal, uniform
from model.optimizer import SGLD, reSGLD, cyclicSGLD
from model.domain import Star, Flower, Cross, Polygon, Heart
from scipy.integrate import dblquad
from scipy.special import rel_entr
from tqdm import tqdm
import numpy as np


class Helper:
    def __init__(self, myClass, max_radius=1, grid_radius=1e-2, grid_curve=1e-3):  # finer grid is much slower
        self.myClass = myClass(radius=max_radius)
        self.cached_points_list = []
        self.grid_radius = grid_radius
        self.max_radius = max_radius
        for radius in np.arange(0., max_radius, max_radius * grid_radius):
            curClass = myClass(radius=radius)
            candidate_points = curClass.position(np.arange(0, 1, grid_curve))
            self.cached_points_list.append(candidate_points)

    def inside_domain(self, test_point=np.array([1, -0.6])):
        test_point = test_point.reshape(1, -1, 1)
        min_rmse = np.min(np.sqrt(np.sum((self.cached_points_list - test_point) ** 2, axis=1)))
        return min_rmse < self.grid_radius * self.max_radius

    def binary_search_boundary(self, left, right):
        if not self.inside_domain(left):
            assert "left should be in the domain."
        if self.inside_domain(right):
            return right

        cnt = 0
        while not self.inside_domain(right) and cnt < 10:
            mid = (left + right) / 2
            if self.inside_domain(mid):
                left = mid.copy()
            else:
                right = mid.copy()
            cnt += 1
        return mid

    def get_reflection(self, left, right):
        boundary = self.binary_search_boundary(left, right)
        nu = right - boundary
        # compute unit normal vector
        grid_arrays = np.arange(0, 1, self.grid_radius)
        points = self.myClass.position(grid_arrays)
        idx = np.argmin(np.sum((points - boundary.reshape(-1, 1)) ** 2, axis=0))
        boundary_t = grid_arrays[idx]
        unit_normal = self.myClass.unit_normal(boundary_t)

        # http://www.sunshine2k.de/articles/coding/vectorreflection/vectorreflection.html
        reflected_nu = nu - 2 * np.inner(nu, unit_normal) * unit_normal
        reflection_points = boundary + reflected_nu
        return boundary + reflected_nu, boundary


def mixture(x):
    try:
        energy = ((x[0] ** 2 + x[1] ** 2) / 10 - (cos(2.0 * pi * x[0]) + cos(2.0 * pi * x[1]))) / 0.5  # 2
    except IndexError:
        x = x[0]
        energy = ((x[0] ** 2 + x[1] ** 2) / 10 - (cos(2.0 * pi * x[0]) + cos(2.0 * pi * x[1]))) / 0.5  # 2
        print('Index Errors')
    regularizer = ((x[0] ** 2 + x[1] ** 2) > 20) * ((x[0] ** 2 + x[1] ** 2) - 20)
    return energy + regularizer


def mixture_expand(x, y): return mixture([x, y])


def function_plot(x, y): return np.exp(-mixture([x, y]))


def select_optimizer(flags, myHelper):
    if flags.optimizer == 'sgd':
        sampler = SGD(
            f=mixture, dim=flags.dims, xinit=[0., 0.], lr=flags.lr,
            decay_lr=flags.decay_rate, l2=flags.regularization, myHelp=myHelper)
    if flags.optimizer == 'sgld':
        sampler = SGLD(
            f=mixture, dim=flags.dims, xinit=[0., 0.], lr=flags.lr, T=flags.T_low,
            decay_lr=flags.decay_rate, l2=flags.regularization, myHelp=myHelper)
    elif flags.optimizer == 'resgld':
        # need modification
        sampler = reSGLD(
            f=mixture, dim=flags.dims, xinit=[0., 0.], lr=flags.lr, lr_gap=flags.lr_gap, decay_lr=flags.decay_rate,
            l2=flags.regularization, myHelp=myHelper, flags=flags)
        print("Optimizer: reSGLD")
    elif flags.optimizer == 'cyclic_sgld':
        sampler = cyclicSGLD(
            f=mixture, dim=flags.dims, xinit=[0., 0.], lr=flags.lr, T=flags.T_low, total_epoch=flags.n_epoch, M=flags.M,
            l2=flags.regularization, myHelp=myHelper)
        print("Optimizer: cyclic SGLD")
    else:
        raise("Unable to identify the optimizer!")
    return sampler


def select_domain(domain_name):
    if domain_name == 'star':
        return Star
    elif domain_name == 'flower':
        return Flower
    elif domain_name == 'cross':
        return Cross
    elif domain_name == 'polygon':
        return Polygon
    elif domain_name == 'heart':
        return Heart
    else:
        raise("Unrecognizable domain, please enter it again.")


def kl_divergence(ground, samples, bound=[-2.5, 2.5], num_grid=100):
    grid_x = np.linspace(bound[0], bound[1], num_grid + 1)
    grid_y = np.linspace(bound[0], bound[1], num_grid + 1)
    grid_size = grid_x[1] - grid_x[0]

    # # Initialize a matrix to store the probability mass for each cell
    # # Define the region and discretization steps
    # x_min, x_max = bound[0], bound[1]
    # y_min, y_max = bound[0], bound[1]
    # n_bins = num_grid
    #
    # # Compute the width and height of each grid cell
    # dx = (x_max - x_min) / n_bins
    # dy = (y_max - y_min) / n_bins
    #
    # # Initialize the probability matrix
    # probability_matrix = np.zeros((n_bins, n_bins))
    #
    # # Iterate over each grid cell
    # for i in range(n_bins):
    #     for j in range(n_bins):
    #         # Find the center of the current grid cell
    #         x_center = x_min + (i + 0.5) * dx
    #         y_center = y_min + (j + 0.5) * dy
    #
    #         # Estimate the probability for the current grid cell
    #         # by evaluating the density function at the cell center
    #         # and multiplying by the cell's area.
    #         probability_matrix[i, j] = func(x_center, y_center) * dx * dy
    #
    #
    # # Normalize the probability masses so that they sum to 1
    # probability_density = probability_matrix / probability_matrix.sum()
    probability_density = ground / np.sum(ground)

    # Initialize a matrix to store the counts for each cell
    counts = np.zeros((len(grid_x) - 1, len(grid_y) - 1))

    # Count how many samples fall into each grid cell
    if samples is None:
        counts += 1
    else:
        for x, y in samples:
            if -2 <= x <= 2 and -2 <= y <= 2:  # Check if the sample is within our region of interest
                # Find the right cell for each sample
                i = np.digitize(x, grid_x) - 1  # np.digitize bins are 1-indexed
                j = np.digitize(y, grid_y) - 1
                if i < len(grid_x) - 1 and j < len(grid_y) - 1:  # Avoid incrementing the overflow bin
                    counts[j, i] += 1

    # Normalize the counts to get a probability density
    try:
        total_samples = samples.shape[0]
    except AttributeError:
        total_samples = int(np.sum(counts))

    # cell_area = (grid_x[1] - grid_x[0]) * (grid_y[1] - grid_y[0])
    empirical_density = counts / total_samples

    kl_div = np.sum(rel_entr(probability_density, empirical_density + np.finfo(float).eps)) * grid_size ** 2
    return kl_div, probability_density, empirical_density


def kl_plot(ground, samples, bound=[-2.5, 2.5], num_grid=100):
    grid_x = np.linspace(bound[0], bound[1], num_grid + 1)
    grid_y = np.linspace(bound[0], bound[1], num_grid + 1)
    grid_size = grid_x[1] - grid_x[0]

    probability_density = ground / np.sum(ground)

    # Initialize a matrix to store the counts for each cell
    counts = np.zeros((len(grid_x) - 1, len(grid_y) - 1))

    # Count how many samples fall into each grid cell
    if samples is None:
        counts += 1
    else:
        kl = []
        for x, y in samples:
            if -2 <= x <= 2 and -2 <= y <= 2:  # Check if the sample is within our region of interest
                # Find the right cell for each sample
                i = np.digitize(x, grid_x) - 1  # np.digitize bins are 1-indexed
                j = np.digitize(y, grid_y) - 1
                if i < len(grid_x) - 1 and j < len(grid_y) - 1:  # Avoid incrementing the overflow bin
                    counts[j, i] += 1
            empirical_density = counts / int(np.sum(counts))
            kl_div = np.sum(rel_entr(probability_density, empirical_density + np.finfo(float).eps)) * grid_size ** 2
            kl.append(kl_div)

    # Normalize the counts to get a probability density
    try:
        total_samples = samples.shape[0]
    except AttributeError:
        total_samples = int(np.sum(counts))
    cell_area = (grid_x[1] - grid_x[0]) * (grid_y[1] - grid_y[0])
    empirical_density = counts / total_samples

    try:
        kl_div = np.sum(rel_entr(probability_density, empirical_density + np.finfo(float).eps)) * grid_size ** 2
        return kl_div, probability_density, empirical_density
    except ValueError:
        return 0, probability_density, empirical_density
