import autograd.numpy as np
# from autograd import grad
# from autograd.numpy import log, sqrt, sin, cos, exp, pi, prod
# from autograd.numpy.random import normal, uniform

from model.tools import Helper

from model.utils import mixture, mixture_expand, function_plot, select_optimizer, select_domain, kl_divergence
from model.event import Event
from model.flags import get_flags

from tqdm import tqdm
import os


def run_func(flags):
    # if_bound = False
    if_bound = flags.if_include_domain

    num_points = flags.num_points
    lower, upper = flags.lower, flags.upper
    axis_x = np.linspace(lower, upper, num_points)
    axis_y = np.linspace(lower, upper, num_points)
    axis_X, axis_Y = np.meshgrid(axis_x, axis_y)
    path = flags.path

    # energy_grid = mixture_expand(axis_X, axis_Y)
    prob_grid = function_plot(axis_X, axis_Y)

    prob_grid_modified = np.hstack((axis_X.reshape(-1, 1), axis_Y.reshape(-1, 1)))

    # myHeart = Flower(petals=5, move_out=3)
    if flags.if_include_domain:
        myHelper = Helper(select_domain(flags.domain_type), max_radius=flags.radius, grid_radius=1e-2, grid_curve=1e-3)
        print('domain: ', flags.domain_type, '. seed: '+str(flags.seed))
    else:
        myHelper = None
        print('no constrained. seed: '+str(flags.seed))
        

    if if_bound:
        try:
            ground_truth = np.load('./logs/data/' + flags.domain_type + '_' + str(flags.num_points) + '.npy')
        except FileNotFoundError:
            ground_truth = np.zeros(prob_grid_modified.shape[0])
            pbar = tqdm(range(ground_truth.size), dynamic_ncols=True, smoothing=0.1, desc='Process Ground Truth')

            for i in pbar:
                ground_truth[i] = myHelper.inside_domain(prob_grid_modified[i])
                ground_truth[i] = prob_grid.reshape(-1)[i] * ground_truth[i]
            ground_truth = ground_truth.reshape(num_points, -1)
            np.save('./logs/data/' + flags.domain_type + '_' + str(flags.num_points) + '.npy', ground_truth)
    else:
        ground_truth = prob_grid

    # # plt.figure(figsize=(8, 8), dpi=300)
    # # sns.heatmap(prob_grid, cmap="Blues", cbar=False, xticklabels=False, yticklabels=False).invert_yaxis()
    # # plt.xlabel('x_sub', size=20)
    # # plt.ylabel('y', size=20)
    # # plt.grid(True)
    # # plt.show()
    kl_divergence(ground_truth, samples=None, num_grid=flags.num_points)
    # sampler = SGLD(
    #     f=mixture, dim=flags.dims, xinit=[0., 0.], lr=3e-3, T=1, decay_lr=3e-3)
    sampler = select_optimizer(flags, myHelper)

    my_images3 = []
    if not os.path.isdir(path):
        try:
            os.makedirs(path)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else:
                raise

    event = Event(sampler, path, [axis_X, axis_Y], ground_truth,
                  warm_up=flags.warm_up, if_record=True, if_gifs=flags.if_gifs, bound_help=myHelper, flags=flags)
    event.update()
    kl, prob_true, prob_empirical = kl_divergence(ground_truth, event.x_record, num_grid=num_points)

    print('Done!')

if __name__ == '__main__':
    # Get flags
    flags = get_flags()

    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)
    # np.random.seed(flags.seed)
    # run_func(flags)
        
    for i in range(1):
        flags.seed = flags.seed + 1
        np.random.seed(flags.seed)
        # # random.seed(flags.seed)
        # # np.random.seed(flags.seed)
        # # torch.manual_seed(flags.seed)
        # # torch.cuda.manual_seed(flags.seed)
    
        print(flags)
        
        run_func(flags)
