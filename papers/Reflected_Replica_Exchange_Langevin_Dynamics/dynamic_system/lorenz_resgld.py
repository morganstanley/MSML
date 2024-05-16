from model.utils import plot_trace
from model.flags import get_flags
from model.sampler import reSGLD
from tqdm import tqdm
import numpy as np
import scipy.io


if __name__ == '__main__':

    data = scipy.io.loadmat('./data/lorenz_100.mat')
    sigma = data['sigma']
    beta = data['beta']
    rho = data['rho']

    dxdt = data['dx']
    x = data['x'][:, 0:1]
    y = data['x'][:, 1:2]
    z = data['x'][:, 2:]
    x0 = data['x0']
    t = data['t']

    args = get_flags()
    args.hat_var = 9
    seed = args.seed
    np.random.seed(seed)

    Xi_correct = np.array([[-10, 28, 0], [10, -1, 0], [0, 0, -8/3], [0, 0, 1], [0, -1, 0]])
    Theta = np.hstack([x, y, z, x*y, x*z])

    dims = [Theta.shape[1], dxdt.shape[1]]
    Xi = np.ones(dims) * 0
    m = Theta.shape[0]
    epoch = int(args.n_epoch)
    batch_size = int(args.batch)
    lr = args.lr

    sampler = reSGLD(
        dim=dims, xinit=Xi, batch_size=batch_size, lr=3e-6, n_chain=2, flags=args)
    epoch_run = tqdm(range(epoch), dynamic_ncols=True, smoothing=0.1, desc='Start Training: ')
    sample_record = []

    for i in epoch_run:
        idx_random = np.random.choice(range(m), size=m, replace=False)
        num_iter = np.floor(m / batch_size).astype(int) + 1
        for j in range(num_iter):
            idx = idx_random[j * batch_size:(j + 1) * batch_size]
            Theta_batch = Theta[idx]
            dxdt_batch = dxdt[idx]
            sampler.update(Theta_batch, dxdt_batch)

        epoch_run.set_postfix({
            'loss': '{0:1.4e}'.format(np.sqrt(np.mean((np.dot(Theta_batch, sampler.x) - dxdt_batch) ** 2))),
            'swap rate': '{0:1.4f}'.format(
                sampler.swap_total / (sampler.swap_total + sampler.no_swap_total) * 100),
            'learn rate': '{0:1.4e}'.format(sampler.lr[0])})
        if i >= 1e4:
            for j in range(len(sampler.lr)):
                sampler.lr[j] *= args.decay_rate
            if i >= 2e4:
                sample_record.append(np.expand_dims(sampler.x_sub[0], axis=2))

    # sampler.x = np.where(np.abs(sampler.x) < 0.08, 0, sampler.x)
    # sampler.x[0, 2], sampler.x[1, 2] = 0, 0
    # samples = np.concatenate(sample_record, axis=-1)
    # sigma_1 = samples[0, 0]
    # sigma_2 = samples[1, 0]
    # rho = samples[0, 1]
    # beta = samples[2, 2]
    #
    # plot_trace(sigma_1)
    # plot_trace(sigma_2)
    # plot_trace(rho)
    # plot_trace(-1*beta)
    #
    # print(sampler.x)
    # print(Xi_correct)
    print('Done')

