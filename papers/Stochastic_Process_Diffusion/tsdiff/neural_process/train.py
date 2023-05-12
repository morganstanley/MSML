import pickle
import numpy as np
from functools import partial
from scipy.stats import multivariate_normal

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from gluonts.evaluation.metrics import mse, quantile_loss

from tsdiff.diffusion import OUDiffusion, GPDiffusion, BetaLinear
from tsdiff.utils import PositionalEncoding

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)

def quantile_loss(target, forecast, q):
    """ Adapted from gluonts """
    return 2 * np.sum(np.abs((forecast - target) * ((target <= forecast) - q)))

def q_mean_loss(target, forecast):
    q_loss = []
    for q in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        forecast_quantile = np.quantile(forecast, q)
        q_loss.append(quantile_loss(target, forecast_quantile, q))
    return np.mean(q_loss) / np.abs(target).sum()

def radial_basis_kernel(x, y, sigma):
    dist = (y[None,:] - x[:,None])**2
    return np.exp(-dist / sigma)

def generate_data(
    N,
    train_ratio=0.5,
    mean_num_points=8,
    min_num_points=5,
    max_num_points=50,
    min_t=0,
    max_t=1,
    sigma=0.05,
    seed=1,
):
    np.random.seed(seed)

    lengths = np.random.poisson(mean_num_points, size=(N,))
    lengths = np.clip(lengths, min_num_points, max_num_points)

    x_time, x_values = [], []
    y_time, y_values = [], []

    q_loss = []
    mse_error = []

    for n in lengths:
        t = np.random.uniform(min_t, max_t, size=n)

        cov = radial_basis_kernel(t, t, sigma) + 1e-4 * np.eye(n) # Add to diagonal for stability
        L = np.linalg.cholesky(cov)

        points = L @ np.random.normal(size=(n,))

        i = int(n * train_ratio)
        i = np.clip(i, 1, n - 1)

        x_time.append(t[:i])
        y_time.append(t[i:])
        x_values.append(points[:i])
        y_values.append(points[i:])

        # Evaluate the quantile loss
        t_ = t[i:]
        t = t[:i]

        kxx = radial_basis_kernel(t, t, sigma)
        kxx = kxx + 1e-2 * np.eye(len(kxx))
        kxx_inv = np.linalg.inv(kxx)
        kyx = radial_basis_kernel(t_, t, sigma)
        kyy = radial_basis_kernel(t_, t_, sigma)

        mean = kyx @ kxx_inv @ points[:i]
        cov = kyy - kyx @ kxx_inv @ kyx.T
        # cov += 1e-3 * np.eye(len(cov))

        dist = multivariate_normal(mean, cov)

        sample = dist.rvs(100)

        q_loss.append(q_mean_loss(points[i:], sample))
        mse_error.append(mse(points[i:], sample.mean(0)))

    q_loss = np.mean(q_loss)
    mse_error = np.mean(mse_error)
    return x_values, x_time, y_values, y_time, q_loss, mse_error


class NumpyDataset(Dataset):
    def __init__(self, x, tx, y, ty):
        super().__init__()
        assert len(x) == len(tx) == len(y) == len(ty)
        assert [len(a) == len(b) for a, b in zip(x, tx)]
        assert [len(a) == len(b) for a, b in zip(y, ty)]

        self.x = x
        self.tx = tx
        self.y = y
        self.ty = ty

    def __getitem__(self, ind):
        return self.x[ind], self.tx[ind], self.y[ind], self.ty[ind]

    def __len__(self):
        return len(self.x)

def collate_fn(batch, device):
    x, tx, y, ty = list(zip(*batch))
    max_x_len = max(map(len, x))
    max_y_len = max(map(len, y))

    def get_mask(arr, max_len):
        mask = np.array([np.concatenate([np.ones(len(x)), np.zeros(max_len - len(x))]) for x in arr])
        return (1 - torch.Tensor(mask)).bool().to(device)
    x_pad = get_mask(x, max_x_len)
    y_pad = get_mask(y, max_y_len)

    pad = lambda arr, max_len: torch.Tensor(np.array([np.pad(s, (0, max_len - len(s))) for s in arr])).unsqueeze(-1).to(device)
    x = pad(x, max_x_len)
    y = pad(y, max_y_len)
    tx = pad(tx, max_x_len)
    ty = pad(ty, max_y_len)

    return x, y, tx, ty, x_pad, y_pad

class Denoiser(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.i_emb = PositionalEncoding(hidden_dim, max_value=100)
        self.t_emb = PositionalEncoding(hidden_dim, max_value=1)

        self.linear1 = nn.Linear(dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.i_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.z_proj = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim),
        )

        self.sigma = nn.Parameter(torch.randn(1))

    def forward(self, y, *, t, i, z, tz, z_pad):
        k = torch.exp(-torch.square(t - tz.transpose(-1, -2)) / self.sigma**2)

        z = self.z_proj(z)
        z = z * (1 - z_pad.float().unsqueeze(-1))

        z = k @ z

        i = self.i_emb(i)
        i = self.i_proj(i)

        y = self.linear1(y) + z
        y = torch.relu(y)

        y = self.linear2(y) + i
        y = torch.relu(y)

        y = self.net(y)
        return y

class NeuralDenoisingProcess(nn.Module):
    def __init__(self, dim, hidden_dim, gp, param):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim

        if gp:
            self.diffusion = GPDiffusion(dim, beta_fn=BetaLinear(1e-4, 0.2), num_steps=100, sigma=param)
        else:
            self.diffusion = OUDiffusion(dim, beta_fn=BetaLinear(1e-4, 0.2), num_steps=100, theta=param)
        self.denoise_fn = Denoiser(dim, hidden_dim)

    def forward(self, x, y, tx, ty, x_pad, y_pad):
        y = torch.cat([x, y], 1)
        ty = torch.cat([tx, ty], 1)
        y_pad = torch.cat([x_pad, y_pad], 1)

        loss = self.diffusion.get_loss(self.denoise_fn, y, t=ty, tz=tx, z=x, z_pad=x_pad)
        loss = loss.mean(-1) * (1 - y_pad.float())
        return loss.mean()

    def sample(self, x, tx, x_pad, ty, num_samples=1):
        x = x.repeat_interleave(num_samples, dim=0)
        tx = tx.repeat_interleave(num_samples, dim=0)
        x_pad = x_pad.repeat_interleave(num_samples, dim=0)
        ty = ty.repeat_interleave(num_samples, dim=0)
        return self.diffusion.sample(self.denoise_fn, num_samples=ty.shape[:-1], device=tx.device,
                                     t=ty, tz=tx, z=x, z_pad=x_pad)

def evaluate(model, testloader):
    num_samples = 100

    targets, forecasts = [], []
    mse_err = []
    q_loss = []

    for _, batch in enumerate(testloader):
        x, y, tx, ty, x_pad, y_pad = batch

        y_pred = model.sample(x, tx, x_pad, ty, num_samples=num_samples)

        target, forecast = y.cpu().squeeze().numpy(), y_pred.cpu().squeeze().numpy()

        targets.append(target)
        forecasts.append(forecast)

        assert target.shape == forecast.mean(0).shape

        q_loss.append(q_mean_loss(target, forecast))
        mse_err.append(mse(target, forecast.mean(0)))

    return np.mean(q_loss), np.mean(mse_err)

def sample(model, testloader, gp, param):
    num_samples = 30
    T = 50

    loader_it = iter(testloader)

    samples = []
    xs = []
    txs = []
    ys = []
    tys = []

    for _ in range(10):
        x, y, tx, ty, x_pad, y_pad = next(loader_it)
        t = torch.linspace(0, 1, T).view(1, -1, 1).to(x)
        y_samples = model.sample(x, tx, x_pad, t, num_samples=num_samples)

        samples.append(y_samples.detach().cpu().numpy())
        xs.append(x.detach().cpu().numpy())
        txs.append(tx.detach().cpu().numpy())
        ys.append(y.detach().cpu().numpy())
        tys.append(ty.detach().cpu().numpy())

    root = '/tsdiff/experiments/neural_process/samples'
    filename = f'{root}/{"gp" if gp else "ou"}-{param}.pkl'

    with open(filename, 'wb') as f:
        data = dict(sample=samples, x=xs, tx=txs, y=ys, ty=tys)
        pickle.dump(data, f)

def train(
    seed: int,
    gp: bool,
    param: float,
    epochs: int = 200,
    batch_size: int = 128,
    train_size: int = 800,
    test_size: int = 200,
    hidden_dim: int = 32,
):
    # Generate data
    x_values, x_time, y_values, y_time, train_q_loss, train_mse_error = generate_data(train_size, seed=seed)
    trainset = NumpyDataset(x_values, x_time, y_values, y_time)
    trainloader = DataLoader(trainset, batch_size=batch_size, collate_fn=partial(collate_fn, device=device), shuffle=True)

    x_values, x_time, y_values, y_time, test_q_loss, test_mse_error = generate_data(test_size, seed=seed + 10)
    testset = NumpyDataset(x_values, x_time, y_values, y_time)
    testloader = DataLoader(testset, batch_size=1, collate_fn=partial(collate_fn, device=device), shuffle=False)

    # Make model
    model = NeuralDenoisingProcess(
        dim=1,
        hidden_dim=hidden_dim,
        gp=gp,
        param=param,
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), weight_decay=1e-5)

    # Train
    for epoch in range(epochs):
        for batch in trainloader:
            optim.zero_grad()
            loss = model(*batch)
            loss.backward()
            optim.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, loss: {loss:.4f}')

    # Evaluate
    q_loss, mse_error = evaluate(model, testloader)

    if seed == 1:
        sample(model, testloader, gp, param)

    result = dict(
        q_loss=q_loss,
        mse_error=mse_error,
        train_q_loss=train_q_loss,
        train_mse_error=train_mse_error,
        test_q_loss=test_q_loss,
        test_mse_error=test_mse_error,
    )

    return model, result

if __name__ == '__main__':
    model, result = train(seed=1, param=0.02, epochs=20, gp=True)
    print(result)
