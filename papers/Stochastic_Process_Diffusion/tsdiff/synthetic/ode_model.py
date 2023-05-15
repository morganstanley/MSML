import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
from pytorch_lightning import LightningModule
from torchdiffeq import odeint_adjoint as odeint



class ODEFunc(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

    def forward(self, t, state):
        y, diff = state
        y = torch.cat([t * diff, y], -1)
        dy = self.net(y) * diff
        return dy, torch.zeros_like(diff).to(dy)


class Model(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.enc_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GRU(hidden_dim, 2 * hidden_dim, num_layers=2, batch_first=True),
        )
        self.net = ODEFunc(dim, hidden_dim)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2 * dim),
        )

    def encoder(self, x):
        _, h = self.enc_net(x)
        h = h[-1].unsqueeze(-2).repeat_interleave(x.shape[-2], dim=-2)
        mu, sigma = h.chunk(2, dim=-1)
        sigma = F.softplus(0.1 + 0.9 * sigma)
        return mu, sigma

    def decoder(self, z, t):
        z = odeint(self.net, (z, t), torch.Tensor([0, 1]).to(t))
        z = z[0][1] # first state, solution at t=1 (with reparam.)
        x_mu, x_sigma = self.proj(z).chunk(2, dim=-1)
        x_sigma = F.softplus(0.1 + 0.9 * x_sigma)
        return x_mu, x_sigma


class ODEModule(LightningModule):
    def __init__(
        self,
        # Data params
        dim: int,
        data_mean: torch.Tensor = None,
        data_std: torch.Tensor = None,
        max_t: float = None,
        # NN params
        hidden_dim: int = None,
        # Training params
        learning_rate: float = None,
        weight_decay: float = None,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.data_mean = data_mean
        self.data_std = data_std
        self.max_t = max_t

        self.save_hyperparameters()

        self.model = Model(dim, hidden_dim)

    def get_loss(self, x, t):
        mu, sigma = self.model.encoder(x.flip(dims=[-2]))
        z = torch.randn_like(mu) * sigma + mu

        x_mu, x_sigma = self.model.decoder(z, t)

        px = td.Normal(x_mu, x_sigma)
        pz = td.Normal(mu, sigma)
        qz = td.Normal(torch.zeros_like(mu), torch.ones_like(sigma))

        kl = td.kl_divergence(pz, qz)

        # loss = -px.log_prob(x) + kl.sum(-1, keepdim=True)
        loss = (x - x_mu)**2 + kl.sum(-1, keepdim=True)
        return loss

    def forward(self, batch, log_name=None):
        x, t = self._normalize_batch(batch)
        loss = self.get_loss(x, t).mean()
        if log_name is not None:
            self.log(log_name, loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self.forward(batch, 'train_loss')

    def validation_step(self, batch, batch_idx):
        return self.forward(batch, 'val_loss')

    def test_step(self, batch, batch_idx):
        x, t = self._normalize_batch(batch)
        log_prob = -self.get_loss(x, t) / x.shape[-2]
        log_prob = log_prob - self.data_std.log().sum()
        self.log('test_log_prob', log_prob.mean())

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    @torch.no_grad()
    def sample(self, t, **kwargs):
        t = t / self.max_t
        z = torch.randn(*t.shape[:-2], 1, self.hidden_dim).to(t)
        z = z.repeat_interleave(t.shape[-2], dim=-2)
        samples, _ = self.model.decoder(z, t)
        return samples * self.data_std.to(t) + self.data_mean.to(t)

    def _normalize_batch(self, batch):
        x, t = batch
        x = (x - self.data_mean.to(x)) / self.data_std.to(x)
        t = t / self.max_t
        return x, t
