import torch
from torch import nn, Tensor
from sde import compute_vp_diffusion

def build_linear(zero_out_last_layer):
    return LinearPolicy(zero_out_last_layer=zero_out_last_layer)


class LinearPolicy(nn.Module):
    def __init__(
        self,
        data_dim,
        beta_min: float,
        beta_max: float,
        beta_r: float,
        interval: int,
    ):
        super(LinearPolicy,self).__init__()

        self.beta_min = beta_min
        self.beta_max = beta_max
        self.beta_r = beta_r
        self.interval = interval

        self.A = nn.Parameter(torch.zeros(data_dim, data_dim))

        self.Sigma = nn.Parameter(torch.zeros(data_dim))
        self.U = nn.utils.parametrizations.orthogonal(nn.Linear(data_dim, data_dim, bias=False))
        self.V = nn.utils.parametrizations.orthogonal(nn.Linear(data_dim, data_dim, bias=False))
        self.U.weight = torch.eye(data_dim)
        self.V.weight = torch.eye(data_dim)

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype

    @property
    def A(self):
        Sigma_mat = torch.diag(self.Sigma)
        return self.U.weight @ Sigma_mat @ self.V.weight.T

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        **kwargs,
    ) -> Tensor:
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        """
        # make sure t.shape = [T]
        if len(t.shape)==0:
            t=t[None]

        out = x @ self.A.T

        diffusion = compute_vp_diffusion(
            t,
            b_min=self.beta_min,
            b_max=self.beta_max,
            b_r=self.beta_r,
            T=self.interval,
        ).view(-1, *[1]*(out.ndim - 1))

        out = out * diffusion
        return out
