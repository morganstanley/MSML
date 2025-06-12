
import torch
from models.utils import *

from ipdb import set_trace as debug

from sde import compute_vp_diffusion


def build_linear_img(config, zero_out_last_layer):
    out_channel=config.out_channel
    image_size = config.image_size

    return LinearImgModel(
        out_channels=out_channel,
        image_size=image_size
    )

class LinearImgModel(nn.Module):
    def __init__(
        self,
        out_channels,
        image_size,
        num_classes=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.AA = nn.Parameter(torch.zeros([out_channels, image_size, image_size]))

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype

    def forward(self, x, timesteps, y=None, beta_min=0.1, beta_max=10., beta_r=1., interval=100.):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        x = torch.einsum('cij,bcij->bcij', self.AA, x)
        x = compute_vp_diffusion(timesteps, b_min=beta_min, b_max=beta_max, b_r=beta_r, T=interval)[:, None, None, None] * x

        return x
