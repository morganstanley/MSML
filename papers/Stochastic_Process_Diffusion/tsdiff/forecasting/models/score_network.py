from torchtyping import TensorType

import torch
import torch.nn as nn
import torch.nn.functional as F

from pts.model import weighted_average
from pts.modules import MeanScaler

from tsdiff.diffusion import OUDiffusion, BetaLinear


class ScoreTrainingNetwork(nn.Module):
    """
    Score training network.

    Args:
        context_length: Size of history
        prediction_length: Size of prediction
        target_dim: Dimension of data
        time_feat_dim: Dimension of covariates
        conditioning_length: Hidden dimension
        beta_end: Final diffusion scale
        diff_steps: Number of diffusion steps
        residual_layers: Number of residual layers
        residual_channels: Number of residual channels
        dilation_cycle_length: Dilation cycle length
    """
    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        target_dim: int,
        time_feat_dim: int,
        conditioning_length: int,
        beta_end: float,
        diff_steps: int,
        residual_layers: int,
        residual_channels: int,
        dilation_cycle_length: int,
        **kwargs,
    ):
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length

        # hidden_dim = conditioning_length
        # self.context_rnn = nn.GRU(target_dim + time_feat_dim, hidden_dim, num_layers=2, bidirectional=True, batch_first=True)

        self.diffusion = OUDiffusion(target_dim, BetaLinear(1e-4, beta_end), diff_steps)
        self.denoise_fn = DenoisingModel(
            dim=target_dim + time_feat_dim,
            residual_channels=residual_channels,
            latent_dim=conditioning_length,
            residual_hidden=conditioning_length,
        )

        self.scaler = MeanScaler(keepdim=True)

    def forward(
        self,
        target_dimension_indicator: TensorType['batch', 'dim'],
        past_time_feat:             TensorType['batch', 'history_length', 'feat_dim'],
        past_target_cdf:            TensorType['batch', 'history_length', 'dim'],
        past_observed_values:       TensorType['batch', 'history_length', 'dim'],
        past_is_pad:                TensorType['batch', 'history_length'],
        future_time_feat:           TensorType['batch', 'prediction_length', 'feat_dim'],
        future_target_cdf:          TensorType['batch', 'prediction_length', 'dim'],
        future_observed_values:     TensorType['batch', 'prediction_length', 'dim'],
    ) -> TensorType[()]:

        past_time_feat = past_time_feat[...,-self.context_length:,:]
        past_target_cdf = past_target_cdf[...,-self.context_length:,:]
        past_observed_values = past_observed_values[...,-self.context_length:,:]
        past_is_pad = past_is_pad[...,-self.context_length:]

        past_observed_values = torch.min(past_observed_values, 1 - past_is_pad.unsqueeze(-1))
        _, scale = self.scaler(past_target_cdf, past_observed_values)

        history = past_target_cdf / scale
        target = future_target_cdf / scale

        t = torch.arange(self.prediction_length).view(1, -1, 1).repeat(target.shape[0], 1, 1).to(target)

        loss = self.diffusion.get_loss(self.denoise_fn, target, t=t, history=history, covariates=future_time_feat)

        loss_weights, _ = future_observed_values.min(dim=-1, keepdim=True)
        loss = weighted_average(loss, weights=loss_weights, dim=1)

        return loss.mean()


class ScorePredictionNetwork(ScoreTrainingNetwork):
    def __init__(self, num_parallel_samples: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_parallel_samples = num_parallel_samples

    def forward(
        self,
        target_dimension_indicator: TensorType['batch', 'dim'],
        past_time_feat:             TensorType['batch', 'history_length', 'feat_dim'],
        past_target_cdf:            TensorType['batch', 'history_length', 'dim'],
        past_observed_values:       TensorType['batch', 'history_length', 'dim'],
        past_is_pad:                TensorType['batch', 'history_length'],
        future_time_feat:           TensorType['batch', 'prediction_length', 'feat_dim'],
    ) -> TensorType['batch', 'num_samples', 'prediction_length', 'dim']:

        past_observed_values = torch.min(past_observed_values, 1 - past_is_pad.unsqueeze(-1))

        rnn_states, scale = self.get_rnn_state(
            past_time_feat=past_time_feat,
            past_target_cdf=past_target_cdf,
            past_observed_values=past_observed_values,
            future_time_feat=future_time_feat,
        )

        t = torch.arange(self.prediction_length).view(1, -1, 1)
        t = t.repeat(rnn_states.shape[0] * self.num_parallel_samples, 1, 1).to(rnn_states)

        rnn_states = rnn_states.repeat_interleave(self.num_parallel_samples, dim=0)

        samples = self.diffusion.sample(self.denoise_fn, t=t, latent=rnn_states)
        samples = samples.unflatten(0, (-1, self.num_parallel_samples)) * scale.unsqueeze(1)

        return samples
