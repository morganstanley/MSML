from torchtyping import TensorType

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from pts.model import weighted_average
from pts.model.time_grad import TimeGradTrainingNetwork, TimeGradPredictionNetwork
from pts.model.time_grad.epsilon_theta import DiffusionEmbedding

from tsdiff.diffusion import GaussianDiffusion, OUDiffusion, GPDiffusion, BetaLinear
from tsdiff.utils import dotdict, EpsilonTheta


class TimeGradTrainingNetwork_AutoregressiveOld(TimeGradTrainingNetwork):
    def __init__(self, **kwargs):
        kwargs.pop('time_feat_dim')
        kwargs.pop('noise')
        super().__init__(**kwargs)

class TimeGradPredictionNetwork_AutoregressiveOld(TimeGradPredictionNetwork):
    def __init__(self, **kwargs):
        kwargs.pop('time_feat_dim')
        kwargs.pop('noise')
        super().__init__(**kwargs)


class DenoiseWrapper(nn.Module):
    def __init__(self, denoise_fn, target_dim, time_input):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.time_input = time_input
        if self.time_input:
            self.time_embedding = DiffusionEmbedding(dim=target_dim, proj_dim=target_dim, max_steps=100)

    def forward(self, x, t=None, i=None, latent=None, **kwargs):
        shape = x.shape

        if self.time_input:
            x = x + self.time_embedding(t.squeeze(-1).long())

        x = x.view(-1, 1, x.shape[-1])
        i = i.view(-1).long()
        latent = latent.reshape(-1, 1, latent.shape[-1])

        y = self.denoise_fn(x, i, latent)
        y = y.view(*shape)
        return y


################################################################################################
#### TimeGrad RNN encoder --> prediction all at once using time positional encoding
#### using the past prediction window sized RNN context
################################################################################################
class TimeGradTrainingNetwork_All(TimeGradTrainingNetwork):
    def __init__(self, **kwargs):
        args = dotdict(kwargs)
        self.noise = args.noise

        kwargs.pop('time_feat_dim')
        kwargs.pop('noise')
        super().__init__(**kwargs)

        self.time_input = (self.noise != 'normal')
        self.rnn_state_proj = nn.Linear(args.num_cells, args.conditioning_length)

        if self.noise == 'normal':
            diffusion = GaussianDiffusion
        elif self.noise == 'ou':
            diffusion = OUDiffusion
        elif self.noise == 'gp':
            diffusion = GPDiffusion
        else:
            raise NotImplementedError

        self.diffusion = diffusion(args.target_dim, args.diff_steps, BetaLinear(1e-4, args.beta_end), sigma=0.05, predict_gaussian_noise=True)

        denoise_fn = EpsilonTheta(
            target_dim=args.target_dim,
            cond_length=args.conditioning_length,
            residual_layers=args.residual_layers,
            residual_channels=args.residual_channels,
            dilation_cycle_length=args.dilation_cycle_length,
        )

        self.denoise_fn = DenoiseWrapper(denoise_fn, args.target_dim, self.time_input)

    def get_rnn_state(self, **kwargs):
        rnn_outputs, _, scale, _, _ = self.unroll_encoder(**kwargs)
        rnn_outputs = self.rnn_state_proj(rnn_outputs)
        return rnn_outputs, scale

    def forward(
        self,
        target_dimension_indicator: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target_cdf: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_is_pad: torch.Tensor,
        future_time_feat: torch.Tensor,
        future_target_cdf: torch.Tensor,
        future_observed_values: torch.Tensor,
    ) -> TensorType[()]:

        latent, scale = self.get_rnn_state(
            past_time_feat=past_time_feat,
            past_target_cdf=past_target_cdf,
            past_observed_values=past_observed_values,
            past_is_pad=past_is_pad,
            future_time_feat=future_time_feat,
            future_target_cdf=None,
            target_dimension_indicator=target_dimension_indicator,
        )

        mean = past_target_cdf[...,-self.prediction_length:,:].mean(1, keepdim=True)
        std = past_target_cdf[...,-self.prediction_length:,:].std(1, keepdim=True).clamp(1e-4)

        # target = future_target_cdf[...,-self.prediction_length:,:] / scale
        target = (future_target_cdf[...,-self.prediction_length:,:] - mean) / std
        # target = (future_target_cdf[...,-self.prediction_length:,:] - past_target_cdf[...,-1:,:] - mean) / std
        # target = (future_target_cdf[...,-self.prediction_length:,:] - past_target_cdf[...,-1:,:]) / scale

        t = torch.arange(self.prediction_length).view(1, -1, 1).repeat(target.shape[0], 1, 1).to(target)
        loss = self.diffusion.get_loss(self.denoise_fn, target, t=t, latent=latent, future_time_feat=future_time_feat)

        loss_weights, _ = future_observed_values.min(dim=-1, keepdim=True)
        loss = weighted_average(loss, weights=loss_weights, dim=1)

        return loss.mean()


class TimeGradPredictionNetwork_All(TimeGradTrainingNetwork_All):
    def __init__(self, num_parallel_samples: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_samples = num_parallel_samples

    def forward(
        self,
        target_dimension_indicator: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target_cdf: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_is_pad: torch.Tensor,
        future_time_feat: torch.Tensor,
    ) -> torch.Tensor:
        mean = past_target_cdf[...,-self.prediction_length:,:].mean(1, keepdim=True)
        std = past_target_cdf[...,-self.prediction_length:,:].std(1, keepdim=True).clamp(1e-4)

        latent, scale = self.get_rnn_state(
            past_time_feat=past_time_feat,
            past_target_cdf=past_target_cdf,
            past_observed_values=past_observed_values,
            past_is_pad=past_is_pad,
            future_time_feat=future_time_feat,
            future_target_cdf=None,
            target_dimension_indicator=target_dimension_indicator,
        )

        num_samples = (self.num_samples * latent.shape[0], *latent.shape[1:-1])
        latent = latent.repeat_interleave(self.num_samples, dim=0)
        future_time_feat = future_time_feat.repeat_interleave(self.num_samples, dim=0)

        t = torch.arange(self.prediction_length).view(*(1,) * len(latent.shape[:-3]), -1, 1)
        t = t.expand_as(latent[...,:1]).to(latent)

        samples = self.diffusion.sample(
            self.denoise_fn,
            num_samples=num_samples,
            latent=latent,
            t=t,
            future_time_feat=future_time_feat,
            device=latent.device,
        )

        samples = samples.unflatten(0, (-1, self.num_samples))
        samples = samples * std.unsqueeze(1) + mean.unsqueeze(1)
        # samples = samples * scale.unsqueeze(1)
        # samples = samples + past_target_cdf[...,-1:,:].unsqueeze(1)
        return samples


################################################################################################
#### TimeGrad Autoregressive -> predicts one by one
################################################################################################
class TimeGradTrainingNetwork_Autoregressive(TimeGradTrainingNetwork_All):
    def forward(
        self,
        target_dimension_indicator: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target_cdf: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_is_pad: torch.Tensor,
        future_time_feat: torch.Tensor,
        future_target_cdf: torch.Tensor,
        future_observed_values: torch.Tensor,
    ) -> TensorType[()]:

        latent, scale = self.get_rnn_state(
            past_time_feat=past_time_feat,
            past_target_cdf=past_target_cdf,
            past_observed_values=past_observed_values,
            past_is_pad=past_is_pad,
            future_time_feat=future_time_feat,
            future_target_cdf=future_target_cdf,
            target_dimension_indicator=target_dimension_indicator,
        )

        target = torch.cat([past_target_cdf[...,-self.context_length:,:], future_target_cdf], 1)
        target = target / scale

        loss = self.diffusion.get_loss(self.denoise_fn, target, latent=latent, future_time_feat=future_time_feat)

        past_observed_values = torch.min(past_observed_values, 1 - past_is_pad.unsqueeze(-1))
        observed_values = torch.cat((past_observed_values[:, -self.context_length:, ...], future_observed_values), dim=1)
        loss_weights, _ = observed_values.min(dim=-1, keepdim=True)

        loss = weighted_average(loss, weights=loss_weights, dim=1)

        return loss.mean()

class TimeGradPredictionNetwork_Autoregressive(TimeGradTrainingNetwork_Autoregressive):
    def __init__(self, num_parallel_samples: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_samples = num_parallel_samples
        self.shifted_lags = [l - 1 for l in self.lags_seq]

    @torch.no_grad()
    def forward(
        self,
        target_dimension_indicator: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target_cdf: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_is_pad: torch.Tensor,
        future_time_feat: torch.Tensor,
    ) -> torch.Tensor:

        past_observed_values = torch.min(past_observed_values, 1 - past_is_pad.unsqueeze(-1))

        past_time_feat = past_time_feat.repeat_interleave(self.num_samples, dim=0)
        past_target_cdf = past_target_cdf.repeat_interleave(self.num_samples, dim=0)
        past_observed_values = past_observed_values.repeat_interleave(self.num_samples, dim=0)
        past_is_pad = past_is_pad.repeat_interleave(self.num_samples, dim=0)
        future_time_feat = future_time_feat.repeat_interleave(self.num_samples, dim=0)
        target_dimension_indicator = target_dimension_indicator.repeat_interleave(self.num_samples, dim=0)

        _, begin_states, scale, _, _ = self.unroll_encoder(
            past_time_feat=past_time_feat,
            past_target_cdf=past_target_cdf,
            past_observed_values=past_observed_values,
            past_is_pad=past_is_pad,
            future_time_feat=future_time_feat,
            future_target_cdf=None,
            target_dimension_indicator=target_dimension_indicator,
        )

        samples = []
        for i in range(self.prediction_length):
            lags = self.get_lagged_subsequences(
                sequence=past_target_cdf,
                sequence_length=self.history_length + i,
                indices=self.shifted_lags,
                subsequences_length=1,
            )

            latent, begin_states, _, _ = self.unroll(
                begin_state=begin_states,
                lags=lags,
                scale=scale,
                time_feat=future_time_feat[:, i : i + 1],
                target_dimension_indicator=target_dimension_indicator,
                unroll_length=1,
            )
            latent = self.rnn_state_proj(latent)

            sample = self.diffusion.sample(
                self.denoise_fn,
                num_samples=latent.shape[:-1],
                latent=latent,
                device=latent.device,
            )
            sample = sample * scale

            samples.append(sample)
            past_target_cdf = torch.cat([past_target_cdf, sample], dim=1)

        samples = torch.cat(samples, dim=1)
        samples = samples.unflatten(0, (-1, self.num_samples))

        return samples


################################################################################################
#### TimeGrad RNN encoder --> predicting all at once with RNN+TimeGrad decoder
#### RNN initial state is the last state from the encoder
################################################################################################
class TimeGradTrainingNetwork_RNN(TimeGradTrainingNetwork_All):
    def __init__(self, **kwargs):
        args = dotdict(kwargs)
        super().__init__(**kwargs)

        self.num_rnn_layers = 2
        self.proj_inputs = nn.Sequential(
            nn.Linear(args.time_feat_dim , args.conditioning_length),
            nn.ReLU(),
            nn.Linear(args.conditioning_length , args.conditioning_length),
        )
        self.prediction_rnn = nn.GRU(args.conditioning_length, args.conditioning_length,
            num_layers=self.num_rnn_layers, bidirectional=False, batch_first=True)

    def get_rnn_state(self, **kwargs):
        states, _, scale, _, _ = self.unroll_encoder(**kwargs)
        states = self.rnn_state_proj(states)

        states = states[...,-1,:].unsqueeze(0).repeat_interleave(self.num_rnn_layers, dim=0)

        inputs = self.proj_inputs(kwargs['future_time_feat'])
        out, _ = self.prediction_rnn(inputs, states)

        return out, scale

class TimeGradPredictionNetwork_RNN(TimeGradTrainingNetwork_RNN):
    forward = TimeGradPredictionNetwork_All.forward

    def __init__(self, num_parallel_samples: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_samples = num_parallel_samples


################################################################################################
#### TimeGrad RNN encoder --> predicting all at once with Transformer decoder+TimeGrad net
################################################################################################
class TimeGradTrainingNetwork_Transformer(TimeGradTrainingNetwork_All):
    def __init__(self, **kwargs):
        args = dotdict(kwargs)
        super().__init__(**kwargs)

        self.pos_enc = DiffusionEmbedding(dim=args.conditioning_length, proj_dim=args.conditioning_length, max_steps=100)
        self.proj_time_feat = nn.Linear(args.time_feat_dim + args.conditioning_length, args.conditioning_length)

        decoder_layer = nn.TransformerDecoderLayer(args.conditioning_length, nhead=1, dim_feedforward=args.conditioning_length, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

    def get_rnn_state(self, **kwargs):
        states, _, scale, _, _ = self.unroll_encoder(**kwargs)
        states = self.rnn_state_proj(states)

        t = torch.arange(self.prediction_length).view(1, -1).repeat(states.shape[0], 1).to(states)
        t = self.pos_enc(t.long())

        x = torch.cat([t, kwargs['future_time_feat']], -1)
        x = self.proj_time_feat(x)
        out = self.transformer_decoder(tgt=x, memory=states)
        return out, scale

class TimeGradPredictionNetwork_Transformer(TimeGradTrainingNetwork_Transformer):
    forward = TimeGradPredictionNetwork_All.forward

    def __init__(self, num_parallel_samples: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_samples = num_parallel_samples


################################################################################################
#### TimeGrad RNN encoder --> predicting all at once with 2D conv similar to TimeGrad version
################################################################################################
class ResidualBlock(nn.Module):
    def __init__(self, dim, hidden_size, residual_channels, dilation, padding_mode):
        super().__init__()
        self.step_projection = nn.Linear(hidden_size, residual_channels)
        self.time_projection = nn.Linear(hidden_size, residual_channels)

        self.x_step_proj = nn.Sequential(
            nn.Conv2d(residual_channels, residual_channels, kernel_size=1, padding='same', padding_mode=padding_mode),
            nn.LeakyReLU(0.4),
        )
        self.x_time_proj = nn.Sequential(
            nn.Conv2d(residual_channels, residual_channels, kernel_size=1, padding='same', padding_mode=padding_mode),
            nn.LeakyReLU(0.4),
        )

        self.latent_projection = nn.Conv2d(
            1, 2 * residual_channels, kernel_size=1, padding='same', padding_mode=padding_mode,
        )
        self.dilated_conv = nn.Conv2d(
            1 * residual_channels,
            2 * residual_channels,
            kernel_size=3,
            dilation=dilation,
            padding='same',
            padding_mode=padding_mode,
        )
        self.output_projection = nn.Conv2d(
            residual_channels, 2 * residual_channels, kernel_size=1, padding='same', padding_mode=padding_mode,
        )

    def forward(self, x, t=None, i=None, latent=None):
        i = self.step_projection(i).transpose(-1, -2).unsqueeze(-1)
        latent = self.latent_projection(latent.unsqueeze(1))

        y = x + i
        y = y + self.x_step_proj(y)

        t = self.time_projection(t).transpose(-1, -2).unsqueeze(-1)
        y = y + self.x_time_proj(y + t)

        y = self.dilated_conv(y) + latent

        gate, filter = y.chunk(2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        y = F.leaky_relu(y, 0.4)

        residual, skip = y.chunk(2, dim=1)
        return (x + residual) / math.sqrt(2), skip

class DenoisingModel(nn.Module):
    def __init__(self, dim, residual_channels, latent_dim, residual_hidden, residual_layers, time_input, padding_mode='circular'):
        super().__init__()
        self.time_input = time_input

        self.input_projection = nn.Conv2d(1, residual_channels, kernel_size=1, padding='same', padding_mode=padding_mode)
        self.step_embedding = DiffusionEmbedding(residual_hidden, proj_dim=residual_hidden)
        self.time_embedding = DiffusionEmbedding(residual_hidden, proj_dim=residual_hidden, max_steps=24)
        self.latent_projection = nn.Sequential(
            nn.Linear(latent_dim, dim // 2),
            nn.LeakyReLU(0.4),
            nn.Linear(dim // 2, dim),
            nn.LeakyReLU(0.4),
        )

        self.residual_layers = nn.ModuleList([
            ResidualBlock(dim, residual_hidden, residual_channels, dilation=2**(i % 2), padding_mode=padding_mode)
            for i in range(residual_layers)
        ])

        self.skip_projection = nn.Conv2d(
            residual_channels, residual_channels, kernel_size=3, padding='same', padding_mode=padding_mode,
        )
        self.output_projection = nn.Conv2d(
            residual_channels, 1, kernel_size=3, padding='same', padding_mode=padding_mode,
        )

        self.time_proj = nn.Sequential(
            nn.Linear(5, residual_hidden),
            nn.LeakyReLU(0.4),
            nn.Linear(residual_hidden, residual_hidden),
            nn.LeakyReLU(0.4),
        )

    def forward(self, x, t=None, i=None, latent=None, future_time_feat=None):
        shape = x.shape

        x = x.unsqueeze(1)
        x = self.input_projection(x)
        x = F.leaky_relu(x, 0.4)

        i = self.step_embedding(i.squeeze(-1).long())
        # if t is not None:
        #     t = self.time_embedding(t.squeeze(-1).long())

        t = self.time_proj(torch.cat([future_time_feat, t / t.max()], -1))

        latent = self.latent_projection(latent)

        skip_agg = 0
        for layer in self.residual_layers:
            x, skip = layer(x, t=t, i=i, latent=latent)
            skip_agg = skip_agg + skip

        x = skip_agg / math.sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.leaky_relu(x, 0.4)
        x = self.output_projection(x).squeeze(1)

        x = x.view(*shape)
        return x

class TimeGradTrainingNetwork_CNN(TimeGradTrainingNetwork_All):
    def __init__(self, **kwargs):
        args = dotdict(kwargs)
        super().__init__(**kwargs)
        self.denoise_fn = DenoisingModel(
            dim=args.target_dim,
            residual_channels=args.residual_channels,
            latent_dim=args.conditioning_length,
            residual_hidden=args.conditioning_length,
            time_input=self.time_input,
            residual_layers=args.residual_layers,
        )

class TimeGradPredictionNetwork_CNN(TimeGradTrainingNetwork_CNN):
    forward = TimeGradPredictionNetwork_All.forward

    def __init__(self, num_parallel_samples: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_samples = num_parallel_samples
        self.shifted_lags = [l - 1 for l in self.lags_seq]
