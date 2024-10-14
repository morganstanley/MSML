from typing import Any, Iterable, Optional

import lightning.pytorch as pl
import torch
import torch.nn as nn
from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import as_stacked_batches
from gluonts.dataset.stat import calculate_dataset_statistics
from gluonts.itertools import Cyclic, prod, select
from gluonts.model import Input, InputSpec
from gluonts.time_feature import (
    TimeFeature,
    get_lags_for_frequency,
    time_features_from_frequency_str,
)
from gluonts.torch.model.estimator import PyTorchLightningEstimator
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.torch.modules.feature import FeatureEmbedder
from gluonts.torch.scaler import MeanScaler, NOPScaler, Scaler, StdScaler
from gluonts.torch.util import repeat_along_dim, unsqueeze_expand
from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    DummyValueImputation,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    MissingValueImputation,
    RemoveFields,
    SetField,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
    VstackFeatures,
)
from gluonts.transform.sampler import InstanceSampler
from pts.util import lagged_sequence_values
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau

from time_match.nn.epsilon_theta import EpsilonTheta
from time_match.nn.unet_1d import UNet1DConditionModel


class BaseForecastModule(nn.Module):
    """
    Parameters
    ----------
    freq
        String indicating the sampling frequency of the data to be processed.
    context_length
        Length of the RNN unrolling prior to the forecast date.
    prediction_length
        Number of time points to predict.
    num_feat_dynamic_real
        Number of dynamic real features that will be provided to ``forward``.
    num_feat_static_real
        Number of static real features that will be provided to ``forward``.
    num_feat_static_cat
        Number of static categorical features that will be provided to
        ``forward``.
    cardinality
        List of cardinalities, one for each static categorical feature.
    embedding_dimension
        Dimension of the embedding space, one for each static categorical
        feature.
    num_layers
        Number of layers in the RNN.
    hidden_size
        Size of the hidden layers in the RNN.
    dropout_rate
        Dropout rate to be applied at training time.
    lags_seq
        Indices of the lagged observations that the RNN takes as input. For
        example, ``[1]`` indicates that the RNN only takes the observation at
        time ``t-1`` to produce the output for time ``t``; instead,
        ``[1, 25]`` indicates that the RNN takes observations at times ``t-1``
        and ``t-25`` as input.
    scaling
        Whether to apply mean scaling to the observations (target).
    default_scale
        Default scale that is applied if the context length window is
        completely unobserved. If not set, the scale in this case will be
        the mean scale in the batch.
    num_parallel_samples
        Number of samples to produce when unrolling the RNN in the prediction
        time range.
    denoising_model
        Name of the denoising network
    velocity_model
        Name of the velocity network
    encoding_model
        Name of the encoding networks
    """
    @validated()
    def __init__(
        self,
        freq: str,
        context_length: int,
        prediction_length: int,
        log_count: int = 10,
        input_size: int = 1,
        target_dim: Optional[int] = None,
        num_feat_dynamic_real: int = 1,
        num_feat_static_real: int = 1,
        num_feat_static_cat: int = 1,
        cardinality: list[int] = [1],
        embedding_dimension: Optional[list[int]] = None,
        num_layers: int = None,
        hidden_size: int = None,
        dropout_rate: float = None,
        lags_seq: Optional[list[int]] = None,
        scaling: Optional[str] = None,
        default_scale: float = 0.0,
        num_parallel_samples: int = None,
        denoising_model: str = None,
        velocity_model: str = None,
        rnn_model: str = None,
    ) -> None:
        super().__init__()
        assert num_feat_dynamic_real > 0
        assert num_feat_static_real > 0
        assert num_feat_static_cat > 0
        assert len(cardinality) == num_feat_static_cat
        assert (
            embedding_dimension is None
            or len(embedding_dimension) == num_feat_static_cat
        )

        if target_dim is None:
            target_dim = input_size

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.input_size = input_size

        self.log_count = log_count

        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.num_feat_static_cat = num_feat_static_cat
        self.num_feat_static_real = num_feat_static_real
        self.embedding_dimension = (
            embedding_dimension
            if embedding_dimension is not None or cardinality is None
            else [min(50, (cat + 1) // 2) for cat in cardinality]
        )
        self.lags_seq = lags_seq or get_lags_for_frequency(freq_str=freq)
        self.lags_seq = [lag - 1 for lag in self.lags_seq]
        self.num_parallel_samples = num_parallel_samples
        self.past_length = self.context_length + max(self.lags_seq)
        self.embedder = FeatureEmbedder(
            cardinalities=cardinality, embedding_dims=self.embedding_dimension
        )
        if scaling == "mean":
            self.scaler: Scaler = MeanScaler(
                dim=1, keepdim=True, default_scale=default_scale, minimum_scale=0.01,
            )
        elif scaling == "std":
            self.scaler: Scaler = StdScaler(
                dim=1, keepdim=True, minimum_scale=0.01,
            )
        else:
            self.scaler: Scaler = NOPScaler(dim=1, keepdim=True)

        self.rnn_input_size = (
            self.input_size * len(self.lags_seq) + self._number_of_features
        )

        if rnn_model == 'lstm':
            self.rnn = nn.LSTM(
                input_size=self.rnn_input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout_rate,
                batch_first=True,
            )

        if denoising_model == 'epsilon_theta':
            self.denoising_model = EpsilonTheta(
                input_dim=input_size,
                target_dim=target_dim,
                cond_dim=hidden_size,
            )
        elif denoising_model == 'unet1d':
            self.denoising_model = UNet1DConditionModel(
                residual_layers=8,
                residual_channels=8,
                hidden_size=hidden_size,
                target_dim=target_dim,
            )

        if velocity_model == 'epsilon_theta':
            self.velocity_model = EpsilonTheta(
                input_dim=input_size,
                target_dim=target_dim,
                cond_dim=hidden_size,
            )
        elif velocity_model == 'unet1d':
            self.velocity_model = UNet1DConditionModel(
                residual_layers=8,
                residual_channels=8,
                hidden_size=hidden_size,
                target_dim=target_dim,
            )

    def describe_inputs(self, batch_size=1) -> InputSpec:
        return InputSpec(
            {
                "feat_static_cat": Input(
                    shape=(batch_size, self.num_feat_static_cat), dtype=torch.long
                ),
                "feat_static_real": Input(
                    shape=(batch_size, self.num_feat_static_real), dtype=torch.float
                ),
                "past_time_feat": Input(
                    shape=(batch_size, self._past_length, self.num_feat_dynamic_real),
                    dtype=torch.float,
                ),
                "past_target": Input(
                    shape=(batch_size, self._past_length)
                    if self.input_size == 1
                    else (batch_size, self._past_length, self.input_size),
                    dtype=torch.float,
                ),
                "past_observed_values": Input(
                    shape=(batch_size, self._past_length)
                    if self.input_size == 1
                    else (batch_size, self._past_length, self.input_size),
                    dtype=torch.float,
                ),
                "future_time_feat": Input(
                    shape=(
                        batch_size,
                        self.prediction_length,
                        self.num_feat_dynamic_real,
                    ),
                    dtype=torch.float,
                ),
            },
            zeros_fn=torch.zeros,
        )

    @property
    def _number_of_features(self) -> int:
        return (
            sum(self.embedding_dimension)
            + self.num_feat_dynamic_real
            + self.num_feat_static_real
            + self.input_size * 2  # the log(scale) and log1p(abs(loc))
        )

    @property
    def _past_length(self) -> int:
        return self.context_length + max(self.lags_seq)

    def prepare_rnn_input(
        self,
        feat_static_cat: Tensor,
        feat_static_real: Tensor,
        past_time_feat: Tensor,
        past_target: Tensor,
        past_observed_values: Tensor,
        future_time_feat: Tensor,
        future_target: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor,]:
        context = past_target[:, -self.context_length :, ...]
        observed_context = past_observed_values[:, -self.context_length :, ...]

        input, loc, scale = self.scaler(context, observed_context)
        future_length = future_time_feat.shape[-2]
        if future_length > 1:
            assert future_target is not None
            input = torch.cat(
                (input, (future_target[:, : future_length - 1, ...] - loc) / scale),
                dim=1,
            )
        prior_input = (past_target[:, : -self.context_length, ...] - loc) / scale

        lags = lagged_sequence_values(self.lags_seq, prior_input, input, dim=1)
        time_feat = torch.cat(
            (past_time_feat[:, -self.context_length + 1 :, ...], future_time_feat),
            dim=1,
        )

        embedded_cat = self.embedder(feat_static_cat)
        log_abs_loc = (
            loc.abs().log1p() if self.input_size == 1 else loc.squeeze(1).abs().log1p()
        )
        log_scale = scale.log() if self.input_size == 1 else scale.squeeze(1).log()

        static_feat = torch.cat(
            (embedded_cat, feat_static_real, log_abs_loc, log_scale), dim=-1
        )
        expanded_static_feat = unsqueeze_expand(
            static_feat, dim=1, size=time_feat.shape[-2]
        )

        features = torch.cat((expanded_static_feat, time_feat), dim=-1)

        return torch.cat((lags, features), dim=-1), loc, scale, static_feat

    def unroll_lagged_rnn(
        self,
        feat_static_cat: Tensor,
        feat_static_real: Tensor,
        past_time_feat: Tensor,
        past_target: Tensor,
        past_observed_values: Tensor,
        future_time_feat: Tensor,
        future_target: Optional[Tensor] = None,
    ) -> tuple[
        tuple[Tensor, ...],
        Tensor,
        Tensor,
        Tensor,
        tuple[Tensor, Tensor],
    ]:
        """
        Applies the underlying RNN to the provided target data and covariates.

        Parameters
        ----------
        feat_static_cat
            Tensor of static categorical features,
            shape: ``(batch_size, num_feat_static_cat)``.
        feat_static_real
            Tensor of static real features,
            shape: ``(batch_size, num_feat_static_real)``.
        past_time_feat
            Tensor of dynamic real features in the past,
            shape: ``(batch_size, past_length, num_feat_dynamic_real)``.
        past_target
            Tensor of past target values,
            shape: ``(batch_size, past_length)``.
        past_observed_values
            Tensor of observed values indicators,
            shape: ``(batch_size, past_length)``.
        future_time_feat
            Tensor of dynamic real features in the future,
            shape: ``(batch_size, prediction_length, num_feat_dynamic_real)``.
        future_target
            (Optional) tensor of future target values,
            shape: ``(batch_size, prediction_length)``.

        Returns
        -------
        tuple
            A tuple containing, in this order:
            - Parameters of the output distribution
            - Scaling factor applied to the target
            - Raw output of the RNN
            - Static input to the RNN
            - Output state from the RNN
        """
        rnn_input, loc, scale, static_feat = self.prepare_rnn_input(
            feat_static_cat,
            feat_static_real,
            past_time_feat,
            past_target,
            past_observed_values,
            future_time_feat,
            future_target,
        )
        output, new_state = self.rnn(rnn_input)

        return loc, scale, output, static_feat, new_state

    def sample(
        self,
        x: Tensor,
        cond: Optional[Tensor] = None,
    ) -> Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def forward(
        self,
        feat_static_cat: Tensor,
        feat_static_real: Tensor,
        past_time_feat: Tensor,
        past_target: Tensor,
        past_observed_values: Tensor,
        future_time_feat: Tensor,
        num_parallel_samples: Optional[int] = None,
    ) -> Tensor:
        """
        Invokes the model on input data, and produce outputs future samples.

        Parameters
        ----------
        feat_static_cat
            Tensor of static categorical features,
            shape: ``(batch_size, num_feat_static_cat)``.
        feat_static_real
            Tensor of static real features,
            shape: ``(batch_size, num_feat_static_real)``.
        past_time_feat
            Tensor of dynamic real features in the past,
            shape: ``(batch_size, past_length, num_feat_dynamic_real)``.
        past_target
            Tensor of past target values,
            shape: ``(batch_size, past_length)``.
        past_observed_values
            Tensor of observed values indicators,
            shape: ``(batch_size, past_length)``.
        future_time_feat
            (Optional) tensor of dynamic real features in the past,
            shape: ``(batch_size, prediction_length, num_feat_dynamic_real)``.
        num_parallel_samples
            How many future samples to produce.
            By default, self.num_parallel_samples is used.
        """
        if num_parallel_samples is None:
            num_parallel_samples = self.num_parallel_samples

        loc, scale, rnn_output, static_feat, state = self.unroll_lagged_rnn(
            feat_static_cat,
            feat_static_real,
            past_time_feat,
            past_target,
            past_observed_values,
            future_time_feat[:, :1],
        )

        repeated_scale = scale.repeat_interleave(repeats=num_parallel_samples, dim=0)
        repeated_loc = loc.repeat_interleave(repeats=num_parallel_samples, dim=0)

        repeated_static_feat = static_feat.repeat_interleave(
            repeats=num_parallel_samples, dim=0
        ).unsqueeze(dim=1)
        repeated_past_target = (
            past_target.repeat_interleave(repeats=num_parallel_samples, dim=0)
            - repeated_loc
        ) / repeated_scale
        repeated_time_feat = future_time_feat.repeat_interleave(
            repeats=num_parallel_samples, dim=0
        )
        repeated_state = [
            s.repeat_interleave(repeats=num_parallel_samples, dim=1) for s in state
        ]
        repeated_outputs = rnn_output.repeat_interleave(
            repeats=num_parallel_samples, dim=0
        )

        # sample
        next_sample = self.sample(
            x=repeated_past_target[:, -1:, ...],
            cond=repeated_outputs[:, -1:, ...],
        )
        future_samples = [repeated_scale * next_sample + repeated_loc]

        for k in range(1, self.prediction_length):
            next_features = torch.cat(
                (repeated_static_feat, repeated_time_feat[:, k : k + 1]),
                dim=-1,
            )
            next_lags = lagged_sequence_values(
                self.lags_seq, repeated_past_target, next_sample, dim=1
            )
            rnn_input = torch.cat((next_lags, next_features), dim=-1)

            repeated_outputs, repeated_state = self.rnn(rnn_input, repeated_state)

            repeated_past_target = torch.cat(
                (repeated_past_target, next_sample), dim=1
            )

            next_sample = self.sample(
                x=repeated_past_target[:, -1:, ...],
                cond=repeated_outputs,
            )
            future_samples.append(repeated_scale * next_sample + repeated_loc)

        future_samples_concat = torch.cat(future_samples, dim=1).reshape(
            (-1, num_parallel_samples, self.prediction_length, self.input_size)
        )

        return future_samples_concat.squeeze(-1)

    def get_loss_values(
        self,
        context: Tensor,
        target: Tensor,
        cond: Tensor,
        observed_values_mask: Tensor,
    ) -> Tensor:
        raise NotImplementedError

    def loss(
        self,
        feat_static_cat: Tensor,
        feat_static_real: Tensor,
        past_time_feat: Tensor,
        past_target: Tensor,
        past_observed_values: Tensor,
        future_time_feat: Tensor,
        future_target: Tensor,
        future_observed_values: Tensor,
        future_only: bool = True,
        aggregate_by=torch.mean,
    ) -> Tensor:

        extra_dims = len(future_target.shape) - len(past_target.shape)
        extra_shape = future_target.shape[:extra_dims]

        repeats = prod(extra_shape)
        feat_static_cat = repeat_along_dim(feat_static_cat, 0, repeats)
        feat_static_real = repeat_along_dim(feat_static_real, 0, repeats)
        past_time_feat = repeat_along_dim(past_time_feat, 0, repeats)
        past_target = repeat_along_dim(past_target, 0, repeats)
        past_observed_values = repeat_along_dim(past_observed_values, 0, repeats)
        future_time_feat = repeat_along_dim(future_time_feat, 0, repeats)

        future_target_reshaped = future_target.reshape(
            -1, *future_target.shape[extra_dims + 1 :]
        )
        future_observed_reshaped = future_observed_values.reshape(
            -1, *future_observed_values.shape[extra_dims + 1 :]
        )

        loc, scale, rnn_outputs, _, _ = self.unroll_lagged_rnn(
            feat_static_cat,
            feat_static_real,
            past_time_feat,
            past_target,
            past_observed_values,
            future_time_feat,
            future_target_reshaped,
        )

        if future_only:
            past_target = torch.cat([
                past_target[:, -1:],
                future_target_reshaped[:, :-1]
            ], 1)

            sliced_rnn_outputs = rnn_outputs[:, -self.prediction_length :]
            observed_values = (
                future_observed_reshaped.all(-1)
                if future_observed_reshaped.ndim == 3
                else future_observed_reshaped
            )

            past_target = (past_target - loc) / scale
            future_target_reshaped = (future_target_reshaped - loc) / scale

            loss = self.get_loss_values(
                context=past_target,
                target=future_target_reshaped,
                cond=sliced_rnn_outputs,
                observed_values_mask=observed_values,
            )
        else:
            context_target = past_target[:, -self.context_length + 1 :, ...]
            target = torch.cat((context_target, future_target_reshaped), dim=1)
            context_observed = past_observed_values[:, -self.context_length + 1 :, ...]
            observed_values = torch.cat(
                (context_observed, future_observed_reshaped), dim=1
            )
            observed_values = (
                observed_values.all(-1)
                if observed_values.ndim == 3
                else observed_values
            )

            past_target = (past_target - loc) / scale
            target = (target - loc) / scale

            loss = self.get_loss_values(
                context=past_target,
                target=target,
                cond=rnn_outputs,
                observed_values_mask=observed_values,
            )

        return aggregate_by(loss, dim=(1,))


class BaseLightningModule(pl.LightningModule):
    """
    A ``pl.LightningModule`` class that can be used to train a
    ``T2TSBModel`` with PyTorch Lightning.

    This is a thin layer around a (wrapped) ``T2TSBModel`` object,
    that exposes the methods to evaluate training and validation loss.

    Parameters
    ----------
    model
        ``T2TSBModel`` to be trained.
    lr
        Learning rate, default: ``1e-3``.
    weight_decay
        Weight decay regularization parameter, default: ``1e-8``.
    patience
        Patience parameter for learning rate scheduler, default: ``10``.
    """

    @validated()
    def __init__(
        self,
        model_kwargs: dict,
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
        patience: int = 10,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience

    @property
    def inputs(self):
        return self.model.describe_inputs()

    @property
    def example_input_array(self):
        return self.inputs.zeros()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx: int):  # type: ignore
        """
        Execute training step.
        """
        train_loss = self.model.loss(
            **select(self.inputs, batch),
            future_observed_values=batch["future_observed_values"],
            future_target=batch["future_target"],
        ).mean()

        self.log(
            "train_loss",
            train_loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )

        return train_loss

    def validation_step(self, batch, batch_idx: int):  # type: ignore
        """
        Execute validation step.
        """
        val_loss = self.model.loss(
            **select(self.inputs, batch),
            future_observed_values=batch["future_observed_values"],
            future_target=batch["future_target"],
        ).mean()

        self.log("val_loss", val_loss, on_epoch=True, on_step=False, prog_bar=True)

        return val_loss

    def configure_optimizers(self):
        """
        Returns the optimizer to use.
        """
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        monitor = (
            "val_loss"
            if self.trainer.fit_loop.epoch_loop.val_loop._data_source.is_defined()
            else "train_loss"
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer=optimizer,
                    mode="min",
                    factor=0.5,
                    patience=self.patience,
                ),
                "monitor": monitor,
            },
        }



PREDICTION_INPUT_NAMES = [
    "feat_static_cat",
    "feat_static_real",
    "past_time_feat",
    "past_target",
    "past_observed_values",
    "future_time_feat",
]

TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
    "future_target",
    "future_observed_values",
]


class BaseEstimator(PyTorchLightningEstimator):
    """
    Estimator class to train a TimeMatch model.

    This class is uses the model defined in ``TimeMatchModel``, and wraps it
    into a ``TimeMatchLightningModule`` for training purposes: training is
    performed using PyTorch Lightning's ``pl.Trainer`` class.

    Parameters
    ----------
    freq
        Frequency of the data to train on and predict.
    prediction_length
        Length of the prediction horizon.
    context_length
        Number of steps to unroll the RNN for before computing predictions
        (default: None, in which case context_length = prediction_length).
    input_size:
        Number of variates in the input time series (default: 1 for univariate).
    num_layers
        Number of RNN layers (default: 2).
    hidden_size
        Number of RNN cells for each layer (default: 40).
    lr
        Learning rate (default: ``1e-3``).
    weight_decay
        Weight decay regularization parameter (default: ``1e-8``).
    dropout_rate
        Dropout regularization parameter (default: 0.1).
    patience
        Patience parameter for learning rate scheduler.
    num_feat_dynamic_real
        Number of dynamic real features in the data (default: 0).
    num_feat_static_real
        Number of static real features in the data (default: 0).
    num_feat_static_cat
        Number of static categorical features in the data (default: 0).
    cardinality
        Number of values of each categorical feature.
        This must be set if ``num_feat_static_cat > 0`` (default: None).
    embedding_dimension
        Dimension of the embeddings for categorical features
        (default: ``[min(50, (cat+1)//2) for cat in cardinality]``).
    scaling
        Whether to automatically scale the target values (default: "mean"). Can be
        set to "none" to disable scaling, to "std" to apply Std Scaling, or to
        "mean" to apply Mean Scaling.
    default_scale
        Default scale that is applied if the context length window is
        completely unobserved. If not set, the scale in this case will be
        the mean scale in the batch.
    lags_seq
        Indices of the lagged target values to use as inputs of the RNN
        (default: None, in which case these are automatically determined
        based on freq).
    time_features
        List of time features, from :py:mod:`gluonts.time_feature`, to use as
        inputs of the RNN in addition to the provided data (default: None,
        in which case these are automatically determined based on freq).
    num_parallel_samples
        Number of samples per time series to that the resulting predictor
        should produce (default: 100).
    batch_size
        The size of the batches to be used for training (default: 32).
    num_batches_per_epoch
        Number of batches to be processed in each training epoch
        (default: 50).
    trainer_kwargs
        Additional arguments to provide to ``pl.Trainer`` for construction.
    train_sampler
        Controls the sampling of windows during training.
    validation_sampler
        Controls the sampling of windows during validation.
    """

    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        input_size: int,
        context_length: Optional[int] = None,
        num_layers: int = None,
        hidden_size: int = None,
        lr: float = None,
        weight_decay: float = None,
        dropout_rate: float = 0.1,
        patience: int = 10,
        num_feat_dynamic_real: int = 0,
        num_feat_static_cat: int = 0,
        num_feat_static_real: int = 0,
        cardinality: Optional[list[int]] = None,
        embedding_dimension: Optional[list[int]] = None,
        scaling: Optional[str] = "mean",
        default_scale: float = 0.0,
        lags_seq: Optional[list[int]] = None,
        time_features: Optional[list[TimeFeature]] = None,
        num_parallel_samples: int = 100,
        batch_size: int = None,
        num_batches_per_epoch: int = 50,
        imputation_method: Optional[MissingValueImputation] = None,
        trainer_kwargs: Optional[dict[str, Any]] = None,
        train_sampler: Optional[InstanceSampler] = None,
        validation_sampler: Optional[InstanceSampler] = None,
        denoising_model: str = None,
        velocity_model: str = None,
        rnn_model: str = None,
    ) -> None:
        default_trainer_kwargs = {
            "max_epochs": 100,
            "gradient_clip_val": 10.0,
        }
        if trainer_kwargs is not None:
            default_trainer_kwargs.update(trainer_kwargs)
        super().__init__(trainer_kwargs=default_trainer_kwargs)

        self.freq = freq
        self.prediction_length = prediction_length
        self.input_size = input_size
        self.context_length = (
            context_length if context_length is not None else prediction_length
        )

        self.patience = patience
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.num_feat_static_cat = num_feat_static_cat
        self.num_feat_static_real = num_feat_static_real
        self.cardinality = (
            cardinality if cardinality and num_feat_static_cat > 0 else [1]
        )
        self.embedding_dimension = embedding_dimension
        self.scaling = scaling
        self.default_scale = default_scale
        self.lags_seq = lags_seq
        self.time_features = (
            time_features
            if time_features is not None
            else time_features_from_frequency_str(self.freq)
        )

        self.num_parallel_samples = num_parallel_samples
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch

        self.imputation_method = (
            imputation_method
            if imputation_method is not None
            else DummyValueImputation(0.0)
        )

        self.train_sampler = train_sampler or ExpectedNumInstanceSampler(
            num_instances=1.0, min_future=prediction_length
        )
        self.validation_sampler = validation_sampler or ValidationSplitSampler(
            min_future=prediction_length
        )

        self.denoising_model = denoising_model
        self.velocity_model = velocity_model
        self.rnn_model = rnn_model

        self.lightning_module = None
        self.additional_model_kwargs = {}

    @classmethod
    def derive_auto_fields(cls, train_iter):
        stats = calculate_dataset_statistics(train_iter)

        return {
            "num_feat_dynamic_real": stats.num_feat_dynamic_real,
            "num_feat_static_cat": len(stats.feat_static_cat),
            "cardinality": [len(cats) for cats in stats.feat_static_cat],
        }

    def create_transformation(self) -> Transformation:
        remove_field_names = []
        if self.num_feat_static_real == 0:
            remove_field_names.append(FieldName.FEAT_STATIC_REAL)
        if self.num_feat_dynamic_real == 0:
            remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)

        return Chain(
            [RemoveFields(field_names=remove_field_names)]
            + (
                [SetField(output_field=FieldName.FEAT_STATIC_CAT, value=[0])]
                if not self.num_feat_static_cat > 0
                else []
            )
            + (
                [SetField(output_field=FieldName.FEAT_STATIC_REAL, value=[0.0])]
                if not self.num_feat_static_real > 0
                else []
            )
            + [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_CAT,
                    expected_ndim=1,
                    dtype=int,
                ),
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_REAL,
                    expected_ndim=1,
                ),
                AsNumpyArray(
                    field=FieldName.TARGET,
                    # in the following line, we add 1 for the time dimension and 1 for the multivariate dimension
                    expected_ndim=2,
                ),
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                    imputation_method=self.imputation_method,
                ),
                AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=self.time_features,
                    pred_length=self.prediction_length,
                ),
                AddAgeFeature(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_AGE,
                    pred_length=self.prediction_length,
                    log_scale=True,
                ),
                VstackFeatures(
                    output_field=FieldName.FEAT_TIME,
                    input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
                    + (
                        [FieldName.FEAT_DYNAMIC_REAL]
                        if self.num_feat_dynamic_real > 0
                        else []
                    ),
                ),
                AsNumpyArray(FieldName.FEAT_TIME, expected_ndim=2),
            ]
        )

    def _create_instance_splitter(self, module: BaseLightningModule, mode: str):
        assert mode in ["training", "validation", "test"]

        instance_sampler = {
            "training": self.train_sampler,
            "validation": self.validation_sampler,
            "test": TestSplitSampler(),
        }[mode]

        return InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            past_length=module.model._past_length,
            future_length=self.prediction_length,
            time_series_fields=[
                FieldName.FEAT_TIME,
                FieldName.OBSERVED_VALUES,
            ],
            dummy_value=0.0,
        )

    def create_training_data_loader(
        self,
        data: Dataset,
        module: BaseLightningModule,
        shuffle_buffer_length: Optional[int] = None,
        **kwargs,
    ) -> Iterable:
        data = Cyclic(data).stream()
        instances = self._create_instance_splitter(module, "training").apply(
            data, is_train=True
        )
        return as_stacked_batches(
            instances,
            batch_size=self.batch_size,
            shuffle_buffer_length=shuffle_buffer_length,
            field_names=TRAINING_INPUT_NAMES,
            output_type=torch.tensor,
            num_batches_per_epoch=self.num_batches_per_epoch,
        )

    def create_validation_data_loader(
        self,
        data: Dataset,
        module: BaseLightningModule,
        **kwargs,
    ) -> Iterable:
        instances = self._create_instance_splitter(module, "validation").apply(
            data, is_train=True
        )
        return as_stacked_batches(
            instances,
            batch_size=self.batch_size,
            field_names=TRAINING_INPUT_NAMES,
            output_type=torch.tensor,
        )


    def create_lightning_module(self) -> BaseLightningModule:
        return self.lightning_module(
            lr=self.lr,
            weight_decay=self.weight_decay,
            patience=self.patience,
            model_kwargs={
                "freq": self.freq,
                "context_length": self.context_length,
                "prediction_length": self.prediction_length,
                "input_size": self.input_size,
                "num_feat_dynamic_real": (
                    1 + self.num_feat_dynamic_real + len(self.time_features)
                ),
                "num_feat_static_real": max(1, self.num_feat_static_real),
                "num_feat_static_cat": max(1, self.num_feat_static_cat),
                "cardinality": self.cardinality,
                "embedding_dimension": self.embedding_dimension,
                "num_layers": self.num_layers,
                "hidden_size": self.hidden_size,
                "dropout_rate": self.dropout_rate,
                "lags_seq": self.lags_seq,
                "scaling": self.scaling,
                "default_scale": self.default_scale,
                "num_parallel_samples": self.num_parallel_samples,
                "denoising_model": self.denoising_model,
                "velocity_model": self.velocity_model,
                "rnn_model": self.rnn_model,
                **self.additional_model_kwargs,
            },
        )

    def create_predictor(
        self,
        transformation: Transformation,
        module: BaseLightningModule,
    ) -> PyTorchPredictor:
        prediction_splitter = self._create_instance_splitter(module, "test")

        return PyTorchPredictor(
            input_transform=transformation + prediction_splitter,
            input_names=PREDICTION_INPUT_NAMES,
            prediction_net=module,
            batch_size=self.batch_size,
            prediction_length=self.prediction_length,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

