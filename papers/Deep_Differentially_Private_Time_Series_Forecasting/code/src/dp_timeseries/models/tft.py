
from gluonts.torch.distributions.studentT import StudentTOutput
from gluonts.torch.model.tft import (TemporalFusionTransformerEstimator,
                                     TemporalFusionTransformerLightningModule)
from gluonts.transform.split import InstanceSplitter
from lightning import LightningModule
from opacus.grad_sample import GradSampleModule
from opacus.optimizers.optimizer import DPOptimizer

from .dp_estimator import DPPyTorchLightningEstimator
from .utils import create_dp_compatible_layer


class DPTemporalFusionTransformerLightningModule(TemporalFusionTransformerLightningModule):
    """TemporalFusionTransformer with Noisy SGD Optimizer.
    """

    def __init__(
        self,
        model_kwargs: dict,
        dp_optimizer_kwargs: dict,
        lr: float = 1e-3,
        weight_decay: float = 1e-8
    ) -> None:
        """
        Args:
            model_kwargs (dict): Arguments for model underlying the
                TemporalFusionTransformerLightningModule.
                See
                gluonts.torch.model.simple_feedforward.module.TemporalFusionTransformerModel.__init__.
            dp_optimizer_kwargs (dict): Arguments for the Noisy SGD Optimizer.
                See opacus.optimizers.optimizer.DPOptimizer.
            lr (float, optional): Learning rate. Defaults to 1e-3.
            weight_decay (float, optional): Weight decay. Defaults to 1e-8.
        """

        super().__init__(model_kwargs=model_kwargs,
                         lr=lr,
                         weight_decay=weight_decay)

        self.dp_optimizer_kwargs = dp_optimizer_kwargs

        # Normal LSTM uses too much cudnn, does not work with gradient hooks
        original_encoder_lstm = self.model.temporal_encoder.encoder_lstm
        original_decoder_lstm = self.model.temporal_encoder.decoder_lstm
        self.model.temporal_encoder.encoder_lstm = create_dp_compatible_layer(original_encoder_lstm)
        self.model.temporal_encoder.decoder_lstm = create_dp_compatible_layer(original_decoder_lstm)

        # Normal MultiHeadAttention uses too much cudnn, does not work with gradient hooks
        original_attention = self.model.temporal_decoder.attention
        self.model.temporal_decoder.attention = create_dp_compatible_layer(original_attention)

        GradSampleModule(self.model)  # Adds gradient hooks etc. as part of constructor

    def configure_optimizers(self):
        # Get rid of LR scheduler for fair comparison with other models
        base_optimizer = super().configure_optimizers()['optimizer']

        return DPOptimizer(
            base_optimizer,
            **self.dp_optimizer_kwargs)


class NonDPTemporalFusionTransformerLightningModule(TemporalFusionTransformerLightningModule):
    """TemporalFusionTransformer with DPLSTM layers and without LR scheduling.
    """

    def __init__(
        self,
        model_kwargs: dict,
        lr: float = 1e-3,
        weight_decay: float = 1e-8
    ) -> None:
        """
        Args:
            model_kwargs (dict): Arguments for model underlying the
                TemporalFusionTransformerLightningModule.
                See
                gluonts.torch.model.simple_feedforward.module.TemporalFusionTransformerModel.__init__.
                lr (float, optional): Learning rate. Defaults to 1e-3.
            weight_decay (float, optional): Weight decay. Defaults to 1e-8.
        """

        super().__init__(model_kwargs=model_kwargs,
                         lr=lr,
                         weight_decay=weight_decay)

        # Use DPLSTM here, in case there are subtle differences in behavior
        # to normal MultiHeadAttention
        original_encoder_lstm = self.model.temporal_encoder.encoder_lstm
        original_decoder_lstm = self.model.temporal_encoder.decoder_lstm
        self.model.temporal_encoder.encoder_lstm = create_dp_compatible_layer(original_encoder_lstm)
        self.model.temporal_encoder.decoder_lstm = create_dp_compatible_layer(original_decoder_lstm)

        # Use DPMultiHeadAttention here, in case there are subtle differences in behavior
        # to normal MultiHeadAttention
        original_attention = self.model.temporal_decoder.attention
        self.model.temporal_decoder.attention = create_dp_compatible_layer(original_attention)

    def configure_optimizers(self):
        # Get rid of LR scheduler for fair comparison with other models
        return super().configure_optimizers()['optimizer']


class DPTemporalFusionTransformerEstimator(DPPyTorchLightningEstimator,
                                           TemporalFusionTransformerEstimator):

    def __init__(self,
                 top_level_mode: str,
                 instances_per_sequence: int,
                 use_dp_lightning_module: bool,
                 *,
                 dp_optimizer_kwargs: None | dict[str] = None,
                 neighboring_relation: None | dict[str] = None,
                 dp_accountant_kwargs: None | dict[str] = None,
                 tight_privacy_loss: bool = False,
                 lower_bound: bool = False,
                 relative_context_length: None | int = 1,
                 **kwargs) -> None:

        if kwargs.get('train_sampler', None) is not None:
            raise ValueError('DPEstimators do not support custom train samplers')

        if kwargs.get('validation_sampler', None) is not None:
            raise ValueError('DPEstimators do not support custom train samplers')

        TemporalFusionTransformerEstimator.__init__(self, **kwargs)

        DPPyTorchLightningEstimator.__init__(self,
                                             top_level_mode,
                                             instances_per_sequence,
                                             self.batch_size,
                                             use_dp_lightning_module,
                                             self.trainer_kwargs,
                                             dp_optimizer_kwargs=dp_optimizer_kwargs,
                                             neighboring_relation=neighboring_relation,
                                             dp_accountant_kwargs=dp_accountant_kwargs,
                                             tight_privacy_loss=tight_privacy_loss,
                                             lower_bound=lower_bound)

        if relative_context_length is not None:
            self.context_length = relative_context_length * self.prediction_length

    def _create_instance_splitter(
            self, *args) -> InstanceSplitter:
        # TemporalFusionTransformerEstimator calls with only mode
        # While DPEstimator calls with module and mode
        # We want to extract mode from both cases
        return TemporalFusionTransformerEstimator._create_instance_splitter(self, args[-1])

    def create_lightning_module(self) -> LightningModule:
        assert isinstance(self.distr_output, StudentTOutput)

        if not self.use_dp_lightning_module:
            return NonDPTemporalFusionTransformerLightningModule(
                lr=self.lr,
                weight_decay=self.weight_decay,
                model_kwargs={
                    "context_length": self.context_length,
                    "prediction_length": self.prediction_length,
                    "d_var": self.variable_dim,
                    "d_hidden": self.hidden_dim,
                    "num_heads": self.num_heads,
                    "distr_output": self.distr_output,
                    "d_past_feat_dynamic_real": self.past_dynamic_dims,
                    "c_past_feat_dynamic_cat": self.past_dynamic_cardinalities,
                    "d_feat_dynamic_real": [1] * max(len(self.time_features), 1)
                    + self.dynamic_dims,
                    "c_feat_dynamic_cat": self.dynamic_cardinalities,
                    "d_feat_static_real": self.static_dims or [1],
                    "c_feat_static_cat": self.static_cardinalities or [1],
                    "dropout_rate": self.dropout_rate,
                })
        else:
            return DPTemporalFusionTransformerLightningModule(
                lr=self.lr,
                weight_decay=self.weight_decay,
                model_kwargs={
                    "context_length": self.context_length,
                    "prediction_length": self.prediction_length,
                    "d_var": self.variable_dim,
                    "d_hidden": self.hidden_dim,
                    "num_heads": self.num_heads,
                    "distr_output": self.distr_output,
                    "d_past_feat_dynamic_real": self.past_dynamic_dims,
                    "c_past_feat_dynamic_cat": self.past_dynamic_cardinalities,
                    "d_feat_dynamic_real": [1] * max(len(self.time_features), 1)
                    + self.dynamic_dims,
                    "c_feat_dynamic_cat": self.dynamic_cardinalities,
                    "d_feat_static_real": self.static_dims or [1],
                    "c_feat_static_cat": self.static_cardinalities or [1],
                    "dropout_rate": self.dropout_rate,
                },
                dp_optimizer_kwargs=self.dp_optimizer_kwargs)
