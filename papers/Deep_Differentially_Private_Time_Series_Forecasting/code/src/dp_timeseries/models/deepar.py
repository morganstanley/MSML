
from gluonts.torch.model.deepar import DeepAREstimator, DeepARLightningModule
from gluonts.transform.split import InstanceSplitter
from lightning import LightningModule
from opacus.grad_sample import GradSampleModule
from opacus.optimizers.optimizer import DPOptimizer

from .dp_estimator import DPPyTorchLightningEstimator
from .utils import create_dp_compatible_layer


class DPDeepARLightningModule(DeepARLightningModule):
    """DeepAR with Noisy SGD Optimizer.
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
            model_kwargs (dict): Arguments for model underlying the DeepARLightningModule.
                See gluonts.torch.model.simple_feedforward.module.DeepARModel.__init__.
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
        original_lstm = self.model.rnn
        self.model.rnn = create_dp_compatible_layer(original_lstm)

        GradSampleModule(self.model)  # Adds gradient hooks etc. as part of constructor

    def configure_optimizers(self):
        # Get rid of LR scheduler for fair comparison with other models
        base_optimizer = super().configure_optimizers()['optimizer']

        return DPOptimizer(
            base_optimizer,
            **self.dp_optimizer_kwargs)


class NonDPDeepARLightningModule(DeepARLightningModule):
    """DeepAR with DPLSTM layers and without LR scheduling.
    """

    def __init__(
        self,
        model_kwargs: dict,
        lr: float = 1e-3,
        weight_decay: float = 1e-8
    ) -> None:
        """
        Args:
            model_kwargs (dict): Arguments for model underlying the DeepARLightningModule.
                See gluonts.torch.model.simple_feedforward.module.DeepARModel.__init__.
            lr (float, optional): Learning rate. Defaults to 1e-3.
            weight_decay (float, optional): Weight decay. Defaults to 1e-8.
        """

        super().__init__(model_kwargs=model_kwargs,
                         lr=lr,
                         weight_decay=weight_decay)

        # Use DPLSTM here, in case there are subtle differences in behavior to nomal LSTM
        original_lstm = self.model.rnn
        self.model.rnn = create_dp_compatible_layer(original_lstm)

    def configure_optimizers(self):
        # Get rid of LR scheduler for fair comparison with other models
        return super().configure_optimizers()['optimizer']


class DPDeepAREstimator(DPPyTorchLightningEstimator, DeepAREstimator):

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

        DeepAREstimator.__init__(self, **kwargs)

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
            self, module: LightningModule, mode: str) -> InstanceSplitter:
        return DeepAREstimator._create_instance_splitter(self, module, mode)

    def create_lightning_module(self) -> LightningModule:
        if not self.use_dp_lightning_module:
            return NonDPDeepARLightningModule(
                lr=self.lr,
                weight_decay=self.weight_decay,
                model_kwargs={
                    "freq": self.freq,
                    "context_length": self.context_length,
                    "prediction_length": self.prediction_length,
                    "num_feat_dynamic_real": (
                        1 + self.num_feat_dynamic_real + len(self.time_features)
                    ),
                    "num_feat_static_real": max(1, self.num_feat_static_real),
                    "num_feat_static_cat": max(1, self.num_feat_static_cat),
                    "cardinality": self.cardinality,
                    "embedding_dimension": self.embedding_dimension,
                    "num_layers": self.num_layers,
                    "hidden_size": self.hidden_size,
                    "distr_output": self.distr_output,
                    "dropout_rate": self.dropout_rate,
                    "lags_seq": self.lags_seq,
                    "scaling": self.scaling,
                    "default_scale": self.default_scale,
                    "num_parallel_samples": self.num_parallel_samples,
                    "nonnegative_pred_samples": self.nonnegative_pred_samples,
                })
        else:
            return DPDeepARLightningModule(
                lr=self.lr,
                weight_decay=self.weight_decay,
                model_kwargs={
                    "freq": self.freq,
                    "context_length": self.context_length,
                    "prediction_length": self.prediction_length,
                    "num_feat_dynamic_real": (
                        1 + self.num_feat_dynamic_real + len(self.time_features)
                    ),
                    "num_feat_static_real": max(1, self.num_feat_static_real),
                    "num_feat_static_cat": max(1, self.num_feat_static_cat),
                    "cardinality": self.cardinality,
                    "embedding_dimension": self.embedding_dimension,
                    "num_layers": self.num_layers,
                    "hidden_size": self.hidden_size,
                    "distr_output": self.distr_output,
                    "dropout_rate": self.dropout_rate,
                    "lags_seq": self.lags_seq,
                    "scaling": self.scaling,
                    "default_scale": self.default_scale,
                    "num_parallel_samples": self.num_parallel_samples,
                    "nonnegative_pred_samples": self.nonnegative_pred_samples,
                },
                dp_optimizer_kwargs=self.dp_optimizer_kwargs)
