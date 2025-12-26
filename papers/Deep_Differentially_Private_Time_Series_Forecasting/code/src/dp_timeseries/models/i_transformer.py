
from gluonts.dataset.field_names import FieldName
from gluonts.torch.model.i_transformer import (ITransformerEstimator,
                                               ITransformerLightningModule)
from gluonts.transform import (AddObservedValuesIndicator, AsNumpyArray,
                               ExpandDimArray, SelectFields, Transformation)
from gluonts.transform.split import InstanceSplitter
from lightning import LightningModule
from opacus.grad_sample import GradSampleModule
from opacus.optimizers.optimizer import DPOptimizer

from .dp_estimator import DPPyTorchLightningEstimator
from .utils import create_dp_compatible_layer


class DPITransformerLightningModule(ITransformerLightningModule):
    """ITransformer with Noisy SGD Optimizer.
    """

    def __init__(
        self,
        model_kwargs: dict,
        dp_optimizer_kwargs: dict,
        num_parallel_samples: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 1e-8
    ) -> None:
        """
        Args:
            model_kwargs (dict): Arguments for model underlying the
                ITransformerLightningModule.
                See
                gluonts.torch.model.simple_feedforward.module.ITransformerModel.__init__.
            dp_optimizer_kwargs (dict): Arguments for the Noisy SGD Optimizer.
                See opacus.optimizers.optimizer.DPOptimizer.
            lr (float, optional): Learning rate. Defaults to 1e-3.
            weight_decay (float, optional): Weight decay. Defaults to 1e-8.
        """

        super().__init__(model_kwargs=model_kwargs,
                         num_parallel_samples=num_parallel_samples,
                         lr=lr,
                         weight_decay=weight_decay)

        self.dp_optimizer_kwargs = dp_optimizer_kwargs

        # Normal Encoder uses too much cudnn, does not work with gradient hooks
        original_encoder = self.model.encoder
        self.model.encoder = create_dp_compatible_layer(original_encoder)

        GradSampleModule(self.model)  # Adds gradient hooks etc. as part of constructor

    def configure_optimizers(self):
        base_optimizer = super().configure_optimizers()

        return DPOptimizer(
            base_optimizer,
            **self.dp_optimizer_kwargs)


class NonDPITransformerLightningModule(ITransformerLightningModule):
    """ITransformer with DPLSTM layers and without LR scheduling.
    """

    def __init__(
        self,
        model_kwargs: dict,
        num_parallel_samples: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 1e-8
    ) -> None:
        """
        Args:
            model_kwargs (dict): Arguments for model underlying the
                ITransformerLightningModule.
                See
                gluonts.torch.model.simple_feedforward.module.ITransformerModel.__init__.
                lr (float, optional): Learning rate. Defaults to 1e-3.
            weight_decay (float, optional): Weight decay. Defaults to 1e-8.
        """

        super().__init__(model_kwargs=model_kwargs,
                         num_parallel_samples=num_parallel_samples,
                         lr=lr,
                         weight_decay=weight_decay)

        # Use DPMultiHeadAttention here, in case there are subtle differences in behavior
        # to normal MultiHeadAttention
        original_encoder = self.model.encoder
        self.model.encoder = create_dp_compatible_layer(original_encoder)


class DPITransformerEstimator(DPPyTorchLightningEstimator,
                              ITransformerEstimator):

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

        ITransformerEstimator.__init__(self, **kwargs)

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

    def create_transformation(self) -> Transformation:
        # We overrode this this to make model work for univariate forecasting
        return (
            SelectFields(
                [
                    FieldName.ITEM_ID,
                    FieldName.INFO,
                    FieldName.START,
                    FieldName.TARGET,
                ],
                allow_missing=True,
            )
            + AsNumpyArray(field=FieldName.TARGET, expected_ndim=1)
            + ExpandDimArray(field=FieldName.TARGET, axis=0)
            + AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            )
        )

    def _create_instance_splitter(
            self, module: LightningModule, mode: str) -> InstanceSplitter:
        return ITransformerEstimator._create_instance_splitter(self, module, mode)

    def create_lightning_module(self) -> LightningModule:

        if not self.use_dp_lightning_module:
            return NonDPITransformerLightningModule(
                lr=self.lr,
                weight_decay=self.weight_decay,
                num_parallel_samples=self.num_parallel_samples,
                model_kwargs={
                    "prediction_length": self.prediction_length,
                    "context_length": self.context_length,
                    "d_model": self.d_model,
                    "nhead": self.nhead,
                    "dim_feedforward": self.dim_feedforward,
                    "dropout": self.dropout,
                    "activation": self.activation,
                    "norm_first": self.norm_first,
                    "num_encoder_layers": self.num_encoder_layers,
                    "distr_output": self.distr_output,
                    "scaling": self.scaling,
                    "nonnegative_pred_samples": self.nonnegative_pred_samples,
                    })
        else:
            return DPITransformerLightningModule(
                lr=self.lr,
                weight_decay=self.weight_decay,
                num_parallel_samples=self.num_parallel_samples,
                model_kwargs={
                    "prediction_length": self.prediction_length,
                    "context_length": self.context_length,
                    "d_model": self.d_model,
                    "nhead": self.nhead,
                    "dim_feedforward": self.dim_feedforward,
                    "dropout": self.dropout,
                    "activation": self.activation,
                    "norm_first": self.norm_first,
                    "num_encoder_layers": self.num_encoder_layers,
                    "distr_output": self.distr_output,
                    "scaling": self.scaling,
                    "nonnegative_pred_samples": self.nonnegative_pred_samples,
                },
                dp_optimizer_kwargs=self.dp_optimizer_kwargs)
