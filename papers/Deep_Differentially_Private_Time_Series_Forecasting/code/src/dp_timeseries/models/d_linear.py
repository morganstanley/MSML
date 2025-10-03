
from gluonts.torch.model.d_linear import (DLinearEstimator,
                                          DLinearLightningModule)
from gluonts.transform.split import InstanceSplitter
from lightning import LightningModule
from opacus.grad_sample import GradSampleModule
from opacus.optimizers.optimizer import DPOptimizer

from .dp_estimator import DPPyTorchLightningEstimator


class DPDLinearLightningModule(DLinearLightningModule):
    """DLinearModule with Noisy SGD Optimizer.
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
            model_kwargs (dict): Arguments for model underlying the DLinearLightningModule.
                See gluonts.torch.model.d_linear.module.DLinearModel.__init__.
            dp_optimizer_kwargs (dict): Arguments for the Noisy SGD Optimizer.
                See opacus.optimizers.optimizer.DPOptimizer.
            lr (float, optional): Learning rate. Defaults to 1e-3.
            weight_decay (float, optional): Weight decay. Defaults to 1e-8.
        """

        super().__init__(model_kwargs=model_kwargs,
                         lr=lr,
                         weight_decay=weight_decay)

        self.dp_optimizer_kwargs = dp_optimizer_kwargs
        GradSampleModule(self.model)  # Adds gradient hooks etc. as part of constructor

    def configure_optimizers(self):
        base_optimizer = super().configure_optimizers()

        return DPOptimizer(
            base_optimizer,
            **self.dp_optimizer_kwargs)


class DPDLinearEstimator(DPPyTorchLightningEstimator, DLinearEstimator):

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
                 relative_context_length: None | int = 10,
                 **kwargs) -> None:

        if kwargs.get('train_sampler', None) is not None:
            raise ValueError('DPEstimators do not support custom train samplers')

        if kwargs.get('validation_sampler', None) is not None:
            raise ValueError('DPEstimators do not support custom train samplers')

        DLinearEstimator.__init__(self, **kwargs)

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
        return DLinearEstimator._create_instance_splitter(self, module, mode)

    def create_lightning_module(self) -> LightningModule:
        if not self.use_dp_lightning_module:
            return super().create_lightning_module()
        else:
            return DPDLinearLightningModule(
                lr=self.lr,
                weight_decay=self.weight_decay,
                model_kwargs={
                    "prediction_length": self.prediction_length,
                    "context_length": self.context_length,
                    "hidden_dimension": self.hidden_dimension,
                    "distr_output": self.distr_output,
                    "kernel_size": self.kernel_size,
                    "scaling": self.scaling},
                dp_optimizer_kwargs=self.dp_optimizer_kwargs)
