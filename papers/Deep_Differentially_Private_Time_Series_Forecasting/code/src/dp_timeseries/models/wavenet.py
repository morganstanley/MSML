
from typing import Iterable

import torch
from gluonts.dataset import Dataset
from gluonts.dataset.loader import as_stacked_batches
from gluonts.torch.model.wavenet import (WaveNetEstimator,
                                         WaveNetLightningModule)
from gluonts.torch.model.wavenet.estimator import \
    TRAINING_INPUT_NAMES as WAVENET_TRAINING_INPUT_NAMES
from gluonts.transform.split import InstanceSplitter
from lightning import LightningModule
from opacus.grad_sample import GradSampleModule
from opacus.optimizers.optimizer import DPOptimizer

from .dp_estimator import DPPyTorchLightningEstimator


class DPWaveNetLightningModule(WaveNetLightningModule):
    """WaveNetModule with Noisy SGD Optimizer.
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
            model_kwargs (dict): Arguments for model underlying the WaveNetLightningModule.
                See gluonts.torch.model.wavenet.module.WaveNetModel.__init__.
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


class DPWaveNetEstimator(DPPyTorchLightningEstimator, WaveNetEstimator):

    def __init__(self,
                 top_level_mode,
                 instances_per_sequence: int,
                 use_dp_lightning_module: bool,
                 *,
                 dp_optimizer_kwargs: None | dict[str] = None,
                 neighboring_relation: None | dict[str] = None,
                 dp_accountant_kwargs: None | dict[str] = None,
                 tight_privacy_loss: bool = False,
                 lower_bound: bool = False,
                 **kwargs) -> None:

        if kwargs.get('train_sampler', None) is not None:
            raise ValueError('DPEstimators do not support custom train samplers')

        if kwargs.get('validation_sampler', None) is not None:
            raise ValueError('DPEstimators do not support custom train samplers')

        WaveNetEstimator.__init__(self, **kwargs)

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

    def _create_instance_splitter(
            self, *args) -> InstanceSplitter:
        # WaveNetEstimator calls with only mode
        # While DPEstimator calls with module and mode
        # We want to extract mode from both cases
        return WaveNetEstimator._create_instance_splitter(self, args[-1])

    def create_lightning_module(self) -> LightningModule:
        if not self.use_dp_lightning_module:
            return super().create_lightning_module()
        else:
            return DPWaveNetLightningModule(
                lr=self.lr,
                weight_decay=self.weight_decay,
                model_kwargs=dict(
                    bin_values=self.bin_centers,
                    num_residual_channels=self.num_residual_channels,
                    num_skip_channels=self.num_skip_channels,
                    dilation_depth=self.dilation_depth,
                    num_stacks=self.num_stacks,
                    num_feat_dynamic_real=1
                    + self.num_feat_dynamic_real
                    + len(self.time_features),
                    num_feat_static_real=max(1, self.num_feat_static_real),
                    cardinality=self.cardinality,
                    embedding_dimension=self.embedding_dimension,
                    pred_length=self.prediction_length,
                    num_parallel_samples=self.num_parallel_samples,
                    temperature=self.temperature,
                    use_log_scale_feature=self.use_log_scale_feature
                ),
                dp_optimizer_kwargs=self.dp_optimizer_kwargs)
