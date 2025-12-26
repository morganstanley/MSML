from typing import Any

import numpy as np
from dp_accounting.pld.privacy_loss_distribution import (
    PrivacyLossDistribution, _create_pld_pmf_from_additive_noise)
from dp_accounting.pld.privacy_loss_mechanism import AdditiveNoisePrivacyLoss
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback


class PLDAccountant(Callback):
    """Numeric privacy accountant that can terminate lightning trainer early.
    """

    def __init__(self,
                 epoch_level: bool,
                 privacy_loss: AdditiveNoisePrivacyLoss,
                 budget_epsilon: float,
                 budget_delta: float,
                 value_discretization_interval: float = 1e-3,
                 use_connect_dots: bool = True):
        """"
        Args:
            epoch_level (False): If True, compose once per epoch.
                If False, compose once per iteration.
            privacy_loss (AdditiveNoisePrivacyLoss): Privacy loss characterizing
                one iteration / epoch.
            budget_epsilon (float): Epsilon at which training will be terminated.
            budget_delta (float): Constant delta to use in computing epsilon.
            value_discretization_interval (float, optional): Resolution of pld quantization.
                Defaults to 1e-3.
            use_connect_dots (bool, optional): Whether to use "connect the dots", 
                i.e., linear interpolation of pld, in quantizing pld.
                Defaults to True.
        """

        super().__init__()

        self.epoch_level = epoch_level
        self.budget_epsilon = budget_epsilon
        self.budget_delta = budget_delta

        if not np.isinf(self.budget_epsilon):
            pld_pmf = _create_pld_pmf_from_additive_noise(
                privacy_loss,
                value_discretization_interval=value_discretization_interval,
                use_connect_dots=use_connect_dots)

            self.pld = PrivacyLossDistribution(pld_pmf)

        else:
            self.pld = None

        self.composed_pld = self.pld

        self.current_epsilon = 0.0

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self.epoch_level:
            return
        self.accounting_step(trainer, pl_module)

    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule,
                             batch: Any, batch_idx: int) -> None:
        if self.epoch_level:
            return
        self.accounting_step(trainer, pl_module)

    def accounting_step(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Computes accumulated epsilon for next iteration/epoch. terminates training if needed.

        Args:
            trainer (Trainer): Trainer to be potentially terminated.
            pl_module (LightningModule): Lightning module that is being trained.
        """
        next_epsilon = self.calc_next_epsilon()

        if next_epsilon > self.budget_epsilon:
            trainer.should_stop = True
        else:
            self.current_epsilon = next_epsilon
            pl_module.log('train_epsilon', self.current_epsilon)

    def calc_next_epsilon(self) -> float:
        """Computes accumulated epsilon for next iteration/epoch.

        Returns:
            float: Next epsilon(budget_delta).
        """
        if np.isinf(self.budget_delta):
            return 0.0

        epsilon = self.composed_pld.get_epsilon_for_delta(self.budget_delta)
        # Compose after computing epsilon
        # because initial pld already specifies next_epsilon for first iteration
        self.composed_pld = self.composed_pld.compose(self.pld)
        return epsilon
