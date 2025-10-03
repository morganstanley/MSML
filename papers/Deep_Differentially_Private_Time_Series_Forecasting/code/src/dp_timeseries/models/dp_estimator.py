import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Optional

import dp_timeseries.privacy.bilevel_subsampling as bilevel_subsampling
import lightning.pytorch as pl
import numpy as np
from dp_accounting.pld.privacy_loss_mechanism import AdditiveNoisePrivacyLoss
from dp_timeseries.data.loaders import (NonCyclicTrainDataLoader,
                                        WithoutReplacementTrainDataLoader)
from dp_timeseries.evaluation import EvaluateCallback
from dp_timeseries.privacy.accounting import PLDAccountant
from dp_timeseries.transformations.additive_noise import AddGaussianNoise
from gluonts.dataset import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import TrainDataLoader
from gluonts.env import env
from gluonts.itertools import Cached
from gluonts.model import Predictor
from gluonts.torch import PyTorchPredictor
from gluonts.torch.batchify import batchify
from gluonts.torch.model.estimator import (PyTorchLightningEstimator,
                                           TrainOutput)
from gluonts.transform._base import Chain, Identity
from gluonts.transform.sampler import InstanceSampler, NumInstanceSampler
from gluonts.transform.split import InstanceSplitter
from lightning import LightningModule

logger = logging.getLogger(__name__)


class DPPyTorchLightningEstimator(PyTorchLightningEstimator, ABC):
    """PytorchLightningEstimator that is trained using DP-SGD with a privacy budget.

    Estimators are a concept from GluonTS that bundles
        - data preprocessing
        - a model
        - a training procedure for the model
    and can be used to create an inference procedure "Predictor" from the trained model.

    See https://ts.gluon.ai/stable/tutorials/forecasting/quick_start_tutorial.html.

    Just like normal GluonTS Estimators, all of this is a hacky mess.
    But it can be used to make arbitrary estimators DP with some boiler plate code.
    See other modules in dp_timeseries/models.
    """

    def __init__(self,
                 top_level_mode: str,
                 instances_per_sequence: int,
                 batch_size: int,
                 use_dp_lightning_module: bool,
                 trainer_kwargs: dict[str],
                 lead_time: int = 0,
                 dp_optimizer_kwargs: None | dict[str] = None,
                 neighboring_relation: None | dict[str] = None,
                 dp_accountant_kwargs: None | dict[str] = None,
                 tight_privacy_loss: bool = False,
                 lower_bound: bool = False) -> None:
        """Prepares DP sampling procedure and privacy accountant.

        Args:
            top_level_mode (str): Top-level subsampling scheme to use. Should be in:
                - "iteration" (Algorithm 3 in paper)
                - "shuffling" (In each epoch, shuffle train set before iterating)
                - "sampling_without_replacement" (Algorithm 5 in paper)
            instances_per_sequence (int): _description_
            batch_size (int): Number of subsequences in a batch (Lambda)
            use_dp_lightning_module (bool): Keep this at True for DP training.
            trainer_kwargs (dict[str]): args to be passed to lightning Trainer
            lead_time (int, optional): Gap between context and forecast window.
                Defaults to 0.
            dp_optimizer_kwargs (None | dict[str]): kwargs to be passed to opacus DPOptimizer.
                Should have keys "noise_multiplier" and "max_grad_norm".
            neighboring_relation (None | dict[str]): Considered neighboring relation
                - "level": "event" or "user"
                - "size": Number of modified elements, ("w" in paper)
                - "target_sensitivity": Maximum absolute change in modified elements ("v" in paper)
            dp_accountant_kwargs (None | dict[str]): kwargs to be passed to accounting.PLDAccountant
                Should specifiy "budget_epsilon" and "budget_delta".
                None will disable privacy accounting.
            tight_privacy_loss (bool): If False, will use loose upper bound from Lemma F.4.
            lower_bound (bool, optional): If True, compute optimistic lower bound on privacy.
                Defaults to False.
        """

        check_val_pred_every_n_epoch = self.trainer_kwargs.pop(
            'check_val_pred_every_n_epoch', None)

        PyTorchLightningEstimator.__init__(self, trainer_kwargs, lead_time)

        self.top_level_mode = top_level_mode
        self.instances_per_sequence = instances_per_sequence
        self.batch_size = batch_size
        self.use_dp_lightning_module = use_dp_lightning_module
        self.dp_optimizer_kwargs = dp_optimizer_kwargs
        self.neighboring_relation = neighboring_relation
        self.dp_accountant_kwargs = dp_accountant_kwargs
        self.tight_privacy_loss = tight_privacy_loss
        self.lower_bound = lower_bound

        self.check_val_pred_every_n_epoch = check_val_pred_every_n_epoch

        self.dp_optimizer_kwargs = deepcopy(self.dp_optimizer_kwargs)

        if self.dp_optimizer_kwargs is not None:
            self.dp_optimizer_kwargs['expected_batch_size'] = self.batch_size
            self.future_target_noise_multiplier = self.dp_optimizer_kwargs.pop(
                'future_target_noise_multiplier', 0.0)
        else:
            self.future_target_noise_multiplier = 0.0

        if (self.batch_size % self.instances_per_sequence) != 0:
            raise ValueError('batch_size must be divisible by instances_per_sequence')

        self.train_sampler = self.create_train_sampler()

    def create_train_sampler(self) -> InstanceSampler:
        return NumInstanceSampler(
            min_past=0,  # i.e. is equivalent to padding of length context_length
            min_future=self.prediction_length,
            N=self.instances_per_sequence)

    @abstractmethod
    def _create_instance_splitter(
            self, module: LightningModule, mode: str) -> InstanceSplitter:

        raise NotImplementedError

    def create_training_data_loader(
            self,
            data: Dataset,
            module: LightningModule,
            shuffle_buffer_length: None | int = None,
            **kwargs) -> TrainDataLoader:

        # We are creating batches from a cyclic iterator
        # To avoid duplicate access, round down number of iterations
        num_batches_per_epoch = (len(data) * self.instances_per_sequence) // self.batch_size

        if num_batches_per_epoch == 0:
            raise ValueError('len(data) * instances_per_sequence not large enough for batch size')

        transform = self._create_instance_splitter(module, "training")

        # Noise for amplification by label perturbation
        if self.neighboring_relation is not None:
            future_sensitivity = self.neighboring_relation.get('future_target_sensitivity', np.inf)

            if not np.isinf(future_sensitivity) and self.future_target_noise_multiplier > 0:
                transform = transform + AddGaussianNoise(
                    target_field=f'future_{FieldName.TARGET}',
                    observed_values_field=f'future_{FieldName.OBSERVED_VALUES}',
                    is_pad_field=None,
                    standard_deviation=(future_sensitivity * self.future_target_noise_multiplier)
                )

        if self.top_level_mode in ['iteration', 'shuffling']:
            if self.top_level_mode == 'shuffling':
                if shuffle_buffer_length is None:
                    shuffle_buffer_length = len(data)
            else:
                shuffle_buffer_length = None

            loader = NonCyclicTrainDataLoader(
                data,
                transform=transform,
                batch_size=self.batch_size,
                stack_fn=batchify,
                num_batches_per_epoch=num_batches_per_epoch,
                shuffle_buffer_length=shuffle_buffer_length
            )

        elif self.top_level_mode == 'sampling_without_replacement':
            # How many sequences we sample at top level
            sample_size = self.batch_size // self.instances_per_sequence

            loader = WithoutReplacementTrainDataLoader(
                data,
                transform=transform,
                sample_size=sample_size,
                batch_size=self.batch_size,
                stack_fn=batchify,
                num_batches_per_epoch=num_batches_per_epoch,
            )

        else:
            raise ValueError('top_level_mode must be in '
                             '["iteration", "shuffling", "sampling_without_replacement"]')

        return loader

    @staticmethod
    def _series_length(ts: dict) -> int:
        target = ts['target']
        if isinstance(target, np.ndarray):
            return target.shape[-1]
        else:
            return len(target)

    def create_privacy_loss(
                self,
                module: LightningModule,
                num_sequences: int,
                min_sequence_length: int
            ) -> AdditiveNoisePrivacyLoss:
        """Privacy loss of module characterizing leakage per iteration/epoch.

        Args:
            module (LightningModule): Model that is trained.
                Its context length etc. are indirectly used to obtain instance splitter.
            num_sequences (int): Number of sequences in dataset.
            min_sequence_length (int): Minimum sequence length within dataset.

        Returns:
            AdditiveNoisePrivacyLoss: Privacy loss.
                Should be wrapped into PrivacyLossPMF and then PrivacyLossDistribution
                from opacus for accounting.
        """

        splitter = self._create_instance_splitter(module, "training")

        if isinstance(splitter, Chain):
            splitter = splitter.transformations[0]

        past_length = splitter.past_length  # Used for padding
        future_length = splitter.future_length  # Does not introduce padding
        lead_time = splitter.lead_time

        sampler = splitter.instance_sampler

        if not isinstance(sampler, NumInstanceSampler):
            raise NotImplementedError('Accounting only supports NumInstanceSampler.')

        min_past = sampler.min_past  # Start of range where target windows can start
        min_future = sampler.min_future  # Negative end of range where target windows can start

        return bilevel_subsampling.create_privacy_loss(
            num_sequences, min_sequence_length,
            self.top_level_mode, sampler.N, self.batch_size,
            past_length, future_length, lead_time,
            min_past, min_future,
            self.dp_optimizer_kwargs['noise_multiplier'],
            self.neighboring_relation,
            self.tight_privacy_loss,
            self.future_target_noise_multiplier,
            lower_bound=self.lower_bound)

    @property
    def privacy_loss_is_epoch_level(self) -> bool:
        """Whether privacy loss describes epoch or iteration-level leakage.

        Returns:
            bool: True means epoch-level leakage.
        """
        return self.top_level_mode in ['iteration', 'shuffling']

    def train_model(
        self,
        training_data: Dataset,
        validation_data: Optional[Dataset] = None,
        from_predictor: Optional[PyTorchPredictor] = None,
        shuffle_buffer_length: Optional[int] = None,
        cache_data: bool = False,
        ckpt_path: Optional[str] = None,
        validation_pred_data: Optional[Dataset] = None,
        **kwargs,
    ) -> TrainOutput:
        transformation = self.create_transformation()

        # Prepare training data loader
        with env._let(max_idle_transforms=max(len(training_data), 100)):
            transformed_training_data: Dataset = transformation.apply(
                training_data, is_train=True
            )
            if cache_data:
                transformed_training_data = Cached(transformed_training_data)

            training_network = self.create_lightning_module()

            training_data_loader = self.create_training_data_loader(
                transformed_training_data,
                training_network,
                shuffle_buffer_length=shuffle_buffer_length,
            )

        validation_data_loader = None

        # Prepare validation data loader
        if validation_data is not None:
            with env._let(max_idle_transforms=max(len(validation_data), 100)):
                transformed_validation_data: Dataset = transformation.apply(
                    validation_data, is_train=True
                )
                if cache_data:
                    transformed_validation_data = Cached(
                        transformed_validation_data
                    )

                validation_data_loader = self.create_validation_data_loader(
                    transformed_validation_data,
                    training_network,
                )

        if validation_pred_data is not None:
            with env._let(max_idle_transforms=max(len(validation_pred_data), 100)):
                transformed_validation_pred_data: Dataset = transformation.apply(
                    validation_pred_data, is_train=False
                )
                if cache_data:
                    transformed_validation_pred_data = Cached(
                        transformed_validation_pred_data
                    )

                # Apply make_evaluation_predictions to validation data
                # Normally GluonTS would only give you the loss value
                eval_callback = EvaluateCallback(
                    transformed_validation_pred_data,
                    transformation=Identity(),
                    estimator=self,
                    eval_every_n_epoch=self.check_val_pred_every_n_epoch)

        if from_predictor is not None:
            training_network.load_state_dict(
                from_predictor.network.state_dict()
            )

        # Prepare logging
        monitor = "train_loss" if validation_data is None else "val_loss"
        checkpoint = pl.callbacks.ModelCheckpoint(
            monitor=monitor, mode="min", verbose=True
        )

        custom_callbacks = self.trainer_kwargs.pop("callbacks", [])

        if validation_pred_data is not None:
            custom_callbacks.append(eval_callback)

        # Set up privacy accounting
        if self.dp_accountant_kwargs is not None:
            if not self.use_dp_lightning_module:
                raise ValueError('Accounting only valid for DP lightning modules.')

            min_sequence_length = min(self._series_length(x) for x in transformed_training_data)

            privacy_loss = self.create_privacy_loss(training_network,
                                                    len(transformed_training_data),
                                                    min_sequence_length)

            accountant = PLDAccountant(
                self.privacy_loss_is_epoch_level,
                privacy_loss,
                **self.dp_accountant_kwargs)

            custom_callbacks.append(accountant)

        trainer = pl.Trainer(
            **{
                "accelerator": "auto",
                "callbacks": [checkpoint] + custom_callbacks,
                **self.trainer_kwargs,
            }
        )

        # Train
        trainer.fit(
            model=training_network,
            train_dataloaders=training_data_loader,
            val_dataloaders=validation_data_loader,
            ckpt_path=ckpt_path,
        )

        # Load best model
        if checkpoint.best_model_path != "":
            logger.info(
                f"Loading best model from {checkpoint.best_model_path}"
            )

            # With DP modules, we also need to recover the optimizer state
            if self.use_dp_lightning_module:
                best_model = training_network.__class__.load_from_checkpoint(
                    checkpoint.best_model_path,
                    dp_optimizer_kwargs=self.dp_optimizer_kwargs)

            else:
                best_model = training_network.__class__.load_from_checkpoint(
                    checkpoint.best_model_path)

        else:
            best_model = training_network

        return TrainOutput(
            transformation=transformation,
            trained_net=best_model,
            trainer=trainer,
            predictor=self.create_predictor(transformation, best_model),
        )

    def train(
        self,
        training_data: Dataset,
        validation_data: Optional[Dataset] = None,
        shuffle_buffer_length: Optional[int] = None,
        cache_data: bool = False,
        ckpt_path: Optional[str] = None,
        validation_pred_data: Optional[Dataset] = None,
        **kwargs,
    ) -> PyTorchPredictor:
        return self.train_model(
            training_data,
            validation_data,
            shuffle_buffer_length=shuffle_buffer_length,
            cache_data=cache_data,
            ckpt_path=ckpt_path,
            validation_pred_data=validation_pred_data
        ).predictor

    def train_from(
        self,
        predictor: Predictor,
        training_data: Dataset,
        validation_data: Optional[Dataset] = None,
        shuffle_buffer_length: Optional[int] = None,
        cache_data: bool = False,
        ckpt_path: Optional[str] = None,
        validation_pred_data: Optional[Dataset] = None,
    ) -> PyTorchPredictor:
        assert isinstance(predictor, PyTorchPredictor)
        return self.train_model(
            training_data,
            validation_data,
            from_predictor=predictor,
            shuffle_buffer_length=shuffle_buffer_length,
            cache_data=cache_data,
            ckpt_path=ckpt_path,
            validation_pred_data=validation_pred_data
        ).predictor
