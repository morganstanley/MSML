from copy import deepcopy

import wandb
from dp_timeseries.data import GluonTSLightningDataModule
from dp_timeseries.evaluation import eval_predictor
from dp_timeseries.models.utils import create_dp_estimator
from gluonts.dataset.common import MetaData
from gluonts.torch.distributions.studentT import StudentTOutput
from gluonts.torch.model.estimator import PyTorchLightningEstimator
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.loggers.logger import Logger

from .utils import seed_everything


def standard_train_standard_eval(
        seed: int,
        save_dir: str,
        experiment_name: str,
        dataset_kwargs: dict[str],
        estimator_name: str,
        top_level_mode: str,
        shuffle_buffer_length: None | int,
        instances_per_sequence: int,
        estimator_kwargs: dict[str],
        use_wandb_logger: bool = False,
        wandb_project_name: None | str = None
        ) -> tuple[str, dict[str], dict[str]]:
    """Trains a model without privacy, then does inference without adding input noise

    Args:
        seed (int): Random seed for experiment
        save_dir (str): Directory in which logs should be stored
        experiment_name (str): Name of experiment for lightning loggers
        dataset_kwargs (dict[str]): kwargs for GluonTSLightningDataModule that loads data.
        estimator_name (str): Class name of GluonTS estimator to use.
        top_level_mode (str): Top-level subsampling scheme to use. Should be in:
            - "iteration" (Algorithm 3 in paper)
            - "shuffling" (In each epoch, shuffle train set before iterating)
            - "sampling_without_replacement" (Algorithm 5 in paper)
        shuffle_buffer_length (None | int): Length of random queue for shuffling.
            If None, entire dataset will be shuffled.
        instances_per_sequence (int): Number of subsequences per sequence ("lambda")
        estimator_kwargs (dict[str]): Other kwargs to be passed to class f"estimator_name"
        use_wandb_logger (bool, optional): If True, use W&B logger on top of CSVLogger.
            Defaults to False.
        wandb_project_name (None | str, optional): weights and biases project name.
            Defaults to None.

    Returns:
        tuple[str, dict[str], dict[str]]: Tuple composed of
            - Directory in which lightning logs are stored
            - Dictionary of gluonTS validation results, see "make_evaluation_predictions"
            - Dictionary of gluonTS test results, see "make_evaluation_predictions"
    """

    seed_everything(seed)

    # Load data
    data_module = GluonTSLightningDataModule(dataset_kwargs)

    data_module.prepare_data()
    data_module.setup(stage='fit')
    meta = data_module.meta

    # Prepare logging
    loggers = [
        CSVLogger(save_dir, experiment_name)
    ]

    if use_wandb_logger:
        loggers.append(WandbLogger(
            save_dir=save_dir,
            name=experiment_name,
            project=wandb_project_name))

    # Create estimator
    estimator, estimator_kwargs = create_estimator(
        estimator_name, top_level_mode, estimator_kwargs,
        instances_per_sequence, meta, loggers)

    # Log all the hyperparameteres
    for logger in loggers:
        logger.log_hyperparams({
            'save_dir': experiment_name,
            'experiment_name': experiment_name,
            'dataset_kwargs': dataset_kwargs,
            'estimator_name': estimator_name,
            'estimator_kwargs': estimator_kwargs,
            'seed': seed})

    # Train
    predictor = estimator.train(
        data_module.dataset_train,
        data_module.dataset_val,
        shuffle_buffer_length,
        cache_data=True,
        validation_pred_data=data_module.dataset_val_pred
    )

    # Evaluate
    metrics_val = eval_predictor(
        data_module.dataset_val, predictor)

    metrics_test = eval_predictor(
        data_module.dataset_test, predictor)

    for logger in loggers:
        logger.log_metrics({f'final_val_{k}': v for k, v in metrics_val.items()})
        logger.log_metrics({f'final_test_{k}': v for k, v in metrics_test.items()})
        logger.save()

    if use_wandb_logger:
        wandb.finish()

    return loggers[0].log_dir, metrics_val, metrics_test


def create_estimator(
        estimator_name: str,
        top_level_mode: str,
        estimator_kwargs: dict,
        instances_per_sequence: int,
        meta: MetaData,
        loggers: None | list[Logger] = None
) -> tuple[PyTorchLightningEstimator, dict]:

    estimator_kwargs = deepcopy(estimator_kwargs)

    estimator_kwargs.update(
        {'prediction_length': meta.prediction_length,
         'top_level_mode': top_level_mode,
         'instances_per_sequence': instances_per_sequence,
         'use_dp_lightning_module': False,
         'dp_optimizer_kwargs': None,
         'neighboring_relation': None,
         'dp_accountant_kwargs': None,
         'tight_privacy_loss': False})

    estimator_kwargs['trainer_kwargs']['logger'] = loggers

    if estimator_name in ['DeepAREstimator',
                          'WaveNetEstimator',
                          'TemporalFusionTransformerEstimator']:

        estimator_kwargs['freq'] = meta.freq

    if estimator_name != 'WaveNetEstimator':
        estimator_kwargs['distr_output'] = StudentTOutput()

    return (create_dp_estimator(estimator_name,
                                estimator_kwargs),
            estimator_kwargs)
