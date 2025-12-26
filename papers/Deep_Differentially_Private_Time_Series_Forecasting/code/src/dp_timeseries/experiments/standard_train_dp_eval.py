import os
from copy import deepcopy

from dp_timeseries.data import GluonTSLightningDataModule
from dp_timeseries.evaluation import eval_predictor
from dp_timeseries.transformations import (SubsamplePoisson,
                                           SubsampleWithoutReplacement)
from gluonts.dataset.field_names import FieldName
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.transform import (AddObservedValuesIndicator, Chain,
                               InstanceSplitter)
from gluonts.transform.feature import *
from seml import get_results

from .standard_train_standard_eval import create_estimator
from .utils import (create_calibrated_noise_transform,
                    nested_dict_to_dot_dict, seed_everything)


def standard_train_dp_eval(
        seed: int,
        save_dir: str,
        experiment_name: str,
        dataset_kwargs: dict[str],
        training_db_collection: str,
        training_seed: int,
        estimator_name: str,
        top_level_mode: str,
        shuffle_buffer_length: None | int,
        instances_per_sequence: int,
        estimator_kwargs: dict[str],
        inference_kwargs: dict[str],
        ) -> tuple[str, dict[str], dict[str]]:
    """Does inference while adding input noise with an already trained model.

    Args:
        seed (int): Random seed for experiment
        save_dir (str): Directory in which logs should be stored
        experiment_name (str): Name of experiment for lightning loggers
        dataset_kwargs (dict[str]): kwargs for GluonTSLightningDataModule that loads data.
        training_db_collection (str): Name of mongoDB databased with SEML experiments from training.
        training_seed (int): Seed used during training
        estimator_name (str): Class name of GluonTS estimator used during training.
        top_level_mode (str): Top-level subsampling scheme used during training. Should be in:
            - "iteration" (Algorithm 3 in paper)
            - "shuffling" (In each epoch, shuffle train set before iterating)
            - "sampling_without_replacement" (Algorithm 5 in paper)
        shuffle_buffer_length (None | int): Length of random queue used during training.
        instances_per_sequence (int): Num. of subsequences per sequence ("lambda") during training.
        estimator_kwargs (dict[str]): Other args passed to class f"estimator_name" during training.
        inference_kwargs (dict[str]): Args for create_dp_predictor below.

    Returns:
        tuple[str, dict[str], dict[str]]: Tuple composed of
            - Directory in which SEML experiment should store results of experiment.
            - Dictionary of gluonTS validation results, see "make_evaluation_predictions"
            - Dictionary of gluonTS test results, see "make_evaluation_predictions"
    """

    seed_everything(seed)

    # Load data
    data_module = GluonTSLightningDataModule(dataset_kwargs)
    data_module.prepare_data()
    data_module.setup(stage='fit')

    # Load trained predictor
    predictor = load_trained_predictor(
        dataset_kwargs,
        training_db_collection,
        training_seed,
        estimator_name,
        top_level_mode,
        shuffle_buffer_length,
        instances_per_sequence,
        estimator_kwargs)

    # Add transformations to make trained predictor DP
    predictor = create_dp_predictor(
        predictor, **inference_kwargs)

    # Evaluate
    metrics_val = eval_predictor(
        data_module.dataset_val, predictor)

    metrics_test = eval_predictor(
        data_module.dataset_test, predictor)

    # Create dir for storing results
    log_dir = os.path.join(save_dir, experiment_name, 'version_0')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    return log_dir, metrics_val, metrics_test


def load_trained_predictor(
        dataset_kwargs: dict,
        training_db_collection: str,
        training_seed: int,
        estimator_name: str,
        top_level_mode: str,
        shuffle_buffer_length: None | int,
        instances_per_sequence: int,
        estimator_kwargs: dict[str]) -> PyTorchPredictor:
    """Utility function to load trained model from seml training experiment.

    Args:
        dataset_kwargs (dict[str]): kwargs for GluonTSLightningDataModule that loads data.
        training_db_collection (str): Name of mongoDB databased with SEML experiments from training.
        training_seed (int): Seed used during training
        estimator_name (str): Class name of GluonTS estimator used during training.
        top_level_mode (str): Top-level subsampling scheme used during training. Should be in:
            - "iteration" (Algorithm 3 in paper)
            - "shuffling" (In each epoch, shuffle train set before iterating)
            - "sampling_without_replacement" (Algorithm 5 in paper)
        shuffle_buffer_length (None | int): Length of random queue used during training.
        instances_per_sequence (int): Num. of subsequences per sequence ("lambda") during training.
        estimator_kwargs (dict[str]): Other args passed to class f"estimator_name" during training.

    Returns:
        tuple[str, dict[str], dict[str]]: Tuple composed of
            - Directory in which SEML experiment should store results of experiment.
            - Dictionary of gluonTS validation results, see "make_evaluation_predictions"
            - Dictionary of gluonTS test results, see "make_evaluation_predictions"
    """

    # Get metadata
    data_module = GluonTSLightningDataModule(dataset_kwargs)
    data_module.prepare_data()
    data_module.setup(stage='fit')
    meta = data_module.meta

    # Create estimator
    estimator, _ = create_estimator(
        estimator_name, top_level_mode,
        estimator_kwargs, instances_per_sequence,
        meta)

    # Load weights into lightning module
    lightning_module = estimator.create_lightning_module()

    checkpoint_path = _get_checkpoint_path(
        dataset_kwargs,
        training_db_collection,
        training_seed,
        estimator_name=estimator_name,
        top_level_mode=top_level_mode,
        shuffle_buffer_length=shuffle_buffer_length,
        instances_per_sequence=instances_per_sequence,
        estimator_kwargs=estimator_kwargs
    )

    best_model = lightning_module.__class__.load_from_checkpoint(checkpoint_path)

    return estimator.create_predictor(
        estimator.create_transformation(), best_model)


def _get_checkpoint_path(
        dataset_kwargs: dict,
        training_db_collection: str,
        training_seed: int,
        **training_kwargs) -> str:
    """Utility function to find path of checkpoint stored during model training."""

    training_kwargs['seed'] = training_seed
    training_kwargs['dataset_kwargs'] = dataset_kwargs

    filter_dict = nested_dict_to_dot_dict(training_kwargs)
    filter_dict = {f'config.{k}': v for k, v in filter_dict.items()}

    log_dirs = get_results(
        training_db_collection, ['result.log_dir'],
        to_data_frame=True,
        states=['COMPLETED'],
        filter_dict=filter_dict,
        parallel=True)['result.log_dir']

    if len(log_dirs) == 0:
        raise ValueError('No training log_dir found.')
    if len(log_dirs) > 1:
        raise ValueError('Multiple training log_dirs found.')

    log_dir = log_dirs[0]
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    checkpoint_names = [x for x in os.listdir(checkpoint_dir)
                        if '.ckpt' in x]

    if len(checkpoint_names) == 0:
        raise ValueError('No checkpoint found.')
    if len(checkpoint_names) > 1:
        raise ValueError('Multiple checkpoints found.')

    return os.path.join(checkpoint_dir, checkpoint_names[0])


def create_dp_predictor(
        predictor: PyTorchPredictor,
        neighboring_relation: dict[str],
        budget_epsilon: float,
        budget_delta: float,
        subsample_transform_name: str,
        subsample_transform_kwargs: dict,
        imputation_transform_name: str,
        imputation_transform_kwargs: dict,
        reset_observed_values_indicator: bool) -> PyTorchPredictor:
    """Adds input noise and subsampling to inference procedure of GluonTS Predictor.

    The input noise is calibrated s.t. a specified privacy level is achieved.

    Args:
        predictor (PyTorchPredictor): Original GluonTS Predictor
        neighboring_relation (dict[str]): Considered neighboring relation
            - "level": "event" or "user"
            - "size": Number of modified elements, ("w" in paper)
            - "target_sensitivity": Maximum absolute change in modified elements ("v" in paper)
        budget_epsilon (float): Privacy parameter epsilon
        budget_delta (float): Privacy parameter delta
        subsample_transform_name (str): Class name from dp_timeseries/transformations/subsampling.
        subsample_transform_kwargs (dict): Args to pass to class "subsample_transform_name"
        imputation_transform_name (str): Class name from gluonts.transform.feature
        imputation_transform_kwargs (dict): Args to pass to class "imputation_transform_name"
        reset_observed_values_indicator (bool): Whether to treat imputed values as observed.
            If True, will overwrite observed_values_indicator generated by imputation method.

    Returns:
        PyTorchPredictor: GluonTS Predictor with input noise and subsampling.
    """

    predictor = deepcopy(predictor)

    original_transform = predictor.input_transform

    assert isinstance(original_transform, Chain)
    assert isinstance(original_transform.transformations[-1],
                      InstanceSplitter)

    target_field = f'past_{FieldName.TARGET}'
    observed_values_field = f'past_{FieldName.OBSERVED_VALUES}'
    is_pad_field = f'past_{FieldName.IS_PAD}'

    if subsample_transform_name == 'SubsamplePoisson':
        subsample_transform = SubsamplePoisson(
            target_field=target_field,
            observed_values_field=observed_values_field,
            is_pad_field=is_pad_field,
            **subsample_transform_kwargs
        )
    elif subsample_transform_name == 'SubsampleWithoutReplacement':
        subsample_transform = SubsampleWithoutReplacement(
            target_field=target_field,
            observed_values_field=observed_values_field,
            is_pad_field=is_pad_field,
            context_length=original_transform[-1].past_length,
            **subsample_transform_kwargs
        )
    else:
        raise ValueError('subsample_transform_name must be in '
                         '["SubsamplePoisson", "SubsampleWithoutReplacement"]')

    imputation_transform = globals()[imputation_transform_name](
        **imputation_transform_kwargs)

    noise_transform = create_calibrated_noise_transform(
        neighboring_relation,
        budget_epsilon, budget_delta,
        target_field,
        observed_values_field,
        is_pad_field,
        subsample_transform,
    )

    original_transform.transformations.extend([noise_transform, subsample_transform])

    original_transform.transformations.append(AddObservedValuesIndicator(
        target_field, observed_values_field,
        imputation_method=imputation_transform
    ))

    # After applying imputation, all values are non-nan
    # so AddObservedValuesIndicator resets is_observed_field to all-ones
    if reset_observed_values_indicator:
        original_transform.transformations.append(AddObservedValuesIndicator(
            target_field, observed_values_field))

    assert not isinstance(predictor.input_transform.transformations[-1],
                          InstanceSplitter)

    return predictor
