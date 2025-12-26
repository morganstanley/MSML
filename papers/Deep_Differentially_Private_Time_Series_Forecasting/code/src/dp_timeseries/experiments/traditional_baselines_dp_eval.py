import os

from dp_timeseries.data import GluonTSLightningDataModule
from dp_timeseries.evaluation import eval_predictor
from dp_timeseries.models.seasonal_naive import \
    PreprocessedSeasonalNaivePredictor
from dp_timeseries.models.statsforecast import (PreprocessedAutoARIMAPredictor,
                                                PreprocessedAutoETSPredictor)
from gluonts.dataset.common import MetaData
from gluonts.dataset.field_names import FieldName
from gluonts.model import Predictor

from .utils import create_calibrated_noise_transform, seed_everything


def traditional_baselines_dp_eval(
        seed: int,
        save_dir: str,
        experiment_name: str,
        dataset_kwargs: dict[str],
        predictor_name: str,
        neighboring_relation: dict,
        budget_epsilon: float,
        budget_delta: float,
        use_season_length: bool
        ) -> tuple[str, dict[str], dict[str]]:
    """Applies statistical forecasting baselines with DP input noise to a dataset.

    Input noise is calibrated s.t. a specified privacy level is achieved.

    Args:
        seed (int): Random seed for experiment
        save_dir (str): Directory in which logs should be stored
        experiment_name (str): Name of experiment for lightning loggers
        dataset_kwargs (dict[str]): kwargs for GluonTSLightningDataModule that loads data.
        predictor_name (str): Class name of predictor to use.
            Should be in "SeasonalNaivePredictor", "AutoARIMAPredictor", "AutoETSPredictor".
        neighboring_relation (dict[str]): Considered neighboring relation
            - "level": "event" or "user"
            - "size": Number of modified elements, ("w" in paper)
            - "target_sensitivity": Maximum absolute change in modified elements ("v" in paper)
        budget_epsilon (float): Privacy parameter epsilon
        budget_delta (float): Privacy parameter delta
        use_season_length (bool): If True, pass "season_length" of respective dataset to predictor.

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

    # Get predictor
    predictor = get_predictor(predictor_name, meta,
                              neighboring_relation,
                              budget_epsilon, budget_delta,
                              use_season_length)

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


def get_predictor(predictor_name: str,
                  meta: MetaData,
                  neighboring_relation: dict,
                  budget_epsilon: float,
                  budget_delta: float,
                  use_season_length:  bool) -> Predictor:

    season_lengths = {
        'H': 24,
        '1H': 24,
        'D': 30,
        '1D': 30,
        'B': 30,
        '1B': 30,
        '10T': 144  # 6 * 10 minutes per hour, 24 hours per day
    }

    noise_transform = create_calibrated_noise_transform(
            neighboring_relation,
            budget_epsilon, budget_delta,
            FieldName.TARGET)

    if predictor_name == 'SeasonalNaivePredictor':
        return PreprocessedSeasonalNaivePredictor(
            prediction_length=meta.prediction_length,
            season_length=season_lengths[meta.freq],
            input_transform=noise_transform)

    elif predictor_name == 'AutoARIMAPredictor':
        if use_season_length:
            return PreprocessedAutoARIMAPredictor(
                prediction_length=meta.prediction_length,
                quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                input_transform=noise_transform,
                season_length=season_lengths[meta.freq])
        else:
            return PreprocessedAutoARIMAPredictor(
                prediction_length=meta.prediction_length,
                quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                input_transform=noise_transform)

    elif predictor_name == 'AutoETSPredictor':
        if use_season_length:
            return PreprocessedAutoETSPredictor(
                prediction_length=meta.prediction_length,
                quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                input_transform=noise_transform,
                season_length=season_lengths[meta.freq])
        else:
            return PreprocessedAutoETSPredictor(
                prediction_length=meta.prediction_length,
                quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                input_transform=noise_transform)

    else:
        raise ValueError(f'{predictor_name} not supported.')