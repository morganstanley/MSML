import os

from dp_timeseries.data import GluonTSLightningDataModule
from dp_timeseries.evaluation import eval_predictor
from gluonts.dataset.common import MetaData
from gluonts.ext.statsforecast import AutoARIMAPredictor, AutoETSPredictor
from gluonts.model import Predictor
from gluonts.model.seasonal_naive import SeasonalNaivePredictor

from .utils import seed_everything


def traditional_baselines_standard_eval(
        seed: int,
        save_dir: str,
        experiment_name: str,
        dataset_kwargs: dict[str],
        predictor_name: str,
        use_season_length: bool
        ) -> tuple[dict[str], dict[str]]:
    """Applies statistical forecasting baselines without DP input noise to a dataset.

    Args:
        seed (int): Random seed for experiment
        save_dir (str): Directory in which logs should be stored
        experiment_name (str): Name of experiment for lightning loggers
        dataset_kwargs (dict[str]): kwargs for GluonTSLightningDataModule that loads data.
        predictor_name (str): Class name of predictor to use.
            Should be in "SeasonalNaivePredictor", "AutoARIMAPredictor", "AutoETSPredictor".
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
    predictor = get_predictor(predictor_name, meta, use_season_length)

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
                  use_season_length: bool) -> Predictor:

    season_lengths = {
        'H': 24,
        '1H': 24,
        'D': 30,
        '1D': 30,
        'B': 30,
        '1B': 30,
        '10T': 144  # 6 * 10 minutes per hour, 24 hours per day
    }

    if predictor_name == 'SeasonalNaivePredictor':
        return SeasonalNaivePredictor(
            prediction_length=meta.prediction_length,
            season_length=season_lengths[meta.freq],
        )
    elif predictor_name == 'AutoARIMAPredictor':
        if use_season_length:
            return AutoARIMAPredictor(
                prediction_length=meta.prediction_length,
                quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                season_length=season_lengths[meta.freq],
            )
        else:
            return AutoARIMAPredictor(
                prediction_length=meta.prediction_length,
                quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            )
    elif predictor_name == 'AutoETSPredictor':
        if use_season_length:
            return AutoETSPredictor(
                prediction_length=meta.prediction_length,
                quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                season_length=season_lengths[meta.freq],
            )
        else:
            return AutoETSPredictor(
                prediction_length=meta.prediction_length,
                quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            )
    else:
        raise ValueError(f'{predictor_name} not supported.')
