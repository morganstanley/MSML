import numpy as np
import pandas as pd
import argparse

import torch
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.repository.datasets import dataset_recipes, get_dataset
from pts.model.deepvar import DeepVAREstimator
from pts import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import MultivariateEvaluator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def energy_score(forecast, target):
    obs_dist = np.mean(np.linalg.norm((forecast - target), axis=-1))
    pair_dist = np.mean(
        np.linalg.norm(forecast[:, np.newaxis, ...] - forecast, axis=-1)
    )
    return obs_dist - pair_dist * 0.5

def train(dataset_name):
    covariance_dim = 4 if dataset_name != 'exchange_rate_nips' else -4

    dataset = get_dataset(dataset_name, regenerate=False)

    train_grouper = MultivariateGrouper(max_target_dim=int(dataset.metadata.feat_static_cat[0].cardinality))
    test_grouper = MultivariateGrouper(num_test_dates=int(len(dataset.test)/len(dataset.train)), max_target_dim=int(dataset.metadata.feat_static_cat[0].cardinality))

    dataset_train = train_grouper(dataset.train)
    dataset_test = test_grouper(dataset.test)

    evaluator = MultivariateEvaluator(quantiles=(np.arange(20)/20.0)[1:], target_agg_funcs={'sum': np.sum})

    target_dim = int(dataset.metadata.feat_static_cat[0].cardinality)

    estimator = DeepVAREstimator(
        input_size=target_dim * 4 + covariance_dim + 3,
        target_dim=target_dim,
        prediction_length=dataset.metadata.prediction_length,
        context_length=dataset.metadata.prediction_length*4,
        freq=dataset.metadata.freq,
        trainer=Trainer(
            device=device,
            epochs=40,
            learning_rate=1e-3,
            num_batches_per_epoch=100,
            batch_size=64,
        )
    )

    predictor = estimator.train(dataset_train)
    forecast_it, ts_it = make_evaluation_predictions(dataset=dataset_test,
                                                predictor=predictor,
                                                num_samples=100)
    forecasts = list(forecast_it)
    targets = list(ts_it)

    agg_metric, _ = evaluator(targets, forecasts, num_series=len(dataset_test))

    score = energy_score(
        forecast=np.array([x.samples for x in forecasts]),
        target=np.array([x[-dataset.metadata.prediction_length:] for x in targets])[:,None,...],
    )

    metrics = dict(
        CRPS=agg_metric['mean_wQuantileLoss'],
        ND=agg_metric['ND'],
        NRMSE=agg_metric['NRMSE'],
        CRPS_sum=agg_metric['m_sum_mean_wQuantileLoss'],
        ND_sum=agg_metric['m_sum_ND'],
        NRMSE_sum=agg_metric['m_sum_NRMSE'],
        energy_score=score,
    )
    metrics = { k: float(v) for k,v in metrics.items() }

    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train forecasting model.')
    parser.add_argument('--dataset', type=str, choices=['exchange_rate_nips', 'electricity_nips', 'solar_nips', 'traffic_nips'])
    args = parser.parse_args()

    metrics = train(args.dataset)

    for key, value in metrics.items():
        print(f'{key}:\t{value:.4f}')
