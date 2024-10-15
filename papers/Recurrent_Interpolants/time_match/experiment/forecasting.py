import argparse
import hashlib
import json
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.evaluation import MultivariateEvaluator
from gluonts.evaluation.backtest import make_evaluation_predictions

from time_match.utils.get_estimator import get_estimator

SAVE_DIR = Path(__file__).resolve().parents[2] / 'results' / 'forecasting'
SAVE_DIR.mkdir(exist_ok=True)

NUM_QUANTILES = 10
MAX_TARGET_DIM = 2000


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def get_dir(config: dict):
    del config['device']
    del config['redo']
    name = json.dumps(config, sort_keys=True).encode()
    name = hashlib.md5(name)
    return SAVE_DIR / f'experiment_{name.hexdigest()}'

def run(args: argparse.Namespace):
    set_seed(args.seed)

    if args.beta_end is None:
        args.beta_end = 20 / args.steps

    config = {**vars(args)}
    args.out_dir = get_dir(config)

    if (args.out_dir / 'forecasts.pickle').exists() and not args.redo:
        print(f'{args.out_dir} Already exists. Skipping...')
        return

    dataset = get_dataset(args.dataset, regenerate=False)
    args.max_target_dim = min(int(dataset.metadata.feat_static_cat[0].cardinality), MAX_TARGET_DIM)
    args.prediction_length = dataset.metadata.prediction_length

    train_grouper = MultivariateGrouper(
        max_target_dim=args.max_target_dim
    )
    test_grouper = MultivariateGrouper(
        num_test_dates=int(len(dataset.test) / len(dataset.train)),
        max_target_dim=args.max_target_dim,
    )
    dataset_train = train_grouper(dataset.train)
    dataset_test = test_grouper(dataset.test)
    evaluator = MultivariateEvaluator(
        quantiles=(np.arange(NUM_QUANTILES) / NUM_QUANTILES)[1:], target_agg_funcs={"sum": np.sum}
    )
    estimator = get_estimator(args, dataset)

    predictor = estimator.train(dataset_train, cache_data=True, shuffle_buffer_length=1024)
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset_test, predictor=predictor, num_samples=100
    )

    t_start = time.perf_counter()
    forecasts = list(forecast_it)
    t_end = time.perf_counter()
    targets = list(ts_it)

    agg_metric, _ = evaluator(targets, forecasts, num_series=len(dataset_test))

    # Save results
    with open(args.out_dir / 'metrics.json', 'w') as f:
        json.dump(agg_metric, f)
    with open(args.out_dir / 'config.json', 'w') as f:
        config['inference_time'] = t_end - t_start
        config['prediction_length'] = args.prediction_length
        json.dump(config, f)
    with open(args.out_dir / 'forecasts.pickle', 'wb') as f:
        pickle.dump(forecasts, f)
    with open(args.out_dir / 'targets.pickle', 'wb') as f:
        pickle.dump(targets, f)

    result_keys = ['MASE', 'MAPE', 'MSE', 'NRMSE', 'ND', 'm_sum_mean_wQuantileLoss']
    for k in result_keys:
        print(k, agg_metric[k])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['constant', 'solar_nips', 'exchange_rate_nips', 'electricity_nips', 'wiki-rolling_nips', 'taxi_30min', 'traffic_nips'])
    parser.add_argument('--estimator', type=str, choices=['ddpm', 'fm', 'sgm', 'si', 'blend'])
    parser.add_argument('--scaling', type=str, choices=['mean', 'std'])
    parser.add_argument('--rnn_model', type=str, default='lstm', choices=['lstm'])
    parser.add_argument('--denoising_model', type=str, default='epsilon_theta', choices=['epsilon_theta', 'unet1d'])
    parser.add_argument('--velocity_model', type=str, default='epsilon_theta', choices=['epsilon_theta', 'unet1d'])

    parser.add_argument('--gamma', type=str, required=False, choices=['trig', 'quad', 'sqrt', 'zero'])
    parser.add_argument('--interpolant', type=str, required=False, choices=['linear', 'trig', 'encdec'])
    parser.add_argument('--epsilon', type=float, default=0.1, required=False)
    parser.add_argument('--start_noise', action='store_true')
    parser.add_argument('--importance_sampling', action='store_true')

    parser.add_argument('--beta_start', type=float, default=1e-4, required=False)
    parser.add_argument('--beta_end', type=float, required=False)
    parser.add_argument('--steps', type=int, required=False)

    parser.add_argument('--sigma_min', type=float, default=0.01, required=False)

    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--hidden_size', type=int)
    parser.add_argument('--num_layers', type=int)
    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--dt', type=float, required=False)
    parser.add_argument('--sde_solver', type=str, default='manual', required=False)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--redo', action='store_true')

    args = parser.parse_args()

    run(args)
