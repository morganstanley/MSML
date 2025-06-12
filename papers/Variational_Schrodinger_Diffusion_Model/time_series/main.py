import argparse
import json
import pickle
import random
import torch
import numpy as np
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent / 'results'

from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.repository.datasets import get_dataset, dataset_recipes
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import MultivariateEvaluator

from estimator import DiffusionEstimator

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

def main(opt):
    set_seed(opt.seed)

    if opt.ddpm:
        prefix = 'ddpm'
    elif opt.forward_opt_steps == 0:
        prefix = 'dsm'
    else:
        prefix = 'sb'

    name = f'{prefix}__{opt.data}__{opt.seed}__{opt.forward_opt_steps}__{opt.backward_opt_steps}'
    dir = ROOT_DIR / name
    dir.mkdir(exist_ok=True, parents=True)

    if (dir / 'forecasts.pickle').exists() and not opt.redo:
        print(f'Experiment {name} already exists. Skipping...')
        return

    dataset = get_dataset(opt.data, regenerate=False)

    opt.data_dim = min(int(dataset.metadata.feat_static_cat[0].cardinality), opt.max_data_dim)

    train_grouper = MultivariateGrouper(
        max_target_dim=opt.data_dim,
    )
    test_grouper = MultivariateGrouper(
        num_test_dates=int(len(dataset.test) / len(dataset.train)),
        max_target_dim=opt.data_dim,
    )

    dataset_train = train_grouper(dataset.train)
    dataset_test = test_grouper(dataset.test)

    estimator = DiffusionEstimator(
        ddpm_baseline=opt.ddpm,
        freq=dataset.metadata.freq,
        input_size=opt.data_dim,
        prediction_length=dataset.metadata.prediction_length,
        context_length=dataset.metadata.prediction_length * 3,
        batch_size=opt.batch_size,
        t0=opt.t0,
        T=opt.T,
        beta_min=opt.beta_min,
        beta_max=opt.beta_max,
        beta_r=opt.beta_r,
        n_timestep=opt.steps,
        forward_opt_steps=opt.forward_opt_steps,
        backward_opt_steps=opt.backward_opt_steps,
        num_layers=2,
        hidden_size=opt.hidden_dim,
        lags_seq=None,
        scaling='std',
        trainer_kwargs=dict(
            max_epochs=opt.epochs,
            accelerator='cpu' if opt.cpu else 'gpu',
            devices=[opt.device],
            callbacks=[ModelCheckpoint(monitor=None)],
            logger=CSVLogger(dir, name='logs'),
        ),
    )

    predictor = estimator.train(dataset_train, cache_data=True, shuffle_buffer_length=1024)

    if prefix == 'sb':
        A = predictor.prediction_net.model.forward_net.A.detach().cpu().numpy()
        np.save(dir / 'forward_matrix.npy', A)

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset_test,
        predictor=predictor,
        num_samples=estimator.num_parallel_samples,
    )
    forecasts = list(forecast_it)
    targets = list(ts_it)

    evaluator = MultivariateEvaluator(
        quantiles=(np.arange(20) / 20.0)[1:], target_agg_funcs={"sum": np.sum}
    )
    agg_metric, _ = evaluator(targets, forecasts, num_series=len(dataset_test))


    summary_metrics = {
        'CRPS': agg_metric["mean_wQuantileLoss"],
        'ND': agg_metric["ND"],
        'NRMSE': agg_metric["NRMSE"],
        'MSE': agg_metric["MSE"],
        'CRPS-Sum': agg_metric["m_sum_mean_wQuantileLoss"],
        'ND-Sum': agg_metric["m_sum_ND"],
        'NRMSE-Sum': agg_metric["m_sum_NRMSE"],
        'MSE-Sum': agg_metric["m_sum_MSE"],
    }

    print(summary_metrics)

    with open(dir / 'metrics.json', 'w') as f:
        json.dump(agg_metric, f)

    with open(dir / 'forecasts.pickle', 'wb') as f:
        pickle.dump([forecasts, targets], f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, choices=list(dataset_recipes.keys()))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max_data_dim", type=int, default=2000)

    # Diffusion params
    parser.add_argument("--t0", type=float, required=False)
    parser.add_argument("--T", type=float, required=False)
    parser.add_argument("--beta_min", type=float, required=False)
    parser.add_argument("--beta_max", type=float, required=False)
    parser.add_argument("--beta_r", type=float, required=False)
    parser.add_argument("--steps", type=int, required=False)

    # Training params
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--hidden_dim", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--forward_opt_steps", type=int)
    parser.add_argument("--backward_opt_steps", type=int)
    parser.add_argument("--ddpm", action='store_true')
    parser.add_argument("--redo", action='store_true')

    opt = parser.parse_args()
    main(opt)
