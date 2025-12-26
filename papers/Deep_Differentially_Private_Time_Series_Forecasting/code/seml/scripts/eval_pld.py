import os

import numpy as np
import torch
from dp_timeseries.experiments.eval_pld import eval_pld
from numpy.typing import NDArray
from seml import Experiment

ex = Experiment()

eval_pld = ex.capture(eval_pld)


@ex.config
def config():
    privacy_loss_kwargs = {
        'num_sequences': 100,
        'min_sequence_length': 1000,
        'top_level_mode': 'sampling_without_replacement',
        'instances_per_sequence': 1,
        'batch_size': 10,
        'past_length': 48,
        'future_length': 24,
        'lead_time': 0,
        'min_past': 0,
        'min_future': 24,
        'noise_multiplier': 1.0,
        'tight_privacy_loss': True,
        'future_target_noise_multiplier': 0,
        'bottom_level_mode': 'sampling_poisson'
    }

    privacy_loss_kwargs['neighboring_relation'] = {
        'level': 'event',
        'size': 1
    }

    epsilon_params = {
        'space': 'linspace',
        'start': 1.0,
        'stop': 8.0,
        'num': 8
    }

    value_discretization_interval = None
    use_connect_dots = True


@ex.capture(prefix='epsilon_params')
def epsilon_from_epsilon_params(
        space: str,
        start: float, stop: float, num: int) -> NDArray[np.float64]:

    if space == 'linspace':
        space_fn = np.linspace
    elif space == 'logspace':
        space_fn = np.logspace
    else:
        raise ValueError(f'{space} for epsilon_params.space not supported.')

    return space_fn(start=start, stop=stop, num=num)


@ex.automain
def run(_config: dict) -> dict:

    db_collection = _config['db_collection']
    run_id = _config['overwrite']

    experiment_name = f'{db_collection}_{str(run_id)}'

    epsilons = epsilon_from_epsilon_params()

    log_dir, results_dict = eval_pld(
        epsilons=epsilons,
        experiment_name=experiment_name)

    results_dict['log_dir'] = log_dir

    save_dict = {
        'config': _config,
        'results': results_dict
    }

    torch.save(save_dict, os.path.join(log_dir, 'config_and_results.pyt'))

    return {'log_dir': log_dir}
