import os
from typing import Sequence

import numpy as np
from dp_accounting.pld.privacy_loss_distribution import (
    PrivacyLossDistribution, _create_pld_pmf_from_additive_noise)
from dp_timeseries.privacy.bilevel_subsampling import create_privacy_loss
from tqdm import tqdm


def eval_pld_accounting(
            save_dir: str,
            experiment_name: str,
            privacy_loss_kwargs: dict[str],
            epsilons: Sequence,
            value_discretization_interval: float = 1e-3,
            use_connect_dots: bool = True,
            max_epochs: None | int = None,
            num_compositions: None | int = None
        ) -> tuple[str, dict[str]]:
    """Evaluates privacy profiles under composition via Theorem 4.2 or 4.4.

    Args:
        save_dir (str): Directory in which results are to be stored by seml experiment.
        experiment_name (str): Name of experiment, will be part saved file name.
        privacy_loss_kwargs (dict[str]): Parameters of amplification bounds.
            - "top_level_mode": Top-level sampling scheme.
                Should be in "iteration" for Alg. 3 or "sampling_without_replacmeent" for Alg. 5.
            - "num_sequences": Dataset size (N)
            - "instances_per_sequence": Number of subsequences to sample per sequence (lambda)
            - "batch_size": Number of subsequences per batch (Lambda)
        epsilons (Sequence): Epsilons at which privacy profile should be evaluated.
        value_discretization_interval (float, optional): Controls quantization of PLD.
            Defaults to 1e-3.
        use_connect_dots (bool, optional): Whether to use optimal quantization or privacy buckets.
            True results in optimal quantization.
            Defaults to True.
        max_epochs (None | int, optional): Number of training epochs.
            Used to compute number of compositions.
            Defaults to None.
        num_compositions (None | int, optional): Overrides number of compositions from max_epochs.
            Defaults to None.

    Returns:
        tuple[str, dict[str]]: Where SEML experiment should save results and the results themselves.
            Results are:
                - "epsilons": epsilons as an array
                - "deltas": Privacy profile values of shape num_compositions x len(epsilons)
    """

    if (max_epochs is None) and (num_compositions is None):
        raise ValueError('Specify max_epochs OR num_compositions!')
    if (max_epochs is not None) and (num_compositions is not None):
        raise ValueError('Specify max_epochs XOR num_compositions!')

    # Create pld
    privacy_loss = create_privacy_loss(**privacy_loss_kwargs)

    pld_pmf = _create_pld_pmf_from_additive_noise(
                privacy_loss,
                value_discretization_interval=value_discretization_interval,
                use_connect_dots=use_connect_dots)

    pld = PrivacyLossDistribution(pld_pmf)

    # Determine how many compositions are needed
    top_level_mode = privacy_loss_kwargs['top_level_mode']
    num_sequences = privacy_loss_kwargs['num_sequences']
    instances_per_sequence = privacy_loss_kwargs['instances_per_sequence']
    batch_size = privacy_loss_kwargs['batch_size']

    if num_compositions is None:
        if top_level_mode in ['sampling_without_replacement', 'sampling_poisson']:
            num_batches_per_epoch = num_sequences * instances_per_sequence // batch_size
            num_compositions = num_batches_per_epoch * max_epochs
        elif top_level_mode in ['iteration', 'shuffling']:
            num_compositions = max_epochs
        else:
            raise ValueError('top_level_mode {top_level_mode} not supported.')

    # Get all composed privacy profiles
    epsilons = np.array(epsilons)

    deltas = []

    pld_composed = pld
    for i in tqdm(range(num_compositions)):
        if i > 0:  # 1-fold self-composition is just mechanism itself
            pld_composed = pld_composed.compose(pld)

        deltas.append(pld_composed.get_delta_for_epsilon(epsilons))

    deltas = np.vstack(deltas)

    results_dict = {
        'epsilons': epsilons,
        'deltas': deltas
    }

    # Create dir for storing results
    log_dir = os.path.join(save_dir, experiment_name, 'version_0')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    return log_dir, results_dict
