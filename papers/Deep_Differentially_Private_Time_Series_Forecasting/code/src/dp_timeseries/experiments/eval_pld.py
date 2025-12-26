import os
from typing import Sequence

import numpy as np
from dp_accounting.pld.privacy_loss_distribution import (
    PrivacyLossDistribution, _create_pld_pmf_from_additive_noise)
from dp_timeseries.privacy.bilevel_subsampling import create_privacy_loss


def eval_pld(
            save_dir: str,
            experiment_name: str,
            privacy_loss_kwargs: dict[str],
            epsilons: Sequence,
            value_discretization_interval: None | float = None,
            use_connect_dots: bool = True
        ) -> tuple[dict[str], dict[str]]:
    """Evaluates privacy profiles via Theorem 4.2 or 4.4.

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
            If None, will not quantize.
            Defaults to 1e-3.
        use_connect_dots (bool, optional): Whether to use optimal quantization or privacy buckets.
            True results in optimal quantization.
            Defaults to True.

    Returns:
        tuple[str, dict[str]]: Where SEML experiment should save results and the results themselves.
            Results are:
                - "epsilons": epsilons as an array
                - "deltas": Privacy profile values of shape len(epsilons)
    """

    # Create pld
    privacy_loss = create_privacy_loss(**privacy_loss_kwargs)
    pld = privacy_loss

    # Quantize, if desired
    if value_discretization_interval is not None:
        pld_pmf = _create_pld_pmf_from_additive_noise(
                    privacy_loss,
                    value_discretization_interval=value_discretization_interval,
                    use_connect_dots=use_connect_dots)

        pld = PrivacyLossDistribution(pld_pmf)

    # Get privacy profile
    epsilons = np.array(epsilons)
    deltas = pld.get_delta_for_epsilon(epsilons)

    # Map to dict
    results_dict = {
        'epsilons': epsilons,
        'deltas': deltas
    }

    # Create dir for storing results
    log_dir = os.path.join(save_dir, experiment_name, 'version_0')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    return log_dir, results_dict
