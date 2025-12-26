import random

import numpy as np
import torch
from dp_accounting.pld.privacy_loss_mechanism import (AdjacencyType,
                                                      GaussianPrivacyLoss)
from dp_timeseries.transformations import (AddGaussianNoise, SubsamplePoisson,
                                           SubsampleTransformation,
                                           SubsampleWithoutReplacement)
from scipy.optimize import bisect


def seed_everything(seed: int) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def nested_dict_to_dot_dict(x: dict) -> dict:
    return {
        k: v
        for k, v in _nested_dict_to_dot_list(x)
    }


def _nested_dict_to_dot_list(x: dict) -> list:
    """Converts nested dict to list of tuples using dot notation.

    The second entry of each tuple is a value from the lowest level of a nested dictionary.
    The first entry of each tuple are the keys leading to this value, in dot notation."""

    ret = []

    for key, value in x.items():
        assert isinstance(key, str)
        if isinstance(value, dict):
            child_ret = _nested_dict_to_dot_list(value)
            ret.extend([
                (f'{key}.{child_dotstring}', child_value)
                for child_dotstring, child_value in child_ret
            ])
        else:
            ret.append((key, value))

    return ret


def create_calibrated_noise_transform(
        neighboring_relation: dict,
        budget_epsilon: float,
        budget_delta: float,
        target_field: str,
        observed_values_field: None | str = None,
        is_pad_field: None | str = None,
        subsample_transform: None | SubsampleTransformation = None,
        absolute_tolerance: float = 1e-8,
) -> AddGaussianNoise:
    """Creates Gaussian input noise transform calibrated to specified privacy level.

    Noise takes into account amplification by subsampling via a used subsample_transform.

    Args:
        neighboring_relation (dict): Considered neighboring relation
            - "level": "event" or "user"
            - "size": Number of modified elements, ("w" in paper)
            - "target_sensitivity": Maximum absolute change in modified elements ("v" in paper
        budget_epsilon (float): Privacy parameter epsilon
        budget_delta (float): Privacy parameter delta
        target_field (str): Name of GluonTS dataset field to which noise should be added.
        observed_values_field (None | str, optional):  observed_values_field in GluonTS dataset
            Defaults to None.
        is_pad_field (None | str, optional): is_pad_field in GluonTS dataset.
            Defaults to None.
        subsample_transform (None | SubsampleTransformation, optional): Subsampling transform.
            Should be a class from transformations/subsampling.
            None implies that no subsampling is performed.
            Defaults to None.
        absolute_tolerance (float, optional): Tolerance for binary search used in calibration.
            Defaults to 1e-8.

    Returns:
        AddGaussianNoise: GluonTS transform to apply after subsample_transform.
    """

    if neighboring_relation['level'] != 'event':
        raise NotImplementedError
    if neighboring_relation['size'] != 1:
        raise NotImplementedError
    if 'target_sensitivity' not in neighboring_relation:
        raise ValueError

    if subsample_transform is None:
        p_bad = 1.0
    elif isinstance(subsample_transform, SubsamplePoisson):
        p_bad = subsample_transform.subsampling_rate
    elif isinstance(subsample_transform, SubsampleWithoutReplacement):
        p_bad = (subsample_transform.num_samples
                 / subsample_transform.context_length)
    else:
        raise NotImplementedError

    target_sensitivity = neighboring_relation['target_sensitivity']

    if budget_epsilon < 0:
        raise ValueError(f'epsilon must be g.e.q. 0, but {budget_epsilon=}')
    if budget_delta < 0:
        raise ValueError(f'delta must be g.e.q. 0, but {budget_delta=}')

    def delta_for_epsilon(standard_deviation: float) -> float:
        # Without clipping, only know pairwise distance between
        # x not sampled, x' not sampled
        # and x sampled, x' sampled
        # --> Just use standard convexity

        privacy_loss = GaussianPrivacyLoss(
            standard_deviation,
            sensitivity=target_sensitivity,
            sampling_prob=1.0,
            adjacency_type=AdjacencyType.REMOVE)

        return p_bad * privacy_loss.get_delta_for_epsilon(budget_epsilon)

    # privacy profile is monotonically decreasing in standard_deviation
    left_bound = 1
    while delta_for_epsilon(left_bound) < budget_delta:
        if np.isclose(left_bound, 0, atol=absolute_tolerance):
            left_bound = absolute_tolerance
            break
        left_bound /= 2
    right_bound = 1
    while delta_for_epsilon(right_bound) > budget_delta:
        right_bound *= 2

    standard_devation = bisect(
        (lambda x: delta_for_epsilon(x) - budget_delta),
        left_bound, right_bound)

    if np.isclose(standard_devation, 0, atol=absolute_tolerance):
        standard_devation = absolute_tolerance
    else:
        assert np.isclose(delta_for_epsilon(standard_devation), budget_delta)

    return AddGaussianNoise(
        target_field,
        observed_values_field,
        is_pad_field,
        standard_devation)
