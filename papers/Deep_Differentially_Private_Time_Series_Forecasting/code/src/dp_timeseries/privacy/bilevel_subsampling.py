import numpy as np
from dp_accounting.pld.privacy_loss_mechanism import (
    AdditiveNoisePrivacyLoss, AdjacencyType, GaussianPrivacyLoss,
    MixtureGaussianPrivacyLoss)
from scipy.stats import binom, hypergeom

from .pld import (DoubleMixtureGaussianPrivacyLoss, PerfectPrivacyLoss,
                  SwitchingPrivacyLoss, WeightedSumPrivacyLoss)


def create_privacy_loss_bottom_wr(
        p_bad_top: float,
        min_sequence_length: int,
        instances_per_sequence: int,
        past_length: int,
        future_length: int,
        lead_time: int,
        min_past: int,
        min_future: int,
        noise_multiplier: float,
        neighboring_relation: dict,
        tight_privacy_loss: bool = False,
        future_target_noise_multiplier: float = 0,
        lower_bound: bool = False
) -> AdditiveNoisePrivacyLoss:
    """Create opacus privacy loss via Theorem 4.2, Theorem 4.5, or Lemma F.4.

    Args:
        p_bad_top (float): Probability of sampling any sequence on top level.
        min_sequence_length (int): Shortest sequence in dataset (L).
        instances_per_sequence (int): Subsequences per sequence (lambda)
        past_length (int): Context length (L_C)
        future_length (int): Forecast length (L_F)
        lead_time (int): Gap between context and forecast window.
            0 to reproduce paper results.
        min_past (int): (L_C - min_past) is how many zeros are padded at start.
            0 to reproduce paper results.
        min_future (int): (L_F - min_future) is how many zeros are padded at end
            L_F to reproduce paper results.
        noise_multiplier (float): Noise scale (sigma)
        neighboring_relation (dict): Considered neighboring relation
            - "level": "event" or "user"
            - "size": Number of modified elements, ("w" in paper)
            - "target_sensitivity": Maximum absolute change in modified elements ("v" in paper)
        tight_privacy_loss (bool): If False, will use loose upper bound from Lemma F.4 on privacy.
        future_target_noise_multiplier (float, optional): Forecast noise scale (sigma_F).
            Defaults to 0.
        lower_bound (bool, optional): If True, compute optimistic lower bound on privacy.
            Defaults to False.

    Returns:
        AdditiveNoisePrivacyLoss: opacus privacy loss.
    """

    future_start_interval_len, num_bad_sample_idx = _calc_support_size(
        min_sequence_length,
        past_length, future_length, lead_time,
        min_past, min_future, neighboring_relation
    )

    p_bad_bottom = num_bad_sample_idx / future_start_interval_len

    # Amplification by label perturbation
    future_sensitivity = neighboring_relation.get('future_target_sensitivity', np.inf)
    if not np.isinf(future_sensitivity) and future_target_noise_multiplier > 0:

        if neighboring_relation['size'] != 1:
            raise NotImplementedError('Hybrid feature-label privacy currently only supports '
                                      '1-event-level and 1-user-level privacy.')

        if instances_per_sequence != 1:
            raise NotImplementedError('Hybrid feature-label privacy currently only supports '
                                      'sampling one instance per sequence.')

        total_variation_distance = _calc_gaussian_tvd(future_target_noise_multiplier,
                                                      sensitivity=1.0)

        relative_future_length = future_length / (past_length + future_length)

        # "Sampled as future" cases gets attenuated based on TVD
        # for more than one instance, we would need to consider "Number of times sampled as future"
        p_bad_bottom *= ((1 - relative_future_length) 
                         + relative_future_length * total_variation_distance)

    # Figure out probability of sampling different-sized subsets on bottom level
    sampling_probs = binom.pmf(np.arange(instances_per_sequence + 1),
                               instances_per_sequence, p_bad_bottom)

    # means of privacy loss distribution
    sensitivities = np.arange(instances_per_sequence + 1)
    # Multiply by two, because grad can change from +clip to -clip when substituting
    sensitivities *= 2

    # Construct the PrivacyLoss
    if (not tight_privacy_loss) and (not lower_bound):
        privacy_loss = SwitchingPrivacyLoss(
            epsilon_threshold=0.0,
            below_threshold_pl=MixtureGaussianPrivacyLoss(
                noise_multiplier,
                sensitivities,
                sampling_probs,
                adjacency_type=AdjacencyType.ADD
            ),
            above_threshold_pl=MixtureGaussianPrivacyLoss(
                noise_multiplier,
                sensitivities,
                sampling_probs,
                adjacency_type=AdjacencyType.REMOVE
            )
        )

        if p_bad_top < 1:
            bounds = privacy_loss.connect_dots_bounds()
            perfect_privacy_loss = PerfectPrivacyLoss(bounds)

            privacy_loss = WeightedSumPrivacyLoss(
                [privacy_loss, perfect_privacy_loss],
                [p_bad_top, 1 - p_bad_top]
            )

    else:
        if tight_privacy_loss and (p_bad_top < 1) and (instances_per_sequence > 1):
            raise NotImplementedError('Tight bi-level WOR-WR for more than one'
                                      'instance not derived yet.')

        sampling_probs *= p_bad_top
        sampling_probs[0] += (1 - p_bad_top)
        assert np.isclose(sampling_probs.sum(), 1.0)

        privacy_loss = SwitchingPrivacyLoss(
            epsilon_threshold=0.0,
            below_threshold_pl=MixtureGaussianPrivacyLoss(
                noise_multiplier,
                sensitivities,
                sampling_probs,
                adjacency_type=AdjacencyType.ADD
            ),
            above_threshold_pl=MixtureGaussianPrivacyLoss(
                noise_multiplier,
                sensitivities,
                sampling_probs,
                adjacency_type=AdjacencyType.REMOVE
            ))

    return privacy_loss


def create_privacy_loss_bottom_poisson(
        p_bad_top: float,
        min_sequence_length: int,
        instances_per_sequence: int,
        past_length: int,
        future_length: int,
        lead_time: int,
        min_past: int,
        min_future: int,
        noise_multiplier: float,
        neighboring_relation: dict,
        tight_privacy_loss: bool = False,
        future_target_noise_multiplier: float = 0,
        lower_bound: bool = False
) -> AdditiveNoisePrivacyLoss:
    """Create opacus privacy loss via Theorem E.8 or Lemma F.4.

    Args:
        p_bad_top (float): Probability of sampling any sequence on top level.
        min_sequence_length (int): Shortest sequence in dataset (L).
        instances_per_sequence (int): Subsequences per sequence (lambda)
        past_length (int): Context length (L_C)
        future_length (int): Forecast length (L_F)
        lead_time (int): Gap between context and forecast window.
            0 to reproduce paper results.
        min_past (int): (L_C - min_past) is how many zeros are padded at start.
            0 to reproduce paper results.
        min_future (int): (L_F - min_future) is how many zeros are padded at end
            L_F to reproduce paper results.
        noise_multiplier (float): Noise scale (sigma)
        neighboring_relation (dict): Considered neighboring relation
            - "level": "event" or "user"
            - "size": Number of modified elements, ("w" in paper)
            - "target_sensitivity": Maximum absolute change in modified elements ("v" in paper)
        tight_privacy_loss (bool): If False, will use loose upper bound from Lemma F.4 on privacy.
        future_target_noise_multiplier (float, optional): Forecast noise scale (sigma_F).
            Defaults to 0.
        lower_bound (bool, optional): If True, compute optimistic lower bound on privacy.
            Defaults to False.

    Returns:
        AdditiveNoisePrivacyLoss: opacus privacy loss.
    """

    future_start_interval_len, num_bad_sample_idx = _calc_support_size(
        min_sequence_length,
        past_length, future_length, lead_time,
        min_past, min_future, neighboring_relation
    )

    sample_rate_bottom = instances_per_sequence / future_start_interval_len
    assert sample_rate_bottom <= 1

    # Figure out probability of sampling different-sized subsets on bottom level
    sampling_probs = binom.pmf(np.arange(num_bad_sample_idx + 1),
                               num_bad_sample_idx, sample_rate_bottom)

    # means of privacy loss distribution
    sensitivities = np.arange(num_bad_sample_idx + 1)
    # Multiply by one, because grad can change from null to +clip or -clip when inserting/removing
    sensitivities *= 1

    # Amplification by label perturbation
    future_sensitivity = neighboring_relation.get('future_target_sensitivity', np.inf)
    if not np.isinf(future_sensitivity) and future_target_noise_multiplier > 0:
        raise NotImplementedError('Label perturbation only implemented for WR bottom level')

    # Construct the PrivacyLoss
    if (not tight_privacy_loss) and (not lower_bound):
        privacy_loss = DoubleMixtureGaussianPrivacyLoss(
            noise_multiplier,
            sensitivities, sensitivities,
            sampling_probs, sampling_probs
        )

        if p_bad_top < 1:
            bounds = privacy_loss.connect_dots_bounds()
            perfect_privacy_loss = PerfectPrivacyLoss(bounds)

            privacy_loss = WeightedSumPrivacyLoss(
                [privacy_loss, perfect_privacy_loss],
                [p_bad_top, 1 - p_bad_top]
            )

    else:
        if tight_privacy_loss and (p_bad_top < 1):
            raise NotImplementedError('Tight bi-level Poisson not derived yet.')

        sampling_probs *= p_bad_top
        sampling_probs[0] += (1 - p_bad_top)
        assert np.isclose(sampling_probs.sum(), 1.0)

        privacy_loss = DoubleMixtureGaussianPrivacyLoss(
            noise_multiplier,
            sensitivities, sensitivities,
            sampling_probs, sampling_probs
        )

    return privacy_loss


def create_privacy_loss(
            num_sequences: int,
            min_sequence_length: int,
            top_level_mode: str,
            instances_per_sequence: int,
            batch_size: int,
            past_length: int,
            future_length: int,
            lead_time: int,
            min_past: int,
            min_future: int,
            noise_multiplier: float,
            neighboring_relation: dict,
            tight_privacy_loss: bool = False,
            future_target_noise_multiplier: float = 0,
            bottom_level_mode: str = 'sampling_with_replacement',
            lower_bound: bool = False
        ) -> AdditiveNoisePrivacyLoss:
    """Create opacus privacy loss via specified subsampling scheme

    Args:
        num_sequence (int): Size of dataset N.
        min_sequence_length (int): Shortest sequence in dataset (L).
        top_level_mode (str): Top-level subsampling scheme to use. Should be in:
            - "iteration" (Algorithm 3 in paper)
            - "shuffling" (In each epoch, shuffle train set before iterating)
            - "sampling_without_replacement" (Algorithm 5 in paper)
        instances_per_sequence (int): Subsequences per sequence (lambda)
        batch_size (int): Number of subsequences in a batch (Lambda)
        past_length (int): Context length (L_C)
        future_length (int): Forecast length (L_F)
        lead_time (int): Gap between context and forecast window.
            0 to reproduce paper results.
        min_past (int): (L_C - min_past) is how many zeros are padded at start.
            0 to reproduce paper results.
        min_future (int): (L_F - min_future) is how many zeros are padded at end
            L_F to reproduce paper results.
        noise_multiplier (float): Noise scale (sigma)
        neighboring_relation (dict): Considered neighboring relation
            - "level": "event" or "user"
            - "size": Number of modified elements, ("w" in paper)
            - "target_sensitivity": Maximum absolute change in modified elements ("v" in paper)
        tight_privacy_loss (bool): If False, will use loose upper bound from Lemma F.4 on privacy.
        future_target_noise_multiplier (float, optional): Forecast noise scale (sigma_F).
            Defaults to 0.
        bottom_level_mode (str): Bottom-level sampling scheme.
            Should be in "sampling_with_replacement" or "sampling_poisson".
        lower_bound (bool, optional): If True, compute optimistic lower bound on privacy.
            Defaults to False.

    Returns:
        AdditiveNoisePrivacyLoss: opacus privacy loss.
    """

    p_bad_top = _calc_p_bad_top_level(
        num_sequences, top_level_mode, instances_per_sequence, batch_size)

    if bottom_level_mode == 'sampling_with_replacement':
        return create_privacy_loss_bottom_wr(
            p_bad_top, min_sequence_length, instances_per_sequence,
            past_length, future_length, lead_time, min_past, min_future,
            noise_multiplier, neighboring_relation, tight_privacy_loss,
            future_target_noise_multiplier, lower_bound)

    elif bottom_level_mode == 'sampling_poisson':
        return create_privacy_loss_bottom_poisson(
            p_bad_top, min_sequence_length, instances_per_sequence,
            past_length, future_length, lead_time, min_past, min_future,
            noise_multiplier, neighboring_relation, tight_privacy_loss,
            future_target_noise_multiplier, lower_bound)

    else:
        raise ValueError(f'{bottom_level_mode=} not supported.')


def _calc_p_bad_top_level(
    num_sequences: int,
    top_level_mode: str,
    instances_per_sequence: int,
    batch_size: int,
) -> float:
    """Probability of sampling a particular sequence on top level."""

    if top_level_mode == 'sampling_without_replacement':
        sample_size = batch_size // instances_per_sequence  # Lambda / lambda
        p_bad_top_level = hypergeom.sf(0, num_sequences, 1, sample_size)  # N' / N
    elif top_level_mode == 'sampling_poisson':
        expected_sample_size = batch_size // instances_per_sequence  # Lambda / lambda
        sample_rate = expected_sample_size / num_sequences  # N' / N
        p_bad_top_level = sample_rate
    elif top_level_mode in ['iteration', 'shuffling']:
        p_bad_top_level = 1.0
    else:
        raise ValueError(f'{top_level_mode=} not supported.')

    return p_bad_top_level


def _calc_support_size(
        min_sequence_length: int,
        past_length: int,
        future_length: int,
        lead_time: int,
        min_past: int,
        min_future: int,
        neighboring_relation: dict) -> tuple[int, int]:
    """Number of subsequences that can be constructed from a min_sequence_length sequence."""

    if lead_time != 0:
        raise NotImplementedError('Accounting does not support Splitter lead times yet.')

    if min_future < future_length:
        raise ValueError('min_future must be g.e.q. future_length '
                         'to avoid underlength targets.')

    if min_sequence_length < future_length:
        raise ValueError('min_sequence_length must be g.e.q. future length '
                         'to avoid underlength targets.')

    if min_past > min_sequence_length - min_future:
        raise ValueError('min_past must be l.e.q. min_sequence_length - min_future'
                         'to avoid undefined sampling distribution.')

    level = neighboring_relation.get('level', None)
    size = neighboring_relation.get('size', 0)

    if size < 1:
        raise ValueError('neighboring_relation.size=1 must be g.e.q. 1.')

    if level == 'event':
        num_bad_sample_idx = past_length + future_length - 1 + size
    elif level == 'user':
        num_bad_sample_idx = (past_length + future_length) * size
    else:
        raise ValueError('neighboring_relation.level must be in ["event", "user"]')

    future_start_interval = [min_past, min_sequence_length - min_future]
    future_start_interval_len = future_start_interval[1] - future_start_interval[0] + 1

    if future_start_interval_len < num_bad_sample_idx:
        raise NotImplementedError(
            'Accountant currently assumes that all sequences are long enough '
            'for modified element to be sampled at every position '
            'in context and target window.')

    return future_start_interval_len, num_bad_sample_idx


def _calc_gaussian_tvd(standard_deviation: float,
                       sensitivity: float) -> float:

    privacy_loss = GaussianPrivacyLoss(standard_deviation, sensitivity)

    tvd = float(privacy_loss.get_delta_for_epsilon(0))
    assert 0 <= tvd <= 1
    return tvd
