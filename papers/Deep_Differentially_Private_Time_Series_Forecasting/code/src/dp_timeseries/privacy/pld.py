import math
import numbers
from typing import Iterable, List, Sequence, Union

import numpy as np
import scipy
from dp_accounting.pld import common
from dp_accounting.pld.privacy_loss_distribution import (
    PrivacyLossDistribution, _create_pld_pmf_from_additive_noise)
from dp_accounting.pld.privacy_loss_mechanism import (
    AdditiveNoisePrivacyLoss, AdjacencyType, ConnectDotsBounds,
    MixtureGaussianPrivacyLoss, TailPrivacyLossDistribution)
from scipy import stats
from scipy.special import logsumexp


class SwitchingPrivacyLoss(AdditiveNoisePrivacyLoss):
    """Privacy loss when privacy profile has binary case distinction based on epsilon"""
    # TODO: Do fancy convex conjugate thing from Zhu et al. 2022
    # for all the NotImplementedError methods.

    def __init__(self,
                 epsilon_threshold: float,
                 below_threshold_pl: AdditiveNoisePrivacyLoss,
                 above_threshold_pl: AdditiveNoisePrivacyLoss):

        self.epsilon_threshold = epsilon_threshold
        self.below_threshold_pl = below_threshold_pl
        self.above_threshold_pl = above_threshold_pl

        if below_threshold_pl.discrete_noise != above_threshold_pl.discrete_noise:
            raise ValueError('PLs must be both discrete or both continuous.')

        self.discrete_noise = below_threshold_pl.discrete_noise

        if self.discrete_noise:
            raise NotImplementedError('Only continuous PLs supported currently.')

        if not np.isclose(
                below_threshold_pl.get_delta_for_epsilon(epsilon_threshold),
                above_threshold_pl.get_delta_for_epsilon(epsilon_threshold)):
            raise ValueError('Tradeoff functions must intersect at epsilon_threshold.')

    def mu_upper_cdf(self, x: Union[float, Iterable[float]]) -> Union[float, np.ndarray]:
        raise NotImplementedError(
            'SwitchingPL is currently only meant for use with connect_the_dots.')

    def mu_lower_log_cdf(self, x: Union[float, Iterable[float]]) -> Union[float, np.ndarray]:
        raise NotImplementedError(
            'SwitchingPL is currently only meant for use with connect_the_dots.')

    def get_delta_for_epsilon(
            self, epsilon: Union[float, List[float]]) -> Union[float, List[float]]:

        is_scalar = isinstance(epsilon, numbers.Number)
        epsilons = np.array([epsilon]) if is_scalar else np.asarray(epsilon)
        deltas = np.zeros_like(epsilons, dtype=float)

        below_threshold_mask = (epsilons < self.epsilon_threshold)
        above_threshold_mask = ~below_threshold_mask

        if below_threshold_mask.sum() > 0:
            deltas[below_threshold_mask] = self.below_threshold_pl.get_delta_for_epsilon(
                epsilons[below_threshold_mask])

        if above_threshold_mask.sum() > 0:
            deltas[above_threshold_mask] = self.above_threshold_pl.get_delta_for_epsilon(
                epsilons[above_threshold_mask])

        return float(deltas[0]) if is_scalar else deltas

    def privacy_loss_tail(self) -> TailPrivacyLossDistribution:
        raise NotImplementedError(
            'SwitchingPL is currently only meant for use with connect_the_dots.')

    def connect_dots_bounds(self) -> ConnectDotsBounds:
        below_threshold_bounds = self.below_threshold_pl.connect_dots_bounds()
        above_threshold_bounds = self.above_threshold_pl.connect_dots_bounds()

        epsilon_upper = max(below_threshold_bounds.epsilon_upper,
                            above_threshold_bounds.epsilon_upper)

        epsilon_lower = min(below_threshold_bounds.epsilon_lower,
                            above_threshold_bounds.epsilon_lower)

        return ConnectDotsBounds(epsilon_upper=epsilon_upper,
                                 epsilon_lower=epsilon_lower)

    def privacy_loss(self, x: float) -> float:
        raise NotImplementedError(
            'SwitchingPL is currently only meant for use with connect_the_dots.')

    def privacy_loss_without_subsampling(self, x: float) -> float:
        raise NotImplementedError(
            'SwitchingPL is currently only meant for use with connect_the_dots.')

    def inverse_privacy_loss(self, privacy_loss: float) -> float:
        raise NotImplementedError(
            'SwitchingPL is currently only meant for use with connect_the_dots.')

    def inverse_privacy_loss_without_subsampling(self,
                                                 privacy_loss: float) -> float:

        raise NotImplementedError(
            'SwitchingPL is currently only meant for use with connect_the_dots.')

    def noise_cdf(self, x: Union[float,
                  Iterable[float]]) -> Union[float, np.ndarray]:

        raise NotImplementedError(
            'SwitchingPL is currently only meant for use with connect_the_dots.')

    def noise_log_cdf(
            self, x: Union[float, Iterable[float]]) -> Union[float, np.ndarray]:

        raise NotImplementedError(
            'SwitchingPL is currently only meant for use with connect_the_dots.')

    @classmethod
    def from_privacy_guarantee(
            cls,
            privacy_parameters: common.DifferentialPrivacyParameters,
            sensitivity: float = 1,
            pessimistic_estimate: bool = True,
            sampling_prob: float = 1.0,
            adjacency_type: AdjacencyType = AdjacencyType.REMOVE) -> 'AdditiveNoisePrivacyLoss':

        raise NotImplementedError(
            'SwitchingPL is currently only meant for use with connect_the_dots.')


class PerfectPrivacyLoss(AdditiveNoisePrivacyLoss):
    """Privacy loss when both distributions in dominating pair are identical."""
    def __init__(self, bounds: ConnectDotsBounds):
        self.bounds = bounds

    def mu_upper_cdf(self, x: Union[float, Iterable[float]]) -> Union[float, np.ndarray]:
        raise NotImplementedError(
            'PerfectPrivacyLoss is currently only meant for use with connect_the_dots.')

    def mu_lower_log_cdf(self, x: Union[float, Iterable[float]]) -> Union[float, np.ndarray]:
        raise NotImplementedError(
            'PerfectPrivacyLoss is currently only meant for use with connect_the_dots.')

    def get_delta_for_epsilon(
            self, epsilon: Union[float, List[float]]) -> Union[float, List[float]]:

        is_scalar = isinstance(epsilon, numbers.Number)
        epsilons = np.array([epsilon]) if is_scalar else np.asarray(epsilon)
        deltas = np.zeros_like(epsilons, dtype=float)

        deltas[epsilons < 0] = 1 - np.exp(epsilons)

        return float(deltas[0]) if is_scalar else deltas

    def privacy_loss_tail(self) -> TailPrivacyLossDistribution:
        raise NotImplementedError(
            'PerfectPrivacyLoss is currently only meant for use with connect_the_dots.')

    def connect_dots_bounds(self) -> ConnectDotsBounds:
        return self.bounds

    def privacy_loss(self, x: float) -> float:
        raise NotImplementedError(
            'PerfectPrivacyLoss is currently only meant for use with connect_the_dots.')

    def privacy_loss_without_subsampling(self, x: float) -> float:
        raise NotImplementedError(
            'PerfectPrivacyLoss is currently only meant for use with connect_the_dots.')

    def inverse_privacy_loss(self, privacy_loss: float) -> float:
        raise NotImplementedError(
            'PerfectPrivacyLoss is currently only meant for use with connect_the_dots.')

    def inverse_privacy_loss_without_subsampling(self,
                                                 privacy_loss: float) -> float:

        raise NotImplementedError(
            'PerfectPrivacyLoss is currently only meant for use with connect_the_dots.')

    def noise_cdf(self, x: Union[float,
                  Iterable[float]]) -> Union[float, np.ndarray]:

        raise NotImplementedError(
            'PerfectPrivacyLoss is currently only meant for use with connect_the_dots.')

    def noise_log_cdf(
            self, x: Union[float, Iterable[float]]) -> Union[float, np.ndarray]:

        raise NotImplementedError(
            'PerfectPrivacyLoss is currently only meant for use with connect_the_dots.')

    @classmethod
    def from_privacy_guarantee(
            cls,
            privacy_parameters: common.DifferentialPrivacyParameters,
            sensitivity: float = 1,
            pessimistic_estimate: bool = True,
            sampling_prob: float = 1.0,
            adjacency_type: AdjacencyType = AdjacencyType.REMOVE) -> 'AdditiveNoisePrivacyLoss':

        raise NotImplementedError(
            'PerfectPrivacyLoss is currently only meant for use with connect_the_dots.')


class WeightedSumPrivacyLoss(AdditiveNoisePrivacyLoss):
    """Privacy loss when privacy profile is a weighted sum."""
    # TODO: Think about whether we can resolve some of the NotImplementedErrors.

    def __init__(self,
                 privacy_losses: Sequence[AdditiveNoisePrivacyLoss],
                 weights: Sequence[float]):

        self.privacy_losses = privacy_losses
        self.weights = weights

        if not np.isclose(np.sum(weights), 1):
            raise ValueError('Weights must add up to 1.0 for valid privacy profile.')

        if len(privacy_losses) != len(weights):
            raise ValueError('privacy_losses and weights must have equal length.')

        num_components = len(privacy_losses)

        if np.sum([pl.discrete_noise for pl in privacy_losses]) not in [0, num_components]:
            raise ValueError('PLs must be all discrete or all continuous.')

        self.discrete_noise = (
            (np.sum([pl.discrete_noise for pl in privacy_losses])) == num_components)

        if self.discrete_noise:
            raise NotImplementedError('Only continuous PLs supported currently.')

    def mu_upper_cdf(self, x: Union[float, Iterable[float]]) -> Union[float, np.ndarray]:
        raise NotImplementedError(
            'WeightedSumPrivacyLoss is currently only meant for use with connect_the_dots.')

    def mu_lower_log_cdf(self, x: Union[float, Iterable[float]]) -> Union[float, np.ndarray]:
        raise NotImplementedError(
            'WeightedSumPrivacyLoss is currently only meant for use with connect_the_dots.')

    def get_delta_for_epsilon(
            self, epsilon: Union[float, List[float]]) -> Union[float, List[float]]:

        is_scalar = isinstance(epsilon, numbers.Number)
        epsilons = np.array([epsilon]) if is_scalar else np.asarray(epsilon)
        deltas = np.zeros_like(epsilons, dtype=float)

        for privacy_loss, weight in zip(self.privacy_losses, self.weights):
            deltas += weight * privacy_loss.get_delta_for_epsilon(epsilons)

        return float(deltas[0]) if is_scalar else deltas

    def privacy_loss_tail(self) -> TailPrivacyLossDistribution:
        raise NotImplementedError(
            'WeightedSumPrivacyLoss is currently only meant for use with connect_the_dots.')

    def connect_dots_bounds(self) -> ConnectDotsBounds:
        connect_dots_bounds = [pl.connect_dots_bounds() for pl in self.privacy_losses]

        epsilon_upper = np.max([cdb.epsilon_upper for cdb in connect_dots_bounds])
        epsilon_lower = np.min([cdb.epsilon_lower for cdb in connect_dots_bounds])

        return ConnectDotsBounds(epsilon_upper=epsilon_upper,
                                 epsilon_lower=epsilon_lower)

    def privacy_loss(self, x: float) -> float:
        raise NotImplementedError(
            'WeightedSumPrivacyLoss is currently only meant for use with connect_the_dots.')

    def privacy_loss_without_subsampling(self, x: float) -> float:
        raise NotImplementedError(
            'WeightedSumPrivacyLoss is currently only meant for use with connect_the_dots.')

    def inverse_privacy_loss(self, privacy_loss: float) -> float:
        raise NotImplementedError(
            'WeightedSumPrivacyLoss is currently only meant for use with connect_the_dots.')

    def inverse_privacy_loss_without_subsampling(self,
                                                 privacy_loss: float) -> float:

        raise NotImplementedError(
            'WeightedSumPrivacyLoss is currently only meant for use with connect_the_dots.')

    def noise_cdf(self, x: Union[float,
                  Iterable[float]]) -> Union[float, np.ndarray]:

        raise NotImplementedError(
            'WeightedSumPrivacyLoss is currently only meant for use with connect_the_dots.')

    def noise_log_cdf(
            self, x: Union[float, Iterable[float]]) -> Union[float, np.ndarray]:

        raise NotImplementedError(
            'WeightedSumPrivacyLoss is currently only meant for use with connect_the_dots.')

    @classmethod
    def from_privacy_guarantee(
            cls,
            privacy_parameters: common.DifferentialPrivacyParameters,
            sensitivity: float = 1,
            pessimistic_estimate: bool = True,
            sampling_prob: float = 1.0,
            adjacency_type: AdjacencyType = AdjacencyType.REMOVE) -> 'AdditiveNoisePrivacyLoss':

        raise NotImplementedError(
            'WeightedSumPrivacyLoss is currently only meant for use with connect_the_dots.')


class DoubleMixtureGaussianPrivacyLoss(AdditiveNoisePrivacyLoss):
    """Privacy loss of the Mixture of Gaussians mechanism with two mixtures.

    That is, let mu be the Gaussian noise PDF with sigma = standard_deviation.
    The privacy loss distribution is generated as follows:
    - Let mu_upper(x) := sum over i of sampling_probs[i] *
      mu(x + sensitivities[i])
    - Let mu_lower(x) := sum over i of sampling_probs[i] *
      mu(x - sensitivities[i])
    - Sample x ~ mu_upper and let the privacy loss be
        ln(mu_upper(x) / mu_lower(x)).
    Reasoning for these signs: So that privacy loss is non-increasing
    """

    def __init__(  # pylint: disable=super-init-not-called
      self,
      standard_deviation: float,
      sensitivities_upper: Sequence[float],
      sensitivities_lower: Sequence[float],
      sampling_probs_upper: Sequence[float],
      sampling_probs_lower: Sequence[float],
      pessimistic_estimate: bool = True,
      log_mass_truncation_bound: float = -50,
    ) -> None:
        """Initializes the privacy loss of the MoG mechanism.

        Args:
        standard_deviation: The standard_deviation of the Gaussian distribution.
        sensitivities_upper: The support of the first sensitivity distribution. Must be the
            same length as sampling_probs_upper, and both should be 1D.
        sensitivities_lower: The support of the second sensitivity distribution. Must be the
            same length as sampling_probs_lower, and both should be 1D.
        sampling_probs_upper: The probabilities associated with the first sensitivities.
        sampling_probs_lower: The probabilities associated with the second sensitivities.
        pessimistic_estimate: A value indicating whether the rounding is done in
            such a way that the resulting epsilon-hockey stick divergence
            computation gives an upper estimate to the real value.
        log_mass_truncation_bound: The ln of the probability mass that might be
            discarded from the noise distribution. The larger this number, the more
            error it may introduce in divergence calculations.

        Raises:
        ValueError: If args are invalid, e.g. standard_deviation is negative or
        sensitivities and sampling_probs are different lengths.
        """
        if standard_deviation <= 0:
            raise ValueError(
                'Standard deviation is not a positive real number: '
                f'{standard_deviation}'
            )

        if log_mass_truncation_bound > 0:
            raise ValueError(
                'Log mass truncation bound is not a non-positive real '
                f'number: {log_mass_truncation_bound}'
            )

        if ((len(sampling_probs_upper) != len(sensitivities_upper))
           or (len(sampling_probs_lower) != len(sensitivities_lower))):

            raise ValueError(
                'sensitivities and sampling_probs must have the same length'
            )

        non_zero_indices_upper = np.asarray(sampling_probs_upper) != 0.0
        sensitivities_upper = np.asarray(sensitivities_upper)[non_zero_indices_upper]
        sampling_probs_upper = np.asarray(sampling_probs_upper)[non_zero_indices_upper]
        non_zero_indices_lower = np.asarray(sampling_probs_lower) != 0.0
        sensitivities_lower = np.asarray(sensitivities_lower)[non_zero_indices_lower]
        sampling_probs_lower = np.asarray(sampling_probs_lower)[non_zero_indices_lower]

        if np.any(sensitivities_upper < 0) or np.any(sensitivities_lower < 0):
            raise ValueError(
                'Sensitivities contain a negative number.'
            )

        if (sensitivities_upper.max() == 0.0) and (sensitivities_lower.max() == 0.0):
            raise ValueError('Must have at least one positive sensitivity.')

        if not (math.isclose(sum(sampling_probs_upper), 1)
                and
                math.isclose(sum(sampling_probs_lower), 1)):
            raise ValueError(
                'Probabilities do not add up to 1'
            )

        if (np.any((sampling_probs_upper <= 0) | (sampling_probs_upper > 1))
           or np.any((sampling_probs_lower <= 0) | (sampling_probs_lower > 1))):

            raise ValueError(
                'Sampling probabilities are in (0,1]'
            )

        self.discrete_noise = False
        # self.adjacency_type = adjacency_type
        self.sampling_probs_upper = sampling_probs_upper
        self.sensitivities_upper = sensitivities_upper
        self.sampling_probs_lower = sampling_probs_lower
        self.sensitivities_lower = sensitivities_lower
        self._standard_deviation = standard_deviation
        self._variance = standard_deviation**2
        self._pessimistic_estimate = pessimistic_estimate
        self._log_mass_truncation_bound = log_mass_truncation_bound

        # Constant properties.
        self._log_sampling_probs_upper = np.log(self.sampling_probs_upper)
        self._pos_sampling_probs_upper = self.sampling_probs_upper[self.sensitivities_upper > 0.0]
        self._sampling_prob_upper = np.clip(self._pos_sampling_probs_upper.sum(), 0, 1)
        self._max_sens_upper = self.sensitivities_upper[self.sampling_probs_upper > 0].max()

        self._log_sampling_probs_lower = np.log(self.sampling_probs_lower)
        self._pos_sampling_probs_lower = self.sampling_probs_lower[self.sensitivities_lower > 0.0]
        self._sampling_prob_lower = np.clip(self._pos_sampling_probs_lower.sum(), 0, 1)
        self._max_sens_lower = self.sensitivities_lower[self.sampling_probs_lower > 0].max()

        self._gaussian_random_variable = stats.norm(scale=standard_deviation)

    def mu_upper_cdf(
        self, x: Union[float, Iterable[float]]
    ) -> Union[float, np.ndarray]:
        """Computes the cumulative density function of the mu_upper distribution.

        mu_upper(x) := sum of sampling_probs[i] * mu(x + sensitivities[i])

        Args:
        x: the point or points at which the cumulative density function is to be
            calculated.

        Returns:
        The cumulative density function of the mu_upper distribution at x, i.e.,
        the probability that mu_upper is less than or equal to x.
        """
        points_per_sens = np.add.outer(np.atleast_1d(x), self.sensitivities_upper)
        output = (self.noise_cdf(points_per_sens) * self.sampling_probs_upper).sum(axis=1)

        if isinstance(x, numbers.Number):
            return output[0]
        else:
            return output

    def mu_lower_log_cdf(
        self, x: Union[float, Iterable[float]]
    ) -> Union[float, np.ndarray]:
        """Computes log cumulative density function of the mu_lower distribution.

        mu_lower(x) := sum of sampling_probs[i] * mu(x - sensitivities[i])

        Args:
        x: the point or points at which the log of the cumulative density function
            is to be calculated.

        Returns:
        The log of the cumulative density function of the mu_lower distribution at
        x, i.e., the log of the probability that mu_lower is less than or equal to
        x.
        """
        points_per_sens = np.add.outer(np.atleast_1d(x), -self.sensitivities_lower)
        logcdf_per_sens = self.noise_log_cdf(points_per_sens)

        output = scipy.special.logsumexp(
            logcdf_per_sens, axis=1, b=self.sampling_probs_lower
        )
        if isinstance(x, numbers.Number):
            return output[0]
        else:
            return output

    def get_delta_for_epsilon(
      self, epsilon: Union[float, Sequence[float]]
    ) -> Union[float, list[float]]:
        """Computes the epsilon-hockey stick divergence of the mechanism.

        Args:
        epsilon: the epsilon, or list-like object of epsilon values, in
            epsilon-hockey stick divergence. Should be non-decreasing if list-like.

        Returns:
        A non-negative real number which is the epsilon-hockey stick divergence of
        the mechanism, or a numpy array if epsilon is list-like.
        """
        epsilons = np.atleast_1d(epsilon)
        if not np.all(epsilons[1:] >= epsilons[:-1]):
            raise ValueError(f'Epsilon values must be non-decreasing: {epsilons}')

        deltas = np.zeros_like(epsilons, dtype=float)            

        # Corresponds to old ADD case
        if (self._sampling_prob_upper == 0.0) and (self._sampling_prob_lower != 1.0):
            inverse_indices = epsilons < -np.log1p(-self._sampling_prob_lower)

        # Corresponds to old REMOVE case
        elif (self._sampling_prob_lower == 0.0) and (self._sampling_prob_upper != 1.0):
            inverse_indices = epsilons > np.log1p(-self._sampling_prob_upper)
            other_indices = np.logical_not(inverse_indices)
            deltas[other_indices] = -np.expm1(epsilons[other_indices])

        else:
            inverse_indices = np.full_like(epsilons, True, dtype=bool)

        x_cutoffs = self.inverse_privacy_losses(epsilons[inverse_indices])

        deltas[inverse_indices] = self.mu_upper_cdf(x_cutoffs) - np.exp(
            epsilons[inverse_indices] + self.mu_lower_log_cdf(x_cutoffs)
        )

        # Clip delta values to lie in [0,1] (to avoid numerical errors)
        deltas = np.clip(deltas, 0, 1)
        if isinstance(epsilon, numbers.Number):
            return float(deltas)
        else:
            # For numerical stability reasons, deltas may not be non-increasing. This
            # is fixed post-hoc at small cost in accuracy.
            for i in reversed(range(deltas.shape[0] - 1)):
                deltas[i] = max(deltas[i], deltas[i + 1])
        return deltas

    def privacy_loss_tail(
        self, precision: float = 1e-4
    ) -> TailPrivacyLossDistribution:
        """Computes the privacy loss at the tail of the random-sensitivity Gaussian.

        For ADD adjacency type: The upper distribution is a single Gaussian and we
        can exactly compute the tails easily.

        For REMOVE adjacency type:  We set upper_x_truncation such that
        CDF(upper_x_truncation) = 1 - 0.5 * exp(log_mass_truncation_bound). It is
        worthwhile to spend some up-front computation getting a more precise value
        for lower_x_truncation to save computation later on. So we binary search
        over the interval [-upper_x_truncation - max(sensitivities),
        -upper_x_truncation] for the point where the cdf of mu_upper is
        0.5 * exp(log_mass_truncation_bound). Since we're binary searching over a
        continuous domain, we proceed until the width of the binary search
        interval is at most some small precision, and then set lower_x_truncation
        to be the left endpoint of this interval.

        Args:
        precision: The additive error we will compute the truncation values
            within. That is, when we binary search for log_mass_truncation_bound in
            the REMOVE case, we terminate the binary search when the interval has
            length at most precision, and then use the more conservative endpoint
            of the interval as our truncation value.

        Returns:
        A TailPrivacyLossDistribution instance representing the tail of the
        privacy loss distribution.
        """
        tail_mass = 0.5 * np.exp(self._log_mass_truncation_bound)
        z_value = self._gaussian_random_variable.ppf(tail_mass)
        upper_x_truncation = -z_value
        # Corresponds to old ADD case
        if self._sampling_prob_upper == 0.0:
            lower_x_truncation = z_value
        else:  # Case: self.adjacency_type == AdjacencyType.REMOVE
            lower_x_truncation = common.inverse_monotone_function(
                self.mu_upper_cdf,
                tail_mass,
                common.BinarySearchParameters(
                    z_value - self._max_sens_upper,
                    z_value,
                    tolerance=precision
                ),
                increasing=True,
            )
        if self._pessimistic_estimate:
            tail_probability_mass_function = {
                math.inf: self.mu_upper_cdf(lower_x_truncation),
                self.privacy_loss(upper_x_truncation): 1 - self.mu_upper_cdf(
                    upper_x_truncation
                ),
            }
        else:
            tail_probability_mass_function = {
                self.privacy_loss(lower_x_truncation): self.mu_upper_cdf(
                    lower_x_truncation
                ),
            }

        return TailPrivacyLossDistribution(
            lower_x_truncation, upper_x_truncation, tail_probability_mass_function
        )

    def connect_dots_bounds(self) -> ConnectDotsBounds:
        """Computes the bounds on epsilon values to use in connect-the-dots algorithm.

        Returns:
        A ConnectDotsBounds instance containing upper and lower values of
        epsilon to use in connect-the-dots algorithm.
        """
        tail_pld = self.privacy_loss_tail()

        return ConnectDotsBounds(
            epsilon_upper=self.privacy_loss(tail_pld.lower_x_truncation),
            epsilon_lower=self.privacy_loss(tail_pld.upper_x_truncation),
        )

    def privacy_loss(self, x: float) -> float:
        """Computes the privacy loss at a given point `x`."""

        p_upper = logsumexp(stats.norm.logpdf(x, loc=-1 * self.sensitivities_upper,
                                              scale=self._standard_deviation),
                            b=self.sampling_probs_upper)

        p_lower = logsumexp(stats.norm.logpdf(x, loc=self.sensitivities_lower,
                                              scale=self._standard_deviation),
                            b=self.sampling_probs_lower)

        return p_upper - p_lower

    def privacy_loss_without_subsampling(self, x: float) -> float:
        raise NotImplementedError(
            'DoubleMixtureGaussianPrivacyLoss uses multiple sensitivities, so '
            'privacy loss without subsampling is ill-defined. Use '
            'privacy_loss_for_single_gaussian instead.'
        )

    def inverse_privacy_loss_without_subsampling(
      self, privacy_loss: float
    ) -> float:
        raise NotImplementedError(
            'MixtureGaussianPrivacyLoss uses multiple sensitivities, so '
            'inverse_privacy_loss_without_subsampling is ill-defined. Use '
            'inverse_privacy_loss_for_single_gaussian instead.'
        )

    def inverse_privacy_loss(
        self, privacy_loss: float, precision: float = 1e-6
    ) -> float:
        """(Approximately) computes the inverse of a given privacy loss.

        Technically, this method can be sped up by rewriting the logic in
        inverse_privacy_losses to take advantage of the fact that we have a
        single privacy loss rather than a list. However, this method is only written
        to complete the abstract class, and the process of generating a PLD from
        this class won't ever call this method. So, we have chosen the simple but
        inefficient implementation of calling inverse_privacy_losses.

        Args:
        privacy_loss: the privacy loss value.
        precision: Precision of the output.

        Returns:
        The largest float x such that the privacy loss at x is at least
        privacy_loss, rounded down to the nearest multiple of precision if
        we are using pessimistic estimates, and otherwise rounded up.
        """
        return float(
            self.inverse_privacy_losses(np.atleast_1d(privacy_loss), precision)[0]
        )

    def inverse_privacy_losses(
        self,
        privacy_losses: np.ndarray,
        precision: float = 1e-6,
    ) -> np.ndarray:
        """(Approximately) computes the inverse of a list of privacy losses.

        Unlike subsampled Gaussians, for mixture Gaussians the privacy loss does
        not have a closed-form inverse, to the best of our knowledge. So, we use
        binary search. This is the main bottleneck in this library, so we optimize
        it by doing one binary search for all values in privacy losses rather than a
        separate binary search for each. This way, we avoid recomputing the privacy
        loss at the same point across different binary searches.

        Args:
        privacy_losses: the privacy losses we wish to invert, in increasing order.
        precision: Precision of the output. In particular, for each entry l in
            privacy_losses, we output the smallest multiple of precision, x, such
            that the privacy loss at x is at most l. This ensures (i) given a
            monotonic privacy_losses, we return a monotonic list of xs, and (ii) the
            approximation results in an overestimate of epsilon, i.e. the final
            epsilon reported is valid.

        Returns:
        For each l in privacy_losses, the smallest multiple of precision, x, such
        that the privacy loss at x is at most l.
        """
        if not (np.diff(privacy_losses) >= 0).all():
            raise ValueError(
                f'Expected non-decreasing privacy_losses, got: {privacy_losses}.'
            )
        if len(privacy_losses) == 0:  # pylint: disable=g-explicit-length-test
            return np.ndarray([])

        # If we have a non-zero probability of choosing sensitivity = 0, then the
        # privacy loss does not take on all values in [-inf, inf], and so we need to
        # make sure all values in privacy_losses are in the proper range for the
        # given adjacency type.

        # Some privacy losses might be close to the privacy loss at x = +/-inf, in
        # which case we report the corresponding infinity for them.
        min_pl = privacy_losses[0]
        max_pl = privacy_losses[-1]

        # Corresponds to old ADD case
        if (self._sampling_prob_upper == 0.0) and (self._sampling_prob_lower != 1.0):
            log_1m_prob = (
                math.log1p(-self._sampling_prob_lower)
            )
            if max_pl > -log_1m_prob:
                raise ValueError(
                    f'max of privacy_losses ({max_pl}) is larger than '
                    f'-log(1 - sampling_prob)={-log_1m_prob}.'
                )
            finite_indices = np.logical_not(np.isclose(privacy_losses, -log_1m_prob))
            max_pl = np.max(privacy_losses[finite_indices])

        # Corresponds to old REMOVE case
        elif (self._sampling_prob_lower == 0.0) and (self._sampling_prob_upper != 1.0):
            log_1m_prob = (math.log1p(-self._sampling_prob_upper))

            if min_pl <= log_1m_prob:
                raise ValueError(
                    f'min of privacy_losses ({min_pl}) is smaller than '
                    f'log(1 - sampling_prob)={log_1m_prob}'
                )
            finite_indices = np.logical_not(np.isclose(privacy_losses, log_1m_prob))
            min_pl = np.min(privacy_losses[finite_indices])

        else:
            finite_indices = np.full_like(privacy_losses, True, dtype=bool)

        # Instead of the fancy method from the original paper,
        # we just iteratively grow our binary search boundaries
        # until they contain minimum and maximum privacy loss

        left_bound = -1
        while True:
            loss = self.privacy_loss(left_bound)
            if not (loss < max_pl):
                break
            left_bound *= 2

        right_bound = 1
        while self.privacy_loss(right_bound) > min_pl:
            right_bound *= 2

        bounds = (left_bound, right_bound)

        # Corresponds to old add case
        if (self._sampling_prob_upper == 0.0) and (self._sampling_prob_lower != 1.0):
            output = np.full_like(privacy_losses, -np.inf)

        else:
            output = np.full_like(privacy_losses, np.inf)

        output[finite_indices] = self._inverse_privacy_losses_with_range(
            privacy_losses[finite_indices], bounds, precision
        )

        return output

    def _inverse_privacy_losses_with_range(
      self,
      privacy_losses: np.ndarray,
      bounds: tuple[float, float],
      precision: float = 1e-6,
    ) -> Iterable[float]:
        """Helper method for performing binary search in inverse_privacy_losses.

        Args:
        privacy_losses: the privacy losses we wish to invert.
        bounds: Range to search over, i.e. the inverses are in the range
            [bounds[0], bounds[1]].
        precision: Precision of the output; in particular, for each entry l in
            privacy_losses, we output the smallest multiple of precision, x, such
            that the privacy loss at x is at most l. This ensures (i) given a
            monotonic privacy_losses, we return a monotonic list of xs, and (ii) the
            approximation results in an overestimate of epsilon, i.e. the final
            epsilon we report is a valid epsilon.

        Returns:
        For each l in privacy_losses, the smallest multiple of precision, x, such
        that the privacy loss at x is at most l.
        """
        if len(privacy_losses) == 0:  # pylint: disable=g-explicit-length-test
            return []
        if bounds[1] - bounds[0] <= precision:
            return np.repeat(
                np.floor(bounds[1] / precision) * precision, len(privacy_losses)
            )

        mid = (bounds[0] + bounds[1]) / 2
        pl_split = self.privacy_loss(mid)
        lower_indices = privacy_losses < pl_split
        higher_indices = privacy_losses >= pl_split
        output = np.zeros_like(privacy_losses)
        output[lower_indices] = self._inverse_privacy_losses_with_range(
            privacy_losses[lower_indices], (mid, bounds[1]), precision
        )
        output[higher_indices] = self._inverse_privacy_losses_with_range(
            privacy_losses[higher_indices], (bounds[0], mid), precision
        )
        return output

    def noise_cdf(
        self, x: Union[float, Iterable[float]]
    ) -> Union[float, np.ndarray]:
        """Computes the cumulative density function of the Gaussian distribution.

        Args:
        x: the point or points at which the cumulative density function is to be
        calculated.

        Returns:
        The cumulative density function of the Gaussian noise at x, i.e., the
        probability that the Gaussian noise is less than or equal to x.
        """
        return self._gaussian_random_variable.cdf(x)

    def noise_log_cdf(
        self, x: Union[float, Iterable[float]]
    ) -> Union[float, np.ndarray]:
        """Computes log of cumulative density function of the Gaussian distribution.

        Args:
        x: the point or points at which the log cumulative density function is to
            be calculated.

        Returns:
        The log cumulative density function of the Gaussian noise at x, i.e., the
        log of the probability that the Gaussian noise is less than or equal to x.
        """
        return self._gaussian_random_variable.logcdf(x)

    @classmethod
    def from_privacy_guarantee(
        cls,
        privacy_parameters: common.DifferentialPrivacyParameters,
        sensitivity: float = 1,
        pessimistic_estimate: bool = True,
        sampling_prob: float = 1.0,
        adjacency_type: AdjacencyType = AdjacencyType.REMOVE,
    ) -> 'MixtureGaussianPrivacyLoss':
        raise NotImplementedError(
            'MixtureGaussianPrivacy loss cannot be uniquely '
            'instantiated from privacy parameters.'
        )
