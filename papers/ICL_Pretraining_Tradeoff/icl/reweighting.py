import jax
import jax.numpy as jnp
from jax import Array
from typing import Dict, Any
import chex
from functools import partial

def compute_alpha_schedule(t: int, T: int, alpha0, T_ramp) -> float:
    """
    Compute alpha value according to linear schedule:
    α_t = 1 - (1 - α_0) * (1 - t / T_ramp)
    
    Args:
        t: Current step
        T: Total number of steps
        alpha0: Initial alpha value (default: 0.5)
        T_ramp_ratio: Ratio of total steps for ramping (default: 0.4, so T_ramp = 0.4*T)
    
    Returns:
        Alpha value for current step
    """
    return jax.lax.select(
            t >= T_ramp,
            1.0,
            1.0 - (1.0 - alpha0) * (1.0 - t / T_ramp)
            )

def soft_clip_log(log_weights: Array, alpha: float) -> Array:
    """
    Apply soft clipping in log domain: ψ_α(w) = α*w + (1-α)*sqrt(w)
    Implemented as: log(ψ_α(exp(log_w))) = log(α*exp(log_w) + (1-α)*exp(log_w / 2))
    
    Args:
        log_weights: Log weights of shape (B,)
        alpha: Clipping parameter in [0.5, 1]
    
    Returns:
        Log of soft-clipped weights
    """
    soft_clipped_weights = jnp.logaddexp(
            jnp.log(alpha) + log_weights,
            jnp.log1p(-alpha) + 0.5 * log_weights
            )
    chex.assert_shape(soft_clipped_weights, log_weights.shape)
    return soft_clipped_weights 

def soft_hard_clip(log_weights: Array) -> Array:
    """
    Apply hard clipping:
    - Upper bound: 99.5% percentile
    - Lower bound: 1e-6 * median
    
    Args:
        weights: Input weights of shape (B,)
    
    Returns:
        Hard-clipped weights
    """
    upper_bound = jnp.percentile(log_weights, 99.5)
    lower_bound = jnp.log(1e-6) + jnp.median(log_weights)
    
    return jnp.clip(log_weights, lower_bound, None)

def renormalize_weights(log_weights: Array) -> Array:
    """
    Renormalize weights to sum to 1.
    
    Args:
        weights: Input weights of shape (B,)
    
    Returns:
        Renormalized weights
    """
    return jax.nn.softmax(log_weights, axis=0)

def compute_gini_coefficient(weights: Array) -> float:
    """
    Compute Gini coefficient for inequality measurement.
    G ∈ [0, 1] where 0 = perfect equality, 1 = perfect inequality
    
    Args:
        weights: Array of weights
    
    Returns:
        Gini coefficient
    """
    # Sort weights in ascending order
    sorted_weights = jnp.sort(weights)
    n = len(sorted_weights)
    
    # Compute Gini coefficient using the standard formula
    index = jnp.arange(1, n + 1)
    gini = 2 * jnp.sum(index * sorted_weights) / (n * jnp.sum(sorted_weights)) - (n + 1) / n
    
    return gini


def kl_divergence_from_uniform(weights: Array) -> float:
    """
    Compute KL divergence from uniform distribution: D(weights || uniform)
    
    Args:
        weights: Array of weights (should sum to 1)
    
    Returns:
        KL divergence (0 = perfectly uniform, higher = more non-uniform)
    """
    n = weights.shape[0]
    uniform = jnp.ones(n) / n
    return jnp.sum(jax.scipy.special.kl_div(weights, uniform))

def compute_diagnostics(weights: Array) -> Dict[str, Any]:
    """
    Compute comprehensive diagnostics for weights.
    
    Args:
        weights: Input weights of shape (B,)
    
    Returns:
        Dictionary with diagnostic information
    """
    diagnostics = {
        # Basic statistics
        "max": jnp.max(weights),
        "median": jnp.median(weights),
        "mean": jnp.mean(weights),
        "std": jnp.std(weights),
        "min": jnp.min(weights),
        
        # Percentiles
        "P95": jnp.percentile(weights, 95),
        "P99": jnp.percentile(weights, 99),
        "P99.5": jnp.percentile(weights, 99.5),
        
        # Coefficient of variation
        "coef_variation": jnp.std(weights) / jnp.mean(weights),
        
        # Gini coefficient for inequality
        "gini": compute_gini_coefficient(weights),
        
        # KL divergence from uniform distribution
        "kl_from_uniform": kl_divergence_from_uniform(weights),
        
        # Effective Sample Size (ESS) per batch
        "ess": (
            jnp.sum(weights) ** 2 / jnp.sum(weights ** 2) / weights.shape[0]
            ),
        
        # Additional useful metrics
        "sum": jnp.sum(weights)
    }
    
    return diagnostics

@partial(jax.jit, static_argnames=["T","T_ramp_ratio", "normalize"])
def process_log_weights(log_weights: Array, t: int, T: int, alpha0: float = 0.5, normalize=False,
                       T_ramp_ratio: float = 0.4) -> tuple[Array, Dict[str, Any]]:
    """
    Main function to process log weights with soft clipping, hard clipping, and renormalization.
    
    Args:
        log_weights: Log of weights of size B
        t: Current step
        T: Total number of steps
        alpha0: Initial alpha value for soft clipping schedule (default: 0.5)
        T_ramp_ratio: Ratio of total steps for alpha ramping (default: 0.4)
    
    Returns:
        Tuple of (processed_weights, diagnostics)
    """
    # Validate inputs
    chex.assert_rank(log_weights, {1, 2})
    if log_weights.ndim == 2:
        chex.assert_axis_dimension(log_weights, 1, 1)

    # 1. Compute alpha according to schedule
    T_ramp = int(T_ramp_ratio * T)
    alpha = compute_alpha_schedule(t, T, alpha0, T_ramp)
    
    # 2. Apply soft clipping
    weights_soft = soft_clip_log(log_weights, alpha)
    
    # 3. Apply hard clipping
    weights_hard = soft_hard_clip(weights_soft)
    
    # 4. Renormalize to sum to 1
    if normalize:
        weights_final = renormalize_weights(weights_hard)
    else:
        weights_final = jnp.exp(weights_hard) / len(weights_hard)
    
    # 5. Compute diagnostics
    original_weights = renormalize_weights(log_weights) 
    soft_clipped_weights = renormalize_weights(weights_soft)
    hard_clipped_weights = renormalize_weights(weights_hard)

    diagnostics = {
        "alpha": alpha,
        "original": compute_diagnostics(original_weights),
        "soft_clipped": compute_diagnostics(soft_clipped_weights),
        "hard_clipped": compute_diagnostics(hard_clipped_weights),
        "final": compute_diagnostics(weights_final),
        "clipping_bounds": {
            "hard_clip_lower": 1e-6 * jnp.median(soft_clipped_weights),
            "hard_clip_upper": jnp.percentile(soft_clipped_weights, 99.5),
            "n_clipped_lower": jnp.sum(soft_clipped_weights < (1e-6 * jnp.median(soft_clipped_weights))),
            "n_clipped_upper": jnp.sum(soft_clipped_weights > jnp.percentile(soft_clipped_weights, 99.5))
        }
    }
    
    return weights_final, diagnostics
