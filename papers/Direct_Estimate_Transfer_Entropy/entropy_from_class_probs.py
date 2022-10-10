import numpy as np
import math


def compute_conditional_entropy_from_cond_probs(pred_prob, sample_weights=None, eps=1e-5):

    assert not np.any(np.isnan(pred_prob))
    if sample_weights is not None:
        assert not np.any(np.isnan(sample_weights))
        if sample_weights.sum() == 0:
            sample_weights = None

    if sample_weights is not None:
        entropy = (-np.log(pred_prob+eps)*sample_weights).sum()/(sample_weights.sum())
    else:
        entropy = (-np.log(pred_prob+eps)).mean()

    assert not math.isnan(entropy), (sample_weights, pred_prob, eps, (np.log(pred_prob+eps)*sample_weights).sum(), sample_weights.sum())

    return entropy
