import importlib
import numpy as np
from hyperopt import STATUS_OK

import entropy_from_class_probs as ecp
importlib.reload(ecp)

import discriminative_model_for_te_estimation as dte
importlib.reload(dte)


class ClfTransferEntropyEstimation(dte.DiscriminatorTransferEntropyEstimation):

    def __init__(
            self,
            num_cores=1, debug=False,
            objective=None,
            lambda_entropy_predictor_cond_hashcodes=1.0,
            seed=0,
    ):
        super(ClfTransferEntropyEstimation, self).__init__(
            num_cores=num_cores,
            debug=debug,
            lambda_entropy_predictor_cond_hashcodes=lambda_entropy_predictor_cond_hashcodes,
            seed=seed,
        )
        self.objective = 'binary' if objective is None else objective
        assert self.objective in ['regression', 'binary']

    def compute_sample_weights(self):

        num_data = self.labels.size
        assert num_data == self.features.shape[0]

        if self.objective == 'binary':
            assert self.labels.dtype == np.bool
            frac_positives = self.labels.mean()
            # print(f'frac_positives={frac_positives}')
            assert 0.0 <= frac_positives <= 1.0
            sample_weights = frac_positives*np.ones(num_data)
            sample_weights[self.labels] = (1.0 - frac_positives)
            frac_positives = None
            assert not np.any(np.isnan(sample_weights)), sample_weights
        else:
            sample_weights = None

        return sample_weights

    def kl_objective_for_tune(self, parameters, eps=1e-5):

        num_data = self.labels.size
        assert num_data == self.features.shape[0]

        sample_weights = self.compute_sample_weights()

        # inferred probs for the ground truth class/value
        infer_probs = self.compute_probs(
            features=self.features,
            labels=self.labels,
            parameters=parameters,
        )

        cond_entropy = ecp.compute_conditional_entropy_from_cond_probs(
            pred_prob=infer_probs,
            sample_weights=sample_weights,
            eps=eps
        )

        if self.adv_reg_obj is not None:

            assert self.features_for_adv_reg is not None

            infer_labels_adv_features = self.predict_labels_for_adv_reg(
                features=self.features,
                labels=self.labels,
                features_for_adv_reg=self.features_for_adv_reg,
                parameters=parameters,
            )

            labels_range = (
                min(0, infer_labels_adv_features.min()),
                max(1, infer_labels_adv_features.max())
            )

            entropy_of_preds = self.adv_reg_obj.compute_entropy_1d_with_histograms(
                x=infer_labels_adv_features,
                num_bins=20,
                x_range=labels_range,
            )

            entropy_cond_hashcodes = self.adv_reg_obj.compute_entropy_1d_with_histograms_cond_hashcodes(
                x=infer_labels_adv_features,
                num_bins=20,
                x_range=labels_range,
            )
        else:
            entropy_of_preds = 0.0
            entropy_cond_hashcodes = 0.0

        if entropy_of_preds == 0.0:
            assert entropy_cond_hashcodes == 0.0
            normalized_entropy_cond_hashcodes = 1.0
        else:
            normalized_entropy_cond_hashcodes = (entropy_cond_hashcodes/entropy_of_preds)

        loss = cond_entropy + self.lambda_entropy_predictor_cond_hashcodes*normalized_entropy_cond_hashcodes

        return {
            'loss': loss,
            'ce': cond_entropy,
            'entropy_cond_hashcodes': entropy_cond_hashcodes,
            'entropy_of_preds': entropy_of_preds,
            'normalized_entropy_cond_hashcodes': normalized_entropy_cond_hashcodes,
            'status': STATUS_OK,
            'parameters': parameters,
        }
