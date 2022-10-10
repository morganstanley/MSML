from ms_specific import *

import numpy as np
import numpy.random as npr
import scipy.stats as scipy_stats

import hyperopt
from hyperopt import hp, fmin, tpe, space_eval, Trials, STATUS_OK


class DiscriminatorTransferEntropyEstimation:

    def __init__(self, num_cores=1, debug=False, seed=0,
        lambda_entropy_predictor_cond_hashcodes=0.075,
    ):
        self.num_cores = num_cores
        self.lambda_entropy_predictor_cond_hashcodes = lambda_entropy_predictor_cond_hashcodes
        self.debug = debug
        self.max_evals = 300
        self.is_normalize = False
        self.seed = seed

    def get_hyperparameters_space(self):
        raise NotImplemented

    def post_process_tuned_parameters(self, hp_space, tuned_parameters):
        raise NotImplemented

    def plot_obj_in_hp_space(
            self,
            ce,
            opt_idx,
            meta_data,
            entropy_of_preds=None,
            entropy_cond_hashcodes=None,
            normalized_entropy_cond_hashcodes=None,
    ):
        pass

    def tune_hyperparameters(
            self,
            features, labels,
            meta_data='',
            features_for_adv_reg=None,
            adv_reg_obj=None,
    ):
        # cross validation or fixed test set randomly from the inputs
        self.features = features
        self.labels = labels
        self.features_for_adv_reg = features_for_adv_reg
        self.adv_reg_obj = adv_reg_obj

        self.data_sampler_for_hp_tune = npr.RandomState(seed=0)
        self.num_batches_for_cross_validation = 5
        self.batch_idx_for_cross_validation = self.data_sampler_for_hp_tune.choice(
            self.num_batches_for_cross_validation, size=self.labels.size, replace=True,
        )

        hp_space = self.get_hyperparameters_space()

        trials = Trials()
        _ = hyperopt.fmin(
            fn=self.kl_objective_for_tune,
            space=hp_space,
            algo=hyperopt.tpe.suggest,
            max_evals=self.max_evals,
            verbose=True,
            trials=trials,
            show_progressbar=True,
            rstate=npr.RandomState(seed=self.seed),
        )

        losses = np.array(trials.losses())
        assert losses.size == self.max_evals

        ce = np.array([curr_trial_result['ce'] for curr_trial_result in trials.results])
        assert ce.size == self.max_evals

        entropy_of_preds = np.array([curr_trial_result['entropy_of_preds'] for curr_trial_result in trials.results])
        assert entropy_of_preds.size == self.max_evals

        entropy_cond_hashcodes = np.array([curr_trial_result['entropy_cond_hashcodes'] for curr_trial_result in trials.results])
        assert entropy_cond_hashcodes.size == self.max_evals

        normalized_entropy_cond_hashcodes = np.array([curr_trial_result['normalized_entropy_cond_hashcodes'] for curr_trial_result in trials.results])
        assert normalized_entropy_cond_hashcodes.size == self.max_evals

        if self.is_normalize:
            losses_z = scipy_stats.zscore(ce)
            losses_z += scipy_stats.zscore(normalized_entropy_cond_hashcodes)
            opt_idx = losses_z.argmin()
            losses_z = None
        else:
            losses_weighted = ce.copy()
            losses_weighted += (self.lambda_entropy_predictor_cond_hashcodes*normalized_entropy_cond_hashcodes)
            opt_idx = losses_weighted.argmin()
            losses_weighted = None

        tuned_parameters = trials.results[opt_idx]['parameters']
        tuned_parameters = self.post_process_tuned_parameters(
            hp_space=hp_space,
            tuned_parameters=tuned_parameters,
        )

        self.parameters = tuned_parameters

        print(meta_data, tuned_parameters)

        self.plot_obj_in_hp_space(
            ce=ce,
            opt_idx=losses.argmin(),
            meta_data=meta_data,
            normalized_entropy_cond_hashcodes=normalized_entropy_cond_hashcodes,
            entropy_cond_hashcodes=entropy_cond_hashcodes,
            entropy_of_preds=entropy_of_preds,
        )

        return tuned_parameters

    def kl_objective_for_tune(self, parameters):
        raise NotImplemented

    def predict_labels_for_adv_reg(
            self,
            features, labels,
            features_for_adv_reg,
            parameters=None,
    ):
        raise NotImplemented

    def compute_log_lkl(
            self,
            features, labels,
            parameters=None,
            test_features=None,
            test_labels=None,
    ):
        raise NotImplemented

    def compute_probs(
            self,
            features, labels,
            parameters=None,
            test_features=None,
            test_labels=None,
    ):
        raise NotImplemented
