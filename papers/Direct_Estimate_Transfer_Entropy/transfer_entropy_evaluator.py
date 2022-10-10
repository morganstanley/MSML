import importlib
import time
import numpy as np

import constants

if constants.venv_sites_path is not None:
    import site
    site.addsitedir(constants.venv_sites_path)

import hashcode_model_for_te as hmte
importlib.reload(hmte)

import transfer_entropy_estimator as tee
importlib.reload(tee)

import neuro_events as ne
importlib.reload(ne)


class TransferEntropyEvaluator:

    # no forecasting but pure evaluation of transfer entropy
    # since we don't have ground truth on transfer entropy, we compute transfer entropy for a rolling window of a given size,
    # there are different ways to analyze the results the rolling window.

    def __init__(
            # see parameters description for TransferEntropyEstimator class
            self, num_ms_per_timestep, time_lag, num_cores,
            debug=False,
            window_size=10000,
            window_slide_size=1000,
            estimator_type = 'gbt',
            estimation_objective='binary',
            negative_sample_for_te=False,
            max_negative_sample_rate=1.0,
            tune_hyperparameters=True,
            is_hashcode_regularization=True,
            is_augment_data_from_hashcodes_for_tuning=False,
            augment_multiplier=1,
            alpha_for_gen_sampling=0.1,
            num_hash_funcs=10,
            lambda_entropy_predictor_cond_hashcodes=0.075,
            dir_path_for_saving_hash_models = '/v/campus/ny/cs/aiml_build/sahigarg/saved_hash_models',
            normalize_te=True,
            is_data_augment_for_te_estimate=False,
            augment_multiplier_for_te_estimate=1,
            is_regression_else_classification=False,
            seed=0,
    ):
        self.num_ms_per_timestep = num_ms_per_timestep
        self.time_lag = time_lag
        self.num_cores = num_cores

        self.window_size = window_size
        self.window_slide_size = window_slide_size

        self.estimator_type = estimator_type
        assert self.estimator_type in ['gbt', 'nbeats'], self.estimator_type

        self.estimation_objective = estimation_objective
        assert self.estimation_objective in ['regression', 'binary']

        self.negative_sample_for_te = negative_sample_for_te
        self.max_negative_sample_rate = max_negative_sample_rate

        self.tune_hyperparameters = tune_hyperparameters

        self.is_hashcode_regularization = is_hashcode_regularization
        self.is_augment_data_from_hashcodes_for_tuning = is_augment_data_from_hashcodes_for_tuning
        self.augment_multiplier = augment_multiplier
        self.alpha_for_gen_sampling = alpha_for_gen_sampling
        self.num_hash_funcs = num_hash_funcs

        self.lambda_entropy_predictor_cond_hashcodes = lambda_entropy_predictor_cond_hashcodes

        self.debug = debug

        self.dir_path_for_saving_hash_models = dir_path_for_saving_hash_models

        self.eps = 1e-5

        self.normalize_te = normalize_te

        self.is_regression_else_classification = is_regression_else_classification
        self.is_data_augment_for_te_estimate = is_data_augment_for_te_estimate
        self.augment_multiplier_for_te_estimate = augment_multiplier_for_te_estimate

        self.seed = seed

    def init_hash_model(self, C=1, penalty='l1'):
        return hmte.HashcodeModel(
            debug=self.debug,
            dir_path_for_saving_hash_models=self.dir_path_for_saving_hash_models,
            C=C, penalty=penalty,
        )

    def transfer_entropy_optimize(self, train_events, event_names, trial_ids=None):

        start_time = time.time()
        print('Computing transfer entropy ...')

        train_te_obj = tee.TransferEntropy(
            time_lag=self.time_lag,
            num_cores=self.num_cores,
            negative_sample=self.negative_sample_for_te,
            max_negative_sample_rate=self.max_negative_sample_rate,
            estimator_type=self.estimator_type,
            estimation_objective=self.estimation_objective,
            tune_hyperparameters=self.tune_hyperparameters,
            is_hashcode_regularization=self.is_hashcode_regularization,
            is_augment_data_from_hashcodes_for_tuning=self.is_augment_data_from_hashcodes_for_tuning,
            augment_multiplier=self.augment_multiplier,
            alpha_for_gen_sampling=self.alpha_for_gen_sampling,
            num_hash_funcs=self.num_hash_funcs,
            lambda_entropy_predictor_cond_hashcodes=self.lambda_entropy_predictor_cond_hashcodes,
            dir_path_for_saving_hash_models=self.dir_path_for_saving_hash_models,
            normalize_te=self.normalize_te,
            is_regression_else_classification=self.is_regression_else_classification,
            is_data_augment_for_te_estimate=self.is_data_augment_for_te_estimate,
            augment_multiplier_for_te_estimate=self.augment_multiplier_for_te_estimate,
            seed=self.seed,
        )

        train_te_obj.build(time_series_data=train_events, ticker_names=event_names, trial_ids=trial_ids)

        te, ce = train_te_obj.compute_transfer_entropy()

        print(f'Computed transfer entropy in {round(time.time()-start_time, 1)}s.')

        return te, ce

    def learn_hash_model_for_adversarial_regularization_neuro_multiple_trials(
            self,
            df,
            num_hash_functions=20,
            alpha=128,
            max_num_samples_in_cluster_for_entropy_compute=1000,
            max_num_samples_in_sel_cluster_for_kl_div_compute=300,
            max_num_samples_in_cluster_for_kl_div_compute=30,
            min_set_size_to_split=8,
            max_set_size_for_info_cluster_optimal=32,
            is_equal_split=True,
            is_unique_per_binary_dtype=True,
            visualize_quality_of_hashcodes=False,
            C=1,
            penalty='l1',
            num_trials=None,
            trial_ids=None,
    ):
        assert self.is_regression_else_classification
        events, event_names, trial_ids = ne.NeuroMultivariateTimeseries().events_from_df(
            df=df,
            dtype=np.float,
            is_sparse=False,
            num_trials=num_trials,
            trial_ids=trial_ids,
        )
        print(f'events.shape={events.shape} events.dtype={events.dtype}')

        num_time_series, num_timesteps = events.shape

        features = {}
        for curr_trial_id in np.unique(trial_ids):
            idx_for_curr_trial_id = np.where(trial_ids == curr_trial_id)[0]
            assert np.all((idx_for_curr_trial_id[1:] - idx_for_curr_trial_id[:-1]) == 1)
            curr_trial_events = events[:, idx_for_curr_trial_id]

            features[curr_trial_id] = np.stack(
                tuple(
                    curr_trial_events[:, curr_lag_idx:(-self.time_lag+curr_lag_idx)] for curr_lag_idx in range(self.time_lag)
                ),
                axis=2,
            )

        features = np.concatenate(tuple(features.values()), axis=1)
        print(f'features.shape={features.shape}')
        assert features.shape[0] == num_time_series
        assert features.shape[2] == self.time_lag

        for curr_event_idx, curr_event_name in enumerate(event_names):

            curr_event_features = features[curr_event_idx, :, :]

            if is_unique_per_binary_dtype:
                _, unique_events_idx = np.unique(curr_event_features.astype(np.bool), axis=0, return_index=True)
                curr_event_features = curr_event_features[unique_events_idx, :]
                unique_events_idx = None
            else:
                curr_event_features = np.unique(curr_event_features, axis=0)

            print(f'{curr_event_name}={curr_event_features}')

            hash_model_obj = self.init_hash_model(C=C, penalty=penalty)
            hash_model_obj.compute_hashcodes(
                features=curr_event_features,
                is_optimize=True,
                num_cores=self.num_cores,
                num_hash_functions=num_hash_functions,
                alpha=alpha,
                max_num_samples_in_cluster_for_entropy_compute=max_num_samples_in_cluster_for_entropy_compute,
                max_num_samples_in_sel_cluster_for_kl_div_compute=max_num_samples_in_sel_cluster_for_kl_div_compute,
                max_num_samples_in_cluster_for_kl_div_compute=max_num_samples_in_cluster_for_kl_div_compute,
                min_set_size_to_split=min_set_size_to_split,
                max_set_size_for_info_cluster_optimal=max_set_size_for_info_cluster_optimal,
                is_equal_split=is_equal_split,
                model_name=curr_event_name,
                visualize_quality_of_hashcodes=visualize_quality_of_hashcodes,
            )

    def te_regression_neuro_multi_trials(self, df, num_trials=None, trial_ids=None):

        events, event_names, trial_ids = ne.NeuroMultivariateTimeseries().events_from_df(
            df=df, is_sparse=True, num_trials=num_trials, trial_ids=trial_ids,
        )

        te, ce = self.transfer_entropy_optimize(
            train_events=events,
            event_names=event_names,
            trial_ids=trial_ids,
        )

        return te, ce
