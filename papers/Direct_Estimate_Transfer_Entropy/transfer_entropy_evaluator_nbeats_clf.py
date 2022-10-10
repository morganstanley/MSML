import time
import importlib

import transfer_entropy_evaluator as tee
importlib.reload(tee)

import transfer_entropy_estimator_nbeats_clf as tenbeats
importlib.reload(tenbeats)


class TransferEntropyEvaluatorNBeatsClf(tee.TransferEntropyEvaluator):

    def __init__(self,
            num_ms_per_timestep,
            time_lag, num_cores,
            debug=False,
            window_size=None,
            window_slide_size=1000,
            negative_sample_for_te=False,
            max_negative_sample_rate=1.0,
            tune_hyperparameters=False,
            is_hashcode_regularization=False,
            is_augment_data_from_hashcodes_for_tuning=False,
            augment_multiplier=1,
            alpha_for_gen_sampling=0.1,
            num_hash_funcs=10,
            lambda_entropy_predictor_cond_hashcodes=0.0,
            dir_path_for_saving_hash_models = '/v/campus/ny/cs/aiml_build/sahigarg/saved_hash_models',
            normalize_te=False,
            is_data_augment_for_te_estimate=False,
            augment_multiplier_for_te_estimate=1,
            # nbeats hyper-parameters
            nb_blocks_per_stack=8,
            thetas_dim=(8, 8),
            share_weights_in_stack=True,
            hidden_layer_units=1024,
            batchSize=2**12,
            lr=1e-3,
            nbEpochs=10,
            seed=0,
    ):

        super(TransferEntropyEvaluatorNBeatsClf, self).__init__(
            num_ms_per_timestep=num_ms_per_timestep,
            time_lag=time_lag,
            num_cores=num_cores,
            debug=debug,
            window_size=window_size,
            window_slide_size=window_slide_size,
            estimator_type='nbeats',
            estimation_objective='binary',
            is_regression_else_classification=False,
            negative_sample_for_te=negative_sample_for_te,
            max_negative_sample_rate=max_negative_sample_rate,
            tune_hyperparameters=tune_hyperparameters,
            is_hashcode_regularization=is_hashcode_regularization,
            is_augment_data_from_hashcodes_for_tuning=is_augment_data_from_hashcodes_for_tuning,
            augment_multiplier=augment_multiplier,
            alpha_for_gen_sampling=alpha_for_gen_sampling,
            num_hash_funcs=num_hash_funcs,
            lambda_entropy_predictor_cond_hashcodes=lambda_entropy_predictor_cond_hashcodes,
            dir_path_for_saving_hash_models=dir_path_for_saving_hash_models,
            normalize_te=normalize_te,
            is_data_augment_for_te_estimate=is_data_augment_for_te_estimate,
            augment_multiplier_for_te_estimate=augment_multiplier_for_te_estimate,
            seed=seed,
        )

        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.thetas_dim = thetas_dim
        self.share_weights_in_stack = share_weights_in_stack
        self.hidden_layer_units = hidden_layer_units
        self.batchSize = batchSize
        self.lr = lr
        self.nbEpochs = nbEpochs

    def transfer_entropy_optimize(self, train_events, event_names, trial_ids=None):

        start_time = time.time()
        print('Computing transfer entropy ...')

        train_te_obj = tenbeats.TransferEntropyNBeatsClf(
            time_lag=self.time_lag,
            num_cores=self.num_cores,
            negative_sample=self.negative_sample_for_te,
            max_negative_sample_rate=self.max_negative_sample_rate,
            tune_hyperparameters=self.tune_hyperparameters,
            is_hashcode_regularization=self.is_hashcode_regularization,
            is_augment_data_from_hashcodes_for_tuning=self.is_augment_data_from_hashcodes_for_tuning,
            augment_multiplier=self.augment_multiplier,
            alpha_for_gen_sampling=self.alpha_for_gen_sampling,
            num_hash_funcs=self.num_hash_funcs,
            lambda_entropy_predictor_cond_hashcodes=self.lambda_entropy_predictor_cond_hashcodes,
            dir_path_for_saving_hash_models=self.dir_path_for_saving_hash_models,
            normalize_te=self.normalize_te,
            is_data_augment_for_te_estimate=self.is_data_augment_for_te_estimate,
            augment_multiplier_for_te_estimate=self.augment_multiplier_for_te_estimate,
            nb_blocks_per_stack=self.nb_blocks_per_stack,
            thetas_dim=self.thetas_dim,
            share_weights_in_stack=self.share_weights_in_stack,
            hidden_layer_units=self.hidden_layer_units,
            batchSize=self.batchSize,
            lr=self.lr,
            nbEpochs=self.nbEpochs,
            seed=self.seed,
        )

        train_te_obj.build(time_series_data=train_events, ticker_names=event_names, trial_ids=trial_ids)

        te, ce = train_te_obj.compute_transfer_entropy()

        print(f'Computed transfer entropy in {round(time.time()-start_time, 1)}s.')

        return te, ce
