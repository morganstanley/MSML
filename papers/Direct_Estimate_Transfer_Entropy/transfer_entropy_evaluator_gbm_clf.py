import time
import importlib

import transfer_entropy_evaluator as tee
importlib.reload(tee)

import transfer_entropy_estimator_gbm_clf as tegbm
importlib.reload(tegbm)


class TransferEntropyEvaluatorGBMClf(tee.TransferEntropyEvaluator):

    def __init__(self,
            num_ms_per_timestep, time_lag, num_cores,
            debug=False,
            window_size=None,
            window_slide_size=1000,
            negative_sample_for_te=False,
            max_negative_sample_rate=1.0,
            tune_hyperparameters=True,
            is_hashcode_regularization=True,
            is_augment_data_from_hashcodes_for_tuning=False,
            augment_multiplier=1,
            alpha_for_gen_sampling=0.1,
            num_hash_funcs=10,
            lambda_entropy_predictor_cond_hashcodes=1.0,
            dir_path_for_saving_hash_models = '/v/campus/ny/cs/aiml_build/sahigarg/saved_hash_models',
            normalize_te=False,
            is_data_augment_for_te_estimate=False,
            augment_multiplier_for_te_estimate=1,
            #  default parameters of lightgbm, put here explicitly
            gbm_bagging_fraction=1.0,
            gbm_bagging_freq = 5,
            gbm_learning_rate=1.0,
            gbm_feature_fraction=1.0,
            gbm_max_depth=-1,
            gbm_is_unbalance=False,
            gbm_pos_bagging_fraction=1.0,
            gbm_neg_bagging_fraction=1.0,
            gbm_min_data_in_leaf=20,
            gbm_num_leaves=31,
            gbm_lambda_l1=0.0,
            gbm_lambda_l2=0.0,
            gbm_path_smooth=0.0,
            # number of evaluations for tuning hyper-parameters
            num_max_evals_for_tuning=300,
            seed=0,
    ):

        super(TransferEntropyEvaluatorGBMClf, self).__init__(
                num_ms_per_timestep=num_ms_per_timestep,
                time_lag=time_lag,
                num_cores=num_cores,
                debug=debug,
                window_size=window_size,
                window_slide_size=window_slide_size,
                estimator_type='gbt',
                estimation_objective='binary',
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
                is_regression_else_classification=False,
                seed=seed,
        )

        self.gbm_bagging_fraction = gbm_bagging_fraction
        self.gbm_bagging_freq = gbm_bagging_freq
        self.gbm_learning_rate = gbm_learning_rate
        self.gbm_feature_fraction = gbm_feature_fraction
        self.gbm_max_depth = gbm_max_depth
        self.gbm_is_unbalance = gbm_is_unbalance
        self.gbm_pos_bagging_fraction = gbm_pos_bagging_fraction
        self.gbm_neg_bagging_fraction = gbm_neg_bagging_fraction
        self.gbm_min_data_in_leaf = gbm_min_data_in_leaf
        self.gbm_num_leaves = gbm_num_leaves
        self.gbm_lambda_l1 = gbm_lambda_l1
        self.gbm_lambda_l2 = gbm_lambda_l2
        self.gbm_path_smooth = gbm_path_smooth
        self.num_max_evals_for_tuning = num_max_evals_for_tuning

    def transfer_entropy_optimize(self, train_events, event_names, trial_ids=None):

        start_time = time.time()
        print('Computing transfer entropy ...')

        train_te_obj = tegbm.TransferEntropyGBMClf(
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
            gbm_bagging_fraction=self.gbm_bagging_fraction,
            gbm_bagging_freq=self.gbm_bagging_freq,
            gbm_learning_rate=self.gbm_learning_rate,
            gbm_feature_fraction=self.gbm_feature_fraction,
            gbm_max_depth=self.gbm_max_depth,
            gbm_is_unbalance=self.gbm_is_unbalance,
            gbm_pos_bagging_fraction=self.gbm_pos_bagging_fraction,
            gbm_neg_bagging_fraction=self.gbm_neg_bagging_fraction,
            gbm_min_data_in_leaf=self.gbm_min_data_in_leaf,
            gbm_num_leaves=self.gbm_num_leaves,
            gbm_lambda_l1=self.gbm_lambda_l1,
            gbm_lambda_l2=self.gbm_lambda_l2,
            gbm_path_smooth=self.gbm_path_smooth,
            num_max_evals_for_tuning=self.num_max_evals_for_tuning,
            seed=self.seed,
        )

        train_te_obj.build(time_series_data=train_events, ticker_names=event_names, trial_ids=trial_ids)

        te, ce = train_te_obj.compute_transfer_entropy()

        print(f'Computed transfer entropy in {round(time.time()-start_time, 1)}s.')

        return te, ce
