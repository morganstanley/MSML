import importlib

import transfer_entropy_estimator as tee
importlib.reload(tee)

import gbm_clf
importlib.reload(gbm_clf)


class TransferEntropyGBMClf(tee.TransferEntropy):

    def __init__(
            self,
            time_lag=10,
            num_cores=1,
            negative_sample=False,
            max_negative_sample_rate=1.0,
            tune_hyperparameters=True,
            is_hashcode_regularization=True,
            is_augment_data_from_hashcodes_for_tuning=False,
            augment_multiplier=1,
            is_data_augment_for_te_estimate=False,
            augment_multiplier_for_te_estimate=1.0,
            alpha_for_gen_sampling=0.1,
            num_hash_funcs=10,
            lambda_entropy_predictor_cond_hashcodes=1.0,
            debug=False,
            dir_path_for_saving_hash_models='/v/campus/ny/cs/aiml_build/sahigarg/saved_hash_models',
            seed=0,
            normalize_te=False,
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
            #
            num_max_evals_for_tuning=300,
    ):

        super(TransferEntropyGBMClf, self).__init__(
                time_lag=time_lag,
                num_cores=num_cores,
                estimator_type='gbt',
                estimation_objective='binary',
                negative_sample=negative_sample,
                max_negative_sample_rate=max_negative_sample_rate,
                tune_hyperparameters=tune_hyperparameters,
                is_regression_else_classification=False,
                is_hashcode_regularization=is_hashcode_regularization,
                is_augment_data_from_hashcodes_for_tuning=is_augment_data_from_hashcodes_for_tuning,
                augment_multiplier=augment_multiplier,
                is_data_augment_for_te_estimate=is_data_augment_for_te_estimate,
                augment_multiplier_for_te_estimate=augment_multiplier_for_te_estimate,
                alpha_for_gen_sampling=alpha_for_gen_sampling,
                num_hash_funcs=num_hash_funcs,
                lambda_entropy_predictor_cond_hashcodes=lambda_entropy_predictor_cond_hashcodes,
                debug=debug,
                dir_path_for_saving_hash_models=dir_path_for_saving_hash_models,
                seed=seed,
                normalize_te=normalize_te,
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

    def create_model(self):

        return gbm_clf.GradientBoostedClfTree(
            num_cores=1,
            debug=self.debug,
            objective=self.estimation_objective,
            lambda_entropy_predictor_cond_hashcodes=self.lambda_entropy_predictor_cond_hashcodes,
            bagging_fraction=self.gbm_bagging_fraction,
            bagging_freq=self.gbm_bagging_freq,
            learning_rate=self.gbm_learning_rate,
            feature_fraction=self.gbm_feature_fraction,
            max_depth=self.gbm_max_depth,
            is_unbalance=self.gbm_is_unbalance,
            pos_bagging_fraction=self.gbm_pos_bagging_fraction,
            neg_bagging_fraction=self.gbm_neg_bagging_fraction,
            min_data_in_leaf=self.gbm_min_data_in_leaf,
            num_leaves=self.gbm_num_leaves,
            lambda_l1=self.gbm_lambda_l1,
            lambda_l2=self.gbm_lambda_l2,
            path_smooth=self.gbm_path_smooth,
            num_max_evals_for_tuning=self.num_max_evals_for_tuning,
            seed=self.seed,
        )
