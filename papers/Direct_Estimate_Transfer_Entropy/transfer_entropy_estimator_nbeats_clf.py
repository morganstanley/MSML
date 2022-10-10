import importlib

import transfer_entropy_estimator as tee
importlib.reload(tee)

import nbeats_clf as nc
importlib.reload(nc)


class TransferEntropyNBeatsClf(tee.TransferEntropy):

    def __init__(
            self,
            time_lag,
            num_cores=1,
            seed=0,
            negative_sample=False,
            max_negative_sample_rate=1.0,
            tune_hyperparameters=False,
            is_hashcode_regularization=False,
            is_augment_data_from_hashcodes_for_tuning=False,
            augment_multiplier=1,
            is_data_augment_for_te_estimate=False,
            augment_multiplier_for_te_estimate=1.0,
            alpha_for_gen_sampling=0.1,
            num_hash_funcs=10,
            lambda_entropy_predictor_cond_hashcodes=0.0,
            debug=False,
            dir_path_for_saving_hash_models='/v/campus/ny/cs/aiml_build/sahigarg/saved_hash_models',
            normalize_te=False,
            # nbeats hyperparameters
            nb_blocks_per_stack=8,
            thetas_dim=(8, 8),
            share_weights_in_stack=True,
            hidden_layer_units=1024,
            # this can be adjusted as per dataset size
            batchSize=2**12,
            lr=1e-3,
            nbEpochs=10,
    ):

        super(TransferEntropyNBeatsClf, self).__init__(
            time_lag=time_lag,
            num_cores=num_cores,
            estimator_type='nbeats',
            estimation_objective='binary',
            is_regression_else_classification=False,
            negative_sample=negative_sample,
            max_negative_sample_rate=max_negative_sample_rate,
            tune_hyperparameters=tune_hyperparameters,
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

        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.thetas_dim = thetas_dim
        self.share_weights_in_stack = share_weights_in_stack
        self.hidden_layer_units = hidden_layer_units
        self.batchSize = batchSize
        self.lr = lr
        self.nbEpochs = nbEpochs

    def create_model(self):

        return nc.NBeatsClfNetwork(
            num_cores=1,
            debug=self.debug,
            lambda_entropy_predictor_cond_hashcodes=self.lambda_entropy_predictor_cond_hashcodes,
            nb_blocks_per_stack=self.nb_blocks_per_stack,
            thetas_dim=self.thetas_dim,
            share_weights_in_stack=self.share_weights_in_stack,
            hidden_layer_units=self.hidden_layer_units,
            batchSize=self.batchSize,
            lr=self.lr,
            nbEpochs=self.nbEpochs,
            seed=self.seed,
        )
