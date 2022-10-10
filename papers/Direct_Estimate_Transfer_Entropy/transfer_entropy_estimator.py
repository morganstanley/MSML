import importlib
import math

import time
import multiprocessing

import numpy as np
import numpy.random as npr

import scipy.sparse as scipy_sparse

import hashcode_model_for_te as hmte
importlib.reload(hmte)

import adversarial_hash_regularizer as ahr
importlib.reload(ahr)

import data_augment_from_hash_bins as dahb
importlib.reload(dahb)

from hash_finance import hash_function_representations_info_theoretic_optimization as hf_ito
importlib.reload(hf_ito)

from hash_finance import information_theoretic_measures_hashcodes as itmh
importlib.reload(itmh)

import entropy_from_class_probs as ecp
importlib.reload(ecp)

import constants

if constants.venv_sites_path is not None:
    import site
    site.addsitedir(constants.venv_sites_path)

import sparse


def map_multivariate_timeseries_to_features_and_labels_wrapper(time_series_data, time_lag):

    # time_lag is the number of timesteps from the past for inferring in the present timestep

    # assuming that timeseries data is sparse, which is applicable for neuroscience, finance domains.
    # even for stock price datasets, change in price can be zero at microscale
    # time lag corresponds to feature vector and the observation in the current timestep is a label
    # sparse 3-D arrays
    # time lag is assumed to be in a few hundreds, so this code block should be efficient for practical purposes
    features = sparse.stack(
        tuple(
            time_series_data[:, curr_lag_idx:(-time_lag+curr_lag_idx)] for curr_lag_idx in range(time_lag)
        ),
        axis=2,
    )

    # sparse 2-D array using the standard scipy sparse module
    labels = time_series_data[:, time_lag:].tocsr().astype(np.int)

    return features, labels


def map_multivariate_timeseries_to_features_and_labels(time_series_data, time_lag, trial_ids=None):

    # trial ids are applicable for neuroscience data
    # it is more efficient to process data for each trial separately
    # code can be extended in the future for multiprocess across trials

    print('Mapping timeseries to features and labels ...')
    start_time = time.time()

    if trial_ids is None:
        return map_multivariate_timeseries_to_features_and_labels_wrapper(
            time_series_data=time_series_data, time_lag=time_lag
        )
    else:
        features = {}
        labels = {}
        unique_trial_ids = np.unique(trial_ids)

        for curr_trial_id in unique_trial_ids:
            idx_from_curr_trial_id = np.where(trial_ids == curr_trial_id)[0]
            assert np.all(idx_from_curr_trial_id[1:] - idx_from_curr_trial_id[:-1] == 1)
            curr_trial_time_series_data = time_series_data[:, idx_from_curr_trial_id]
            features[curr_trial_id], labels[curr_trial_id] = map_multivariate_timeseries_to_features_and_labels_wrapper(
                time_series_data=curr_trial_time_series_data,
                time_lag=time_lag,
            )

        features = sparse.concatenate(tuple(features.values()), axis=1)
        assert features.shape[2] == time_lag
        labels = scipy_sparse.hstack(tuple(labels.values()), format='csr')

    print(features.shape, labels.shape)
    assert features.shape[1] == labels.shape[1]
    assert features.shape[0] == labels.shape[0]

    print(f'Mapped in {round(time.time()-start_time, 2)}s.')

    return features, labels


class NegativeSampler:

    # sampling from negatives
    # in timeseries, if there are many zeros, many of the feature vectors and the corresponding labels would be zero. In such case, it is important to perform negative sampling.

    def __init__(self, max_negative_sample_rate=1.0, seed=0):
        self.max_negative_sample_rate = max_negative_sample_rate
        self.negative_sampler = npr.RandomState(seed=seed)

    def negative_samples_idx(self, labels_i, features_i):

        num_pos = np.count_nonzero(labels_i)
        num_neg = (labels_i.size-num_pos)
        neg_to_pos_ratio = (float(num_neg)/num_pos) if num_pos > 0 else num_neg

        if neg_to_pos_ratio > 1.0:

            logical_idx = (labels_i > 0)

            # logic for negative sampling is as follows. It is expected to see zero labels (i.e. negative samples) if feature vector (timesteps from the recent past) have zeros as well.
            # a data point which has nonzeros in feature vector but zero label is more informative

            # minimal sum for a nonzero vector is 1, 0.1 is for zero vectors
            # that is basically 1/10 weight of a minimal non-zero vector
            weights = np.asarray(features_i[(~logical_idx), :].sum(1)).flatten()+0.1
            weights /= weights.sum()
            sampled_neg_idx = self.negative_sampler.choice(
                np.where(~logical_idx)[0],
                size=int(num_pos*min(math.sqrt(neg_to_pos_ratio), self.max_negative_sample_rate)) if num_pos > 0 else int(math.sqrt(num_neg)),
                p=weights,
            )
            logical_idx[sampled_neg_idx] = True
            sampled_neg_idx, weights = (None,)*2

            return logical_idx
        else:
            print('Not sampling negatives.')
            return None


class TransferEntropy:

    def __init__(
            self,
            time_lag=10,
            num_cores=1,
            estimator_type = 'gbt',
            estimation_objective='binary',
            negative_sample=True,
            max_negative_sample_rate=1.0,
            tune_hyperparameters=True,
            is_regression_else_classification=False,
            is_hashcode_regularization=False,
            is_augment_data_from_hashcodes_for_tuning=False,
            # fraction of samples to generate for augmentation for tuning specifically
            augment_multiplier=1,
            is_data_augment_for_te_estimate=False,
            augment_multiplier_for_te_estimate=1.0,
            # parameter of Dirichlet for sampling from within a hashcode
            # each data point in a hashcode is a dimension of Dirichlet distribution
            # multinomial vectors are samples from the distribution for weighted convex combination of the data points
            alpha_for_gen_sampling=0.1,
            # number of hash function for LSH based regularization
            num_hash_funcs=10,
            # weight for LSH regularization in the objective
            lambda_entropy_predictor_cond_hashcodes=0.075,
            debug=False,
            dir_path_for_saving_hash_models='./saved_hash_models',
            seed=0,
            # normalize transfer entropy
            normalize_te=True,
    ):
        # number of timesteps as observations
        self.time_lag = time_lag
        self.num_cores = num_cores

        # not performing regression for now, classification also includes poisson distribution as target
        self.is_regression_else_classification = is_regression_else_classification
        # to deal with log of probabilities if zero values
        self.eps = 1e-5

        self.tune_hyperparameters = tune_hyperparameters

        self.is_hashcode_regularization = is_hashcode_regularization
        self.is_augment_data_from_hashcodes_for_tuning = is_augment_data_from_hashcodes_for_tuning
        self.augment_multiplier = augment_multiplier

        self.is_data_augment_for_te_estimate = is_data_augment_for_te_estimate
        self.augment_multiplier_for_te_estimate = augment_multiplier_for_te_estimate

        self.alpha_for_gen_sampling = alpha_for_gen_sampling
        self.num_hash_funcs = num_hash_funcs

        self.lambda_entropy_predictor_cond_hashcodes = lambda_entropy_predictor_cond_hashcodes

        self.debug = debug
        if self.num_cores > 1:
            assert not self.debug

        self.dir_path_for_saving_hash_models = dir_path_for_saving_hash_models

        self.estimator_type = estimator_type
        assert self.estimator_type in ['gbt', 'nbeats']

        self.estimation_objective = estimation_objective
        assert self.estimation_objective in ['regression', 'binary']

        # it is not standard negative sampling, customized for TE
        self.negative_sample = negative_sample
        self.max_negative_sample_rate = max_negative_sample_rate

        if self.negative_sample:
            self.negative_sampler_obj = NegativeSampler(
                max_negative_sample_rate=max_negative_sample_rate,
                seed=seed,
            )

        # sanity checks to ensure that data passed is int, not binary, if modeling target as poisson distributed
        self.labels_dtype = np.int
        self.normalize_te = normalize_te

        self.seed = seed

    def create_model(self):
        raise NotImplemented

    def prepare_conditional_density_estimator(
            self,
            features_for_tune=None,
            labels_for_tune=None,
            meta_data='',
            features_for_adv_reg=None,
            adv_reg_obj=None,
    ):
        cond_density_estimator = self.create_model()
        cond_density_estimator.lambda_entropy_predictor_cond_hashcodes = self.lambda_entropy_predictor_cond_hashcodes

        if self.tune_hyperparameters:
            assert self.estimator_type not in ['lr']
            cond_density_estimator.tune_hyperparameters(
                features=features_for_tune,
                labels=labels_for_tune,
                meta_data=meta_data,
                features_for_adv_reg=features_for_adv_reg,
                adv_reg_obj=adv_reg_obj,
            )

        return cond_density_estimator

    def __compute_cond_entropy__(self, features, labels,
            eps=1e-5,
            cond_density_estimator=None,
            features_ad=None,
            labels_ad=None,
    ):

        if cond_density_estimator is None:
            cond_density_estimator = self.create_model()

        if features_ad is not None:
            assert labels_ad is not None
            features = self.__concatenate_features__(features1=features, features2=features_ad)
            labels = np.concatenate((labels, labels_ad))

        # infer probabilities for labels conditioning upon features
        # P(y|X)
        # for high dimensions, obtaining high dimensional probabilities is difficult,
        # so computing it using a discriminative model with low capacity

        pred_prob = cond_density_estimator.compute_probs(
            features=features,
            labels=labels,
        )
        # assert not np.any(np.isnan(pred_prob))

        assert labels.dtype == np.bool
        positives_frac = labels.mean()
        sample_weights = positives_frac*np.ones(labels.size)
        sample_weights[labels] = (1.0 - positives_frac)
        positives_frac = None

        # negative of expectation of log probabilities is entropy
        # as standard practice, assuming p(x,y)=1/N for N datapoints.
        # in a separate piece of code, I compute P(x,y) explicitly, a work in progress.
        entropy = ecp.compute_conditional_entropy_from_cond_probs(
            pred_prob=pred_prob,
            sample_weights=sample_weights,
            eps=eps,
        )

        return entropy

    def generate_labeled_data_for_augment_from_hashing(self, features_arr, labels, ticker, is_optimize=False, num_cores=1):

        hashcode_model_obj = hmte.HashcodeModel(
            dir_path_for_saving_hash_models=self.dir_path_for_saving_hash_models,
        )

        hashcodes_for_features = hashcode_model_obj.compute_hashcodes(
            features=features_arr,
            is_optimize=is_optimize,
            model_name=ticker,
            num_cores=num_cores,
        )
        assert hashcodes_for_features.shape[0] == features_arr.shape[0]

        hda_obj = dahb.HashDataAugment(
            C=hashcodes_for_features,
            alpha=self.alpha_for_gen_sampling,
            num_hash_funcs=self.num_hash_funcs,
        )

        gen_features, gen_labels = hda_obj.generate_data_from_hashcodes(
            events=features_arr,
            labels=labels,
            multiple=self.augment_multiplier_for_te_estimate,
        )

        return gen_features, gen_labels

    def generate_labeled_data_for_augment_from_hashing_for_transfer(self, features1, features2, labels2, ticker1, ticker2, is_optimize=False, num_cores=1):

        if not self.is_regression_else_classification:
            assert scipy_sparse.issparse(features1) and isinstance(features1, scipy_sparse.csr_matrix)
            assert scipy_sparse.issparse(features2) and isinstance(features2, scipy_sparse.csr_matrix)
            features1 = features1.toarray()
            features2 = features2.toarray()

        hashcode_model_obj = hmte.HashcodeModel(
            dir_path_for_saving_hash_models=self.dir_path_for_saving_hash_models,
        )

        hashcodes_for_features1 = hashcode_model_obj.compute_hashcodes(
            features=features1,
            is_optimize=is_optimize,
            model_name=ticker1,
            num_cores=num_cores,
        )
        assert hashcodes_for_features1.shape[0] == features1.shape[0]

        hashcodes_for_features2 = hashcode_model_obj.compute_hashcodes(
            features=features2,
            is_optimize=is_optimize,
            model_name=ticker2,
            num_cores=num_cores,
        )
        assert hashcodes_for_features2.shape[0] == features2.shape[0]

        hashcodes_for_features12 = np.hstack((hashcodes_for_features1, hashcodes_for_features2))

        hda_obj = dahb.HashDataAugment(
            C=hashcodes_for_features12,
            alpha=self.alpha_for_gen_sampling,
            num_hash_funcs=hashcodes_for_features12.shape[1],
        )

        features12 = np.hstack((features1, features2))

        gen_features, gen_labels = hda_obj.generate_data_from_hashcodes(
            events=features12,
            labels=labels2,
            multiple=self.augment_multiplier_for_te_estimate,
        )
        if not self.is_regression_else_classification:
            gen_features = scipy_sparse.csr_matrix(gen_features)

        gen_features1 = gen_features[:, :features1.shape[1]]
        gen_features2 = gen_features[:, features1.shape[1]:]

        # not necessary, using only while coding
        if not self.is_regression_else_classification:
            assert scipy_sparse.issparse(gen_features1) and isinstance(gen_features1, scipy_sparse.csr_matrix)
            assert scipy_sparse.issparse(gen_features2) and isinstance(gen_features2, scipy_sparse.csr_matrix)

        gen_features = self.__join_features_from_two_timeseries__(
            features1=gen_features1, features2=gen_features2,
        )
        gen_features1, gen_features2 = (None,)*2

        return gen_features, gen_labels

    def generate_data_through_hashing(self, features_arr, ticker, is_optimize=False, num_cores=1):

        hashcode_model_obj = hmte.HashcodeModel(
            dir_path_for_saving_hash_models=self.dir_path_for_saving_hash_models,
        )

        hashcodes_for_features = hashcode_model_obj.compute_hashcodes(
            features=features_arr,
            is_optimize=is_optimize,
            model_name=ticker,
            num_cores=num_cores,
        )
        assert hashcodes_for_features.shape[0] == features_arr.shape[0]

        adv_reg_obj = ahr.AdversarialHashRegularizer(
            C=hashcodes_for_features,
            log_base_for_k=4.0,
            max_val_for_k=8,
            alpha=self.alpha_for_gen_sampling,
            num_hash_funcs=self.num_hash_funcs,
        )
        features_from_hash_generative_process = adv_reg_obj.generated_data_from_hashcodes(
            events=features_arr,
            multiple=self.augment_multiplier,
        )
        hashcodes_of_generated_features = hashcode_model_obj.compute_hashcodes(
            features=features_from_hash_generative_process,
            is_optimize=False,
            model_name=ticker,
            num_cores=num_cores,
        )

        return hashcodes_for_features, features_from_hash_generative_process, hashcodes_of_generated_features

    def __join_features_from_two_timeseries__(self, features1, features2):

        if self.is_regression_else_classification:
            if self.estimator_type in ['tcn', 'nbeats']:
                features12 = np.stack(
                    (features2, features1),
                    axis=-2,
                )
                assert features12.shape[1] == 2
            else:
                features12 = np.hstack((features2, features1))
        else:
            if self.estimator_type in ['tcn', 'nbeats']:
                features12 = sparse.stack(
                    tuple([
                        sparse.COO.from_scipy_sparse(features2),
                        sparse.COO.from_scipy_sparse(features1),
                    ]),
                    axis=-2,
                )
                assert features12.shape[1] == 2
            else:
                assert scipy_sparse.issparse(features1), features1
                assert scipy_sparse.issparse(features2), features2
                features12 = scipy_sparse.hstack((features2, features1), format='csr')
        # features1 = None

        return features12

    def __concatenate_features__(self, features1, features2):

        if isinstance(features1, (np.ndarray, np.generic)):
            assert isinstance(features2, (np.ndarray, np.generic))
            features12 = np.vstack((features1, features2))
        else:
            if scipy_sparse.issparse(features1):
                assert scipy_sparse.issparse(features2)
                features12 = scipy_sparse.vstack((features1, features2))
                assert isinstance(features12, scipy_sparse.csr_matrix), features12
            else:
                features12 = sparse.concatenate(
                    (features1, features2),
                    axis=0,
                )

        return features12

    def __compute_transfer_entropy_func__(
            self,
            features1, features2,
            labels2,
            cond_density_estimator=None,
            ticker2=None,
            ticker1=None,
            features_ad_2=None,
            labels_ad_2=None,
    ):
        # inferring timeseries 2, using the knowledge of timeseries 2 as well as timeseries 1
        # features1 is the previous timesteps from timeseries1 treated as features
        # features2 is the previous timesteps from timeseries2 treated as features
        # labels2 is the current timestep (under inference) from timeseries 2

        # transfer entropy is a directional measure, see wiki link:
        # https://en.wikipedia.org/wiki/Transfer_entropy

        # labels are modeled as poisson distributed, so ensuring that it is int, and not float or boolean
        assert labels2.dtype in [np.int, np.bool]
        if (self.estimation_objective == 'binary') and (labels2.dtype != np.bool):
            labels2 = labels2.astype(np.bool)

        # basic sanity checks to ensure that features and labels are of same dimensions
        assert labels2.size == features1.shape[0], (labels2.shape, features1.shape)
        assert features1.shape[0] == features2.shape[0], (features1.shape, features2.shape)

        # transfer entropy 1 -> 2
        # H(Y2|X2) - H(Y2|X1,X2)
        # inferring the labels from timeseries 2 using the past of timeseries 1 as well as timeseries 2
        features12 = self.__join_features_from_two_timeseries__(
            features1=features1, features2=features2
        )

        if self.estimator_type in ['gbt']:
            features12 = features12.tocsr()
        elif self.estimator_type in ['nbeats']:
            pass
        else:
            raise AssertionError

        # entropy (uncertainty in inferring timeseries 2 using info from self only)
        entropy2 = self.__compute_cond_entropy__(
            features=features2,
            labels=labels2,
            cond_density_estimator=cond_density_estimator,
            features_ad=features_ad_2,
            labels_ad=labels_ad_2,
        )

        if self.is_data_augment_for_te_estimate:
            # assert self.is_regression_else_classification, 'not implemented'
            features_ad_12, labels_ad_12 = self.generate_labeled_data_for_augment_from_hashing_for_transfer(
                features1=features1,
                features2=features2,
                labels2=labels2,
                ticker1=ticker1,
                ticker2=ticker2,
            )
            if labels2.dtype == np.bool:
                assert labels_ad_12.dtype == np.bool
        else:
            features_ad_12, labels_ad_12 = (None,)*2

        # entropy (uncertainty in inferring timeseries 2 using info from self as well as timeseries 1)
        entropy12 = self.__compute_cond_entropy__(
            features=features12,
            labels=labels2,
            cond_density_estimator=cond_density_estimator,
            features_ad=features_ad_12,
            labels_ad=labels_ad_12,
        )

        if self.normalize_te:

            entropy12 = max(entropy12, 0.0)
            entropy2 = max(entropy2, 0.0)

            if entropy2 == 0.0:
                transfer_entropy = 0.0
            else:
                # although not standard, normalizing transfer entropy, i.e. relative reduction in uncertainty from using the knowledge of timeseries 1
                # normalized transfer entropy is always in range [0.0, 1.0]
                assert entropy2 > 0.0, entropy2
                transfer_entropy = (entropy2 - entropy12)/float(entropy2)

            # in practical scenarios, adding features can hurt due to the curse of dimensionality
            # (although information never hurts in information theory),
            # transfer entropy can be negative in practical settings (not in theory)
            transfer_entropy = max(transfer_entropy, 0.0)
        else:
            transfer_entropy = (entropy2 - entropy12)

        return transfer_entropy, entropy2

    def compute_transfer_entropy_for_i(self, i):

        # for timeseries i, compute transfer entropies from j to i,
        # specifically, reduction in uncertainty for inferring timeseries i conditioning upon timeseries j

        transfer_entropies = np.zeros(self.num_time_series, dtype=np.float)

        # uncertainty for inferring ith timeseries
        # values should be same for all j, this is just for validating that it is so
        conditional_entropies_i = np.zeros(self.num_time_series, dtype=np.float)

        # sparse arrays
        labels_i = self.labels[i].toarray().flatten()
        features_i = self.features[i].tocsr()

        print(features_i.shape, labels_i.shape)

        meta_data = f'{self.ticker_names[i] if self.ticker_names is not None else i}\n '
        meta_data += f'% of positive labels={round((100.0*np.count_nonzero(labels_i))/labels_i.size, 1)}\n '
        meta_data += f'% of non-zero features={round((100.0*np.count_nonzero(features_i.sum(1)))/features_i.shape[0], 1)}\n'
        print(meta_data)

        logical_idx = None
        if self.negative_sample:
            assert not self.is_regression_else_classification
            logical_idx = self.negative_sampler_obj.negative_samples_idx(labels_i=labels_i, features_i=features_i)
            # print(f'Negative Sampling: {labels_i.size} to {logical_idx.sum()}')
            if logical_idx is not None:
                assert logical_idx.size == labels_i.size
                # print(features_i.shape, logical_idx.shape, logical_idx)
                labels_i = labels_i[logical_idx]
                features_i = features_i[logical_idx, :]

        if (self.estimation_objective == 'binary') and (labels_i.dtype != np.bool):
            labels_i = labels_i.astype(np.bool)

        if self.tune_hyperparameters:

            features_for_adv_reg = None
            adv_reg_obj = None

            if self.is_hashcode_regularization:
                if self.is_augment_data_from_hashcodes_for_tuning:
                    hashcode_model_obj = hmte.HashcodeModel(
                        dir_path_for_saving_hash_models=self.dir_path_for_saving_hash_models,
                    )
                    features_i_arr = features_i if self.is_regression_else_classification else features_i.toarray()
                    hashcodes_for_features_i = hashcode_model_obj.compute_hashcodes(
                        features=features_i_arr,
                        is_optimize=False,
                        model_name=self.ticker_names[i],
                    )
                    assert hashcodes_for_features_i.shape[0] == features_i_arr.shape[0]
                    adv_reg_obj = ahr.AdversarialHashRegularizer(
                        C=hashcodes_for_features_i,
                        log_base_for_k=4.0,
                        max_val_for_k=8,
                        alpha=self.alpha_for_gen_sampling,
                        num_hash_funcs=self.num_hash_funcs,
                    )
                    features_for_adv_reg = adv_reg_obj.generated_data_from_hashcodes(
                        events=features_i_arr,
                        multiple=self.augment_multiplier,
                    )
                    hashcodes_for_adv_reg = hashcode_model_obj.compute_hashcodes(
                        features=features_for_adv_reg,
                        is_optimize=False,
                        model_name=self.ticker_names[i],
                    )
                    if self.debug:
                        print(f'Generated {features_for_adv_reg.shape} with hashcodes {hashcodes_for_adv_reg.shape}.')
                    features_for_adv_reg = np.vstack((features_i_arr, features_for_adv_reg))
                    features_i_arr = None
                    hashcodes_for_adv_reg = np.vstack((hashcodes_for_features_i, hashcodes_for_adv_reg))
                    hashcodes_for_features_i = None
                    adv_reg_obj = ahr.AdversarialHashRegularizer(
                        C=hashcodes_for_adv_reg,
                        log_base_for_k=4.0,
                        max_val_for_k=8,
                        alpha=self.alpha_for_gen_sampling,
                        num_hash_funcs=self.num_hash_funcs,
                    )
                else:
                    print('hashing ...')
                    print(self.dir_path_for_saving_hash_models)
                    hashcode_model_obj = hmte.HashcodeModel(
                        dir_path_for_saving_hash_models=self.dir_path_for_saving_hash_models,
                    )
                    features_for_adv_reg, hashcodes_for_adv_reg = hashcode_model_obj.load_pretrained_hashcodes_and_events(
                        model_name=self.ticker_names[i],
                    )
                    print(features_for_adv_reg.shape, hashcodes_for_adv_reg.shape)
                    adv_reg_obj = ahr.AdversarialHashRegularizer(
                        C=hashcodes_for_adv_reg,
                        log_base_for_k=4.0,
                        max_val_for_k=8,
                        alpha=self.alpha_for_gen_sampling,
                        num_hash_funcs=self.num_hash_funcs,
                    )
                hashcode_model_obj = None

            cond_density_estimator = self.prepare_conditional_density_estimator(
                features_for_tune=features_i,
                labels_for_tune=labels_i,
                meta_data=meta_data,
                features_for_adv_reg=features_for_adv_reg,
                adv_reg_obj=adv_reg_obj,
            )
            assert cond_density_estimator is not None
        else:
            cond_density_estimator = None

        if self.is_data_augment_for_te_estimate:
            # assert self.is_regression_else_classification, 'not implemented'
            features_ad_i, labels_ad_i = self.generate_labeled_data_for_augment_from_hashing(
                features_arr=features_i, labels=labels_i,
                ticker=self.ticker_names[i],
            )
            if labels_i.dtype == np.bool:
                assert labels_ad_i.dtype == np.bool
        else:
            features_ad_i, labels_ad_i = (None,)*2

        # compute efficient, but not memory efficient since it
        # requires access to all the timeseries rather than a subset
        for j in range(self.num_time_series):

            # print(self.features.shape, logical_idx.shape if logical_idx is not None else None)
            features_j = self.features[j].tocsr() if logical_idx is None else self.features[j, logical_idx].tocsr()
            transfer_entropies[j], conditional_entropies_i[j] = self.__compute_transfer_entropy_func__(
                features1=features_j,
                features2=features_i,
                labels2=labels_i,
                cond_density_estimator=cond_density_estimator,
                ticker2=self.ticker_names[i],
                ticker1=self.ticker_names[j],
                features_ad_2=features_ad_i,
                labels_ad_2=labels_ad_i,
            )
            features_j = None

        print(f'{self.ticker_names[i]}: {round(transfer_entropies.mean(), 2)}+-{round(transfer_entropies.std(), 2)}')

        return transfer_entropies, conditional_entropies_i[0]

    def build(self, time_series_data, ticker_names=None, trial_ids=None):

        self.num_time_steps = time_series_data.shape[1]
        assert self.time_lag < self.num_time_steps
        self.num_data = self.num_time_steps-self.time_lag
        self.num_time_series = time_series_data.shape[0]

        assert (ticker_names is None) or (ticker_names.size == self.num_time_series)

        self.features, self.labels = map_multivariate_timeseries_to_features_and_labels(
            time_series_data=time_series_data, time_lag=self.time_lag,
            trial_ids=trial_ids,
        )

        print(self.features.shape, self.labels.shape)

        self.ticker_names = ticker_names

        return self.features, self.labels

    def compute_transfer_entropy(self):

        # (j, i): reduction in uncertainty (entropy) for inferring timseries i given the knowledge of timeseries j
        transfer_entropies = np.zeros((self.num_time_series, self.num_time_series), dtype=np.float)
        conditional_entropies = np.zeros(self.num_time_series, dtype=np.float)

        if self.num_cores == 1:
            for i in range(self.num_time_series):
                transfer_entropies[:, i], conditional_entropies[i] = self.compute_transfer_entropy_for_i(i=i)
        else:
            assert self.num_cores > 1
            with multiprocessing.Pool(processes=min(self.num_cores, self.num_time_series)) as pool:
                results = pool.map(self.compute_transfer_entropy_for_i, np.arange(self.num_time_series, dtype=np.int))

            for i in range(self.num_time_series):
                transfer_entropies[:, i], conditional_entropies[i] = results[i]
                print('.', end='')
            print('')

        return transfer_entropies, conditional_entropies
