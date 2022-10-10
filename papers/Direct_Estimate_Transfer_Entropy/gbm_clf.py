from ms_specific import *

import importlib
import numpy as np

import matplotlib.pyplot as plt

import lightgbm as lgb

from hyperopt import hp
from hyperopt.pyll import scope

import clf_model_for_te_estimation as cte
importlib.reload(cte)


class GradientBoostedClfTree(cte.ClfTransferEntropyEstimation):

    def __init__(
            self,
            num_cores=1, debug=False,
            objective='binary',
            lambda_entropy_predictor_cond_hashcodes=0.0,
            #
            bagging_fraction=1.0,
            bagging_freq = 5,
            learning_rate=1.0,
            feature_fraction=1.0,
            max_depth=-1,
            is_unbalance=False,
            pos_bagging_fraction=1.0,
            neg_bagging_fraction=1.0,
            min_data_in_leaf=20,
            num_leaves=31,
            lambda_l1=0.0,
            lambda_l2=0.0,
            path_smooth=0.0,
            #
            num_max_evals_for_tuning=300,
            seed=0,
    ):

        assert objective == 'binary'

        super(GradientBoostedClfTree, self).__init__(
            num_cores=num_cores,
            debug=debug,
            objective=objective,
            lambda_entropy_predictor_cond_hashcodes=lambda_entropy_predictor_cond_hashcodes,
            seed=seed,
        )

        self.max_evals = num_max_evals_for_tuning
        self.num_cores = num_cores

        self.parameters = {
            'objective': objective,
            'tree_learner': 'serial',
            'verbose': -1,
            'boosting': 'gbdt',
            'metric': '',
            'device_type': 'cpu',
            'num_threads': 1,
            'seed': self.seed,
            'bagging_fraction': bagging_fraction,
            'bagging_freq': bagging_freq,
            'learning_rate': learning_rate,
            'feature_fraction': feature_fraction,
            'is_unbalance': is_unbalance,
            'lambda_l1': lambda_l1,
            'lambda_l2': lambda_l2,
            'max_depth': max_depth,
            'min_data_in_leaf': min_data_in_leaf,
            'num_leaves': num_leaves,
            'path_smooth': path_smooth,
            'pos_bagging_fraction': pos_bagging_fraction,
            'neg_bagging_fraction': neg_bagging_fraction,
        }

    def plot_obj_in_hp_space(
            self,
            ce,
            opt_idx,
            meta_data,
            entropy_of_preds=None,
            entropy_cond_hashcodes=None,
            normalized_entropy_cond_hashcodes=None,
    ):
        if (entropy_of_preds is not None) and (np.any(entropy_of_preds != 0)):

            plt.close()
            plt.plot(entropy_cond_hashcodes, entropy_of_preds, 'kx', label='search space')
            plt.plot(entropy_cond_hashcodes[opt_idx], entropy_of_preds[opt_idx], 'g+', ms=12, mew=3, label='min-loss')
            plt.title(meta_data)
            plt.xlabel('H(f|C)')
            plt.ylabel('H(f)')
            plt.show()

        if (entropy_cond_hashcodes is not None) and (np.any(entropy_cond_hashcodes != 0)):

            print(ce.shape, entropy_cond_hashcodes.shape)

            plt.close()
            plt.plot(entropy_cond_hashcodes, ce, 'kx', label='search space')
            plt.plot(entropy_cond_hashcodes[opt_idx], ce[opt_idx], 'g+', ms=12, mew=3, label='min-loss')
            plt.title(meta_data)
            plt.xlabel('H(f|C)')
            plt.ylabel('H(y|Y)')
            plt.show()

        if (normalized_entropy_cond_hashcodes is not None) and (np.any(normalized_entropy_cond_hashcodes != 0)):

            print(ce.shape, normalized_entropy_cond_hashcodes.shape)

            plt.close()
            plt.plot(normalized_entropy_cond_hashcodes, ce, 'kx', label='search space')
            plt.plot(normalized_entropy_cond_hashcodes[opt_idx], ce[opt_idx], 'g+', ms=12, mew=3, label='min-loss')
            plt.title(meta_data)
            plt.xlabel('H(f|C)/H(f)')
            plt.ylabel('H(y|Y)')
            plt.show()

    def get_hyperparameters_space(self):

        return {

            'objective': self.objective,
            'tree_learner': 'serial',
            'verbose': -1,
            'boosting': 'gbdt',
            'metric': '',
            'device_type': 'cpu',
            'num_threads': 1,
            'seed': self.seed,

            'is_unbalance': hp.choice('is_unbalance', [False, True]),

            'bagging_fraction': scope.float(
                hp.choice('bagging_fraction', [0.05, 0.10, 0.30, 1.00]),
            ),

            'bagging_freq': scope.int(
                hp.choice('bagging_freq',  [1, 3, 5, 10]),
            ),

            'pos_bagging_fraction': scope.float(
                hp.choice('pos_bagging_fraction', [0.05, 0.10, 0.30, 1.0]),
            ),

            'neg_bagging_fraction': scope.float(
                hp.choice('neg_bagging_fraction', [0.05, 0.10, 0.30, 1.0]),
            ),

            'feature_fraction': scope.float(
                hp.choice('feature_fraction', [0.05, 0.10, 0.30]),
            ),

            'learning_rate': scope.float(
                hp.choice('learning_rate', [0.6, 1.0]),
            ),

            'max_depth': scope.int(
                hp.choice('max_depth',  [3, 5, 10]),
            ),

            'min_data_in_leaf': scope.int(
                hp.choice('min_data_in_leaf',  [30, 100]),
            ),

            'num_leaves': scope.int(
                hp.choice('num_leaves', [30, 100]),
            ),

            'lambda_l1': scope.float(
                hp.choice('lambda_l1', [1.0, 3.0, 10.0])
            ),

            'lambda_l2': scope.float(
                hp.choice('lambda_l2', [1.0, 3.0, 10.0])
            ),

            'path_smooth': scope.float(
                hp.choice('path_smooth', [0.0, 0.1, 0.3, 1.0, 3.0, 10.0]),
            ),

        }

    def post_process_tuned_parameters(self, hp_space, tuned_parameters):

        tuned_parameters['is_unbalance'] = bool(tuned_parameters['is_unbalance'])
        tuned_parameters['bagging_freq'] = int(tuned_parameters['bagging_freq'])
        tuned_parameters['max_depth'] = int(tuned_parameters['max_depth'])
        tuned_parameters['min_data_in_leaf'] = int(tuned_parameters['min_data_in_leaf'])
        tuned_parameters['num_leaves'] = int(tuned_parameters['num_leaves'])

        tuned_parameters['objective'] = hp_space['objective']
        tuned_parameters['tree_learner'] = hp_space['tree_learner']
        tuned_parameters['verbose'] = hp_space['verbose']
        tuned_parameters['boosting'] = hp_space['boosting']
        tuned_parameters['metric'] = hp_space['metric']
        tuned_parameters['device_type'] = hp_space['device_type']
        tuned_parameters['num_threads'] = hp_space['num_threads']


        return tuned_parameters

    def predict_labels_for_adv_reg(
            self,
            features, labels,
            features_for_adv_reg,
            parameters=None,
    ):
        dataset = lgb.Dataset(
            features, labels,
            free_raw_data=False,
        )

        lgb_obj = lgb.train(
            parameters if parameters is not None else self.parameters,
            dataset,
            verbose_eval=-1,
        )

        # assuming it to be prediction probabilities if binary labels and intensities otherwise
        inferred_labels = lgb_obj.predict(features_for_adv_reg)
        assert inferred_labels.dtype == np.float

        return inferred_labels

    def compute_probs(
            self,
            features, labels,
            test_features=None,
            test_labels=None,
            parameters=None,
    ):
        # assert labels.sum() > 0
        if labels.sum() == 0:
            labels[0] = 1

        if labels.dtype != np.bool:
            print('warning: labels are not bool')
            labels = labels.astype(np.bool)

        if (test_labels is not None) and (test_labels.dtype != np.bool):
            test_labels = test_labels.astype(np.bool)

        dataset = lgb.Dataset(
            features, labels,
            free_raw_data=False,
        )

        lgb_obj = lgb.train(
            parameters if parameters is not None else self.parameters,
            dataset,
            verbose_eval=-1,
        )

        pred_prob = lgb_obj.predict(features)
        assert pred_prob.dtype == np.float
        # print(pred_prob)
        zero_idx = (labels == 0)
        pred_prob[zero_idx] = 1.0 - pred_prob[zero_idx]
        zero_idx = None

        if test_features is not None:

            assert test_labels is not None

            test_pred_prob = lgb_obj.predict(test_features)
            test_zero_idx = (test_labels == 0)
            test_pred_prob[test_zero_idx] = 1.0 - test_pred_prob[test_zero_idx]
            test_zero_idx = None

        if test_features is not None:
            return pred_prob, test_pred_prob
        else:
            return pred_prob
