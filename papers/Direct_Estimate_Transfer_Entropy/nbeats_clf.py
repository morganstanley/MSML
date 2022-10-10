import importlib

import numpy as np

from hyperopt.pyll import scope
from hyperopt import hp, fmin, tpe, space_eval, Trials, STATUS_OK

import constants

if constants.venv_sites_path is not None:
    import site
    site.addsitedir(constants.venv_sites_path)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['KERAS_BACKEND'] = "tensorflow"
os.environ['TF_DETERMINISTIC_OPS'] = '1'

import tensorflow as tf

tf.random.set_seed(0)
tf.autograph.set_verbosity(0)
tf.get_logger().setLevel('ERROR')
tf.config.threading.set_intra_op_parallelism_threads(1)

import tensorflow.keras as tfk
import tensorflow.keras.initializers as tfki

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Input, Dense, Lambda, Subtract, Add, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model

import clf_model_for_te_estimation as cte
importlib.reload(cte)


class NBeatsNet:

    GENERIC_BLOCK = 'generic'
    TREND_BLOCK = 'trend'
    SEASONALITY_BLOCK = 'seasonality'
    _FORECAST = 'forecast'

    def __init__(self,
                 seed=0,
                 input_dim=1,
                 output_dim=1,
                 exo_dim=0,
                 backcast_length=10,
                 forecast_length=1,
                 stack_types=(TREND_BLOCK, SEASONALITY_BLOCK),
                 nb_blocks_per_stack=3,
                 thetas_dim=(4, 8),
                 share_weights_in_stack=False,
                 hidden_layer_units=256,
                 nb_harmonics=None
    ):

        self.seed= seed

        self.stack_types = stack_types
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.thetas_dim = thetas_dim
        self.units = hidden_layer_units
        self.share_weights_in_stack = share_weights_in_stack
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.exo_dim = exo_dim
        self.input_shape = (self.backcast_length, self.input_dim)
        self.exo_shape = (self.backcast_length, self.exo_dim)
        self.output_shape = (self.forecast_length, self.output_dim)
        self.weights = {}
        self.nb_harmonics = nb_harmonics
        assert len(self.stack_types) == len(self.thetas_dim)

    def create_model(self):

        x = Input(shape=self.input_shape, name='input_variable')
        x_ = {}
        for k in range(self.input_dim):
            x_[k] = Lambda(lambda z: z[..., k])(x)

        e_ = {}
        if self.has_exog():
            e = Input(shape=self.exo_shape, name='exos_variables')
            for k in range(self.exo_dim):
                e_[k] = Lambda(lambda z: z[..., k])(e)
        else:
            e = None

        y_ = {}
        for stack_id in range(len(self.stack_types)):
            stack_type = self.stack_types[stack_id]
            nb_poly = self.thetas_dim[stack_id]

            for block_id in range(self.nb_blocks_per_stack):
                forecast = self.create_block(x_, e_, stack_id, block_id, stack_type, nb_poly)

                for k in range(self.input_dim):
                    if stack_id == 0 and block_id == 0:
                        y_[k] = forecast[k]
                    else:
                        y_[k] = Add()([y_[k], forecast[k]])

        for k in range(self.input_dim):
            y_[k] = Reshape(target_shape=(self.forecast_length, 1))(y_[k])

        if self.input_dim > 1:
            y_ = Concatenate()([y_[ll] for ll in range(self.input_dim)])
        else:
            y_ = y_[0]

        assert self.output_dim == 1
        # print(y_.shape)
        y_ = Dense(
            2,
            activation='softmax',
            kernel_initializer=tfki.glorot_uniform(seed=self.seed),
            name='reg_y',
            use_bias=True,
            bias_initializer='zeros',
        )(y_)

        inputs_x = [x, e] if self.has_exog() else x
        model = Model(inputs_x, y_, name=self._FORECAST)

        return model

    def has_exog(self):
        # exo/exog is short for 'exogenous variable', i.e. any input
        # features other than the target time-series itself.
        return self.exo_dim > 0

    @staticmethod
    def load(filepath, custom_objects=None, compile=True):
        return load_model(filepath, custom_objects, compile)

    def _r(self, layer_with_weights, stack_id):
        # mechanism to restore weights when block share the same weights.
        # only useful when share_weights_in_stack=True.
        if self.share_weights_in_stack:
            layer_name = layer_with_weights.name.split('/')[-1]
            try:
                reused_weights = self.weights[stack_id][layer_name]
                return reused_weights
            except KeyError:
                pass
            if stack_id not in self.weights:
                self.weights[stack_id] = {}
            self.weights[stack_id][layer_name] = layer_with_weights
        return layer_with_weights

    def create_block(self, x, e, stack_id, block_id, stack_type, nb_poly):

        # register weights (useful when share_weights_in_stack=True)
        def reg(layer):
            return self._r(layer, stack_id)

        # update name (useful when share_weights_in_stack=True)
        def n(layer_name):
            return '/'.join([str(stack_id), str(block_id), stack_type, layer_name])

        forecast_ = {}

        d1 = reg(Dense(
            self.units, activation='relu',
            kernel_initializer=tfki.glorot_uniform(seed=self.seed),
            name=n('d1'),
        ))

        d2 = reg(Dense(
            self.units, activation='relu',
            kernel_initializer=tfki.glorot_uniform(seed=self.seed),
            name=n('d2'),
        ))

        d3 = reg(Dense(
            self.units, activation='relu',
            kernel_initializer=tfki.glorot_uniform(seed=self.seed),
            name=n('d3'),
        ))

        d4 = reg(Dense(
            self.units, activation='relu',
            kernel_initializer=tfki.glorot_uniform(seed=self.seed),
            name=n('d4'),
        ))

        if stack_type == 'generic':

            theta_f = reg(Dense(
                nb_poly, activation='linear',
                use_bias=False,
                kernel_initializer=tfki.glorot_uniform(seed=self.seed),
                name=n('theta_f'),
            ))

            forecast = reg(Dense(
                self.forecast_length,
                activation='linear',
                kernel_initializer=tfki.glorot_uniform(seed=self.seed),
                name=n('forecast'),
            ))

        elif stack_type == 'trend':
            theta_f = reg(Dense(
                nb_poly, activation='linear',
                use_bias=False,
                kernel_initializer=tfki.glorot_uniform(seed=self.seed),
                name=n('theta_f_b'),
            ))

            forecast = Lambda(
                trend_model,
                arguments={
                    'forecast_length': self.forecast_length
                }
            )
        else:  # 'seasonality'
            theta_f = reg(Dense(
                self.forecast_length, activation='linear',
                use_bias=False,
                kernel_initializer=tfki.glorot_uniform(seed=self.seed),
                name=n('theta_f')
            ))

            forecast = Lambda(
                seasonality_model,
                arguments={'forecast_length': self.forecast_length}
            )

        for k in range(self.input_dim):

            if self.has_exog():
                d0 = Concatenate()([x[k]] + [e[ll] for ll in range(self.exo_dim)])
            else:
                d0 = x[k]

            d1_ = d1(d0)
            d2_ = d2(d1_)
            #             d3_ = d3(d2_)
            #             d4_ = d4(d3_)
            #             theta_f_ = theta_f(d4_)
            theta_f_ = theta_f(d1_)
            forecast_[k] = forecast(theta_f_)

        return forecast_

    def __getattr__(self, name):

        attr = getattr(self.model, name)

        if not callable(attr):
            return attr

        def wrapper(*args, **kwargs):
            cast_type = self._FORECAST
            return getattr(self.model, attr.__name__)(*args, **kwargs)

        return wrapper


def linear_space(forecast_length):
    horizon = forecast_length
    return K.arange(0, horizon) / horizon


def seasonality_model(thetas, forecast_length):

    p = thetas.get_shape().as_list()[-1]
    p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)

    t = linear_space(forecast_length)

    s1 = K.stack([K.cos(2 * np.pi * i * t) for i in range(p1)])
    s2 = K.stack([K.sin(2 * np.pi * i * t) for i in range(p2)])

    if p == 1:
        s = s2
    else:
        s = K.concatenate([s1, s2], axis=0)

    s = K.cast(s, np.float32)

    return K.dot(thetas, s)


def trend_model(thetas, forecast_length):
    p = thetas.shape[-1]
    t = linear_space(forecast_length)
    t = K.transpose(K.stack([t ** i for i in range(p)]))
    t = K.cast(t, np.float32)

    return K.dot(thetas, K.transpose(t))


class NBeatsClfNetwork(cte.ClfTransferEntropyEstimation):
    # Official one from same source as N-beats

    def __init__(
            self,
            num_cores=1, debug=False,
            nb_blocks_per_stack = 8,
            thetas_dim = (8, 8),
            share_weights_in_stack = True,
            hidden_layer_units = 1024,
            lambda_entropy_predictor_cond_hashcodes=0.0,
            batchSize=2**12,
            lr=1e-3,
            nbEpochs=10,
            seed=0,
    ):

        super(NBeatsClfNetwork, self).__init__(
            num_cores=num_cores,
            debug=debug,
            objective='binary',
            lambda_entropy_predictor_cond_hashcodes=lambda_entropy_predictor_cond_hashcodes,
        )

        self.num_cores = num_cores
        self.objective = 'binary'

        self.parameters = {
            'batchSize': batchSize,
            'lr': lr,
            'nbEpochs': nbEpochs,
        }

        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.thetas_dim = thetas_dim
        self.share_weights_in_stack = share_weights_in_stack
        self.hidden_layer_units = hidden_layer_units

        self.seed = seed

        self.max_evals = 45

    def get_hyperparameters_space(self):

        return {
            'lr': scope.float(hp.quniform('learning_rate', 0.0001, 0.0005, 0.001)),
            'nbEpochs': scope.int(hp.choice('nbEpochs',  [1, 3, 10])),
            'batchSize': scope.int(hp.choice('batchSize',  [2**10, 2**11, 2**12])),
        }

    def post_process_tuned_parameters(self, hp_space, tuned_parameters):

        tuned_parameters['nbEpochs'] = int(tuned_parameters['nbEpochs'])
        tuned_parameters['batchSize'] = int(tuned_parameters['batchSize'])
        tuned_parameters['lr'] = float(tuned_parameters['lr'])

        return tuned_parameters

    def compile_dl_model(self, time_lag, lr, num_timeseries=1):

        obj = NBeatsNet(
            input_dim=num_timeseries,
            output_dim=1,
            exo_dim=0,
            backcast_length=time_lag,
            forecast_length=1,
            nb_blocks_per_stack=self.nb_blocks_per_stack,
            thetas_dim=self.thetas_dim,
            share_weights_in_stack=self.share_weights_in_stack,
            hidden_layer_units=self.hidden_layer_units,
            nb_harmonics=None,
            stack_types=(
                NBeatsNet.GENERIC_BLOCK,
                NBeatsNet.GENERIC_BLOCK
            ),
            seed=self.seed,
        )

        deepModel = obj.create_model()
        opt = tfk.optimizers.Nadam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-6)

        deepModel.compile(
            optimizer=opt,
            loss='binary_crossentropy',
        )

        return deepModel

    def __format_features__(self, features):

        if features is None:
            return features

        if features.ndim == 3:
            features = features.todense()
        else:
            # csr 2-D to dense
            assert features.ndim == 2
            features = features.toarray()

        return features

    def predict_labels_for_adv_reg(
            self,
            features, labels,
            features_for_adv_reg,
            parameters=None,
    ):

        assert labels.ndim == 1
        assert features.shape[0] == labels.shape[0]

        time_lag = features.shape[-1]

        if parameters is None:
            parameters = self.parameters

        assert (features.ndim == 2) or (features.shape[1] == 2)
        num_timeseries = features.shape[-2] if features.ndim == 3 else 1

        features = self.__format_features__(features)

        deep_model = self.compile_dl_model(
            time_lag,
            lr=parameters['lr'],
            num_timeseries=num_timeseries,
        )

        assert (features.ndim == 2) or (features.shape[1] == 2)
        features = features[:, :, None] if features.ndim == 2 else np.moveaxis(features, source=-2, destination=-1)

        #Train model
        deep_model.fit(
            x=np.copy(features),
            y=labels,
            batch_size=parameters['batchSize'],
            epochs=parameters['nbEpochs'],
            workers=1,
            use_multiprocessing=False,
            shuffle=False,
        )

        assert (features_for_adv_reg.ndim == 2) or (features_for_adv_reg.shape[1] == 2)
        features_for_adv_reg = features_for_adv_reg[:, :, None] if features_for_adv_reg.ndim == 2 else np.moveaxis(features_for_adv_reg, source=-2, destination=-1)

        # assuming it to be prediction probabilities if binary labels and intensities otherwise
        inferred_labels = deep_model.predict(
            np.copy(features_for_adv_reg)
        )[:, 0, 1]

        return inferred_labels

    def compute_probs(
            self,
            features, labels,
            test_features=None,
            test_labels=None,
            parameters=None,
    ):

        if test_labels is not None:
            assert test_labels.dtype == np.float

        time_lag = features.shape[-1]
        assert (test_features is None) or (time_lag == test_features.shape[-1])

        num_timeseries = features.shape[-2] if features.ndim == 3 else 1

        if parameters is None:
            parameters = self.parameters

        features = self.__format_features__(features)
        test_features = self.__format_features__(test_features)

        deep_model = self.compile_dl_model(
            time_lag,
            lr=parameters['lr'],
            num_timeseries=num_timeseries,
        )

        assert (features.ndim == 2) or (features.shape[1] == 2)
        features = features[:, :, None] if features.ndim == 2 else np.moveaxis(features, source=-2, destination=-1)

        deep_model.fit(
            x=np.copy(
                features
            ),
            y=labels,
            batch_size=parameters['batchSize'],
            verbose=0,
            epochs=parameters['nbEpochs'],
        )

        pred_prob = deep_model.predict(
            np.copy(
                features
            )
        )[:, 0, 1]
        pred_prob[~labels] = 1.0 - pred_prob[~labels]

        if test_features is not None:
            assert test_labels is not None
            assert test_labels.dtype == np.float

            assert (test_features.ndim == 2) or (test_features.shape[1] == 2)
            test_features = test_features[:, :, None] if test_features.ndim == 2 else np.moveaxis(test_features, source=-2, destination=-1)

            test_pred_prob = deep_model.predict(np.copy(test_features))[:, 0, 1]
            test_pred_prob[~test_labels] = 1.0 - test_pred_prob[~test_labels]

            return pred_prob, test_pred_prob
        else:
            return pred_prob
