from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import Array
import chex

import icl.utils as u
from icl.gpt2 import GPT2Config, GPT2Model, init_fn


########################################################################################################################
# Utilities                                                                                                            #
########################################################################################################################


def get_model_name(model):
    if isinstance(model, Ridge):
        return "Ridge"
    elif isinstance(model, DiscreteMMSE):
        return "dMMSE"
    elif isinstance(model, Transformer):
        return "Transformer"
    elif isinstance(model, SingleSeqTransformer):
        return "SingleSeqTransformer"
    elif isinstance(model, LastValue):
        return "LastValue"
    elif isinstance(model, ARMA):
        return "ARMA"
    elif isinstance(model, CorrectedLastValue):
        return "CorrectedLastValue"
    else:
        raise ValueError(f"model type={type(model)} not supported")


########################################################################################################################
# Transformer                                                                                                          #
########################################################################################################################


class Transformer(nn.Module):
    n_points: int
    n_layer: int
    n_embd: int
    n_head: int
    seed: int
    dtype: Any
    use_ln: bool = True
    use_linear_attention: bool = False

    def setup(self):
        config = GPT2Config(
            block_size=2 * self.n_points,
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_embd=self.n_embd,
            dtype=self.dtype,
            use_ln=self.use_ln,
            use_linear_attention=self.use_linear_attention,
        )
        self._in = nn.Dense(self.n_embd, False, self.dtype, kernel_init=init_fn)
        self._h = GPT2Model(config)
        self._out = nn.Dense(1, False, self.dtype, kernel_init=init_fn)

    def __call__(self, data: Array, targets: Array, attention_mask: Array, training: bool = False) -> Array:
        # Batch size
        batch_size = data.shape[0]
        # Get actual sequence length before padding
        actual_seq_len = data.shape[1]
        # Data features
        n_features = data.shape[2]
        
        chex.assert_shape(data, (batch_size, actual_seq_len, n_features))
        chex.assert_shape(targets, (batch_size, actual_seq_len))
        
        # Pad input sequence to match the model's expected block_size
        target_seq_len = self.n_points  # Expected number of data points

        chex.assert_shape(attention_mask, (batch_size, 2 * self.n_points, 2 * self.n_points))

        input_seq = u.to_seq(data, targets, target_seq_len=target_seq_len)

        embds = self._in(input_seq)
        outputs = self._h(input_embds=embds, attention_mask=attention_mask, training=training)
        preds = self._out(outputs)
        preds = u.seq_to_targets(preds, actual_seq_len=actual_seq_len)
        return preds

########################################################################################################################
# Single Seq Transformer                                                                                                          #
########################################################################################################################


class SingleSeqTransformer(nn.Module):
    n_points: int
    n_layer: int
    n_embd: int
    n_head: int
    seed: int
    dtype: Any
    use_ln: bool = True
    use_linear_attention: bool = False
    n_out: int = 1

    def setup(self):
        config = GPT2Config(
            block_size=self.n_points,
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_embd=self.n_embd,
            dtype=self.dtype,
            use_ln=self.use_ln,
            use_linear_attention=self.use_linear_attention,
        )
        self._in = nn.Dense(self.n_embd, False, self.dtype, kernel_init=init_fn)
        self._h = GPT2Model(config)
        self._out = nn.Dense(self.n_out, False, self.dtype, kernel_init=init_fn)

    def __call__(self, data: Array, targets: Array, attention_mask: Array, training: bool = False) -> Array:
        # Batch size
        batch_size = data.shape[0]
        # Get actual sequence length before padding
        actual_seq_len = data.shape[1]
        # Target features
        n_data_features = data.shape[2]
        
        chex.assert_shape(data, (batch_size, actual_seq_len, n_data_features))
        
        # Pad input sequence to match the model's expected block_size
        data_seq_len = self.n_points  # Expected number of data points

        chex.assert_shape(attention_mask, (batch_size, self.n_points, self.n_points))

        input_seq = u.pad_sequence(data, target_seq_len=data_seq_len)
        chex.assert_shape(input_seq, (batch_size, self.n_points, n_data_features))

        embds = self._in(input_seq)
        outputs = self._h(input_embds=embds, attention_mask=attention_mask, training=training)
        preds = self._out(outputs)
        chex.assert_shape(preds, (batch_size, self.n_points, n_data_features))

        preds = u.unpad_sequence(preds, actual_seq_len=actual_seq_len)
        chex.assert_shape(preds, (batch_size, actual_seq_len, n_data_features))

        return preds

class LastValue(nn.Module):
    """
    A simple model that returns the last value of the input sequence.
    This is useful for tasks where the last value is the target.
    """

    def __call__(self, data: Array, targets: Array) -> Array:
        """
        Args:
            data: batch_size x n_points x n_dims (float)
            targets: batch_size x n_points x n_dims (float)
        Return:
            batch_size x n_points x n_dims (float)
        """
        batch_size, n_points, n_dims = data.shape

        return data

class CorrectedLastValue(nn.Module):
    """
    A simple model that returns the last value of the input sequence adjusted by the mean change.
    This is useful for tasks where the last value is the target but with a bias correction.
    """

    def __call__(self, data: Array, targets: Array) -> Array:
        """
        Args:
            data: batch_size x n_points x n_dims (float)
            targets: batch_size x n_points x n_dims (float)
        Return:
            batch_size x n_points x n_dims (float)
        """
        batch_size, n_points, n_dims = data.shape
        
        change = data[:, 1:, :] - data[:, :-1, :]  # batch_size x (n_points - 1) x n_dims
        chex.assert_shape(change, (batch_size, n_points - 1, n_dims))

        cum_sum_change = jnp.cumsum(change, axis=1)  # batch_size x (n_points - 1) x n_dims
        chex.assert_shape(cum_sum_change, (batch_size, n_points - 1, n_dims))

        average_change = cum_sum_change / jnp.arange(1, n_points).reshape(1, -1, 1)  # batch_size x (n_points - 1) x n_dims
        chex.assert_shape(average_change, (batch_size, n_points - 1, n_dims))

        padded_average_change = jnp.concatenate([jnp.zeros((batch_size, 1, n_dims), dtype=data.dtype), average_change], axis=1)  # batch_size x n_points x n_dims
        chex.assert_shape(padded_average_change, (batch_size, n_points, n_dims))

        estimate = data + padded_average_change  # batch_size x n_points x n_dims
        chex.assert_shape(estimate, (batch_size, n_points, n_dims))

        return estimate

########################################################################################################################
# ARMA (AutoRegressive Moving Average)                                                                               #
########################################################################################################################


class ARMA(nn.Module):
    """
    Optimized ARMA(p,q) baseline model for time series prediction using JAX vectorization.
    Fits autoregressive components online using sliding window with JIT compilation.
    
    Model: X_t = φ₁X_{t-1} + ... + φₚX_{t-p} + ε_t (AR-only for computational efficiency)
    """
    ar_order: int = 2        # AR(p) order - number of autoregressive terms
    min_points: int = 4      # Minimum points needed to fit (should be > ar_order)
    window_size: int = 20    # Sliding window size for parameter estimation
    reg_lambda: float = 1e-6 # Regularization for numerical stability
    dtype: Any = jnp.float32

    def __call__(self, data: Array, targets: Array) -> Array:
        """
        Args:
            data: batch_size x n_points x n_dims (float) - unused, ARMA only uses targets
            targets: batch_size x n_points x n_dims (float)
        Return:
            batch_size x n_points x n_dims (float)
        """
        batch_size, n_points, n_dims = targets.shape
        chex.assert_shape(targets, (batch_size, n_points, n_dims))
        
        # Use vectorized prediction with vmap instead of nested loops
        preds = self._predict_all_dimensions_vectorized(targets)
        
        chex.assert_shape(preds, (batch_size, n_points, n_dims))
        return preds

    def _predict_all_dimensions_vectorized(self, targets: Array) -> Array:
        """
        Vectorized prediction across all dimensions and time steps, eliminating loops.
        
        Args:
            targets: batch_size x n_points x n_dims
        Returns:
            predictions: batch_size x n_points x n_dims
        """
        batch_size, n_points, n_dims = targets.shape
        
        # Vectorize over dimensions first
        def predict_single_dimension(targets_1d):
            # targets_1d: batch_size x n_points
            return self._predict_timeseries_vectorized(targets_1d)
        
        # Apply to all dimensions: vmap over n_dims axis (axis 2)
        all_preds = jax.vmap(predict_single_dimension, in_axes=2, out_axes=2)(targets)
        
        return all_preds  # batch_size x n_points x n_dims
    
    def _predict_timeseries_vectorized(self, targets_1d: Array) -> Array:
        """
        Vectorized AR prediction for a single dimension time series.
        
        Args:
            targets_1d: batch_size x n_points
        Returns:
            predictions: batch_size x n_points
        """
        batch_size, n_points = targets_1d.shape
        preds = jnp.zeros_like(targets_1d)
        
        # Vectorize time step processing where possible
        for t in range(n_points):
            if t < self.min_points:
                # Use fallback: zero for first step, last value for subsequent steps
                if t == 0:
                    pred_t = jnp.zeros(batch_size, dtype=self.dtype)
                else:
                    pred_t = targets_1d[:, t-1]  # Last value
            else:
                # Use vectorized AR prediction
                pred_t = self._predict_ar_vectorized(targets_1d, t)
            
            preds = preds.at[:, t].set(pred_t)
        
        return preds
    
    def _predict_ar_vectorized(self, targets_1d: Array, t: int) -> Array:
        """
        Vectorized AR prediction for all batch samples at time t.
        
        Args:
            targets_1d: batch_size x n_points (single dimension)
            t: current timestep
        Returns:
            predictions: batch_size
        """
        batch_size = targets_1d.shape[0]
        
        # Define sliding window
        start_idx = max(0, t - self.window_size)
        history = targets_1d[:, start_idx:t]  # batch_size x history_len
        history_len = t - start_idx
        
        if history_len < self.min_points:
            return targets_1d[:, t-1]  # Fallback to last value
        
        # Vectorized AR fitting across all batches
        return self._fit_ar_batched(history, history_len)
    
    def _fit_ar_batched(self, history: Array, history_len: int) -> Array:
        """
        Fit AR model across all batch samples simultaneously using vectorized operations.
        
        Args:
            history: batch_size x history_len
            history_len: length of history
        Returns:
            predictions: batch_size
        """
        batch_size = history.shape[0]
        effective_order = min(self.ar_order, history_len - 1)
        
        if effective_order < 1:
            return history[:, -1]  # Return last value
        
        num_samples = history_len - effective_order
        if num_samples < 1:
            return history[:, -1]
        
        # Create lagged feature matrix using vectorized slicing
        # X will be batch_size x num_samples x effective_order
        # y will be batch_size x num_samples
        
        # Use advanced indexing to create all lag features at once
        sample_indices = jnp.arange(num_samples)
        lag_indices = jnp.arange(effective_order)
        
        # Create feature indices: for each sample i, features are [i+eff_order-1, i+eff_order-2, ..., i]
        feature_matrix_indices = effective_order + sample_indices[:, None] - 1 - lag_indices[None, :]
        
        # Extract features using advanced indexing
        X = history[:, feature_matrix_indices]  # batch_size x num_samples x effective_order
        
        # Extract targets
        target_indices = jnp.arange(effective_order, effective_order + num_samples)
        y = history[:, target_indices]  # batch_size x num_samples
        
        # Batched least squares solution
        # (X^T X + λI)^{-1} X^T y for each batch sample
        XtX = jnp.matmul(X.transpose(0, 2, 1), X)  # batch_size x effective_order x effective_order
        reg_matrix = self.reg_lambda * jnp.eye(effective_order, dtype=self.dtype)
        XtX_reg = XtX + reg_matrix
        
        Xty = jnp.matmul(X.transpose(0, 2, 1), y[:, :, None])  # batch_size x effective_order x 1
        
        # Solve for AR coefficients
        phi = jnp.linalg.solve(XtX_reg, Xty)[:, :, 0]  # batch_size x effective_order
        
        # Make predictions using most recent values
        recent_values = history[:, -effective_order:]  # batch_size x effective_order
        recent_reversed = jnp.flip(recent_values, axis=1)  # Reverse for [y_{t-1}, y_{t-2}, ...]
        
        predictions = jnp.sum(phi * recent_reversed, axis=1)  # batch_size
        
        return predictions


########################################################################################################################
# Ridge                                                                                                                #
########################################################################################################################


class Ridge(nn.Module):
    lam: float
    dtype: Any

    def __call__(self, data: Array, targets: Array) -> Array:
        """
        Args:
            xs: batch_size x n_points x n_dims (float)
            ys: batch_size x n_points (float)
        Return:
            batch_size x n_points (float)
        """
        batch_size, n_points, _ = data.shape
        targets = jnp.expand_dims(targets, -1)  # batch_size x n_points x 1
        preds = [jnp.zeros(batch_size, dtype=self.dtype)]
        preds.extend(
            [self.predict(data[:, :_i], targets[:, :_i], data[:, _i : _i + 1], self.lam) for _i in range(1, n_points)]
        )
        preds = jnp.stack(preds, axis=1)
        return preds

    def predict(self, X: Array, Y: Array, test_x: Array, lam: float) -> Array:
        """
        Args:
            X: batch_size x i x n_dims (float)
            Y: batch_size x i x 1 (float)
            test_x: batch_size x 1 x n_dims (float)
            lam: (float)
        Return:
            batch_size (float)
        """
        _, _, n_dims = X.shape
        XT = X.transpose((0, 2, 1))  # batch_size x n_dims x i
        XT_Y = XT @ Y  # batch_size x n_dims x 1, @ should be ok (batched matrix-vector product)
        ridge_matrix = jnp.matmul(XT, X, precision=jax.lax.Precision.HIGHEST) + lam * jnp.eye(n_dims, dtype=self.dtype)  # batch_size x n_dims x n_dims
        # batch_size x n_dims x 1
        ws = jnp.linalg.solve(ridge_matrix.astype(jnp.float32), XT_Y.astype(jnp.float32)).astype(self.dtype)
        pred = test_x @ ws  # @ should be ok (batched row times column)
        return pred[:, 0, 0]


########################################################################################################################
# MMSE                                                                                                                #
########################################################################################################################


class DiscreteMMSE(nn.Module):
    scale: float
    task_pool: Array  # n_tasks x n_dims x 1
    dtype: Any

    def __call__(self, data: Array, targets: Array) -> Array:
        """
        Args:
            data: batch_size x n_points x n_dims (float)
            targets: batch_size x n_points (float)
        Return:
            batch_size x n_points (float)
        """
        _, n_points, _ = data.shape
        targets = jnp.expand_dims(targets, -1)  # batch_size x n_points x 1
        W = self.task_pool.squeeze().T  # n_dims x n_tasks  (maybe do squeeze and transpose in setup?)
        preds = [data[:, 0] @ W.mean(axis=1)]  # batch_size
        preds.extend(
            [
                self.predict(data[:, :_i], targets[:, :_i], data[:, _i : _i + 1], W, self.scale)
                for _i in range(1, n_points)
            ]
        )
        preds = jnp.stack(preds, axis=1)  # batch_size x n_points
        return preds

    def predict(self, X: Array, Y: Array, test_x: Array, W: Array, scale: float) -> Array:
        """
        Args:
            X: batch_size x i x n_dims (float)
            Y: batch_size x i x 1 (float)
            test_x: batch_size x 1 x n_dims (float)
            W: n_dims x n_tasks (float)
            scale: (float)
        Return:
            batch_size (float)
        """
        # X @ W is batch_size x i x n_tasks, Y is batch_size x i x 1, so broadcasts to alpha being batch_size x n_tasks
        # alpha = tfd.Normal(0, scale).log_prob(Y - jnp.matmul(X, W, precision=jax.lax.Precision.HIGHEST)).astype(self.dtype).sum(axis=1)
        alpha = jax.scipy.stats.norm.logpdf(Y - jnp.matmul(X, W, precision=jax.lax.Precision.HIGHEST), loc=0, scale=scale).astype(self.dtype).sum(axis=1)
        # softmax is batch_size x n_tasks, W.T is n_tasks x n_dims, so w_mmse is batch_size x n_dims x 1
        w_mmse = jnp.expand_dims(jnp.matmul(jax.nn.softmax(alpha, axis=1), W.T, precision=jax.lax.Precision.HIGHEST), -1)
        # test_x is batch_size x 1 x n_dims, so pred is batch_size x 1 x 1. NOTE: @ should be ok (batched row times column)
        pred = test_x @ w_mmse
        return pred[:, 0, 0]


########################################################################################################################
# Get Model                                                                                                            #
########################################################################################################################

Model = Transformer | Ridge | DiscreteMMSE | ARMA


def get_model(name: str, **kwargs) -> Model:
    models = {
            "transformer": Transformer,
            "ridge": Ridge,
            "discrete_mmse": DiscreteMMSE,
            "single_seq_transformer": SingleSeqTransformer,
            "last_value": LastValue,
            "corrected_last_value": CorrectedLastValue,
            "arma": ARMA
            }
    return models[name](**kwargs)
