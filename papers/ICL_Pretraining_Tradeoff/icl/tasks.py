import dataclasses
from typing import Any, Callable, List
import logging

import jax
import jax.numpy as jnp
from jax import Array, tree_util
from functools import partial
import chex
import flax.linen as nn

from icl.models import Model, get_model

########################################################################################################################
# Utilities                                                                                                            #
########################################################################################################################


Sampler = Callable[[int], tuple[Array, Array, Array, Array, Array]]


def get_task_name(task: "Task") -> str:
    return task.name 

@partial(jax.jit, static_argnames=("shape", "dtype"))
def sample_truncated_student(
        key: jax.random.PRNGKey,
        loc: Array,
        scale: Array,
        df: float,
        clip: float,
        shape: tuple[int, ...],
        dtype: Any = jnp.float32
        ) -> Array:
    adjusted_scale = scale * jnp.sqrt((df - 2) / df)  # Adjust scale for variance
    def cond_fun(val):
        _, x = val
        return jnp.any(jnp.abs(x) > clip)

    def body_fun(val):
        key, x = val
        key, subkey = jax.random.split(key)
        new_sample = jax.random.t(key, df, shape=shape, dtype=dtype) * adjusted_scale + loc
        new_x = jax.lax.select(jnp.abs(x) > clip, new_sample, x)
        return key, new_x

    key, subkey = jax.random.split(key)
    init_x = jax.random.t(subkey, df, shape=shape, dtype=dtype) * adjusted_scale + loc
    init_val = (key, init_x)
    _, final_x = jax.lax.while_loop(cond_fun, body_fun, init_val)
    return final_x

def sample_multivariate_gaussian(
        key: jax.random.PRNGKey,
        loc: Array,
        scale: Array,
        clip: float | None,
        shape: tuple[int, ...],
        dtype: Any = jnp.float32
        ) -> Array:
    if clip is not None:
        return jax.random.truncated_normal(
            key,
            lower=(-clip - loc) / scale,
            upper=(clip - loc) / scale,
            shape=shape,
            dtype=dtype
            ) * scale + loc
    else:
        return jax.random.normal(key, shape=shape, dtype=dtype) * scale + loc


def sample_student_t(
        key: jax.random.PRNGKey,
        loc: Array,
        scale: Array,
        df: float,
        shape: tuple[int, ...],
        dtype: Any = jnp.float32
        ) -> Array:
    """Sample from Student-t distribution with location and scale."""
    adjusted_scale = scale * jnp.sqrt((df - 2) / df)  # Adjust scale for variance
    return jax.random.t(key, df, shape=shape, dtype=dtype) * adjusted_scale + loc

def sample_generalized_normal(
        key: jax.random.PRNGKey,
        loc: Array,
        scale: Array,
        beta: float,
        shape: tuple[int, ...],
        dtype: Any = jnp.float32
        ) -> Array:
    """Sample from Generalized Normal distribution with location, scale, and shape parameter beta."""
    adjusted_scale = scale / jnp.sqrt(jax.scipy.special.gamma(3 / beta) / jax.scipy.special.gamma(1 / beta))
    return loc + adjusted_scale * jax.random.generalized_normal(key, beta, shape=shape, dtype=dtype)

def sample_distrib(
        key: jax.random.PRNGKey,
        loc: Array,
        scale: Array,
        clip: float | None,
        distrib_name: str,
        distrib_param: float | None,
        shape: tuple[int, ...],
        dtype: Any = jnp.float32,
        fixed_support: bool = False,
        ref_distrib_name: str = "student",
        ref_distrib_param: float = 3.0,
        ) -> Array:
    """Dispatch to appropriate sampling function based on distribution name."""
    if not fixed_support:
        if distrib_name == "normal" or (distrib_name == "student" and distrib_param == float("inf")):
            # jax.debug.print("Sampling from normal distribution with loc {}, scale {}, clip {}", loc, scale, clip)
            return sample_multivariate_gaussian(key, loc, scale, clip, shape, dtype)
        elif distrib_name == "student":
            # jax.debug.print("Sampling from student-t distribution with loc {}, scale {}, df {}, clip {}", loc, scale, distrib_param, clip)
            if distrib_param is None:
                raise ValueError("distrib_param (degrees of freedom) must be specified for student-t distribution")
            if clip is None:
                return sample_student_t(key, loc, scale, distrib_param, shape, dtype)
            else:
                return sample_truncated_student(key, loc, scale, distrib_param, clip, shape, dtype)
        elif distrib_name == "generalized_normal":
            # jax.debug.print("Sampling from generalized normal distribution with loc {}, scale {}, beta {}", loc, scale, distrib_param)
            if distrib_param is None:
                raise ValueError("distrib_param (shape parameter) must be specified for generalized normal distribution")
            if clip is not None:
                raise NotImplementedError("Clipping not implemented for generalized normal distribution")
            return sample_generalized_normal(key, loc, scale, distrib_param, shape, dtype)
        else:
            raise ValueError(f"Unknown distribution name: {distrib_name}")
    else:
        return sample_distrib(
            key, loc, scale, clip, ref_distrib_name, ref_distrib_param, shape, dtype, fixed_support=False
            )


#@partial(jax.jit, static_argnames=("clip",))
def aux_task_log_weights(
        tasks: Array,
        loc: float,
        scale: float,
        clip: float | None,
        distrib_name: str,
        distrib_param: float | None,
        use_weights: bool = False,
        reduce_axis: int = -1
        ) -> Array:
    if not use_weights:
        return jnp.zeros_like(tasks).sum(axis=reduce_axis)
    
    if distrib_name == "normal" or (distrib_name == "student" and distrib_param == float("inf")):
        if clip is None:
            log_weights = jax.scipy.stats.norm.logpdf(tasks, loc=loc, scale=scale)
        else:
            log_weights = jax.scipy.stats.truncnorm.logpdf(tasks, -clip, clip, loc=loc, scale=scale)
    elif distrib_name == "student":
        if clip is not None:
            raise NotImplementedError("Student-t distribution with clipping not implemented")
        if distrib_param is None:
            raise ValueError("distrib_param (degrees of freedom) must be specified for student-t distribution")
        # Student-t logpdf: loc and scale parameters
        assert distrib_param > 2, "Degrees of freedom must be greater than 2 for Student-t distribution"
        # Match the scale so that variance stays constant as distrib_param changes
        adjusted_scale = scale * jnp.sqrt((distrib_param - 2) / distrib_param)
        standardized = (tasks - loc) / adjusted_scale
        log_weights = jax.scipy.stats.t.logpdf(standardized, df=distrib_param) - jnp.log(adjusted_scale)
    elif distrib_name == "generalized_normal":
        if clip is not None:
            raise NotImplementedError("Generalized normal distribution with clipping not implemented")
        if distrib_param is None:
            raise ValueError("distrib_param (shape parameter) must be specified for generalized normal distribution")
        beta = distrib_param
        adjusted_scale = scale / jnp.sqrt(jax.scipy.special.gamma(3 / beta) / jax.scipy.special.gamma(1 / beta))
        standardized = (tasks - loc) / adjusted_scale
        log_weights = jax.scipy.stats.gennorm.logpdf(standardized, beta)
    else:
        raise ValueError(f"Unknown distribution name: {distrib_name}")
    
    return jnp.sum(log_weights, axis=reduce_axis)

def task_weights_trunc_norm_factor(
        loc: float,
        scale: float,
        clip: float | None,
        use_weights: bool,
        n_dims: int
        ) -> float:
    if not use_weights:
        return 1.0
    # Compute integral of exp(-log_weights) over R^d
    assert clip is not None, "clip must be specified to compute normalization factor"
    x = jnp.linspace(-clip, clip, 1000)
    Z_1d = jax.numpy.trapezoid(
            jnp.exp(-jax.scipy.stats.truncnorm.logpdf(x, -clip, clip, loc=loc, scale=scale)),
            x,
            )
    jax.debug.print("Trunc norm factor 1d: {}", Z_1d)
    return Z_1d ** n_dims

def task_log_weights(
        tasks: Array,
        loc: float,
        scale: float,
        clip: float | None,
        distrib_name: str,
        distrib_param: float | None,
        ref_distrib_name: str = "student",
        ref_distrib_param: float = 3.0,
        use_weights: bool = False,
        reduce_axis: int = -1,
        fixed_support: bool = False,
        ) -> Array:
    if not fixed_support:
        res = jnp.clip(
            #aux_task_log_weights(
             #  tasks, loc, scale, clip, ref_distrib_name, ref_distrib_param, use_weights, reduce_axis
             #   ) \
            - aux_task_log_weights(
                tasks, loc, scale, clip, distrib_name, distrib_param, use_weights, reduce_axis
                ),
            0, None)
    else:
        res = (
            aux_task_log_weights(
                tasks, loc, scale, clip, distrib_name, distrib_param, use_weights=True, reduce_axis=reduce_axis
                ) 
            - aux_task_log_weights(
                tasks, loc, scale, clip, ref_distrib_name, ref_distrib_param, use_weights=True, reduce_axis=reduce_axis
                )
            )
    return res


########################################################################################################################
# Noisy Linear Regression                                                                                              #
########################################################################################################################

"""
Noisy Linear Regression Task for In-Context Learning

This implements a noisy linear regression task y = w^T x + ε where:
- x: input data points (n_dims dimensional)  
- w: task vector (n_dims dimensional)
- ε: Gaussian noise
- y: noisy target values

The task supports two evaluation modes based on task distribution:

**Latent Tasks** (n_tasks > 0):
- Uses a fixed pool of pre-generated task vectors
- Tasks are sampled from this finite pool during training/evaluation
- Model can learn the latent structure of this specific task distribution
- Better for studying how models specialize on repeated task patterns
- Task name ends with the pool size, e.g., "NoisyLinReg(16)"

**Pretrain Tasks** (n_tasks = 0):  
- Generates fresh task vectors from Gaussian distribution each time
- Mimics the diverse task distribution seen during pretraining
- More challenging as model must generalize to completely novel tasks
- Better for studying few-shot learning on unseen tasks
- Task name is "NoisyLinReg(0)"

The task also supports two data sampling modes:

**Fixed Data Pool** (n_data > 0):
- Uses a fixed pool of pre-generated data points
- Data points are sampled from this finite pool during training/evaluation
- Allows studying performance on repeated data patterns

**Fresh Data Sampling** (n_data = 0):
- Generates fresh data points from Gaussian distribution each time
- Each batch contains completely novel data points
- Default behavior for maximum data diversity

Evaluation compares Transformer performance against:
- Ground truth (noise-free predictions)
- Ridge regression baseline (optimal linear predictor given noise/task scales)
"""


@dataclasses.dataclass
class NoisyLinearRegression:
    n_tasks: int
    n_data: int
    n_dims: int
    n_points: int
    batch_size: int
    data_seed: int
    task_seed: int
    noise_seed: int
    data_scale: float
    task_scale: float
    noise_scale: float
    dtype: Any
    task_center: float | None = None
    n_max_points: int | None = None  # Optional, used for padding in some models
    clip: float | None = None  # Optional, clip task vectors to [-clip, clip]^d
    name: str | None = None  # Optional, can be set to override default name
    eval_ridge: bool = True  # Optional, whether to include Ridge baseline in evaluation
    use_weights: bool = False  # Optional, whether to use task importance weights
    use_weight_sampling: bool = False  # Whether to use weighted sampling for tasks
    distrib_name: str = "normal"  # Distribution name: "normal" or "student"
    distrib_param: float | None = None  # Distribution parameter (degrees of freedom for student-t)
    use_curriculum: bool = False  # Whether to use curriculum learning
    curriculum_n_points_increment: int = 2  # Increment for curriculum learning
    curriculum_steps_thresh: int = 2_000  # Steps after which to increment n_points in curriculum learning
    _skip_init: bool = False  # Private parameter to skip __post_init__ logic
    fixed_support: bool = False  # Whether to use fixed support for task weights


    def __post_init__(self):
        if self._skip_init:
            return
        # Validation
        self.data_key = jax.random.PRNGKey(self.data_seed)
        self.task_key = jax.random.PRNGKey(self.task_seed)
        self.noise_key = jax.random.PRNGKey(self.noise_seed)
        self.n_max_points = self.n_points if self.n_max_points is None else self.n_max_points
        self.task_center = 0.0 if self.task_center is None else self.task_center
        task_pool, weights = self.generate_task_pool() if self.n_tasks > 0 else (None, None)
        self.task_pool = task_pool
        self.weights = weights
        self.data_pool = self.generate_data_pool() if self.n_data > 0 else None
        self.name = f"NoisyLinReg({self.n_tasks})" if self.name is None else self.name

    @classmethod
    def from_task_pool(cls, task_pool: Array, weights: Array, **kwargs) -> "NoisyLinearRegression":
        assert kwargs["n_tasks"] == task_pool.shape[0]
        task = cls(**kwargs)
        task.task_pool = task_pool
        task.weights = weights
        return task

    def generate_task_pool(self) -> Array:
        key = jax.random.fold_in(self.task_key, 0)
        shape = self.n_tasks, self.n_dims, 1
        tasks = sample_distrib(key, self.task_center, self.task_scale, self.clip, 
                              self.distrib_name, self.distrib_param, shape, self.dtype, fixed_support=self.fixed_support)

        log_weights = task_log_weights(
                tasks,
                self.task_center,
                self.task_scale,
                self.clip, 
                self.distrib_name,
                self.distrib_param,
                use_weights=self.use_weights,
                reduce_axis=1,
                fixed_support=self.fixed_support
                )
        #weights = jax.nn.softmax(log_weights, axis=0)
        weights = log_weights
        return tasks, weights

    def generate_data_pool(self) -> Array:
        key = jax.random.fold_in(self.data_key, 0)
        shape = self.n_data, self.n_points, self.n_dims
        data = jax.random.normal(key, shape, self.dtype) * self.data_scale
        return data

    @jax.jit
    def sample_data(self, step: int) -> Array:
        key = jax.random.fold_in(self.data_key, step)
        if self.n_data > 0:
            idxs = jax.random.choice(key, self.n_data, (self.batch_size,))
            data = self.data_pool[idxs]
        else:
            shape = self.batch_size, self.n_points, self.n_dims
            data = jax.random.normal(key, shape, self.dtype) * self.data_scale + self.task_center
        return data

    @jax.jit
    def sample_tasks(self, step: int) -> Array:
        key = jax.random.fold_in(self.task_key, step)
        if self.n_tasks > 0:
            idxs = jax.random.choice(key, self.n_tasks, (self.batch_size,))
            # jax.debug.print("Sampled indices for tasks: {}", idxs)
            tasks = self.task_pool[idxs]
            # log_weights = self.weights[idxs] 
            log_weights = task_log_weights(tasks, self.task_center, self.task_scale, self.clip, 
                                         self.distrib_name, self.distrib_param, use_weights=self.use_weights, reduce_axis=1, fixed_support=self.fixed_support)
            #weights = jax.nn.softmax(log_weights, axis=0) * self.batch_size  # Scale weights to match batch size
        else:
            shape = self.batch_size, self.n_dims, 1
            tasks = sample_distrib(key, self.task_center, self.task_scale, self.clip, 
                                 self.distrib_name, self.distrib_param, shape, self.dtype)
            log_weights = task_log_weights(
                    tasks,
                    self.task_center,
                    self.task_scale,
                    self.clip, 
                    self.distrib_name,
                    self.distrib_param,
                    use_weights=self.use_weights,
                    reduce_axis=1,
                    fixed_support=self.fixed_support
                    )
            #weights = jax.nn.softmax(log_weights, axis=0) * self.batch_size  # Scale weights to match batch size
        weights = log_weights
        chex.assert_shape(tasks, (self.batch_size, self.n_dims, 1))
        chex.assert_shape(weights, (self.batch_size, 1))
        # jax.debug.print("Weights sum: {}", jnp.sum(weights))
        # jax.debug.print("Batch statistics: tasks min {}, max {}, mean {}",
        #                jnp.min(tasks), jnp.max(tasks), jnp.mean(tasks))
        return tasks, weights

    @jax.jit
    def evaluate(self, data: Array, tasks: Array, step: int) -> Array:
        targets = (data @ tasks)[:, :, 0]
        key = jax.random.fold_in(self.noise_key, step)
        noise = jax.random.normal(key, targets.shape, self.dtype) * self.noise_scale
        return targets + noise

    @jax.jit
    def generate_attention_mask(self) -> Array:
        """Generate causal attention mask for the sequence with right padding.
        
        Creates a mask of size (2*n_max_points, 2*n_max_points) where:
        - First 2*n_points positions are valid (actual data) 
        - Remaining positions are padded and masked out (right padding)
        - Within valid positions, uses causal attention (can only attend to previous positions)
        """
        effective_seq_len = 2 * self.n_points      # Valid data: positions 0 to this-1
        max_seq_len = 2 * self.n_max_points        # Total padded length
        
        # Start with all positions masked (False)
        mask = jnp.zeros((max_seq_len, max_seq_len), dtype=bool)
        
        # Valid region gets causal attention pattern
        valid_mask = jnp.tril(jnp.ones((effective_seq_len, effective_seq_len))).astype(bool)
        
        # Insert valid causal mask into full mask 
        mask = mask.at[:effective_seq_len, :effective_seq_len].set(valid_mask)
        return mask

    def curriculum_increment(self):
        old_n_points = self.n_points
        self.n_points = min(self.n_points + self.curriculum_n_points_increment, 
                           self.n_max_points)
        if self.n_points > old_n_points:
            logging.info(f"Curriculum increment: n_points {old_n_points} -> {self.n_points}")

    def sample_batch(self, step: int, evl=False) -> tuple[Array, Array, Array, Array]:
        if step % self.curriculum_steps_thresh == self.curriculum_steps_thresh - 1 and self.use_curriculum:
            self.curriculum_increment()
        data, (tasks, weights) = self.sample_data(step), self.sample_tasks(step)
        # print(f"Evl={evl}, weights: {weights}") 
        targets = self.evaluate(data, tasks, step)
        attention_mask = self.generate_attention_mask()
        return data, tasks, weights, targets, attention_mask

    @staticmethod
    @jax.jit
    def evaluate_oracle(data: Array, tasks: Array, targets) -> Array:
        targets = (data @ tasks)[:, :, 0]
        return targets

    def get_default_eval_tasks(
            self, batch_size: int, task_seed: int, data_seed: int, noise_seed: int, eval_n_points: List[int], task_centers: List[float] | None = None, **kwargs
    ) -> list["NoisyLinearRegression"]:
        del kwargs
        assert task_seed != self.task_seed
        assert data_seed != self.data_seed
        assert noise_seed != self.noise_seed
        config = dataclasses.asdict(self)
        config["batch_size"] = batch_size
        config["task_seed"] = task_seed
        config["data_seed"] = data_seed
        config["noise_seed"] = noise_seed
        config["n_tasks"] = 0
        config["n_data"] = 0
        config["n_max_points"] = self.n_max_points
        config["use_curriculum"] = False  # Disable curriculum for evaluation
        config["use_weights"] = False
        eval_tasks = []
        n_points = eval_n_points
        assert n_points <= self.n_max_points, f"n_points {n_points} exceeds n_max_points {self.n_max_points}"
        config["n_points"] = n_points
        # Test  with fresh tasks from training distribution
        name = f"Test tasks"
        config["name"] = name
        eval_tasks.append(self.__class__(**config))

        # Test with same tasks as training distribution
        if self.n_tasks > 0:
            name = f"Train tasks"
            config["n_tasks"] = self.n_tasks
            config["name"] = name
            eval_tasks.append(NoisyLinearRegression.from_task_pool(**config, task_pool=self.task_pool.copy(), weights=self.weights.copy()))
        
        config["n_tasks"] = 0  # Reset for fresh tasks

        # Test with fixed task centers
        if task_centers is not None:
            config["distrib_name"] = "normal"  # Reset to normal distribution for fixed tasks
            config["fixed_support"] = False 
            for task_center in task_centers:
                config["task_center"] = task_center
                # config["task_scale"] = 0.
                config["clip"] = None
                name = f"Fixed task {task_center}"
                config["name"] = name
                eval_tasks.append(self.__class__(**config))
        return eval_tasks

    def get_default_eval_models(self) -> list[Model]:
        if self.eval_ridge:
            models = [get_model(name="ridge", lam=self.noise_scale**2 / self.task_scale**2, dtype=self.dtype)]
            return models
        else:
            return []

    def _tree_flatten(self):
        # Dynamic values (arrays, keys, and values that can change)
        children = (
            self.data_key,
            self.task_key, 
            self.noise_key,
            self.task_pool,
            self.weights,
            self.data_pool,
            self.data_scale,
            self.task_scale,
            self.noise_scale,
            self.task_center,
            self.clip,
        )
        
        # Static values (configuration that doesn't change during execution)
        aux_data = {
            'n_tasks': self.n_tasks,
            'n_data': self.n_data,
            'n_dims': self.n_dims,
            'n_points': self.n_points,
            'batch_size': self.batch_size,
            'data_seed': self.data_seed,
            'task_seed': self.task_seed,
            'noise_seed': self.noise_seed,
            'dtype': self.dtype,
            'n_max_points': self.n_max_points,
            'name': self.name,
            'eval_ridge': self.eval_ridge,
            'use_weights': self.use_weights,
            'distrib_name': self.distrib_name,
            'distrib_param': self.distrib_param,
            'use_curriculum': self.use_curriculum,
            'curriculum_n_points_increment': self.curriculum_n_points_increment,
            'curriculum_steps_thresh': self.curriculum_steps_thresh,
            'fixed_support': self.fixed_support,
        }
        
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        (data_key, task_key, noise_key, task_pool, weights, data_pool,
         data_scale, task_scale, noise_scale, task_center, clip) = children
        
        # Create object with aux_data parameters and placeholder scale values
        obj = cls(data_scale=1.0, task_scale=1.0, noise_scale=1.0, 
                 task_center=0.0, clip=None, _skip_init=True, **aux_data)
        
        # Set the dynamic values
        obj.data_key = data_key
        obj.task_key = task_key
        obj.noise_key = noise_key
        obj.task_pool = task_pool
        obj.weights = weights
        obj.data_pool = data_pool
        obj.data_scale = data_scale
        obj.task_scale = task_scale
        obj.noise_scale = noise_scale
        obj.task_center = task_center
        obj.clip = clip
        
        return obj


# Register NoisyLinearRegression as a PyTree
tree_util.register_pytree_node(NoisyLinearRegression,
                               NoisyLinearRegression._tree_flatten,
                               NoisyLinearRegression._tree_unflatten)

########################################################################################################################
# MLP for SDE drift function
########################################################################################################################

class MLPDrift(nn.Module):
    """Small MLP for computing drift function b(X_t) in SDE: dX_t = -b(X_t) dt + sigma dW_t"""
    n_dims: int
    hidden_size: int
    dtype: Any = jnp.float32
    
    def setup(self):
        self.dense1 = nn.Dense(self.hidden_size, dtype=self.dtype)
        self.dense2 = nn.Dense(self.n_dims, dtype=self.dtype)
    
    def __call__(self, x: Array) -> Array:
        """
        Args:
            x: Input state of shape (batch_size, n_dims) or (batch_size, n_points, n_dims)
        Returns:
            drift: Drift vector of same shape as input
        """
        h = self.dense1(x)
        h = nn.gelu(h)
        h = self.dense2(h)
        normalized_h = jnp.clip(h, -1.0, 1.0)  # Clip to prevent extreme drift values
        res = h - 1e-4 * x  # Add small linear term for stability
        return res

########################################################################################################################
# Ornstein-Uhlenbeck Process Task for In-Context Learning
########################################################################################################################



@dataclasses.dataclass
class OrnsteinUhlenbeckTask:
    n_tasks: int
    n_data: int
    n_dims: int
    n_points: int
    batch_size: int
    data_seed: int
    task_seed: int
    noise_seed: int
    data_scale: float
    task_scale: float
    noise_scale: float
    dtype: Any
    task_center: float | None = None
    n_max_points: int | None = None  # Optional, used for padding in some models
    clip: float | None = None  # Optional, clip task vectors to [-clip, clip]^d
    name: str | None = None  # Optional, can be set to override default name
    eval_ridge: bool = True  # Optional, whether to include Ridge baseline in evaluation
    use_weights: bool = False  # Optional, whether to use task importance weights
    use_weight_sampling: bool = False  # Whether to use weighted sampling for tasks
    distrib_name: str = "normal"  # Distribution name: "normal" or "student"
    distrib_param: float | None = None  # Distribution parameter (degrees of freedom for student-t)
    use_curriculum: bool = False  # Whether to use curriculum learning
    curriculum_n_points_increment: int = 2  # Increment for curriculum learning
    curriculum_steps_thresh: int = 2_000  # Steps after which to increment n_points in curriculum learning
    ou_step: float = 1e-2
    task_n_dims: int | None = None  # Automatically set to 2*n_dims in __post_init__
    _skip_init: bool = False  # Private parameter to skip __post_init__ logic
    fixed_support: bool = False  # Whether to use fixed support for task weights


    def __post_init__(self):
        if self._skip_init:
            return
        # Validation
        self.data_key = jax.random.PRNGKey(self.data_seed)
        self.task_key = jax.random.PRNGKey(self.task_seed)
        self.noise_key = jax.random.PRNGKey(self.noise_seed)
        self.task_n_dims = 2 * self.n_dims  # Each task vector has 3*n_dims: mean, decay rate
        self.n_max_points = self.n_points if self.n_max_points is None else self.n_max_points
        self.task_center = 0.0 if self.task_center is None else self.task_center
        task_pool, weights = self.generate_task_pool() if self.n_tasks > 0 else (None, None)
        self.task_pool = task_pool
        self.weights = weights
        self.data_pool = None
        self.name = f"NoisyLinReg({self.n_tasks})" if self.name is None else self.name

    @classmethod
    def from_task_pool(cls, task_pool: Array, weights: Array, **kwargs):
        assert kwargs["n_tasks"] == task_pool.shape[0]
        task = cls(**kwargs)
        task.task_pool = task_pool
        task.weights = weights
        return task
    
    def get_params_from_tasks(self, tasks: Array) -> tuple[Array, Array]:
        bs = tasks.shape[0]
        chex.assert_shape(tasks, (bs, self.task_n_dims, 1))

        mu = tasks[:, :self.n_dims, 0]  # Mean of the OU process
        chex.assert_shape(mu, (bs, self.n_dims))

        # Decay rate, rescaled to (0.05, 0.15)
        # The higher the tasks values, the slower the decay rate
        #theta = jnp.tanh(- tasks[:, -self.n_dims:, 0]) * 0.1 + 0.2
        theta = jax.nn.sigmoid(-0.4 * tasks[:, -self.n_dims:, 0]) * 0.2 + 0.3
        chex.assert_shape(theta, (bs, self.n_dims))

        return mu, theta

    def generate_task_pool(self) -> Array:
        key = jax.random.fold_in(self.task_key, 0)
        shape = self.n_tasks, self.task_n_dims, 1
        tasks = sample_distrib(key, self.task_center, self.task_scale, self.clip, 
                              self.distrib_name, self.distrib_param, shape, self.dtype, fixed_support=self.fixed_support)

        log_weights = task_log_weights(tasks, self.task_center, self.task_scale, self.clip, 
                                     self.distrib_name, self.distrib_param, use_weights=self.use_weights, reduce_axis=1, fixed_support=self.fixed_support)
        #weights = jax.nn.softmax(log_weights, axis=0)
        weights = log_weights
        # jax.debug.print("Tasks: {tasks}", tasks=tasks)
        # jax.debug.print("Log weights: {log_weights}", log_weights=log_weights)
        return tasks, weights

    def generate_data_pool(self) -> Array:
        key = jax.random.fold_in(self.data_key, 0)
        shape = self.n_data, self.n_points, self.n_dims
        data = jax.random.normal(key, shape, self.dtype) * self.data_scale
        return data

    @jax.jit
    def sample_data(self, step: int) -> Array:
        key = jax.random.fold_in(self.data_key, step)
        if self.n_data > 0:
            idxs = jax.random.choice(key, self.n_data, (self.batch_size,))
            data = self.data_pool[idxs]
        else:
            shape = self.batch_size, self.n_points, self.n_dims
            data = jax.random.normal(key, shape, self.dtype) * self.data_scale + self.task_center
        return data

    @jax.jit
    def sample_tasks(self, step: int) -> Array:
        key = jax.random.fold_in(self.task_key, step)
        if self.n_tasks > 0:
            if self.use_weight_sampling:
                # Sample tasks with replacement based on weights
                idxs = jax.random.categorical(key, self.weights, axis=0, shape=(self.batch_size,))
                log_weights = jnp.zeros((self.batch_size, 1), self.dtype)  # No need to sample weights here
                # jax.debug.print("Using weighted sampling for tasks")
                tasks = self.task_pool[idxs]
            else:
                idxs = jax.random.choice(key, self.n_tasks, (self.batch_size,))
                # log_weights = self.weights[idxs] 
                tasks = self.task_pool[idxs]
                log_weights = task_log_weights(tasks, self.task_center, self.task_scale, self.clip, 
                                             self.distrib_name, self.distrib_param, use_weights=self.use_weights, reduce_axis=1, fixed_support=self.fixed_support)
                # weights = jax.nn.softmax(log_weights, axis=0) * self.batch_size  # Scale weights to match batch size
            # jax.debug.print("Sampled indices for tasks: {}", idxs)
        else:
            shape = self.batch_size, self.task_n_dims, 1
            tasks = sample_distrib(key, self.task_center, self.task_scale, self.clip, 
                                 self.distrib_name, self.distrib_param, shape, self.dtype, fixed_support=self.fixed_support)
            log_weights = task_log_weights(tasks, self.task_center, self.task_scale, self.clip, 
                                         self.distrib_name, self.distrib_param, use_weights=self.use_weights, reduce_axis=1, fixed_support=self.fixed_support)
            # weights = jax.nn.softmax(log_weights, axis=0) * self.batch_size  # Scale weights to match batch size
        weights = log_weights
        chex.assert_shape(tasks, (self.batch_size, self.task_n_dims, 1))
        chex.assert_shape(weights, (self.batch_size, 1))
        # jax.debug.print("Weights sum: {}", jnp.sum(weights))
        # jax.debug.print("Batch statistics: tasks min {}, max {}, mean {}",
        #                jnp.min(tasks), jnp.max(tasks), jnp.mean(tasks))
        # Now: returns unnormalized log weights
        return tasks, weights

    @jax.jit
    def evaluate(self, tasks: Array, step: int) -> Array:
        chex.assert_shape(tasks, (self.batch_size, self.task_n_dims, 1))

        mu, theta = self.get_params_from_tasks(tasks)
        chex.assert_shape(mu, (self.batch_size, self.n_dims))
        chex.assert_shape(theta, (self.batch_size, self.n_dims))

        key = jax.random.fold_in(self.noise_key, step)
        all_noise = jax.random.normal(key, (self.n_points+1, self.batch_size, self.n_dims), self.dtype) * self.noise_scale

        init = all_noise[0, :, :]
        chex.assert_shape(init, (self.batch_size, self.n_dims))

        all_noise = all_noise[1:, :, :]
        chex.assert_shape(all_noise, (self.n_points, self.batch_size, self.n_dims))
        
        def ou_step(carry, noise):
            prev_state = carry
            chex.assert_shape(prev_state, (self.batch_size, self.n_dims))

            next_state = prev_state + theta * (mu - prev_state) * self.ou_step  + jnp.sqrt(self.ou_step) * noise
            chex.assert_shape(next_state, (self.batch_size, self.n_dims))

            return next_state, next_state
        # Run the OU process for n_points steps
        _, ou_steps = jax.lax.scan(ou_step, init, all_noise)
        chex.assert_shape(ou_steps, (self.n_points, self.batch_size, self.n_dims))

        ou_steps = jnp.transpose(ou_steps, (1, 0, 2))  # Shape: (batch_size, n_points, n_dims)
        chex.assert_shape(ou_steps, (self.batch_size, self.n_points, self.n_dims))

        return init, ou_steps

    @jax.jit
    def generate_attention_mask(self) -> Array:
        """Generate causal attention mask for the sequence with right padding.
        
        Creates a mask of size (n_max_points, n_max_points) where:
        - First n_points positions are valid (actual data) 
        - Remaining positions are padded and masked out (right padding)
        - Within valid positions, uses causal attention (can only attend to previous positions)
        """
        effective_seq_len = self.n_points      # Valid data: positions 0 to this-1
        max_seq_len = self.n_max_points        # Total padded length
        
        # Start with all positions masked (False)
        mask = jnp.zeros((max_seq_len, max_seq_len), dtype=bool)
        
        # Valid region gets causal attention pattern
        valid_mask = jnp.tril(jnp.ones((effective_seq_len, effective_seq_len))).astype(bool)
        
        # Insert valid causal mask into full mask 
        mask = mask.at[:effective_seq_len, :effective_seq_len].set(valid_mask)
        return mask

    def curriculum_increment(self):
        old_n_points = self.n_points
        self.n_points = min(self.n_points + self.curriculum_n_points_increment, 
                           self.n_max_points)
        if self.n_points > old_n_points:
            logging.info(f"Curriculum increment: n_points {old_n_points} -> {self.n_points}")

    def sample_batch(self, step: int, evl=False) -> tuple[Array, Array, Array, Array]:
        if step % self.curriculum_steps_thresh == self.curriculum_steps_thresh - 1 and self.use_curriculum:
            self.curriculum_increment()

        (tasks, weights) = self.sample_tasks(step)
        chex.assert_shape(tasks, (self.batch_size, self.task_n_dims, 1))
        chex.assert_shape(weights, (self.batch_size, 1))

        #print(f"Evl={evl}, Weights: {weights}")

        init, targets = self.evaluate(tasks, step)
        chex.assert_shape(init, (self.batch_size, self.n_dims))
        chex.assert_shape(targets, (self.batch_size, self.n_points, self.n_dims))

        data = jnp.concatenate((init[:, None, :], targets[:, :-1, :]), axis=1)
        chex.assert_shape(data, (self.batch_size, self.n_points, self.n_dims))

        attention_mask = self.generate_attention_mask()
        chex.assert_shape(attention_mask, (self.n_max_points, self.n_max_points))

        return data, tasks, weights, targets, attention_mask

    @jax.jit
    def evaluate_oracle(self, data: Array, tasks: Array, targets) -> Array:
        targets = data
        n_points = targets.shape[1]
        batch_size = targets.shape[0]
        chex.assert_shape(targets, (batch_size, n_points, self.n_dims))

        mu, theta = self.get_params_from_tasks(tasks)
        chex.assert_shape(mu, (batch_size, self.n_dims))
        chex.assert_shape(theta, (batch_size, self.n_dims))

        prev_states = data 
        chex.assert_shape(prev_states, (batch_size, n_points, self.n_dims))

        oracle_states = prev_states + theta[:, None, :] * (mu[:, None, :] - prev_states) * self.ou_step
        chex.assert_shape(oracle_states, (batch_size, n_points, self.n_dims))

        return oracle_states

    def get_default_eval_tasks(
            self, batch_size: int, task_seed: int, data_seed: int, noise_seed: int, eval_n_points: List[int], task_centers: List[float] | None = None, **kwargs
            ) -> list["OrnsteinUhlenbeckTask"]:
        del kwargs
        assert task_seed != self.task_seed
        assert data_seed != self.data_seed
        assert noise_seed != self.noise_seed
        config = dataclasses.asdict(self)
        config["batch_size"] = batch_size
        config["task_seed"] = task_seed
        config["data_seed"] = data_seed
        config["noise_seed"] = noise_seed
        config["n_tasks"] = 0
        config["n_data"] = 0
        config["n_max_points"] = self.n_max_points
        config["use_curriculum"] = False  # Disable curriculum for evaluation
        config["use_weights"] = False
        eval_tasks = []
        n_points = eval_n_points
        assert n_points <= self.n_max_points, f"n_points {n_points} exceeds n_max_points {self.n_max_points}"
        config["n_points"] = n_points
        # Increment seeds
        config["task_seed"] += 1
        config["data_seed"] += 1
        config["noise_seed"] += 1

        # Test  with fresh tasks from training distribution
        name = f"Test tasks"
        config["name"] = name
        eval_tasks.append(self.__class__(**config))

        # Test with same tasks as training distribution
        if self.n_tasks > 0:
            # Increment seeds
            config["task_seed"] += 1
            config["data_seed"] += 1
            config["noise_seed"] += 1

            name = f"Train tasks"
            config["n_tasks"] = self.n_tasks
            config["name"] = name
            eval_tasks.append(OrnsteinUhlenbeckTask.from_task_pool(**config, task_pool=self.task_pool.copy(), weights=self.weights.copy()))
        
        config["n_tasks"] = 0  # Reset for fresh tasks

        # Test with fixed task centers
        if task_centers is not None:
            config["distrib_name"] = "normal"  # Reset to normal distribution for fixed tasks
            config["fixed_support"] = False
            for task_center in task_centers:
                # Increment seeds
                config["task_seed"] += 1
                config["data_seed"] += 1
                config["noise_seed"] += 1

                config["task_center"] = task_center
                # config["task_scale"] = 0.
                config["clip"] = None
                name = f"Fixed task {task_center}"
                config["name"] = name
                eval_tasks.append(self.__class__(**config))
        return eval_tasks

    def get_default_eval_models(self) -> list[Model]:
        return [get_model(name="last_value"), get_model(name="arma", dtype=self.dtype)]

    def _tree_flatten(self):
        # Dynamic values (arrays, keys, and values that can change)
        children = (
            self.data_key,
            self.task_key, 
            self.noise_key,
            self.task_pool,
            self.weights,
            self.data_pool,
            self.data_scale,
            self.task_scale,
            self.noise_scale,
            self.task_center,
            self.clip,
        )
        
        # Static values (configuration that doesn't change during execution)
        aux_data = {
            'n_tasks': self.n_tasks,
            'n_data': self.n_data,
            'n_dims': self.n_dims,
            'n_points': self.n_points,
            'batch_size': self.batch_size,
            'data_seed': self.data_seed,
            'task_seed': self.task_seed,
            'noise_seed': self.noise_seed,
            'dtype': self.dtype,
            'n_max_points': self.n_max_points,
            'name': self.name,
            'eval_ridge': self.eval_ridge,
            'use_weights': self.use_weights,
            'use_weight_sampling': self.use_weight_sampling,
            'distrib_name': self.distrib_name,
            'distrib_param': self.distrib_param,
            'use_curriculum': self.use_curriculum,
            'curriculum_n_points_increment': self.curriculum_n_points_increment,
            'curriculum_steps_thresh': self.curriculum_steps_thresh,
            'ou_step': self.ou_step,
            'task_n_dims': self.task_n_dims,
            'fixed_support': self.fixed_support,
        }
        
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        (data_key, task_key, noise_key, task_pool, weights, data_pool,
         data_scale, task_scale, noise_scale, task_center, clip) = children
        
        # Create object with aux_data parameters and placeholder scale values
        obj = cls(data_scale=1.0, task_scale=1.0, noise_scale=1.0, 
                 task_center=0.0, clip=None, _skip_init=True, **aux_data)
        
        # Set the dynamic values
        obj.data_key = data_key
        obj.task_key = task_key
        obj.noise_key = noise_key
        obj.task_pool = task_pool
        obj.weights = weights
        obj.data_pool = data_pool
        obj.data_scale = data_scale
        obj.task_scale = task_scale
        obj.noise_scale = noise_scale
        obj.task_center = task_center
        obj.clip = clip
        
        return obj


# Register OrnsteinUhlenbeckTask as a PyTree
tree_util.register_pytree_node(OrnsteinUhlenbeckTask,
                               OrnsteinUhlenbeckTask._tree_flatten,
                               OrnsteinUhlenbeckTask._tree_unflatten)


########################################################################################################################
# MLP SDE Task for In-Context Learning
########################################################################################################################

@dataclasses.dataclass
class MLPSDETask:
    """
    MLP-driven SDE Task: dX_t = -b(X_t) dt + sigma dW_t
    where b(X_t) is a small MLP instead of linear function.
    
    **CRITICAL CHANGE from OU Task**: The drift function is now b(X_t) = MLP(X_t)
    instead of b(X_t) = theta * (mu - X_t).
    """
    n_tasks: int
    n_data: int
    n_dims: int
    n_points: int
    batch_size: int
    data_seed: int
    task_seed: int
    noise_seed: int
    data_scale: float
    task_scale: float
    noise_scale: float
    dtype: Any
    task_center: float | None = None
    n_max_points: int | None = None  # Optional, used for padding in some models
    clip: float | None = None  # Optional, clip task vectors to [-clip, clip]^d
    name: str | None = None  # Optional, can be set to override default name
    eval_ridge: bool = True  # Optional, whether to include Ridge baseline in evaluation
    use_weights: bool = False  # Optional, whether to use task importance weights
    use_weight_sampling: bool = False  # Whether to use weighted sampling for tasks
    distrib_name: str = "normal"  # Distribution name: "normal" or "student"
    distrib_param: float | None = None  # Distribution parameter (degrees of freedom for student-t)
    use_curriculum: bool = False  # Whether to use curriculum learning
    curriculum_n_points_increment: int = 2  # Increment for curriculum learning
    curriculum_steps_thresh: int = 2_000  # Steps after which to increment n_points in curriculum learning
    ou_step: float = 1e-2
    hidden_size: int | None = None  # MLP hidden size, defaults to 2*n_dims
    task_n_dims: int | None = None  # Automatically computed based on MLP parameters
    
    # Extended curriculum learning parameters - use defaults that match base parameters
    max_hidden_size: int = 0  # Maximum MLP hidden size (will be set to hidden_size if 0)
    curriculum_hidden_increment: int = 1  # How much to increase hidden size each step
    min_hidden_size: int = 1  # Starting hidden size for curriculum
    max_n_dims: int = 0  # Maximum state dimensionality (will be set to n_dims if 0)
    curriculum_dims_increment: int = 1  # How much to increase dimensions each step
    min_n_dims: int = 1  # Starting dimensions for curriculum
    current_hidden_size: int | None = None
    current_n_dims: int | None = None
    
    _skip_init: bool = False  # Private parameter to skip __post_init__ logic

    def __post_init__(self):
        if self._skip_init:
            return
        # Validation
        self.data_key = jax.random.PRNGKey(self.data_seed)
        self.task_key = jax.random.PRNGKey(self.task_seed)
        self.noise_key = jax.random.PRNGKey(self.noise_seed)
        
        # Set hidden size if not provided
        if self.hidden_size is None:
            self.hidden_size = 2 * self.n_dims
            
        # Set up extended curriculum parameters
        self.max_hidden_size = self.hidden_size if self.max_hidden_size == 0 else self.max_hidden_size
        self.max_n_dims = self.n_dims if self.max_n_dims == 0 else self.max_n_dims
        
        # Initialize current curriculum state
        self.current_hidden_size = self.min_hidden_size if self.use_curriculum else self.max_hidden_size
        self.current_n_dims = self.min_n_dims if self.use_curriculum else self.max_n_dims
        
        # **CRITICAL CHANGE**: Compute task_n_dims based on MAX MLP parameters for shape consistency
        # MLP has: W1 (max_n_dims x max_hidden_size), b1 (max_hidden_size), W2 (max_hidden_size x max_n_dims), b2 (max_n_dims)
        self.task_n_dims = self.max_n_dims * self.max_hidden_size + self.max_hidden_size + self.max_hidden_size * self.max_n_dims + self.max_n_dims
        
        self.n_max_points = self.n_points if self.n_max_points is None else self.n_max_points
        self.n_points = self.n_max_points if not self.use_curriculum else self.n_points
        self.task_center = 0.0 if self.task_center is None else self.task_center
        task_pool, weights = self.generate_task_pool() if self.n_tasks > 0 else (None, None)
        self.task_pool = task_pool
        self.weights = weights
        self.data_pool = None
        self.name = f"MLPSDE({self.n_tasks})" if self.name is None else self.name

    @classmethod
    def from_task_pool(cls, task_pool: Array, weights: Array, **kwargs):
        assert kwargs["n_tasks"] == task_pool.shape[0]
        task = cls(**kwargs)
        task.task_pool = task_pool
        task.weights = weights
        return task
    
    def get_params_from_tasks(self, tasks: Array) -> dict:
        """
        **CRITICAL CHANGE**: Extract MLP parameters instead of mu/theta.
        
        Args:
            tasks: Task parameters of shape (batch_size, task_n_dims, 1)
        
        Returns:
            dict with MLP parameters: {'W1', 'b1', 'W2', 'b2'}
        """
        batch_size_actual, task_n_dims_actual, one_dim = tasks.shape
        chex.assert_shape(tasks, (batch_size_actual, task_n_dims_actual, one_dim))
        chex.assert_equal(task_n_dims_actual, self.task_n_dims)
        chex.assert_equal(one_dim, 1)
        
        # Flatten task parameters
        params = tasks[:, :, 0]  # Shape: (batch_size, task_n_dims)
        chex.assert_shape(params, (batch_size_actual, self.task_n_dims))
        
        # Extract MLP parameters using MAX dimensions for shape consistency
        idx = 0
        
        # W1: (batch_size, max_n_dims, max_hidden_size)
        W1_size = self.max_n_dims * self.max_hidden_size
        W1 = params[:, idx:idx+W1_size].reshape(batch_size_actual, self.max_n_dims, self.max_hidden_size)
        idx += W1_size
        chex.assert_shape(W1, (batch_size_actual, self.max_n_dims, self.max_hidden_size))
        
        # b1: (batch_size, max_hidden_size)  
        b1 = params[:, idx:idx+self.max_hidden_size]
        idx += self.max_hidden_size
        chex.assert_shape(b1, (batch_size_actual, self.max_hidden_size))
        
        # W2: (batch_size, max_hidden_size, max_n_dims)
        W2_size = self.max_hidden_size * self.max_n_dims
        W2 = params[:, idx:idx+W2_size].reshape(batch_size_actual, self.max_hidden_size, self.max_n_dims)
        idx += W2_size
        chex.assert_shape(W2, (batch_size_actual, self.max_hidden_size, self.max_n_dims))
        
        # b2: (batch_size, max_n_dims)
        b2 = params[:, idx:idx+self.max_n_dims]
        chex.assert_shape(b2, (batch_size_actual, self.max_n_dims))
        
        # Verify we consumed all parameters
        chex.assert_equal(idx + self.max_n_dims, self.task_n_dims)
        
        return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    def apply_mlp_drift(self, x: Array, mlp_params: dict) -> Array:
        """
        **CRITICAL CHANGE**: Apply MLP drift function b(x) with curriculum masking.
        
        Args:
            x: State of shape (batch_size, n_points, current_n_dims) - ALWAYS has n_points dimension
            mlp_params: Dictionary with MLP parameters (using max dimensions)
        
        Returns:
            drift: Drift vector of same shape as x
        """
        batch_size_actual, n_points_actual, n_dims_actual = x.shape
        chex.assert_shape(x, (batch_size_actual, n_points_actual, n_dims_actual))
        assert n_dims_actual <= self.max_n_dims, f"Input n_dims {n_dims_actual} exceeds max_n_dims {self.max_n_dims}"
        
        # MLP parameters should match batch size and MAX dimensions
        chex.assert_shape(mlp_params['W1'], (batch_size_actual, self.max_n_dims, self.max_hidden_size))
        chex.assert_shape(mlp_params['b1'], (batch_size_actual, self.max_hidden_size))
        chex.assert_shape(mlp_params['W2'], (batch_size_actual, self.max_hidden_size, self.max_n_dims))
        chex.assert_shape(mlp_params['b2'], (batch_size_actual, self.max_n_dims))
        
        def single_point_mlp_with_masking(x_point, W1, b1, W2, b2):
            """Apply MLP to a single point with curriculum masking."""
            chex.assert_shape(x_point, (n_dims_actual,))
            chex.assert_shape(W1, (self.max_n_dims, self.max_hidden_size))
            chex.assert_shape(b1, (self.max_hidden_size,))
            chex.assert_shape(W2, (self.max_hidden_size, self.max_n_dims))
            chex.assert_shape(b2, (self.max_n_dims,))
            
            # Pad input to max dimensions (zero-pad extra dimensions)
            x_padded = jnp.concatenate([x_point, jnp.zeros(self.max_n_dims - n_dims_actual, dtype=x_point.dtype)])
            chex.assert_shape(x_padded, (self.max_n_dims,))
            
            # Forward pass: x -> h -> drift with masking
            h_full = jax.nn.tanh(x_padded @ W1 + b1)  # (max_hidden_size,)
            chex.assert_shape(h_full, (self.max_hidden_size,))
            
            # Apply hidden dimension mask
            hidden_mask = jnp.concatenate([
                jnp.ones(self.current_hidden_size, dtype=h_full.dtype),
                jnp.zeros(self.max_hidden_size - self.current_hidden_size, dtype=h_full.dtype)
            ])
            h_masked = h_full * hidden_mask
            chex.assert_shape(h_masked, (self.max_hidden_size,))
            
            # Second layer
            drift_full = h_masked @ W2 + b2  # (max_n_dims,)
            chex.assert_shape(drift_full, (self.max_n_dims,))

            # Normalize drift to prevent explosion
            drift_full = jnp.clip(drift_full, -1.0, 1.0)  # Clip to prevent extreme drift values
            drift_full = drift_full + 0.1 * x_padded  # Add small linear term for stability
            
            # Apply output dimension mask and truncate to current dimensions
            drift_current = drift_full[:n_dims_actual]  # (n_dims_actual,)
            chex.assert_shape(drift_current, (n_dims_actual,))
            
            return drift_current
        
        # Use vmap to handle n_points dimension: vmap over both points and batch
        # First vmap over n_points, then over batch
        batched_mlp = jax.vmap(jax.vmap(single_point_mlp_with_masking, in_axes=(0, None, None, None, None)), 
                              in_axes=(0, 0, 0, 0, 0))
        
        drift = batched_mlp(x, mlp_params['W1'], mlp_params['b1'], mlp_params['W2'], mlp_params['b2'])
        
        chex.assert_shape(drift, (batch_size_actual, n_points_actual, n_dims_actual))
        return drift

    def generate_task_pool(self) -> Array:
        chex.assert_scalar_positive(self.n_tasks)  # Should be positive since we're generating a pool
        
        key = jax.random.fold_in(self.task_key, 0)
        shape = self.n_tasks, self.task_n_dims, 1
        tasks = sample_distrib(key, self.task_center, self.task_scale, self.clip, 
                              self.distrib_name, self.distrib_param, shape, self.dtype)

        # Assert generated task pool shape
        chex.assert_shape(tasks, (self.n_tasks, self.task_n_dims, 1))

        log_weights = task_log_weights(tasks, self.task_center, self.task_scale, self.clip, 
                                     self.distrib_name, self.distrib_param, use_weights=self.use_weights, reduce_axis=1)
        weights = log_weights
        
        # Assert weights shape
        chex.assert_shape(weights, (self.n_tasks, 1))
        
        return tasks, weights

    def generate_data_pool(self) -> Array:
        chex.assert_scalar_positive(self.n_data)  # Should be positive since we're generating a pool
        
        key = jax.random.fold_in(self.data_key, 0)
        shape = self.n_data, self.n_points, self.n_dims
        data = jax.random.normal(key, shape, self.dtype) * self.data_scale
        
        # Assert generated data pool shape
        chex.assert_shape(data, (self.n_data, self.n_points, self.n_dims))
        
        return data

    @jax.jit
    def sample_data(self, step: int) -> Array:
        key = jax.random.fold_in(self.data_key, step)
        if self.n_data > 0:
            idxs = jax.random.choice(key, self.n_data, (self.batch_size,))
            chex.assert_shape(idxs, (self.batch_size,))
            data = self.data_pool[idxs]
            # data_pool has shape (n_data, n_points, n_dims), so indexed data should be:
            chex.assert_shape(data, (self.batch_size, self.n_points, self.n_dims))
        else:
            shape = self.batch_size, self.n_points, self.n_dims
            data = jax.random.normal(key, shape, self.dtype) * self.data_scale + self.task_center
            chex.assert_shape(data, (self.batch_size, self.n_points, self.n_dims))
        
        # Final assertion on returned data
        batch_size_actual, n_points_actual, n_dims_actual = data.shape
        chex.assert_shape(data, (batch_size_actual, n_points_actual, n_dims_actual))
        chex.assert_equal(batch_size_actual, self.batch_size)
        chex.assert_equal(n_points_actual, self.n_points) 
        chex.assert_equal(n_dims_actual, self.n_dims)
        
        return data

    @jax.jit
    def sample_tasks(self, step: int) -> Array:
        key = jax.random.fold_in(self.task_key, step)
        if self.n_tasks > 0:
            if self.use_weight_sampling:
                idxs = jax.random.categorical(key, self.weights, axis=0, shape=(self.batch_size,))
                chex.assert_shape(idxs, (self.batch_size,))
                log_weights = jnp.zeros((self.batch_size, 1), self.dtype)
                chex.assert_shape(log_weights, (self.batch_size, 1))
                tasks = self.task_pool[idxs]
                # task_pool has shape (n_tasks, task_n_dims, 1), so indexed tasks should be:
                chex.assert_shape(tasks, (self.batch_size, self.task_n_dims, 1))
            else:
                idxs = jax.random.choice(key, self.n_tasks, (self.batch_size,))
                chex.assert_shape(idxs, (self.batch_size,))
                tasks = self.task_pool[idxs]
                chex.assert_shape(tasks, (self.batch_size, self.task_n_dims, 1))
                log_weights = task_log_weights(tasks, self.task_center, self.task_scale, self.clip, 
                                             self.distrib_name, self.distrib_param, use_weights=self.use_weights, reduce_axis=1)
                chex.assert_shape(log_weights, (self.batch_size, 1))
        else:
            shape = self.batch_size, self.task_n_dims, 1
            tasks = sample_distrib(key, self.task_center, self.task_scale, self.clip, 
                                 self.distrib_name, self.distrib_param, shape, self.dtype)
            chex.assert_shape(tasks, (self.batch_size, self.task_n_dims, 1))
            log_weights = task_log_weights(tasks, self.task_center, self.task_scale, self.clip, 
                                         self.distrib_name, self.distrib_param, use_weights=self.use_weights, reduce_axis=1)
            chex.assert_shape(log_weights, (self.batch_size, 1))
        
        weights = log_weights
        
        # Final assertions on returned values  
        batch_size_actual, task_n_dims_actual, one_dim = tasks.shape
        chex.assert_shape(tasks, (batch_size_actual, task_n_dims_actual, one_dim))
        chex.assert_equal(batch_size_actual, self.batch_size)
        chex.assert_equal(task_n_dims_actual, self.task_n_dims)
        chex.assert_equal(one_dim, 1)
        
        batch_size_w, one_dim_w = weights.shape
        chex.assert_shape(weights, (batch_size_w, one_dim_w))
        chex.assert_equal(batch_size_w, self.batch_size)
        chex.assert_equal(one_dim_w, 1)
        
        return tasks, weights

    @jax.jit
    def evaluate(self, tasks: Array, step: int) -> Array:
        """
        **CRITICAL CHANGE**: Use MLP drift instead of linear OU drift.
        """
        batch_size_actual, task_n_dims_actual, one_dim = tasks.shape
        chex.assert_shape(tasks, (batch_size_actual, task_n_dims_actual, one_dim))
        chex.assert_equal(batch_size_actual, self.batch_size)
        chex.assert_equal(task_n_dims_actual, self.task_n_dims)
        chex.assert_equal(one_dim, 1)

        # Extract MLP parameters instead of mu/theta
        mlp_params = self.get_params_from_tasks(tasks)

        key = jax.random.fold_in(self.noise_key, step)
        # Generate noise using CURRENT dimensions (curriculum masking)
        all_noise = jax.random.normal(key, (self.n_points+1, self.batch_size, self.current_n_dims), self.dtype) * self.noise_scale
        chex.assert_shape(all_noise, (self.n_points+1, self.batch_size, self.current_n_dims))

        init = all_noise[0, :, :]
        chex.assert_shape(init, (self.batch_size, self.current_n_dims))

        all_noise = all_noise[1:, :, :]
        chex.assert_shape(all_noise, (self.n_points, self.batch_size, self.current_n_dims))
        
        def mlp_sde_step(carry, noise):
            """**CRITICAL CHANGE**: Use MLP drift with curriculum dimension masking."""
            prev_state = carry
            chex.assert_shape(prev_state, (self.batch_size, self.current_n_dims))
            chex.assert_shape(noise, (self.batch_size, self.current_n_dims))

            # MLP drift: -b(X_t) where b is MLP with curriculum masking
            # apply_mlp_drift expects 3D input (batch_size, n_points, current_n_dims)
            # So we add a singleton n_points dimension
            prev_state_3d = prev_state[:, None, :]  # (batch_size, 1, current_n_dims)
            chex.assert_shape(prev_state_3d, (self.batch_size, 1, self.current_n_dims))
            
            drift_3d = self.apply_mlp_drift(prev_state_3d, mlp_params)
            chex.assert_shape(drift_3d, (self.batch_size, 1, self.current_n_dims))
            
            # Remove singleton dimension
            drift = drift_3d[:, 0, :]  # (batch_size, current_n_dims)
            chex.assert_shape(drift, (self.batch_size, self.current_n_dims))
            
            next_state = prev_state - drift * self.ou_step + jnp.sqrt(self.ou_step) * noise
            chex.assert_shape(next_state, (self.batch_size, self.current_n_dims))

            return next_state, next_state
            
        # Run the MLP SDE for n_points steps
        _, sde_steps = jax.lax.scan(mlp_sde_step, init, all_noise)
        chex.assert_shape(sde_steps, (self.n_points, self.batch_size, self.current_n_dims))

        sde_steps = jnp.transpose(sde_steps, (1, 0, 2))  # Shape: (batch_size, n_points, current_n_dims)
        chex.assert_shape(sde_steps, (self.batch_size, self.n_points, self.current_n_dims))

        # Final assertions on return values using current curriculum dimensions
        init_bs, init_dims = init.shape
        chex.assert_shape(init, (init_bs, init_dims))
        chex.assert_equal(init_bs, self.batch_size)
        chex.assert_equal(init_dims, self.current_n_dims)
        
        sde_bs, sde_points, sde_dims = sde_steps.shape
        chex.assert_shape(sde_steps, (sde_bs, sde_points, sde_dims))
        chex.assert_equal(sde_bs, self.batch_size)
        chex.assert_equal(sde_points, self.n_points)
        chex.assert_equal(sde_dims, self.current_n_dims)

        new_init = jnp.concatenate([init, jnp.zeros((self.batch_size, self.max_n_dims - self.current_n_dims), dtype=init.dtype)], axis=1)
        chex.assert_shape(new_init, (self.batch_size, self.max_n_dims))

        new_sde_steps = jnp.concatenate([sde_steps, jnp.zeros((self.batch_size, self.n_points, self.max_n_dims - self.current_n_dims), dtype=sde_steps.dtype)], axis=2)
        chex.assert_shape(new_sde_steps, (self.batch_size, self.n_points, self.max_n_dims))

        return new_init, new_sde_steps

    @jax.jit
    def generate_attention_mask(self) -> Array:
        """Generate causal attention mask for the sequence with right padding."""
        effective_seq_len = self.n_points      # Valid data: positions 0 to this-1
        max_seq_len = self.n_max_points        # Total padded length
        
        chex.assert_scalar_non_negative(effective_seq_len)
        chex.assert_scalar_positive(max_seq_len)
        assert effective_seq_len <= max_seq_len, f"effective_seq_len {effective_seq_len} > max_seq_len {max_seq_len}"
        
        # Start with all positions masked (False)
        mask = jnp.zeros((max_seq_len, max_seq_len), dtype=bool)
        chex.assert_shape(mask, (max_seq_len, max_seq_len))
        
        # Valid region gets causal attention pattern
        valid_mask = jnp.tril(jnp.ones((effective_seq_len, effective_seq_len))).astype(bool)
        chex.assert_shape(valid_mask, (effective_seq_len, effective_seq_len))
        
        # Insert valid causal mask into full mask 
        mask = mask.at[:effective_seq_len, :effective_seq_len].set(valid_mask)
        
        # Final assertion on return value
        chex.assert_shape(mask, (self.n_max_points, self.n_max_points))
        
        return mask

    def curriculum_increment(self):
        """Enhanced curriculum learning that can increment multiple dimensions."""
        changes = []
        
        # Increment n_points (original curriculum)
        old_n_points = self.n_points
        self.n_points = min(self.n_points + self.curriculum_n_points_increment, 
                           self.n_max_points)
        if self.n_points > old_n_points:
            changes.append(f"n_points {old_n_points} -> {self.n_points}")
        
        # Increment hidden size curriculum
        old_hidden_size = self.current_hidden_size
        self.current_hidden_size = min(self.current_hidden_size + self.curriculum_hidden_increment,
                                     self.max_hidden_size)
        if self.current_hidden_size > old_hidden_size:
            changes.append(f"hidden_size {old_hidden_size} -> {self.current_hidden_size}")
        
        # Increment dimension curriculum  
        old_n_dims = self.current_n_dims
        self.current_n_dims = min(self.current_n_dims + self.curriculum_dims_increment,
                                self.max_n_dims)
        if self.current_n_dims > old_n_dims:
            changes.append(f"n_dims {old_n_dims} -> {self.current_n_dims}")
        
        # Log all changes
        if changes:
            logging.info(f"Curriculum increment: {', '.join(changes)}")

    def sample_batch(self, step: int, evl=False) -> tuple[Array, Array, Array, Array]:
        if step % self.curriculum_steps_thresh == self.curriculum_steps_thresh - 1 and self.use_curriculum:
            self.curriculum_increment()

        (tasks, weights) = self.sample_tasks(step)
        chex.assert_shape(tasks, (self.batch_size, self.task_n_dims, 1))
        chex.assert_shape(weights, (self.batch_size, 1))

        init, targets = self.evaluate(tasks, step)
        chex.assert_shape(init, (self.batch_size, self.max_n_dims))
        chex.assert_shape(targets, (self.batch_size, self.n_points, self.max_n_dims))

        data = jnp.concatenate((init[:, None, :], targets[:, :-1, :]), axis=1)
        chex.assert_shape(data, (self.batch_size, self.n_points, self.max_n_dims))

        attention_mask = self.generate_attention_mask()
        chex.assert_shape(attention_mask, (self.n_max_points, self.n_max_points))

        return data, tasks, weights, targets, attention_mask

    @jax.jit
    def evaluate_oracle(self, data: Array, tasks: Array, targets) -> Array:
        """Oracle prediction using MLP drift."""
        # Identify actual dimensions from input (should match current curriculum dimensions)
        batch_size_actual, n_points_actual, n_dims_actual = data.shape
        chex.assert_shape(data, (batch_size_actual, n_points_actual, n_dims_actual))
        chex.assert_equal(n_dims_actual, self.max_n_dims)  # Data should always be padded to max_n_dims
        
        task_bs_actual, task_n_dims_actual, one_dim = tasks.shape
        chex.assert_shape(tasks, (task_bs_actual, task_n_dims_actual, one_dim))
        chex.assert_equal(one_dim, 1)

        mlp_params = self.get_params_from_tasks(tasks)
        prev_states = data 
        chex.assert_shape(prev_states, (batch_size_actual, n_points_actual, n_dims_actual))

        # Oracle: apply MLP drift with curriculum masking
        drift = self.apply_mlp_drift(prev_states, mlp_params)
        chex.assert_shape(drift, (batch_size_actual, n_points_actual, n_dims_actual))
        
        oracle_states = prev_states - drift * self.ou_step
        chex.assert_shape(oracle_states, (batch_size_actual, n_points_actual, n_dims_actual))

        # Final assertion on return value
        oracle_bs, oracle_points, oracle_dims = oracle_states.shape
        chex.assert_shape(oracle_states, (oracle_bs, oracle_points, oracle_dims))
        chex.assert_equal(oracle_bs, batch_size_actual)
        chex.assert_equal(oracle_points, n_points_actual)
        chex.assert_equal(oracle_dims, n_dims_actual)

        return oracle_states

    def get_default_eval_tasks(
            self, batch_size: int, task_seed: int, data_seed: int, noise_seed: int, eval_n_points: List[int], task_centers: List[float] | None = None, **kwargs
            ) -> list["MLPSDETask"]:
        del kwargs
        assert task_seed != self.task_seed
        assert data_seed != self.data_seed
        assert noise_seed != self.noise_seed
        config = dataclasses.asdict(self)
        config["batch_size"] = batch_size
        config["task_seed"] = task_seed
        config["data_seed"] = data_seed
        config["noise_seed"] = noise_seed
        config["n_tasks"] = 0
        config["n_data"] = 0
        config["n_max_points"] = self.n_max_points
        config["use_curriculum"] = False  # Disable curriculum for evaluation
        config["use_weights"] = False
        eval_tasks = []
        n_points = eval_n_points
        assert n_points <= self.n_max_points, f"n_points {n_points} exceeds n_max_points {self.n_max_points}"
        config["n_points"] = n_points
        # Increment seeds
        config["task_seed"] += 1
        config["data_seed"] += 1
        config["noise_seed"] += 1

        # Test  with fresh tasks from training distribution
        name = f"Test tasks"
        config["name"] = name
        eval_tasks.append(self.__class__(**config))

        # Test with same tasks as training distribution
        if self.n_tasks > 0:
            # Increment seeds
            config["task_seed"] += 1
            config["data_seed"] += 1
            config["noise_seed"] += 1

            name = f"Train tasks"
            config["n_tasks"] = self.n_tasks
            config["name"] = name
            eval_tasks.append(MLPSDETask.from_task_pool(**config, task_pool=self.task_pool.copy(), weights=self.weights.copy()))
        
        config["n_tasks"] = 0  # Reset for fresh tasks

        # Test with fixed task centers
        if task_centers is not None:
            config["distrib_name"] = "normal"  # Reset to normal distribution for fixed tasks
            for task_center in task_centers:
                # Increment seeds
                config["task_seed"] += 1
                config["data_seed"] += 1
                config["noise_seed"] += 1

                config["task_center"] = task_center
                config["clip"] = None
                name = f"Fixed task {task_center}"
                config["name"] = name
                eval_tasks.append(self.__class__(**config))
        return eval_tasks

    def get_default_eval_models(self) -> list[Model]:
        return [get_model(name="last_value"), get_model(name="arma", dtype=self.dtype), get_model(name="corrected_last_value")]

    def _tree_flatten(self):
        # Dynamic values (arrays, keys, and values that can change)
        children = (
            self.data_key,
            self.task_key, 
            self.noise_key,
            self.task_pool,
            self.weights,
            self.data_pool,
            self.data_scale,
            self.task_scale,
            self.noise_scale,
            self.task_center,
            self.clip,
        )
        
        # Static values (configuration that doesn't change during execution)
        aux_data = {
            'n_tasks': self.n_tasks,
            'n_data': self.n_data,
            'n_dims': self.n_dims,
            'n_points': self.n_points,
            'batch_size': self.batch_size,
            'data_seed': self.data_seed,
            'task_seed': self.task_seed,
            'noise_seed': self.noise_seed,
            'dtype': self.dtype,
            'n_max_points': self.n_max_points,
            'name': self.name,
            'eval_ridge': self.eval_ridge,
            'use_weights': self.use_weights,
            'use_weight_sampling': self.use_weight_sampling,
            'distrib_name': self.distrib_name,
            'distrib_param': self.distrib_param,
            'use_curriculum': self.use_curriculum,
            'curriculum_n_points_increment': self.curriculum_n_points_increment,
            'curriculum_steps_thresh': self.curriculum_steps_thresh,
            'ou_step': self.ou_step,
            'hidden_size': self.hidden_size,
            'task_n_dims': self.task_n_dims,
            'max_hidden_size': self.max_hidden_size,
            'curriculum_hidden_increment': self.curriculum_hidden_increment,
            'min_hidden_size': self.min_hidden_size,
            'max_n_dims': self.max_n_dims,
            'curriculum_dims_increment': self.curriculum_dims_increment,
            'min_n_dims': self.min_n_dims,
            'current_hidden_size': self.current_hidden_size,
            'current_n_dims': self.current_n_dims,
        }
        
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        (data_key, task_key, noise_key, task_pool, weights, data_pool,
         data_scale, task_scale, noise_scale, task_center, clip) = children
        
        # Create object with aux_data parameters and placeholder scale values
        obj = cls(data_scale=1.0, task_scale=1.0, noise_scale=1.0, 
                 task_center=0.0, clip=None, _skip_init=True, **aux_data)
        
        # Set the dynamic values
        obj.data_key = data_key
        obj.task_key = task_key
        obj.noise_key = noise_key
        obj.task_pool = task_pool
        obj.weights = weights
        obj.data_pool = data_pool
        obj.data_scale = data_scale
        obj.task_scale = task_scale
        obj.noise_scale = noise_scale
        obj.task_center = task_center
        obj.clip = clip
        
        return obj


# Register MLPSDETask as a PyTree
tree_util.register_pytree_node(MLPSDETask,
                               MLPSDETask._tree_flatten,
                               MLPSDETask._tree_unflatten)


@dataclasses.dataclass
class VolterraTask:
    """
    X_t = sum_{s < t} G(t-s) (b(X_s) + sigma Z_s)
    where Z_s truncated normal noise, and
    where b(X_t) is a small MLP instead of linear function.
    """
    n_tasks: int
    n_data: int
    n_dims: int
    n_points: int
    batch_size: int
    data_seed: int
    task_seed: int
    noise_seed: int
    data_scale: float
    task_scale: float
    noise_scale: float
    dtype: Any
    task_center: float | None = None
    n_max_points: int | None = None  # Optional, used for padding in some models
    clip: float | None = None  # Optional, clip task vectors to [-clip, clip]^d
    name: str | None = None  # Optional, can be set to override default name
    eval_ridge: bool = True  # Optional, whether to include Ridge baseline in evaluation
    use_weights: bool = False  # Optional, whether to use task importance weights
    use_weight_sampling: bool = False  # Whether to use weighted sampling for tasks
    distrib_name: str = "normal"  # Distribution name: "normal" or "student"
    distrib_param: float | None = None  # Distribution parameter (degrees of freedom for student-t)
    use_curriculum: bool = False  # Whether to use curriculum learning
    curriculum_n_points_increment: int = 2  # Increment for curriculum learning
    curriculum_steps_thresh: int = 2_000  # Steps after which to increment n_points in curriculum learning
    ou_step: float = 1e-2
    hidden_size: int | None = None  # MLP hidden size, defaults to 2*n_dims
    task_n_dims: int | None = None  # Automatically computed based on MLP parameters
    data_noise_trunc_radius: float = 10  # Truncation radius Gaussian noise for Volterra SDE
    kernel_exponent: float = 1.
    inner_steps:int = 10  # Number of inner steps for Volterra SDE
    drift_scale: float = 10.0
    drift_clip: float = 1.5
    drift_reg: float = 0.1  # Reference value for drift normalization

    
    # Extended curriculum learning parameters - use defaults that match base parameters
    max_hidden_size: int = 0  # Maximum MLP hidden size (will be set to hidden_size if 0)
    curriculum_hidden_increment: int = 1  # How much to increase hidden size each step
    min_hidden_size: int = 1  # Starting hidden size for curriculum
    max_n_dims: int = 0  # Maximum state dimensionality (will be set to n_dims if 0)
    curriculum_dims_increment: int = 1  # How much to increase dimensions each step
    min_n_dims: int = 1  # Starting dimensions for curriculum
    current_hidden_size: int | None = None
    current_n_dims: int | None = None
    
    _skip_init: bool = False  # Private parameter to skip __post_init__ logic

    def __post_init__(self):
        if self._skip_init:
            return
        # Validation
        self.data_key = jax.random.PRNGKey(self.data_seed)
        self.task_key = jax.random.PRNGKey(self.task_seed)
        self.noise_key = jax.random.PRNGKey(self.noise_seed)
        
        # Set hidden size if not provided
        if self.hidden_size is None:
            self.hidden_size = 2 * self.n_dims
            
        # Set up extended curriculum parameters
        self.max_hidden_size = self.hidden_size if self.max_hidden_size == 0 else self.max_hidden_size
        self.max_n_dims = self.n_dims if self.max_n_dims == 0 else self.max_n_dims
        
        # Initialize current curriculum state
        self.current_hidden_size = self.min_hidden_size if self.use_curriculum else self.max_hidden_size
        self.current_n_dims = self.min_n_dims if self.use_curriculum else self.max_n_dims
        
        # **CRITICAL CHANGE**: Compute task_n_dims based on MAX MLP parameters for shape consistency
        # MLP has: W1 (max_n_dims x max_hidden_size), b1 (max_hidden_size), W2 (max_hidden_size x max_n_dims), b2 (max_n_dims)
        self.task_n_dims = self.max_n_dims * self.max_hidden_size + self.max_hidden_size + self.max_hidden_size * self.max_n_dims + self.max_n_dims
        
        self.n_max_points = self.n_points if self.n_max_points is None else self.n_max_points
        self.n_points = self.n_max_points if not self.use_curriculum else self.n_points
        self.task_center = 0.0 if self.task_center is None else self.task_center
        task_pool, weights = self.generate_task_pool() if self.n_tasks > 0 else (None, None)
        self.task_pool = task_pool
        self.weights = weights
        self.data_pool = None
        self.name = f"Volterra({self.n_tasks})" if self.name is None else self.name

    @classmethod
    def from_task_pool(cls, task_pool: Array, weights: Array, **kwargs):
        assert kwargs["n_tasks"] == task_pool.shape[0]
        task = cls(**kwargs)
        task.task_pool = task_pool
        task.weights = weights
        return task
    
    def get_params_from_tasks(self, tasks: Array) -> dict:
        """
        **CRITICAL CHANGE**: Extract MLP parameters instead of mu/theta.
        
        Args:
            tasks: Task parameters of shape (batch_size, task_n_dims, 1)
        
        Returns:
            dict with MLP parameters: {'W1', 'b1', 'W2', 'b2'}
        """
        batch_size_actual, task_n_dims_actual, one_dim = tasks.shape
        chex.assert_shape(tasks, (batch_size_actual, task_n_dims_actual, one_dim))
        chex.assert_equal(task_n_dims_actual, self.task_n_dims)
        chex.assert_equal(one_dim, 1)
        
        # Flatten task parameters
        params = tasks[:, :, 0]  # Shape: (batch_size, task_n_dims)
        chex.assert_shape(params, (batch_size_actual, self.task_n_dims))
        
        # Extract MLP parameters using MAX dimensions for shape consistency
        idx = 0
        
        # W1: (batch_size, max_n_dims, max_hidden_size)
        W1_size = self.max_n_dims * self.max_hidden_size
        W1 = params[:, idx:idx+W1_size].reshape(batch_size_actual, self.max_n_dims, self.max_hidden_size)
        idx += W1_size
        chex.assert_shape(W1, (batch_size_actual, self.max_n_dims, self.max_hidden_size))
        
        # b1: (batch_size, max_hidden_size)  
        b1 = params[:, idx:idx+self.max_hidden_size]
        idx += self.max_hidden_size
        chex.assert_shape(b1, (batch_size_actual, self.max_hidden_size))
        
        # W2: (batch_size, max_hidden_size, max_n_dims)
        W2_size = self.max_hidden_size * self.max_n_dims
        W2 = params[:, idx:idx+W2_size].reshape(batch_size_actual, self.max_hidden_size, self.max_n_dims)
        idx += W2_size
        chex.assert_shape(W2, (batch_size_actual, self.max_hidden_size, self.max_n_dims))
        
        # b2: (batch_size, max_n_dims)
        b2 = params[:, idx:idx+self.max_n_dims]
        chex.assert_shape(b2, (batch_size_actual, self.max_n_dims))
        
        # Verify we consumed all parameters
        chex.assert_equal(idx + self.max_n_dims, self.task_n_dims)
        
        return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    def apply_mlp_drift(self, x: Array, mlp_params: dict) -> Array:
        """
        **CRITICAL CHANGE**: Apply MLP drift function b(x) with curriculum masking.
        
        Args:
            x: State of shape (batch_size, n_points, current_n_dims) - ALWAYS has n_points dimension
            mlp_params: Dictionary with MLP parameters (using max dimensions)
        
        Returns:
            drift: Drift vector of same shape as x
        """
        batch_size_actual, n_points_actual, n_dims_actual = x.shape
        chex.assert_shape(x, (batch_size_actual, n_points_actual, n_dims_actual))
        assert n_dims_actual <= self.max_n_dims, f"Input n_dims {n_dims_actual} exceeds max_n_dims {self.max_n_dims}"
        
        # MLP parameters should match batch size and MAX dimensions
        chex.assert_shape(mlp_params['W1'], (batch_size_actual, self.max_n_dims, self.max_hidden_size))
        chex.assert_shape(mlp_params['b1'], (batch_size_actual, self.max_hidden_size))
        chex.assert_shape(mlp_params['W2'], (batch_size_actual, self.max_hidden_size, self.max_n_dims))
        chex.assert_shape(mlp_params['b2'], (batch_size_actual, self.max_n_dims))
        
        def single_point_mlp_with_masking(x_point, W1, b1, W2, b2):
            """Apply MLP to a single point with curriculum masking."""
            chex.assert_shape(x_point, (n_dims_actual,))
            chex.assert_shape(W1, (self.max_n_dims, self.max_hidden_size))
            chex.assert_shape(b1, (self.max_hidden_size,))
            chex.assert_shape(W2, (self.max_hidden_size, self.max_n_dims))
            chex.assert_shape(b2, (self.max_n_dims,))
            
            # Pad input to max dimensions (zero-pad extra dimensions)
            x_padded = jnp.concatenate([x_point, jnp.zeros(self.max_n_dims - n_dims_actual, dtype=x_point.dtype)])
            chex.assert_shape(x_padded, (self.max_n_dims,))
            
            # Forward pass: x -> h -> drift with masking
            h_full = jax.nn.tanh(x_padded @ W1 + b1)  # (max_hidden_size,)
            chex.assert_shape(h_full, (self.max_hidden_size,))
            
            # Apply hidden dimension mask
            hidden_mask = jnp.concatenate([
                jnp.ones(self.current_hidden_size, dtype=h_full.dtype),
                jnp.zeros(self.max_hidden_size - self.current_hidden_size, dtype=h_full.dtype)
            ])
            h_masked = h_full * hidden_mask
            chex.assert_shape(h_masked, (self.max_hidden_size,))
            
            # Second layer
            drift_full = h_masked @ W2 + b2  # (max_n_dims,)
            chex.assert_shape(drift_full, (self.max_n_dims,))

            # Normalize drift to prevent explosion
            drift_full = self.drift_scale * drift_full
            clip_thresh = self.drift_clip
            drift_full = jnp.clip(drift_full, -clip_thresh,clip_thresh) 
            drift_full = drift_full + self.drift_reg * x_padded  # Add small linear term for stability
            
            # Apply output dimension mask and truncate to current dimensions
            drift_current = drift_full[:n_dims_actual]  # (n_dims_actual,)
            chex.assert_shape(drift_current, (n_dims_actual,))
            
            return drift_current
        
        # Use vmap to handle n_points dimension: vmap over both points and batch
        # First vmap over n_points, then over batch
        batched_mlp = jax.vmap(jax.vmap(single_point_mlp_with_masking, in_axes=(0, None, None, None, None)), 
                              in_axes=(0, 0, 0, 0, 0))
        
        drift = batched_mlp(x, mlp_params['W1'], mlp_params['b1'], mlp_params['W2'], mlp_params['b2'])
        
        chex.assert_shape(drift, (batch_size_actual, n_points_actual, n_dims_actual))
        return drift

    def generate_task_pool(self) -> Array:
        chex.assert_scalar_positive(self.n_tasks)  # Should be positive since we're generating a pool
        
        key = jax.random.fold_in(self.task_key, 0)
        shape = self.n_tasks, self.task_n_dims, 1
        tasks = sample_distrib(key, self.task_center, self.task_scale, self.clip, 
                              self.distrib_name, self.distrib_param, shape, self.dtype)

        # Assert generated task pool shape
        chex.assert_shape(tasks, (self.n_tasks, self.task_n_dims, 1))

        log_weights = task_log_weights(tasks, self.task_center, self.task_scale, self.clip, 
                                     self.distrib_name, self.distrib_param, use_weights=self.use_weights, reduce_axis=1)
        weights = log_weights
        
        # Assert weights shape
        chex.assert_shape(weights, (self.n_tasks, 1))
        
        return tasks, weights

    def generate_data_pool(self) -> Array:
        chex.assert_scalar_positive(self.n_data)  # Should be positive since we're generating a pool
        
        key = jax.random.fold_in(self.data_key, 0)
        shape = self.n_data, self.n_points, self.n_dims
        data = jax.random.normal(key, shape, self.dtype) * self.data_scale
        
        # Assert generated data pool shape
        chex.assert_shape(data, (self.n_data, self.n_points, self.n_dims))
        
        return data

    @jax.jit
    def sample_data(self, step: int) -> Array:
        key = jax.random.fold_in(self.data_key, step)
        if self.n_data > 0:
            idxs = jax.random.choice(key, self.n_data, (self.batch_size,))
            chex.assert_shape(idxs, (self.batch_size,))
            data = self.data_pool[idxs]
            # data_pool has shape (n_data, n_points, n_dims), so indexed data should be:
            chex.assert_shape(data, (self.batch_size, self.n_points, self.n_dims))
        else:
            shape = self.batch_size, self.n_points, self.n_dims
            data = jax.random.normal(key, shape, self.dtype) * self.data_scale + self.task_center
            chex.assert_shape(data, (self.batch_size, self.n_points, self.n_dims))
        
        # Final assertion on returned data
        batch_size_actual, n_points_actual, n_dims_actual = data.shape
        chex.assert_shape(data, (batch_size_actual, n_points_actual, n_dims_actual))
        chex.assert_equal(batch_size_actual, self.batch_size)
        chex.assert_equal(n_points_actual, self.n_points) 
        chex.assert_equal(n_dims_actual, self.n_dims)
        
        return data

    @jax.jit
    def sample_tasks(self, step: int) -> Array:
        key = jax.random.fold_in(self.task_key, step)
        if self.n_tasks > 0:
            if self.use_weight_sampling:
                idxs = jax.random.categorical(key, self.weights, axis=0, shape=(self.batch_size,))
                chex.assert_shape(idxs, (self.batch_size,))
                log_weights = jnp.zeros((self.batch_size, 1), self.dtype)
                chex.assert_shape(log_weights, (self.batch_size, 1))
                tasks = self.task_pool[idxs]
                # task_pool has shape (n_tasks, task_n_dims, 1), so indexed tasks should be:
                chex.assert_shape(tasks, (self.batch_size, self.task_n_dims, 1))
            else:
                idxs = jax.random.choice(key, self.n_tasks, (self.batch_size,))
                chex.assert_shape(idxs, (self.batch_size,))
                tasks = self.task_pool[idxs]
                chex.assert_shape(tasks, (self.batch_size, self.task_n_dims, 1))
                log_weights = task_log_weights(tasks, self.task_center, self.task_scale, self.clip, 
                                             self.distrib_name, self.distrib_param, use_weights=self.use_weights, reduce_axis=1)
                chex.assert_shape(log_weights, (self.batch_size, 1))
        else:
            shape = self.batch_size, self.task_n_dims, 1
            tasks = sample_distrib(key, self.task_center, self.task_scale, self.clip, 
                                 self.distrib_name, self.distrib_param, shape, self.dtype)
            chex.assert_shape(tasks, (self.batch_size, self.task_n_dims, 1))
            log_weights = task_log_weights(tasks, self.task_center, self.task_scale, self.clip, 
                                         self.distrib_name, self.distrib_param, use_weights=self.use_weights, reduce_axis=1)
            chex.assert_shape(log_weights, (self.batch_size, 1))
        
        weights = log_weights
        
        # Final assertions on returned values  
        batch_size_actual, task_n_dims_actual, one_dim = tasks.shape
        chex.assert_shape(tasks, (batch_size_actual, task_n_dims_actual, one_dim))
        chex.assert_equal(batch_size_actual, self.batch_size)
        chex.assert_equal(task_n_dims_actual, self.task_n_dims)
        chex.assert_equal(one_dim, 1)
        
        batch_size_w, one_dim_w = weights.shape
        chex.assert_shape(weights, (batch_size_w, one_dim_w))
        chex.assert_equal(batch_size_w, self.batch_size)
        chex.assert_equal(one_dim_w, 1)
        
        return tasks, weights

    @partial(jax.jit, static_argnames=('evl',))
    def evaluate(self, tasks: Array, step: int, evl=False) -> Array:
        """
        **CRITICAL CHANGE**: Use MLP drift instead of linear OU drift.
        """
        batch_size_actual, task_n_dims_actual, one_dim = tasks.shape
        chex.assert_shape(tasks, (batch_size_actual, task_n_dims_actual, one_dim))
        chex.assert_equal(batch_size_actual, self.batch_size)
        chex.assert_equal(task_n_dims_actual, self.task_n_dims)
        chex.assert_equal(one_dim, 1)

        # Extract MLP parameters instead of mu/theta
        mlp_params = self.get_params_from_tasks(tasks)

        key = jax.random.fold_in(self.noise_key, step)
        # Generate noise using CURRENT dimensions (curriculum masking)
        n_points_loop = self.n_points * self.inner_steps
        all_noise = jax.random.truncated_normal(
                key,
                -self.data_noise_trunc_radius / self.noise_scale,
                self.data_noise_trunc_radius / self.noise_scale,
                shape = (n_points_loop+1, self.batch_size, self.current_n_dims),
                dtype = self.dtype
                ) * self.noise_scale
        chex.assert_shape(all_noise, (n_points_loop+1, self.batch_size, self.current_n_dims))

        init = all_noise[0, :, :]
        chex.assert_shape(init, (self.batch_size, self.current_n_dims))

        all_noise = all_noise[1:, :, :]
        chex.assert_shape(all_noise, (n_points_loop, self.batch_size, self.current_n_dims))

        indices = jnp.arange(n_points_loop + 1)
        ou_step = self.ou_step / self.inner_steps

        def compute_change(state, noise):
            """
            Compute b(X_t) dt + sigma sqrt(dt) Z_t
            """
            chex.assert_shape(state, (self.batch_size, self.current_n_dims))
            chex.assert_shape(noise, (self.batch_size, self.current_n_dims))

            state_3d = state[:, None, :]  # (batch_size, 1, current_n_dims)
            chex.assert_shape(state_3d, (self.batch_size, 1, self.current_n_dims))

            drift_3d = self.apply_mlp_drift(state_3d, mlp_params)
            chex.assert_shape(drift_3d, (self.batch_size, 1, self.current_n_dims))

            drift = drift_3d[:, 0, :]  # (batch_size, current_n_dims)
            chex.assert_shape(drift, (self.batch_size, self.current_n_dims))

            change = -drift * ou_step + jnp.sqrt(ou_step) * noise
            chex.assert_shape(change, (self.batch_size, self.current_n_dims))

            return change

        def mlp_sde_step(carry, t):
            """
            Compute X_(t+1) = sum_{s <= t} G(t-s) (b(X_s) dt + sigma sqrt(dt) Z_s)
            where all_changes[s] = b(X_s) dt + sigma sqrt(dt) Z_s
            Returns X_(t+1) and updates all_changes.
            """
            all_changes = carry
            chex.assert_shape(all_changes, (n_points_loop+1, self.batch_size, self.current_n_dims))
            total_time = 1 #self.n_points * self.ou_step
            g_coefs = jax.lax.select(indices <= t, (t/total_time - indices/total_time + 1) ** (-self.kernel_exponent), indices * 0.0)
            chex.assert_shape(g_coefs, (n_points_loop+1,))

            new_state = jnp.einsum('s,sbd->bd', g_coefs, all_changes)
            chex.assert_shape(new_state, (self.batch_size, self.current_n_dims))

            noise = all_changes[t+1, :, :]  # (batch_size, current_n_dims)
            chex.assert_shape(noise, (self.batch_size, self.current_n_dims))

            new_change = compute_change(new_state, noise)
            chex.assert_shape(new_change, (self.batch_size, self.current_n_dims))

            new_all_changes = all_changes.at[t+1, :, :].set(new_change)

            return new_all_changes, new_state
            
        # Run the MLP SDE for n_points steps
        all_changes_init =  jnp.zeros((n_points_loop+1, self.batch_size, self.current_n_dims), dtype=self.dtype)
        # Note all_changes_init[self.n_points, :, :] is unused
        all_changes_init = all_changes_init.at[1:n_points_loop, :, :].set(all_noise[1:, :, :])

        init_change = compute_change(init, all_noise[0, :, :])
        chex.assert_shape(init_change, (self.batch_size, self.current_n_dims))

        all_changes_init = all_changes_init.at[0, :, :].set(init_change)

        _, sde_steps = jax.lax.scan(mlp_sde_step, all_changes_init, indices[:-1])
        chex.assert_shape(sde_steps, (n_points_loop, self.batch_size, self.current_n_dims))

        if evl:
            target_sde_steps = sde_steps - all_noise * jnp.sqrt(ou_step)
        else:
            target_sde_steps = sde_steps

        # Downsample to original n_points
        sde_steps = sde_steps[self.inner_steps-1::self.inner_steps, :, :]  # Shape: (n_points, batch_size, current_n_dims)
        chex.assert_shape(sde_steps, (self.n_points, self.batch_size, self.current_n_dims))

        target_sde_steps = target_sde_steps[self.inner_steps-1::self.inner_steps, :, :]
        chex.assert_shape(target_sde_steps, (self.n_points, self.batch_size, self.current_n_dims))

        sde_steps = jnp.transpose(sde_steps, (1, 0, 2))  # Shape: (batch_size, n_points, current_n_dims)
        chex.assert_shape(sde_steps, (self.batch_size, self.n_points, self.current_n_dims))

        target_sde_steps = jnp.transpose(target_sde_steps, (1, 0, 2))
        chex.assert_shape(target_sde_steps, (self.batch_size, self.n_points, self.current_n_dims))

        # Final assertions on return values using current curriculum dimensions
        init_bs, init_dims = init.shape
        chex.assert_shape(init, (init_bs, init_dims))
        chex.assert_equal(init_bs, self.batch_size)
        chex.assert_equal(init_dims, self.current_n_dims)
        
        sde_bs, sde_points, sde_dims = sde_steps.shape
        chex.assert_shape(sde_steps, (sde_bs, sde_points, sde_dims))
        chex.assert_equal(sde_bs, self.batch_size)
        chex.assert_equal(sde_points, self.n_points)
        chex.assert_equal(sde_dims, self.current_n_dims)

        new_init = jnp.concatenate([init, jnp.zeros((self.batch_size, self.max_n_dims - self.current_n_dims), dtype=init.dtype)], axis=1)
        chex.assert_shape(new_init, (self.batch_size, self.max_n_dims))

        new_sde_steps = jnp.concatenate([sde_steps, jnp.zeros((self.batch_size, self.n_points, self.max_n_dims - self.current_n_dims), dtype=sde_steps.dtype)], axis=2)
        chex.assert_shape(new_sde_steps, (self.batch_size, self.n_points, self.max_n_dims))

        new_target_sde_steps = jnp.concatenate([target_sde_steps, jnp.zeros((self.batch_size, self.n_points, self.max_n_dims - self.current_n_dims), dtype=target_sde_steps.dtype)], axis=2)
        chex.assert_shape(new_target_sde_steps, (self.batch_size, self.n_points, self.max_n_dims))

        return new_init, new_sde_steps, new_target_sde_steps

    @jax.jit
    def generate_attention_mask(self) -> Array:
        """Generate causal attention mask for the sequence with right padding."""
        effective_seq_len = self.n_points      # Valid data: positions 0 to this-1
        max_seq_len = self.n_max_points        # Total padded length
        
        chex.assert_scalar_non_negative(effective_seq_len)
        chex.assert_scalar_positive(max_seq_len)
        assert effective_seq_len <= max_seq_len, f"effective_seq_len {effective_seq_len} > max_seq_len {max_seq_len}"
        
        # Start with all positions masked (False)
        mask = jnp.zeros((max_seq_len, max_seq_len), dtype=bool)
        chex.assert_shape(mask, (max_seq_len, max_seq_len))
        
        # Valid region gets causal attention pattern
        valid_mask = jnp.tril(jnp.ones((effective_seq_len, effective_seq_len))).astype(bool)
        chex.assert_shape(valid_mask, (effective_seq_len, effective_seq_len))
        
        # Insert valid causal mask into full mask 
        mask = mask.at[:effective_seq_len, :effective_seq_len].set(valid_mask)
        
        # Final assertion on return value
        chex.assert_shape(mask, (self.n_max_points, self.n_max_points))
        
        return mask

    def curriculum_increment(self):
        """Enhanced curriculum learning that can increment multiple dimensions."""
        changes = []
        
        # Increment n_points (original curriculum)
        old_n_points = self.n_points
        self.n_points = min(self.n_points + self.curriculum_n_points_increment, 
                           self.n_max_points)
        if self.n_points > old_n_points:
            changes.append(f"n_points {old_n_points} -> {self.n_points}")
        
        # Increment hidden size curriculum
        old_hidden_size = self.current_hidden_size
        self.current_hidden_size = min(self.current_hidden_size + self.curriculum_hidden_increment,
                                     self.max_hidden_size)
        if self.current_hidden_size > old_hidden_size:
            changes.append(f"hidden_size {old_hidden_size} -> {self.current_hidden_size}")
        
        # Increment dimension curriculum  
        old_n_dims = self.current_n_dims
        self.current_n_dims = min(self.current_n_dims + self.curriculum_dims_increment,
                                self.max_n_dims)
        if self.current_n_dims > old_n_dims:
            changes.append(f"n_dims {old_n_dims} -> {self.current_n_dims}")
        
        # Log all changes
        if changes:
            logging.info(f"Curriculum increment: {', '.join(changes)}")

    def sample_batch(self, step: int, evl=False) -> tuple[Array, Array, Array, Array]:
        if step % self.curriculum_steps_thresh == self.curriculum_steps_thresh - 1 and self.use_curriculum:
            self.curriculum_increment()

        (tasks, weights) = self.sample_tasks(step)
        chex.assert_shape(tasks, (self.batch_size, self.task_n_dims, 1))
        chex.assert_shape(weights, (self.batch_size, 1))

        init, sde_steps, targets = self.evaluate(tasks, step, evl=evl)
        chex.assert_shape(init, (self.batch_size, self.max_n_dims))
        chex.assert_shape(sde_steps, (self.batch_size, self.n_points, self.max_n_dims))
        chex.assert_shape(targets, (self.batch_size, self.n_points, self.max_n_dims))

        data = jnp.concatenate((init[:, None, :], sde_steps[:, :-1, :]), axis=1)
        chex.assert_shape(data, (self.batch_size, self.n_points, self.max_n_dims))

        attention_mask = self.generate_attention_mask()
        chex.assert_shape(attention_mask, (self.n_max_points, self.n_max_points))

        return data, tasks, weights, targets, attention_mask

    @jax.jit
    def evaluate_oracle(self, data: Array, tasks: Array, targets) -> Array:
        """Oracle prediction using MLP drift."""
        del data, tasks
        return targets


    def get_default_eval_tasks(
            self, batch_size: int, task_seed: int, data_seed: int, noise_seed: int, eval_n_points: List[int], task_centers: List[float] | None = None, **kwargs
            ) -> list["MLPSDETask"]:
        del kwargs
        assert task_seed != self.task_seed
        assert data_seed != self.data_seed
        assert noise_seed != self.noise_seed
        config = dataclasses.asdict(self)
        config["batch_size"] = batch_size
        config["task_seed"] = task_seed
        config["data_seed"] = data_seed
        config["noise_seed"] = noise_seed
        config["n_tasks"] = 0
        config["n_data"] = 0
        config["n_max_points"] = self.n_max_points
        config["use_curriculum"] = False  # Disable curriculum for evaluation
        config["use_weights"] = False
        eval_tasks = []
        n_points = eval_n_points
        assert n_points <= self.n_max_points, f"n_points {n_points} exceeds n_max_points {self.n_max_points}"
        config["n_points"] = n_points
        # Increment seeds
        config["task_seed"] += 1
        config["data_seed"] += 1
        config["noise_seed"] += 1

        # Test  with fresh tasks from training distribution
        name = f"Test tasks"
        config["name"] = name
        eval_tasks.append(self.__class__(**config))

        # Test with same tasks as training distribution
        if self.n_tasks > 0:
            # Increment seeds
            config["task_seed"] += 1
            config["data_seed"] += 1
            config["noise_seed"] += 1

            name = f"Train tasks"
            config["n_tasks"] = self.n_tasks
            config["name"] = name
            eval_tasks.append(VolterraTask.from_task_pool(**config, task_pool=self.task_pool.copy(), weights=self.weights.copy()))
        
        config["n_tasks"] = 0  # Reset for fresh tasks

        # Test with fixed task centers
        if task_centers is not None:
            config["distrib_name"] = "normal"  # Reset to normal distribution for fixed tasks
            for task_center in task_centers:
                # Increment seeds
                config["task_seed"] += 1
                config["data_seed"] += 1
                config["noise_seed"] += 1

                config["task_center"] = task_center
                config["clip"] = None
                name = f"Fixed task {task_center}"
                config["name"] = name
                eval_tasks.append(self.__class__(**config))
        return eval_tasks

    def get_default_eval_models(self) -> list[Model]:
        return [get_model(name="last_value"), get_model(name="arma", dtype=self.dtype), get_model(name="corrected_last_value")]

    def _tree_flatten(self):
        # Dynamic values (arrays, keys, and values that can change)
        children = (
            self.data_key,
            self.task_key, 
            self.noise_key,
            self.task_pool,
            self.weights,
            self.data_pool,
            self.data_scale,
            self.task_scale,
            self.noise_scale,
            self.task_center,
            self.clip,
        )
        
        # Static values (configuration that doesn't change during execution)
        aux_data = {
            'n_tasks': self.n_tasks,
            'n_data': self.n_data,
            'n_dims': self.n_dims,
            'n_points': self.n_points,
            'batch_size': self.batch_size,
            'data_seed': self.data_seed,
            'task_seed': self.task_seed,
            'noise_seed': self.noise_seed,
            'dtype': self.dtype,
            'n_max_points': self.n_max_points,
            'name': self.name,
            'eval_ridge': self.eval_ridge,
            'use_weights': self.use_weights,
            'use_weight_sampling': self.use_weight_sampling,
            'distrib_name': self.distrib_name,
            'distrib_param': self.distrib_param,
            'use_curriculum': self.use_curriculum,
            'curriculum_n_points_increment': self.curriculum_n_points_increment,
            'curriculum_steps_thresh': self.curriculum_steps_thresh,
            'ou_step': self.ou_step,
            'hidden_size': self.hidden_size,
            'task_n_dims': self.task_n_dims,
            'max_hidden_size': self.max_hidden_size,
            'curriculum_hidden_increment': self.curriculum_hidden_increment,
            'min_hidden_size': self.min_hidden_size,
            'max_n_dims': self.max_n_dims,
            'curriculum_dims_increment': self.curriculum_dims_increment,
            'min_n_dims': self.min_n_dims,
            'current_hidden_size': self.current_hidden_size,
            'current_n_dims': self.current_n_dims,
            'data_noise_trunc_radius': self.data_noise_trunc_radius,
            'kernel_exponent': self.kernel_exponent,
            'inner_steps': self.inner_steps,
            'drift_scale': self.drift_scale,
            'drift_clip': self.drift_clip,
            'drift_reg': self.drift_reg,
        }
        
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        (data_key, task_key, noise_key, task_pool, weights, data_pool,
         data_scale, task_scale, noise_scale, task_center, clip) = children
        
        # Create object with aux_data parameters and placeholder scale values
        obj = cls(data_scale=1.0, task_scale=1.0, noise_scale=1.0, 
                 task_center=0.0, clip=None, _skip_init=True, **aux_data)
        
        # Set the dynamic values
        obj.data_key = data_key
        obj.task_key = task_key
        obj.noise_key = noise_key
        obj.task_pool = task_pool
        obj.weights = weights
        obj.data_pool = data_pool
        obj.data_scale = data_scale
        obj.task_scale = task_scale
        obj.noise_scale = noise_scale
        obj.task_center = task_center
        obj.clip = clip
        
        return obj


tree_util.register_pytree_node(
        VolterraTask,
        VolterraTask._tree_flatten,
        VolterraTask._tree_unflatten
        )



########################################################################################################################
# Get Task                                                                                                             #
########################################################################################################################

Task = NoisyLinearRegression


def get_task(name: str, **kwargs) -> Task:
    tasks = {
            "noisy_linear_regression": NoisyLinearRegression,
            "ornstein_uhlenbeck": OrnsteinUhlenbeckTask,
            "mlp_sde": MLPSDETask,
            "volterra": VolterraTask,
            }
    return tasks[name](**kwargs)
