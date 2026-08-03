#!/usr/bin/env python3
"""
Mean, minimum, and best MSE analysis functionality.
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import jax
from jax import jit
from functools import partial

from loading import load_log_with_safetensors
from task_shift import (
    extract_task_shift_distance, normalize_error_values, compute_min_mean_end_mse_over_context,
    compute_auc_trapz, extract_swept_params, get_param_value_from_config, group_runs_by_other_params,
    find_best_param_combination_by_auc, format_parameter_legend, find_valid_multirun_subdirs
)


def compute_min_mean_end_mse_for_prefix(mse_values: jnp.ndarray, prefix_length: int) -> tuple[float, float, float]:
    """Compute min, mean, and end MSE over a specific sequence length prefix (NOT JIT compiled to handle dynamic prefix).

    Args:
        mse_values: MSE values over context positions
        prefix_length: Number of positions to include from the beginning (must be <= len(mse_values))

    Returns:
        tuple: (min_mse_excluding_first, mean_mse_prefix, end_mse_prefix)
    """
    # Ensure prefix_length doesn't exceed available data
    actual_prefix = min(prefix_length, len(mse_values))

    # Take only the first prefix_length positions using numpy-style slicing
    prefix_values = mse_values[:actual_prefix]

    # Skip first position (index 0) for min MSE as per original code
    if len(prefix_values) > 1:
        min_mse = jnp.min(prefix_values[1:])
    else:
        min_mse = prefix_values[0]

    mean_mse = jnp.mean(prefix_values)
    end_mse = prefix_values[actual_prefix - 1]  # MSE at last position in prefix
    return min_mse, mean_mse, end_mse


def compute_min_mean_end_mse_std_for_prefix(mse_values: jnp.ndarray, mse_std_values: jnp.ndarray, prefix_length: int) -> tuple[float, float, float]:
    """Compute std values for min, mean, and end MSE over a specific sequence length prefix.

    Args:
        mse_values: MSE values over context positions (needed to find min position)
        mse_std_values: MSE std values over context positions
        prefix_length: Number of positions to include from the beginning (must be <= len(mse_values))

    Returns:
        tuple: (min_mse_std_at_min_pos, mean_mse_std_aggregated, end_mse_std_prefix)
    """
    # Ensure prefix_length doesn't exceed available data
    actual_prefix = min(prefix_length, len(mse_values))

    # Take only the first prefix_length positions
    prefix_mse_values = mse_values[:actual_prefix]
    prefix_std_values = mse_std_values[:actual_prefix]

    # Find min MSE position (skip first position as per original code)
    if len(prefix_mse_values) > 1:
        min_pos = jnp.argmin(prefix_mse_values[1:]) + 1  # +1 because we skipped index 0
        min_mse_std = prefix_std_values[min_pos]
    else:
        min_mse_std = prefix_std_values[0]

    # Mean MSE std: sqrt of average of std²
    mean_mse_std = jnp.sqrt(jnp.mean(prefix_std_values ** 2))

    # End MSE std: std at last position in prefix
    end_mse_std = prefix_std_values[actual_prefix - 1]

    return min_mse_std, mean_mse_std, end_mse_std


@partial(jit, static_argnames=('num_steps', 'num_tasks'))
def find_best_step_by_auc(all_min_mse: jnp.ndarray, all_mean_mse: jnp.ndarray, all_end_mse: jnp.ndarray,
                         shift_distances: jnp.ndarray, num_steps: int, num_tasks: int) -> tuple[int, int, int]:
    """Find evaluation steps with minimal AUC for log of min, mean, and end MSE over shift distance (JIT compiled).
    
    Args:
        all_min_mse: Array of shape (num_steps, num_tasks) with min MSE values
        all_mean_mse: Array of shape (num_steps, num_tasks) with mean MSE values  
        all_end_mse: Array of shape (num_steps, num_tasks) with end MSE values
        shift_distances: Array of shape (num_tasks,) with shift distances
        num_steps: Number of evaluation steps (static arg for JIT)
        num_tasks: Number of tasks (static arg for JIT)
        
    Returns:
        tuple: (best_step_for_min_mse, best_step_for_mean_mse, best_step_for_end_mse)
    """
    def compute_step_auc(step_idx):
        # Take logarithm of MSE values before computing AUC
        min_log_mse = jnp.log(all_min_mse[step_idx])
        mean_log_mse = jnp.log(all_mean_mse[step_idx])
        end_log_mse = jnp.log(all_end_mse[step_idx])
        
        min_auc = compute_auc_trapz(shift_distances, min_log_mse)
        mean_auc = compute_auc_trapz(shift_distances, mean_log_mse) 
        end_auc = compute_auc_trapz(shift_distances, end_log_mse)
        return min_auc, mean_auc, end_auc
    
    # Vectorized computation across all steps
    step_aucs = jax.vmap(compute_step_auc)(jnp.arange(num_steps))
    min_aucs, mean_aucs, end_aucs = step_aucs
    
    best_min_step = jnp.argmin(min_aucs)
    best_mean_step = jnp.argmin(mean_aucs)
    best_end_step = jnp.argmin(end_aucs)
    return num_steps-1, num_steps-1, num_steps-1
    # return best_min_step, best_mean_step, best_end_step


def compute_best_auc_for_baseline(log: dict, baseline_type: str) -> float:
    """Compute the best AUC (minimal mean MSE AUC) for a given baseline type.
    
    Args:
        log: The log dictionary
        baseline_type: Either 'Ridge' or 'True' to specify which baseline to use
        
    Returns:
        float: The minimal mean MSE AUC value, or float('inf') if computation fails
    """
    try:
        # Reuse the existing logic from extract_min_mse_params_for_baseline
        eval_steps = log.get("eval/step", [])
        if not eval_steps:
            return float('inf')
        
        # Extract evaluation metrics for all steps
        eval_metrics = {}
        for key, value in log.items():
            if key.startswith("eval/") and key != "eval/step":
                task_name = key.split("/")[1]
                if task_name not in eval_metrics:
                    eval_metrics[task_name] = {}
                for metric_name, metric_values in value.items():
                    eval_metrics[task_name][metric_name] = metric_values
        
        # Find tasks that match our criteria and baseline
        task_data = {}
        for task_name, metrics in eval_metrics.items():
            if task_name == "Test tasks" or task_name.startswith("Fixed task"):
                selected_metric = None
                for metric_name, values in metrics.items():
                    if f"Transformer | {baseline_type}" in metric_name and "(RelErr)" not in metric_name and values:
                        selected_metric = (metric_name, values)
                        break
                
                if selected_metric:
                    metric_name, values = selected_metric
                    shift_distance = extract_task_shift_distance(task_name)
                    task_data[task_name] = (shift_distance, values)
        
        if not task_data:
            return float('inf')
        
        # Sort tasks by shift distance for consistent ordering
        sorted_tasks = sorted(task_data.items(), key=lambda x: x[1][0])
        task_names = [task_name for task_name, _ in sorted_tasks]
        shift_distances = jnp.array([shift_dist for _, (shift_dist, _) in sorted_tasks])
        
        # Collect MSE data for all steps and tasks
        num_steps = len(eval_steps)
        num_tasks = len(sorted_tasks)
        
        all_min_mse = np.zeros((num_steps, num_tasks))
        all_mean_mse = np.zeros((num_steps, num_tasks))
        all_end_mse = np.zeros((num_steps, num_tasks))
        
        for task_idx, (task_name, (shift_dist, values)) in enumerate(sorted_tasks):
            for step_idx in range(num_steps):
                if step_idx < len(values):
                    mse_values = normalize_error_values(values[step_idx])
                    if mse_values is not None and len(mse_values) > 0:
                        mse_jax = jnp.array(mse_values)
                        min_mse, mean_mse, end_mse = compute_min_mean_end_mse_over_context(mse_jax)
                        all_min_mse[step_idx, task_idx] = float(min_mse)
                        all_mean_mse[step_idx, task_idx] = float(mean_mse)
                        all_end_mse[step_idx, task_idx] = float(end_mse)
                    else:
                        all_min_mse[step_idx, task_idx] = float('inf')
                        all_mean_mse[step_idx, task_idx] = float('inf')
                        all_end_mse[step_idx, task_idx] = float('inf')
                else:
                    all_min_mse[step_idx, task_idx] = float('inf')
                    all_mean_mse[step_idx, task_idx] = float('inf')
                    all_end_mse[step_idx, task_idx] = float('inf')
        
        # Convert to JAX arrays for optimized computation
        all_min_mse_jax = jnp.array(all_min_mse)
        all_mean_mse_jax = jnp.array(all_mean_mse)
        all_end_mse_jax = jnp.array(all_end_mse)
        
        # Find best step and return the minimal mean AUC
        def compute_step_auc(step_idx):
            mean_log_mse = jnp.log(all_mean_mse_jax[step_idx])
            return compute_auc_trapz(shift_distances, mean_log_mse)
        
        step_aucs = jax.vmap(compute_step_auc)(jnp.arange(num_steps))
        min_auc = float(jnp.min(step_aucs))
        
        return min_auc
        
    except Exception as e:
        return float('inf')


def extract_std_mse_params_for_baseline_with_prefix(log: dict, baseline_type: str, prefix_length: int, return_selected_steps: bool = False) -> tuple[dict, dict, dict] | tuple[dict, dict, dict, dict]:
    """Extract std values for min, mean, and end MSE over sequence length prefix for iteration with minimal AUC of log MSE over shift distance for all tasks for a specific baseline and prefix.

    Args:
        log: The log dictionary
        baseline_type: Either 'Ridge' or 'True' to specify which baseline to use
        prefix_length: Number of sequence positions to include from the beginning
        return_selected_steps: If True, also return which steps were selected

    Returns:
        tuple: (min_mse_std_dict, mean_mse_std_dict, end_mse_std_dict) or (min_mse_std_dict, mean_mse_std_dict, end_mse_std_dict, selected_steps_dict) where:
            min_mse_std_dict: {task_name: std_at_min_mse_pos_over_prefix}
            mean_mse_std_dict: {task_name: aggregated_std_over_prefix}
            end_mse_std_dict: {task_name: std_at_prefix_end}
            selected_steps_dict: {task_name: (min_mse_step, mean_mse_step, end_mse_step)} - only if return_selected_steps=True
    """
    eval_steps = log.get("eval/step", [])
    if not eval_steps:
        if return_selected_steps:
            return {}, {}, {}, {}
        else:
            return {}, {}, {}

    # Extract evaluation metrics for all steps
    eval_metrics = {}
    for key, value in log.items():
        if key.startswith("eval/") and key != "eval/step":
            task_name = key.split("/")[1]
            if task_name not in eval_metrics:
                eval_metrics[task_name] = {}
            for metric_name, metric_values in value.items():
                eval_metrics[task_name][metric_name] = metric_values

    # Find tasks that match our criteria and baseline - look for both MSE and MSE_Std
    task_data = {}  # {task_name: (shift_distance, mse_values, mse_std_values)}

    for task_name, metrics in eval_metrics.items():
        # Include both Test tasks and Fixed task
        if task_name == "Test tasks" or task_name.startswith("Fixed task"):
            # Look for the specific baseline type MSE values
            selected_mse_metric = None
            selected_std_metric = None
            for metric_name, values in metrics.items():
                if f"Transformer | {baseline_type}" in metric_name and "(RelErr)" not in metric_name and values:
                    if "(Std)" in metric_name:
                        selected_std_metric = (metric_name, values)
                    else:
                        selected_mse_metric = (metric_name, values)

            if selected_mse_metric and selected_std_metric:
                _, mse_values = selected_mse_metric
                _, std_values = selected_std_metric
                shift_distance = extract_task_shift_distance(task_name)
                task_data[task_name] = (shift_distance, mse_values, std_values)

    if not task_data:
        if return_selected_steps:
            return {}, {}, {}, {}
        else:
            return {}, {}, {}

    # Sort tasks by shift distance for consistent ordering
    sorted_tasks = sorted(task_data.items(), key=lambda x: x[1][0])
    task_names = [task_name for task_name, _ in sorted_tasks]
    shift_distances = jnp.array([shift_dist for _, (shift_dist, _, _) in sorted_tasks])

    # Collect MSE and MSE_Std data for all steps and tasks using the prefix length
    num_steps = len(eval_steps)
    num_tasks = len(sorted_tasks)

    all_min_mse = np.zeros((num_steps, num_tasks))
    all_mean_mse = np.zeros((num_steps, num_tasks))
    all_end_mse = np.zeros((num_steps, num_tasks))

    all_min_mse_std = np.zeros((num_steps, num_tasks))
    all_mean_mse_std = np.zeros((num_steps, num_tasks))
    all_end_mse_std = np.zeros((num_steps, num_tasks))

    for task_idx, (task_name, (shift_dist, mse_values, std_values)) in enumerate(sorted_tasks):
        for step_idx in range(num_steps):
            if step_idx < len(mse_values) and step_idx < len(std_values):
                mse_vals = normalize_error_values(mse_values[step_idx])
                std_vals = normalize_error_values(std_values[step_idx])
                if mse_vals is not None and std_vals is not None and len(mse_vals) > 0 and len(std_vals) > 0:
                    # Convert to JAX arrays for computation
                    mse_jax = jnp.array(mse_vals)
                    std_jax = jnp.array(std_vals)

                    # Compute MSE statistics for AUC computation (unchanged)
                    min_mse, mean_mse, end_mse = compute_min_mean_end_mse_for_prefix(mse_jax, prefix_length)
                    all_min_mse[step_idx, task_idx] = float(min_mse)
                    all_mean_mse[step_idx, task_idx] = float(mean_mse)
                    all_end_mse[step_idx, task_idx] = float(end_mse)

                    # Compute corresponding std statistics
                    min_mse_std, mean_mse_std, end_mse_std = compute_min_mean_end_mse_std_for_prefix(mse_jax, std_jax, prefix_length)
                    all_min_mse_std[step_idx, task_idx] = float(min_mse_std)
                    all_mean_mse_std[step_idx, task_idx] = float(mean_mse_std)
                    all_end_mse_std[step_idx, task_idx] = float(end_mse_std)
                else:
                    all_min_mse[step_idx, task_idx] = float('inf')
                    all_mean_mse[step_idx, task_idx] = float('inf')
                    all_end_mse[step_idx, task_idx] = float('inf')
                    all_min_mse_std[step_idx, task_idx] = 0.0
                    all_mean_mse_std[step_idx, task_idx] = 0.0
                    all_end_mse_std[step_idx, task_idx] = 0.0
            else:
                all_min_mse[step_idx, task_idx] = float('inf')
                all_mean_mse[step_idx, task_idx] = float('inf')
                all_end_mse[step_idx, task_idx] = float('inf')
                all_min_mse_std[step_idx, task_idx] = 0.0
                all_mean_mse_std[step_idx, task_idx] = 0.0
                all_end_mse_std[step_idx, task_idx] = 0.0

    # Convert to JAX arrays for optimized computation
    all_min_mse_jax = jnp.array(all_min_mse)
    all_mean_mse_jax = jnp.array(all_mean_mse)
    all_end_mse_jax = jnp.array(all_end_mse)

    # Find best steps with minimal AUC (JIT compiled) - use MSE values for selection, not std
    best_min_step, best_mean_step, best_end_step = find_best_step_by_auc(
        all_min_mse_jax, all_mean_mse_jax, all_end_mse_jax, shift_distances, num_steps, num_tasks
    )

    # Extract std results from the best steps (determined by MSE AUC)
    min_mse_std_results = {}
    mean_mse_std_results = {}
    end_mse_std_results = {}
    selected_steps = {}

    for task_idx, task_name in enumerate(task_names):
        min_mse_std_results[task_name] = float(all_min_mse_std[int(best_min_step), task_idx])
        mean_mse_std_results[task_name] = float(all_mean_mse_std[int(best_mean_step), task_idx])
        end_mse_std_results[task_name] = float(all_end_mse_std[int(best_end_step), task_idx])
        if return_selected_steps:
            # Convert JAX array indices to Python ints and then to actual step numbers
            min_step_num = eval_steps[int(best_min_step)]
            mean_step_num = eval_steps[int(best_mean_step)]
            end_step_num = eval_steps[int(best_end_step)]
            selected_steps[task_name] = (min_step_num, mean_step_num, end_step_num)

    if return_selected_steps:
        return min_mse_std_results, mean_mse_std_results, end_mse_std_results, selected_steps
    else:
        return min_mse_std_results, mean_mse_std_results, end_mse_std_results


def extract_min_mse_params_for_baseline_with_prefix(log: dict, baseline_type: str, prefix_length: int, return_selected_steps: bool = False) -> tuple[dict, dict, dict] | tuple[dict, dict, dict, dict]:
    """Extract minimum MSE over sequence length prefix, mean MSE over prefix, and end MSE for iteration with minimal AUC of log MSE over shift distance for all tasks for a specific baseline and prefix.
    
    Args:
        log: The log dictionary
        baseline_type: Either 'Ridge' or 'True' to specify which baseline to use
        prefix_length: Number of sequence positions to include from the beginning
        return_selected_steps: If True, also return which steps were selected
    
    Returns:
        tuple: (min_mse_dict, mean_mse_dict, end_mse_dict) or (min_mse_dict, mean_mse_dict, end_mse_dict, selected_steps_dict) where:
            min_mse_dict: {task_name: min_mse_over_prefix}
            mean_mse_dict: {task_name: mean_mse_over_prefix}
            end_mse_dict: {task_name: end_mse_at_prefix}
            selected_steps_dict: {task_name: (min_mse_step, mean_mse_step, end_mse_step)} - only if return_selected_steps=True
    """
    eval_steps = log.get("eval/step", [])
    if not eval_steps:
        if return_selected_steps:
            return {}, {}, {}, {}
        else:
            return {}, {}, {}
    
    # Extract evaluation metrics for all steps
    eval_metrics = {}
    for key, value in log.items():
        if key.startswith("eval/") and key != "eval/step":
            task_name = key.split("/")[1]
            if task_name not in eval_metrics:
                eval_metrics[task_name] = {}
            for metric_name, metric_values in value.items():
                eval_metrics[task_name][metric_name] = metric_values
    
    # Find tasks that match our criteria and baseline
    task_data = {}  # {task_name: (shift_distance, metric_values)}
    
    for task_name, metrics in eval_metrics.items():
        # Include both Test tasks and Fixed task
        if task_name == "Test tasks" or task_name.startswith("Fixed task"):
            # Look for the specific baseline type
            selected_metric = None
            for metric_name, values in metrics.items():
                if f"Transformer | {baseline_type}" in metric_name and "(RelErr)" not in metric_name and values:
                    selected_metric = (metric_name, values)
                    break
            
            if selected_metric:
                metric_name, values = selected_metric
                shift_distance = extract_task_shift_distance(task_name)
                task_data[task_name] = (shift_distance, values)
    
    if not task_data:
        if return_selected_steps:
            return {}, {}, {}, {}
        else:
            return {}, {}, {}
    
    # Sort tasks by shift distance for consistent ordering
    sorted_tasks = sorted(task_data.items(), key=lambda x: x[1][0])
    task_names = [task_name for task_name, _ in sorted_tasks]
    shift_distances = jnp.array([shift_dist for _, (shift_dist, _) in sorted_tasks])
    
    # Collect MSE data for all steps and tasks using the prefix length
    num_steps = len(eval_steps)
    num_tasks = len(sorted_tasks)
    
    all_min_mse = np.zeros((num_steps, num_tasks))
    all_mean_mse = np.zeros((num_steps, num_tasks))
    all_end_mse = np.zeros((num_steps, num_tasks))
    
    for task_idx, (task_name, (shift_dist, values)) in enumerate(sorted_tasks):
        for step_idx in range(num_steps):
            if step_idx < len(values):
                mse_values = normalize_error_values(values[step_idx])
                if mse_values is not None and len(mse_values) > 0:
                    # Convert to JAX array for JIT computation
                    mse_jax = jnp.array(mse_values)
                    # Use the prefix-specific function instead of the full context one
                    min_mse, mean_mse, end_mse = compute_min_mean_end_mse_for_prefix(mse_jax, prefix_length)
                    all_min_mse[step_idx, task_idx] = float(min_mse)
                    all_mean_mse[step_idx, task_idx] = float(mean_mse)
                    all_end_mse[step_idx, task_idx] = float(end_mse)
                else:
                    all_min_mse[step_idx, task_idx] = float('inf')
                    all_mean_mse[step_idx, task_idx] = float('inf')
                    all_end_mse[step_idx, task_idx] = float('inf')
            else:
                all_min_mse[step_idx, task_idx] = float('inf')
                all_mean_mse[step_idx, task_idx] = float('inf')
                all_end_mse[step_idx, task_idx] = float('inf')
    
    # Convert to JAX arrays for optimized computation
    all_min_mse_jax = jnp.array(all_min_mse)
    all_mean_mse_jax = jnp.array(all_mean_mse)
    all_end_mse_jax = jnp.array(all_end_mse)
    
    # Find best steps with minimal AUC (JIT compiled)
    best_min_step, best_mean_step, best_end_step = find_best_step_by_auc(
        all_min_mse_jax, all_mean_mse_jax, all_end_mse_jax, shift_distances, num_steps, num_tasks
    )
    
    # Extract results from the best steps
    min_mse_results = {}
    mean_mse_results = {}
    end_mse_results = {}
    selected_steps = {}
    
    for task_idx, task_name in enumerate(task_names):
        min_mse_results[task_name] = float(all_min_mse[int(best_min_step), task_idx])
        mean_mse_results[task_name] = float(all_mean_mse[int(best_mean_step), task_idx])
        end_mse_results[task_name] = float(all_end_mse[int(best_end_step), task_idx])
        if return_selected_steps:
            # Convert JAX array indices to Python ints and then to actual step numbers
            min_step_num = eval_steps[int(best_min_step)]
            mean_step_num = eval_steps[int(best_mean_step)]
            end_step_num = eval_steps[int(best_end_step)]
            selected_steps[task_name] = (min_step_num, mean_step_num, end_step_num)
    
    if return_selected_steps:
        return min_mse_results, mean_mse_results, end_mse_results, selected_steps
    else:
        return min_mse_results, mean_mse_results, end_mse_results


def extract_min_mse_params(log: dict) -> tuple[dict, dict, dict]:
    """Extract minimum MSE over context length, mean MSE over context length, and end MSE for iteration with minimal AUC of log MSE over shift distance for all tasks.
    
    Returns:
        tuple: (min_mse_dict, mean_mse_dict, end_mse_dict) where:
            min_mse_dict: {task_name: min_mse}
            mean_mse_dict: {task_name: mean_mse_over_context}
            end_mse_dict: {task_name: end_mse}
    """
    # This function needs to be imported or we need to create a baseline version
    # Let's create a simple version for now:
    from .loading import log  # This would need to be fixed
    
    # Try full sequence length with Ridge first
    min_mse_ridge, mean_mse_ridge, end_mse_ridge = extract_min_mse_params_for_baseline_with_prefix(log, 'Ridge', 1000)  # Large prefix to get full sequence
    if min_mse_ridge:
        return min_mse_ridge, mean_mse_ridge, end_mse_ridge
    else:
        return extract_min_mse_params_for_baseline_with_prefix(log, 'True', 1000)


def load_all_logs(run_paths: list, run_labels: list = None) -> dict:
    """Load all log files and metadata upfront to avoid duplicate I/O.
    
    Args:
        run_paths: List of Path objects pointing to runs or multirun subdirs
        run_labels: Custom labels for runs (optional)
    
    Returns:
        dict: {
            'logs': {run_label: log_data},
            'metadata': {run_label: (config, task_centers)},
            'run_labels': [ordered_list_of_run_labels]
        }
    """
    loaded_data = {
        'logs': {},
        'metadata': {},
        'run_labels': []
    }
    
    if not run_paths:
        return loaded_data
    
    actual_run_labels = []

    
    for i, run_path in enumerate(run_paths):
        run_path = Path(run_path)

        
        # Determine run label
        if run_labels and i < len(run_labels):
            run_label = run_labels[i]
        else:
            run_label = run_path.name
        
        # Check if this is a multirun directory or single run
        if (run_path / "multirun.yaml").exists():
            # This is a multirun directory - we want to analyze each subrun separately
            subdirs = find_valid_multirun_subdirs(run_path, return_paths=True)
            
            # Extract parameter names from multirun.yaml if no custom labels provided
            if run_labels is None:
                from task_shift import create_run_display_names
                param_names = create_run_display_names(run_path, [subdir.name for subdir in subdirs])
                if param_names:
                    subrun_labels = [param_names.get(int(subdir.name), subdir.name) for subdir in subdirs]
                else:
                    subrun_labels = [f"{run_label}-{subdir.name}" for subdir in subdirs]
            else:
                subrun_labels = [f"{run_label}-{subdir.name}" for subdir in subdirs]
            
            # Process each subrun as a separate run
            for subdir_idx, subdir in enumerate(subdirs):
                # Load config and log for this subrun
                config_path = subdir / "config.json"
                
                if not config_path.exists():
                    continue
                
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    log = load_log_with_safetensors(subdir)
                    
                    # Extract task centers from config
                    task_centers = config.get('eval', {}).get('task_centers', [])
                    
                    # Create run data for this subrun
                    if subdir_idx < len(subrun_labels):
                        subrun_label = subrun_labels[subdir_idx].strip()
                    else:
                        subrun_label = f"{run_label}-{subdir.name}"
                    
                    # Store loaded data
                    loaded_data['logs'][subrun_label] = log
                    loaded_data['metadata'][subrun_label] = (config, task_centers)
                    actual_run_labels.append(subrun_label)
                    
                except Exception as e:
                    print(f"Warning: Failed to load data for {subdir}: {e}")
                    raise e
                    continue
        
        elif (run_path / "log.json").exists():
            # This is a single run
            config_path = run_path / "config.json"
            
            if not config_path.exists():
                continue
            
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                log = load_log_with_safetensors(run_path)
                
                # Extract task centers from config
                task_centers = config.get('eval', {}).get('task_centers', [])
                
                # Store loaded data
                loaded_data['logs'][run_label] = log
                loaded_data['metadata'][run_label] = (config, task_centers)
                actual_run_labels.append(run_label)
                
            except Exception as e:
                print(f"Warning: Failed to load data for {run_path}: {e}")
                raise e
                continue
        
        else:
            print(f"Warning: {run_path} is neither a valid run nor multirun directory")
            continue
    
    loaded_data['run_labels'] = actual_run_labels
    return loaded_data


def load_all_logs_with_param_optimization(run_paths: list, run_labels: list = None, optimize_params: list = None, baseline_type: str = 'Ridge') -> dict:
    """Load all log files with parameter optimization for multirun experiments.
    
    Args:
        run_paths: List of Path objects pointing to runs or multirun subdirs
        run_labels: Custom labels for runs (optional)
        optimize_params: List of parameters to optimize over
        baseline_type: 'Ridge' or 'True' for MSE baseline type
    
    Returns:
        dict: Same format as load_all_logs but with optimized parameter combinations
    """
    print(f"Performing parameter optimization over: {optimize_params} using baseline: {baseline_type}")
    if not optimize_params:
        return load_all_logs(run_paths, run_labels)
    
    loaded_data = {
        'logs': {},
        'metadata': {},
        'run_labels': []
    }

    FAST = False
    
    actual_run_labels = []

    
    for i, run_path in enumerate(run_paths):
        run_path = Path(run_path)

        # Determine run label
        if run_labels and i < len(run_labels):
            run_label = run_labels[i]
        else:
            run_label = run_path.name
        
        # Check if this is a multirun directory
        if (run_path / "multirun.yaml").exists():
            # This is a multirun directory - apply parameter optimization
            subdirs = find_valid_multirun_subdirs(run_path, return_paths=True)
            swept_params = extract_swept_params(run_path)
            
            if not swept_params:
                print(f"Warning: No swept parameters found in {run_path}/multirun.yaml")
                continue
            
            # Validate that optimize_params exist in swept_params
            invalid_params = [param for param in optimize_params if param not in swept_params]
            if invalid_params:
                print(f"Error: Parameter optimization requested for parameters not swept in multirun: {invalid_params}")
                print(f"Available swept parameters: {swept_params}")
                continue
            
            # Collect all run data
            all_runs = []
            for j, subdir in enumerate(subdirs):
                config_path = subdir / "config.json"
                if not config_path.exists():
                    continue
                if FAST and j > 4:
                    print("Fast mode: only loading first 5 subruns")
                    break
                
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    log = load_log_with_safetensors(subdir)
                    
                    all_runs.append({
                        'config': config,
                        'log': log,
                        'path': subdir,
                        'name': subdir.name
                    })
                except Exception as e:
                    print(f"Warning: Failed to load run {subdir}: {e}")
                    raise e
                    continue
            
            # Group runs by non-optimized parameters
            run_groups = group_runs_by_other_params(all_runs, optimize_params, swept_params)
            
            # Create AUC cache for performance optimization
            auc_cache = {}
            
            # For each group, find best parameter combination
            for group_key, run_group in run_groups.items():
                best_run, best_auc, best_param_values = find_best_param_combination_by_auc(
                    run_group, optimize_params, baseline_type, auc_cache
                )
                
                if best_run is None:
                    continue
                
                # Create display name for this group using smart formatting
                group_label = format_parameter_legend(group_key, best_param_values, max_length=100)
                
                # Store the best run's data
                config = best_run['config']
                log = best_run['log']
                task_centers = config.get('eval', {}).get('task_centers', [])

                print(f"Eval steps:", log.get("eval/step", []))
                
                loaded_data['logs'][group_label] = log
                loaded_data['metadata'][group_label] = (config, task_centers)
                actual_run_labels.append(group_label)
        
        elif (run_path / "log.json").exists():
            # This is a single run - use standard processing
            try:
                config_path = run_path / "config.json"
                
                if not config_path.exists():
                    continue
                
                with open(config_path, 'r') as f:
                    config = json.load(f)
                log = load_log_with_safetensors(run_path)
                
                # Extract task centers from config
                task_centers = config.get('eval', {}).get('task_centers', [])
                
                # Store loaded data
                loaded_data['logs'][run_label] = log
                loaded_data['metadata'][run_label] = (config, task_centers)
                actual_run_labels.append(run_label)
                
            except Exception as e:
                print(f"Warning: Failed to load data for {run_path}: {e}")
                raise e
                continue
        
        else:
            print(f"Warning: {run_path} is neither a valid run nor multirun directory")
            continue
    
    loaded_data['run_labels'] = actual_run_labels
    return loaded_data


def process_loaded_data_for_baseline_with_prefixes(loaded_data: dict, baseline_type: str) -> dict[int, tuple[dict, dict, dict, dict, dict, dict, dict]]:
    """Process pre-loaded data for a specific baseline at multiple sequence length prefixes without any I/O.

    Args:
        loaded_data: Dictionary returned by load_all_logs()
        baseline_type: Either 'Ridge' or 'True' to specify which baseline to use

    Returns:
        dict: {prefix_length: (min_mse_dict, mean_mse_dict, end_mse_dict, min_std_dict, mean_std_dict, end_std_dict, selected_steps_dict)} where:
            prefix_length: Sequence length prefix (16, 32, 48, etc.)
            min_mse_dict: {run_label: [(task_center, min_mse, task_name), ...]}
            mean_mse_dict: {run_label: [(task_center, mean_mse, task_name), ...]}
            end_mse_dict: {run_label: [(task_center, end_mse, task_name), ...]}
            min_std_dict: {run_label: [(task_center, min_mse_std, task_name), ...]}
            mean_std_dict: {run_label: [(task_center, mean_mse_std, task_name), ...]}
            end_std_dict: {run_label: [(task_center, end_mse_std, task_name), ...]}
            selected_steps_dict: {run_label: {task_name: (min_step, mean_step, end_step)}}
    """
    # First, determine the sequence length of the data by examining one run
    max_seq_length = None
    for run_label in loaded_data['run_labels']:
        log = loaded_data['logs'][run_label]
        # Look for MSE data to determine sequence length
        for key, value in log.items():
            if key.startswith("eval/") and key != "eval/step":
                for metric_name, metric_values in value.items():
                    if f"Transformer | {baseline_type}" in metric_name and "(RelErr)" not in metric_name and metric_values is not None:
                        # Get the first non-empty MSE values to check length
                        first_values = normalize_error_values(metric_values[0])
                        if first_values is not None and len(first_values) > 0:
                            max_seq_length = len(first_values)
                            break
                if max_seq_length is not None:
                    break
            if max_seq_length is not None:
                break
        if max_seq_length is not None:
            break
    
    if max_seq_length is None:
        print(f"Warning: Could not determine sequence length for {baseline_type} baseline")
        return {}
    
    # Generate prefix lengths: multiples of 16 up to max_seq_length
    prefix_lengths = []
    for prefix in range(16, max_seq_length + 1, 16):
        prefix_lengths.append(prefix)
    
    # Also include the full length if it's not already included
    if max_seq_length not in prefix_lengths:
        prefix_lengths.append(max_seq_length)
    
    print(f"Processing {baseline_type} baseline for sequence prefixes: {prefix_lengths}")
    
    # Process data for each prefix length
    results_by_prefix = {}

    for prefix_length in prefix_lengths:
        min_mse_data = {}
        mean_mse_data = {}
        end_mse_data = {}
        min_std_data = {}
        mean_std_data = {}
        end_std_data = {}
        selected_steps_data = {}

        for run_label in loaded_data['run_labels']:
            log = loaded_data['logs'][run_label]
            config, task_centers = loaded_data['metadata'][run_label]

            # Extract MSE data for this specific prefix length
            try:
                min_mse_params, mean_mse_params, end_mse_params, selected_steps = extract_min_mse_params_for_baseline_with_prefix(
                    log, baseline_type, prefix_length, return_selected_steps=True
                )

                # Try to extract std data - this may fail if std data is not available
                try:
                    min_std_params, mean_std_params, end_std_params, _ = extract_std_mse_params_for_baseline_with_prefix(
                        log, baseline_type, prefix_length, return_selected_steps=True
                    )
                    has_std_data = True
                except Exception:
                    # Std data not available, create empty dicts
                    min_std_params, mean_std_params, end_std_params = {}, {}, {}
                    has_std_data = False

                min_mse_run_data = []
                mean_mse_run_data = []
                end_mse_run_data = []
                min_std_run_data = []
                mean_std_run_data = []
                end_std_run_data = []

                # Add Test tasks (task center = 0)
                if "Test tasks" in min_mse_params:
                    min_mse = min_mse_params["Test tasks"]
                    mean_mse = mean_mse_params.get("Test tasks", 0)
                    end_mse = end_mse_params.get("Test tasks", 0)
                    min_mse_run_data.append((0.0, min_mse, "Test tasks"))
                    mean_mse_run_data.append((0.0, mean_mse, "Test tasks"))
                    end_mse_run_data.append((0.0, end_mse, "Test tasks"))

                    if has_std_data:
                        min_std = min_std_params.get("Test tasks", 0.0)
                        mean_std = mean_std_params.get("Test tasks", 0.0)
                        end_std = end_std_params.get("Test tasks", 0.0)
                        min_std_run_data.append((0.0, min_std, "Test tasks"))
                        mean_std_run_data.append((0.0, mean_std, "Test tasks"))
                        end_std_run_data.append((0.0, end_std, "Test tasks"))
                    else:
                        min_std_run_data.append((0.0, 0.0, "Test tasks"))
                        mean_std_run_data.append((0.0, 0.0, "Test tasks"))
                        end_std_run_data.append((0.0, 0.0, "Test tasks"))

                # Add Fixed tasks
                for task_center in task_centers:
                    task_name = f"Fixed task {task_center}"
                    if task_name in min_mse_params:
                        min_mse = min_mse_params[task_name]
                        mean_mse = mean_mse_params.get(task_name, 0)
                        end_mse = end_mse_params.get(task_name, 0)
                        min_mse_run_data.append((task_center, min_mse, task_name))
                        mean_mse_run_data.append((task_center, mean_mse, task_name))
                        end_mse_run_data.append((task_center, end_mse, task_name))

                        if has_std_data:
                            min_std = min_std_params.get(task_name, 0.0)
                            mean_std = mean_std_params.get(task_name, 0.0)
                            end_std = end_std_params.get(task_name, 0.0)
                            min_std_run_data.append((task_center, min_std, task_name))
                            mean_std_run_data.append((task_center, mean_std, task_name))
                            end_std_run_data.append((task_center, end_std, task_name))
                        else:
                            min_std_run_data.append((task_center, 0.0, task_name))
                            mean_std_run_data.append((task_center, 0.0, task_name))
                            end_std_run_data.append((task_center, 0.0, task_name))

                if min_mse_run_data:
                    min_mse_data[run_label] = min_mse_run_data
                    mean_mse_data[run_label] = mean_mse_run_data
                    end_mse_data[run_label] = end_mse_run_data
                    min_std_data[run_label] = min_std_run_data
                    mean_std_data[run_label] = mean_std_run_data
                    end_std_data[run_label] = end_std_run_data
                    selected_steps_data[run_label] = selected_steps

            except Exception as e:
                print(f"Warning: Failed to process {baseline_type} baseline for {run_label} at prefix {prefix_length}: {e}")
                continue

        results_by_prefix[prefix_length] = (min_mse_data, mean_mse_data, end_mse_data, min_std_data, mean_std_data, end_std_data, selected_steps_data)
    
    return results_by_prefix


def plot_min_mse_analysis(run_paths: list, output_dir: Path = None, run_labels: list = None, optimize_params: list = None):
    """Plot minimum MSE vs task shift and mean MSE over context length for last iteration for multiple runs.
    
    Args:
        run_paths: List of Path objects pointing to runs or multirun subdirs
        output_dir: Directory to save plots (optional)
        run_labels: Custom labels for runs (optional)
        optimize_params: List of parameters to optimize over (e.g., ['task.n_tasks', 'train.clip_max_norm'])
    """
    if not run_paths:
        print("No run paths provided for minimum MSE analysis")
        return
    
    # PHASE 1: Load all data once (eliminates duplicate I/O)
    print("Loading all log files...")
    if optimize_params:
        # Parameter optimization mode - load and optimize parameter combinations for both baselines
        print("Loading with Ridge baseline optimization...")
        #ridge_loaded_data = load_all_logs_with_param_optimization(run_paths, run_labels, optimize_params, 'Ridge')
        ridge_loaded_data = {'logs': {}, 'metadata': {}, 'run_labels': []}  # Empty Ridge data
        print("Loading with True baseline optimization...")  
        true_loaded_data = load_all_logs_with_param_optimization(run_paths, run_labels, optimize_params, 'True')
    else:
        # Standard mode
        loaded_data = load_all_logs(run_paths, run_labels)
        ridge_loaded_data = loaded_data
        true_loaded_data = loaded_data
    
    if not ridge_loaded_data['logs'] and not true_loaded_data['logs']:
        print("No valid log data found")
        return
    
    # PHASE 2: Process data for each baseline at multiple sequence length prefixes (no I/O, uses cached logs)
    print("Processing Ridge baseline with multiple prefixes...")
    ridge_results_by_prefix = process_loaded_data_for_baseline_with_prefixes(ridge_loaded_data, 'Ridge')
    
    print("Processing True baseline with multiple prefixes...")
    true_results_by_prefix = process_loaded_data_for_baseline_with_prefixes(true_loaded_data, 'True')
    
    # PHASE 3: Free memory by clearing loaded data
    try:
        del ridge_loaded_data
        del true_loaded_data
    except:
        pass  # Ignore deletion errors
    
    # Determine which plots to create for each prefix
    create_ridge_plots = bool(ridge_results_by_prefix)
    create_true_plots = bool(true_results_by_prefix)
    
    if not create_ridge_plots and not create_true_plots:
        print("No valid data found for minimum MSE analysis")
        return
    
    # Get all prefix lengths
    all_prefixes = set()
    if ridge_results_by_prefix:
        all_prefixes.update(ridge_results_by_prefix.keys())
    if true_results_by_prefix:
        all_prefixes.update(true_results_by_prefix.keys())
    
    if not all_prefixes:
        print("No sequence length prefixes found")
        return
    
    sorted_prefixes = sorted(all_prefixes)
    print(f"Creating plots for sequence length prefixes: {sorted_prefixes}")
    
    # Helper function to create a plot for a specific baseline and prefix
    def create_mse_plot(min_mse_data, mean_mse_data, end_mse_data, min_std_data, mean_std_data, end_std_data, selected_steps_data, baseline_type: str, fig_suffix: str):
        # Extract prefix length from fig_suffix (e.g., 'ridge_prefix_32' -> 32)
        prefix_length = None
        if 'prefix_' in fig_suffix:
            try:
                prefix_length = int(fig_suffix.split('prefix_')[1])
            except (IndexError, ValueError):
                pass
        # Create the plot with three subplots in a separate figure
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))
        
        # Choose colormap based on number of runs
        num_runs = len(min_mse_data)
        if num_runs > 10:
            # Use colormap for many runs
            cmap = plt.get_cmap('tab20' if num_runs <= 20 else 'hsv')
            colors = [cmap(i / num_runs) for i in range(num_runs)]
        else:
            # Use discrete colors for few runs
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
        
        max_shift = float('inf')
        
        # Plot minimum MSE data (left subplot)
        for i, (run_label, run_data) in enumerate(min_mse_data.items()):
            if not run_data:
                continue

            # Sort by task center
            run_data.sort(key=lambda x: x[0])

            task_centers = [x[0] for x in run_data if x[0] <= max_shift]
            min_mses = [x[1] for x in run_data if x[0] <= max_shift]

            color = colors[i] if i < len(colors) else colors[i % len(colors)]

            # Create label with selected step information for min MSE
            steps_info = selected_steps_data.get(run_label, {})
            if steps_info:
                # Get a representative step for min MSE (use first task's min step)
                first_task_steps = next(iter(steps_info.values()), (None, None))
                min_step = first_task_steps[0]
                label_with_step = f"{run_label} (best step: {min_step})" if min_step is not None else run_label
            else:
                label_with_step = run_label

            # Get corresponding std data for shaded region
            std_run_data = min_std_data.get(run_label, [])
            if std_run_data:
                std_run_data.sort(key=lambda x: x[0])
                min_stds = [x[1] for x in std_run_data if x[0] <= max_shift]

                # Create upper and lower bounds for shaded region
                if len(min_stds) == len(min_mses):
                    upper_bounds = [max(1e-10, mse + std) for mse, std in zip(min_mses, min_stds)]
                    lower_bounds = [max(1e-10, mse - std) for mse, std in zip(min_mses, min_stds)]

                    # Add shaded region
                    ax1.fill_between(task_centers, lower_bounds, upper_bounds,
                                   alpha=0.2, color=color)

            # Plot minimum MSE vs task center
            ax1.plot(task_centers, min_mses, 'o-', color=color, linewidth=2,
                    markersize=6, label=label_with_step)
        
        # Configure minimum MSE plot with prefix information
        ax1.set_xlabel("Task Center (Task Shift)")
        ax1.set_ylabel(f"Best MSE vs {baseline_type} Baseline (Optimal Iteration)")
        title_suffix = f" (Seq Len: {prefix_length})" if prefix_length else ""
        ax1.set_title(f"Best MSE vs {baseline_type} Baseline vs Task Shift{title_suffix}")
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Plot mean MSE over context length data (middle subplot)
        for i, (run_label, run_data) in enumerate(mean_mse_data.items()):
            if not run_data:
                continue

            # Sort by task center
            run_data.sort(key=lambda x: x[0])

            task_centers = [x[0] for x in run_data if x[0] <= max_shift]
            mean_mses = [x[1] for x in run_data if x[0] <= max_shift]

            color = colors[i] if i < len(colors) else colors[i % len(colors)]

            # Create label with selected step information for mean MSE
            steps_info = selected_steps_data.get(run_label, {})
            if steps_info:
                # Get a representative step for mean MSE (use first task's mean step)
                first_task_steps = next(iter(steps_info.values()), (None, None))
                mean_step = first_task_steps[1]
                label_with_step = f"{run_label} (best step: {mean_step})" if mean_step is not None else run_label
            else:
                label_with_step = run_label

            # Get corresponding std data for shaded region
            std_run_data = mean_std_data.get(run_label, [])
            if std_run_data:
                std_run_data.sort(key=lambda x: x[0])
                mean_stds = [x[1] for x in std_run_data if x[0] <= max_shift]

                # Create upper and lower bounds for shaded region
                if len(mean_stds) == len(mean_mses):
                    upper_bounds = [max(1e-10, mse + std) for mse, std in zip(mean_mses, mean_stds)]
                    lower_bounds = [max(1e-10, mse - std) for mse, std in zip(mean_mses, mean_stds)]

                    # Add shaded region
                    ax2.fill_between(task_centers, lower_bounds, upper_bounds,
                                   alpha=0.2, color=color)

            # Plot mean MSE over context length vs task center
            ax2.plot(task_centers, mean_mses, 'o-', color=color, linewidth=2,
                    markersize=6, label=label_with_step)
        
        # Configure mean MSE over context length plot with prefix information
        ax2.set_xlabel("Task Center (Task Shift)")
        ax2.set_ylabel(f"Mean MSE vs {baseline_type} Baseline (Optimal Iteration)")
        ax2.set_title(f"Mean MSE vs {baseline_type} Baseline vs Task Shift{title_suffix}")
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # Plot end MSE data (right subplot)
        for i, (run_label, run_data) in enumerate(end_mse_data.items()):
            if not run_data:
                continue

            # Sort by task center
            run_data.sort(key=lambda x: x[0])

            task_centers = [x[0] for x in run_data if x[0] <= max_shift]
            end_mses = [x[1] for x in run_data if x[0] <= max_shift]

            color = colors[i] if i < len(colors) else colors[i % len(colors)]

            # Create label with selected step information for end MSE
            steps_info = selected_steps_data.get(run_label, {})
            if steps_info:
                # Get a representative step for end MSE (use first task's end step)
                first_task_steps = next(iter(steps_info.values()), (None, None, None))
                end_step = first_task_steps[2] if len(first_task_steps) > 2 else None
                label_with_step = f"{run_label} (best step: {end_step})" if end_step is not None else run_label
            else:
                label_with_step = run_label

            # Get corresponding std data for shaded region
            std_run_data = end_std_data.get(run_label, [])
            if std_run_data:
                std_run_data.sort(key=lambda x: x[0])
                end_stds = [x[1] for x in std_run_data if x[0] <= max_shift]

                # Create upper and lower bounds for shaded region
                if len(end_stds) == len(end_mses):
                    upper_bounds = [max(1e-10, mse + std) for mse, std in zip(end_mses, end_stds)]
                    lower_bounds = [max(1e-10, mse - std) for mse, std in zip(end_mses, end_stds)]

                    # Add shaded region
                    ax3.fill_between(task_centers, lower_bounds, upper_bounds,
                                   alpha=0.2, color=color)

            # Plot end MSE vs task center
            ax3.plot(task_centers, end_mses, 'o-', color=color, linewidth=2,
                    markersize=6, label=label_with_step)
        
        # Configure end MSE plot with prefix information
        ax3.set_xlabel("Task Center (Task Shift)")
        ax3.set_ylabel(f"End MSE vs {baseline_type} Baseline (Optimal Iteration)")
        ax3.set_title(f"End MSE vs {baseline_type} Baseline vs Task Shift{title_suffix}")
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
        
        # Add legend below plots
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles3, labels3 = ax3.get_legend_handles_labels()
        
        # Use handles and labels from first subplot (they should be the same)
        fig.legend(handles1, labels1, loc='lower center', ncol=min(len(labels1), 4), 
                  bbox_to_anchor=(0.5, -0.02))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        
        # Save plot
        if output_dir is None:
            # Check if we're analyzing multiruns by looking at the first run path
            if run_paths and (run_paths[0] / "multirun.yaml").exists():
                output_dir_to_use = run_paths[0]  # Use the multirun directory
            else:
                output_dir_to_use = Path("outputs")
        else:
            output_dir_to_use = output_dir
            
        output_path = output_dir_to_use / f"min_mse_analysis_{fig_suffix}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Minimum MSE analysis ({baseline_type}) plot saved to: {output_path}")
        
        plt.show()
        
        # Print summary for this baseline
        print(f"\n=== Minimum MSE Analysis Summary ({baseline_type} Baseline) ===")
        for run_label, run_data in min_mse_data.items():
            if not run_data:
                continue
            print(f"\n{run_label}:")
            mean_data = mean_mse_data.get(run_label, [])
            end_data = end_mse_data.get(run_label, [])
            min_std_data_run = min_std_data.get(run_label, [])
            mean_std_data_run = mean_std_data.get(run_label, [])
            end_std_data_run = end_std_data.get(run_label, [])

            mean_dict = {x[2]: x[1] for x in mean_data}  # task_name -> mean_mse
            end_dict = {x[2]: x[1] for x in end_data}  # task_name -> end_mse
            min_std_dict = {x[2]: x[1] for x in min_std_data_run}  # task_name -> min_mse_std
            mean_std_dict = {x[2]: x[1] for x in mean_std_data_run}  # task_name -> mean_mse_std
            end_std_dict = {x[2]: x[1] for x in end_std_data_run}  # task_name -> end_mse_std

            for task_center, min_mse, task_name in sorted(run_data, key=lambda x: x[0]):
                mean_mse = mean_dict.get(task_name, "N/A")
                end_mse = end_dict.get(task_name, "N/A")
                min_std = min_std_dict.get(task_name, 0.0)
                mean_std = mean_std_dict.get(task_name, 0.0)
                end_std = end_std_dict.get(task_name, 0.0)

                # Format std values - show as 0.000000 if they are zero (no std data available)
                if isinstance(min_std, (int, float)) and min_std > 0:
                    min_std_str = f"±{min_std:.6f}"
                else:
                    min_std_str = "±0.000000"

                if isinstance(mean_std, (int, float)) and mean_std > 0:
                    mean_std_str = f"±{mean_std:.6f}"
                else:
                    mean_std_str = "±0.000000"

                if isinstance(end_std, (int, float)) and end_std > 0:
                    end_std_str = f"±{end_std:.6f}"
                else:
                    end_std_str = "±0.000000"

                print(f"  {task_name} (center={task_center}): min_mse={min_mse:.6f}{min_std_str}, mean_mse={mean_mse:.6f}{mean_std_str}, end_mse={end_mse:.6f}{end_std_str}")
    
    # Create plots for each prefix and available baselines
    for prefix_length in sorted_prefixes:
        print(f"\nCreating plots for sequence prefix length: {prefix_length}")
        
        # Create Ridge plots for this prefix
        if create_ridge_plots and prefix_length in ridge_results_by_prefix:
            ridge_min_data, ridge_mean_data, ridge_end_data, ridge_min_std_data, ridge_mean_std_data, ridge_end_std_data, ridge_steps_data = ridge_results_by_prefix[prefix_length]
            if ridge_min_data:  # Only create plot if we have data
                create_mse_plot(ridge_min_data, ridge_mean_data, ridge_end_data, ridge_min_std_data, ridge_mean_std_data, ridge_end_std_data, ridge_steps_data,
                              'Ridge', f'ridge_prefix_{prefix_length}')

        # Create True plots for this prefix
        if create_true_plots and prefix_length in true_results_by_prefix:
            true_min_data, true_mean_data, true_end_data, true_min_std_data, true_mean_std_data, true_end_std_data, true_steps_data = true_results_by_prefix[prefix_length]
            if true_min_data:  # Only create plot if we have data
                create_mse_plot(true_min_data, true_mean_data, true_end_data, true_min_std_data, true_mean_std_data, true_end_std_data, true_steps_data,
                              'True', f'true_prefix_{prefix_length}')
