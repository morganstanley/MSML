import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from loading import load_log_with_safetensors
from task_shift import normalize_error_values, find_valid_multirun_subdirs
from mean_min_best_mse import load_all_logs, load_all_logs_with_param_optimization

def format_task_name_for_display(task_name, metric_name):
    """Format task name for display in legends, replacing 'Fixed task' with 'Shifted task'."""
    if task_name.startswith("Fixed task"):
        task_name = task_name.replace("Fixed task", "Shifted task")
    splitted_metric = metric_name.split(" | ")
    name = f"{task_name} ({splitted_metric[0]})"
    return name

def plot_icl_for_all_steps(log: dict, run_id: str, output_dir: Path = None):
    """Plot ICL performance for all evaluation steps and save each step as a separate file."""
    eval_steps = log.get("eval/step", [])
    if not eval_steps:
        print("No evaluation steps found in log")
        return
    
    # Extract evaluation metrics
    eval_metrics = {}
    for key, value in log.items():
        if key.startswith("eval/") and key != "eval/step":
            task_name = key.split("/")[1]
            if task_name not in eval_metrics:
                eval_metrics[task_name] = {}
            for metric_name, metric_values in value.items():
                eval_metrics[task_name][metric_name] = metric_values
    
    # Check which baselines are available
    ridge_available = False
    true_available = False
    
    for task_name, metrics in eval_metrics.items():
        for metric_name in metrics.keys():
            if "Transformer | Ridge" in metric_name:
                ridge_available = True
            if "Transformer | True" in metric_name:
                true_available = True
    
    if not ridge_available and not true_available:
        print("No Transformer baseline metrics found in log")
        return
    
    # Helper function to create plots for a specific baseline
    def create_icl_plots_for_baseline(baseline_type: str, baseline_suffix: str):
        # Create output directories for ICL plots
        if output_dir is None:
            base_output_dir = Path("outputs") / run_id
        else:
            base_output_dir = output_dir
            
        icl_mse_dir = base_output_dir / f"icl_plots_mse_{baseline_suffix}"
        icl_rel_err_dir = base_output_dir / f"icl_plots_rel_err_{baseline_suffix}"
        icl_mse_dir.mkdir(exist_ok=True)
        icl_rel_err_dir.mkdir(exist_ok=True)
        
        # Colors for different tasks
        n_curves = len(eval_metrics)
        colors = cm.get_cmap('tab20', n_curves)  # or 'tab10', 'Set3', 'hsv', etc.
        linestyles = ['-', '--', '-.', ':']
        
        # Generate plots for each evaluation step
        for step_idx, eval_step in enumerate(eval_steps):
            
            # MSE Plot
            plt.figure(figsize=(12, 8))
            color_idx = 0
            
            for task_name, metrics in eval_metrics.items():
                linestyle_idx = 0
                for metric_name, values in metrics.items():
                    # print(f"Processing {task_name} - {metric_name}...", end=" ")
                    if f" | {baseline_type}" in metric_name and "(RelErr)" not in metric_name and "(Std)" not in metric_name and values and step_idx < len(values):
                        # print("Found relevant metric.")
                        # Get MSE by position for this step
                        mse_by_position = normalize_error_values(values[step_idx])  # List of MSE values by position
                        n_points = len(mse_by_position)
                        positions = list(range(1, n_points + 1))  # Context length positions
                        
                        plt.plot(positions, mse_by_position,
                                color=colors(color_idx),
                                linewidth=2,
                                #marker='.,
                                #markersize=1,
                                linestyle=linestyles[linestyle_idx % len(linestyles)],
                                label=f"{format_task_name_for_display(task_name, metric_name)}")
                        linestyle_idx += 1
                    else:
                        #print("Metric not relevant or data missing.")
                        pass
                color_idx += 1
            
            plt.xlabel("Context Length (Position)")
            plt.ylabel(f"MSE (Transformer vs {baseline_type})")
            plt.title(f"ICL MSE vs {baseline_type} Baseline at Step {eval_step} - {run_id}")
            plt.grid(True, alpha=0.3)
            plt.yscale('log')
            #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.legend(loc='lower left', bbox_to_anchor=(0, 1), fontsize='small')
            plt.tight_layout()
            
            # Save MSE plot for this step
            output_path = icl_mse_dir / f"icl_step_{eval_step:04d}.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()  # Close to save memory
            
            # Relative Error Plot
            plt.figure(figsize=(12, 8))
            color_idx = 0
            
            for task_name, metrics in eval_metrics.items():
                for metric_name, values in metrics.items():
                    if f"Transformer | {baseline_type} (RelErr)" in metric_name and values and step_idx < len(values)  and "(Std)" not in metric_name:
                        # Get Relative Error by position for this step
                        rel_err_by_position = normalize_error_values(values[step_idx])  # List of RelErr values by position
                        n_points = len(rel_err_by_position)
                        positions = list(range(1, n_points + 1))  # Context length positions
                        
                        plt.plot(positions, rel_err_by_position,
                                color=colors(color_idx),
                                linewidth=2,
                                marker='o',
                                markersize=6,
                                label=f"{format_task_name_for_display(task_name, metric_name)}")
                        color_idx += 1
            
            plt.xlabel("Context Length (Position)")
            plt.ylabel(f"Relative Error (Transformer vs {baseline_type})")
            plt.title(f"ICL Relative Error vs {baseline_type} Baseline at Step {eval_step} - {run_id}")
            plt.grid(True, alpha=0.3)
            plt.yscale('log')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            # Save Relative Error plot for this step
            output_path = icl_rel_err_dir / f"icl_step_{eval_step:04d}.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()  # Close to save memory
        
        print(f"ICL MSE plots ({baseline_type} baseline) for {len(eval_steps)} steps saved to: {icl_mse_dir}")
        print(f"ICL Relative Error plots ({baseline_type} baseline) for {len(eval_steps)} steps saved to: {icl_rel_err_dir}")
    
    # Create plots for available baselines
    if ridge_available:
        create_icl_plots_for_baseline('Ridge', 'ridge')
    
    if true_available:
        create_icl_plots_for_baseline('True', 'true')

