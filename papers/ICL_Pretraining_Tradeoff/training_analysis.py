#!/usr/bin/env python3
"""
Training analysis functionality - training loss plots, MSE curve fitting, and summaries.
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit

from task_shift import normalize_error_values, icl_power_law
from icl_plots import format_task_name_for_display


def plot_training_loss(log: dict, run_id: str, output_dir: Path = None):
    """Plot training loss over steps."""
    steps = log["train/step"]
    eval_steps = log.get("eval/step", [])
    lr_values = log["train/lr"]
    train_losses = log.get("train/loss", [])
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 18))
    
    # Plot training loss
    if train_losses:
        axes[0].plot(steps, train_losses, 'r-', linewidth=2)
        axes[0].set_xlabel("Training Step")
        axes[0].set_ylabel("Training Loss")
        axes[0].set_title(f"Training Loss - {run_id}")
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')
    
    # Plot evaluation metrics
    eval_metrics = {}
    for key, value in log.items():
        if key.startswith("eval/") and key != "eval/step":
            task_name = key.split("/")[1]
            if task_name not in eval_metrics:
                eval_metrics[task_name] = {}
            for metric_name, metric_values in value.items():
                eval_metrics[task_name][metric_name] = metric_values
    
    # Plot MSE for each task and baseline
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    color_idx_mse = 0
    color_idx_rel = 0
    
    for task_name, metrics in eval_metrics.items():
        for metric_name, values in metrics.items():
            if "Transformer |" in metric_name and "(RelErr)" not in metric_name:
                # MSE metrics (exclude relative error)
                mean_values = [np.mean(normalize_error_values(v)) for v in values]
                axes[1].plot(eval_steps, mean_values, 
                        color=colors[color_idx_mse % len(colors)], 
                        linewidth=2,
                        label=f"{format_task_name_for_display(task_name, metric_name)}: {metric_name}")
                color_idx_mse += 1
    
    # Plot min MSE over context length as function of training step
    for task_name, metrics in eval_metrics.items():
        for metric_name, values in metrics.items():
            if "Transformer |" in metric_name and "(RelErr)" not in metric_name:
                # Calculate min MSE over context length for each training step
                min_values = [np.min(normalize_error_values(v)) for v in values]
                axes[2].plot(eval_steps, min_values, 
                        color=colors[color_idx_rel % len(colors)], 
                        linewidth=2,
                        label=f"{format_task_name_for_display(task_name, metric_name)}: {metric_name}")
                color_idx_rel += 1
    
    # Configure MSE plot (axes[1])
    axes[1].set_xlabel("Training Step")
    axes[1].set_ylabel("Mean Squared Error")
    axes[1].set_title(f"MSE Evaluation Metrics - {run_id}")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Configure Min MSE plot (axes[2])
    axes[2].set_xlabel("Training Step")
    axes[2].set_ylabel("Min MSE over Context Length")
    axes[2].set_title(f"Min MSE over Context Length - {run_id}")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_yscale('log')
    axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Save plot
    if output_dir is None:
        output_dir = Path("outputs") / run_id
    output_path = output_dir / "training_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    plt.show()


def fit_mse_curves_and_compute_metrics(log: dict, run_id: str):
    """Fit MSE curves and compute ICL performance metrics."""
    eval_steps = log.get("eval/step", [])
    if not eval_steps:
        print("No evaluation steps found in log")
        return
    
    # Extract evaluation metrics for the final step
    final_step_idx = -1
    eval_metrics = {}
    for key, value in log.items():
        if key.startswith("eval/") and key != "eval/step":
            task_name = key.split("/")[1]
            if task_name not in eval_metrics:
                eval_metrics[task_name] = {}
            for metric_name, metric_values in value.items():
                eval_metrics[task_name][metric_name] = metric_values
    
    print(f"\n=== ICL Performance Metrics Analysis: {run_id} ===")
    print("Fitting MSE curves with formula: D/(k+1)^alpha + C")
    print("where k is context length (0-indexed), D = init error at k=0 - C")
    
    for task_name, metrics in eval_metrics.items():
        print(f"\n{task_name}:")
        
        # Look for preferred metric (Ridge) first, then fallback to True
        preferred_metric = None
        fallback_metric = None
        
        for metric_name, values in metrics.items():
            if "(RelErr)" not in metric_name and values:
                if "Transformer | Ridge" in metric_name:
                    preferred_metric = (metric_name, values)
                elif "Transformer | True" in metric_name:
                    fallback_metric = (metric_name, values)
        
        # Use preferred metric if available, otherwise fallback
        selected_metric = preferred_metric or fallback_metric
        
        if selected_metric:
            metric_name, values = selected_metric
            # Get MSE values for final step
            mse_values = normalize_error_values(values[final_step_idx])
            if mse_values is None or len(mse_values) < 3:
                continue
                
            k = np.arange(len(mse_values))  # Context lengths: 0, 1, 2, ...
            
            try:
                # Fit the power law curve
                # Initial guess: D = mse_values[0] - min(mse_values), alpha = 1, C = min(mse_values)
                initial_guess = [mse_values[0] - np.min(mse_values), 1.0, np.min(mse_values)]
                
                popt, pcov = curve_fit(icl_power_law, k, mse_values, p0=initial_guess, 
                                     bounds=([0, 0, 0], [np.inf, np.inf, np.inf]), maxfev=5000)
                
                D_fit, alpha_fit, C_fit = popt
                
                # Compute metrics
                avg_performance = np.mean(mse_values)
                
                # Compute R-squared
                y_pred = icl_power_law(k, D_fit, alpha_fit, C_fit)
                ss_res = np.sum((mse_values - y_pred) ** 2)
                ss_tot = np.sum((mse_values - np.mean(mse_values)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                print(f"  {metric_name}:")
                print(f"    Average MSE over k: {avg_performance:.6f}")
                print(f"    Power law fit - alpha: {alpha_fit:.4f}, C: {C_fit:.6f}")
                print(f"    D (init error - C): {D_fit:.6f}")
                print(f"    R²: {r_squared:.4f}")
                
            except Exception as e:
                print(f"  {metric_name}: Failed to fit curve - {str(e)}")
                avg_performance = np.mean(mse_values)
                print(f"    Average MSE over k: {avg_performance:.6f}")


def print_summary(log: dict, run_id: str):
    """Print a summary of the training run."""
    steps = log["train/step"]
    lr_values = log["train/lr"]
    train_losses = log.get("train/loss", [])
    
    print(f"\n=== Training Summary: {run_id} ===")
    print(f"Total steps: {steps[-1] if steps else 0}")
    print(f"Final learning rate: {lr_values[-1]:.2e}" if lr_values else "N/A")
    if train_losses:
        print(f"Final training loss: {train_losses[-1]:.6f}")
        print(f"Initial training loss: {train_losses[0]:.6f}")
        if len(train_losses) > 1:
            improvement = (train_losses[0] - train_losses[-1]) / train_losses[0] * 100
            print(f"Loss improvement: {improvement:.1f}%")
    
    # Print final evaluation metrics
    print("\nFinal Evaluation Metrics:")
    for key, value in log.items():
        if key.startswith("eval/") and key != "eval/step":
            task_name = key.split("/")[1]
            print(f"\n{task_name}:")
            # Look for preferred metric (Ridge) first, then fallback to any Transformer metric
            preferred_metrics = []
            fallback_metrics = []
            
            for metric_name, metric_values in value.items():
                if metric_values:
                    if "Transformer | Ridge" in metric_name:
                        preferred_metrics.append((metric_name, metric_values))
                    elif "Transformer |" in metric_name:
                        fallback_metrics.append((metric_name, metric_values))
            
            # Show preferred metrics first, then fallback metrics
            metrics_to_show = preferred_metrics if preferred_metrics else fallback_metrics
            
            for metric_name, metric_values in metrics_to_show:
                final_mse = np.mean(normalize_error_values(metric_values[-1]))
                print(f"  {metric_name}: {final_mse:.6f}")
