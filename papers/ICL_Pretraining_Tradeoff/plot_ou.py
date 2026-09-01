#!/usr/bin/env python3
"""
Plot Ornstein-Uhlenbeck process samples using task.py

Usage:
    python plot_ou.py --config-path=icl/configs --config-name=fast
    python plot_ou.py --config-path=icl/configs --config-name=fast task.batch_size=16
"""

import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import jax
import jax.numpy as jnp

# Import utils first to avoid circular import issues
import icl.utils
from icl.tasks import get_task


@hydra.main(version_base=None, config_path="icl/configs", config_name="fast")
def main(cfg: DictConfig) -> None:
    """Plot OU samples for each evaluation task."""
    
    # Create task from config
    task_config = cfg.task
    task = get_task(
        name=task_config.name,
        n_tasks=task_config.n_tasks,
        n_data=task_config.n_data,
        n_dims=task_config.n_dims,
        n_points=task_config.n_points,
        n_max_points=task_config.n_max_points,
        batch_size=task_config.batch_size,
        data_seed=task_config.data_seed,
        task_seed=task_config.task_seed,
        noise_seed=task_config.noise_seed,
        data_scale=task_config.data_scale,
        task_scale=task_config.task_scale,
        noise_scale=task_config.noise_scale,
        dtype=jnp.float32,
        clip=task_config.clip,
        use_weights=task_config.use_weights,
        distrib_name=task_config.distrib_name,
        distrib_param=task_config.distrib_param,
        ou_step=task_config.ou_step,
    )
    
    # Get evaluation tasks
    eval_tasks = task.get_default_eval_tasks(
        batch_size=cfg.eval.batch_size,
        task_seed=cfg.eval.task_seed,
        data_seed=cfg.eval.data_seed,
        noise_seed=cfg.eval.noise_seed,
        eval_n_points=cfg.eval.eval_n_points,
        task_centers=cfg.eval.task_centers,
    )
    
    print(f"Generated {len(eval_tasks)} evaluation tasks")
    
    # Create output directory
    output_dir = Path(cfg.output_dir) / "ou_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot samples for each evaluation task
    for task_idx, eval_task in enumerate(eval_tasks):
        print(f"Plotting task {task_idx + 1}/{len(eval_tasks)}: {eval_task.name}")
        
        # Sample a batch from this task
        data, tasks, weights, targets, attention_mask = eval_task.sample_batch(step=0)
        
        # Convert to numpy for plotting
        targets_np = np.array(targets)  # Shape: (batch_size, n_points, n_dims)
        batch_size, n_points, n_dims = targets_np.shape
        
        print(f"  Batch size: {batch_size}, Points: {n_points}, Dims: {n_dims}")
        
        # Create figure with subplots for each dimension
        fig, axes = plt.subplots(n_dims, 1, figsize=(12, 4 * n_dims))
        if n_dims == 1:
            axes = [axes]  # Make it iterable
        
        # Time points
        time_points = np.arange(n_points) * eval_task.ou_step
        
        # Get task parameters for individual trajectories
        tasks_np = np.array(tasks)  # Shape: (batch_size, 2*n_dims, 1)
        mu_values, theta_values = eval_task.get_params_from_tasks(tasks_np)

        # Get error when predicting the previous value
        error = jnp.mean(
                    jnp.square(targets_np[:, 1:, :] - targets_np[:, :-1, :]),
                )
        print(f"  Mean prediction error from previous point: {error:.4f}")

        # Plot each sample trajectory
        for dim in range(n_dims):
            ax = axes[dim]
            
            # Plot all batch samples for this dimension with μ parameter in label
            for batch_idx in range(batch_size):
                trajectory = targets_np[batch_idx, :, dim]
                mu_val = mu_values[batch_idx, dim]
                theta_val = theta_values[batch_idx, dim]
                
                # Plot trajectory
                ax.plot(time_points, trajectory, alpha=0.7, linewidth=1.5)
                color = ax.lines[-1].get_color()  # Get color of last line for legend

                # Plot mean parameter as horizontal dashed line
                ax.plot(time_points, mu_val * np.ones_like(time_points), '--', 
                       alpha=0.7, linewidth=1.5, 
                       label=f'μ={mu_val:.3f}, θ={theta_val:.3f}', color=color)
            
            ax.set_xlabel("Time")
            ax.set_ylabel(f"X_{dim+1}(t)")
            ax.set_title(f"{eval_task.name} - Dimension {dim+1}")
            ax.grid(True, alpha=0.3)
            if batch_size <= 8:  # Only show legend if not too many trajectories
                ax.legend()
        
        # Task parameters were already extracted above
        
        # Add parameter info as figure suptitle
        mu_mean = np.mean(mu_values, axis=0)
        theta_mean = np.mean(theta_values, axis=0)
        if n_dims == 1:
            param_info = f"μ: {mu_mean[0]:.3f}, θ: {theta_mean[0]:.3f}, dt: {eval_task.ou_step}"
        else:
            param_info = f"μ: {mu_mean}, θ: {theta_mean}, dt: {eval_task.ou_step}"
        fig.suptitle(f"{eval_task.name} - OU Process Samples\n{param_info}", fontsize=14)
        
        plt.tight_layout()
        
        # Save plot
        safe_name = eval_task.name.replace(" ", "_").replace(".", "_")
        output_path = output_dir / f"ou_samples_{safe_name}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved plot to: {output_path}")
        
        plt.close()  # Close to save memory
    
    print(f"\nAll plots saved to: {output_dir}")
    
    # Create a summary plot showing one trajectory from each task
    if len(eval_tasks) > 1:
        print("Creating summary plot...")
        
        fig, axes = plt.subplots(n_dims, 1, figsize=(12, 4 * n_dims))
        if n_dims == 1:
            axes = [axes]
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(eval_tasks)))
        
        for task_idx, eval_task in enumerate(eval_tasks):
            # Sample one trajectory from this task
            data, tasks, weights, targets, attention_mask = eval_task.sample_batch(step=0)
            targets_np = np.array(targets)
            
            # Time points for this task
            time_points = np.arange(targets_np.shape[1]) * eval_task.ou_step
            
            # Plot first sample from this task
            for dim in range(n_dims):
                ax = axes[dim]
                trajectory = targets_np[0, :, dim]  # First sample only
                ax.plot(time_points, trajectory, 
                       color=colors[task_idx], 
                       linewidth=2, 
                       label=eval_task.name,
                       alpha=0.8)
        
        # Configure summary plot
        for dim in range(n_dims):
            ax = axes[dim]
            ax.set_xlabel("Time")
            ax.set_ylabel(f"X_{dim+1}(t)")
            ax.set_title(f"OU Process Comparison - Dimension {dim+1}")
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        fig.suptitle("Ornstein-Uhlenbeck Process - All Evaluation Tasks", fontsize=16)
        plt.tight_layout()
        
        # Save summary plot
        summary_path = output_dir / "ou_summary_all_tasks.png"
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        print(f"Summary plot saved to: {summary_path}")
        
        plt.close()


if __name__ == "__main__":
    main()
