#!/usr/bin/env python
# coding=utf-8

import os
import sys
import json
import logging
import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Any, Tuple
import datetime
import random
from datasets import load_dataset
import torch
import gc

# Add utils directory to path
sys.path.append(str(Path(__file__).parent))
from utils.problem_solver import ProblemSolver
from utils.dataset_manager import copy_initial_problems, create_datasets, get_solved_problems
from utils.evaluator import ModelEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(Path(__file__).parent, 'bootstrap.log'))
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Run the bootstrap Q model improvement loop")
    
    # Source directories
    parser.add_argument(
        "--tasks-dir",
        type=str,
        default="../tasks",
        help="Path to the tasks directory"
    )
    parser.add_argument(
        "--leetcode-tasks-dir",
        type=str,
        default="../leet_code_tasks",
        help="Path to the LeetCode tasks directory"
    )
    
    # Model parameters
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen3-32B",
        help="Base model to start with"
    )
    parser.add_argument(
        "--pretrained-model-path",
        type=str,
        default="pretrained_model",
        help="Path to pretrained model directory (optional)"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="huggingface",
        choices=["openai", "anthropic", "huggingface"],
        help="Type of model"
    )
    
    # Loop parameters
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=30,
        help="Maximum number of bootstrap iterations"
    )
    parser.add_argument(
        "--target-problems",
        type=int,
        default=None,
        help="Target number of solved problems (default: all available)"
    )
    
    # Training parameters
    parser.add_argument(
        "--train-steps",
        type=int,
        default=1000,
        help="Number of training steps per iteration"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate for training"
    )
    
    # Misc
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for model inference (cuda or cpu)"
    )
    parser.add_argument(
        "--wandb-name-prefix",
        type=str,
        default="qwen-q-bootstrap",
        help="Prefix for W&B run name"
    )
    parser.add_argument(
        "--resume-iteration",
        type=int,
        default=0,
        help="Iteration to resume from (0 = start fresh)"
    )
    
    # Debug mode
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode (one problem, one training iteration, minimal steps)"
    )
    
    return parser.parse_args()

def train_model(
    model_name: str,
    train_file: str,
    eval_file: str,
    output_dir: str,
    training_args: Dict[str, Any]
) -> str:
    """
    Train a model on the given data.
    
    Args:
        model_name: Name of the base model
        train_file: Path to training data file
        eval_file: Path to evaluation data file
        output_dir: Output directory for the trained model
        training_args: Additional training arguments
    
    Returns:
        Path to the trained model
    """
    logger.info(f"Training model: {model_name} -> {output_dir}")
    
    # Use the SFT training script from the sft directory
    script_path = os.path.join(Path(__file__).parent.parent, "sft", "run_sft.py")
    
    # Check if script exists
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Training script not found: {script_path}")
    
    # Construct command with arguments for run_sft.py
    cmd = [
        "python",
        script_path,
        "--base_model", model_name,
        "--train_file", train_file,
        "--eval_file", eval_file,
        "--output_dir", output_dir,
        "--max_steps", str(training_args.get("max_steps", 1000)),
        "--learning_rate", str(training_args.get("learning_rate", 2e-5)),
        "--experiment_name", training_args.get("wandb_name", "qwen-q-bootstrap"),
        "--wandb"
    ]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    process = subprocess.run(
        cmd,
        check=True,
        text=True,
        capture_output=True,
        cwd=os.path.join(Path(__file__).parent.parent, "sft")  # Run from sft directory
    )
    logger.info(f"Training completed successfully")
    logger.info(f"Output: {process.stdout}")
    
    return output_dir
    

def main():
    args = parse_args()
    
    # Set up directories
    bootstrap_dir = Path(__file__).parent
    initial_problems_dir = bootstrap_dir / "new_initial_problems"
    iterations_dir = bootstrap_dir / "new_iterations"
    high_level_summary_dir = bootstrap_dir / "new_high_level_summary"
    
    # Create directories
    os.makedirs(initial_problems_dir, exist_ok=True)
    os.makedirs(iterations_dir, exist_ok=True)
    os.makedirs(high_level_summary_dir, exist_ok=True)
    
    # Load the primary dataset to determine the total universe of problems
    logger.info("Loading Hugging Face dataset to determine total problem space...")
    hf_dataset = load_dataset("greengerong/leetcode")['train']
    num_total_hf_problems = len(hf_dataset)
    logger.info(f"Found {num_total_hf_problems} problems in greengerong/leetcode dataset.")
    # This counts the number of description files *locally available* after copying.
    # These files are expected to correspond to problems from the HF dataset.
    locally_available_descriptions = len([f for f in os.listdir(initial_problems_dir) if f.endswith("_description.txt")])
    logger.info(f"Found {locally_available_descriptions} problem description files in {initial_problems_dir}.")

    # Set the target number of problems to solve
    if args.target_problems is not None:
        if args.target_problems > num_total_hf_problems:
            logger.warning(
                f"--target-problems ({args.target_problems}) is greater than "
                f"the total number of problems in the dataset ({num_total_hf_problems}). "
                f"Will use the dataset size as the target."
            )
            total_problems_to_solve = num_total_hf_problems
        else:
            total_problems_to_solve = args.target_problems
    else:
        total_problems_to_solve = num_total_hf_problems
    
    # In debug mode, we only aim to solve one problem
    if args.debug:
        logger.info("Running in DEBUG MODE - only one problem and one iteration will be processed")
        total_problems_to_solve = 1

    logger.info(f"Target: Solve {total_problems_to_solve} problems out of {num_total_hf_problems} available in the dataset.")
    
    # Initialize variables
    current_iteration = args.resume_iteration
    all_solved_problems = set()

    # Populate initially solved problems from the initial_problems_dir
    logger.info(f"Checking for initially solved problems in {initial_problems_dir}...")
    initially_solved_from_dir = set()
    if os.path.exists(initial_problems_dir):
        for filename in os.listdir(initial_problems_dir):
            if filename.endswith("_description.txt"):
                # Extract the part before the first underscore if present, or the whole name if no underscore
                # This assumes IDs like "1" or "123" and filenames like "1_description.txt" or "1_some_title_description.txt"
                base_name = filename.replace("_description.txt", "")
                problem_id = base_name.split('_')[0]
                initially_solved_from_dir.add(problem_id)
    
    if initially_solved_from_dir:
        logger.info(f"Found {len(initially_solved_from_dir)} problem(s) in {initial_problems_dir}, adding to all_solved_problems.")
        all_solved_problems.update(initially_solved_from_dir)
    else:
        logger.info(f"No pre-solved problems found in {initial_problems_dir}.")

    # Initialize model paths
    base_model = args.base_model
    trained_model = args.base_model
    solver_model = args.base_model
    q_base_acc = []
    q_trained_acc = []

    # Check for a pre-existing trained model directory if not resuming
    # This allows starting the bootstrap with an already fine-tuned model
    potential_pretrained_dir = Path(args.pretrained_model_path)
    
    if args.resume_iteration == 0 and potential_pretrained_dir.exists() and potential_pretrained_dir.is_dir() and any(potential_pretrained_dir.iterdir()):
        logger.info(f"Found existing pretrained model directory: {potential_pretrained_dir}")
        logger.info(f"This will override the --base-model ('{args.base_model}') for the initial state.")
        base_model = str(args.base_model)  # Training will start from this model
        trained_model = str(potential_pretrained_dir) # This is our current best model
        solver_model = str(potential_pretrained_dir)  # Solver will use this model
        logger.info(f"Initial base_model set to: {base_model}")
        logger.info(f"Initial trained_model set to: {trained_model}")
        logger.info(f"Initial solver_model set to: {solver_model}")
    elif args.resume_iteration == 0:
        logger.info(f"No pre-existing pretrained model directory found. Starting with base model: {args.base_model}")
    
    if current_iteration > 0:
        logger.info(f"Resuming from iteration {current_iteration}. Loading solved problems from previous bootstrap iterations...")
        for i in range(1, current_iteration + 1):
            iter_dir_resume = iterations_dir / f"iteration_{i}"
            solved_dir_resume = iter_dir_resume / "solved_problems"
            if os.path.exists(solved_dir_resume):
                iter_solved = get_solved_problems(str(solved_dir_resume))
                all_solved_problems.update(iter_solved)
            model_dir_path_resume = iter_dir_resume / "model"
            if os.path.exists(model_dir_path_resume) and os.listdir(model_dir_path_resume):
                pass
                # trained_model = str(model_dir_path_resume)
                # solver_model = str(model_dir_path_resume)
        logger.info(f"After loading from initial_problems_dir and previous iterations, total {len(all_solved_problems)} problems are marked as solved.")
        logger.info(f"Resuming with solver_model set to: {solver_model}")
    else:
        logger.info(f"Starting fresh. Total {len(all_solved_problems)} problems initially marked as solved (from {initial_problems_dir}).")
    
    while (len(all_solved_problems) < total_problems_to_solve and current_iteration < args.max_iterations):
        current_iteration += 1
        logger.info(f"Starting iteration {current_iteration}")
        
        iter_dir = iterations_dir / f"iteration_{current_iteration}"
        solved_dir = iter_dir / "solved_problems"
        eval_dir = iter_dir / "evaluation"
        model_dir = iter_dir / "model"
        
        os.makedirs(iter_dir, exist_ok=True)
        os.makedirs(solved_dir, exist_ok=True)
        os.makedirs(eval_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        
        metadata = {
            "iteration": current_iteration,
            "timestamp": datetime.datetime.now().isoformat(),
            "base_model": base_model,
            "trained_model": trained_model,
            "solver_model": solver_model,
            "problems_solved_before": len(all_solved_problems),
            "debug_mode": args.debug
        }
        with open(os.path.join(iter_dir, "metadata.json"), 'w') as f: json.dump(metadata, f, indent=2)
        
        newly_solved = set() # Initialize newly_solved for the iteration

        # Skip problem solving attempt only on the very first iteration of a fresh run

        logger.info(f"Attempting to solve problems for iteration {current_iteration}...")
        solver = ProblemSolver(model_name=solver_model, model_type=args.model_type, device=args.device)
        
        logger.info("Normal mode: Attempting all unsolved problems from the Hugging Face dataset.")
        ## TESTING
        
        newly_solved = solver.attempt_all_problems(
            hf_dataset=hf_dataset, 
            all_descriptions_dir=args.leetcode_tasks_dir, 
            output_dir=str(solved_dir),
            solved_problems=all_solved_problems,
            problem_id_column="id"
        )
        if newly_solved:
                logger.info(f"Solved {len(newly_solved)} new problems in iteration {current_iteration}")
                all_solved_problems.update(newly_solved)
        else:
            logger.info(f"No new problems solved in iteration {current_iteration} by the solver.")

        # Ensure newly_solved is a set, even if problem solving was skipped
        if not isinstance(newly_solved, set):
            newly_solved = set()
            
        metadata["problems_solved_new"] = len(newly_solved)
        metadata["problems_solved_total"] = len(all_solved_problems)
        metadata["progress_percentage"] = len(all_solved_problems) / total_problems_to_solve * 100 if total_problems_to_solve > 0 else 0
        with open(os.path.join(iter_dir, "metadata.json"), 'w') as f: json.dump(metadata, f, indent=2)

        
        logger.info("Creating training and evaluation datasets...")
        problem_sources_for_dataset = []

        # 1. Add current iteration's solved problems directory
        #    (if any problems were solved in this iteration and files were created)
        if os.path.exists(solved_dir) and os.listdir(solved_dir):
            logger.info(f"Including problems from current iteration: {solved_dir}")
            problem_sources_for_dataset.append(str(solved_dir))
        else:
            logger.info(f"No new problems solved and saved in current iteration directory: {solved_dir}")

        # 2. Add initial problems directory if it exists and has content
        if os.path.exists(initial_problems_dir) and os.listdir(initial_problems_dir):
             logger.info(f"Including problems from {initial_problems_dir} in dataset creation.")
             problem_sources_for_dataset.append(str(initial_problems_dir))
        else:
            logger.info(f"{initial_problems_dir} is empty or does not exist, not adding to global dataset sources for this round.")

        # 3. Add solved problems from all PREVIOUS iterations if resuming or if current_iteration > 1
        #    (If current_iteration is 1, we don't have previous iterations to add beyond initial_problems_dir)
        if current_iteration > 1: # Check > 1 because current_iteration is already incremented for this round
            start_prev_iter = 1
            # If resuming, only add iterations from the resume point up to the one before current
            # However, all_solved_problems set should already track everything solved globally.
            # The goal here is to provide all physical directories where solved problem artifacts might exist.
            for i in range(start_prev_iter, current_iteration): # Up to, but not including, current_iteration
                prev_iter_solved_dir = iterations_dir / f"iteration_{i}" / "solved_problems"
                if prev_iter_solved_dir.exists() and any(prev_iter_solved_dir.iterdir()):
                    logger.info(f"Including problems from previous iteration: {prev_iter_solved_dir}")
                    problem_sources_for_dataset.append(str(prev_iter_solved_dir))
                else:
                    logger.info(f"No solved problems found or directory empty for previous iteration: {prev_iter_solved_dir}")
        
        # Remove duplicates just in case, though list(set(...)) would also work
        problem_sources_for_dataset = sorted(list(set(problem_sources_for_dataset)))
        logger.info(f"Final list of problem source directories for dataset creation: {problem_sources_for_dataset}")

        train_file, test_file, dataset_stats = create_datasets(
            problems_dirs=problem_sources_for_dataset, # Now this list is comprehensive
            output_dir=str(iter_dir),
            test_ratio=0.1
        )
        with open(os.path.join(iter_dir, "dataset_stats.json"), 'w') as f: json.dump(dataset_stats, f, indent=2)
        
        # --- Check if training data exists ---
        if dataset_stats.get("train_pairs", 0) == 0:
            logger.warning(f"No training data generated for iteration {current_iteration}. Skipping training and evaluation.")
            # Ensure solver_model for next iteration remains the current one
            # No need to explicitly set solver_model = solver_model, it persists
            trained_model = solver_model # Keep the last successful trained model (or base if first iter failed)
            
            # Update metadata to reflect skipped steps
            metadata["training_skipped"] = True
            metadata["evaluation_skipped"] = True
            metadata["evaluation"] = {
                "base_model_acc": "N/A (Skipped)",
                "trained_model_acc": "N/A (Skipped)",
                "improvement": "N/A (Skipped)"
            }
            with open(os.path.join(iter_dir, "metadata.json"), 'w') as f: json.dump(metadata, f, indent=2)
            
            # Create a simplified round summary
            round_summary = {
                "iteration_number": current_iteration,
                "timestamp": metadata["timestamp"],
                "model_used_for_solving": solver_model,
                "problems_solved_at_start_of_iteration": metadata["problems_solved_before"],
                "new_problems_successfully_processed_this_iteration": metadata["problems_solved_new"],
                "total_problems_solved_after_iteration": metadata["problems_solved_total"],
                "dataset_creation_stats": dataset_stats,
                "training_details": "SKIPPED - No training data",
                "evaluation_results_on_iteration_test_set": "SKIPPED - No training data"
            }
            round_summary_path = high_level_summary_dir / f"round_{current_iteration}_summary.json"
            with open(round_summary_path, 'w') as f_summary: json.dump(round_summary, f_summary, indent=2)
            logger.info(f"High-level summary for skipped round {current_iteration} saved to {round_summary_path}")
            
            # Continue to the next iteration
            logger.info(f"Iteration {current_iteration} completed (Training Skipped).")
            logger.info(f"Progress: {len(all_solved_problems)}/{total_problems_to_solve} problems solved")
            # No cleanup needed as models weren't loaded for training/eval
            continue # Skip the rest of the loop for this iteration
        # --- End Check ---
        
        logger.info(f"Training new model...")
        wandb_name = f"{args.wandb_name_prefix}-iter{current_iteration}"
        

        # training_steps = args.train_steps
        training_steps = 22
        
        training_args_dict = {
            "max_steps": training_steps,
            "learning_rate": args.learning_rate,
            "wandb_name": wandb_name
        }
        

        ## CHANGED
        trained_model_path = train_model(
            model_name=base_model,
            train_file=train_file,
            eval_file=test_file,
            output_dir=str(model_dir),
            training_args=training_args_dict
        )
        
        # Delete checkpoint subdirectory if it exists
        checkpoint_dir = os.path.join(str(model_dir), 'checkpoint-22')
        if os.path.exists(checkpoint_dir) and os.path.isdir(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
            logger.info(f"Deleted checkpoint directory: {checkpoint_dir}")
        
        # ## TESTING
        trained_model = trained_model_path
        solver_model = trained_model_path

        logger.info(f"Model trained successfully: {trained_model}")
        
        logger.info("Evaluating models on test set...")
        base_eval_dir = os.path.join(eval_dir, "base_model")
        os.makedirs(base_eval_dir, exist_ok=True)
        base_evaluator = ModelEvaluator(model_name=base_model, model_type=args.model_type, device=args.device)
        base_results = base_evaluator.evaluate_all_tasks(test_jsonl_path=test_file, results_dir=base_eval_dir)
        trained_eval_dir = os.path.join(eval_dir, "trained_model")
        os.makedirs(trained_eval_dir, exist_ok=True)
        trained_evaluator = ModelEvaluator(model_name=trained_model, model_type=args.model_type, device=args.device)
        trained_results = trained_evaluator.evaluate_all_tasks(test_jsonl_path=test_file, results_dir=trained_eval_dir)
        
        q_base_acc.append(base_results["overall_success_rate"])
        q_trained_acc.append(trained_results["overall_success_rate"])
        
        metadata["evaluation"] = {
            "base_model_acc": base_results["overall_success_rate"],
            "trained_model_acc": trained_results["overall_success_rate"],
            "improvement": trained_results["overall_success_rate"] - base_results["overall_success_rate"]
        }
        with open(os.path.join(iter_dir, "metadata.json"), 'w') as f: json.dump(metadata, f, indent=2)
        
        logger.info(f"Iteration {current_iteration} completed")
        logger.info(f"Base model accuracy: {base_results['overall_success_rate']:.2%}")
        logger.info(f"Trained model accuracy: {trained_results['overall_success_rate']:.2%}")
        logger.info(f"Progress: {len(all_solved_problems)}/{total_problems_to_solve} problems solved")
        
        # --- Create High-Level Round Summary JSON ---
        round_summary = {
            "iteration_number": current_iteration,
            "timestamp": metadata["timestamp"],
            "model_used_for_solving": solver_model, # Model that attempted to solve problems
            "problems_solved_at_start_of_iteration": metadata["problems_solved_before"],
            # Note: To get 'new_problems_attempted_by_solver', ProblemSolver.attempt_all_problems would need to return it.
            # For now, we can infer it if solve_task is robustly called for all non-globally-solved items.
            "new_problems_successfully_processed_this_iteration": metadata["problems_solved_new"],
            "total_problems_solved_after_iteration": metadata["problems_solved_total"],
            "dataset_creation_stats": dataset_stats,
            "training_details": {
                "base_model_for_this_training_run": base_model, # The model that was fine-tuned
                "training_steps_configured": training_steps,
                "learning_rate": training_args_dict["learning_rate"],
                "wandb_run_name": training_args_dict["wandb_name"]
            },
            "evaluation_results_on_iteration_test_set": {
                "base_model_accuracy": metadata["evaluation"]["base_model_acc"],
                "trained_model_accuracy": metadata["evaluation"]["trained_model_acc"],
                "improvement_over_base_model": metadata["evaluation"]["improvement"]
            }
        }

        round_summary = {
            "iteration_number": current_iteration,
            "timestamp": metadata["timestamp"],
            "model_used_for_solving": solver_model, # Model that attempted to solve problems
            "problems_solved_at_start_of_iteration": metadata["problems_solved_before"],
            # Note: To get 'new_problems_attempted_by_solver', ProblemSolver.attempt_all_problems would need to return it.
            # For now, we can infer it if solve_task is robustly called for all non-globally-solved items.
            "new_problems_successfully_processed_this_iteration": metadata["problems_solved_new"],
            "total_problems_solved_after_iteration": metadata["problems_solved_total"],
            "dataset_creation_stats": dataset_stats,
        }

        round_summary_path = high_level_summary_dir / f"round_{current_iteration}_summary.json"
        with open(round_summary_path, 'w') as f_summary:
            json.dump(round_summary, f_summary, indent=2)
        logger.info(f"High-level summary for round {current_iteration} saved to {round_summary_path}")
        # --- End Summary JSON ---
        
        # Explicitly delete large objects and attempt to clear GPU cache
        logger.info(f"Cleaning up resources at the end of iteration {current_iteration}...")
        del solver
        del base_evaluator
        del trained_evaluator
        
        logger.info("Attempting to empty CUDA cache...")
        torch.cuda.empty_cache()
        
        logger.info("Triggering garbage collection...")
        gc.collect()
        logger.info(f"Resource cleanup for iteration {current_iteration} complete.")

    logger.info("=" * 50)
    logger.info("Bootstrap loop completed")
    logger.info(f"Total iterations: {current_iteration}")
    logger.info(f"Total problems solved: {len(all_solved_problems)}/{total_problems_to_solve}")
    
    if q_base_acc and q_trained_acc:
        logger.info(f"Initial base model accuracy: {q_base_acc[0]:.2%}")
        logger.info(f"Final trained model accuracy: {q_trained_acc[-1]:.2%}")
        logger.info(f"Overall improvement: {q_trained_acc[-1] - q_base_acc[0]:.2%}")
    
    logger.info("=" * 50)
    
    # Save final report
    final_report = {
        "total_iterations": current_iteration,
        "total_problems_solved": len(all_solved_problems),
        "total_problems_available_in_dataset": num_total_hf_problems,
        "target_problems_to_solve": total_problems_to_solve,
        "base_model": base_model,
        "final_model": trained_model,
        "base_model_accuracies": q_base_acc,
        "trained_model_accuracies": q_trained_acc,
        "overall_improvement": q_trained_acc[-1] - q_base_acc[0] if q_base_acc and q_trained_acc else None,
        "debug_mode": args.debug
    }
    
    with open(os.path.join(bootstrap_dir, "final_report.json"), 'w') as f:
        json.dump(final_report, f, indent=2)
    
    logger.info(f"Final report saved to {os.path.join(bootstrap_dir, 'final_report.json')}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())