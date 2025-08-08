#!/usr/bin/env python3
"""
Curriculum Learning SFT Trainer

This script runs SFT training with curriculum learning - training on different
data phases sequentially (e.g., Easy → Medium → Hard).

Usage:
    python train_sft_curriculum.py --curriculum_dir curriculum_data/difficulty/ --base_model Qwen/Qwen2.5-7B-Instruct
"""

import os
import sys
import json
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run Curriculum SFT Training")
    
    # Model arguments
    parser.add_argument("--base_model", type=str, required=True, help="Base model name or path")
    
    # Curriculum arguments
    parser.add_argument("--curriculum_dir", type=str, required=True, help="Directory with curriculum phase files")
    parser.add_argument("--eval_file", type=str, default="../SFT_Data/test.jsonl", help="Evaluation file")
    
    # Training arguments
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for training")
    parser.add_argument("--batch_size", type=int, default=1, help="Per device batch size")
    parser.add_argument("--steps_per_phase", type=int, default=200, help="Training steps per curriculum phase")
    parser.add_argument("--save_every_n_steps", type=int, default=50, help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps", type=int, default=25, help="Evaluate every N steps")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--experiment_name", type=str, default=None, help="Experiment name")
    
    # Misc arguments
    parser.add_argument("--device", type=str, default="cuda", help="Device for training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    return parser.parse_args()

def discover_curriculum_phases(curriculum_dir: str) -> List[str]:
    """Discover curriculum phase files in order."""
    curriculum_path = Path(curriculum_dir)
    if not curriculum_path.exists():
        raise FileNotFoundError(f"Curriculum directory not found: {curriculum_dir}")
    
    # Find all .jsonl files
    phase_files = list(curriculum_path.glob("*.jsonl"))
    
    if not phase_files:
        raise FileNotFoundError(f"No curriculum phase files found in {curriculum_dir}")
    
    # Check if this is a difficulty-based curriculum
    difficulty_files = [f for f in phase_files if any(diff in f.name for diff in ['easy', 'medium', 'hard'])]
    
    if len(difficulty_files) == len(phase_files) and len(difficulty_files) >= 2:
        # This is a difficulty-based curriculum - sort by logical difficulty order
        logger.info("Detected difficulty-based curriculum - sorting by difficulty order")
        
        # Define the correct difficulty order
        difficulty_order = ['easy', 'medium', 'hard']
        
        # Sort files by difficulty level
        def get_difficulty_order(file_path):
            filename = file_path.name.lower()
            for i, difficulty in enumerate(difficulty_order):
                if difficulty in filename:
                    return i
            return 999  # Unknown difficulty goes last
        
        phase_files.sort(key=get_difficulty_order)
        
        logger.info("Difficulty-based curriculum order:")
        for i, phase_file in enumerate(phase_files):
            logger.info(f"  Phase {i+1}: {phase_file.name}")
    
    else:
        # Not a difficulty-based curriculum - use alphabetical sorting
        logger.info("Using alphabetical ordering for curriculum phases")
        phase_files.sort()
        
        logger.info(f"Found {len(phase_files)} curriculum phases:")
        for i, phase_file in enumerate(phase_files):
            logger.info(f"  Phase {i+1}: {phase_file.name}")
    
    return [str(f) for f in phase_files]

def count_examples(file_path: str) -> int:
    """Count number of examples in a JSONL file."""
    count = 0
    with open(file_path, 'r') as f:
        for _ in f:
            count += 1
    return count

def run_phase_training(
    phase_name: str,
    phase_file: str,
    eval_file: str,
    base_model: str,
    output_dir: str,
    args,
    is_first_phase: bool = True
) -> str:
    """Run training for a single curriculum phase."""
    
    logger.info(f"*** Starting training phase: {phase_name} ***")
    
    # Count examples for logging
    num_examples = count_examples(phase_file)
    logger.info(f"Phase {phase_name}: {num_examples} examples")
    
    # Create phase output directory
    phase_output_dir = os.path.join(output_dir, f"phase_{phase_name}")
    os.makedirs(phase_output_dir, exist_ok=True)
    
    # Determine model to start from (base_model is actually the path to use)
    model_path = base_model
    
    # Prepare training command
    cmd = [
        sys.executable, "train_sft.py",
        "--base_model", model_path,
        "--train_file", phase_file,
        "--eval_file", eval_file,
        "--output_dir", phase_output_dir,
        "--max_steps", str(args.steps_per_phase),
        "--learning_rate", str(args.learning_rate),
        "--batch_size", str(args.batch_size),
        "--save_every_n_steps", str(args.save_every_n_steps),
        "--eval_steps", str(args.eval_steps),
        "--device", args.device,
        "--seed", str(args.seed),
        "--wandb_project", "q-sft-curriculum",
        "--run_name", f"{args.experiment_name}_{phase_name}" if args.experiment_name else f"curriculum_{phase_name}"
    ]
    
    logger.info(f"Running phase training command: {' '.join(cmd)}")
    
    # Run training
    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"Phase {phase_name} training completed successfully")
        if args.verbose:
            logger.info(f"Training stdout: {result.stdout}")
        
        return phase_output_dir
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Phase {phase_name} training failed: {e}")
        logger.error(f"Training stderr: {e.stderr}")
        raise

def run_curriculum_training(args) -> str:
    """Run the full curriculum training pipeline."""
    
    logger.info("*** Starting Curriculum SFT Training ***")
    logger.info(f"Base model: {args.base_model}")
    logger.info(f"Curriculum directory: {args.curriculum_dir}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Steps per phase: {args.steps_per_phase}")
    
    # Discover curriculum phases
    phase_files = discover_curriculum_phases(args.curriculum_dir)
    
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save curriculum training config
    config = {
        'base_model': args.base_model,
        'curriculum_dir': args.curriculum_dir,
        'learning_rate': args.learning_rate,
        'steps_per_phase': args.steps_per_phase,
        'batch_size': args.batch_size,
        'num_phases': len(phase_files),
        'phase_files': phase_files,
        'timestamp': datetime.now().isoformat()
    }
    
    config_file = os.path.join(args.output_dir, "curriculum_config.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Run training phases sequentially
    phase_results = []
    current_model = args.base_model
    
    for i, phase_file in enumerate(phase_files):
        phase_name = Path(phase_file).stem
        is_first_phase = (i == 0)
        
        # Update previous phase name for model path
        if i > 0:
            prev_phase_name = Path(phase_files[i-1]).stem
            current_model = os.path.join(args.output_dir, f"phase_{prev_phase_name}")
        
        # Run phase training
        phase_output_dir = run_phase_training(
            phase_name=phase_name,
            phase_file=phase_file,
            eval_file=args.eval_file,
            base_model=current_model,
            output_dir=args.output_dir,
            args=args,
            is_first_phase=is_first_phase
        )
        
        phase_results.append({
            'phase_name': phase_name,
            'phase_file': phase_file,
            'output_dir': phase_output_dir,
            'examples_count': count_examples(phase_file)
        })
        
        # Update current model for next phase
        current_model = phase_output_dir
    
    # Save final results
    results = {
        'config': config,
        'phase_results': phase_results,
        'final_model_path': current_model,
        'total_phases': len(phase_results),
        'completed_timestamp': datetime.now().isoformat()
    }
    
    results_file = os.path.join(args.output_dir, "curriculum_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("*** Curriculum Training Completed ***")
    logger.info(f"Final model saved to: {current_model}")
    logger.info(f"Results saved to: {results_file}")
    
    return current_model

def main():
    """Main training function"""
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Generate experiment name if not provided
    if not args.experiment_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        curriculum_strategy = Path(args.curriculum_dir).name
        model_name_safe = args.base_model.split('/')[-1] if '/' in args.base_model else args.base_model
        args.experiment_name = f"curriculum_{curriculum_strategy}_{model_name_safe}_{timestamp}"
    
    try:
        final_model_path = run_curriculum_training(args)
        logger.info("Curriculum training pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Curriculum training pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main() 