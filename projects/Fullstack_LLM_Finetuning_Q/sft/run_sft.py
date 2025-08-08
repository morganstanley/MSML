#!/usr/bin/env python
"""
SFT Training Runner

This script provides a convenient interface for running SFT training with different configurations.
It handles directory setup, logging, and calls the main training script.

Usage:
    python run_sft.py --base_model Qwen/Qwen2.5-7B-Instruct --max_steps 300 --output_dir outputs/my_sft
    python run_sft.py --base_model Qwen/Qwen2.5-7B-Instruct --max_steps 300 --output_dir outputs/my_sft_lora --use_lora
"""

import os
import sys
import argparse
import logging
import subprocess
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run SFT Training")
    
    # Model arguments
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Base model name or path"
    )
    
    # Data arguments
    parser.add_argument(
        "--train_file",
        type=str,
        default="../SFT_Data/train.jsonl",
        help="Training data file"
    )
    parser.add_argument(
        "--eval_file",
        type=str,
        default="../SFT_Data/test.jsonl",
        help="Evaluation data file"
    )
    parser.add_argument(
        "--use_no_test_cases",
        action="store_true",
        help="Use filtered data files without test case translation prompts"
    )
    
    # Training arguments
    parser.add_argument(
        "--max_steps",
        type=int,
        default=100,
        help="Maximum training steps"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Per device batch size"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps"
    )
    
    # LoRA arguments
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Whether to use LoRA (Low-Rank Adaptation)"
    )
    parser.add_argument(
        "--use_qlora",
        action="store_true",
        help="Whether to use QLoRA (4-bit quantized LoRA)"
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha parameter"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="LoRA dropout"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (auto-generated if not provided)"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Experiment name for tracking"
    )
    
    # Checkpoint arguments
    parser.add_argument(
        "--save_every_n_steps",
        type=int,
        default=200,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=20,
        help="Evaluate every N steps"
    )
    
    # Misc arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for training"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Use Weights & Biases for logging"
    )
    
    return parser.parse_args()

def setup_output_directory(args):
    """Setup output directory with auto-generated name if needed"""
    if args.output_dir is None:
        # Auto-generate directory name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = args.base_model.split('/')[-1] if '/' in args.base_model else args.base_model
        
        # Include LoRA in the name if used
        method_suffix = ""
        if args.use_qlora:
            method_suffix = "_qlora"
        elif args.use_lora:
            method_suffix = "_lora"
        
        dir_name = f"sft_{model_name}_{args.max_steps}steps{method_suffix}_{timestamp}"
        args.output_dir = os.path.join("outputs", dir_name)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")
    
    return args.output_dir

def run_training(args):
    """Run the actual training"""
    logger.info("*** Starting SFT Training ***")
    
    # Setup output directory
    output_dir = setup_output_directory(args)
    
    # Generate experiment name if not provided
    if not args.experiment_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name_safe = args.base_model.split('/')[-1] if '/' in args.base_model else args.base_model
        
        # Include method in experiment name
        method_suffix = ""
        if args.use_qlora:
            method_suffix = "_qlora"
        elif args.use_lora:
            method_suffix = "_lora"
        
        args.experiment_name = f"sft_{model_name_safe}_{args.max_steps}steps{method_suffix}_{timestamp}"
    
    # Prepare training command
    cmd = [
        sys.executable, "train_sft.py",
        "--base_model", args.base_model,
        "--train_file", args.train_file,
        "--eval_file", args.eval_file,
        "--output_dir", output_dir,
        "--max_steps", str(args.max_steps),
        "--learning_rate", str(args.learning_rate),
        "--batch_size", str(args.batch_size),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--save_every_n_steps", str(args.save_every_n_steps),
        "--eval_steps", str(args.eval_steps),
        "--device", args.device,
        "--seed", str(args.seed),
        "--wandb_project", "q-sft",
        "--run_name", args.experiment_name
    ]
    
    # Add LoRA arguments if specified
    if args.use_lora:
        cmd.extend(["--use_lora"])
    if args.use_qlora:
        cmd.extend(["--use_qlora"])
    if args.use_lora or args.use_qlora:
        cmd.extend([
            "--lora_rank", str(args.lora_rank),
            "--lora_alpha", str(args.lora_alpha),
            "--lora_dropout", str(args.lora_dropout)
        ])
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    # Run training
    try:
        result = subprocess.run(cmd, check=True, cwd=Path(__file__).parent)
        logger.info("Training completed successfully!")
        return output_dir
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed with exit code {e.returncode}")
        raise

def main():
    """Main function"""
    args = parse_args()
    
    # Update data file paths if using no test cases
    if args.use_no_test_cases:
        if args.train_file == "../SFT_Data/train.jsonl":
            args.train_file = "../SFT_Data/no_test_case_train.jsonl"
        if args.eval_file == "../SFT_Data/test.jsonl":
            args.eval_file = "../SFT_Data/no_test_case_test.jsonl"
        logger.info("Using filtered data files without test case translation prompts")
    
    # Log configuration
    logger.info("=== SFT Training Configuration ===")
    logger.info(f"Base model: {args.base_model}")
    logger.info(f"Training file: {args.train_file}")
    logger.info(f"Evaluation file: {args.eval_file}")
    logger.info(f"Max steps: {args.max_steps}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    if args.use_lora or args.use_qlora:
        method = "QLoRA" if args.use_qlora else "LoRA"
        logger.info(f"Using {method}: rank={args.lora_rank}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    
    try:
        output_dir = run_training(args)
        logger.info(f"Training completed! Model saved to: {output_dir}")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 