#!/usr/bin/env python
"""
SFT (Supervised Fine-Tuning) Training Script for Q Language Tasks

This script performs supervised fine-tuning on a HuggingFace model using Q language
translation tasks (description→Q, python→Q, Q→python). It saves checkpoints frequently
with a flat output structure (all checkpoints directly in output directory).

Usage:
    python train_sft.py --base_model Qwen/Qwen2.5-7B-Instruct --max_steps 300 --output_dir outputs/my_sft
    python train_sft.py --base_model Qwen/Qwen2.5-7B-Instruct --max_steps 300 --output_dir outputs/my_sft_lora --use_lora
"""

import os
import sys
import logging
import argparse
import json
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    TrainerCallback
)
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
import wandb

# Training-only version - no evaluation imports

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('sft_training.log')
    ]
)
logger = logging.getLogger(__name__)

class CheckpointCallback(TrainerCallback):
    """Custom callback to save checkpoints at specified intervals and log progress"""
    
    def __init__(self, output_dir: str, save_every_n_steps: int = 10, tokenizer=None):
        self.output_dir = output_dir
        self.save_every_n_steps = save_every_n_steps
        self.checkpoints_dir = output_dir  # Flat structure - save directly in output_dir
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        self.tokenizer = tokenizer  # Store tokenizer reference
        self.trainer = None  # Will be set when trainer is available
        logger.info(f"CheckpointCallback initialized: saving every {save_every_n_steps} steps to {self.checkpoints_dir}")
        logger.info(f"Tokenizer provided: {tokenizer is not None}")
        
    def on_train_begin(self, args, state, control, **kwargs):
        """Store trainer reference when training begins"""
        # Look for trainer in kwargs or try to get it from the current context
        if 'trainer' in kwargs:
            self.trainer = kwargs['trainer']
            logger.info("✓ Trainer reference stored in checkpoint callback")
        else:
            logger.warning("⚠ Trainer not found in kwargs, will try alternative saving method")

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Save checkpoint every N steps"""
        if state.global_step > 0 and state.global_step % self.save_every_n_steps == 0:
            checkpoint_dir = os.path.join(self.checkpoints_dir, f"checkpoint-{state.global_step}")
            logger.info(f"*** SAVING CHECKPOINT at step {state.global_step} to {checkpoint_dir} ***")
            
            # Try to use trainer.save_model() for full model saving
            if self.trainer is not None:
                try:
                    # Save the full model using trainer's save_model method
                    # This properly handles all components and ensures complete model save
                    self.trainer.save_model(checkpoint_dir)
                    logger.info(f"✓ Full model saved using trainer.save_model() to: {checkpoint_dir}")
                    
                    # Verify the checkpoint was saved
                    if os.path.exists(checkpoint_dir) and os.listdir(checkpoint_dir):
                        files = os.listdir(checkpoint_dir)
                        file_sizes = []
                        for f in files:
                            file_path = os.path.join(checkpoint_dir, f)
                            if os.path.isfile(file_path):
                                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                                file_sizes.append(f"{f}: {size_mb:.1f}MB")
                        logger.info(f"✓ Checkpoint saved with {len(files)} files")
                        logger.info(f"✓ File sizes: {', '.join(file_sizes)}")
                    else:
                        logger.warning(f"⚠ Checkpoint directory appears empty: {checkpoint_dir}")
                        
                except Exception as e:
                    logger.error(f"✗ Failed to save checkpoint using trainer: {e}")
                    import traceback
                    logger.error(f"Full traceback: {traceback.format_exc()}")
            
            # Fallback to model.save_pretrained if trainer not available
            elif model is not None:
                try:
                    logger.warning("⚠ Using fallback model.save_pretrained() - may not save complete model")
                    model.save_pretrained(checkpoint_dir)
                    logger.info(f"✓ Model saved to: {checkpoint_dir}")
                    
                    # Save tokenizer using the stored reference
                    if self.tokenizer is not None:
                        self.tokenizer.save_pretrained(checkpoint_dir)
                        logger.info(f"✓ Tokenizer saved to: {checkpoint_dir}")
                    else:
                        logger.warning("⚠ No tokenizer available to save")
                    
                    # Verify the checkpoint was saved
                    if os.path.exists(checkpoint_dir) and os.listdir(checkpoint_dir):
                        files = os.listdir(checkpoint_dir)
                        logger.info(f"✓ Checkpoint saved with {len(files)} files: {files}")
                    else:
                        logger.warning(f"⚠ Checkpoint directory appears empty: {checkpoint_dir}")
                        
                except Exception as e:
                    logger.error(f"✗ Failed to save checkpoint: {e}")
                    import traceback
                    logger.error(f"Full traceback: {traceback.format_exc()}")
            else:
                logger.error(f"✗ Neither trainer nor model available in callback. Available kwargs: {list(kwargs.keys())}")
                
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log training progress"""
        if logs and state.is_world_process_zero:
            step = state.global_step
            if 'loss' in logs:
                logger.info(f"Step {step}: Training Loss = {logs['loss']:.4f}")
            if 'eval_loss' in logs:
                logger.info(f"Step {step}: Eval Loss = {logs['eval_loss']:.4f}")
            
            # Also log checkpointing status
            if step > 0 and step % self.save_every_n_steps == 0:
                logger.info(f"*** Step {step}: This is a checkpoint step ***")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="SFT Training for Q Language Tasks")
    
    # Model arguments
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Base model name or path (HuggingFace model)"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="huggingface",
        help="Model type (always huggingface for SFT training)"
    )
    
    # Data arguments
    parser.add_argument(
        "--train_file",
        type=str,
        default="train.jsonl",
        help="Path to training data JSONL file"
    )
    parser.add_argument(
        "--eval_file", 
        type=str,
        default="test.jsonl",
        help="Path to evaluation data JSONL file"
    )
    
    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sft_output",
        help="Output directory for model and logs"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=100,
        help="Maximum number of training steps"
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
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=10,
        help="Number of warmup steps"
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
    
    # Checkpoint and evaluation arguments
    parser.add_argument(
        "--save_every_n_steps",
        type=int,
        default=10,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=20,
        help="Evaluate every N steps"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=5,
        help="Log every N steps"
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
        "--wandb_project",
        type=str,
        default="q-sft",
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Run name for logging"
    )
    
    return parser.parse_args()

def setup_wandb(args, model_name: str):
    """Initialize Weights & Biases logging"""
    if not args.run_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = model_name.split('/')[-1] if '/' in model_name else model_name
        args.run_name = f"sft_{model_short}_{args.max_steps}steps_{timestamp}"
    
    wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        config={
            "base_model": args.base_model,
            "max_steps": args.max_steps,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_seq_length": args.max_seq_length,
            "warmup_steps": args.warmup_steps,
            "save_every_n_steps": args.save_every_n_steps,
        }
    )
    logger.info(f"Initialized W&B run: {args.run_name}")

def load_datasets(train_file: str, eval_file: str):
    """Load training and evaluation datasets"""
    logger.info(f"Loading training dataset from {train_file}")
    train_dataset = load_dataset("json", data_files=train_file, split="train")
    
    logger.info(f"Loading evaluation dataset from {eval_file}")
    eval_dataset = load_dataset("json", data_files=eval_file, split="train")
    
    logger.info(f"Training dataset: {len(train_dataset)} examples")
    logger.info(f"Evaluation dataset: {len(eval_dataset)} examples")
    
    # Log sample data
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        logger.info(f"Sample training example:")
        logger.info(f"  Prompt: {sample['prompt'][:200]}...")
        logger.info(f"  Completion: {sample['completion'][:200]}...")
    
    return train_dataset, eval_dataset

def train_model(args, train_dataset, eval_dataset):
    """Train the model using SFT"""
    logger.info(f"Loading tokenizer for {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Model loading configuration
    model_kwargs = dict(
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    
    # Add quantization config for QLoRA
    if args.use_qlora:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        logger.info("Using QLoRA with 4-bit quantization")
    
    # Set up PEFT config if using LoRA
    peft_config = None
    if args.use_lora or args.use_qlora:
        logger.info(f"Setting up {'QLoRA' if args.use_qlora else 'LoRA'} configuration")
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM"
        )
        logger.info(f"LoRA config: rank={args.lora_rank}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    
    # Training configuration
    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="no",  # We handle saving via callback
        warmup_steps=args.warmup_steps,
        bf16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="wandb",
        logging_dir=os.path.join(args.output_dir, "logs"),
        max_seq_length=args.max_seq_length,
        model_init_kwargs=model_kwargs,
        eos_token=tokenizer.eos_token,
        load_best_model_at_end=False,
    )
    
    # Initialize trainer with callback (pass tokenizer to callback)
    checkpoint_callback = CheckpointCallback(args.output_dir, args.save_every_n_steps, tokenizer=tokenizer)
    
    logger.info("Initializing SFT Trainer...")
    trainer = SFTTrainer(
        model=args.base_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[checkpoint_callback]
    )
    
    # Set trainer reference in callback for proper model saving
    checkpoint_callback.trainer = trainer
    logger.info("✓ Trainer reference set in checkpoint callback")
    
    # Start training
    logger.info("*** Starting SFT Training ***")
    logger.info(f"Total training steps: {args.max_steps}")
    logger.info(f"Checkpointing every {args.save_every_n_steps} steps")
    logger.info(f"Evaluating every {args.eval_steps} steps")
    if args.use_lora or args.use_qlora:
        logger.info(f"Using {'QLoRA' if args.use_qlora else 'LoRA'} training")
    
    train_result = trainer.train()
    
    # Save final model (flat structure)
    logger.info("*** Saving final model ***")
    final_model_dir = args.output_dir  # Save directly to output dir
    trainer.save_model(final_model_dir)
    
    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    logger.info("*** SFT Training completed ***")
    logger.info(f"Final model saved to: {final_model_dir}")
    
    return final_model_dir, metrics

# Evaluation function removed - training only version

def save_training_summary(args, training_metrics: Dict, eval_metrics: Dict):
    """Save a summary of training metrics."""
    summary = {
        'model_name': args.base_model,
        'final_model_path': args.output_dir,  # Flat structure
        'training_args': vars(args),
        'training_metrics': training_metrics,
        'final_evaluation_metrics': eval_metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    summary_file = os.path.join(args.output_dir, "training_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Training summary saved to: {summary_file}")

def main():
    args = parse_args()
    set_seed(args.seed)
    
    logger.info("--- Starting SFT Training Run ---")
    logger.info(f"Model: {args.base_model}, Max Steps: {args.max_steps}")
    
    # Setup directories and logging
    os.makedirs(args.output_dir, exist_ok=True)
    setup_wandb(args, args.base_model)
    
    # Load datasets
    train_dataset, eval_dataset = load_datasets(args.train_file, args.eval_file)
    
    # Train model
    final_model_path, training_metrics = train_model(args, train_dataset, eval_dataset)
    
    # Save summary (no evaluation)
    eval_metrics = {"status": "skipped", "reason": "Training-only mode"}
    save_training_summary(args, training_metrics, eval_metrics)
    
    logger.info("--- SFT Training Run Finished ---")

if __name__ == "__main__":
    main() 