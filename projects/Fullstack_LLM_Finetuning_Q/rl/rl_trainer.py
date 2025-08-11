#!/usr/bin/env python
"""
Q Language Training Script using TRL's GRPOTrainer

"""

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Dict, Any

import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig

# Optional wandb import
import wandb
WANDB_AVAILABLE = True

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import existing utilities
from rl_training.rl_datasets import get_q_code_dataloaders
from rl_training.rl_evaluator import RLEvaluator
from rl_training.rl_utils import extract_q_code_with_reasoning_support

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_system_prompt(use_reasoning_format: bool = True) -> str:
    """Create system prompt for Q language training."""
    base_prompt = """You are an expert Q language programmer. Your task is to write correct Q code that solves the given programming problem.


Always provide complete, runnable Q code that solves the problem correctly."""

    if use_reasoning_format:
        base_prompt += """

Use this exact format for your response:
<reasoning>
Explain your approach here
</reasoning>

<answer>
// Your Q code here
</answer>"""
    else:
        base_prompt += " Provide only the Q code without explanations."
    
    return base_prompt

def format_problem_for_training(problem: Dict, use_reasoning_format: bool = True) -> str:
    """Format a problem for training prompt."""
    system_prompt = create_system_prompt(use_reasoning_format)
    user_prompt = f"Problem:\n{problem['description']}"
    
    return f"{system_prompt}\n\nHuman: {user_prompt}\n\nAssistant:"

def prepare_dataset_for_trl(problems: List[Dict], use_reasoning_format: bool = True, repeat_for_pass_at_k: int = 1) -> Dataset:
    """Prepare dataset in the format expected by TRL."""
    data = []
    
    for problem in problems:
        # Create the prompt
        prompt = format_problem_for_training(problem, use_reasoning_format)
        
        # Store problem info for reward computation
        problem_data = {
            'prompt': prompt,
            'problem_id': problem['id'],
            'problem_description': problem['description'], 
            'test_cases': problem['tests']
        }
        
        # Repeat each problem multiple times for pass@k-like evaluation
        for repeat_idx in range(repeat_for_pass_at_k):
            if repeat_for_pass_at_k > 1:
                # Add repeat index to make each instance unique
                repeated_data = problem_data.copy()
                repeated_data['problem_id'] = f"{problem['id']}_rep{repeat_idx}"
                data.append(repeated_data)
            else:
                data.append(problem_data)
    
    return Dataset.from_list(data)

def create_dummy_reward_function():
    """Create a dummy reward function that rewards based on bracket usage."""
    
    def reward_function(completions: List[str], prompts: List[str], **kwargs) -> List[float]:
        """
        Dummy reward function that counts [ and ] characters.
        
        Args:
            completions: List of model completions
            prompts: List of prompts
            **kwargs: Additional arguments from TRL
            
        Returns:
            List of rewards (one per completion)
        """
        rewards = []
        
        for i, completion in enumerate(completions):
            # Count [ and ] characters
            bracket_count = completion.count('[') + completion.count(']')
            # Normalize to reasonable range (e.g., 0-3 scale)
            reward = min(bracket_count / 5.0, 3.0)
            rewards.append(reward)
            
            if i < 3:  # Log first few for debugging
                logger.info(f"Completion {i}: {bracket_count} brackets, reward: {reward:.2f}")
        
        return rewards
    
    return reward_function

def create_q_reward_function(train_dataset: Dataset, eval_dataset: Dataset = None, use_reasoning_format: bool = True, max_tests: int = 3, perfect_bonus: float = 2.0, base_reward_weight: float = 1.0, perfect_reward_weight: float = 1.0):
    """Create a reward function for Q language training."""
    
    evaluator = RLEvaluator(reward_method="test_cases")
    
    # Create a mapping from prompt to test cases for both train and eval
    prompt_to_tests = {}
    for item in train_dataset:
        prompt_to_tests[item['prompt']] = item['test_cases']
    
    if eval_dataset:
        for item in eval_dataset:
            prompt_to_tests[item['prompt']] = item['test_cases']
    
    print(f"Reward function created with {len(prompt_to_tests)} prompts (train + eval)")
    
    # Statistics tracking
    stats = {
        'total_calls': 0,
        'code_extraction_failures': 0,
        'timeout_failures': 0,
        'execution_failures': 0,
        'successful_executions': 0
    }
    
    def reward_function(completions: List[str], prompts: List[str], **kwargs) -> List[float]:
        """
        Compute rewards for Q language completions.
        
        Args:
            completions: List of model completions
            prompts: List of prompts (we'll extract problem info from these)
            **kwargs: Additional arguments from TRL
            
        Returns:
            List of rewards (one per completion)
        """
        rewards = []
        stats['total_calls'] += len(completions)
        
        for i, (completion, prompt) in enumerate(zip(completions, prompts)):
            try:
                # Get test cases from the prompt mapping
                if prompt in prompt_to_tests:
                    test_cases = prompt_to_tests[prompt]
                    if not isinstance(test_cases, list):
                        test_cases = [test_cases]
                else:
                    logger.warning(f"No test cases found for prompt {i}")
                    rewards.append(0.0)
                    continue
                
                # Limit test cases for training efficiency
                if max_tests and len(test_cases) > max_tests:
                    test_cases = test_cases[:max_tests]
                
                # Extract Q code from completion
                reasoning_part, answer_part, q_code = extract_q_code_with_reasoning_support(completion)
                
                if not q_code.strip():
                    stats['code_extraction_failures'] += 1
                    if i < 3:  # Log first few failures for debugging
                        logger.warning(f"No Q code found in completion {i}. Completion preview: {completion[:200]}...")
                    rewards.append(0.0)
                    continue
                
                # Run Q code against test cases
                passed_tests = 0
                total_tests = len(test_cases)
                
                for test_idx, test_case in enumerate(test_cases):
                    # Extract test info
                    test_code = test_case.get('test_code', '')
                    expected_output = test_case.get('expected_output', '')
                    
                    if not test_code:
                        continue
                    
                    # Run the test with increased timeout
                    from rl_training.rl_evaluator import run_q_code
                    
                    # Remove exit 0; from solution if present
                    clean_q_code = q_code.strip()
                    if clean_q_code.endswith("exit 0;"):
                        clean_q_code = clean_q_code[:-7].strip()
                    
                    # Combine solution with test
                    full_code = f"{clean_q_code}\n\n{test_code}"
                    
                    # Execute with longer timeout (increased from 0.2 to 1.0 seconds)
                    success, stdout, stderr = run_q_code(full_code, timeout=1.0)
                    
                    # Check if test passed
                    if success and stdout is not None:
                        actual_stripped = stdout.strip()
                        expected_stripped = expected_output.strip()
                        
                        # Handle empty string cases
                        if ((actual_stripped == "" and expected_stripped == '""') or 
                            (actual_stripped == '""' and expected_stripped == "") or
                            (actual_stripped == expected_stripped)):
                            passed_tests += 1
                    else:
                        # Track failure reasons
                        if "Timeout" in stderr:
                            stats['timeout_failures'] += 1
                        else:
                            stats['execution_failures'] += 1
                        
                        if i < 3 and test_idx < 2:  # Debug first few failures
                            logger.debug(f"Test failed for completion {i}, test {test_idx}: {stderr}")
                
                # Calculate reward: percentage passed + bonus for perfect
                if total_tests > 0:
                    base_reward = (passed_tests / total_tests) * base_reward_weight
                    perfect_reward = (perfect_bonus if passed_tests == total_tests else 0.0) * perfect_reward_weight
                    final_reward = base_reward + perfect_reward
                    stats['successful_executions'] += 1
                else:
                    final_reward = 0.0
                
                rewards.append(final_reward)
                
                if i < 3:  # Log first few for debugging
                    logger.info(f"Completion {i}: {passed_tests}/{total_tests} tests passed, reward: {final_reward:.2f}")
                    
            except Exception as e:
                logger.error(f"Error computing reward for completion {i}: {e}")
                stats['execution_failures'] += 1
                rewards.append(0.0)
        
        # Log statistics periodically
        if stats['total_calls'] % 100 == 0:
            logger.info(f"Reward function stats after {stats['total_calls']} calls:")
            logger.info(f"  Code extraction failures: {stats['code_extraction_failures']}")
            logger.info(f"  Timeout failures: {stats['timeout_failures']}")
            logger.info(f"  Execution failures: {stats['execution_failures']}")
            logger.info(f"  Successful executions: {stats['successful_executions']}")
        
        return rewards
    
    return reward_function

def create_pass_at_k_evaluator(eval_dataset: Dataset, use_reasoning_format: bool = True, k: int = 4, use_all_tests: bool = True):
    """Create a pass@k evaluator for clean evaluation metrics (separate from training reward)."""
    
    # Create a mapping from prompt to test cases
    prompt_to_tests = {}
    for item in eval_dataset:
        prompt_to_tests[item['prompt']] = item['test_cases']
    
    def evaluate_pass_at_k(model, tokenizer, eval_prompts_sample=None, num_samples: int = 40):
        """
        Evaluate model using pass@k on a sample of evaluation problems.
        This gives us a clean, unnoisy metric that matches the user's true evaluation.
        """
        from transformers import GenerationConfig
        import random
        import torch
        
        logger.info(f"Running pass@{k} evaluation on {num_samples} problems...")
        
        # Sample evaluation problems
        if eval_prompts_sample is None:
            eval_indices = random.sample(range(len(eval_dataset)), min(num_samples, len(eval_dataset)))
            eval_prompts_sample = [eval_dataset[i]['prompt'] for i in eval_indices]
        
        # Handle distributed/wrapped models (vLLM compatibility fix)
        eval_model = model
        if hasattr(model, 'module'):
            eval_model = model.module
        elif hasattr(model, '_orig_mod'):
            eval_model = model._orig_mod
        
        eval_model.eval()
        total_problems = 0
        pass_at_k_successes = 0
        
        stats = {
            'code_extraction_failures': 0,
            'timeout_failures': 0,
            'execution_failures': 0,
        }
        
        with torch.no_grad():
            for prompt_idx, prompt in enumerate(eval_prompts_sample):
                if prompt not in prompt_to_tests:
                    continue
                    
                total_problems += 1
                test_cases = prompt_to_tests[prompt]
                if not isinstance(test_cases, list):
                    test_cases = [test_cases]
                
                # Use all test cases for evaluation (more reliable)
                if not use_all_tests:
                    test_cases = test_cases[:5]
                
                try:
                    # Generate k completions for this prompt
                    inputs = tokenizer(prompt, return_tensors="pt")
                    
                    # Move inputs to the same device as the model
                    device = next(eval_model.parameters()).device
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    # Fix pad_token_id if needed
                    if tokenizer.pad_token_id is None:
                        tokenizer.pad_token_id = tokenizer.eos_token_id
                    
                    generation_config = GenerationConfig(
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=0.7,
                        num_return_sequences=k,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                    
                    # Use the unwrapped model for generation
                    outputs = eval_model.generate(**inputs, generation_config=generation_config)
                    
                    # Decode completions
                    completions = []
                    for output in outputs:
                        completion = tokenizer.decode(
                            output[inputs['input_ids'].shape[1]:], 
                            skip_special_tokens=True
                        )
                        completions.append(completion)
                    
                except Exception as gen_error:
                    logger.warning(f"Generation failed for prompt {prompt_idx}: {gen_error}")
                    stats['execution_failures'] += 1
                    continue
                
                # Check if ANY completion passes ALL tests (pass@k)
                any_completion_passed = False
                
                for completion in completions:
                    try:
                        # Extract Q code
                        reasoning_part, answer_part, q_code = extract_q_code_with_reasoning_support(completion)
                        
                        if not q_code.strip():
                            stats['code_extraction_failures'] += 1
                            continue
                        
                        # Test against all test cases
                        passed_tests = 0
                        total_tests = len(test_cases)
                        
                        for test_case in test_cases:
                            test_code = test_case.get('test_code', '')
                            expected_output = test_case.get('expected_output', '')
                            
                            if not test_code:
                                continue
                            
                            from rl_training.rl_evaluator import run_q_code
                            
                            clean_q_code = q_code.strip()
                            if clean_q_code.endswith("exit 0;"):
                                clean_q_code = clean_q_code[:-7].strip()
                            
                            full_code = f"{clean_q_code}\n\n{test_code}"
                            success, stdout, stderr = run_q_code(full_code, timeout=2.0)
                            
                            if success and stdout is not None:
                                actual_stripped = stdout.strip()
                                expected_stripped = expected_output.strip()
                                
                                if ((actual_stripped == "" and expected_stripped == '""') or 
                                    (actual_stripped == '""' and expected_stripped == "") or
                                    (actual_stripped == expected_stripped)):
                                    passed_tests += 1
                            else:
                                if "Timeout" in stderr:
                                    stats['timeout_failures'] += 1
                                else:
                                    stats['execution_failures'] += 1
                        
                        # If this completion passed all tests, we have success
                        if passed_tests == total_tests and total_tests > 0:
                            any_completion_passed = True
                            break  # Found a working solution
                            
                    except Exception as e:
                        stats['execution_failures'] += 1
                        continue
                
                if any_completion_passed:
                    pass_at_k_successes += 1
                
                # Log progress
                if (prompt_idx + 1) % 10 == 0:
                    current_rate = pass_at_k_successes / total_problems if total_problems > 0 else 0
                    logger.info(f"Pass@{k} eval progress: {prompt_idx + 1}/{len(eval_prompts_sample)}, current rate: {current_rate:.3f}")
        
        # Final results
        pass_at_k_rate = pass_at_k_successes / total_problems if total_problems > 0 else 0.0
        
        logger.info(f"Pass@{k} Evaluation Results:")
        logger.info(f"  Problems evaluated: {total_problems}")
        logger.info(f"  Pass@{k} successes: {pass_at_k_successes}")
        logger.info(f"  Pass@{k} rate: {pass_at_k_rate:.3f}")
        logger.info(f"  Code extraction failures: {stats['code_extraction_failures']}")
        logger.info(f"  Timeout failures: {stats['timeout_failures']}")
        logger.info(f"  Execution failures: {stats['execution_failures']}")
        
        eval_model.train()  # Switch back to training mode
        return {
            f'pass_at_{k}': pass_at_k_rate,
            f'pass_at_{k}_successes': pass_at_k_successes,
            f'pass_at_{k}_total': total_problems,
            'code_extraction_failures': stats['code_extraction_failures'],
            'timeout_failures': stats['timeout_failures'],
            'execution_failures': stats['execution_failures'],
        }
    
    return evaluate_pass_at_k

from transformers import TrainerCallback

class PassAtKTrainerCallback(TrainerCallback):
    """Proper TrainerCallback implementation for pass@k evaluation."""
    
    def __init__(self, pass_at_k_evaluator, eval_steps: int = 50):
        super().__init__()
        self.pass_at_k_evaluator = pass_at_k_evaluator
        self.eval_steps = eval_steps
        self.step_count = 0
        
    def on_step_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        """Called at the end of each training step."""
        self.step_count += 1
        
        # Run pass@k evaluation at specified intervals
        if self.step_count % self.eval_steps == 0:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running Pass@K Evaluation at Step {self.step_count}")
            logger.info(f"{'='*60}")
            
            try:
                # Run pass@k evaluation
                eval_results = self.pass_at_k_evaluator(model, tokenizer)
                
                # Log results
                for key, value in eval_results.items():
                    logger.info(f"  {key}: {value}")
                
                logger.info(f"{'='*60}\n")
                
            except Exception as e:
                logger.error(f"Error during pass@k evaluation: {e}")
                import traceback
                traceback.print_exc()

def test_reward_function_baseline(reward_function, train_dataset: Dataset, num_samples: int = 5):
    """Test reward function on a few examples to validate it's working."""
    logger.info(f"Testing reward function on {num_samples} examples...")
    
    # Sample a few examples
    import random
    sample_indices = random.sample(range(len(train_dataset)), min(num_samples, len(train_dataset)))
    
    for i, idx in enumerate(sample_indices):
        example = train_dataset[idx]
        prompt = example['prompt']
        
        # Test with a simple valid Q code
        simple_q_code = "1+1"
        logger.info(f"\nTesting example {i+1} with simple Q code: '{simple_q_code}'")
        rewards = reward_function([simple_q_code], [prompt])
        logger.info(f"Simple Q code reward: {rewards[0]:.3f}")
        
        # Test with empty response
        logger.info(f"Testing example {i+1} with empty response")
        rewards = reward_function([""], [prompt])
        logger.info(f"Empty response reward: {rewards[0]:.3f}")
        
        # Test with reasoning format (if enabled)
        reasoning_response = """<reasoning>
This is a simple addition problem.
</reasoning>

<answer>
1+1
</answer>"""
        logger.info(f"Testing example {i+1} with reasoning format")
        rewards = reward_function([reasoning_response], [prompt])
        logger.info(f"Reasoning format reward: {rewards[0]:.3f}")
    
    logger.info("Reward function baseline test completed.")

def test_base_model_output(model_name: str, train_dataset: Dataset, use_reasoning_format: bool = True, num_samples: int = 3):
    """Test what the base model generates to understand format issues."""
    logger.info(f"Testing base model output format with {num_samples} examples...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model.eval()
        
        # Sample a few examples
        import random
        sample_indices = random.sample(range(len(train_dataset)), min(num_samples, len(train_dataset)))
        
        for i, idx in enumerate(sample_indices):
            example = train_dataset[idx]
            prompt = example['prompt']
            
            logger.info(f"\n--- Testing base model on example {i+1} ---")
            logger.info(f"Prompt preview: {prompt[:200]}...")
            
            # Tokenize and generate
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            # Decode response
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            logger.info(f"Base model response: {response}")
            
            # Test code extraction
            reasoning_part, answer_part, q_code = extract_q_code_with_reasoning_support(response)
            logger.info(f"Extracted Q code: '{q_code}'")
            logger.info(f"Has reasoning: {len(reasoning_part) > 0}")
            logger.info(f"Has answer: {len(answer_part) > 0}")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        
    except Exception as e:
        logger.warning(f"Could not test base model output: {e}")
        logger.info("Continuing with training...")

def main():
    parser = argparse.ArgumentParser(description="Simple Q Language Training with TRL")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-0.5B-Instruct", help="Model name")
    parser.add_argument("--output_dir", type=str, default="q_language_trl_output", help="Output directory")
    parser.add_argument("--max_train_problems", type=int, default=540, help="Max training problems")
    parser.add_argument("--max_test_problems", type=int, default=40, help="Max test problems")
    parser.add_argument("--max_tests_per_problem", type=int, default=5, help="Max tests per problem for reward")
    parser.add_argument("--perfect_bonus", type=float, default=1.0, help="Bonus for perfect solutions")
    parser.add_argument("--use_reasoning_format", action="store_true", help="Use reasoning format")
    parser.add_argument("--eval_steps", type=int, default=25, help="Evaluate every N steps")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM server mode")
    parser.add_argument("--dummy_reward", action="store_true", help="Use dummy reward (count [ and ] characters)")
    parser.add_argument("--test_reward_function", action="store_true", help="Test reward function on a few examples before training")
    parser.add_argument("--eval_pass_at_k", action="store_true", help="Run pass@k evaluation during training for clean metrics")
    parser.add_argument("--eval_k", type=int, default=4, help="K value for pass@k evaluation")
    parser.add_argument("--eval_problems", type=int, default=40, help="Number of problems to evaluate with pass@k")
    parser.add_argument("--eval_only", action="store_true", help="Run pass@k evaluation only (no training)")
    parser.add_argument("--repeat_eval_problems", type=int, default=4, help="Repeat each eval problem N times for pass@k-like behavior")
    parser.add_argument("--simple_eval_problems", type=int, default=20, help="Number of unique problems for simple repeated evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_generations", type=int, default=8, help="Number of generations per prompt")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per device")
    
    # Weights & Biases arguments
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="q-language-training", help="W&B project name")
    parser.add_argument("--wandb_name", type=str, help="W&B experiment name (run name)")
    parser.add_argument("--wandb_entity", type=str, help="W&B entity (username or team name)")
    parser.add_argument("--wandb_tags", type=str, nargs="*", help="W&B tags for the run")
    
    # LoRA arguments
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA (Low-Rank Adaptation)")
    parser.add_argument("--use_qlora", action="store_true", help="Use QLoRA (4-bit quantized LoRA)")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    
    # Generation and reward arguments
    parser.add_argument("--generation_temp", type=float, default=1.0, help="Temperature for GRPO generation")
    parser.add_argument("--base_reward_weight", type=float, default=1.0, help="Weight for base percentage reward")
    parser.add_argument("--perfect_reward_weight", type=float, default=1.0, help="Weight for perfect solution bonus")
    
    args = parser.parse_args()
    
    logger.info("Starting Simple Q Language Training with TRL")
    logger.info(f"Model: {args.model}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Max train problems: {args.max_train_problems}")
    logger.info(f"Max tests per problem: {args.max_tests_per_problem}")
    logger.info(f"Perfect bonus: {args.perfect_bonus}")
    logger.info(f"Use reasoning format: {args.use_reasoning_format}")
    logger.info(f"Evaluation every: {args.eval_steps} steps")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Use vLLM: {args.use_vllm}")
    logger.info(f"Dummy reward mode: {args.dummy_reward}")
    logger.info(f"Simple eval problems: {args.simple_eval_problems}")
    logger.info(f"Repeat eval problems: {args.repeat_eval_problems}")
    logger.info(f"Total eval samples: {args.simple_eval_problems} × {args.repeat_eval_problems} = {args.simple_eval_problems * args.repeat_eval_problems}")
    logger.info(f"This simulates pass@{args.repeat_eval_problems} behavior with reliable built-in evaluation!")
    logger.info(f"Per device batch size: {args.per_device_train_batch_size}")
    logger.info(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    logger.info(f"Generations per prompt: {args.num_generations}")
    logger.info(f"Generation temperature: {args.generation_temp}")
    logger.info(f"Base reward weight: {args.base_reward_weight}")
    logger.info(f"Perfect reward weight: {args.perfect_reward_weight}")
    
    # Weights & Biases setup
    if args.use_wandb:
        if not WANDB_AVAILABLE:
            logger.error("❌ Weights & Biases requested but not installed. Install with: pip install wandb")
            sys.exit(1)
        
        logger.info("🚀 Initializing Weights & Biases...")
        logger.info(f"Project: {args.wandb_project}")
        if args.wandb_name:
            logger.info(f"Run name: {args.wandb_name}")
        if args.wandb_entity:
            logger.info(f"Entity: {args.wandb_entity}")
        if args.wandb_tags:
            logger.info(f"Tags: {args.wandb_tags}")
        
        # Initialize wandb
        wandb_config = {
            "model": args.model,
            "output_dir": args.output_dir,
            "max_train_problems": args.max_train_problems,
            "max_test_problems": args.max_test_problems,
            "max_tests_per_problem": args.max_tests_per_problem,
            "perfect_bonus": args.perfect_bonus,
            "use_reasoning_format": args.use_reasoning_format,
            "eval_steps": args.eval_steps,
            "learning_rate": args.learning_rate,
            "use_vllm": args.use_vllm,
            "dummy_reward": args.dummy_reward,
            "simple_eval_problems": args.simple_eval_problems,
            "repeat_eval_problems": args.repeat_eval_problems,
            "seed": args.seed,
            "num_generations": args.num_generations,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "generation_temp": args.generation_temp,
            "base_reward_weight": args.base_reward_weight,
            "perfect_reward_weight": args.perfect_reward_weight,
            "use_lora": args.use_lora,
            "use_qlora": args.use_qlora,
        }
        
        if args.use_lora or args.use_qlora:
            wandb_config.update({
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
            })
        
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            entity=args.wandb_entity,
            tags=args.wandb_tags,
            config=wandb_config,
            resume="allow"  # Allow resuming if run with same name exists
        )
        
        # Set environment variable for transformers integration
        os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_name:
            os.environ["WANDB_RUN_NAME"] = args.wandb_name
    else:
        logger.info("📊 Weights & Biases logging disabled. Use --use_wandb to enable.")
    
    # LoRA configuration logging
    if args.use_lora or args.use_qlora:
        logger.info(f"Using {'QLoRA' if args.use_qlora else 'LoRA'} training")
        logger.info(f"LoRA config: rank={args.lora_rank}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
        
        # Warning about vLLM compatibility
        if args.use_vllm:
            logger.warning("⚠️  vLLM is not compatible with LoRA/QLoRA. Disabling vLLM server mode.")
            args.use_vllm = False
    
    # Calculate and log completions per gradient step
    logger.info("Completions per gradient step calculation:")
    logger.info(f"  Formula: per_device_batch_size × gradient_accumulation_steps × num_generations × num_processes")
    logger.info(f"  With 4 processes: {args.per_device_train_batch_size} × {args.gradient_accumulation_steps} × {args.num_generations} × 4 = {args.per_device_train_batch_size * args.gradient_accumulation_steps * args.num_generations * 4}")
    logger.info(f"  With 6 processes: {args.per_device_train_batch_size} × {args.gradient_accumulation_steps} × {args.num_generations} × 6 = {args.per_device_train_batch_size * args.gradient_accumulation_steps * args.num_generations * 6}")
    
    # Load datasets
    logger.info("Loading Q language datasets...")
    train_loader, test_loader = get_q_code_dataloaders(
        max_train_problems=args.max_train_problems,
        max_test_problems=args.max_test_problems,
        seed=args.seed
    )
    print("here12367")
    
    # Convert to list of problems
    train_problems = []
    for batch in train_loader:
        if isinstance(batch, dict) and "id" in batch:
            train_problems.append({
                'id': batch['id'],
                'description': batch['description'],
                'tests': batch['tests']
            })
        else:
            train_problems.append(batch)
    print("here123245")
    
    # Get test problems directly from the QCodeDataLoader
    test_problems = test_loader.problems
    
    logger.info(f"Loaded {len(train_problems)} training problems")
    logger.info(f"Loaded {len(test_problems)} test problems")
    print("here1234")
    
    # Prepare datasets for TRL
    logger.info("Preparing datasets for TRL...")
    train_dataset = prepare_dataset_for_trl(train_problems, args.use_reasoning_format)
    
    # For evaluation: use fewer unique problems but repeat them for pass@k-like behavior
    eval_problems_subset = test_problems[:args.simple_eval_problems]
    eval_dataset = prepare_dataset_for_trl(
        eval_problems_subset, 
        args.use_reasoning_format, 
        repeat_for_pass_at_k=args.repeat_eval_problems
    )
    
    logger.info(f"Training dataset: {len(train_dataset)} samples")
    logger.info(f"Evaluation dataset: {len(eval_problems_subset)} unique problems × {args.repeat_eval_problems} repeats = {len(eval_dataset)} total evaluations")
    logger.info(f"This simulates pass@{args.repeat_eval_problems} evaluation with simple built-in metrics")
    
    # Create reward function
    logger.info("Creating reward function...")
    if args.dummy_reward:
        logger.info("Using dummy reward function (bracket counting)")
        reward_function = create_dummy_reward_function()
    else:
        logger.info("Using Q code test case reward function")
        reward_function = create_q_reward_function(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            use_reasoning_format=args.use_reasoning_format,
            max_tests=args.max_tests_per_problem,
            perfect_bonus=args.perfect_bonus,
            base_reward_weight=args.base_reward_weight,
            perfect_reward_weight=args.perfect_reward_weight
        )
    
    # Test reward function if requested
    if args.test_reward_function:
        test_reward_function_baseline(reward_function, train_dataset)
    
    # Test base model output if requested
    if args.test_reward_function: # Reusing the same flag for now, can be changed if needed
        test_base_model_output(args.model, train_dataset, args.use_reasoning_format)
    
    # Create training configuration
    training_config = {
        "output_dir": args.output_dir,
        "num_train_epochs": 100,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "logging_steps": 1,
        "save_steps": 50,
        "learning_rate": args.learning_rate,
        "max_completion_length": 2048,
        "bf16": True,
        "num_generations": args.num_generations,
        "temperature": args.generation_temp,
    }
    
    # Add Weights & Biases configuration
    if args.use_wandb:
        training_config.update({
            "report_to": "wandb",
            "run_name": args.wandb_name,
            "logging_strategy": "steps",
            "logging_steps": 1,  # Log every step for detailed monitoring
        })
        logger.info("📈 Enabled Weights & Biases reporting in training config")
    else:
        training_config["report_to"] = "none"
    
    # Configure evaluation strategy - simple repeated evaluation for pass@k-like behavior
    training_config.update({
        "eval_strategy": "steps",
        "eval_steps": args.eval_steps,
        "eval_on_start": True,
    })
    logger.info(f"Using simple built-in evaluation with {args.repeat_eval_problems}x repeated problems for pass@{args.repeat_eval_problems}-like behavior")
    
    # Create LoRA configuration separately
    peft_config = None
    model_kwargs = {}
    
    if args.use_lora or args.use_qlora:
        logger.info("Creating LoRA configuration...")
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Add quantization config for QLoRA
        if args.use_qlora:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model_kwargs["quantization_config"] = quantization_config
            logger.info("Added QLoRA quantization configuration")
    
    # Add vLLM configuration if requested
    if args.use_vllm:
        training_config.update({
            "use_vllm": True,
            "vllm_mode": "server",
        })
        logger.info("Added vLLM server configuration")
    
    training_args = GRPOConfig(**training_config)

    # Simple evaluation approach - no complex pass@k callbacks needed
    logger.info("Using simple repeated evaluation approach - no additional callbacks needed")

    # Create trainer
    logger.info("Creating GRPO trainer...")
    trainer_kwargs = {
        "model": args.model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "reward_funcs": reward_function,
    }
    
    # Add PEFT config if using LoRA
    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config
        logger.info("Added PEFT config to trainer")
    
    # Add model kwargs if any (e.g., quantization config)
    if model_kwargs:
        trainer_kwargs["model_kwargs"] = model_kwargs
        logger.info("Added model kwargs to trainer")
    
    trainer = GRPOTrainer(**trainer_kwargs)
    
    # Handle eval-only mode
    if args.eval_only:
        logger.error("--eval_only is disabled in simple evaluation mode")
        logger.info("Use the built-in evaluation during training instead")
        return
    
    # Train
    logger.info("Starting training...")
    
    # Simple evaluation uses built-in TRL evaluation with repeated problems
    logger.info("Using simple built-in evaluation with repeated problems for pass@k-like behavior")
    
    trainer.train()
    
    # Save
    logger.info("Saving model...")
    trainer.save_model()
    
    # Finish wandb run
    if args.use_wandb and wandb is not None:
        logger.info("🏁 Finishing Weights & Biases run...")
        wandb.finish()
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main() 
