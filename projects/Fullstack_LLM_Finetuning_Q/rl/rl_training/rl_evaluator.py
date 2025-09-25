"""
Q code evaluator for RL training.

This evaluator provides a reward signal based on the fraction of test cases passed.
Uses direct Q code execution without LLM validation calls for speed.

Environment Variables:
    Q_INTERPRETER_PATH: Path to Q executable (default: "q")
"""
import torch
import subprocess
import tempfile
import os
import shutil
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import logging
import time
import re
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))
from rl_training.rl_utils import extract_q_code_with_reasoning_support

logger = logging.getLogger(__name__)

# Constants for Q execution
DEFAULT_TIMEOUT = 0.2  # Reduced for faster training
Q_EXECUTABLE = os.getenv("Q_INTERPRETER_PATH", "q")  # Match main config.py pattern

def sanitize_filename(s: str) -> str:
    """Remove problematic characters from filenames."""
    return re.sub(r'[^a-zA-Z0-9_-]', '_', s)

def run_q_code(code: str, timeout: float = DEFAULT_TIMEOUT, task_id: str = "eval") -> Tuple[bool, str, str]:
    """
    Execute Q code and return success status, stdout, stderr.
    
    Args:
        code: Q code to execute
        timeout: Execution timeout in seconds
        task_id: Identifier for temporary files
        
    Returns:
        Tuple of (success, stdout, stderr)
    """
    # Ensure Q code ends with exit 0;
    if not code.strip().endswith("exit 0;"):
        code = code.strip() + "\nexit 0;"
    
    safe_task_id = sanitize_filename(task_id)
    temp_dir = Path(f"./temp_q_{safe_task_id}_{time.time_ns()}")
    temp_dir.mkdir(exist_ok=True)
    temp_script = temp_dir / "script.q"

    try:
        with open(temp_script, 'w') as f:
            f.write(code)

        # Use the configured Q executable path
        q_cmd = Q_EXECUTABLE
        
        process = subprocess.run(
            [q_cmd, temp_script.name],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            cwd=str(temp_dir)
        )
        
        success = process.returncode == 0 and "error" not in process.stderr.lower()
        return success, process.stdout.strip(), process.stderr.strip()
    
    except subprocess.TimeoutExpired:
        return False, "", f"Timeout after {timeout}s"
    except Exception as e:
        return False, "", str(e)
    finally:
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass

class RLEvaluator:
    """
    Evaluator for RL that provides a reward based on test case performance.
    Uses direct Q code execution without LLM validation for speed.
    
    Reward structure:
    - Binary mode: Reward is 1 if all test cases pass, 0 otherwise
    - Test cases mode: Reward is the fraction of test cases passed (e.g., 3/4 passed = 0.75 reward)
    - If code fails to execute, reward is 0.
    """
    
    def __init__(self, reward_method: str = "binary"):
        self.timeout = DEFAULT_TIMEOUT
        self.reward_method = reward_method
        if reward_method not in ["binary", "test_cases"]:
            raise ValueError(f"reward_method must be 'binary' or 'test_cases', got {reward_method}")
    
    def _extract_q_code_for_logging(self, completion: str) -> str:
        """Extract Q code from completion for logging purposes."""
        _, _, q_code = extract_q_code_with_reasoning_support(completion)
        return q_code

    def compute_rewards(
        self,
        completions: List[str],
        tests: List[Dict],
        device: str,
        iteration: int = None,
        debug_output_dir: str = None,
        max_tests: int = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute rewards based on the fraction of passed test cases.
        
        Returns:
            rewards: Tensor of rewards (one per completion)
            metrics: Dictionary with aggregate metrics
        """
        rewards = []
        metrics_counts = {
            'total_tests_run': 0,
            'total_tests_passed': 0,
            'completions_fully_passing': 0,
            'total_completions': len(completions)
        }
        
        # Limit the number of tests if specified
        tests_to_run = tests
        if max_tests is not None and max_tests > 0 and len(tests) > max_tests:
            tests_to_run = tests[:max_tests]

        iteration_debug_dir = None
        if debug_output_dir and iteration is not None:
            iteration_debug_dir = Path(debug_output_dir) / f"iteration_{iteration}"
            iteration_debug_dir.mkdir(parents=True, exist_ok=True)
            
        if not tests_to_run:
            # Return zero rewards if there are no tests to run
            return torch.zeros(len(completions), dtype=torch.float32, device=device), {
                'mean_reward': 0.0,
                'test_case_pass_rate': 0.0,
                'problem_success_rate': 0.0
            }

        for i, completion in enumerate(completions):
            rollout_log = []
            rollout_log.append(f"--- ROLLOUT {i} ---")
            rollout_log.append("=" * 40)
            rollout_log.append("1. RAW MODEL COMPLETION:")
            rollout_log.append(completion)
            rollout_log.append("-" * 20)
            
            passed_count = 0
            # Extract Q code from the completion with reasoning support
            reasoning_part, answer_part, q_code = extract_q_code_with_reasoning_support(completion)
            
            rollout_log.append("2. EXTRACTED PARTS:")
            if reasoning_part:
                rollout_log.append("   REASONING:")
                rollout_log.append(f"   {reasoning_part}")
            if answer_part:
                rollout_log.append("   ANSWER:")
                rollout_log.append(f"   {answer_part}")
            rollout_log.append("   Q CODE:")
            rollout_log.append(q_code)
            rollout_log.append("-" * 20)
            
            if not q_code.strip():
                rewards.append(0.0)
                rollout_log.append("RESULT: No code found. Reward: 0.0")
            else:
                rollout_log.append("3. RUNNING ALL TEST CASES:")
                
                # Remove exit 0; from solution if present (will interfere with test execution)
                if q_code.strip().endswith("exit 0;"):
                    q_code = q_code.strip()[:-7].strip()
                
                for test_info in tests_to_run:
                    test_idx = test_info['test_idx']
                    test_code = test_info['test_code']
                    expected_output = test_info['expected_output']
                    
                    rollout_log.append(f"  - Test Case {test_idx}:")
                    
                    # Combine solution with test harness
                    full_code = f"{q_code}\n\n{test_code}"
                    
                    # Execute the code
                    task_id = f"rl_eval_{i}_t{test_idx}"
                    success, stdout, stderr = run_q_code(full_code, timeout=self.timeout, task_id=task_id)
                    
                    # Check if output matches exactly (after stripping whitespace)
                    test_passed = False
                    if success and stdout is not None:
                        actual_stripped = stdout.strip()
                        expected_stripped = expected_output.strip()
                        
                        # Special case: handle empty string vs empty output
                        # When expected is '""' (empty string literal) and actual is empty
                        if (actual_stripped == "" and expected_stripped == '""') or \
                           (actual_stripped == '""' and expected_stripped == ""):
                            test_passed = True
                        else:
                            test_passed = actual_stripped == expected_stripped
                    
                    rollout_log.append(f"    - FULL SCRIPT: {full_code}")
                    rollout_log.append(f"    - EXPECTED OUTPUT: '{expected_output}'")
                    rollout_log.append(f"    - ACTUAL STDOUT: '{stdout}'")
                    rollout_log.append(f"    - ACTUAL STDERR: '{stderr}'")
                    rollout_log.append(f"    - EXECUTION SUCCESS: {success}")
                    rollout_log.append(f"    - TEST PASSED: {test_passed}")
                    
                    if test_passed:
                        passed_count += 1
                
                metrics_counts['total_tests_run'] += len(tests_to_run)
                metrics_counts['total_tests_passed'] += passed_count
                
                # Calculate reward based on the chosen method
                if self.reward_method == "binary":
                    # All tests must pass for a reward of 1
                    reward = 1.0 if passed_count == len(tests_to_run) else 0.0
                else: # test_cases
                    # Reward is the fraction of tests passed
                    reward = passed_count / len(tests_to_run)
                
                rewards.append(reward)
                
                if reward == 1.0:
                    metrics_counts['completions_fully_passing'] += 1
                    
                rollout_log.append(f"RESULT: Passed {passed_count}/{len(tests_to_run)} tests. Reward: {reward:.2f}")

            if iteration_debug_dir:
                log_file = iteration_debug_dir / f"rollout_{i}.txt"
                with open(log_file, 'w') as f:
                    f.write("\n".join(rollout_log))

        # Calculate final metrics
        mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
        
        total_tests_run = metrics_counts['total_tests_run']
        total_tests_passed = metrics_counts['total_tests_passed']
        total_completions = metrics_counts['total_completions']
        completions_fully_passing = metrics_counts['completions_fully_passing']

        # Avoid division by zero
        test_case_pass_rate = total_tests_passed / total_tests_run if total_tests_run > 0 else 0.0
        problem_success_rate = completions_fully_passing / total_completions if total_completions > 0 else 0.0
        
        metrics = {
            'mean_reward': mean_reward,
            'test_case_pass_rate': test_case_pass_rate,
            'problem_success_rate': problem_success_rate
        }

        return torch.tensor(rewards, dtype=torch.float32, device=device), metrics 