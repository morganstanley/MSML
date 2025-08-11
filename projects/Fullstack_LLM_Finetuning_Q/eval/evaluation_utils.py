#!/usr/bin/env python3
"""
Evaluation Utilities - Common functions for both generation and evaluation phases.
"""

import os
import json
import re
import time
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# With configuration-based path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_config

config = get_config()
Q_PATH = config.dataset.q_interpreter_path

# Configuration  
DEFAULT_TIMEOUT = 2
Q_CMD = "taskset -c 0-23"
Q_EXECUTABLE = f"{Q_CMD} {Q_PATH}"


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file operations."""
    return re.sub(r'[^\w\-_.]', '_', filename)


def load_file_content(file_path: str) -> str:
    """Load content from a file safely."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return ""


def load_test_problems(test_dir: str, max_problems: Optional[int] = None, tasks: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Load test problems from directory."""
    test_path = Path(test_dir)
    if not test_path.exists():
        logger.error(f"Test directory not found: {test_dir}")
        return []
    
    problems = []
    problem_dirs = [d for d in test_path.iterdir() if d.is_dir()]
    if max_problems:
        problem_dirs = problem_dirs[:max_problems]
    
    for problem_dir in problem_dirs:
        problem_name = problem_dir.name
        
        # Check required files exist
        required_files = {
            'problem_description.txt': problem_dir / 'problem_description.txt',
            'python_sol.py': problem_dir / 'python_sol.py',
            'q_sol.q': problem_dir / 'q_sol.q'
        }
        
        if not all(f.exists() for f in required_files.values()):
            logger.debug(f"Missing required files for {problem_name}")
            continue
        
        # Find test case indices
        test_case_indices = []
        for file in problem_dir.iterdir():
            if file.name.startswith('test_case_') and file.name.endswith('.py'):
                try:
                    index = int(file.name.replace('test_case_', '').replace('.py', ''))
                    test_case_indices.append(index)
                except ValueError:
                    continue
        
        if not test_case_indices:
            logger.debug(f"No test cases found for {problem_name}")
            continue
        
        test_case_indices.sort()
        
        # Load problem content
        problem_description = load_file_content(str(required_files['problem_description.txt']))
        python_code = load_file_content(str(required_files['python_sol.py']))
        q_code = load_file_content(str(required_files['q_sol.q']))
        
        # Load test cases
        test_cases = []
        for test_idx in test_case_indices:
            py_test_file = problem_dir / f'test_case_{test_idx}.py'
            q_test_file = problem_dir / f'q_test_case_{test_idx}.q'
            
            # Check for exact answer files (new format) or fallback to old format
            q_exact_file = problem_dir / f'q_exact_ans_test_case_{test_idx}.txt'
            python_exact_file = problem_dir / f'python_exact_ans_test_case_{test_idx}.txt'
            old_answer_file = problem_dir / f'test_case_{test_idx}_correct_ans.txt'
            
            if not all(f.exists() for f in [py_test_file, q_test_file]):
                continue
            
            py_test_content = load_file_content(str(py_test_file))
            q_test_content = load_file_content(str(q_test_file))
            
            # Use exact answer files if available, otherwise fall back to old format
            if q_exact_file.exists() and python_exact_file.exists():
                q_expected_output = load_file_content(str(q_exact_file))
                python_expected_output = load_file_content(str(python_exact_file))
            elif old_answer_file.exists():
                # Use old format for both
                expected_output = load_file_content(str(old_answer_file))
                q_expected_output = expected_output
                python_expected_output = expected_output
            else:
                continue
            
            if all([py_test_content, q_test_content, q_expected_output, python_expected_output]):
                test_cases.append({
                    'index': test_idx,
                    'py_test_content': py_test_content,
                    'q_test_content': q_test_content,
                    'q_expected_output': q_expected_output,
                    'python_expected_output': python_expected_output
                })
        
        if not test_cases:
            logger.debug(f"No valid test cases for {problem_name}")
            continue
        
        # Create problem entries for each task type
        task_types = tasks if tasks else ['description_to_q', 'python_to_q', 'q_to_python']
        
        for task_type in task_types:
            problems.append({
                'problem_name': problem_name,
                'task_type': task_type,
                'problem_description': problem_description,
                'python_code': python_code,
                'q_code': q_code,
                'test_cases': test_cases,
                'total_test_cases': len(test_cases)
            })
    
    logger.info(f"Loaded {len(problems)} test problems")
    return problems


def create_prompt(problem: Dict[str, Any], context: Optional[str] = None, thinking_mode: bool = False) -> str:
    """Create prompt for the given problem and task type."""
    task_type = problem['task_type']
    problem_description = problem['problem_description']
    python_code = problem['python_code']
    q_code = problem['q_code']
    
    # Add context if provided
    context_prefix = ""
    if context:
        context_prefix = f"""Here is reference material for Q programming:

{context}

---

"""
    
    if task_type == 'description_to_q':
        if thinking_mode:
            prompt = f"""{context_prefix}Write a Q solve function that solves the following problem. Use a structured approach with reasoning and answer.

Problem:
{problem_description}

Please provide your response in this format:
<reasoning>
[Your step-by-step reasoning about how to solve this problem]
</reasoning>

<answer>
[Your Q solve function code]
</answer>

Make sure your Q solve function is complete and executable."""
        else:
            prompt = f"""{context_prefix}Write a Q solve function that solves the following problem:

{problem_description}

You must implement a function called 'solve' that accepts the arguments specified in the problem description. The function should return the correct output for the given inputs.

Output ONLY the Q solve function implementation. Do not include any other text, explanations, or test harnesses.

Example format:
solve:{{[args] // your implementation here}}"""

    elif task_type == 'python_to_q':
        if thinking_mode:
            prompt = f"""{context_prefix}Translate the following Python solve function to an equivalent Q solve function. Use a structured approach with reasoning and answer.

Python code:
```python
{python_code}
```

Please provide your response in this format:
<reasoning>
[Your step-by-step reasoning about how to translate this Python code to Q]
</reasoning>

<answer>
[Your Q solve function code]
</answer>

Make sure your Q solve function is complete and executable."""
        else:
            prompt = f"""{context_prefix}Translate the following Python solve function to an equivalent Q solve function:

```python
{python_code}
```

You must implement a Q function called 'solve' that accepts the same arguments as the Python function and returns equivalent results. Pay attention to the function signature and argument types.

Output ONLY the Q solve function implementation. Do not include any other text, explanations, or test harnesses.

Example format:
solve:{{[args] // your implementation here}}"""

    elif task_type == 'q_to_python':
        if thinking_mode:
            prompt = f"""{context_prefix}Translate the following Q solve function to an equivalent Python solve function. Use a structured approach with reasoning and answer.

Q code:
```q
{q_code}
```

Please provide your response in this format:
<reasoning>
[Your step-by-step reasoning about how to translate this Q code to Python]
</reasoning>

<answer>
[Your Python solve function code]
</answer>

Make sure your Python solve function is complete and executable."""
        else:
            prompt = f"""{context_prefix}Translate the following Q solve function to an equivalent Python solve function:

```q
{q_code}
```

You must implement a Python function called 'solve' that accepts the same arguments as the Q function and returns equivalent results.

Output ONLY the Python solve function implementation. Do not include any other text, explanations, or test harnesses.

Example format:
def solve(args):
    # your implementation here"""

    return prompt


def extract_q_code_with_reasoning_support(response: str) -> Tuple[str, str, str]:
    """Extract Q code from response with reasoning support."""
    reasoning_part = ""
    answer_part = ""
    q_code = ""
    
    # Extract reasoning section
    reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', response, re.DOTALL | re.IGNORECASE)
    if reasoning_match:
        reasoning_part = reasoning_match.group(1).strip()
    
    # Extract answer section
    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL | re.IGNORECASE)
    if answer_match:
        answer_part = answer_match.group(1).strip()
        q_code = extract_code_from_response(answer_part, "q", False)
    else:
        # If no answer tags found, try to extract Q code from the full response
        q_code = extract_code_from_response(response, "q", False)
    
    return reasoning_part, answer_part, q_code


def extract_code_from_response(response: str, language: str = None, thinking_mode: bool = False) -> str:
    """Extract code from LLM response, removing markdown code blocks."""
    if thinking_mode:
        # Use reasoning format extraction
        reasoning_part, answer_part, q_code = extract_q_code_with_reasoning_support(response)
        if language == "python":
            # For python, still try to extract from answer part or full response
            if answer_part:
                return extract_code_from_response(answer_part, language, False)
            else:
                return extract_code_from_response(response, language, False)
        else:
            # For Q code, use the extracted q_code
            return q_code
    
    print(response)
    # Original logic for non-thinking mode
    if language:
        pattern = f"```{language}(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return matches[0].strip()
    
    # Try generic code blocks
    matches = re.findall(r"```(.*?)```", response, re.DOTALL)
    if matches:
        return matches[0].strip()
    
    return response.strip()


def run_q_code(code: str, timeout: int = DEFAULT_TIMEOUT, task_id: str = "eval") -> Tuple[bool, str, str]:
    """Execute Q code and return success status, stdout, stderr."""
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

        process = subprocess.run(
            Q_EXECUTABLE.split() + [temp_script.name],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            cwd=str(temp_dir)
        )
        
        # Check for license daemon connection issues (indicates system overload)
        stderr_clean = process.stderr.strip()
        if "couldn't connect to license daemon" in stderr_clean:
            # This is a system/infrastructure issue, not a code issue
            return False, "", "LICENSE_DAEMON_OVERLOAD: " + stderr_clean
        
        success = process.returncode == 0 and "error" not in process.stderr.lower()
        return success, process.stdout.strip(), stderr_clean
    
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


def run_python_code(code: str, timeout: int = DEFAULT_TIMEOUT, task_id: str = "eval") -> Tuple[bool, str, str]:
    """Execute Python code and return success status, stdout, stderr."""
    safe_task_id = sanitize_filename(task_id)
    temp_dir = Path(f"./temp_py_{safe_task_id}_{time.time_ns()}")
    temp_dir.mkdir(exist_ok=True)
    temp_script = temp_dir / "script.py"

    try:
        with open(temp_script, 'w') as f:
            f.write(code)

        process = subprocess.run(
            ["python", temp_script.name],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            cwd=str(temp_dir)
        )

        success = process.returncode == 0
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


def get_file_extension(task_type: str) -> str:
    """Get appropriate file extension for task type."""
    if task_type in ['description_to_q', 'python_to_q']:
        return '.q'
    else:  # q_to_python
        return '.py'


def get_problem_id(problem: Dict[str, Any]) -> str:
    """Get unique identifier for a problem."""
    return f"{problem['problem_name']}_{problem['task_type']}" 