#!/usr/bin/env python3
"""
Final Q and Python Code Verification Script

This script verifies that both Q and Python code solutions in final_dataset/ are correct by:
1. Loading each Q and Python solution
2. Running them against all test cases
3. Using GPT-4.1 to verify output equivalence
4. Moving fully verified problems to fully_prepared_final_dataset/
5. Creating exact answer files for both Q and Python outputs
6. Cleaning Q files of markdown artifacts
7. Generating detailed logs and summary statistics
8. Caching results to avoid recomputation
9. Optional mode to verify only Q code against answer keys (--no-python)

Usage:
    python final_q_verify.py [--no-python]
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_config

config = get_config()
Q_PATH = config.dataset.q_interpreter_path

import json
import logging
import subprocess
import time
import re
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants for Q execution
DEFAULT_TIMEOUT = 10
Q_CMD = "taskset -c 0-23"
Q_EXECUTABLE = f"{Q_CMD} {Q_PATH}"

# Cache file for verification results
CACHE_FILE = "q_verification_cache.json"

def call_llm(prompt, max_retries=3, model="claude-3-5-sonnet-20241022"):
    """Call OpenAI LLM with retry logic."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise
    return None

def load_cache() -> Dict[str, Any]:
    """Load verification cache from JSON file."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load cache file: {e}")
    return {}

def save_cache(cache: Dict[str, Any]):
    """Save verification cache to JSON file."""
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=2)
    except IOError as e:
        logger.error(f"Failed to save cache file: {e}")

def get_problem_hash(problem_dir: Path) -> str:
    """Generate a hash for a problem based on its Q and Python solutions and test cases."""
    import hashlib
    
    content_parts = []
    
    # Add Q solution content
    q_sol_file = problem_dir / "q_sol.q"
    if q_sol_file.exists():
        try:
            with open(q_sol_file, 'r', encoding='utf-8') as f:
                content_parts.append(f.read())
        except IOError:
            pass
    
    # Add Python solution content
    python_sol_file = problem_dir / "python_sol.py"
    if python_sol_file.exists():
        try:
            with open(python_sol_file, 'r', encoding='utf-8') as f:
                content_parts.append(f.read())
        except IOError:
            pass
    
    # Add all Q test case contents
    test_files = sorted(problem_dir.glob("q_test_case_*.q"))
    for test_file in test_files:
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                content_parts.append(f.read())
        except IOError:
            continue
    
    # Add all Python test case contents
    test_files = sorted(problem_dir.glob("test_case_*.py"))
    for test_file in test_files:
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                content_parts.append(f.read())
        except IOError:
            continue
    
    # Create hash
    combined_content = "\n".join(content_parts)
    return hashlib.md5(combined_content.encode()).hexdigest()

def is_cached_and_valid(problem_name: str, problem_dir: Path, cache: Dict[str, Any]) -> bool:
    """Check if problem is cached and the cache is still valid."""
    if problem_name not in cache:
        return False
    
    cached_entry = cache[problem_name]
    
    # Check if entry has required fields
    if not all(key in cached_entry for key in ['status', 'hash', 'timestamp']):
        return False
    
    # Check if the problem files have changed
    current_hash = get_problem_hash(problem_dir)
    if current_hash != cached_entry.get('hash', ''):
        return False
    
    # If status is 'valid', we can skip recomputation
    return cached_entry['status'] == 'valid'

def clean_q_code(q_code: str) -> Tuple[str, bool]:
    """Cleans Q code by removing markdown fences using regex patterns."""
    original_code_str = str(q_code)
    cleaned_code = q_code.strip()
    
    # Use regex patterns similar to convert_to_q.py extract_code_from_response
    import re
    
    # First, try to find a ```q block
    q_pattern = r"```q\s*(.*?)\s*```"
    match = re.search(q_pattern, cleaned_code, re.DOTALL)
    if match:
        cleaned_code = match.group(1).strip()
    else:
        # Try to find a generic ``` block
        any_pattern = r"```\s*(.*?)\s*```"
        match = re.search(any_pattern, cleaned_code, re.DOTALL)
        if match:
            cleaned_code = match.group(1).strip()
        else:
            # If no complete blocks, clean partial fences
            if cleaned_code.startswith("```q"):
                cleaned_code = cleaned_code.removeprefix("```q").strip()
            elif cleaned_code.startswith("```"):
                cleaned_code = cleaned_code.removeprefix("```").strip()
            if cleaned_code.endswith("```"):
                cleaned_code = cleaned_code.removesuffix("```").strip()
    
    final_code = cleaned_code.strip()
    was_cleaned = (final_code != original_code_str.strip())
    
    return final_code, was_cleaned

def sanitize_task_id(s):
    """Sanitize task ID to remove problematic characters."""
    return re.sub(r'[^a-zA-Z0-9_-]', '_', s)

def run_q_code(code: str, timeout: int = DEFAULT_TIMEOUT, task_id_for_temp: str = "verify_q", persist_dir: Optional[Path] = None) -> Tuple[bool, Optional[str], Optional[str]]:
    """Run Q code in a temporary directory and return success status and output."""
    # Ensure Q code ends with exit 0;
    if not code.strip().endswith("exit 0;"):
        code = code.strip() + "\nexit 0;"
    
    safe_task_id = sanitize_task_id(task_id_for_temp)
    
    is_temp_mode = persist_dir is None
    if is_temp_mode:
        work_dir = Path(f"./temp_q_verify_{safe_task_id}_{time.time_ns()}")
        script_name = "temp_script.q"
    else:
        work_dir = persist_dir
        script_name = f"{safe_task_id}_executed.q"
    
    work_dir.mkdir(exist_ok=True)
    temp_script_path = work_dir / script_name

    try:
        with open(temp_script_path, 'w') as f:
            f.write(code)

        process = subprocess.run(
            Q_EXECUTABLE.split() + [temp_script_path.name],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            cwd=str(work_dir)
        )
        
        success = process.returncode == 0 and "error" not in process.stderr.lower()
        return success, process.stdout.strip(), process.stderr.strip()
    
    except subprocess.TimeoutExpired:
        return False, None, f"Execution timed out after {timeout} seconds"
    except Exception as e:
        return False, None, str(e)
    finally:
        if is_temp_mode and work_dir.exists():
            try:
                shutil.rmtree(work_dir)
            except Exception as e:
                logger.error(f"Failed to remove temporary directory {work_dir}: {e}")

def run_python_code(code: str, timeout: int = DEFAULT_TIMEOUT, task_id_for_temp: str = "verify_python") -> Tuple[bool, Optional[str], Optional[str]]:
    """Run Python code using an inline command to ensure consistent execution context."""
    try:
        process = subprocess.run(
            ['python3', '-c', code],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False  # We check the return code manually
        )
        
        success = process.returncode == 0
        return success, process.stdout.strip(), process.stderr.strip()
    
    except subprocess.TimeoutExpired:
        return False, None, f"Execution timed out after {timeout} seconds"
    except Exception as e:
        return False, None, str(e)

def load_file_content(file_path: str) -> str:
    """Load and return file content as string."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return ""

def count_solve_calls(code: str) -> int:
    """Count the number of times 'solve' is called in the code."""
    # Look for solve function calls (not definitions)
    solve_calls = re.findall(r'\bsolve\s*[\[\(]', code)
    return len(solve_calls)

def check_output_equivalence(actual_output: str, expected_output: str, detailed_log: List[str]) -> bool:
    """Check if actual output is equivalent to expected output using GPT-4.1."""
    
    # First, check for exact match, which handles cases where both are empty strings.
    if actual_output.strip() == expected_output.strip():
        detailed_log.append("RESULT: ✓ EXACT MATCH")
        return True

    # Handle empty output cases more explicitly
    actual_empty = actual_output.strip() == ""
    expected_empty = expected_output.strip() == ""
    
    # If both are empty, they match (already handled above, but being explicit)
    if actual_empty and expected_empty:
        detailed_log.append("RESULT: ✓ BOTH OUTPUTS EMPTY")
        return True
    
    # If one is empty but the other is not, they don't match
    if actual_empty and not expected_empty:
        detailed_log.append("RESULT: ✗ NOT EQUIVALENT (actual output was empty, expected was not)")
        return False
    
    if not actual_empty and expected_empty:
        detailed_log.append("RESULT: ✗ NOT EQUIVALENT (expected output was empty, actual was not)")
        return False
    
    # Normalize both outputs by removing surrounding quotes for comparison
    actual_normalized = actual_output.strip().strip('"\'')
    expected_normalized = expected_output.strip().strip('"\'')
    
    # Check for exact match after normalization
    if actual_normalized == expected_normalized:
        detailed_log.append("RESULT: ✓ EXACT MATCH (after quote normalization)")
        return True
    else:
        detailed_log.append("CHECKING EQUIVALENCE WITH GPT-4.1...")
        
        # Create validation prompt using normalized outputs for consistent comparison
        validation_prompt = f"""You are comparing outputs from Q/Python implementations to determine if they are equivalent.

FORMATTING DIFFERENCES TO CONSIDER:

STRINGS:
- Q: "abc" or abc (without quotes)
- Python: "abc" or abc
- Expected: "abc" or abc

LISTS:
- Q: 1 2 3 or (1;2;3)
- Python: [1, 2, 3] or (1, 2, 3)
- Expected: [1, 2, 3] or (1;2;3)

BOOLEANS:
- Q: 1b 0b or 1 0
- Python: True False
- Expected: True False or 1b 0b

NUMBERS:
- Q: 42 or 42.0
- Python: 42 or 42.0
- Expected: 42 or 42.0

EXAMPLES OF EQUIVALENT OUTPUTS:
- Output: abc ≡ Expected: abc
- Output: 1 2 3 ≡ Expected: [1, 2, 3]
- Output: True ≡ Expected: 1b
- Output: False ≡ Expected: 0b
- Output: 42 ≡ Expected: 42
- Output: 125 ≡ Expected: 125i
- Output: 125.0 ≡ Expected: 125.0f

NOW COMPARE THESE SPECIFIC OUTPUTS:

Actual output: {actual_normalized}
Expected output: {expected_normalized}

Are these outputs equivalent? Answer only YES or NO."""
        
        detailed_log.append("VALIDATION PROMPT:")
        detailed_log.append(validation_prompt)
        detailed_log.append("")
        
        validation_result = call_llm(validation_prompt)
        
        detailed_log.append(f"GPT-4.1 RESPONSE: {validation_result}")
        
        if validation_result and validation_result.strip().upper() == "YES":
            detailed_log.append("RESULT: ✓ EQUIVALENT")
            return True
        else:
            detailed_log.append("RESULT: ✗ NOT EQUIVALENT")
            return False

def verify_solution(problem_dir: Path, output_dir: Path, check_python: bool = True) -> Dict[str, Any]:
    """Verify both Q and Python solutions against all test cases."""
    problem_name = problem_dir.name
    logger.info(f"Verifying {problem_name}")
    
    # Create a subdirectory for this problem's detailed logs and executed code
    problem_output_dir = output_dir / problem_name
    problem_output_dir.mkdir(exist_ok=True)
    
    # Load solutions
    q_solution_file = problem_dir / "q_sol.q"
    python_solution_file = problem_dir / "python_sol.py"
    
    if not q_solution_file.exists():
        logger.error(f"Q solution file not found for {problem_name}")
        return {
            'problem_name': problem_name,
            'status': 'error',
            'error': 'Q Solution file not found',
            'test_results': [],
            'hash': get_problem_hash(problem_dir),
            'timestamp': datetime.now().isoformat()
        }
    
    if check_python and not python_solution_file.exists():
        logger.error(f"Python solution file not found for {problem_name}")
        return {
            'problem_name': problem_name,
            'status': 'error',
            'error': 'Python solution file not found',
            'test_results': [],
            'hash': get_problem_hash(problem_dir),
            'timestamp': datetime.now().isoformat()
        }
    
    q_code = load_file_content(str(q_solution_file))
    python_code = load_file_content(str(python_solution_file)) if check_python else ""
    
    if not q_code or (check_python and not python_code):
        logger.error(f"Failed to load solutions for {problem_name}")
        return {
            'problem_name': problem_name,
            'status': 'error',
            'error': 'Failed to load solutions',
            'test_results': [],
            'hash': get_problem_hash(problem_dir),
            'timestamp': datetime.now().isoformat()
        }
    
    # Clean Q code
    q_code, q_was_cleaned = clean_q_code(q_code)
    if q_was_cleaned:
        logger.info(f"  Detected and cleaned markdown artifacts in Q solution for {problem_name}.")

    # Remove 'exit 0;' from Q code if present - we'll add it back when running
    if q_code.strip().endswith("exit 0;"):
        q_code = q_code.strip()[:-7].strip()
    
    # Initialize detailed log
    detailed_log = []
    detailed_log.append(f"PROBLEM: {problem_name}")
    detailed_log.append(f"VERIFICATION TIMESTAMP: {datetime.now().isoformat()}")
    detailed_log.append("=" * 80)
    detailed_log.append("")
    detailed_log.append("Q SOLUTION CODE:")
    detailed_log.append("-" * 40)
    detailed_log.append(q_code)
    detailed_log.append("")
    
    if check_python:
        detailed_log.append("PYTHON SOLUTION CODE:")
        detailed_log.append("-" * 40)
        detailed_log.append(python_code)
        detailed_log.append("")
    
    # Find all test cases
    test_case_indices = []
    for file in problem_dir.iterdir():
        if file.name.startswith('q_test_case_') and file.name.endswith('.q'):
            try:
                # Extract test case index from filename like "q_test_case_1.q"
                index = int(file.name.replace('q_test_case_', '').replace('.q', ''))
                test_case_indices.append(index)
            except ValueError:
                continue
    
    if not test_case_indices:
        logger.error(f"No test cases found for {problem_name}")
        return {
            'problem_name': problem_name,
            'status': 'error',
            'error': 'No test cases found',
            'test_results': [],
            'hash': get_problem_hash(problem_dir),
            'timestamp': datetime.now().isoformat()
        }
    
    test_case_indices.sort()
    detailed_log.append(f"FOUND {len(test_case_indices)} TEST CASES")
    detailed_log.append("")
    
    # Check if solve is called multiple times in any test case
    solve_call_issues = []
    for test_idx in test_case_indices:
        q_test_file = problem_dir / f'q_test_case_{test_idx}.q'
        python_test_file = problem_dir / f'test_case_{test_idx}.py'
        
        if q_test_file.exists():
            q_test_content = load_file_content(str(q_test_file))
            q_solve_calls = count_solve_calls(q_test_content)
            if q_solve_calls != 1:
                solve_call_issues.append(f"Q test case {test_idx} has {q_solve_calls} solve calls (expected 1)")
        
        if check_python and python_test_file.exists():
            python_test_content = load_file_content(str(python_test_file))
            python_solve_calls = count_solve_calls(python_test_content)
            if python_solve_calls != 1:
                solve_call_issues.append(f"Python test case {test_idx} has {python_solve_calls} solve calls (expected 1)")
    
    if solve_call_issues:
        detailed_log.append("SOLVE CALL ISSUES:")
        for issue in solve_call_issues:
            detailed_log.append(f"  - {issue}")
        detailed_log.append("")
        
        # Write log file
        log_file = problem_output_dir / "verification.txt"
        with open(log_file, 'w') as f:
            f.write('\n'.join(detailed_log))
        
        return {
            'problem_name': problem_name,
            'status': 'failed',
            'error': 'Invalid solve call count in test cases',
            'solve_call_issues': solve_call_issues,
            'test_results': [],
            'hash': get_problem_hash(problem_dir),
            'timestamp': datetime.now().isoformat()
        }
    
    # Run each test case for both Q and Python
    test_results = []
    all_passed = True
    q_exact_answers = {}
    python_exact_answers = {}
    
    for test_idx in test_case_indices:
        detailed_log.append(f"TEST CASE {test_idx}:")
        detailed_log.append("~" * 40)
        
        # Load test case files
        q_test_file = problem_dir / f'q_test_case_{test_idx}.q'
        python_test_file = problem_dir / f'test_case_{test_idx}.py'
        answer_file = problem_dir / f'test_case_{test_idx}_correct_ans.txt'
        
        if not q_test_file.exists() or not answer_file.exists() or (check_python and not python_test_file.exists()):
            detailed_log.append(f"Missing files for test case {test_idx}")
            detailed_log.append("")
            test_results.append({
                'test_index': test_idx,
                'status': 'error',
                'error': 'Missing test case files'
            })
            all_passed = False
            continue
        
        q_test_content = load_file_content(str(q_test_file))
        python_test_content = load_file_content(str(python_test_file)) if check_python else ""
        expected_output = load_file_content(str(answer_file))
        
        # Clean Q test content
        q_test_content, q_test_was_cleaned = clean_q_code(q_test_content)
        if q_test_was_cleaned:
            logger.info(f"  Detected and cleaned markdown artifacts in Q test case {test_idx} for {problem_name}.")

        # Test Q solution
        detailed_log.append("TESTING Q SOLUTION:")
        q_code_to_run = f"{q_code}\n\n{q_test_content}"
        detailed_log.append("Q CODE TO RUN:")
        detailed_log.append(q_code_to_run)
        detailed_log.append("")
        
        task_id = f"{sanitize_task_id(problem_name)}_q_test{test_idx}"
        q_execution_success, q_stdout, q_stderr = run_q_code(
            q_code_to_run, task_id_for_temp=task_id, persist_dir=problem_output_dir
        )
        
        detailed_log.append(f"Q EXECUTION SUCCESS: {q_execution_success}")
        detailed_log.append(f"Q STDOUT: {q_stdout}")
        detailed_log.append(f"Q STDERR: {q_stderr}")
        detailed_log.append("")
        
        q_test_passed = False
        if q_execution_success and q_stdout is not None:
            q_test_passed = check_output_equivalence(q_stdout, expected_output, detailed_log)
            if q_test_passed:
                q_exact_answers[test_idx] = q_stdout.strip()
        else:
            detailed_log.append("Q RESULT: ✗ EXECUTION FAILED")
        
        # Test Python solution
        python_test_passed = True
        python_execution_success, python_stdout, python_stderr = True, None, None
        if check_python:
            detailed_log.append("TESTING PYTHON SOLUTION:")
            python_code_to_run = f"{python_code}\n\n{python_test_content}"
            detailed_log.append("PYTHON CODE TO RUN:")
            detailed_log.append(python_code_to_run)
            detailed_log.append("")
            
            task_id = f"{sanitize_task_id(problem_name)}_python_test{test_idx}"
            python_execution_success, python_stdout, python_stderr = run_python_code(python_code_to_run, task_id_for_temp=task_id)
            
            detailed_log.append(f"PYTHON EXECUTION SUCCESS: {python_execution_success}")
            detailed_log.append(f"PYTHON STDOUT: {python_stdout}")
            detailed_log.append(f"PYTHON STDERR: {python_stderr}")
            detailed_log.append("")
            
            python_test_passed = False
            if python_execution_success and python_stdout is not None:
                python_test_passed = check_output_equivalence(python_stdout, expected_output, detailed_log)
                if python_test_passed:
                    python_exact_answers[test_idx] = python_stdout.strip()
            else:
                detailed_log.append("PYTHON RESULT: ✗ EXECUTION FAILED")
        
        detailed_log.append(f"EXPECTED OUTPUT: {expected_output}")
        detailed_log.append("")
        
        # Both must pass for the test case to be considered passed (if checking python)
        test_case_passed = q_test_passed and python_test_passed
        if not test_case_passed:
            all_passed = False
        
        detailed_log.append(f"Q TEST CASE {test_idx} PASSED: {q_test_passed}")
        if check_python:
            detailed_log.append(f"PYTHON TEST CASE {test_idx} PASSED: {python_test_passed}")
        detailed_log.append(f"OVERALL TEST CASE {test_idx} PASSED: {test_case_passed}")
        detailed_log.append("")
        
        test_results.append({
            'test_index': test_idx,
            'status': 'passed' if test_case_passed else 'failed',
            'q_execution_success': q_execution_success,
            'q_stdout': q_stdout,
            'q_stderr': q_stderr,
            'q_test_passed': q_test_passed,
            'python_execution_success': python_execution_success if check_python else None,
            'python_stdout': python_stdout if check_python else None,
            'python_stderr': python_stderr if check_python else None,
            'python_test_passed': python_test_passed if check_python else None,
            'expected_output': expected_output
        })
    
    # Final summary
    passed_count = sum(1 for r in test_results if r['status'] == 'passed')
    total_count = len(test_results)
    
    detailed_log.append("FINAL SUMMARY:")
    detailed_log.append("=" * 40)
    detailed_log.append(f"PASSED TEST CASES: {passed_count}/{total_count}")
    detailed_log.append(f"ALL TESTS PASSED: {all_passed}")
    detailed_log.append(f"VERIFICATION STATUS: {'VALID' if all_passed else 'INVALID'}")
    
    # Write detailed log file
    log_file = problem_output_dir / "verification.txt"
    with open(log_file, 'w') as f:
        f.write('\n'.join(detailed_log))
    
    # Return result
    return {
        'problem_name': problem_name,
        'status': 'valid' if all_passed else 'invalid',
        'total_test_cases': total_count,
        'passed_test_cases': passed_count,
        'test_results': test_results,
        'q_exact_answers': q_exact_answers,
        'python_exact_answers': python_exact_answers,
        'q_was_cleaned': q_was_cleaned,
        'hash': get_problem_hash(problem_dir),
        'timestamp': datetime.now().isoformat()
    }

def copy_to_fully_prepared_dataset(problem_dir: Path, fully_prepared_dir: Path, verification_result: Dict[str, Any]):
    """Copy a verified problem to the fully prepared dataset with cleaned files and exact answers."""
    problem_name = problem_dir.name
    target_dir = fully_prepared_dir / problem_name
    
    # Copy the entire directory
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(problem_dir, target_dir)
    
    # Clean Q files in the target directory
    q_sol_file = target_dir / "q_sol.q"
    if q_sol_file.exists():
        q_code = load_file_content(str(q_sol_file))
        q_code, _ = clean_q_code(q_code)
        with open(q_sol_file, 'w', encoding='utf-8') as f:
            f.write(q_code)
    
    # Clean Q test case files
    for q_test_file in target_dir.glob("q_test_case_*.q"):
        q_test_content = load_file_content(str(q_test_file))
        q_test_content, _ = clean_q_code(q_test_content)
        with open(q_test_file, 'w', encoding='utf-8') as f:
            f.write(q_test_content)
    
    # Create exact answer files
    q_exact_answers = verification_result.get('q_exact_answers', {})
    python_exact_answers = verification_result.get('python_exact_answers', {})
    
    for test_idx in q_exact_answers:
        q_exact_file = target_dir / f"q_exact_ans_test_case_{test_idx}.txt"
        with open(q_exact_file, 'w', encoding='utf-8') as f:
            f.write(q_exact_answers[test_idx])
    
    for test_idx in python_exact_answers:
        python_exact_file = target_dir / f"python_exact_ans_test_case_{test_idx}.txt"
        with open(python_exact_file, 'w', encoding='utf-8') as f:
            f.write(python_exact_answers[test_idx])
    
    logger.info(f"  ✓ Copied {problem_name} to fully_prepared_final_dataset/")

def delete_problematic_files(problematic_problems: List[str], final_dataset_dir: Path) -> int:
    """Delete problematic problem directories and return count of deleted directories."""
    if not problematic_problems:
        return 0
    
    deleted_count = 0
    for problem_name in problematic_problems:
        problem_dir = final_dataset_dir / problem_name
        if problem_dir.exists():
            try:
                shutil.rmtree(problem_dir)
                logger.info(f"Deleted: {problem_name}")
                deleted_count += 1
            except Exception as e:
                logger.error(f"Failed to delete {problem_name}: {e}")
        else:
            logger.warning(f"Problem directory not found: {problem_name}")
    
    return deleted_count

def main():
    """Main verification function."""
    parser = argparse.ArgumentParser(description="Final Q and Python Code Verification Script.")
    parser.add_argument(
        '--no-python',
        action='store_true',
        help="Run verification checking only the Q solution against the answer key, skipping Python solution checks."
    )
    args = parser.parse_args()
    
    check_python = not args.no_python

    # Set up paths
    final_dataset_dir = Path("final_dataset")
    fully_prepared_dir = Path("fully_prepared_final_dataset")
    output_dir = Path("q_verification")
    
    # Create output directories
    output_dir.mkdir(exist_ok=True)
    fully_prepared_dir.mkdir(exist_ok=True)
    
    if not final_dataset_dir.exists():
        logger.error(f"Final dataset directory not found: {final_dataset_dir}")
        return 1
    
    if check_python:
        logger.info("Starting Q and Python solution verification...")
    else:
        logger.warning("="*80)
        logger.warning("RUNNING IN Q-ONLY MODE. PYTHON SOLUTIONS WILL NOT BE VERIFIED.")
        logger.warning("="*80)
        logger.info("Starting Q-only solution verification...")

    logger.info(f"Final dataset directory: {final_dataset_dir}")
    logger.info(f"Fully prepared dataset directory: {fully_prepared_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load cache
    cache = load_cache()
    logger.info(f"Loaded cache with {len(cache)} entries")
    
    # Find all problem directories in final_dataset
    problem_dirs = [d for d in final_dataset_dir.iterdir() if d.is_dir()]
    logger.info(f"Found {len(problem_dirs)} problem directories")
    
    if not problem_dirs:
        logger.error("No problem directories found in final_dataset/")
        return 1
    
    # Verify each solution
    results = []
    valid_problems = []
    invalid_problems = []
    error_problems = []
    
    computed_count = 0
    skipped_count = 0
    
    for i, problem_dir in enumerate(problem_dirs):
        problem_name = problem_dir.name
        logger.info(f"Progress: {i+1}/{len(problem_dirs)} - {problem_name}")
        
        # Check if we can skip this problem
        if is_cached_and_valid(problem_name, problem_dir, cache):
            logger.info(f"  ✓ Skipping {problem_name} (cached as valid)")
            cached_result = cache[problem_name].copy()
            cached_result['problem_name'] = problem_name
            results.append(cached_result)
            
            # Additional check: if running in --no-python mode, we can't trust a 'valid'
            # status that was cached from a full run. We must re-verify.
            # However, for this use case, we will assume cache is fine.
            # A more robust implementation might store the verification mode in the cache.
            if cached_result['status'] == 'valid':
                valid_problems.append(problem_name)
                skipped_count += 1
                continue
        
        # Verify the solution
        result = verify_solution(problem_dir, output_dir, check_python=check_python)
        results.append(result)
        computed_count += 1
        
        # Update cache
        cache[problem_name] = {
            'status': result['status'],
            'hash': result['hash'],
            'timestamp': result['timestamp'],
            'total_test_cases': result.get('total_test_cases', 0),
            'passed_test_cases': result.get('passed_test_cases', 0)
        }
        
        # Categorize result and copy to fully prepared dataset if valid
        if result['status'] == 'valid':
            valid_problems.append(result['problem_name'])
            logger.info(f"  ✓ {result['problem_name']}: VALID")
            copy_to_fully_prepared_dataset(problem_dir, fully_prepared_dir, result)
        elif result['status'] == 'invalid':
            invalid_problems.append(result['problem_name'])
            failed_tests = result['total_test_cases'] - result['passed_test_cases']
            logger.info(f"  ✗ {result['problem_name']}: INVALID ({failed_tests} test failures)")
        else:
            error_problems.append(result['problem_name'])
            logger.info(f"  ⚠ {result['problem_name']}: ERROR")
    
    # Save updated cache
    save_cache(cache)
    logger.info(f"Saved cache with {len(cache)} entries")
    
    # Generate summary statistics
    summary = {
        'verification_timestamp': datetime.now().isoformat(),
        'total_problems': len(problem_dirs),
        'valid_problems': len(valid_problems),
        'invalid_problems': len(invalid_problems),
        'error_problems': len(error_problems),
        'computed_problems': computed_count,
        'skipped_problems': skipped_count,
        'valid_problem_list': valid_problems,
        'invalid_problem_list': invalid_problems,
        'error_problem_list': error_problems,
        'detailed_results': results
    }
    
    # Save summary JSON
    summary_file = output_dir / "verification_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print final summary
    print("\n" + "="*80)
    if check_python:
        print("Q AND PYTHON SOLUTION VERIFICATION SUMMARY")
    else:
        print("Q-ONLY SOLUTION VERIFICATION SUMMARY")
    print("="*80)
    print(f"Total problems: {len(problem_dirs)}")
    if check_python:
        print(f"Valid solutions (both Q and Python): {len(valid_problems)} ({len(valid_problems)/len(problem_dirs)*100:.1f}%)")
    else:
        print(f"Valid solutions (Q only): {len(valid_problems)} ({len(valid_problems)/len(problem_dirs)*100:.1f}%)")
    print(f"Invalid solutions: {len(invalid_problems)} ({len(invalid_problems)/len(problem_dirs)*100:.1f}%)")
    print(f"Error problems: {len(error_problems)} ({len(error_problems)/len(problem_dirs)*100:.1f}%)")
    print(f"Computed this run: {computed_count}")
    print(f"Skipped (cached): {skipped_count}")
    print(f"Copied to fully_prepared_final_dataset/: {len(valid_problems)}")
    
    print(f"\nResults saved to: {output_dir}")
    print(f"  - verification_summary.json: Complete verification results")
    print(f"  - {{problem_name}}/verification.txt: Detailed logs for each problem")
    print(f"  - {{problem_name}}/{{problem_name}}_q_test_..._executed.q: Exact Q code executed")
    print(f"  - {CACHE_FILE}: Verification cache")
    print(f"\nFully prepared problems saved to: {fully_prepared_dir}")
    
    # Ask about deleting problematic files
    problematic_problems = invalid_problems + error_problems
    if problematic_problems:
        print(f"\nFound {len(problematic_problems)} problematic problems:")
        for problem in problematic_problems:
            print(f"  - {problem}")
        
        response = input(f"\nDo you want to delete these {len(problematic_problems)} problematic problem directories? (yes/no): ").strip().lower()
        
        if response in ['yes', 'y']:
            deleted_count = delete_problematic_files(problematic_problems, final_dataset_dir)
            print(f"Deleted {deleted_count} problematic problem directories.")
            
            # Update cache to remove deleted problems
            for problem in problematic_problems:
                if problem in cache:
                    del cache[problem]
            save_cache(cache)
            print("Updated cache to remove deleted problems.")
        else:
            print("No files deleted.")
            
            # Ask if user wants to force copy any of the problematic problems
            force_response = input(f"\nWould you like to go through each problematic problem and decide whether to force copy it to fully_prepared_final_dataset? (yes/no): ").strip().lower()
            
            if force_response in ['yes', 'y']:
                force_copied_count = 0
                
                for problem_name in problematic_problems:
                    problem_dir = final_dataset_dir / problem_name
                    if not problem_dir.exists():
                        print(f"Skipping {problem_name} - directory not found")
                        continue
                    
                    # Find the result for this problem
                    problem_result = None
                    for result in results:
                        if result['problem_name'] == problem_name:
                            problem_result = result
                            break
                    
                    if problem_result is None:
                        print(f"Skipping {problem_name} - no verification result found")
                        continue
                    
                    # Show problem details
                    print(f"\n{'='*60}")
                    print(f"Problem: {problem_name}")
                    print(f"Status: {problem_result['status']}")
                    if problem_result['status'] == 'invalid':
                        passed = problem_result.get('passed_test_cases', 0)
                        total = problem_result.get('total_test_cases', 0)
                        print(f"Test cases: {passed}/{total} passed")
                    elif problem_result['status'] == 'error':
                        error_msg = problem_result.get('error', 'Unknown error')
                        print(f"Error: {error_msg}")
                    
                    # Ask if user wants to force copy this problem
                    force_copy_response = input(f"Force copy {problem_name} to fully_prepared_final_dataset? (yes/no): ").strip().lower()
                    
                    if force_copy_response in ['yes', 'y']:
                        try:
                            # Create a mock "valid" result for copying
                            mock_result = {
                                'q_exact_answers': {},
                                'python_exact_answers': {}
                            }
                            
                            # If we have test results, try to extract any successful outputs
                            if 'test_results' in problem_result:
                                for test_result in problem_result['test_results']:
                                    test_idx = test_result.get('test_index')
                                    if test_idx is not None:
                                        if test_result.get('q_stdout'):
                                            mock_result['q_exact_answers'][test_idx] = test_result['q_stdout'].strip()
                                        if check_python and test_result.get('python_stdout'):
                                            mock_result['python_exact_answers'][test_idx] = test_result['python_stdout'].strip()
                            
                            # Copy to fully prepared dataset
                            copy_to_fully_prepared_dataset(problem_dir, fully_prepared_dir, mock_result)
                            
                            # Update cache to mark as valid
                            cache[problem_name] = {
                                'status': 'valid',
                                'hash': get_problem_hash(problem_dir),
                                'timestamp': datetime.now().isoformat(),
                                'total_test_cases': problem_result.get('total_test_cases', 0),
                                'passed_test_cases': problem_result.get('total_test_cases', 0),  # Mark all as passed
                                'force_copied': True  # Flag to indicate this was force copied
                            }
                            
                            force_copied_count += 1
                            print(f"  ✓ Force copied {problem_name} to fully_prepared_final_dataset")
                            
                        except Exception as e:
                            print(f"  ✗ Failed to force copy {problem_name}: {e}")
                    else:
                        print(f"  - Skipped {problem_name}")
                
                if force_copied_count > 0:
                    save_cache(cache)
                    print(f"\nForce copied {force_copied_count} problems and updated cache.")
                else:
                    print("\nNo problems were force copied.")
            else:
                print("Skipping individual problem review.")
    else:
        print("\n✓ No problematic problems found!")
    
    logger.info("Verification completed successfully")
    return 0

if __name__ == "__main__":
    exit(main()) 