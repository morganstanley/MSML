#!/usr/bin/env python3
"""
Script to validate the filtered LeetCode dataset by running test cases.
Creates validated_dataset/ with only problems that pass all test cases.
"""

import os
import json
import argparse
import re
import shutil
from pathlib import Path
from openai import OpenAI
import time
from tqdm import tqdm
import sys
from io import StringIO
import traceback
from datetime import datetime
import threading
import signal

# Initialize OpenAI client
client = OpenAI()

def call_gpt(prompt, max_retries=3):
    """Call GPT-4o-mini with retry logic."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise
    return None

def validate_output_with_gpt(actual_output, expected_output):
    """Use GPT to validate if outputs are equivalent."""
    prompt = f"""Compare these two outputs and determine if they represent the same value:

Actual output: {actual_output}
Expected output: {expected_output}

These should be considered EQUIVALENT if they represent the same value, even with different formatting:

EXAMPLES OF EQUIVALENT OUTPUTS:
- "hello" and hello (same string, one with quotes, one without)
- [1, 2, 3] and [1,2,3] (same list, different spacing)
- True and true (same boolean)
- 5 and 5.0 (same number)
- None and null (same null value)
- ['a', 'b'] and ["a", "b"] (same list, different quote styles)

Focus on whether the VALUES are the same, not the exact formatting.

Answer with only "YES" or "NO".
"""
    
    result = call_gpt(prompt)
    return result and result.strip().upper() == "YES"

class TimeoutError(Exception):
    """Custom timeout exception."""
    pass

def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutError("Execution timed out")

def execute_test_case(solution_code, test_case_code, timeout=4):
    """Execute a test case and return the output with timeout."""
    try:
        # Combine solution and test case
        full_code = solution_code + "\n\n" + test_case_code
        
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        # Set up timeout signal (Unix/Linux only)
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        
        try:
            # Create execution environment with necessary imports
            exec_globals = {
                '__builtins__': __builtins__,
                'List': list,  # Add List from typing as an alias to list
                'Optional': type(None),  # Basic Optional support
            }
            
            # Import typing modules explicitly
            try:
                from typing import List, Optional, Dict, Any
                exec_globals.update({
                    'List': List,
                    'Optional': Optional, 
                    'Dict': Dict,
                    'Any': Any
                })
            except ImportError:
                pass  # Use the basic fallbacks above
            
            # Execute the code with proper globals
            exec(full_code, exec_globals)
            
            # Get the output
            output = captured_output.getvalue().strip()
            
            # Cancel the alarm
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            
            # Restore stdout
            sys.stdout = old_stdout
            
            return output, None
            
        except TimeoutError:
            # Cancel the alarm and restore handler
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            sys.stdout = old_stdout
            return None, f"Execution timed out after {timeout} seconds"
            
    except Exception as e:
        # Make sure to clean up in case of any other error
        try:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        except:
            pass
        sys.stdout = old_stdout
        return None, str(e)

def check_already_processed(problem_name, validated_dir):
    """Check if a problem has already been validated."""
    return (validated_dir / problem_name).exists()

def copy_problem_files(source_dir, dest_dir):
    """Copy all files from source to destination."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    for file_path in source_dir.iterdir():
        if file_path.is_file():
            shutil.copy2(file_path, dest_dir / file_path.name)

def validate_single_problem(problem_dir, validated_dir, stats, log_data):
    """Validate a single problem by running all test cases."""
    problem_name = problem_dir.name
    print(f"\n=== Validating: {problem_name} ===")
    
    # Initialize problem log entry
    problem_log = {
        'problem_name': problem_name,
        'timestamp': datetime.now().isoformat(),
        'status': 'unknown',
        'test_cases': [],
        'error': None,
        'skipped': False,
        'passed': False
    }
    
    # Check if already processed
    if check_already_processed(problem_name, validated_dir):
        print(f"  ✓ Already validated, skipping")
        problem_log['status'] = 'skipped'
        problem_log['skipped'] = True
        log_data['problems'].append(problem_log)
        stats['skipped'] += 1
        return True
    
    # Load solution code
    sol_path = problem_dir / "python_sol.py"
    if not sol_path.exists():
        print(f"  ✗ python_sol.py not found")
        problem_log['status'] = 'failed'
        problem_log['error'] = 'python_sol.py not found'
        log_data['problems'].append(problem_log)
        stats['failed'] += 1
        return False
    
    with open(sol_path, 'r', encoding='utf-8') as f:
        solution_code = f.read()
    
    print(f"  → Loaded solution code ({len(solution_code)} chars)")
    problem_log['solution_code_length'] = len(solution_code)
    
    # Get all test case files
    test_case_files = sorted(problem_dir.glob("python_test_case_*.py"))
    expected_files = sorted(problem_dir.glob("python_test_case_*_correct_ans.txt"))
    
    if len(test_case_files) != len(expected_files):
        print(f"  ✗ Mismatch in test files: {len(test_case_files)} test cases, {len(expected_files)} expected answers")
        problem_log['status'] = 'failed'
        problem_log['error'] = f'Mismatch in test files: {len(test_case_files)} test cases, {len(expected_files)} expected answers'
        log_data['problems'].append(problem_log)
        stats['failed'] += 1
        return False
    
    print(f"  → Found {len(test_case_files)} test cases")
    problem_log['num_test_cases'] = len(test_case_files)
    
    # Run each test case
    all_passed = True
    for i, (test_file, expected_file) in enumerate(zip(test_case_files, expected_files), 1):
        print(f"    Test case {i}:")
        
        # Initialize test case log entry
        test_log = {
            'test_case_number': i,
            'test_file': test_file.name,
            'expected_file': expected_file.name,
            'passed': False,
            'execution_error': None,
            'actual_output': None,
            'expected_output': None,
            'exact_match': False,
            'timed_out': False,
            'gpt_validation': {
                'used': False,
                'result': None,
                'error': None
            }
        }
        
        # Load test case and expected output
        with open(test_file, 'r', encoding='utf-8') as f:
            test_code = f.read()
        
        with open(expected_file, 'r', encoding='utf-8') as f:
            expected_output = f.read().strip()
        
        test_log['expected_output'] = expected_output
        
        # Execute test case
        actual_output, error = execute_test_case(solution_code, test_code)
        
        if error:
            print(f"      ✗ Execution error: {error}")
            test_log['execution_error'] = error
            # Check if it's a timeout error
            if "timed out" in error.lower():
                test_log['timed_out'] = True
                print(f"      ⏱️ Test case timed out after 4 seconds")
            problem_log['test_cases'].append(test_log)
            all_passed = False
            break
        
        if actual_output is None:
            print(f"      ✗ No output generated")
            test_log['execution_error'] = 'No output generated'
            problem_log['test_cases'].append(test_log)
            all_passed = False
            break
        
        test_log['actual_output'] = actual_output
        
        print(f"      → Actual: {actual_output}")
        print(f"      → Expected: {expected_output}")
        
        # Check if outputs are equivalent using GPT
        if actual_output.strip() == expected_output.strip():
            print(f"      ✓ Exact match")
            test_log['exact_match'] = True
            test_log['passed'] = True
        else:
            print(f"      → Checking with GPT...")
            test_log['gpt_validation']['used'] = True
            try:
                is_equivalent = validate_output_with_gpt(actual_output, expected_output)
                test_log['gpt_validation']['result'] = is_equivalent
                if is_equivalent:
                    print(f"      ✓ GPT confirmed equivalent")
                    test_log['passed'] = True
                else:
                    print(f"      ✗ GPT says not equivalent")
                    test_log['passed'] = False
                    problem_log['test_cases'].append(test_log)
                    all_passed = False
                    break
            except Exception as e:
                print(f"      ✗ GPT validation failed: {e}")
                test_log['gpt_validation']['error'] = str(e)
                test_log['passed'] = False
                problem_log['test_cases'].append(test_log)
                all_passed = False
                break
        
        problem_log['test_cases'].append(test_log)
    
    if all_passed:
        print(f"  ✓ All test cases passed! Moving to validated_dataset")
        dest_dir = validated_dir / problem_name
        copy_problem_files(problem_dir, dest_dir)
        problem_log['status'] = 'passed'
        problem_log['passed'] = True
        stats['passed'] += 1
        log_data['problems'].append(problem_log)
        return True
    else:
        print(f"  ✗ Some test cases failed")
        problem_log['status'] = 'failed'
        problem_log['passed'] = False
        stats['failed'] += 1
        log_data['problems'].append(problem_log)
        return False

def main():
    parser = argparse.ArgumentParser(description='Validate filtered LeetCode dataset')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Maximum number of files to process (default: all)')
    args = parser.parse_args()
    
    # Set up directories
    filtered_dir = Path("filtered_dataset")
    validated_dir = Path("validated_dataset")
    validated_dir.mkdir(exist_ok=True)
    
    if not filtered_dir.exists():
        print("ERROR: filtered_dataset directory not found!")
        return
    
    # Get all problem directories
    problem_dirs = [d for d in filtered_dir.iterdir() if d.is_dir()]
    
    if args.max_files:
        problem_dirs = problem_dirs[:args.max_files]
    
    print(f"Found {len(problem_dirs)} problems to validate")
    
    # Statistics
    stats = {
        'passed': 0,
        'failed': 0,
        'skipped': 0,
        'total': len(problem_dirs)
    }
    
    # Initialize log data structure
    log_data = {
        'timestamp': datetime.now().isoformat(),
        'total_problems': len(problem_dirs),
        'max_files_limit': args.max_files,
        'problems': []
    }
    
    # Process each problem
    with tqdm(total=len(problem_dirs), desc="Validating problems") as pbar:
        for problem_dir in problem_dirs:
            try:
                validate_single_problem(problem_dir, validated_dir, stats, log_data)
                
                # Rate limiting
                if (stats['passed'] + stats['failed']) % 5 == 0:
                    time.sleep(1)
                
            except Exception as e:
                print(f"\nERROR processing {problem_dir.name}: {e}")
                traceback.print_exc()
                
                # Log the error
                error_log = {
                    'problem_name': problem_dir.name,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'error',
                    'error': str(e),
                    'traceback': traceback.format_exc(),
                    'test_cases': [],
                    'skipped': False,
                    'passed': False
                }
                log_data['problems'].append(error_log)
                stats['failed'] += 1
            
            pbar.set_postfix({
                'passed': stats['passed'],
                'failed': stats['failed'],
                'skipped': stats['skipped']
            })
            pbar.update(1)
    
    # Save log data to JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"validation_log_{timestamp}.json"
    
    # Add final statistics to log data
    log_data['final_stats'] = {
        'total_problems': stats['total'],
        'passed': stats['passed'],
        'failed': stats['failed'],
        'skipped': stats['skipped'],
        'pass_rate': stats['passed']/stats['total']*100 if stats['total'] > 0 else 0
    }
    
    # Save log file
    with open(log_filename, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    # Final statistics
    print(f"\n{'='*50}")
    print(f"VALIDATION COMPLETE")
    print(f"{'='*50}")
    print(f"Total problems: {stats['total']}")
    print(f"Passed: {stats['passed']} ({stats['passed']/stats['total']*100:.1f}%)")
    print(f"Failed: {stats['failed']} ({stats['failed']/stats['total']*100:.1f}%)")
    print(f"Skipped: {stats['skipped']} ({stats['skipped']/stats['total']*100:.1f}%)")
    print(f"Results saved to: {validated_dir.absolute()}")
    print(f"Detailed log saved to: {log_filename}")

if __name__ == "__main__":
    main() 