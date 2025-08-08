#!/usr/bin/env python3
"""
Phase 2: Evaluate Completions

Evaluates all generated completions against test cases and creates final results.
Processes completions with parallel execution and resume capability.

Usage:
    python evaluate_completions.py --output_dir results
    python evaluate_completions.py --output_dir results --max_workers 20
"""

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
from collections import defaultdict

from evaluation_utils import (
    load_test_problems, run_q_code, run_python_code, 
    get_file_extension, get_problem_id, logger
)


def is_infrastructure_failure(error_msg: str) -> bool:
    """Check if an error indicates infrastructure failure that should cause job exit."""
    infrastructure_keywords = [
        "INFRASTRUCTURE_FAILURE",
        "License daemon overloaded",
        "couldn't connect to license daemon", 
        "Connection refused",
        "Service unavailable",
        "Internal server error",
        "Rate limit exceeded",
        "Server overloaded"
    ]
    return any(keyword.lower() in error_msg.lower() for keyword in infrastructure_keywords)


def calculate_pass_at_k(n_correct: int, n_total: int, k: int) -> float:
    """
    Calculate pass@k metric using the combinatorial formula.
    
    pass@k = 1 - C(n_total - n_correct, k) / C(n_total, k)
    """
    if n_correct == 0:
        return 0.0
    if k > n_total:
        return 1.0
    if n_total - n_correct < k:
        # Not enough incorrect samples to fill k slots
        return 1.0
    
    # Calculate using log to avoid overflow
    def log_comb(n, k):
        if k > n or k < 0:
            return float('-inf')
        return sum(math.log(n - i) - math.log(i + 1) for i in range(k))
    
    log_prob_fail = log_comb(n_total - n_correct, k) - log_comb(n_total, k)
    return 1.0 - math.exp(log_prob_fail)


def evaluate_single_completion(args) -> Dict[str, Any]:
    """Worker function to evaluate a single completion against all test cases."""
    
    completion_file, problem, problem_id, completion_num = args
    
    try:
        # Load completion code
        with open(completion_file, 'r') as f:
            generated_code = f.read().strip()
        
        if not generated_code:
            return {
                'problem_id': problem_id,
                'completion_num': completion_num,
                'success': False,
                'error': 'Empty completion file',
                'passed_count': 0,
                'total_count': len(problem['test_cases']),
                'test_results': []
            }
        
        # Determine language and expected output key
        task_type = problem['task_type']
        if task_type in ['description_to_q', 'python_to_q']:
            language = "q"
            expected_key = 'q_expected_output'
        else:  # q_to_python
            language = "python"
            expected_key = 'python_expected_output'
        
        # Test against all test cases
        test_results = []
        passed_count = 0
        
        for test_case in problem['test_cases']:
            test_idx = test_case['index']
            expected_output = test_case[expected_key]
            
            if language == "q":
                test_harness = test_case['q_test_content']
            else:
                test_harness = test_case['py_test_content']
            
            # Combine generated code with test harness
            full_code = f"{generated_code}\n\n{test_harness}"
            
            # Execute code with retry on license daemon issues
            task_id = f"{problem_id}_completion{completion_num}_test{test_idx}"
            max_retries = 3
            retry_delay = 0.5  # Start with 0.5 second delay
            
            for retry in range(max_retries + 1):
                if language == "q":
                    success, stdout, stderr = run_q_code(full_code, task_id=task_id)
                else:
                    success, stdout, stderr = run_python_code(full_code, task_id=task_id)
                
                # If it's a license daemon issue, retry with exponential backoff
                if stderr.startswith("LICENSE_DAEMON_OVERLOAD:"):
                    if retry < max_retries:
                        import time
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    else:
                        # After max retries, mark as infrastructure failure
                        # stderr = "INFRASTRUCTURE_FAILURE: License daemon overloaded after retries"
                        stderr = stderr
                        # print(stderr)
                        success = False
                        stdout = ""
                
                # If not a license issue, break out of retry loop
                break
            
            # Check if output matches exactly (after stripping whitespace)
            test_passed = False
            if success and stdout is not None:
                actual_stripped = stdout.strip()
                expected_stripped = expected_output.strip()
                
                # Special case: handle empty string vs empty output
                if (actual_stripped == "" and expected_stripped == '""') or \
                   (actual_stripped == '""' and expected_stripped == ""):
                    test_passed = True
                else:
                    test_passed = actual_stripped == expected_stripped
            
            if test_passed:
                passed_count += 1
            
            test_results.append({
                'test_index': test_idx,
                'passed': test_passed,
                'execution_success': success,
                'actual_output': stdout,
                'expected_output': expected_output,
                'stderr': stderr
            })
        
        all_passed = passed_count == len(problem['test_cases'])
        
        return {
            'problem_id': problem_id,
            'completion_num': completion_num,
            'success': all_passed,
            'passed_count': passed_count,
            'total_count': len(problem['test_cases']),
            'test_results': test_results,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'problem_id': problem_id,
            'completion_num': completion_num,
            'success': False,
            'error': str(e),
            'passed_count': 0,
            'total_count': len(problem['test_cases']),
            'test_results': [],
            'timestamp': datetime.now().isoformat()
        }


class CompletionEvaluator:
    """Evaluates completions against test cases."""
    
    def __init__(self, output_dir: Path, max_workers: int = 10):
        self.output_dir = output_dir
        self.max_workers = max_workers
        self.completions_dir = output_dir / "completions"
        self.results_dir = output_dir / "results"
        
    def verify_completions_exist(self, problems: List[Dict[str, Any]]) -> bool:
        """Verify that all problems have at least one completion."""
        if not self.completions_dir.exists():
            logger.error(f"Completions directory not found: {self.completions_dir}")
            return False
        
        missing_problems = []
        for problem in problems:
            problem_id = get_problem_id(problem)
            problem_dir = self.completions_dir / problem_id
            
            if not problem_dir.exists():
                missing_problems.append(problem_id)
                continue
            
            # Check if there's at least one completion file
            ext = get_file_extension(problem['task_type'])
            completions = list(problem_dir.glob(f"completion_*{ext}"))
            if not completions:
                missing_problems.append(problem_id)
        
        if missing_problems:
            logger.error(f"Missing completions for {len(missing_problems)} problems:")
            for problem_id in missing_problems[:5]:  # Show first 5
                logger.error(f"  - {problem_id}")
            if len(missing_problems) > 5:
                logger.error(f"  ... and {len(missing_problems) - 5} more")
            return False
        
        return True
    
    def collect_evaluation_tasks(self, problems: List[Dict[str, Any]]) -> List[Tuple]:
        """Collect all evaluation tasks, skipping those already completed."""
        tasks = []
        self.results_dir.mkdir(exist_ok=True)
        
        for problem in problems:
            problem_id = get_problem_id(problem)
            problem_dir = self.completions_dir / problem_id
            results_problem_dir = self.results_dir / problem_id
            results_problem_dir.mkdir(exist_ok=True)
            
            if not problem_dir.exists():
                continue
            
            # Find all completion files
            ext = get_file_extension(problem['task_type'])
            completion_files = sorted(problem_dir.glob(f"completion_*{ext}"))
            
            for completion_file in completion_files:
                try:
                    completion_num = int(completion_file.stem.replace('completion_', ''))
                except ValueError:
                    continue
                
                # Check if results already exist
                result_file = results_problem_dir / f"completion_{completion_num}_results.json"
                if result_file.exists():
                    continue  # Skip already evaluated completions
                
                tasks.append((completion_file, problem, problem_id, completion_num))
        
        return tasks
    
    def evaluate_all_completions(self, problems: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate all completions using parallel processing."""
        
        # Collect evaluation tasks
        tasks = self.collect_evaluation_tasks(problems)
        
        if not tasks:
            logger.info("✓ All completions already evaluated")
        else:
            logger.info(f"Evaluating {len(tasks)} completions using {self.max_workers} workers")
            
            # Process tasks in parallel
            completed = 0
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_task = {executor.submit(evaluate_single_completion, task): task for task in tasks}
                
                # Process completed tasks
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    result = future.result()
                    
                    # Check for infrastructure failures in test results
                    infrastructure_failure = False
                    for test_result in result.get('test_results', []):
                        stderr = test_result.get('stderr', '')
                        # print(stderr)
                        if stderr and is_infrastructure_failure(stderr):
                            logger.error(f"🚨 INFRASTRUCTURE FAILURE DETECTED: {stderr}")
                            logger.error("🔄 Skipping this evaluation and continuing...")
                            infrastructure_failure = True
                            break
                    
                    # Skip saving this result if infrastructure failure detected
                    if infrastructure_failure:
                        continue
                    
                    completed += 1
                    
                    # Save individual result
                    _, _, problem_id, completion_num = task
                    results_problem_dir = self.results_dir / problem_id
                    result_file = results_problem_dir / f"completion_{completion_num}_results.json"
                    
                    with open(result_file, 'w') as f:
                        json.dump(result, f, indent=2)
                    
                    if result.get('success'):
                        logger.debug(f"✓ Evaluated {problem_id} completion {completion_num}")
                    else:
                        logger.debug(f"✗ Failed {problem_id} completion {completion_num}")
                    
                    # Progress update
                    if completed % 20 == 0 or completed == len(tasks):
                        logger.info(f"Progress: {completed}/{len(tasks)} evaluations completed")
        
        # Collect all results and aggregate
        return self.aggregate_results(problems)
    
    def aggregate_results(self, problems: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate all individual results into final statistics."""
        
        all_problem_results = []
        
        for problem in problems:
            problem_id = get_problem_id(problem)
            results_problem_dir = self.results_dir / problem_id
            
            if not results_problem_dir.exists():
                logger.warning(f"No results found for {problem_id}")
                continue
            
            # Load all completion results for this problem
            completion_results = []
            result_files = sorted(results_problem_dir.glob("completion_*_results.json"))
            
            for result_file in result_files:
                try:
                    with open(result_file, 'r') as f:
                        result = json.load(f)
                    completion_results.append(result)
                except Exception as e:
                    logger.warning(f"Could not load {result_file}: {e}")
            
            if not completion_results:
                logger.warning(f"No completion results found for {problem_id}")
                continue
            
            # Calculate statistics for this problem
            total_completions = len(completion_results)
            successful_completions = sum(1 for r in completion_results if r.get('success', False))
            
            # Find best result
            best_result = max(completion_results, key=lambda x: x.get('passed_count', 0))
            
            problem_result = {
                'problem_name': problem['problem_name'],
                'task_type': problem['task_type'],
                'problem_id': problem_id,
                'total_completions': total_completions,
                'successful_completions': successful_completions,
                'success_rate': successful_completions / total_completions if total_completions > 0 else 0,
                'best_passed_count': best_result.get('passed_count', 0),
                'total_test_cases': problem['total_test_cases'],
                'best_pass_rate': best_result.get('passed_count', 0) / problem['total_test_cases'],
                'all_completion_results': completion_results
            }
            
            all_problem_results.append(problem_result)
        
        # Calculate overall statistics
        total_problems = len(all_problem_results)
        if total_problems == 0:
            logger.error("No problem results found!")
            return {}
        
        # Overall success rates
        problems_with_success = sum(1 for r in all_problem_results if r['successful_completions'] > 0)
        overall_problem_success_rate = problems_with_success / total_problems
        
        # Test case statistics
        total_test_cases = sum(r['total_test_cases'] for r in all_problem_results)
        total_best_passed = sum(r['best_passed_count'] for r in all_problem_results)
        overall_test_case_pass_rate = total_best_passed / total_test_cases if total_test_cases > 0 else 0
        
        # Calculate pass@k metrics
        pass_at_k_metrics = {}
        for k in [1, 2, 4, 6, 8, 16]:
            empirical_passed = 0
            probabilistic_sum = 0
            valid_problems = 0
            
            for problem_result in all_problem_results:
                completions = problem_result['all_completion_results']
                n_total = len(completions)
                
                if k <= n_total:
                    valid_problems += 1
                    
                    # Empirical: Did any of the first k attempts pass?
                    if any(completions[i].get('success', False) for i in range(k)):
                        empirical_passed += 1
                    
                    # Probabilistic: What's the probability that k random samples contain a correct one?
                    n_correct = sum(1 for comp in completions if comp.get('success', False))
                    prob = calculate_pass_at_k(n_correct, n_total, k)
                    probabilistic_sum += prob
            
            if valid_problems > 0:
                pass_at_k_metrics[f'pass@{k}'] = {
                    'empirical': empirical_passed / valid_problems,
                    'probabilistic': probabilistic_sum / valid_problems,
                    'valid_problems': valid_problems
                }
        
        # Task-specific statistics
        task_stats = defaultdict(lambda: {
            'problems': 0, 'successful_problems': 0, 'total_test_cases': 0, 'passed_test_cases': 0
        })
        
        for result in all_problem_results:
            task_type = result['task_type']
            task_stats[task_type]['problems'] += 1
            task_stats[task_type]['total_test_cases'] += result['total_test_cases']
            task_stats[task_type]['passed_test_cases'] += result['best_passed_count']
            
            if result['successful_completions'] > 0:
                task_stats[task_type]['successful_problems'] += 1
        
        # Convert to regular dict and add rates
        task_stats = dict(task_stats)
        for task_type in task_stats:
            stats = task_stats[task_type]
            stats['problem_success_rate'] = stats['successful_problems'] / stats['problems'] if stats['problems'] > 0 else 0
            stats['test_case_pass_rate'] = stats['passed_test_cases'] / stats['total_test_cases'] if stats['total_test_cases'] > 0 else 0
        
        # Final results
        final_results = {
            'timestamp': datetime.now().isoformat(),
            'total_problems': total_problems,
            'problems_with_success': problems_with_success,
            'overall_problem_success_rate': overall_problem_success_rate,
            'total_test_cases': total_test_cases,
            'total_best_passed_test_cases': total_best_passed,
            'overall_test_case_pass_rate': overall_test_case_pass_rate,
            'pass_at_k_metrics': pass_at_k_metrics,
            'task_stats': task_stats,
            'problem_results': all_problem_results
        }
        
        return final_results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated completions against test cases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic evaluation
    python evaluate_completions.py --output_dir results

    # With custom test directory and workers
    python evaluate_completions.py --output_dir results --test_dir ../SFT_Data/test --max_workers 20

    # Evaluate specific tasks only
    python evaluate_completions.py --output_dir results --tasks description_to_q python_to_q
        """
    )
    
    # Required arguments
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory containing completions")
    
    # Optional arguments
    parser.add_argument("--test_dir", type=str, default="../SFT_Data/test",
                       help="Test data directory (to reload problem metadata)")
    parser.add_argument("--max_problems", type=int,
                       help="Maximum number of problems to evaluate")
    parser.add_argument("--tasks", nargs='+', 
                       choices=['description_to_q', 'python_to_q', 'q_to_python'],
                       help="Tasks to evaluate")
    parser.add_argument("--max_workers", type=int, default=6,
                       help="Maximum number of worker processes (default: 6 - reduced for Q license stability)")
    parser.add_argument("--max_q_workers", type=int, default=4,
                       help="Maximum concurrent Q executions (default: 4 - prevents license daemon overload)")
    
    parser.add_argument("--force", action="store_true",
                       help="Force evaluation even if some completions are missing.")

    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        logger.error(f"Output directory does not exist: {output_dir}")
        return 1
    
    # Load test problems to get metadata
    logger.info(f"Loading test problems from {args.test_dir}")
    problems = load_test_problems(args.test_dir, args.max_problems, args.tasks)
    if not problems:
        logger.error("No test problems found!")
        return 1
    
    # Create evaluator
    evaluator = CompletionEvaluator(output_dir, args.max_workers)
    
    # Verify completions exist
    logger.info("Verifying completions exist...")
    if not evaluator.verify_completions_exist(problems):
        if args.force:
            logger.warning("⚠️ Missing completions detected, but --force is enabled. Evaluating available completions.")
        else:
            logger.error("✗ Missing completions detected. Please run generation phase first or use --force.")
            return 1
    
    logger.info(f"🚀 Starting completion evaluation")
    logger.info(f"📁 Output directory: {output_dir}")
    logger.info(f"🔢 Problems: {len(problems)}")
    logger.info(f"👥 Max workers: {args.max_workers}")
    print()
    
    start_time = time.time()
    
    # Evaluate completions
    results = evaluator.evaluate_all_completions(problems)
    
    if not results:
        logger.error("✗ No results generated")
        return 1
    
    total_time = time.time() - start_time
    
    # Save final results
    results_file = output_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    logger.info(f"\n🎉 Evaluation completed!")
    logger.info(f"⏱️ Total time: {total_time/60:.1f}m")
    logger.info(f"📊 Problems evaluated: {results['total_problems']}")
    logger.info(f"✅ Problems with success: {results['problems_with_success']}/{results['total_problems']} ({results['overall_problem_success_rate']:.1%})")
    logger.info(f"🎯 Test case pass rate: {results['total_best_passed_test_cases']}/{results['total_test_cases']} ({results['overall_test_case_pass_rate']:.1%})")
    
    # Print pass@k metrics
    logger.info("\nPass@k metrics (empirical):")
    for k in [1, 2, 4, 6, 8, 16]:
        key = f'pass@{k}'
        if key in results['pass_at_k_metrics']:
            metric = results['pass_at_k_metrics'][key]
            logger.info(f"  {key}: {metric['empirical']:.1%} ({metric['valid_problems']} problems)")
    
    # Print task breakdown
    logger.info("\nBy task type:")
    for task_type, stats in results['task_stats'].items():
        logger.info(f"  {task_type}:")
        logger.info(f"    Problems: {stats['successful_problems']}/{stats['problems']} ({stats['problem_success_rate']:.1%})")
        logger.info(f"    Test cases: {stats['passed_test_cases']}/{stats['total_test_cases']} ({stats['test_case_pass_rate']:.1%})")
    
    logger.info(f"\n📄 Final results saved to: {results_file}")
    logger.info("✅ Evaluation completed successfully!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 