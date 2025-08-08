#!/usr/bin/env python3
"""
Script to convert validated Python dataset to Q programming language.
Creates final_dataset/ with both Python and Q versions of problems.
"""

import os
import json
import argparse
import re
import shutil
import subprocess
import time
import random
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
import traceback
from datetime import datetime
from typing import Optional, Tuple

# Optional imports for local model
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
TRANSFORMERS_AVAILABLE = True

# With configuration-based path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_config

config = get_config()
Q_PATH = config.dataset.q_interpreter_path

# Initialize OpenAI client
client = OpenAI()

# Initialize Anthropic client
try:
    import anthropic
    anthropic_client = anthropic.Anthropic()
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic_client = None

# Global variable to hold the LLM instance
llm_instance = None
llm_type = "gpt"  # Track which LLM type is being used

# Constants for Q execution
DEFAULT_TIMEOUT = 5
Q_CMD = "taskset -c 0-23"
Q_EXECUTABLE = f"{Q_CMD} {Q_PATH}"

class LocalLLM:
    """Local Hugging Face model for Q code generation."""
    
    def __init__(self, model_path: str = None):
        """Initialize the local model."""
        
        if model_path is None:
            model_path = config.model.base_model

        print(f"Loading local model from: {model_path}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Add padding token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Local model loaded on: {self.device}")
    
    def _format_prompt(self, prompt: str) -> str:
        """Format the prompt for the local model."""
        return prompt
    
    def generate(self, prompt: str) -> str:
        """Generate response using the local model."""
        formatted_prompt = self._format_prompt(prompt)

        messages = [
            {"role": "system", "content": "You are an expert Q programming language coder, specializing in financial analysis."},
            {"role": "user", "content": formatted_prompt}
        ]

        chat = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        generation_params = {
            "max_new_tokens": 500,
            "do_sample": True,
        }
        generation_params.update({
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
        })

        with torch.no_grad():
            inputs = self.tokenizer(chat, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **generation_params
            )
        
        # Trim outputs to be after inputs
        outputs = outputs[:,inputs["input_ids"].shape[1]:][0]
        response = self.tokenizer.decode(outputs, skip_special_tokens=True)
        
        return response.strip()

def call_llm(prompt, max_retries=3, model="gpt-4.1", log_entry=None):
    """Call LLM (GPT, Claude, or local) with retry logic and optional logging."""
    global llm_instance
    
    for attempt in range(max_retries):
        try:
            if llm_instance is None:
                if model.startswith("claude"):
                    # Use Claude via Anthropic API
                    if not ANTHROPIC_AVAILABLE:
                        raise Exception("Anthropic library not available. Install with: pip install anthropic")
                    
                    response = anthropic_client.messages.create(
                        model=model,
                        max_tokens=3000,
                        temperature=0.1,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    result = response.content[0].text.strip()
                    model_used = model
                else:
                    # Use GPT
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1,
                        max_tokens=3000
                    )
                    result = response.choices[0].message.content.strip()
                    model_used = model
            else:
                # Use local model
                result = llm_instance.generate(prompt)
                model_used = "local_model"
            
            # Log the interaction if log_entry is provided
            if log_entry is not None:
                interaction = {
                    'timestamp': datetime.now().isoformat(),
                    'model': model_used,
                    'prompt': prompt,
                    'response': result,
                    'attempt': attempt + 1
                }
                log_entry.append(interaction)
            
            return result
            
        except Exception as e:
            print(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise
    return None

# Keep backward compatibility
def call_gpt(prompt, max_retries=3, model="gpt-4.1", log_entry=None):
    """Backward compatibility wrapper for call_llm."""
    return call_llm(prompt, max_retries, model, log_entry)

def extract_code_from_response(response: str, language: str = None) -> str:
    """Extract code from an LLM response, removing markdown code blocks if present."""
    if language:
        pattern = f"```{language}(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return matches[0].strip()
    
    matches = re.findall(r"```(.*?)```", response, re.DOTALL)
    if matches:
        return matches[0].strip()
    
    return response.strip()

def run_q_code(code: str, timeout: int = DEFAULT_TIMEOUT, test_case_content: Optional[str] = None, task_id_for_temp: str = "eval_q") -> Tuple[bool, Optional[str], Optional[str]]:
    """Run Q code in a temporary directory and return success status and output."""
    temp_dir = Path(f"./temp_q_eval_{task_id_for_temp}_{time.time_ns()}")
    temp_dir.mkdir(exist_ok=True)
    temp_script_path = temp_dir / "temp_script.q"
    test_case_file_path = temp_dir / "test_case.txt"

    try:
        # Ensure Q code ends with exit 0; to properly terminate
        if not code.strip().endswith("exit 0;"):
            code = code.strip() + "\nexit 0;"
        
        with open(temp_script_path, 'w') as f:
            f.write(code)
        
        if test_case_content:
            with open(test_case_file_path, 'w') as f:
                f.write(test_case_content)

        process = subprocess.run(
            Q_EXECUTABLE.split() + [temp_script_path.name],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            cwd=str(temp_dir)
        )
        
        success = process.returncode == 0 and "error" not in process.stderr.lower()
        return success, process.stdout.strip(), process.stderr.strip()
    
    except subprocess.TimeoutExpired:
        return False, None, f"Execution timed out after {timeout} seconds"
    except Exception as e:
        return False, None, str(e)
    finally:
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Failed to remove temporary directory {temp_dir}: {e}")

def count_solve_calls(code: str) -> int:
    """Count the number of times 'solve' is called in the code."""
    # Look for solve function calls (not definitions)
    solve_calls = re.findall(r'\bsolve\s*[\[\(]', code)
    return len(solve_calls)

def enhance_problem_description(problem_desc):
    """Add Q-specific instructions to the problem description."""
    enhancement = """

NOTE: This problem is described for Python, but your task is to implement it in Q programming language. Q is a vector programming language developed by Kx Systems, designed for high-performance data analysis and time-series processing with a concise, functional syntax.
- Use the same solve function approach and equivalent logic
- Follow Q syntax and conventions
- Your Q implementation should have the same function signature and return the same results
- Use Q data structures and operations that are equivalent to the Python version
"""
    return problem_desc + enhancement

def translate_python_to_q(python_code, problem_desc, log_entry=None):
    """Use GPT to translate Python code to Q."""
    q_examples = """
            Example Q code patterns:

            1. Simple function with max operation:
            solve: {max_area:0;
                    left:0;
                    right:(count x)-1;
                    while[left<right;
                        height:min x[left],x[right];
                        width:right - left;
                        area:height*width;
                        max_area:max max_area,area;
                        if[x[left]<x[right];
                            left:left+1;
                        ];
                        if[not x[left]<x[right];
                            right:right-1;
                        ];
                    ];
                    max_area}

            2. Function with parameter and string operations:
            solve:{[num]
            roman:();
            val:(1000;900;500;400;100;90;50;40;10;9;5;4;1);
            symb:("M";"CM";"D";"CD";"C";"XC";"L";"XL";"X";"IX";"V";"IV";"I");
            i:0;
            while[num>0;
            if[num>=val[i];
            roman,:symb[i];
            num:num-val[i]];
            if[num<val[i];
            i+:1]];
            :roman}

            3. Function with list operations:
            solve:{[words]
                if[0=count words; :""];
                min_length:min count each words;
                prefix:"";
                i:0;
                while[i<min_length;
                    current_char:words[0][i];
                    if[all words[i]=current_char;
                        prefix,:current_char;
                        i+:1];
                    if[not all words[i]=current_char;
                        :prefix]];
                :prefix};
            """

    # prompt = f"""You are an expert Q programmer. Translate the following Python code to Q programming language.

    #     PROBLEM CONTEXT:
    #     {problem_desc}

    #     PYTHON CODE TO TRANSLATE:
    #     {python_code}

    #     {q_examples}

    #     Q SYNTAX NOTES:
    #     - Functions defined as: solve:{{[params] ...code...}}
    #     - Variables assigned with colon: var:value;
    #     - Lists use parentheses: (1;2;3)
    #     - Strings use quotes: "hello"
    #     - Indexing with brackets: list[index]
    #     - Conditionals: if[condition; action];
    #     - Loops: while[condition; action];
    #     - Return with colon: :result

    #     Please provide only the Q code (no markdown, no explanations):
    #     """
     
    prompt = f"""You are an expert Q programmer. Translate the following Python code to Q programming language.

        PROBLEM CONTEXT:
        {problem_desc}

        PYTHON CODE TO TRANSLATE:
        {python_code}


        Please provide only the Q code (no markdown, no explanations):
        """

    model = "claude-sonnet-4-20250514" if llm_type == "claude" else "gpt-4.1"
    response = call_llm(prompt, model=model, log_entry=log_entry)
    q_code = extract_code_from_response(response, "q")
    
    # Remove 'exit 0;' from generated Q code if present - we don't want it in the solution
    if q_code and q_code.strip().endswith("exit 0;"):
        q_code = q_code.strip()[:-7].strip()  # Remove 'exit 0;' and trailing whitespace
    
    return q_code

def translate_test_case_to_q(python_test_case, expected_output, log_entry=None, max_retries=3):
    """Translate a Python test case to Q format."""
    
    # Extract the function call from python test case
    # Expected format: print(solve(args...))
    match = re.search(r'solve\((.*?)\)', python_test_case)
    if not match:
        return None
    
    args = match.group(1)
    
    q_test_examples = """
Example Q test format:
result:solve[58];
show result;

result:solve enlist ("dog "; "racecar "; "car ");
show result;

result:solve[(0,1,1)];
show result;
"""

    prompt = f"""Convert this Python test case to Q format:

Python test: print(solve({args}))
Expected output: {expected_output}

{q_test_examples}

Q SYNTAX NOTES:
- Call function: result:solve[args];
- Display result: show result;
- Lists in Q: (item1;item2;item3)
- Strings: "text"
- For single-item lists use 'enlist': enlist "item"

CRITICAL REQUIREMENT: Your Q test case MUST include a call to the solve function. The first line must be something like "result:solve[...];" where you call the solve function with the appropriate arguments.

 Provide only the Q test code (2 lines: assignment and show):
 """
    
    # Retry logic for test case generation
    for attempt in range(max_retries):
        try:
            model = "claude-sonnet-4-20250514" if llm_type == "claude" else "gpt-4.1"
            response = call_llm(prompt, model=model, log_entry=log_entry)
            q_test = extract_code_from_response(response, "q")
            
            # Validate that the Q test case actually calls solve
            if q_test and "solve" in q_test:
                return q_test
            else:
                print(f"        ⚠️ Attempt {attempt + 1}: Generated Q test case doesn't call solve function")
                if attempt < max_retries - 1:
                    print(f"        → Retrying test case generation...")
                    continue
                else:
                    print(f"        ✗ Failed to generate valid test case after {max_retries} attempts")
                    return None
                    
        except Exception as e:
            print(f"        ✗ Error generating test case (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                return None
    
    return None

def validate_q_outputs(actual_output, expected_output, log_entry=None):
    """Check if Q output matches expected Python output."""
    prompt = f"""You are comparing outputs from Q implementations to determine if they are equivalent.

FORMATTING DIFFERENCES TO CONSIDER:

STRINGS:
- Q: "abc" or abc (without quotes)
- Expected: "abc" or abc

LISTS:
- Q: 1 2 3 or (1;2;3)
- Expected: [1, 2, 3] or (1;2;3)

BOOLEANS:
- Q: 1b 0b or 1 0
- Expected: True False or 1b 0b

NUMBERS:
- Q: 42 or 42.0
- Expected: 42 or 42.0

EXAMPLES OF EQUIVALENT OUTPUTS:
- Q: "abc" ≡ Expected: "abc"
- Q: 1 2 3 ≡ Expected: [1, 2, 3]
- Q: 1b ≡ Expected: True
- Q: 0b ≡ Expected: False
- Q: 42 ≡ Expected: 42
- Q: 125i ≡ Expected: 125
- Q: 125.0f ≡ Expected: 125.0

NOW COMPARE THESE SPECIFIC OUTPUTS:

Q output: {actual_output}
Expected output: {expected_output}

Are these outputs equivalent? Answer only YES or NO."""
     
    model = "claude-sonnet-4-20250514" if llm_type == "claude" else "gpt-4.1"
    result = call_llm(prompt, model=model, log_entry=log_entry)
    return result and result.strip().upper() == "YES"

def check_already_processed(problem_name, final_dir):
    """Check if a problem has already been converted."""
    problem_path = final_dir / problem_name
    if not problem_path.exists():
        return False
    
    # Check if Q files exist
    q_sol_exists = (problem_path / "q_sol.q").exists()
    q_test_exists = len(list(problem_path.glob("q_test_case_*.q"))) > 0
    
    return q_sol_exists and q_test_exists

def copy_problem_files(source_dir, dest_dir, exclude_test_files=False):
    """Copy files from source to destination, optionally excluding test files."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    for file_path in source_dir.iterdir():
        if file_path.is_file():
            # Skip python test files if we're excluding them (we'll create clean versions)
            if exclude_test_files and file_path.name.startswith("python_test_case_"):
                continue
            
            shutil.copy2(file_path, dest_dir / file_path.name)

def save_detailed_logs(problem_name, logging_dir, llm_interactions, q_code, q_test_cases, execution_results, attempt_number=1):
    """Save comprehensive logs for a problem conversion."""
    problem_log_dir = logging_dir / problem_name / f"attempt_{attempt_number}"
    problem_log_dir.mkdir(parents=True, exist_ok=True)
    
    # Save LLM interactions
    llm_log = {
        'problem_name': problem_name,
        'timestamp': datetime.now().isoformat(),
        'total_interactions': len(llm_interactions),
        'interactions': llm_interactions
    }
    
    with open(problem_log_dir / "llm_interactions.json", 'w', encoding='utf-8') as f:
        json.dump(llm_log, f, indent=2, ensure_ascii=False)
    
    # Save generated Q solution
    if q_code:
        with open(problem_log_dir / "generated_q_sol.q", 'w', encoding='utf-8') as f:
            f.write(q_code)
    
    # Save generated Q test cases
    for i, q_test in enumerate(q_test_cases, 1):
        with open(problem_log_dir / f"generated_q_test_case_{i}.q", 'w', encoding='utf-8') as f:
            f.write(q_test)
    
    # Save execution results
    execution_log = {
        'problem_name': problem_name,
        'timestamp': datetime.now().isoformat(),
        'execution_results': execution_results
    }
    
    with open(problem_log_dir / "execution_results.json", 'w', encoding='utf-8') as f:
        json.dump(execution_log, f, indent=2, ensure_ascii=False)
    
    # Save combined Q code that was executed for each test case
    for i, (q_test, result) in enumerate(zip(q_test_cases, execution_results), 1):
        combined_code = q_code + "\n\n" + q_test
        with open(problem_log_dir / f"combined_q_code_test_{i}.q", 'w', encoding='utf-8') as f:
            f.write(combined_code)

def attempt_single_conversion(problem_dir, problem_name, problem_desc, python_code, test_case_files, expected_files, 
                            attempt_number, q_code_from_previous=None):
    """Attempt a single conversion with optional reuse of previous Q code."""
    llm_interactions = []
    q_test_cases = []
    execution_results = []
    
    # Use existing Q code or generate new one
    if q_code_from_previous:
        q_code = q_code_from_previous
        print(f"  → Reusing Q solution from previous attempt")
    else:
        print(f"  → Generating new Q solution...")
        q_code = translate_python_to_q(python_code, problem_desc, log_entry=llm_interactions)
        if not q_code:
            return None, llm_interactions, q_code, q_test_cases, execution_results, "Failed to translate Python code to Q"
    
    print(f"  → Q solution ready ({len(q_code)} chars)")
    
    # Generate/regenerate test cases starting from specified test case number
    all_tests_passed = True
    failed_test_number = None
    
    for i, (test_file, expected_file) in enumerate(zip(test_case_files, expected_files), 1):
        print(f"    Test case {i}:")
        
        # Load test case and expected output
        with open(test_file, 'r', encoding='utf-8') as f:
            python_test = f.read()
        
        with open(expected_file, 'r', encoding='utf-8') as f:
            expected_output = f.read().strip()
        
        # Note: For simplicity, we regenerate all test cases in each attempt
        # Future optimization could reuse successful test cases
        
        # Translate test case to Q
        q_test = translate_test_case_to_q(python_test, expected_output, log_entry=llm_interactions)
        
        if not q_test:
            failed_test_number = i
            all_tests_passed = False
            error_msg = f"Failed to translate test case {i} to Q"
            return None, llm_interactions, q_code, q_test_cases, execution_results, error_msg
        
        q_test_cases.append(q_test)
        
        # NEW: Check for valid solve call count before execution
        solve_calls = count_solve_calls(q_test)
        if solve_calls != 1:
            print(f"      ✗ Invalid solve call count: {solve_calls} (expected 1)")
            failed_test_number = i
            all_tests_passed = False
            execution_results.append({
                'test_case_number': i,
                'combined_q_code': q_code + "\n\n" + q_test,
                'success': False,
                'q_output': None,
                'q_error': f"Generated test case has {solve_calls} solve calls (expected 1)",
                'expected_output': expected_output,
                'outputs_match': False
            })
            break # Exit loop over test cases

        # Create combined Q code (solution + test)
        combined_q_code = q_code + "\n\n" + q_test
        
        # Run Q code
        success, q_output, q_error = run_q_code(combined_q_code, timeout=DEFAULT_TIMEOUT, task_id_for_temp=problem_name)
        
        # Track execution results
        execution_result = {
            'test_case_number': i,
            'combined_q_code': combined_q_code,
            'success': success,
            'q_output': q_output,
            'q_error': q_error,
            'expected_output': expected_output,
            'outputs_match': False
        }
        execution_results.append(execution_result)
        
        if not success:
            print(f"      ✗ Q execution failed: {q_error}")
            failed_test_number = i
            all_tests_passed = False
            break
        
        print(f"      → Q output: {q_output}")
        print(f"      → Expected: {expected_output}")
        
        # Validate outputs match
        outputs_match = False
        if q_output.strip() == "" and expected_output.strip() != "":
            outputs_match = False
            print(f"      ✗ NOT EQUIVALENT (Q output was empty)")
        elif q_output.strip() == expected_output.strip():
            outputs_match = True
            print(f"      ✓ Exact match")
        else:
            # Use GPT to check equivalence
            try:
                equivalent = validate_q_outputs(q_output, expected_output, log_entry=llm_interactions)
                outputs_match = equivalent
                
                if equivalent:
                    print(f"      ✓ GPT confirmed equivalent")
                else:
                    print(f"      ✗ Outputs don't match")
            except Exception as e:
                print(f"      ✗ GPT validation failed: {e}")
                outputs_match = False  # Treat validation failures as non-equivalent
        
        execution_result['outputs_match'] = outputs_match

        if not outputs_match:
            failed_test_number = i
            all_tests_passed = False
            break
    
    if all_tests_passed:
        return True, llm_interactions, q_code, q_test_cases, execution_results, None
    else:
        return False, llm_interactions, q_code, q_test_cases, execution_results, f"Failed on test case {failed_test_number}"

def convert_single_problem(problem_dir, final_dir, stats, log_data, logging_dir, max_retries=3):
    """Convert a single problem from Python to Q with retry logic."""
    problem_name = problem_dir.name
    print(f"\n=== Converting: {problem_name} ===")
    
    # Initialize problem log entry
    problem_log = {
        'problem_name': problem_name,
        'timestamp': datetime.now().isoformat(),
        'status': 'unknown',
        'attempts': [],
        'final_attempt': None,
        'error': None,
        'skipped': False,
        'converted': False
    }
    
    # Check if already processed (check at processing time, not pre-computed)
    # This allows multiple instances to run safely
    if check_already_processed(problem_name, final_dir):
        print(f"  ✓ Already converted, skipping")
        problem_log['status'] = 'skipped'
        problem_log['skipped'] = True
        log_data['problems'].append(problem_log)
        stats['skipped'] += 1
        return True
    
    try:
        # Load problem description
        desc_path = problem_dir / "problem_description.txt"
        if not desc_path.exists():
            raise Exception("problem_description.txt not found")
        
        with open(desc_path, 'r', encoding='utf-8') as f:
            problem_desc = f.read()
        
        # Load Python solution
        python_sol_path = problem_dir / "python_sol.py"
        if not python_sol_path.exists():
            raise Exception("python_sol.py not found")
        
        with open(python_sol_path, 'r', encoding='utf-8') as f:
            python_code = f.read()
        
        print(f"  → Loaded Python solution ({len(python_code)} chars)")
        
        # Get test cases
        test_case_files = sorted(problem_dir.glob("python_test_case_*.py"))
        expected_files = sorted(problem_dir.glob("python_test_case_*_correct_ans.txt"))
        
        if len(test_case_files) != len(expected_files):
            raise Exception(f"Mismatch in test files: {len(test_case_files)} vs {len(expected_files)}")
        
        print(f"  → Found {len(test_case_files)} test cases to convert")
        
        # Retry logic
        final_success = False
        q_code_to_reuse = None
        final_q_code = None
        final_q_test_cases = []
        
        for attempt in range(1, max_retries + 1):
            print(f"  ⭐ Attempt {attempt}/{max_retries}")
            
            # Determine what to regenerate based on previous attempt
            regenerate_solution = (attempt == 1 or q_code_to_reuse is None)
            
            # Attempt conversion
            success, llm_interactions, q_code, q_test_cases, execution_results, error_msg = attempt_single_conversion(
                problem_dir, problem_name, problem_desc, python_code, test_case_files, expected_files,
                attempt, q_code_from_previous=q_code_to_reuse if not regenerate_solution else None
            )
            
            # Log this attempt
            attempt_log = {
                'attempt_number': attempt,
                'timestamp': datetime.now().isoformat(),
                'regenerated_solution': regenerate_solution,
                'success': success,
                'error': error_msg,
                'llm_interactions_count': len(llm_interactions),
                'q_code_length': len(q_code) if q_code else 0,
                'test_cases_generated': len(q_test_cases),
                'execution_results': execution_results
            }
            problem_log['attempts'].append(attempt_log)
            
            # Save detailed logs for this attempt
            save_detailed_logs(problem_name, logging_dir, llm_interactions, q_code, q_test_cases, execution_results, attempt)
            
            if success:
                print(f"  ✓ Attempt {attempt} succeeded!")
                final_success = True
                final_q_code = q_code
                final_q_test_cases = q_test_cases
                problem_log['final_attempt'] = attempt
                break
            else:
                print(f"  ✗ Attempt {attempt} failed: {error_msg}")
                
                # Determine retry strategy for next attempt
                if "Failed on test case 1" in error_msg:
                    # First test case failed - regenerate entire solution
                    print(f"    → Next attempt will regenerate Q solution (failed on first test case)")
                    q_code_to_reuse = None
                elif "Failed on test case" in error_msg:
                    # Later test case failed - keep solution, regenerate test cases
                    print(f"    → Next attempt will reuse Q solution (failed on later test case)")
                    q_code_to_reuse = q_code
                else:
                    # Other error - regenerate everything
                    print(f"    → Next attempt will regenerate Q solution (translation error)")
                    q_code_to_reuse = None
        
        if final_success:
            print(f"  🎉 Conversion successful after {problem_log['final_attempt']} attempt(s)")
            
            # Create final dataset directory
            dest_dir = final_dir / problem_name
            copy_problem_files(problem_dir, dest_dir, exclude_test_files=True)
            
            # Save enhanced problem description
            enhanced_desc = enhance_problem_description(problem_desc)
            with open(dest_dir / "problem_description.txt", 'w', encoding='utf-8') as f:
                f.write(enhanced_desc)
            
            # Save Q solution (the validated one)
            with open(dest_dir / "q_sol.q", 'w', encoding='utf-8') as f:
                f.write(final_q_code)
            
            # Copy Python test cases and expected answers
            print(f"  → Saving all {len(test_case_files)} test cases for final dataset")
            
            # Copy and rename Python test cases
            for i, (test_file, expected_file) in enumerate(zip(test_case_files, expected_files), 1):
                # Copy Python test case (remove python_ prefix)
                python_test_dest = dest_dir / f"test_case_{i}.py"
                shutil.copy2(test_file, python_test_dest)
                
                # Copy expected answer (remove python_ prefix) 
                expected_dest = dest_dir / f"test_case_{i}_correct_ans.txt"
                shutil.copy2(expected_file, expected_dest)
            
            # Save the VALIDATED Q test cases (not regenerated ones!)
            print(f"  → Saving {len(final_q_test_cases)} validated Q test cases")
            for i, q_test_case in enumerate(final_q_test_cases, 1):
                q_test_dest = dest_dir / f"q_test_case_{i}.q"
                with open(q_test_dest, 'w', encoding='utf-8') as f:
                    f.write(q_test_case)
                print(f"    ✓ Saved validated Q test case {i}")
            
            problem_log['status'] = 'converted'
            problem_log['converted'] = True
            stats['converted'] += 1
            
        else:
            print(f"  ❌ All {max_retries} attempts failed")
            problem_log['status'] = 'failed'
            stats['failed'] += 1
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        problem_log['status'] = 'error'
        problem_log['error'] = str(e)
        stats['failed'] += 1
        
        # Save error log
        error_log = {
            'attempt_number': 'error',
            'timestamp': datetime.now().isoformat(),
            'error': str(e),
            'traceback': traceback.format_exc()
        }
        save_detailed_logs(problem_name, logging_dir, [], None, [], [error_log], 'error')
    
    log_data['problems'].append(problem_log)
    return problem_log.get('converted', False)

def main():
    parser = argparse.ArgumentParser(description='Convert validated Python dataset to Q')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Maximum number of files to process (default: all)')
    parser.add_argument('--retries', type=int, default=8,
                        help='Maximum number of retry attempts per problem (default: 3)')
    parser.add_argument('--llm-to-use', choices=['gpt', 'local', 'claude'], default='gpt',
                        help='LLM to use: gpt (OpenAI), local (Hugging Face), or claude (Claude Sonnet 4)')
    parser.add_argument('--local-model-path', type=str, default=None,
                   help="Path to local model. If not provided, uses base_model from config.")
    parser.add_argument('--difficulty', choices=['Easy', 'Medium', 'Hard'], default=None,
                        help='Filter problems by difficulty level (default: all difficulties)')
    parser.add_argument('--problem-id', type=str, default=None,
                        help='Process only the specified problem ID/name (default: process all problems)')
    args = parser.parse_args()
    

        # parser.add_argument('--local-model-path', type=str, default="../../paper_code/rl_training/rl_output_new/checkpoint_1000/",
    # Initialize LLM
    global llm_instance
    global llm_type
    if args.llm_to_use == 'local':
        if not TRANSFORMERS_AVAILABLE:
            print("ERROR: torch and transformers are required for local model support!")
            print("Install with: pip install torch transformers")
            return
        
        print(f"Initializing local model...")
        try:
            llm_instance = LocalLLM(model_path=args.local_model_path)
            llm_type = "local"
            print(f"✓ Local model ready")
        except Exception as e:
            print(f"ERROR: Failed to load local model: {e}")
            return
    elif args.llm_to_use == 'claude':
        if not ANTHROPIC_AVAILABLE:
            print("ERROR: anthropic library is required for Claude support!")
            print("Install with: pip install anthropic")
            return
        llm_type = "claude"
        print(f"Using Claude Sonnet 4 model")
        llm_instance = None  # Claude is handled via Anthropic API
    else:
        llm_type = "gpt"
        print(f"Using OpenAI GPT model")
        llm_instance = None  # Use GPT
    
    # Set up directories
    validated_dir = Path("validated_dataset")
    final_dir = Path("final_dataset")
    logging_dir = Path("logging_final_dataset")
    final_dir.mkdir(exist_ok=True)
    logging_dir.mkdir(exist_ok=True)
    
    if not validated_dir.exists():
        print("ERROR: validated_dataset directory not found!")
        return
    
    # Get all problem directories
    all_problem_dirs = [d for d in validated_dir.iterdir() if d.is_dir()]
    
    # Filter by specific problem ID if specified
    if args.problem_id:
        print(f"Filtering by problem ID: {args.problem_id}")
        try:
            problem_id_num = int(args.problem_id)
            if problem_id_num < 1 or problem_id_num > len(all_problem_dirs):
                print(f"ERROR: Problem ID {problem_id_num} is out of range!")
                print(f"Available range: 1 to {len(all_problem_dirs)}")
                return
            
            # Sort problems using natural numeric sorting for consistent ordering
            def natural_sort_key(path):
                # Extract numbers from the path name and sort numerically
                import re
                parts = re.split(r'(\d+)', path.name)
                return [int(part) if part.isdigit() else part for part in parts]
            
            sorted_problem_dirs = sorted(all_problem_dirs, key=natural_sort_key)
            selected_problem = sorted_problem_dirs[problem_id_num - 1]  # Convert to 0-based index
            problem_dirs = [selected_problem]
            
            print(f"Found problem {problem_id_num}: {selected_problem.name}")
            
        except ValueError:
            print(f"ERROR: Problem ID '{args.problem_id}' must be a number!")
            print(f"Available range: 1 to {len(all_problem_dirs)}")
            return
    else:
        # Filter by difficulty if specified
        if args.difficulty:
            print(f"Filtering by difficulty: {args.difficulty}")
            problem_dirs = []
            for problem_dir in all_problem_dirs:
                entry_json_path = problem_dir / "entry.json"
                if entry_json_path.exists():
                    try:
                        with open(entry_json_path, 'r', encoding='utf-8') as f:
                            entry_data = json.load(f)
                        
                        problem_difficulty = entry_data.get('difficulty', '')
                        if problem_difficulty == args.difficulty:
                            problem_dirs.append(problem_dir)
                    except (json.JSONDecodeError, IOError) as e:
                        print(f"Warning: Could not read entry.json for {problem_dir.name}: {e}")
            
            print(f"Found {len(problem_dirs)} problems with difficulty '{args.difficulty}' out of {len(all_problem_dirs)} total")
        else:
            problem_dirs = all_problem_dirs
    
    if args.max_files and not args.problem_id:
        problem_dirs = problem_dirs[:args.max_files]
    
    # Randomize the order of problems to avoid conflicts when running multiple instances
    if not args.problem_id:  # Don't randomize if processing a specific problem
        random.shuffle(problem_dirs)
        print(f"Shuffled problem order randomly for parallel processing")
    
    print(f"Will process {len(problem_dirs)} problems")
    print(f"Max retries per problem: {args.retries}")
    print(f"LLM: {args.llm_to_use}")
    if args.problem_id:
        print(f"Target problem: {args.problem_id}")
    
    # Statistics
    stats = {
        'converted': 0,
        'failed': 0,
        'skipped': 0,
        'total': len(problem_dirs)
    }
    
    # Initialize log data structure
    log_data = {
        'timestamp': datetime.now().isoformat(),
        'total_problems': len(problem_dirs),
        'max_files_limit': args.max_files,
        'max_retries': args.retries,
        'llm_used': args.llm_to_use,
        'local_model_path': args.local_model_path if args.llm_to_use == 'local' else None,
        'difficulty_filter': args.difficulty,
        'problem_id_filter': args.problem_id,
        'problems': []
    }
    
    # Process each problem
    with tqdm(total=len(problem_dirs), desc="Converting to Q") as pbar:
        for problem_dir in problem_dirs:
            try:
                convert_single_problem(problem_dir, final_dir, stats, log_data, logging_dir, args.retries)
                
                # Rate limiting
                if (stats['converted'] + stats['failed']) % 5 == 0:
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
                    'skipped': False,
                    'converted': False
                }
                log_data['problems'].append(error_log)
                stats['failed'] += 1
            
            pbar.set_postfix({
                'converted': stats['converted'],
                'failed': stats['failed'],
                'skipped': stats['skipped']
            })
            pbar.update(1)
    
    # Save log data to JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"q_conversion_log_{timestamp}.json"
    
    # Add final statistics to log data
    log_data['final_stats'] = {
        'total_problems': stats['total'],
        'converted': stats['converted'],
        'failed': stats['failed'],
        'skipped': stats['skipped'],
        'conversion_rate': stats['converted']/stats['total']*100 if stats['total'] > 0 else 0
    }
    
    # Save log file
    with open(log_filename, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    # Final statistics
    print(f"\n{'='*50}")
    print(f"Q CONVERSION COMPLETE")
    print(f"{'='*50}")
    print(f"LLM used: {args.llm_to_use}")
    if args.llm_to_use == 'local':
        print(f"Local model: {args.local_model_path}")
    if args.llm_to_use == 'claude':
        print(f"Claude Sonnet 4 model")
    if args.difficulty:
        print(f"Difficulty filter: {args.difficulty}")
    if args.problem_id:
        print(f"Problem ID filter: {args.problem_id}")
    print(f"Total problems: {stats['total']}")
    print(f"Converted: {stats['converted']} ({stats['converted']/stats['total']*100:.1f}%)")
    print(f"Failed: {stats['failed']} ({stats['failed']/stats['total']*100:.1f}%)")
    print(f"Skipped: {stats['skipped']} ({stats['skipped']/stats['total']*100:.1f}%)")
    print(f"Results saved to: {final_dir.absolute()}")
    print(f"Detailed logs saved to: {logging_dir.absolute()}")
    print(f"Summary log saved to: {log_filename}")

if __name__ == "__main__":
    main() 