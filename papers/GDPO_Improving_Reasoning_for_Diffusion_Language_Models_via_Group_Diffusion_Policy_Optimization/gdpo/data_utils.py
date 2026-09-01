import os
import re
import random

from datasets import load_dataset, Dataset
import pandas as pd
import numpy as np
import torch

from reward_func import extract_hash_answer




def set_random_seed(seed: int = 42):
    # Set the seed for Python's built-in random module
    random.seed(seed)
    # Set the seed for NumPy
    np.random.seed(seed)
    # Set the seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in cuDNN (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False






CODE_PROMPT = """
Always respond with Python code in this exact format:
<reasoning>
...
</reasoning>

<answer>
```python
# python code only, no tests, no I/O, no prints, no asserts, no examples
```
</answer>

Do not include unit tests, assertions, prints, examples for debugging.
"""

# Constants for prompts
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

SUDOKU_SYSTEM_PROMPT = """
Please solve the following 4x4 Sudoku puzzle. The puzzle is provided as a 16-character string reading left-to-right, top-to-bottom, where '0' represents empty cells.

Rules:
- Fill empty cells with digits 1-4
- Each row must contain digits 1-4 exactly once
- Each column must contain digits 1-4 exactly once
- Each 2x2 box must contain digits 1-4 exactly once

Important: Your solution must be a COMPLETE 16-character string with only the digits 1-4, representing your final solved grid.

Respond in this exact format:
<reasoning>
Your step-by-step solving process
</reasoning>
<answer>
[16-character solution string with no spaces or separators]
</answer>
"""


XML_COT_FORMAT = """
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""


def get_gsm8k_questions(split="train") -> Dataset:
    data = load_dataset("openai/gsm8k", "main")[split]
    return data.map(
        lambda x: {
            "prompt": [
                {"role": "user", "content": SYSTEM_PROMPT + "\n\n" + x["question"]},
            ],
            "answer": extract_hash_answer(x["answer"]),
        }
    )


def get_countdown_questions(split="train") -> Dataset:
    data = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split=split)
    data = data.filter(lambda x: len(x["nums"]) == 3)

    return data.map(
        lambda x: {
            "prompt": [
                {
                    "role": "user",
                    "content": f"{SYSTEM_PROMPT}\nUsing only the numbers {x['nums']}, create an arithmetic expression that evaluates to exactly {x['target']}. You must use all numbers from the list, and each number must be used exactly once. You may use the operations +, -, *, and / as needed. After reasoning, provide only your final expression inside <answer></answer> tags without including an equals sign or the target number. For example, if the numbers are [2, 3, 4] and the target is 5, a valid answer is: <answer>\n2*4-3\n</answer>",
                },
            ],
            "target": x["target"],
            "numbers": x["nums"],
        }
    )


def get_sudoku_questions() -> Dataset:
    """Load the Sudoku dataset for training or evaluation."""
    cur_path = os.path.dirname(os.path.abspath(__file__))
    sudoku_file_path = "../dataset/4x4_sudoku_unique_puzzles.csv"
    sudoku_file_path = os.path.join(cur_path, sudoku_file_path)
    df = pd.read_csv(sudoku_file_path, dtype={"Puzzle": str, "Solution": str})
    data = Dataset.from_pandas(df)

    return data.map(
        lambda x: {
            "prompt": [
                {
                    "role": "user",
                    "content": f"{SUDOKU_SYSTEM_PROMPT}\n\nSolve the following Sudoku puzzle: {x['Puzzle']}\n",
                },
            ],
            "puzzle": x["Puzzle"],
            "solution": x["Solution"],
        }
    )


def get_math_questions(split="train") -> Dataset:
    data = load_dataset("ankner/math-500", split=split)  # type: ignore
    data = data.map(
        lambda x: {  # type: ignore
            "prompt": [
                {
                    "role": "user",
                    "content": f"{SYSTEM_PROMPT}\n\nYou are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{{}}. \n\n{x['problem']}",
                },
            ],
            "answer": x["solution"],
        }
    )  # type: ignore
    return data  # type: ignore


# def humaneval_prompt(func): # prompt for humaneval
#      prompt = f'''Role: You are a professional Python coding assistant
# Task: Complete the follow function implementation strictly and clearly without any additional comments or explanations.
# {func}'''
#      return prompt


# from https://github.com/NEUIR/PC-Sampler/blob/master/src/template.py
def get_code_prompt(task, func_name_pars):
    return f'''Role: You are a professional Python coding assistant
Task: Complete the follow function implementation strictly and clearly without any additional comments or explanations.
{task}
the function name and the parameters is {func_name_pars}
Again, implement the above function strictly without any additional comments, explanations, unit tests, assertions, prints, examples in the code.
'''

def get_KodCode_light_rl_10k(split: str = "train"):
    dataset = load_dataset("KodCode/KodCode-Light-RL-10K", split=split)

    def extract_test_cases(test_str):
        pattern = r'^\s*(assert\s+.*)'
        asserts = re.findall(pattern, test_str, re.MULTILINE)
        return asserts

    dataset = dataset.map(lambda x: {"test_list": extract_test_cases(x["test"])})
    dataset = dataset.filter(lambda x: len(x["test_list"]) >= 3)
    dataset = dataset.map(lambda x: {'func_name_pars': x['test_info'][0]['function_name'] + x['test_info'][0]['parameter_list'] })

    return dataset.map(
        lambda x: {
            "prompt": [
                {
                    "role": "user",
                    "content": (get_code_prompt(x['question'], x['func_name_pars'])),
                }
            ],
            # "test_list": x["test_list"],
            "solution": x["solution"],
        }
    )



def extract_function_name(s: str) -> str:
    funcs = re.findall(r'([a-zA-Z_]\w*)\s*\(', s)
    builtins = {"int", "float", "str", "list", "dict", "set", "tuple", "len", "assert"}
    funcs = [f for f in funcs if f not in builtins]
    return funcs[0] if funcs else None

def get_func_name_pars(code_str, func_name):
    pattern = r"def\s+([A-Za-z_]\w*)\s*\(([^)]*)\)\s*:"

    matches = re.findall(pattern, code_str)
    for name, args in matches:
        if func_name == name:
            return f"{name}({args})"
    return ""


def get_mbpp_questions() -> Dataset:
    """Load the MBPP dataset for training or evaluation."""
    cur_path = os.path.dirname(os.path.abspath(__file__))
    mbpp_file_path = "../dataset/mbpp/mbpp.jsonl"
    mbpp_file_path = os.path.join(cur_path, mbpp_file_path)

    df = pd.read_json(mbpp_file_path, lines=True)
    data = Dataset.from_pandas(df)
    data = data.map(lambda x: {'func_name': extract_function_name(x['test_list'][0])})
    data = data.map(lambda x: {'func_name_pars': get_func_name_pars(x['code'], x['func_name'])})

    return data.map(
        lambda x: {
            "prompt": [
                {
                    "role": "user",
                    "content": (get_code_prompt(x['text'], x['func_name_pars'])),
                },
            ],
            # "test_list": x["test_list"],
            "solution": x["code"],
        }
    )



def get_humaneval_questions(split: str = "test"):
    # Load HumanEval (only has "test" split)
    dataset = load_dataset("openai/openai_humaneval", split=split)

    def extract_humaneval_test_cases(test_str: str, entry_point):
        pattern = r'^\s*(assert\s+.*)'
        cases = re.findall(pattern, test_str, re.MULTILINE)
        cases = [re.sub(r"\bcandidate\(", f"{entry_point}(", a) for a in cases]
        return cases
    dataset = dataset.map(lambda x: {"test_list": extract_humaneval_test_cases(x["test"], x["entry_point"])})

    dataset = dataset.map(
        lambda x: {
            "prompt": [
                {
                    "role": "user",
                    "content": (
                        f"You are an expert Python programmer. Task: Complete the follow function implementation strictly and clearly without any additional comments or explanations. {x['prompt']}."
                    ),
                }
            ],
            # "test_list": x["test_list"],
            "solution": x["canonical_solution"],
        }
    )

    return dataset

