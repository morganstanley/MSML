#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
# Adapted from huggingface/open-r1: https://github.com/huggingface/open-r1/blob/6a0cd5c8ad031fc75118a4ce7f42a4860c3d8dea/src/open_r1/rewards.py


"""Reward functions for GRPO training."""
import asyncio
import json
import math
import re
from functools import partial, update_wrapper
from typing import Callable, Dict, Optional

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

from utils_code.code_providers import get_provider

from utils_code.competitive_programming import (
    SubtaskResult,
    add_includes,
    get_morph_client_from_env,
    get_piston_client_from_env,
)

from utils_code.competitive_programming import patch_code as cf_patch_code
from utils_code.competitive_programming import score_submission as cf_score_submission
from utils_code.competitive_programming import score_subtask



def accuracy_reward(completions: list[list[dict[str, str]]], solution: list[str], **kwargs) -> list[Optional[float]]:
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
        )
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # Compute binary rewards if verifiable, `None` otherwise to skip this example
            try:
                reward = float(verify(gold_parsed, answer_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
                reward = None
        else:
            # If the gold solution is not parseable, we assign `None` to skip this example
            reward = None
            print("Failed to parse gold solution: ", sol)
        rewards.append(reward)

    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def tag_count_reward(completions, **kwargs) -> list[float]:
    """Reward function that checks if we produce the desired number of think and answer tags associated with `format_reward()`.

    Adapted from: https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb#file-grpo_demo-py-L90
    """

    def count_tags(text: str) -> float:
        count = 0.0
        if text.count("<think>\n") == 1:
            count += 0.25
        if text.count("\n</think>\n") == 1:
            count += 0.25
        if text.count("\n<answer>\n") == 1:
            count += 0.25
        if text.count("\n</answer>") == 1:
            count += 0.25
        return count

    contents = [completion[0]["content"] for completion in completions]
    return [count_tags(c) for c in contents]


def reasoning_steps_reward(completions, **kwargs):
    r"""Reward function that checks for clear step-by-step reasoning.
    Regex pattern:
        Step \d+: - matches "Step 1:", "Step 2:", etc.
        ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
        \n- - matches bullet points with hyphens
        \n\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [len(re.findall(pattern, content)) for content in completion_contents]

    # Magic number 3 to encourage 3 steps and more, otherwise partial reward
    return [min(1.0, count / 3) for count in matches]


def len_reward(completions: list[Dict[str, str]], solution: list[str], **kwargs) -> float:
    """Compute length-based rewards to discourage overthinking and promote token efficiency.

    Taken from the Kimi 1.5 tech report: https://arxiv.org/abs/2501.12599

    Args:
        completions: List of model completions
        solution: List of ground truth solutions

    Returns:
        List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """
    contents = [completion[0]["content"] for completion in completions]

    # First check correctness of answers
    correctness = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) == 0:
            # Skip unparseable examples
            correctness.append(True)  # Treat as correct to avoid penalizing
            print("Failed to parse gold solution: ", sol)
            continue

        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        correctness.append(verify(answer_parsed, gold_parsed))

    # Calculate lengths
    lengths = [len(content) for content in contents]
    min_len = min(lengths)
    max_len = max(lengths)

    # If all responses have the same length, return zero rewards
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

        if is_correct:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)

        rewards.append(float(reward))

    return rewards


def get_cosine_scaled_reward(
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
):
    def cosine_scaled_reward(completions, solution, **kwargs):
        """Reward function that scales based on completion length using a cosine schedule.

        Shorter correct solutions are rewarded more than longer ones.
        Longer incorrect solutions are penalized less than shorter ones.

        Args:
            completions: List of model completions
            solution: List of ground truth solutions

        This function is parameterized by the following arguments:
            min_value_wrong: Minimum reward for wrong answers
            max_value_wrong: Maximum reward for wrong answers
            min_value_correct: Minimum reward for correct answers
            max_value_correct: Maximum reward for correct answers
            max_len: Maximum length for scaling
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol in zip(contents, solution):
            gold_parsed = parse(
                sol,
                extraction_mode="first_match",
                extraction_config=[LatexExtractionConfig()],
            )
            if len(gold_parsed) == 0:
                rewards.append(1.0)  # Skip unparseable examples
                print("Failed to parse gold solution: ", sol)
                continue

            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed=True,
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

            is_correct = verify(answer_parsed, gold_parsed)
            gen_len = len(content)

            # Apply cosine scaling based on length
            progress = gen_len / max_len
            cosine = math.cos(progress * math.pi)

            if is_correct:
                min_value = min_value_correct
                max_value = max_value_correct
            else:
                # Swap min/max for incorrect answers
                min_value = max_value_wrong
                max_value = min_value_wrong

            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))

        return rewards

    return cosine_scaled_reward


def get_repetition_penalty_reward(ngram_size: int, max_penalty: float, language: str = "en"):
    """
    Computes N-gram repetition penalty as described in Appendix C.2 of https://arxiv.org/abs/2502.03373.
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
    ngram_size: size of the n-grams
    max_penalty: Maximum (negative) penalty for wrong answers
    language: Language of the text, defaults to `en`. Used to choose the way to split the text into n-grams.
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    if language == "en":

        def zipngram(text: str, ngram_size: int):
            words = text.lower().split()
            return zip(*[words[i:] for i in range(ngram_size)]), words

    elif language == "zh":
        from transformers.utils.import_utils import _is_package_available

        if not _is_package_available("jieba"):
            raise ValueError("Please install jieba to use Chinese language")

        def zipngram(text: str, ngram_size: int):
            import jieba

            seg_list = list(jieba.cut(text))
            return zip(*[seg_list[i:] for i in range(ngram_size)]), seg_list

    else:
        raise ValueError(
            f"Word splitting for language `{language}` is not yet implemented. Please implement your own zip-ngram function."
        )

    def repetition_penalty_reward(completions, **kwargs) -> float:
        """
        reward function the penalizes repetitions
        ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

        Args:
            completions: List of model completions
        """

        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion in contents:
            if completion == "":
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            ngram_array, words = zipngram(completion, ngram_size)

            if len(words) < ngram_size:
                rewards.append(0.0)
                continue

            for ng in ngram_array:
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty
            rewards.append(reward)
        return rewards

    return repetition_penalty_reward


def _init_event_loop():
    """Initialize or get the current event loop."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def ioi_code_reward(completions, test_batch_size: int = 1, provider_type: str = "piston", **kwargs) -> list[float]:
    """Reward function that evaluates IOI problems using a specified execution client.

    Assumes the dataset has the same format as hf.co/datasets/open-r1/ioi

    Args:
        completions: List of model completions to evaluate
        test_batch_size: Evaluate these many test cases in parallel, then check if any of them failed (0 score):
                       if so stop evaluating; otherwise continue with the next batch of test cases.
        provider_type: The execution provider to use (default: "piston"). Supported values: "piston", "morph"
        **kwargs: Additional arguments passed from the dataset
    """
    # Get the appropriate client based on provider_type
    if provider_type == "morph":
        execution_client = get_morph_client_from_env()
    else:
        # for info on setting up piston workers, see slurm/piston/README.md
        execution_client = get_piston_client_from_env()

    code_snippets = [
        # note: grading is automatically skipped if no code is extracted
        add_includes(extract_code(completion[-1]["content"], "cpp"), problem_id)
        for completion, problem_id in zip(completions, kwargs["id"])
    ]

    async def run_catch_exceptions(task):
        try:
            return await task
        except Exception as e:
            print(f"Error from {provider_type} worker: {e}")
            return SubtaskResult()

    problems_data = [dict(zip(kwargs.keys(), values)) for values in zip(*kwargs.values())]

    loop = _init_event_loop()
    evals = [
        loop.create_task(
            run_catch_exceptions(
                score_subtask(
                    execution_client,
                    problem_data,
                    code,
                    test_batch_size=test_batch_size,
                )
            )
        )
        for problem_data, code in zip(problems_data, code_snippets)
    ]
    results = loop.run_until_complete(asyncio.gather(*evals))

    return [result.score for result in results]


def extract_code(completion: str, language: str = "python") -> str:
    pattern = re.compile(rf"```{language}\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    extracted_answer = matches[0] if len(matches) >= 1 else ""
    return extracted_answer


def binary_code_reward(
    completions,
    num_parallel: int = 2,
    provider_type: str = "e2b",
    enforce_same_language: bool = False,
    **kwargs,
) -> list[float]:
    rewards = code_reward(
        completions,
        num_parallel=num_parallel,
        provider_type=provider_type,
        enforce_same_language=enforce_same_language,
        **kwargs,
    )
    BINARY_THRESHOLD = 0.99

    output = []
    for reward in rewards:
        if reward is None:
            output.append(None)
        else:
            output.append(1.0 if reward > BINARY_THRESHOLD else 0.0)

    return output


def code_reward(
    completions,
    num_parallel: int = 1, # change to 1 sep 14 11pm trial
    provider_type: str = "e2b",
    enforce_same_language: bool = False,
    **kwargs,
) -> list[float]:
    """Reward function that evaluates code snippets using a code execution provider.

    Assumes the dataset contains a `verification_info` column with test cases.

    Args:
        completions: List of model completions to evaluate
        num_parallel: Number of parallel code executions (default: 2)
        provider_type: Which code execution provider to use (default: "e2b")
        enforce_same_language: If True, verify all problems use the same language (default: False)
        **kwargs: Additional arguments passed to the verification
    """
    evaluation_script_template = """
import subprocess
import json

def evaluate_code(code, test_cases):
    passed = 0
    total = len(test_cases)
    exec_timeout = 30 # original is 5, to avoid stuck at 1.5k iterations

    for case in test_cases:
        process = subprocess.run(
            ["python3", "-c", f"{{code}}\\n{{case}}"],
            text=True,
            capture_output=True,
            timeout=exec_timeout
        )

        if process.returncode != 0:  # Error in execution
            continue

        # If we get here, the assertion passed (no error)
        passed += 1

    success_rate = (passed / total)
    return success_rate

code_snippet = {code_literal}
test_cases = {test_cases_literal}
rate = evaluate_code(code_snippet, test_cases)
print(rate)
"""
    # 1. compute format rewards
    format_rewards = get_code_format_reward(language='python')(completions)
    # 2. collect scripts and their indices in the original array
    template = evaluation_script_template
    scripts = []
    valid_indices = []
    code_list = []
    test_cases = []
    for i, (reward, completion) in enumerate(zip(format_rewards, completions)):
        if reward < 1:
            continue
        code = extract_code(completion[-1]["content"])
        code_list.append(code)
        tc = kwargs["test_list"][i]
        test_cases.append(tc)
        scripts.append(
            template.format(
                code_literal=repr(code),
                test_cases_literal=repr(tc),
            )
        )
        valid_indices.append(i)
    # 3. execute scripts in parallel
    execution_provider = get_provider(
        provider_type=provider_type,
        num_parallel=num_parallel,
        **kwargs,
    )

    results = execution_provider.execute_scripts(scripts, ["python"] * len(scripts))
    # print(f"Results: {repr(results)}")
    # 4. fill results into a list of the same length as completions, and keep None for reward=0
    final_results = [0.0] * len(completions)
    for idx, res in zip(valid_indices, results):
        final_results[idx] = res

    try:
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        BLUE = "\033[94m"
        RESET = "\033[0m"
        print("*" * 100,
            f"\n{GREEN}Code:{RESET}\n{code_list[0]}\n",
            "-" * 20,
            f"\n{BLUE}Test Case:{RESET}\n{test_cases[0]}\n",
            "-" * 20,
            f"\n{YELLOW}Results:{RESET}\n{final_results}\n")
    except:
        pass

    return final_results


def get_code_format_reward(language: str = "python"):
    """Format reward function specifically for code responses.

    Args:
        language: Programming language supported by E2B https://e2b.dev/docs/code-interpreting/supported-languages
    """
    import ast
    pattern = re.compile(
        rf"^"
        r"(?:(?!```)[\s\S])*?"
        rf"```{language}\n"    # match ```language\n
        r"(?:(?!```)[\s\S])*?"         # match any character, but not ```
        rf"```\n$",                     # match ``` and end of string
        re.DOTALL
    )

    def code_format_reward(completions, **kwargs):
        completion_contents = [completion[0]["content"] for completion in completions]
        rewards = []
        
        for content in completion_contents:
            # # First check if the format matches
            # import pdb; pdb.set_trace();
            
            # ignore this part
            #format_match = pattern.fullmatch(content)
            #if not format_match:
            #    rewards.append(0.0)
            #    continue
                
            # Extract code from between code blocks
            code_blocks = re.findall(rf"```{language}\n(.*?)```", content, re.DOTALL)
            if not code_blocks:
                rewards.append(0.0)
                continue
                
            # Get the first code block (in case there are multiple)
            code = code_blocks[0].strip()
            
            # Check syntax if it's Python code
            if language == "python":
                try:
                    ast.parse(code)
                    syntax_valid = True
                except SyntaxError:
                    syntax_valid = False
                rewards.append(1.0 if syntax_valid else 0.5) #  grammar error get partial reward
            else:
                # For other languages, just check format for now
                rewards.append(1.0)
                
        return rewards

    return code_format_reward


def get_soft_overlong_punishment(max_completion_len, soft_punish_cache):
    """
    Reward function that penalizes overlong completions. It is used to penalize overlong completions,
    but not to reward shorter completions. Reference: Eq. (13) from the DAPO paper (https://huggingface.co/papers/2503.14476)

    Args:
        max_completion_len: Maximum length of the completion
        soft_punish_cache: Minimum length of the completion. If set to 0, no minimum length is applied.
    """

    def soft_overlong_punishment_reward(completion_ids: list[list[int]], **kwargs) -> list[float]:
        """Reward function that penalizes overlong completions."""
        rewards = []
        for ids in completion_ids:
            completion_length = len(ids)
            if completion_length <= max_completion_len - soft_punish_cache:
                rewards.append(0.0)
            elif max_completion_len - soft_punish_cache < completion_length <= max_completion_len:
                rewards.append((max_completion_len - soft_punish_cache - completion_length) / soft_punish_cache)
            else:
                rewards.append(-1.0)
        return rewards

    return soft_overlong_punishment_reward


def get_reward_funcs(script_args) -> list[Callable]:
    REWARD_FUNCS_REGISTRY = {
        "accuracy": accuracy_reward,
        "format": format_reward,
        "reasoning_steps": reasoning_steps_reward,
        "cosine": get_cosine_scaled_reward(
            min_value_wrong=script_args.cosine_min_value_wrong,
            max_value_wrong=script_args.cosine_max_value_wrong,
            min_value_correct=script_args.cosine_min_value_correct,
            max_value_correct=script_args.cosine_max_value_correct,
            max_len=script_args.cosine_max_len,
        ),
        "repetition_penalty": get_repetition_penalty_reward(
            ngram_size=script_args.repetition_n_grams,
            max_penalty=script_args.repetition_max_penalty,
        ),
        "length": len_reward,
        "code": update_wrapper(
            partial(
                code_reward,
                num_parallel=script_args.parallel_code_exec_per_proc,
                provider_type=script_args.code_provider,
                enforce_same_language=getattr(script_args, "enforce_same_language", False),
            ),
            code_reward,
        ),
        "binary_code": update_wrapper(
            partial(
                binary_code_reward,
                num_parallel=script_args.parallel_code_exec_per_proc,
                provider_type=script_args.code_provider,
                enforce_same_language=getattr(script_args, "enforce_same_language", False),
            ),
            binary_code_reward,
        ),
        "ioi_code": update_wrapper(
            partial(
                ioi_code_reward,
                test_batch_size=script_args.code_eval_test_batch_size,
                provider_type=getattr(script_args, "ioi_provider", "piston"),
            ),
            ioi_code_reward,
        ),
        "code_format": get_code_format_reward(language=script_args.code_language),
        "tag_count": tag_count_reward,
        "soft_overlong_punishment": get_soft_overlong_punishment(
            max_completion_len=script_args.max_completion_len,
            soft_punish_cache=script_args.soft_punish_cache,
        ),
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    return reward_funcs
