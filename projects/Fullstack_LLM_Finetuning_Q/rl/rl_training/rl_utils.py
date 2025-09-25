"""
Utility functions for RL training.

This module provides core utilities for RL training including code extraction,
execution helpers, and training utilities.
"""
import torch
import torch.nn.functional as F
import numpy as np
import random
import os
import re
from typing import Tuple

def seed_everything(seed: int) -> None:
    """
    Set all random seeds for reproducibility.
    
    Args:
        seed: Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def selective_log_softmax(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    A memory-efficient implementation of the common `log_softmax -> gather` operation.

    This function is equivalent to the following naive implementation:
    ```python
    logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    ```

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`.
        index (`torch.Tensor`):
            Index tensor of shape `(...)`, specifying the positions to gather from the log-softmax output.

    Returns:
        `torch.Tensor`:
            Gathered log probabilities with the same shape as `index`.
    """
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficent approach
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index):  # loop to reduce peak mem consumption
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps

def get_per_token_logps(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, logits_to_keep: int) -> torch.Tensor:
    # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
    logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
    logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

    input_ids = input_ids[:, -logits_to_keep:]
    # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
    # See https://github.com/huggingface/trl/issues/2770
    logits = logits[:, -logits_to_keep:]
    return selective_log_softmax(logits, input_ids)  #  compute logprobs for the input tokens

def extract_reasoning_from_response(response: str) -> str:
    """
    Extract the reasoning part from a response with reasoning/answer structure.
    
    Args:
        response: The LLM response text
        
    Returns:
        The extracted reasoning text
    """
    # Try different reasoning extraction patterns
    patterns = [
        r'<reasoning>(.*?)</reasoning>',
        r'reasoning:\s*(.*?)(?=<answer>|answer:|$)',
        r'let me think\s*(.*?)(?=<answer>|answer:|$)',
        r'step \d+\s*(.*?)(?=<answer>|answer:|$)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return ""

def extract_answer_from_response(response: str) -> str:
    """
    Extract the final answer from a response with reasoning/answer structure.
    
    Args:
        response: The LLM response text
        
    Returns:
        The extracted answer text
    """
    # Try different answer extraction patterns
    patterns = [
        r'<answer>(.*?)</answer>',
        r'answer:\s*(.*?)(?=\n\n|\Z)',
        r'the answer is\s*(.*?)(?=\n\n|\Z)',
        r'final answer:\s*(.*?)(?=\n\n|\Z)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # If no structured answer found, return empty string
    return ""

def extract_q_code_with_reasoning_support(text: str) -> Tuple[str, str, str]:
    """
    Extract Q code from text with support for reasoning format.
    
    Args:
        text: The input text (could be raw code or reasoning+answer format)
        
    Returns:
        tuple of (reasoning_part, answer_part, q_code)
        - reasoning_part: extracted reasoning text (empty if no reasoning format)
        - answer_part: extracted answer text (empty if no reasoning format)  
        - q_code: the final Q code to execute
    """
    # Check if this looks like reasoning format
    has_reasoning, has_answer = check_reasoning_and_answer_tags(text)
    
    if has_reasoning or has_answer:
        # Extract reasoning and answer parts
        reasoning_part = extract_reasoning_from_response(text)
        answer_part = extract_answer_from_response(text)
        
        # Extract Q code from the answer part
        if answer_part:
            q_code = extract_q_code(answer_part)
        else:
            # Fallback: try to extract from the whole text
            q_code = extract_q_code(text)
        
        return reasoning_part, answer_part, q_code
    else:
        # No reasoning format, treat as direct code
        q_code = extract_q_code(text)
        return "", "", q_code

def check_reasoning_and_answer_tags(response: str) -> Tuple[bool, bool]:
    """
    Check if the response follows proper reasoning and answer structure.
    
    Args:
        response: The LLM response text
        
    Returns:
        has_reasoning: Whether response contains reasoning structure
        has_answer: Whether response contains answer structure
    """
    # Look for common reasoning patterns
    reasoning_patterns = [
        r'<reasoning>.*?</reasoning>',
        r'reasoning:.*?(?=answer:|$)',
        r'let me think.*?(?=answer:|$)',
        r'first.*?then.*?(?=answer:|$)',
        r'step \d+.*?(?=answer:|$)',
    ]
    
    has_reasoning = any(re.search(pattern, response.lower(), re.DOTALL) for pattern in reasoning_patterns)
    
    # Look for answer patterns
    answer_patterns = [
        r'<answer>.*?</answer>',
        r'answer:.*?(?=\n\n|\Z)',
        r'the answer is.*?(?=\n\n|\Z)',
        r'final answer.*?(?=\n\n|\Z)',
    ]
    
    has_answer = any(re.search(pattern, response.lower(), re.DOTALL) for pattern in answer_patterns)
    
    return has_reasoning, has_answer

def extract_q_code(text: str) -> str:
    """
    Robustly extracts Q code from a string, handling markdown and partial fences.
    """
    # First, try to find a ```q block
    q_pattern = r"```q\s*(.*?)\s*```"
    match = re.search(q_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # If not, try to find a generic ``` block
    any_pattern = r"```\s*(.*?)\s*```"
    match = re.search(any_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # If no blocks, assume the whole text is code and clean partial fences
    cleaned_code = text.strip()
    if cleaned_code.startswith("```q"):
        cleaned_code = cleaned_code.removeprefix("```q").strip()
    if cleaned_code.startswith("```"):
        cleaned_code = cleaned_code.removeprefix("```").strip()
    if cleaned_code.endswith("```"):
        cleaned_code = cleaned_code.removesuffix("```").strip()
        
    return cleaned_code 