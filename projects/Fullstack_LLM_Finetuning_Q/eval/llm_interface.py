#!/usr/bin/env python3
"""
Simple LLM Interface for Evaluation

Clean interface for OpenAI, Anthropic, HuggingFace, vLLM API, and Ground Truth models.
Assumes prompts are fully prepared (including context) before calling.
"""

import os
import re
import time
import random
from pathlib import Path
import openai
import anthropic
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import Optional
import requests

from evaluation_utils import logger

# import google.generativeai as genai
# from google.generativeai.types import GenerateContentConfig, ThinkingConfig


def exponential_backoff_retry(func, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0, 
                            jitter: bool = True):
    """
    Retry function with exponential backoff.
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        jitter: Whether to add random jitter to delays
    """
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries:
                raise e
            
            # Calculate delay with exponential backoff
            delay = min(base_delay * (2 ** attempt), max_delay)
            
            # Add jitter to prevent thundering herd
            if jitter:
                delay = delay * (0.5 + random.random() * 0.5)
            
            print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s...")
            time.sleep(delay)


class LLMInterface:
    """Simple interface for generating text from language models."""
    
    def __init__(self, model_type: str, model_name: str, device: str = "cuda", test_dir: str = "../SFT_Data/test", 
                 api_base: str = None, api_key: str = None, max_new_tokens: int = 4096):
        self.model_type = model_type
        self.model_name = model_name
        self.device = device
        self.test_dir = test_dir
        self.api_base = api_base
        self.api_key = api_key
        self.max_new_tokens = max_new_tokens
        
        if model_type == "openai":
            if api_key:
                self.client = openai.OpenAI(api_key=api_key)
            else:
                self.client = openai.OpenAI()  # Uses OPENAI_API_KEY env var
        elif model_type == "xai":
            if not api_key:
                raise ValueError("API key is required for xAI models")
            self.client = openai.OpenAI(
                base_url="https://api.x.ai/v1",
                api_key=api_key
            )
        elif model_type == "google":
            if genai is None:
                raise ImportError("google-generativeai package not found. Please install it with 'pip install google-generativeai'")
            if api_key:
                genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(self.model_name)
        elif model_type == "anthropic":
            if api_key:
                self.client = anthropic.Anthropic(api_key=api_key)
            else:
                self.client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var
        elif model_type == "vllm_api":
            # vLLM API uses OpenAI-compatible interface
            if not api_base:
                raise ValueError("api_base is required for vllm_api model type")
            self.client = openai.OpenAI(
                base_url=api_base,
                api_key=api_key or "dummy-key",  # vLLM doesn't require a real key by default
                timeout=30.0  # Add 30 second timeout to prevent hanging
            )
        elif model_type == "huggingface":
            # Check if this is a LoRA checkpoint
            model_path = Path(model_name)
            base_model_name = "Qwen/Qwen2.5-14B-Instruct"  # Base model for LoRA
            
            if model_path.exists() and (model_path / "adapter_config.json").exists():
                # This is a LoRA checkpoint
                print(f"Detected LoRA checkpoint: {model_name}")
                print(f"Loading base model: {base_model_name}")
                
                # Load tokenizer from base model
                self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
                
                # Load base model with optimizations
                print("Loading base model with Flash Attention 2...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                    attn_implementation="flash_attention_2"  # Enable Flash Attention 2
                )
                
                # Load and apply LoRA adapter
                print(f"Loading LoRA adapter from: {model_name}")
                self.model = PeftModel.from_pretrained(self.model, model_name)
                print("LoRA adapter loaded successfully")
                
            else:
                # Regular model loading
                print(f"Loading standard Hugging Face model: {model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                print("Loading model with Flash Attention 2...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                    attn_implementation="flash_attention_2"  # Enable Flash Attention 2
                )
            
            # Apply torch.compile() for additional speedup
            print("Compiling model with torch.compile()...")
            try:
                self.model = torch.compile(self.model, mode="default")
                print("Model compilation successful")
            except Exception as e:
                print(f"Warning: torch.compile() failed, continuing without compilation: {e}")
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        elif model_type == "ground_truth":
            # No initialization needed for ground truth
            pass
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _extract_ground_truth_info(self, prompt: str) -> tuple[str, str]:
        """Extract problem name and task type from prompt for ground truth lookup."""
        # Look for patterns like "GROUND_TRUTH_INFO: problem_name=268_missing-number, task_type=python_to_q"
        match = re.search(r'GROUND_TRUTH_INFO:\s*problem_name=([^,\s]+),\s*task_type=([^,\s]+)', prompt)
        if match:
            return match.group(1), match.group(2)
        
        # If no explicit info, try to infer from prompt context
        # This is a fallback - the evaluation script should provide explicit info
        return None, None
    
    def _get_ground_truth_solution(self, problem_name: str, task_type: str) -> str:
        """Get the ground truth solution for the given problem and task type."""
        test_path = Path(self.test_dir)
        problem_dir = test_path / problem_name
        
        if not problem_dir.exists():
            return f"ERROR: Problem directory not found: {problem_name}"
        
        if task_type in ['description_to_q', 'python_to_q']:
            # Return Q solution
            q_sol_file = problem_dir / 'q_sol.q'
            if q_sol_file.exists():
                try:
                    with open(q_sol_file, 'r', encoding='utf-8') as f:
                        return f.read().strip()
                except Exception as e:
                    return f"ERROR: Failed to read Q solution: {e}"
            else:
                return f"ERROR: Q solution file not found for {problem_name}"
        
        elif task_type == 'q_to_python':
            # Return Python solution
            python_sol_file = problem_dir / 'python_sol.py'
            if python_sol_file.exists():
                try:
                    with open(python_sol_file, 'r', encoding='utf-8') as f:
                        return f.read().strip()
                except Exception as e:
                    return f"ERROR: Failed to read Python solution: {e}"
            else:
                return f"ERROR: Python solution file not found for {problem_name}"
        
        else:
            return f"ERROR: Unknown task type: {task_type}"

    def generate(self, prompt: str, problem_name: Optional[str] = None, task_type: Optional[str] = None) -> str:
        """Generate response from the model."""
        if self.model_type == "ground_truth":
            # For ground truth, extract problem info and return actual solution
            if problem_name and task_type:
                return self._get_ground_truth_solution(problem_name, task_type)
            else:
                # Try to extract from prompt
                extracted_name, extracted_task = self._extract_ground_truth_info(prompt)
                if extracted_name and extracted_task:
                    return self._get_ground_truth_solution(extracted_name, extracted_task)
                else:
                    return "ERROR: Ground truth model requires problem_name and task_type"
        
        elif self.model_type in ["openai", "xai"]:
            # o3 models require temperature 1.0, others use 0.7
            temperature = 1.0 if "o3" in self.model_name.lower() else 0.7
            
            params = {
                'model': self.model_name,
                'messages': [{"role": "user", "content": prompt}],
                'temperature': temperature
            }

            # Add reasoning_effort only for specific Google models that support it.
            # Based on API errors, this feature appears to be model-specific.
            if self.model_type == 'google' and self.model_name in ['gemini-2.5-flash']:
                params['reasoning_effort'] = 'low'

            # o3 models use max_completion_tokens instead of max_tokens
            if "o3" in self.model_name.lower():
                params['max_completion_tokens'] = self.max_new_tokens
            else:
                params['max_tokens'] = self.max_new_tokens
            
            response = self.client.chat.completions.create(**params)
            
            logger.info(f"Full API Response: {response.model_dump_json(indent=2)}")

            if not response.choices:
                logger.warning("API response contained no choices.")
                return ""
            
            choice = response.choices[0]
            if not choice.message:
                logger.warning(f"Choice {choice.index} contained no message.")
                return ""

            content = choice.message.content
            
            if content is None:
                logger.warning(f"Choice {choice.index} message content was None. Finish reason: {choice.finish_reason}")
                return ""
                
            return content.strip()

        elif self.model_type == "google":
            generation_config = GenerateContentConfig(
                max_output_tokens=self.max_new_tokens,
                temperature=0.7,
                thinking_config=ThinkingConfig(thinking_budget=1024)
            )
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            return response.text.strip()
        
        elif self.model_type == "anthropic":
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.max_new_tokens,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
        
        elif self.model_type == "vllm_api":
            # Use exponential backoff for API calls
            def make_api_call():
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=self.max_new_tokens
                )
                return response.choices[0].message.content.strip()
            
            return exponential_backoff_retry(make_api_call, max_retries=3)
        
        elif self.model_type == "huggingface":
            messages = [{"role": "user", "content": prompt}]
            chat = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            
            with torch.no_grad():
                inputs = self.tokenizer(chat, return_tensors="pt").to(self.device)
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=self.max_new_tokens, # Increased max tokens for larger models
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Extract only the new tokens
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            return response.strip()
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")


def create_llm_interface(model_type: str, model_name: str, device: str = "cuda", test_dir: str = "../SFT_Data/test",
                        api_base: str = None, api_key: str = None, max_new_tokens: int = 4096) -> LLMInterface:
    """Factory function to create LLM interface."""
    return LLMInterface(model_type, model_name, device, test_dir, api_base, api_key, max_new_tokens) 