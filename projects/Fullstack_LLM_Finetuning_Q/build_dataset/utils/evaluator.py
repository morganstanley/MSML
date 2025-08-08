#!/usr/bin/env python3
"""
Model Evaluator for Bootstrap Loop

This module provides a ModelEvaluator class for evaluating model performance.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Evaluates model performance on test tasks."""
    
    def __init__(self, model_name: str, model_type: str = "huggingface", device: str = "cuda"):
        """Initialize the model evaluator."""
        self.model_name = model_name
        self.model_type = model_type
        self.device = device
        
        if model_type == "huggingface":
            self._load_huggingface_model()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _load_huggingface_model(self):
        """Load a Hugging Face model."""
        logger.info(f"Loading model for evaluation: {self.model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            logger.info(f"Model loaded successfully for evaluation")
        except Exception as e:
            logger.error(f"Failed to load model for evaluation: {e}")
            raise
    
    def evaluate_all_tasks(self, test_jsonl_path: str, results_dir: str) -> Dict[str, Any]:
        """Evaluate model on all test tasks."""
        logger.info(f"Evaluating model on {test_jsonl_path}")
        
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)
        
        # Load test data
        test_data = []
        with open(test_jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    test_data.append(json.loads(line))
        
        logger.info(f"Loaded {len(test_data)} test examples")
        
        if len(test_data) == 0:
            logger.warning("No test data found")
            return {
                "overall_success_rate": 0.0,
                "total_examples": 0,
                "successful_examples": 0
            }
        
        # Evaluate each example
        successful_examples = 0
        total_examples = len(test_data)
        
        for i, example in enumerate(test_data):
            instruction = example.get("instruction", "")
            expected_response = example.get("response", "")
            
            # Generate response
            try:
                generated_response = self._generate_response(instruction)
                
                # Simple evaluation: check if response contains expected content
                # In a real implementation, this would be more sophisticated
                if self._evaluate_response(generated_response, expected_response):
                    successful_examples += 1
                
                logger.info(f"Example {i+1}/{total_examples}: {'✓' if successful_examples > 0 else '✗'}")
                
            except Exception as e:
                logger.error(f"Error evaluating example {i+1}: {e}")
        
        success_rate = successful_examples / total_examples if total_examples > 0 else 0.0
        
        results = {
            "overall_success_rate": success_rate,
            "total_examples": total_examples,
            "successful_examples": successful_examples
        }
        
        # Save results
        with open(results_path / "evaluation_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Evaluation complete: {successful_examples}/{total_examples} successful ({success_rate:.2%})")
        return results
    
    def _generate_response(self, instruction: str) -> str:
        """Generate a response for the given instruction."""
        prompt = f"Instruction: {instruction}\n\nResponse:"
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return ""
    
    def _evaluate_response(self, generated_response: str, expected_response: str) -> bool:
        """Evaluate if the generated response matches the expected response."""
        # Simple evaluation: check if key elements are present
        # In a real implementation, this would be more sophisticated
        
        # Remove whitespace and normalize
        generated_clean = generated_response.lower().replace(" ", "")
        expected_clean = expected_response.lower().replace(" ", "")
        
        # Check if expected content is present in generated response
        if expected_clean in generated_clean:
            return True
        
        # Check for partial matches (e.g., function names)
        expected_words = expected_clean.split()
        generated_words = generated_clean.split()
        
        common_words = set(expected_words) & set(generated_words)
        if len(common_words) >= len(expected_words) * 0.5:  # 50% overlap
            return True
        
        return False 