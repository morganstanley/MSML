#!/usr/bin/env python3
"""
Configuration management for Q language model training pipeline.
Handles paths, model settings, and API configurations through environment variables.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

@dataclass
class DatasetConfig:
    """Configuration for dataset building and processing"""
    initial_dataset_dir: Path = Path("initial_dataset")
    validated_dataset_dir: Path = Path("validated_dataset")
    final_dataset_dir: Path = Path("final_dataset")
    q_interpreter_path: str = os.getenv("Q_INTERPRETER_PATH", "q")  # Default to system q

@dataclass
class ModelConfig:
    """Configuration for model training and inference"""
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    max_seq_length: int = 2048
    learning_rate: float = 2e-5
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    warmup_steps: int = 100
    max_steps: int = 1000
    save_steps: int = 50
    
    # LoRA settings
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1

@dataclass
class APIConfig:
    """Configuration for external API access"""
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    anthropic_api_key: Optional[str] = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    xai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("XAI_API_KEY"))
    gemini_api_key: Optional[str] = field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))

@dataclass
class Config:
    """Main configuration class"""
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    api: APIConfig = field(default_factory=APIConfig)
    
    # Output directories
    output_dir: Path = Path("outputs")
    log_dir: Path = Path("logs")
    
    def save(self, path: str) -> None:
        """Save configuration to YAML file"""
        # Convert Path objects to strings for YAML serialization
        config_dict = {
            "dataset": {
                "initial_dataset_dir": str(self.dataset.initial_dataset_dir),
                "validated_dataset_dir": str(self.dataset.validated_dataset_dir),
                "final_dataset_dir": str(self.dataset.final_dataset_dir),
                "q_interpreter_path": self.dataset.q_interpreter_path
            },
            "model": {k: v for k, v in vars(self.model).items()},
            "output_dir": str(self.output_dir),
            "log_dir": str(self.log_dir)
        }
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    @classmethod
    def load(cls, path: str) -> 'Config':
        """Load configuration from YAML file"""
        with open(path) as f:
            config_dict = yaml.safe_load(f)
        
        # Convert string paths back to Path objects
        dataset_config = DatasetConfig(
            initial_dataset_dir=Path(config_dict["dataset"]["initial_dataset_dir"]),
            validated_dataset_dir=Path(config_dict["dataset"]["validated_dataset_dir"]),
            final_dataset_dir=Path(config_dict["dataset"]["final_dataset_dir"]),
            q_interpreter_path=config_dict["dataset"]["q_interpreter_path"]
        )
        
        model_config = ModelConfig(**config_dict["model"])
        
        return cls(
            dataset=dataset_config,
            model=model_config,
            output_dir=Path(config_dict["output_dir"]),
            log_dir=Path(config_dict["log_dir"])
        )

def get_config(config_path: Optional[str] = None) -> Config:
    """Get configuration, either from file or defaults"""
    if config_path and os.path.exists(config_path):
        return Config.load(config_path)
    return Config() 