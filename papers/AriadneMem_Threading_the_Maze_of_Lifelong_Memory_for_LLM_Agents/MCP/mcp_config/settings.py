"""
Settings configuration for AriadneMem MCP Server

Reads from environment variables with sensible defaults.
For LLM/embedding configuration, imports from AriadneMem's config.py.
"""

import os
import sys
from dataclasses import dataclass, field
from functools import lru_cache

# Add parent directory to path so we can import AriadneMem's config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@dataclass
class Settings:
    """Application settings"""

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # Security (for production, override via environment variables)
    api_token: str = field(default_factory=lambda: os.getenv(
        "ARIADNEMEM_API_TOKEN",
        "ariadnemem-dev-token"  # Default for local development
    ))

    # Database Paths
    data_dir: str = field(default_factory=lambda: os.getenv(
        "DATA_DIR",
        "./data"
    ))

    # AriadneMem Configuration (imported from parent config.py)
    openai_api_key: str = ""
    openai_base_url: str = ""
    llm_model: str = ""
    embedding_model: str = ""
    embedding_dimension: int = 384

    # Memory Building
    window_size: int = 40
    overlap_size: int = 2
    redundancy_threshold: float = 0.6
    coarsening_threshold: float = 0.6

    # Retrieval
    semantic_top_k: int = 25
    keyword_top_k: int = 5
    max_reasoning_path_depth: int = 3
    max_reasoning_paths: int = 10

    # LLM
    enable_thinking: bool = False
    use_streaming: bool = True
    use_json_format: bool = True
    debug_llm_context: bool = False  # Disable in MCP mode for cleaner output

    def __post_init__(self):
        """Load settings from AriadneMem config.py"""
        os.makedirs(self.data_dir, exist_ok=True)
        self._load_ariadne_config()

    def _load_ariadne_config(self):
        """Import settings from AriadneMem's config.py"""
        try:
            import config as ariadne_config
            self.openai_api_key = getattr(ariadne_config, 'OPENAI_API_KEY', self.openai_api_key)
            self.openai_base_url = getattr(ariadne_config, 'OPENAI_BASE_URL', self.openai_base_url) or ""
            self.llm_model = getattr(ariadne_config, 'LLM_MODEL', self.llm_model)
            self.embedding_model = getattr(ariadne_config, 'EMBEDDING_MODEL', self.embedding_model)
            self.embedding_dimension = getattr(ariadne_config, 'EMBEDDING_DIMENSION', self.embedding_dimension)
            self.window_size = getattr(ariadne_config, 'WINDOW_SIZE', self.window_size)
            self.overlap_size = getattr(ariadne_config, 'OVERLAP_SIZE', self.overlap_size)
            self.redundancy_threshold = getattr(ariadne_config, 'REDUNDANCY_THRESHOLD', self.redundancy_threshold)
            self.coarsening_threshold = getattr(ariadne_config, 'COARSENING_THRESHOLD', self.coarsening_threshold)
            self.semantic_top_k = getattr(ariadne_config, 'SEMANTIC_TOP_K', self.semantic_top_k)
            self.keyword_top_k = getattr(ariadne_config, 'KEYWORD_TOP_K', self.keyword_top_k)
            self.max_reasoning_path_depth = getattr(ariadne_config, 'MAX_REASONING_PATH_DEPTH', self.max_reasoning_path_depth)
            self.max_reasoning_paths = getattr(ariadne_config, 'MAX_REASONING_PATHS', self.max_reasoning_paths)
            self.enable_thinking = getattr(ariadne_config, 'ENABLE_THINKING', self.enable_thinking)
            self.use_streaming = getattr(ariadne_config, 'USE_STREAMING', self.use_streaming)
            self.use_json_format = getattr(ariadne_config, 'USE_JSON_FORMAT', self.use_json_format)
            print(f"Loaded AriadneMem config: model={self.llm_model}, embedding={self.embedding_model}")
        except ImportError:
            print("Warning: AriadneMem config.py not found, using defaults")


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
