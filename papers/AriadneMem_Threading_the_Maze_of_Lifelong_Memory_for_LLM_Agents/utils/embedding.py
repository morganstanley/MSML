"""
Embedding utilities - Generate vector embeddings using SentenceTransformers
Supports Qwen3 Embedding models through SentenceTransformers interface
"""
from typing import List, Optional, Dict, Any
import numpy as np
import config
import os
import hashlib
import threading

# Global model cache to avoid reloading models in the same process
_model_cache: Dict[str, Any] = {}
_cache_lock = threading.Lock()


class EmbeddingModel:
    """
    Embedding model using SentenceTransformers (supports Qwen3 and other models)
    """
    def __init__(self, model_name: str = None, use_optimization: bool = True):
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.use_optimization = use_optimization
        
        # Query embedding cache (same query -> same embedding, no need to recompute)
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_max = 1000
        
        # Check global cache first to avoid reloading models
        cache_key = f"{self.model_name}_{use_optimization}"
        with _cache_lock:
            if cache_key in _model_cache:
                print(f"Reusing cached embedding model: {self.model_name}")
                cached = _model_cache[cache_key]
                self.model = cached['model']
                self.dimension = cached['dimension']
                self.model_type = cached['model_type']
                self.supports_query_prompt = cached.get('supports_query_prompt', False)
                return
        
        # Model not in cache, load it
        print(f"Loading embedding model: {self.model_name}")
        
        # Check if it's a Qwen3 model (through SentenceTransformers)
        # Support both "qwen3-0.6b" format and "Qwen/Qwen3-Embedding-0.6B" format
        model_lower = self.model_name.lower()
        if model_lower.startswith("qwen3") or "qwen3" in model_lower:
            self._init_qwen3_sentence_transformer()
        else:
            self._init_standard_sentence_transformer()
        
        # Cache the loaded model
        with _cache_lock:
            _model_cache[cache_key] = {
                'model': self.model,
                'dimension': self.dimension,
                'model_type': self.model_type,
                'supports_query_prompt': getattr(self, 'supports_query_prompt', False)
            }

    def _init_qwen3_sentence_transformer(self):
        """Initialize Qwen3 model using SentenceTransformers"""
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            
            # Map model names to actual model paths
            qwen3_models = {
                "qwen3-0.6b": "Qwen/Qwen3-Embedding-0.6B",
                "qwen3-4b": "Qwen/Qwen3-Embedding-4B", 
                "qwen3-8b": "Qwen/Qwen3-Embedding-8B"
            }
            
            model_path = qwen3_models.get(self.model_name, self.model_name)
            
            # [SPEED] Explicitly set device to GPU if available
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            print(f"Loading Qwen3 model on device: {device}")
            
            # Initialize with optimization settings
            if self.use_optimization and device == "cuda":
                try:
                    # Try to use flash_attention_2 for CUDA
                    self.model = SentenceTransformer(
                        model_path,
                        device=device,
                        model_kwargs={
                            "attn_implementation": "flash_attention_2", 
                            "torch_dtype": torch.float16  # [SPEED] Use FP16 for faster inference
                        },
                        tokenizer_kwargs={"padding_side": "left"},
                        trust_remote_code=True
                    )
                    print("Qwen3 loaded with flash_attention_2 + FP16 optimization")
                except Exception as e:
                    print(f"Flash attention failed ({e}), using standard loading...")
                    self.model = SentenceTransformer(model_path, device=device, trust_remote_code=True)
            else:
                self.model = SentenceTransformer(model_path, device=device, trust_remote_code=True)
            
            self.dimension = self.model.get_sentence_embedding_dimension()
            self.model_type = "qwen3_sentence_transformer"
            
            # Check if Qwen3 supports query prompts
            self.supports_query_prompt = hasattr(self.model, 'prompts') and 'query' in getattr(self.model, 'prompts', {})
            
            print(f"Qwen3 model loaded successfully with dimension: {self.dimension}")
            if self.supports_query_prompt:
                print("Query prompt support detected")
                
        except Exception as e:
            print(f"Failed to load Qwen3 model: {e}")
            print("Falling back to default SentenceTransformers model...")
            self._fallback_to_sentence_transformer()

    def _init_standard_sentence_transformer(self):
        """Initialize standard SentenceTransformer model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            self.model_type = "sentence_transformer"
            self.supports_query_prompt = False
            print(f"SentenceTransformer model loaded with dimension: {self.dimension}")
        except Exception as e:
            print(f"Failed to load SentenceTransformer model: {e}")
            raise

    def _fallback_to_sentence_transformer(self):
        """Fallback to default SentenceTransformer model"""
        fallback_model = "sentence-transformers/all-MiniLM-L6-v2"
        print(f"Using fallback model: {fallback_model}")
        self.model_name = fallback_model
        self._init_standard_sentence_transformer()

    def encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """
        Encode list of texts to vectors
        
        Args:
        - texts: List of texts to encode
        - is_query: Whether these are query texts (for Qwen3 prompt optimization)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Use query prompt for Qwen3 models when encoding queries
        if self.model_type == "qwen3_sentence_transformer" and self.supports_query_prompt and is_query:
            return self._encode_with_query_prompt(texts)
        else:
            return self._encode_standard(texts)

    def encode_single(self, text: str, is_query: bool = False) -> np.ndarray:
        """
        Encode single text with caching for queries.
        Same text always produces same embedding - safe to cache.
        """
        # Use cache for queries (most common case in retrieval)
        if is_query:
            cache_key = hashlib.md5(text.encode()).hexdigest()
            if cache_key in self._cache:
                return self._cache[cache_key]
            
            result = self.encode([text], is_query=True)[0]
            
            # Simple cache management
            if len(self._cache) >= self._cache_max:
                # Remove first 10% when full
                keys = list(self._cache.keys())[:self._cache_max // 10]
                for k in keys:
                    del self._cache[k]
            
            self._cache[cache_key] = result
            return result
        
        return self.encode([text], is_query=is_query)[0]
    
    def encode_query(self, queries: List[str]) -> np.ndarray:
        """
        Encode queries with optimal settings for Qwen3
        """
        return self.encode(queries, is_query=True)
    
    def encode_documents(self, documents: List[str]) -> np.ndarray:
        """
        Encode documents (no query prompt)
        """
        return self.encode(documents, is_query=False)
    
    def _encode_with_query_prompt(self, texts: List[str]) -> np.ndarray:
        """Encode texts using Qwen3 query prompt"""
        try:
            embeddings = self.model.encode(
                texts, 
                prompt_name="query",  # Use Qwen3's query prompt
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True  # Ensure numpy output (auto handles GPU→CPU transfer)
            )
            return embeddings
        except Exception as e:
            print(f"Query prompt encoding failed: {e}, falling back to standard encoding")
            return self._encode_standard(texts)
    
    def _encode_standard(self, texts: List[str]) -> np.ndarray:
        """Encode texts using standard method"""
        embeddings = self.model.encode(
            texts, 
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True  # Ensure numpy output (auto handles GPU→CPU transfer)
        )
        return embeddings
