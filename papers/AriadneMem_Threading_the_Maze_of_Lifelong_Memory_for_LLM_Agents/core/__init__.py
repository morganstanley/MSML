"""
AriadneMem: Threading the Maze of Lifelong Memory for LLM Agents

Paper Reference: AriadneMem (Zhu et al., 2025)
A structured memory system with:
- Phase I: Asynchronous Memory Construction (entropy-aware gating + conflict-aware coarsening)
- Phase II: Real-Time Structural Reasoning (bridge discovery + topology-aware synthesis)
"""
from .ariadne_memory_builder import AriadneMemoryBuilder
from .ariadne_graph_retriever import AriadneGraphRetriever
from .ariadne_answer_generator import AriadneAnswerGenerator

__all__ = ['AriadneMemoryBuilder', 'AriadneGraphRetriever', 'AriadneAnswerGenerator']
