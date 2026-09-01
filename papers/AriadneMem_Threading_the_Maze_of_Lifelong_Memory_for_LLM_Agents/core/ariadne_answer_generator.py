"""
AriadneAnswerGenerator - Topology-Aware Synthesis

Paper Reference: Section 2.3 - Topology-Aware Reasoning (Eq. 11)
Key Features:
1. Graph Context Serialization: Converts G_q into C_graph for LLM
2. Chronological Reasoning: Time-based fact ordering
3. Bridge Utilization: Highlights inferred connections from Steiner approximation
4. Semantic Normalization: Post-processes answers for evaluation metrics
"""
from typing import List, Dict, Any, Optional
import json

# Imports from your project structure
from utils.llm_client import LLMClient
from models.memory_entry import MemoryEntry
from core.semantic_normalizer import SemanticNormalizer
import config

# Token counting for context (using tiktoken if available, fallback to simple estimation)
try:
    import tiktoken
    _encoding = tiktoken.encoding_for_model("gpt-4")
    def count_tokens(text: str) -> int:
        return len(_encoding.encode(text))
except ImportError:
    def count_tokens(text: str) -> int:
        # Fallback: estimate ~4 chars per token
        return len(text) // 4

class AriadneAnswerGenerator:
    """
    AriadneAnswerGenerator - Topology-Aware Synthesis
    
    Paper Reference: Section 2.3 - Topology-Aware Reasoning
    Generates answer via: a = LLM(q, C_graph) where C_graph = Serialize(G_q)
    Uses structural context (nodes + reasoning paths) instead of flat fact list
    """
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        
        # Semantic normalizer for answer post-processing
        self.normalizer = SemanticNormalizer()
        
        # Token Cost tracking (only context tokens, for fair comparison with SimpleMem)
        self.total_context_tokens = 0
        self.context_token_counts = []  # Per-query context tokens
        
        # Context compression settings
        # DISABLED: LLM compression loses relative time expressions (e.g., "the week before [date]")
        # Node budget is handled in AriadneGraphRetriever
        self.enable_compression = False

    def generate_answer(self, query: str, graph_path) -> str:
        """
        Topology-Aware Synthesis - Eq. 11: a = LLM(q, C_graph)
        
        Paper Reference: Section 2.3 - Topology-Aware Reasoning
        Single LLM call with structured context (vs iterative planning in baselines)
        """
        # Edge case: Empty graph
        if not graph_path or not graph_path.nodes:
            return "No relevant information found"

        # Build C_graph = Serialize(G_q) - topology-aware context
        context_str = self._build_topology_context(graph_path, query)
        
        # Track context tokens (Token Cost metric for paper comparison)
        context_tokens = count_tokens(context_str)
        self.total_context_tokens += context_tokens
        self.context_token_counts.append(context_tokens)

        # Get prompt templates from config (customizable)
        system_prompt = getattr(config, 'ANSWER_SYSTEM_PROMPT', 
            "You are a QA system with graph-based memory. Analyze the graph structure, reason through the facts, then answer. Output JSON only.")

        # Get target entity and graph metadata
        target_entity = getattr(graph_path, 'target_entity', None)
        num_nodes = len(graph_path.nodes) if graph_path and graph_path.nodes else 0
        
        entity_hint = f"[Focus Entity: {target_entity}]\n" if target_entity else ""
        graph_hint = f"[Graph: {num_nodes} nodes retrieved]\n"

        # Build user prompt from template (customizable in config.py)
        user_prompt_template = getattr(config, 'ANSWER_USER_PROMPT_TEMPLATE', None)
        if user_prompt_template:
            user_prompt = user_prompt_template.format(
                query=query,
                entity_hint=entity_hint,
                graph_hint=graph_hint,
                context_str=context_str
            )
        else:
            # Fallback to default template if not configured
            user_prompt = f"""Q: {query}
{entity_hint}{graph_hint}
{context_str}

**STEP 1: GRAPH REASONING (1-2 sentences)**

Analyze the graph facts and reasoning paths. Reference facts by their labels [F1], [F2], etc.
- ATTRIBUTE: What specific fact directly answers this?
- RELATIONSHIP: Trace entity connections (use [Reasoning Paths] if provided)
- COUNT: List and count each occurrence
- INFERENCE: What evidence supports/contradicts?
- TEMPORAL: Note exact timestamp format from facts

**STEP 2: ANSWER RULES (CRITICAL!)**

**ANSWER LENGTH - MATCH QUESTION TYPE:**
- "What is X's identity?" → ONLY the core identity (2-3 words max)
  - Example: "Transgender woman" NOT "transgender single parent artist advocate"
- "What does X like/do?" → 2-4 KEY items only, most relevant to question
  - Example: "dinosaurs, nature" NOT "beach, camping, hiking, nature, swimming"
- "What books/movies?" → Use EXACT quoted titles from facts
  - Example: "Charlotte's Web" NOT "a book about friendship"
- "When did X?" → Copy EXACT time expression from facts
  - If facts say "the week before 9 June" → answer "The week before 9 June 2023"
  - If facts say "2022" or "last year" → answer "2022"
  - NEVER convert "the week before X" to date range like "2-8 June"

**INFERENCE ANSWERS (Would/Could/Likely questions):**
- If evidence is CLEAR: "Yes" or "No" (no explanation needed)
- If evidence needs INFERENCE: "Likely yes; [1 reason from facts]" or "Likely no; [1 reason]"
- Match the question's asking style

**FORMAT RULES:**
- Date: "DD Month YYYY" (NOT ISO format YYYY-MM-DD)
- Lists: Comma-separated, 2-4 items max
- Single answer: 1-5 words, exact from facts
- Quotes: Keep original quotes around titles

**BE CONCISE! Answer only what's asked, nothing extra.**

Output JSON:
{{
  "reasoning": "1-2 sentences: cite [F1], [F2] etc. to show which facts answered this",
  "answer": "concise exact answer"
}}"""
        # Single LLM call for answer synthesis (Eq. 11)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Debug: Print context sent to LLM (disable via config.DEBUG_LLM_CONTEXT = False)
        if getattr(config, "DEBUG_LLM_CONTEXT", True):
            print("\n" + "=" * 60 + " [DEBUG] GRAPH STRUCTURE " + "=" * 60)
            print(f"[Nodes]: {num_nodes} nodes retrieved")
            num_edges = len(graph_path.edges) if hasattr(graph_path, 'edges') and graph_path.edges else 0
            num_paths = len(graph_path.reasoning_paths) if hasattr(graph_path, 'reasoning_paths') and graph_path.reasoning_paths else 0
            print(f"[Edges]: {num_edges} edges (connections between nodes)")
            print(f"[Reasoning Paths]: {num_paths} multi-hop chains discovered")
            if num_edges > 0:
                direct_edges = len([e for e in graph_path.edges if e.get('info') == 'direct'])
                bridge_edges = len([e for e in graph_path.edges if e.get('info') == 'inferred'])
                print(f"  - Direct connections: {direct_edges}")
                print(f"  - Bridge (inferred): {bridge_edges}")
            print("\n" + "=" * 60 + " [DEBUG] FINAL CONTEXT SENT TO LLM " + "=" * 60)
            print("[System prompt]\n" + system_prompt)
            print("\n[User prompt (question + graph facts)]\n" + user_prompt)
            print("=" * 60 + "\n")

        # Retry up to 3 times
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Use JSON format if configured
                response_format = None
                if hasattr(config, 'USE_JSON_FORMAT') and config.USE_JSON_FORMAT:
                    response_format = {"type": "json_object"}

                response = self.llm_client.chat_completion(
                    messages, 
                    temperature=0.1,  # Low temp for factual accuracy
                    response_format=response_format
                )

                # Parse JSON response
                result = self.llm_client.extract_json(response)
                
                # Extract reasoning (optional, for debugging/analysis)
                reasoning = result.get("reasoning", "")
                if reasoning:
                    print(f"  [Reasoning] {reasoning}")
                
                # Extract answer
                answer = result.get("answer", response.strip())
                
                # CRITICAL: Convert list to comma-separated string
                if isinstance(answer, list):
                    answer = ", ".join(str(item) for item in answer)
                elif not isinstance(answer, str):
                    answer = str(answer)
                
                # Apply semantic normalization (for better F1 matching)
                normalized_answer = self.normalizer.normalize(answer.strip())
                
                return normalized_answer

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Answer generation attempt {attempt + 1}/{max_retries} failed: {e}. Retrying...")
                else:
                    print(f"Warning: Failed to parse JSON response after {max_retries} attempts: {e}")
                    # Fallback to raw response
                    if 'response' in locals():
                        return response.strip()
                    else:
                        return "Failed to generate answer"

    def _build_topology_context(self, graph_path, query: str = "") -> str:
        """
        Serialize G_q into C_graph - topology-aware context for LLM
        
        Paper Reference: Section 2.3 - Topology-Aware Contextualization
        Format: Labeled facts [F1], [F2], ... + reasoning paths + bridge connections
        Provides path-oriented grounding (not flat fact list)
        """
        lines = []
        
        # Part 1: Labeled facts (nodes) - sorted by time
        # Build node_id -> label mapping for reasoning paths
        node_label_map = {}
        fact_idx = 0
        
        lines.append("[Facts from Graph]")
        for n in graph_path.nodes:
            content = (n.lossless_restatement or "").replace("\n", " ").strip()
            if not content:
                continue
            
            fact_idx += 1
            label = f"F{fact_idx}"
            node_label_map[n.entry_id] = label
            
            if n.timestamp:
                lines.append(f"[{label}] {n.timestamp}: {content}")
            else:
                lines.append(f"[{label}] {content}")
        
        # Part 2: Reasoning Paths (reference facts by label)
        # Filter valid paths first, then number sequentially to avoid gaps.
        # Deduplicate by label sequence to prevent displaying identical chains.
        if hasattr(graph_path, 'reasoning_paths') and graph_path.reasoning_paths:
            valid_paths = []
            seen_label_seqs = set()
            for path in graph_path.reasoning_paths:
                labels = []
                for node in path:
                    label = node_label_map.get(node.entry_id)
                    if label:
                        labels.append(label)
                if len(labels) >= 2:
                    label_seq = tuple(labels)
                    if label_seq not in seen_label_seqs:
                        seen_label_seqs.add(label_seq)
                        valid_paths.append(labels)
            if valid_paths:
                lines.append("")
                lines.append("[Reasoning Paths]")
                for i, labels in enumerate(valid_paths, 1):
                    lines.append(f"  {i}. {' → '.join(labels)}")
        
        # Part 3: Edges summary (if available)
        if hasattr(graph_path, 'edges') and graph_path.edges:
            bridge_edges = [e for e in graph_path.edges if e.get('info') == 'inferred']
            if bridge_edges:
                lines.append("")
                lines.append(f"[Bridge Connections: {len(bridge_edges)} inferred links found]")
        
        return "\n".join(lines)
    
    def _compress_context(self, raw_context: str, query: str, raw_tokens: int) -> str:
        """
        Use LLM to compress context while preserving answer-relevant info.
        Target: 200-300 tokens.
        """
        compress_prompt = f"""Extract facts needed to answer: "{query}"

{raw_context}

Output ONLY relevant facts as bullet points. Keep exact dates/names/numbers."""

        try:
            compressed = self.llm_client.chat_completion(
                [{"role": "user", "content": compress_prompt}],
                temperature=0.0,
                max_tokens=400
            )
            return compressed.strip()
        except Exception as e:
            print(f"Compression failed: {e}, using raw context")
            return raw_context

    # ============================================================================
    # Token Cost Statistics (Paper Table 1 metric)
    # ============================================================================
    
    def get_context_token_stats(self) -> Dict[str, Any]:
        """
        Get context token statistics for Token Cost metric (Table 1 in paper)
        
        Paper Reference: Section 4.1 - Token Cost metric
        Measures retrieved memory context only (fair comparison with baselines)
        """
        query_count = len(self.context_token_counts)
        return {
            "total_context_tokens": self.total_context_tokens,
            "avg_context_tokens_per_query": self.total_context_tokens / max(1, query_count),
            "query_count": query_count,
            "min_context_tokens": min(self.context_token_counts) if self.context_token_counts else 0,
            "max_context_tokens": max(self.context_token_counts) if self.context_token_counts else 0,
        }
    
    def reset_context_token_stats(self):
        """Reset context token counters."""
        self.total_context_tokens = 0
        self.context_token_counts = []