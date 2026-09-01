"""
AriadneAnswerGenerator Example - Topology-Aware Reasoning

Paper Reference: Section 2.3 - Topology-Aware Reasoning (Eq. 11)
This is an example/older version of the answer generator.
See ariadne_answer_generator.py for the current implementation.
"""
from typing import List, Dict, Any, Optional
import json

# Imports from your project structure
from utils.llm_client import LLMClient
from models.memory_entry import MemoryEntry
import config

class AriadneAnswerGeneratorExample:
    """
    AriadneAnswerGenerator Example (for reference)
    
    This is an older/example version. See ariadne_answer_generator.py for current implementation.
    Uses topology-aware synthesis to guide LLM reasoning.
    """
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def generate_answer(self, query: str, graph_path) -> str:
        """
        [Main Interface]
        Generates the final answer based on the retrieved subgraph.
        
        Args:
            query: The user's question
            graph_path: GraphPath object from ariadne_graph_retriever
        """
        # 1. Edge Case: No info retrieved
        if not graph_path or not graph_path.nodes:
            return "No relevant information found"

        # 2. Build Structural Context (The Core Logic)
        context_str = self._build_topology_context(graph_path)

        # 3. Construct Graph-Native Prompt with JSON output
        system_prompt = (
            "You are an intelligent Memory Graph Decoder. "
            "Answer questions concisely based on the provided knowledge path. "
            "You must output valid JSON format."
        )

        user_prompt = f"""Answer the question based on the knowledge path.

[Query]
{query}

[Retrieved Knowledge Path]
{context_str}

CRITICAL RULES:
1. EXTRACT exact words/phrases from facts - do NOT paraphrase or describe
2. Trust LATER dated info if conflicts exist
3. Dates: "15 January 2023" or "in 2022" or "January 2023"
4. Yes/No questions: answer exactly "yes" or "no"
5. "What is X's identity/status?": Answer with a LABEL (e.g., "Transgender woman", "Single", "Engineer"), NOT a description
6. Lists (hobbies/activities/books/places): List ALL directly mentioned items, comma-separated. Use exact names, not descriptions
7. If not in knowledge path: "Not mentioned in the conversation"

Output: {{"reasoning": "...", "answer": "..."}}

Examples:
- "When did Alice meet Bob?" → {{"answer": "15 January 2023"}}
- "What is Bob's identity?" → {{"answer": "Transgender man"}} (NOT "a person who identifies as...")
- "What is Bob's relationship status?" → {{"answer": "single"}} or {{"answer": "married"}}
- "What hobbies does Bob have?" (facts mention: painting, swimming, reading) → {{"answer": "painting, swimming, reading"}}
- "What books has Alice read?" (facts: "Book A", "Book B") → {{"answer": "Book A, Book B"}} (use exact titles!)
- "Where has Alice camped?" (facts: beach, mountains, forest) → {{"answer": "beach, mountains, forest"}} (list ALL)

Return ONLY the JSON.
"""
        # 4. LLM Call with JSON format
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

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
                return result.get("answer", response.strip())

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

    def _build_topology_context(self, graph_path) -> str:
        """
        Converts the GraphPath object into a structured string representation.
        
        Args:
            graph_path: GraphPath object from ariadne_graph_retriever
            
        Example Output:
        
        [NODES]
        [1] 2023-01-01: Bob started Project X. (Keywords: Bob, Project X)
        [2] 2023-01-05: Project X was delayed. (Keywords: Project X, Delay)
        
        [RELATIONSHIPS]
        [1] -> [2] (Type: entity_link via 'Project X')
        """
        nodes = graph_path.nodes
        edges = graph_path.edges
        
        # 1. Map Objects to IDs for cleanliness
        # Use entry_id or index as reference ID
        node_map = {n.entry_id: i+1 for i, n in enumerate(nodes)}
        
        lines = []
        
        # --- Section A: Nodes (The Facts) ---
        lines.append("[NODES (Chronological Order)]")
        for n in nodes:
            nid = node_map.get(n.entry_id)
            date_str = n.timestamp if n.timestamp else "Unknown Date"
            # Cleaning the text slightly
            content = n.lossless_restatement.replace("\n", " ").strip()
            lines.append(f"[{nid}] Date: {date_str} | Fact: {content}")

        lines.append("")

        # --- Section B: Edges (The Logic) ---
        lines.append("[RELATIONSHIPS (Logic Flow)]")
        if not edges:
            lines.append("(No explicit edges found. Infer based on time.)")
        
        for edge in edges:
            # Edges from Stage 3 store MemoryEntry objects, we need to convert them back to IDs
            src_obj = edge.get('source')
            tgt_obj = edge.get('target')
            
            # Safety check: ensure source/target is MemoryEntry (or at least has entry_id)
            if hasattr(src_obj, 'entry_id') and hasattr(tgt_obj, 'entry_id'):
                src_id = node_map.get(src_obj.entry_id, "?")
                tgt_id = node_map.get(tgt_obj.entry_id, "?")
                
                rel_type = edge.get('type', 'related')
                info = edge.get('info', '')
                
                # Format output: [1] --(temporal_flow)--> [2]
                lines.append(f"[{src_id}] --({rel_type}: {info})--> [{tgt_id}]")
        
        lines.append("")
        
        # --- Section C: Multi-hop Reasoning Paths (NEW) ---
        if hasattr(graph_path, 'reasoning_paths') and graph_path.reasoning_paths:
            lines.append("[REASONING PATHS (Multi-hop Logic Chains)]")
            lines.append("These paths show how facts connect through multiple steps:")
            
            for i, path in enumerate(graph_path.reasoning_paths, 1):
                # Build path string: [1] → [3] → [5]
                path_ids = [str(node_map.get(n.entry_id, "?")) for n in path]
                path_str = " → ".join([f"[{pid}]" for pid in path_ids])
                
                # Add brief content summary
                path_summary = " → ".join([
                    f"{n.lossless_restatement[:40]}..." if len(n.lossless_restatement) > 40 else n.lossless_restatement
                    for n in path
                ])
                
                lines.append(f"\nPath {i}: {path_str}")
                lines.append(f"  Summary: {path_summary}")
        else:
            lines.append("[REASONING PATHS]")
            lines.append("(No multi-hop paths discovered)")
            
        return "\n".join(lines)