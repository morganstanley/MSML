"""
MCP Protocol Handler - JSON-RPC 2.0 over Streamable HTTP

Implements the Model Context Protocol for AriadneMem.

AriadneMem-specific tools:
- memory_add / memory_add_batch: Phase I pipeline (entropy gating + coarsening)
- memory_query: Phase II pipeline (graph retrieval + topology-aware synthesis)
- memory_retrieve: Raw hybrid retrieval (semantic + lexical)
- memory_graph_inspect: Inspect graph structure (nodes, edges, reasoning paths)
- memory_stats: Memory statistics
- memory_clear: Clear all memories
"""

import json
import os
import sys
from typing import Any, Optional
from dataclasses import dataclass

# Add AriadneMem root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from main import AriadneMemSystem
from models.memory_entry import Dialogue


@dataclass
class JsonRpcRequest:
    jsonrpc: str
    method: str
    id: Optional[int | str]
    params: Optional[dict] = None


@dataclass
class JsonRpcResponse:
    jsonrpc: str = "2.0"
    id: Optional[int | str] = None
    result: Optional[Any] = None
    error: Optional[dict] = None

    def to_dict(self):
        d = {"jsonrpc": self.jsonrpc, "id": self.id}
        if self.error:
            d["error"] = self.error
        else:
            d["result"] = self.result
        return d


# MCP Protocol Constants
MCP_VERSION = "2025-03-26"  # Streamable HTTP transport
SERVER_NAME = "ariadnemem"
SERVER_VERSION = "1.0.0"


class MCPHandler:
    """
    Handles MCP protocol messages for AriadneMem.

    Wraps the AriadneMemSystem and exposes its capabilities as MCP tools.
    """

    def __init__(self, system: AriadneMemSystem):
        self.system = system
        self.initialized = False
        self._dialogue_count = 0

    async def handle_message(self, message: str) -> str:
        """Handle a JSON-RPC message and return response"""
        try:
            data = json.loads(message)
            request = JsonRpcRequest(
                jsonrpc=data.get("jsonrpc", "2.0"),
                method=data.get("method", ""),
                id=data.get("id"),
                params=data.get("params", {}),
            )
            response = await self._dispatch(request)
            return json.dumps(response.to_dict(), ensure_ascii=False)
        except json.JSONDecodeError as e:
            return json.dumps(JsonRpcResponse(
                error={"code": -32700, "message": f"Parse error: {e}"}
            ).to_dict())
        except Exception as e:
            return json.dumps(JsonRpcResponse(
                error={"code": -32603, "message": f"Internal error: {e}"}
            ).to_dict())

    async def _dispatch(self, request: JsonRpcRequest) -> JsonRpcResponse:
        """Dispatch request to appropriate handler"""
        method = request.method
        params = request.params or {}

        handlers = {
            "initialize": self._handle_initialize,
            "initialized": self._handle_initialized,
            "ping": self._handle_ping,
            "tools/list": self._handle_tools_list,
            "tools/call": self._handle_tools_call,
            "resources/list": self._handle_resources_list,
            "resources/read": self._handle_resources_read,
        }

        handler = handlers.get(method)
        if not handler:
            return JsonRpcResponse(
                id=request.id,
                error={"code": -32601, "message": f"Method not found: {method}"}
            )

        try:
            result = await handler(params)
            return JsonRpcResponse(id=request.id, result=result)
        except Exception as e:
            return JsonRpcResponse(
                id=request.id,
                error={"code": -32603, "message": str(e)}
            )

    async def _handle_initialize(self, params: dict) -> dict:
        """Handle initialize request"""
        self.initialized = True
        return {
            "protocolVersion": MCP_VERSION,
            "capabilities": {
                "tools": {},
                "resources": {},
            },
            "serverInfo": {
                "name": SERVER_NAME,
                "version": SERVER_VERSION,
                "description": "AriadneMem - Threading the Maze of Lifelong Memory for LLM Agents. "
                              "A graph-based memory system with entropy-aware gating, "
                              "conflict-aware coarsening, Steiner tree bridge discovery, "
                              "and topology-aware synthesis via single LLM call.",
            },
            "instructions": """AriadneMem is a graph-based long-term memory system for LLM agents.

1. STORE conversations: Use memory_add or memory_add_batch to save dialogues.
   Phase I pipeline automatically:
   - Filters low-information messages (entropy-aware gating)
   - Extracts atomic facts via LLM (F_theta transformation)
   - Merges duplicates while preserving state updates (conflict-aware coarsening)

2. RECALL information: Use memory_query to ask questions about past conversations.
   Phase II pipeline automatically:
   - Checks fast paths (O(1) cache for count/list/relation queries)
   - Performs hybrid retrieval (semantic + lexical search)
   - Discovers bridge nodes via Steiner tree approximation
   - Mines multi-hop reasoning paths via DFS
   - Generates answer via single topology-aware LLM call

3. INSPECT graph: Use memory_graph_inspect to see the graph structure,
   reasoning paths, and bridge connections for any query.

4. BROWSE memories: Use memory_retrieve for raw fact retrieval.

5. MANAGE: Use memory_stats to check status, memory_clear to reset.

Key advantage: AriadneMem uses graph algorithms (not multiple LLM calls)
for multi-hop reasoning, resulting in lower cost and latency.""",
        }

    async def _handle_initialized(self, params: dict) -> dict:
        return {}

    async def _handle_ping(self, params: dict) -> dict:
        return {}

    async def _handle_tools_list(self, params: dict) -> dict:
        """Handle tools/list request - expose AriadneMem capabilities"""
        return {
            "tools": [
                {
                    "name": "memory_add",
                    "description": """Add a dialogue to AriadneMem long-term memory.

AriadneMem Phase I pipeline processes the dialogue:
1. Entropy-Aware Gating (Eq.3): Filters redundant/low-info messages
2. Atomic Extraction F_theta (Eq.4): LLM extracts structured facts
3. Conflict-Aware Coarsening (Eq.5-6): Merges duplicates, preserves state updates

The dialogue is processed and stored immediately.

Example: memory_add(speaker="Alice", content="Let's change the meeting to 3pm")
→ Stored as atomic fact with temporal edges linking to previous "2pm" entry""",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "speaker": {
                                "type": "string",
                                "description": "Name of the speaker",
                            },
                            "content": {
                                "type": "string",
                                "description": "Content of the dialogue",
                            },
                            "timestamp": {
                                "type": "string",
                                "description": "ISO 8601 timestamp. Defaults to current time.",
                            },
                        },
                        "required": ["speaker", "content"],
                    },
                },
                {
                    "name": "memory_add_batch",
                    "description": """Add multiple dialogues to AriadneMem at once.

Efficient for importing conversation history. All dialogues go through
the full Phase I pipeline: entropy gating → atomic extraction → coarsening.

After adding, call with finalize=true to complete the memory graph construction.""",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "dialogues": {
                                "type": "array",
                                "description": "List of dialogues to add",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "speaker": {"type": "string", "description": "Speaker name"},
                                        "content": {"type": "string", "description": "Dialogue content"},
                                        "timestamp": {"type": "string", "description": "ISO 8601 timestamp"},
                                    },
                                    "required": ["speaker", "content"],
                                },
                            },
                            "finalize": {
                                "type": "boolean",
                                "description": "Whether to finalize memory graph after adding. Default: true",
                            },
                        },
                        "required": ["dialogues"],
                    },
                },
                {
                    "name": "memory_query",
                    "description": """Query AriadneMem and get a graph-reasoned answer.

This is the primary retrieval tool. AriadneMem Phase II pipeline:
1. Fast Paths: O(1) cache lookup for count/list/relation queries
2. Hybrid Retrieval (Eq.7): Semantic + lexical search for terminal nodes
3. Base Graph Construction (Eq.8): Entity/temporal edge inference
4. Bridge Discovery (Eq.9): Steiner tree approximation for missing links
5. Multi-Hop Path Mining (Eq.10): DFS reasoning chain discovery
6. Topology-Aware Synthesis (Eq.11): Single LLM call with graph context

Returns: answer, reasoning, graph statistics (nodes, edges, paths).
Uses only 1 LLM call for answer generation (vs 4-6 in planning-based methods).""",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "Natural language question about stored memories",
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Max retrieval results. Default: 5",
                            },
                        },
                        "required": ["question"],
                    },
                },
                {
                    "name": "memory_retrieve",
                    "description": """Retrieve relevant memory entries without generating an answer.

Returns raw memory entries from hybrid retrieval (semantic + lexical).
Use this when you need direct access to stored facts.

Each entry contains: content, timestamp, persons, location, topic, entities.""",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query for hybrid retrieval",
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Maximum entries to return. Default: 10",
                            },
                        },
                        "required": ["query"],
                    },
                },
                {
                    "name": "memory_graph_inspect",
                    "description": """Inspect the graph structure for a query.

Returns detailed graph information including:
- Retrieved nodes (facts) with labels [F1], [F2], etc.
- Edges (direct entity/temporal links and inferred bridge connections)
- Multi-hop reasoning paths discovered by DFS
- Bridge nodes found via Steiner tree approximation

Use this for debugging, explainability, or understanding how AriadneMem
reasons about a query.""",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Query to inspect graph structure for",
                            },
                        },
                        "required": ["query"],
                    },
                },
                {
                    "name": "memory_stats",
                    "description": """Get AriadneMem memory statistics.

Returns: total memory entries, graph configuration parameters,
model info, and system status.""",
                    "inputSchema": {
                        "type": "object",
                        "properties": {},
                    },
                },
                {
                    "name": "memory_clear",
                    "description": """Clear ALL memories. This action CANNOT be undone.

Removes all stored memory entries and reinitializes the database.""",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "confirm": {
                                "type": "boolean",
                                "description": "Must be true to confirm deletion",
                            },
                        },
                        "required": ["confirm"],
                    },
                },
            ]
        }

    async def _handle_tools_call(self, params: dict) -> dict:
        """Handle tools/call request"""
        name = params.get("name", "")
        arguments = params.get("arguments", {})

        tool_handlers = {
            "memory_add": self._tool_memory_add,
            "memory_add_batch": self._tool_memory_add_batch,
            "memory_query": self._tool_memory_query,
            "memory_retrieve": self._tool_memory_retrieve,
            "memory_graph_inspect": self._tool_memory_graph_inspect,
            "memory_stats": self._tool_memory_stats,
            "memory_clear": self._tool_memory_clear,
        }

        handler = tool_handlers.get(name)
        if not handler:
            raise ValueError(f"Unknown tool: {name}")

        result = await handler(arguments)
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(result, ensure_ascii=False, indent=2),
                }
            ]
        }

    async def _tool_memory_add(self, args: dict) -> dict:
        """Add single dialogue through Phase I pipeline"""
        speaker = args["speaker"]
        content = args["content"]
        timestamp = args.get("timestamp")

        self.system.add_dialogue(speaker, content, timestamp)
        self._dialogue_count += 1
        self.system.finalize()

        return {
            "success": True,
            "message": f"Dialogue from {speaker} processed through Phase I pipeline",
            "details": {
                "speaker": speaker,
                "content_preview": content[:100] + ("..." if len(content) > 100 else ""),
                "total_dialogues": self._dialogue_count,
                "pipeline": "entropy_gating → atomic_extraction → conflict_coarsening",
            },
        }

    async def _tool_memory_add_batch(self, args: dict) -> dict:
        """Add batch of dialogues through Phase I pipeline"""
        dialogues_data = args["dialogues"]
        finalize = args.get("finalize", True)

        dialogues = []
        for i, d in enumerate(dialogues_data):
            dialogues.append(Dialogue(
                dialogue_id=self._dialogue_count + i + 1,
                speaker=d["speaker"],
                content=d["content"],
                timestamp=d.get("timestamp"),
            ))

        self.system.add_dialogues(dialogues)
        self._dialogue_count += len(dialogues)

        if finalize:
            self.system.finalize()

        return {
            "success": True,
            "message": f"Added {len(dialogues)} dialogues through Phase I pipeline",
            "details": {
                "dialogues_added": len(dialogues),
                "total_dialogues": self._dialogue_count,
                "finalized": finalize,
                "pipeline": "entropy_gating → atomic_extraction → conflict_coarsening",
            },
        }

    async def _tool_memory_query(self, args: dict) -> dict:
        """Query with full Phase II pipeline"""
        question = args["question"]
        top_k = args.get("top_k", 5)

        # Full pipeline: retrieve graph + generate answer
        graph_path = self.system.graph_retriever.retrieve(question)
        answer = self.system.answer_generator.generate_answer(question, graph_path)

        # Collect graph statistics
        num_nodes = len(graph_path.nodes) if graph_path and graph_path.nodes else 0
        num_edges = len(graph_path.edges) if hasattr(graph_path, 'edges') and graph_path.edges else 0
        num_paths = len(graph_path.reasoning_paths) if hasattr(graph_path, 'reasoning_paths') and graph_path.reasoning_paths else 0

        bridge_count = 0
        if hasattr(graph_path, 'edges') and graph_path.edges:
            bridge_count = len([e for e in graph_path.edges if e.get('info') == 'inferred'])

        return {
            "question": question,
            "answer": answer,
            "graph_stats": {
                "nodes_retrieved": num_nodes,
                "edges": num_edges,
                "reasoning_paths": num_paths,
                "bridge_nodes_discovered": bridge_count,
                "llm_calls": 1,  # AriadneMem always uses single LLM call
            },
        }

    async def _tool_memory_retrieve(self, args: dict) -> dict:
        """Raw retrieval without answer generation"""
        query = args["query"]
        top_k = args.get("top_k", 10)

        graph_path = self.system.graph_retriever.retrieve(query)

        results = []
        if graph_path and graph_path.nodes:
            for node in graph_path.nodes[:top_k]:
                results.append({
                    "content": node.lossless_restatement,
                    "timestamp": node.timestamp,
                    "persons": node.persons,
                    "location": node.location,
                    "topic": node.topic,
                    "entities": node.entities,
                })

        return {
            "query": query,
            "results": results,
            "total": len(results),
        }

    async def _tool_memory_graph_inspect(self, args: dict) -> dict:
        """Inspect graph structure for explainability"""
        query = args["query"]

        graph_path = self.system.graph_retriever.retrieve(query)

        if not graph_path or not graph_path.nodes:
            return {
                "query": query,
                "message": "No relevant memories found",
                "graph": {"nodes": [], "edges": [], "reasoning_paths": []},
            }

        # Format nodes
        nodes = []
        for i, node in enumerate(graph_path.nodes, 1):
            nodes.append({
                "label": f"F{i}",
                "content": node.lossless_restatement,
                "timestamp": node.timestamp,
                "persons": node.persons,
                "entry_id": node.entry_id,
            })

        # Format edges
        edges = []
        if hasattr(graph_path, 'edges') and graph_path.edges:
            for edge in graph_path.edges:
                edges.append({
                    "source": edge.get("source", ""),
                    "target": edge.get("target", ""),
                    "type": edge.get("info", "direct"),
                    "reason": edge.get("reason", ""),
                })

        # Format reasoning paths
        paths = []
        node_id_to_label = {node.entry_id: f"F{i}" for i, node in enumerate(graph_path.nodes, 1)}
        if hasattr(graph_path, 'reasoning_paths') and graph_path.reasoning_paths:
            for i, path in enumerate(graph_path.reasoning_paths, 1):
                path_labels = [node_id_to_label.get(n.entry_id, "?") for n in path]
                path_summaries = [n.lossless_restatement[:60] for n in path]
                paths.append({
                    "path_id": i,
                    "hops": len(path),
                    "chain": " → ".join(path_labels),
                    "summary": " → ".join(path_summaries),
                })

        # Count bridge edges
        direct_edges = len([e for e in (graph_path.edges or []) if e.get('info') == 'direct'])
        bridge_edges = len([e for e in (graph_path.edges or []) if e.get('info') == 'inferred'])

        return {
            "query": query,
            "graph": {
                "nodes": nodes,
                "edges": edges,
                "reasoning_paths": paths,
            },
            "summary": {
                "total_nodes": len(nodes),
                "direct_edges": direct_edges,
                "bridge_edges": bridge_edges,
                "reasoning_paths": len(paths),
                "target_entity": getattr(graph_path, 'target_entity', None),
            },
        }

    async def _tool_memory_stats(self, args: dict) -> dict:
        """Get memory statistics"""
        memories = self.system.get_all_memories()

        return {
            "total_entries": len(memories),
            "total_dialogues_processed": self._dialogue_count,
            "configuration": {
                "redundancy_threshold": self.system.memory_builder.redundancy_threshold if hasattr(self.system.memory_builder, 'redundancy_threshold') else "N/A",
                "coarsening_threshold": self.system.memory_builder.coarsening_threshold if hasattr(self.system.memory_builder, 'coarsening_threshold') else "N/A",
                "semantic_top_k": getattr(self.system.graph_retriever, 'semantic_top_k', "N/A"),
            },
            "model_info": {
                "llm_model": self.system.llm_client.model if hasattr(self.system, 'llm_client') else "N/A",
            },
        }

    async def _tool_memory_clear(self, args: dict) -> dict:
        """Clear all memories"""
        if not args.get("confirm", False):
            return {
                "success": False,
                "message": "Please set confirm=true to clear all memories. This cannot be undone.",
            }

        # Reinitialize system with clear_db=True
        self.system = AriadneMemSystem(clear_db=True)
        self._dialogue_count = 0

        return {
            "success": True,
            "message": "All memories cleared and system reinitialized",
        }

    async def _handle_resources_list(self, params: dict) -> dict:
        """Handle resources/list request"""
        return {
            "resources": [
                {
                    "uri": "memory://ariadnemem/stats",
                    "name": "Memory Statistics",
                    "description": "Current memory store statistics and configuration",
                    "mimeType": "application/json",
                },
                {
                    "uri": "memory://ariadnemem/all",
                    "name": "All Memories",
                    "description": "All stored memory entries",
                    "mimeType": "application/json",
                },
            ]
        }

    async def _handle_resources_read(self, params: dict) -> dict:
        """Handle resources/read request"""
        uri = params.get("uri", "")

        if uri.endswith("/stats"):
            stats = await self._tool_memory_stats({})
            content = json.dumps(stats, ensure_ascii=False)
        elif uri.endswith("/all"):
            memories = self.system.get_all_memories()
            entries = []
            for m in memories:
                entries.append({
                    "content": m.lossless_restatement,
                    "timestamp": m.timestamp,
                    "persons": m.persons,
                    "location": m.location,
                    "topic": m.topic,
                })
            content = json.dumps({
                "entries": entries,
                "total": len(entries),
            }, ensure_ascii=False)
        else:
            raise ValueError(f"Unknown resource: {uri}")

        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": content,
                }
            ]
        }
