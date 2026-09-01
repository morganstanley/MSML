<p align="center">
  <img src="https://raw.githubusercontent.com/LLM-VLM-GSL/AriadneMem.github.io/main/image/000.png" width="360" alt="AriadneMem Logo">
</p>

<p align="center">
  <strong>Threading the Maze of Lifelong Memory for LLM Agents</strong>
</p>

<p align="center">
  AriadneMem is a structured memory system that addresses disconnected evidence and state update challenges in long-horizon LLM agents through a decoupled two-phase pipeline.
</p>

<p align="center">
  <a href="https://llm-vlm-gsl.github.io/AriadneMem.github.io/"><strong>🌐 Project Page</strong></a> | 
  <a href="https://arxiv.org/pdf/2603.03290"><strong>📥 Paper (PDF)</strong></a>
</p>



If you find our work is useful in your research, please consider raising a star  :star:  and citing:

```
@misc{zhu2026ariadnememthreadingmazelifelong,
      title={AriadneMem: Threading the Maze of Lifelong Memory for LLM Agents}, 
      author={Wenhui Zhu and Xiwen Chen and Zhipeng Wang and Jingjing Wang and Xuanzhao Dong and Minzhou Huang and Rui Cai and Hejian Sang and Hao Wang and Peijie Qiu and Yueyue Deng and Prayag Tiwari and Brendan Hogan Rappazzo and Yalin Wang},
      year={2026},
      eprint={2603.03290},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2603.03290}, 
}
```


---

<h3 align="center">🚀 Platform Compatibility</h3>

<p align="center">
  Works seamlessly with any AI platform supporting <b>MCP</b> or <b>Python integration</b>.
</p>

<div align="center">

| <img src="https://cdn.simpleicons.org/cursor/000000" width="40" alt="Cursor"/><br>**Cursor** | <img src="https://cdn.simpleicons.org/claude/D97757" width="40" alt="Claude"/><br>**Claude** | <img src="https://cdn.simpleicons.org/github/181717" width="40" alt="GitHub"/><br>**Copilot** | <img src="https://cdn.simpleicons.org/python/3776AB" width="40" alt="Python"/><br>**Python** | <img src="https://cdn.simpleicons.org/fastapi/009688" width="40" alt="MCP"/><br>**MCP Client** |
| :---: | :---: | :---: | :---: | :---: |
| ✅ Fully Tested | 🤝 Compatible | 🤝 Compatible | ✅ Fully Tested | 🔗 Universal |

</div>

---

### 🌟 Key Features

* **Two-Phase Pipeline:** Decouples memory retrieval and state updates for enhanced stability.
* **Evidence Threading:** Successfully bridges disconnected information across long-horizon tasks.
* **Plug & Play:** Easy integration with modern AI IDEs and development workflows.

---

## Quick Start

> **💡 Hardware**: AriadneMem works on **both GPU and CPU**. Uses remote LLM APIs (OpenAI/Qwen) and local embedding models.

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

```bash
cp config.py.example config.py
```

Edit `config.py`:

```python
OPENAI_API_KEY = "your-api-key"
OPENAI_BASE_URL = None  # or Qwen: "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL = "gpt-4o"    # or "qwen-plus-2025-07-28"

# Per-component model overrides (optional, falls back to LLM_MODEL)
BUILDER_LLM_MODEL = None   # Phase I: e.g. "gpt-4.1-mini" for cost savings
ANSWER_LLM_MODEL  = None   # Phase II: e.g. "gpt-4o" for better quality

# Reasoning mode
REASONING_MODE = "eco"     # "eco" | "pro" | "custom"

# Local Embedding Model (no API needed)
# Lightweight option (fast on CPU):
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Or for better retrieval quality (GPU accelerates):
# EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
```

### Basic Usage

```python
from main import AriadneMemSystem
from models.memory_entry import Dialogue

# Initialize system
system = AriadneMemSystem(clear_db=True)

# Add dialogues
dialogues = [
    Dialogue(speaker="Alice", content="Let's meet at Starbucks tomorrow at 2pm", timestamp="2024-01-15T14:30:00"),
    Dialogue(speaker="Bob", content="Sorry, can we change to 3pm?", timestamp="2024-01-15T15:00:00"),
    Dialogue(speaker="Alice", content="Sure, 3pm works for me", timestamp="2024-01-15T15:05:00"),
]
system.add_dialogues(dialogues)

# Build memory graph
system.finalize()

# Query
answer = system.ask("What time (in hour) will Alice and Bob meet?")
# Output: "3pm" (correctly handles the state update from 2pm to 3pm)
```

---

## API Reference

### AriadneMemSystem

```python
class AriadneMemSystem:
    def __init__(
        self,
        api_key: str = None,           # Uses config.OPENAI_API_KEY if None
        model: str = None,             # Default LLM model (falls back to config.LLM_MODEL)
        base_url: str = None,          # Uses config.OPENAI_BASE_URL if None
        clear_db: bool = False,        # Clear existing database
        db_path: str = None,           # Custom database path
        redundancy_threshold: float = None,
        coarsening_threshold: float = None,
        builder_model: str = None,     # Phase I model override (extraction + coarsening)
        answer_model: str = None,      # Phase II model override (topology-aware synthesis)
        reasoning_mode: str = None     # "eco" | "pro" | "custom"
    )
    
    def add_dialogue(self, speaker: str, content: str, timestamp: str = None)
    def add_dialogues(self, dialogues: List[Dialogue])
    def finalize(self)  # Build memory graph
    def ask(self, question: str) -> str
    def get_all_memories() -> List[MemoryEntry]
    def print_memories()
```

#### Per-Component LLM Models

Different phases can use different LLM models. Set via `__init__` params or `config.py` (init params take priority):

```python
# Option 1: via __init__ (runtime)
system = AriadneMemSystem(
    builder_model="gpt-4.1-mini",   # Phase I: cheaper model
    answer_model="gpt-4o",          # Phase II: stronger model
)

# Option 2: via config.py (global default)
BUILDER_LLM_MODEL = "gpt-4.1-mini"
ANSWER_LLM_MODEL  = "gpt-4o"
```

If both are `None`, all phases use `model` (or `config.LLM_MODEL`).

#### Reasoning Mode

Control retrieval depth and prompt verbosity. Set via `__init__` or `config.py`:

| Mode | `MAX_REASONING_PATHS` | `MAX_REASONING_PATH_DEPTH` | Reasoning Length |
|------|-----------------------|----------------------------|-----------------|
| `"eco"` (default) | 10 | 3 | 1-2 sentences |
| `"pro"` | 25 | 3 | 9-10 sentences |
| `"custom"` | User-defined | User-defined | User-defined template |

```python
# Option 1: via __init__ (runtime)
system = AriadneMemSystem(reasoning_mode="pro")

# Option 2: via config.py (global default)
REASONING_MODE = "eco"   # "eco" | "pro" | "custom"
```

---

## Configuration Reference

### LLM Configuration

| Parameter | Description | Example |
|-----------|-------------|---------|
| `LLM_MODEL` | Default model for all components | `gpt-4o`, `qwen-plus-2025-07-28` |
| `BUILDER_LLM_MODEL` | Phase I model override (set `None` to use default) | `gpt-4.1-mini` |
| `ANSWER_LLM_MODEL` | Phase II model override (set `None` to use default) | `gpt-4o` |
| `OPENAI_BASE_URL` | API endpoint | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `ENABLE_THINKING` | Qwen deep thinking mode | `True` / `False` |
| `USE_JSON_FORMAT` | Force JSON output | `True` (recommended) |

### Reasoning Modes

| Mode | `MAX_REASONING_PATHS` | Reasoning Depth | Token Cost | Use Case |
|------|----------------------|-----------------|------------|----------|
| `"eco"` | 10 | 1-2 sentences | Low | Simple queries, batch testing |
| `"pro"` | 25 | 9-10 sentences | High | Multi-hop, complex reasoning |
| `"custom"` | User-defined | User-defined | Varies | Fine-tuned for specific tasks |

```python
# Switch mode in config.py
REASONING_MODE = "eco"     # Fast & token-efficient
REASONING_MODE = "pro"     # Thorough & detailed
REASONING_MODE = "custom"  # Your own settings + prompt template
```

### Phase I Parameters (Memory Construction)

| Parameter | Default | Paper | Description |
|-----------|---------|-------|-------------|
| `REDUNDANCY_THRESHOLD` | 0.6 | λ_red (Eq.3) | Entropy-aware gating threshold |
| `COARSENING_THRESHOLD` | 0.6 | λ_coal (Eq.6) | Merge vs Link decision threshold |
| `WINDOW_SIZE` | 40 | - | Dialogues per processing window |
| `OVERLAP_SIZE` | 2 | - | Window overlap for context continuity |

### Phase II Parameters (Retrieval & Reasoning)

| Parameter | Default | Paper | Description |
|-----------|---------|-------|-------------|
| `SEMANTIC_TOP_K` | 25 | - | Max nodes from semantic search |
| `KEYWORD_TOP_K` | 5 | - | Max nodes from keyword search |
| `MAX_REASONING_PATH_DEPTH` | 3 | L (Eq.10) | Max hops in DFS path discovery (auto-set by mode) |
| `MAX_REASONING_PATHS` | 10/25 | - | Max reasoning paths (eco=10, pro=25, auto-set by mode) |

### Prompt Templates (Customizable)

Prompt templates are auto-selected based on `REASONING_MODE`. You can also define a fully custom template:

```python
# System prompt for topology-aware synthesis
ANSWER_SYSTEM_PROMPT = "You are a QA system with graph-based memory..."

# Custom mode: define your own template
REASONING_MODE = "custom"
_CUSTOM_USER_PROMPT_TEMPLATE = """Q: {query}
{entity_hint}{graph_hint}
{context_str}
... your own reasoning instructions ...
"""
```

---

## Running Tests

### Quick Test

```bash
python quick_test.py
```

### LoCoMo Benchmark

```bash
# Run on 3 sessions with parallel question processing
python test_locomo10.py --num_sessions 3 --parallel_questions

# Run with LLM-as-Judge evaluation
python test_locomo10.py --num_sessions 3 --use_llm_judge
```

### Multi-hop Reasoning Demo

```bash
python demo_multihop.py
```

### MCP Server (Cursor Integration)

AriadneMem can be used as an MCP server in Cursor, providing long-term memory tools directly in your AI chat.

**stdio mode (recommended for Cursor):**

Edit `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "ariadnemem": {
      "command": "/path/to/python",
      "args": ["/path/to/MCP/server/stdio_server.py"]
    }
  }
}
```

For remote compute (e.g. Slurm clusters), use SSH to jump to the GPU node:

```json
{
  "mcpServers": {
    "ariadnemem": {
      "command": "ssh",
      "args": [
        "-o", "StrictHostKeyChecking=no",
        "-o", "LogLevel=ERROR",
        "gpu-node-name",
        "/path/to/python",
        "/path/to/MCP/server/stdio_server.py"
      ]
    }
  }
}
```

**HTTP mode (for programmatic clients):**

```bash
cd MCP
pip install -r requirements.txt
python run.py
```

See [MCP/README.md](MCP/README.md) for full setup guide with step-by-step CoreWeave/Slurm example, tool reference, and troubleshooting.

---

> **🚧 Under Active Development**: We are currently optimizing memory construction for **code** and **math** domains to better handle technical content and formal reasoning.

## Key Features

| Feature | Paper Reference | Benefit |
|---------|-----------------|---------|
| **Entropy-Aware Gating** | Eq. 2-3 | Filters noise before LLM extraction |
| **Conflict-Aware Coarsening** | Eq. 5-6 | Merges duplicates while preserving state updates |
| **Hybrid Retrieval** | Eq. 7 | Semantic + Lexical search for terminal nodes |
| **Bridge Discovery** | Eq. 9 | Steiner tree approximation for missing links |
| **Multi-Hop Path Mining** | Eq. 10 | DFS-based reasoning chain discovery |
| **Topology-Aware Synthesis** | Eq. 11 | Single LLM call with graph-guided reasoning |

### Comparison with Baselines

| Dimension | Flat RAG | Planning-based | AriadneMem |
|-----------|----------|----------------|------------|
| **Retrieval** | Vector search | Multi-round LLM | Graph + Algorithm |
| **Multi-hop** | Not supported | 3-4 LLM calls | DFS (0 LLM calls) |
| **State Updates** | Keep all / Conflict | Keep all | Smart merge + temporal edges |
| **LLM Calls/Query** | 1 | 4-6 | 1 |
| **Latency** | Fast | Slow | Fast |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      AriadneMem Pipeline                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ═══════════════════════════════════════════════════════════════   │
│  ║           PHASE I: Asynchronous Memory Construction          ║   │
│  ═══════════════════════════════════════════════════════════════   │
│                                                                     │
│  [Dialogue Stream D]                                                │
│         │                                                           │
│         ▼                                                           │
│  ┌────────────────────────────────┐                                │
│  │  Entropy-Aware Gating (Eq.3)  │  ← Φ_gate: filter low-info     │
│  │  H(m) < τ → block             │                                │
│  └────────────────────────────────┘                                │
│         │                                                           │
│         ▼                                                           │
│  ┌────────────────────────────────┐                                │
│  │  Atomic Extraction F_θ (Eq.4) │  ← LLM: dialogue → entries     │
│  │  De-linearization transform   │                                │
│  └────────────────────────────────┘                                │
│         │                                                           │
│         ▼                                                           │
│  ┌────────────────────────────────┐                                │
│  │  Conflict-Aware Coarsening    │  ← Merge/Link/Add (Eq.6)       │
│  │  (Eq.5-6)                     │                                │
│  │  • Static duplicates → Merge  │                                │
│  │  • State updates → Link edge  │                                │
│  └────────────────────────────────┘                                │
│         │                                                           │
│         ▼                                                           │
│  ┌────────────────────────────────┐                                │
│  │  VectorStore (LanceDB)        │  ← Multi-view indexing         │
│  │  • Semantic (dense vectors)   │                                │
│  │  • Lexical (keyword/BM25)     │                                │
│  │  • Symbolic (metadata)        │                                │
│  └────────────────────────────────┘                                │
│                                                                     │
│  ═══════════════════════════════════════════════════════════════   │
│  ║           PHASE II: Real-Time Structural Reasoning           ║   │
│  ═══════════════════════════════════════════════════════════════   │
│                                                                     │
│  [Query q]                                                          │
│         │                                                           │
│         ▼                                                           │
│  ┌────────────────────────────────┐                                │
│  │  Fast Paths (O(1) lookup)     │  ← Cache/regex short-circuit   │
│  │  Count/List/Relation queries  │                                │
│  └────────────────────────────────┘                                │
│         │ (if miss)                                                 │
│         ▼                                                           │
│  ┌────────────────────────────────┐                                │
│  │  Hybrid Retrieval (Eq.7)      │  ← Find terminal nodes V_term  │
│  │  score = α·sim_sem + β·sim_lex│                                │
│  └────────────────────────────────┘                                │
│         │                                                           │
│         ▼                                                           │
│  ┌────────────────────────────────┐                                │
│  │  Base Graph Construction      │  ← Entity/temporal edges       │
│  │  (Eq.8)                       │                                │
│  └────────────────────────────────┘                                │
│         │                                                           │
│         ▼                                                           │
│  ┌────────────────────────────────┐                                │
│  │  Bridge Discovery (Eq.9)      │  ← Steiner tree approximation  │
│  │  Find b* to connect V_term    │     (no LLM calls!)            │
│  └────────────────────────────────┘                                │
│         │                                                           │
│         ▼                                                           │
│  ┌────────────────────────────────┐                                │
│  │  Multi-Hop Path Mining (Eq.10)│  ← DFS reasoning chains        │
│  │  Discover logical paths P_q   │                                │
│  └────────────────────────────────┘                                │
│         │                                                           │
│         ▼                                                           │
│  ┌────────────────────────────────┐                                │
│  │  Topology-Aware Synthesis     │  ← Single LLM call             │
│  │  (Eq.11)                      │                                │
│  │  a = LLM(q, Serialize(G_q))   │                                │
│  └────────────────────────────────┘                                │
│         │                                                           │
│         ▼                                                           │
│  [Answer a]                                                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
AriadneMem/
├── main.py                          # Main system entry point
├── config.py                        # Configuration (LLM, thresholds, prompts, modes)
├── requirements.txt                 # Dependencies
│
├── core/
│   ├── ariadne_memory_builder.py    # Phase I: Memory Construction
│   ├── ariadne_graph_retriever.py   # Phase II: Structural Reasoning
│   ├── ariadne_answer_generator.py  # Topology-Aware Synthesis
│   ├── semantic_normalizer.py       # Answer post-processing
│   └── aggregation_builder.py       # Entity aggregation
│
├── models/
│   ├── memory_entry.py              # MemoryEntry, Dialogue dataclasses
│   └── enhanced_structures.py       # EnhancedMemoryIndex, caches
│
├── database/
│   └── vector_store.py              # LanceDB vector store
│
├── utils/
│   ├── llm_client.py                # OpenAI-compatible LLM client
│   └── embedding.py                 # SentenceTransformers embeddings
│
├── dataset/
│   └── locomo10.json                # LoCoMo benchmark data
│
├── MCP/                             # MCP Server (Model Context Protocol)
│   ├── README.md                    # MCP documentation
│   ├── run.py                       # HTTP server entry point
│   ├── requirements.txt             # MCP dependencies
│   ├── mcp_config/
│   │   └── settings.py              # Server settings (inherits from config.py)
│   └── server/
│       ├── stdio_server.py          # stdio transport (recommended for Cursor)
│       ├── http_server.py           # HTTP transport (FastAPI + Streamable HTTP)
│       └── mcp_handler.py           # MCP protocol handler (7 tools)
│
├── test_locomo10.py                 # Full benchmark evaluation
├── quick_test.py                    # Quick functionality test
└── demo_multihop.py                 # Multi-hop reasoning demo
```

---

## Troubleshooting

### Q: How to switch to Qwen models?

```python
# config.py
OPENAI_API_KEY = "your-qwen-api-key"
OPENAI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL = "qwen-plus-2025-07-28"
ENABLE_THINKING = True  # Enable Qwen's deep thinking mode
```

### Q: Multi-hop reasoning not working?

Check:
1. Nodes have shared entities or temporal proximity
2. Inspect discovered paths: `graph_path.reasoning_paths`
3. Increase `MAX_REASONING_PATH_DEPTH` for longer chains

### Q: How to adjust filtering strength?

```python
# More aggressive filtering (fewer nodes, faster)
REDUNDANCY_THRESHOLD = 0.5
COARSENING_THRESHOLD = 0.5

# More permissive (more nodes, better recall)
REDUNDANCY_THRESHOLD = 0.7
COARSENING_THRESHOLD = 0.7
```

---

## Citation

```bibtex
@article{zhu2026ariadnemem,
  title   = {AriadneMem: Threading the Maze of Lifelong Memory for LLM Agents},
  author  = {Zhu, Wenhui and Chen, Xiwen and Wang, Zhipeng and Wang, Jingjing and Dong, Xuanzhao and Huang, Minzhou and Cai, Rui and Sang, Hejian and Wang, Hao and Qiu, Peijie and Deng, Yueyue and Tiwari, Prayag and Hogan Rappazzo, Brendan and Wang, Yalin},
  journal = {Preprint},
  year    = {2026},
  url     = {https://github.com/LLM-VLM-GSL/AriadneMem}
}
```

---

## Acknowledgments

We would like to thank the following projects and teams:
- **Codebase**: [SimpleMem](https://github.com/aiming-lab/SimpleMem/tree/main) (Special thanks to their open-source contribution!)
- **Embedding Models**: 
  - [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (Sentence Transformers) - Lightweight and CPU-friendly
  - [Qwen3-Embedding](https://github.com/QwenLM/Qwen) - State-of-the-art retrieval performance
- **Vector Database**: [LanceDB](https://lancedb.com/) - High-performance columnar storage
- **Benchmark**: [LoCoMo](https://github.com/snap-research/locomo) - Long-context memory evaluation framework

---

## License

This work is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).

You are free to use, share, and adapt this work for **non-commercial purposes** with proper attribution. For commercial licensing, please contact the authors.
