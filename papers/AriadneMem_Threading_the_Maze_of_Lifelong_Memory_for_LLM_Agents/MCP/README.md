# AriadneMem MCP Server

**Graph-Based Lifelong Memory Service for LLM Agents via Model Context Protocol (MCP)**

AriadneMem MCP Server exposes the AriadneMem memory system as an MCP service, enabling AI assistants to store, retrieve, and query conversational memories using graph-based reasoning.

**Supported Platforms**

| [<img src="https://cdn.simpleicons.org/cursor/000000" width="40" alt="Cursor"/>](https://cursor.com) | [<img src="https://cdn.simpleicons.org/claude/D97757" width="40" alt="Claude Desktop"/>](https://www.anthropic.com/claude) | [<img src="https://cdn.simpleicons.org/github/181717" width="40" alt="GitHub Copilot"/>](https://github.com/features/copilot) | **+ Any MCP Client** |
|:---:|:---:|:---:|:---:|
| **Cursor** | **Claude Desktop** | **GitHub Copilot** | |
| Fully Tested | Compatible | Compatible | |

## Key Differentiators

| Feature | AriadneMem | Planning-based (e.g. SimpleMem) |
|---------|------------|-------------------------------|
| **Multi-hop Reasoning** | Graph algorithms (DFS) | Multiple LLM calls |
| **State Updates** | Conflict-aware coarsening | Keep all / overwrite |
| **Bridge Discovery** | Steiner tree approximation | N/A |
| **LLM Calls per Query** | 1 (topology-aware synthesis) | 4-6 (plan + reflect) |
| **Latency** | Low | High |

## Transport Modes

AriadneMem MCP supports two transport modes:

| Mode | Use Case | Setup Complexity |
|------|----------|-----------------|
| **stdio + SSH** (recommended) | Cursor / Claude Desktop with remote compute | Medium |
| **stdio local** | Cursor / Claude Desktop on the same machine | Low |
| **HTTP** (Streamable HTTP) | Programmatic clients, multi-client, REST API | Low |

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                   AriadneMem MCP Server                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Transport Layer (choose one):                                  │
│   ┌─────────────────────────────┐  ┌──────────────────────────┐ │
│   │  stdio (stdin/stdout)       │  │  HTTP (FastAPI)          │ │
│   │  - Cursor spawns process    │  │  - POST /mcp  (JSON-RPC) │ │
│   │  - JSON-RPC via pipes       │  │  - GET  /mcp  (SSE)      │ │
│   │  - SSH for remote compute   │  │  - DELETE /mcp (close)    │ │
│   └─────────────────────────────┘  └──────────────────────────┘ │
│                         │                     │                   │
│                         └──────────┬──────────┘                  │
│                                    ▼                             │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │                  MCP Handler (JSON-RPC 2.0)               │  │
│   │  7 Tools: memory_add, memory_query, memory_graph_inspect  │  │
│   │  2 Resources: stats, all memories                         │  │
│   └──────────────────────────────────────────────────────────┘  │
│                                    │                             │
│                                    ▼                             │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │               AriadneMemSystem                            │  │
│   │  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  │  │
│   │  │ Phase I:    │  │ Phase II:    │  │ Topology-Aware │  │  │
│   │  │ Memory      │  │ Structural   │  │ Synthesis      │  │  │
│   │  │ Construction│  │ Reasoning    │  │ (Single LLM)   │  │  │
│   │  └─────────────┘  └──────────────┘  └────────────────┘  │  │
│   └──────────────────────────────────────────────────────────┘  │
│                                    │                             │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │  LanceDB Vector Store (Semantic + Lexical + Symbolic)    │  │
│   └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │  LLM API (OpenAI / Qwen / compatible)                    │  │
│   └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

> **💡 Hardware**: Works on **both GPU and CPU**. Uses remote LLM APIs and local embedding models.

---

## Cursor Integration (Slurm / Remote Compute)

This is the most common setup when using a shared cluster (e.g. CoreWeave) where:
- Cursor connects to a **login node** via Remote SSH
- Compute runs on a **GPU node** allocated by Slurm
- The login node cannot run compute tasks directly

The solution: Cursor spawns an `ssh` command that jumps to the GPU node and runs the stdio server there.

### Step-by-Step (CoreWeave Example)

**Step 1: Allocate a GPU node**

Open a terminal (inside Cursor or via SSH) and request a node:

```bash
srun --partition=hpc-mid --gpus=1 --time=02:00:00 --pty bash
```

> Use `tmux` or `screen` on the login node first so the allocation persists if Cursor reconnects.

**Step 2: Note the node name**

```bash
squeue -u $USER
```

```
JOBID   PARTITION   NAME   USER    ST   TIME   NODES   NODELIST(REASON)
32837   hpc-mid     bash   your_user  R    54:40  1       slurm-h100-206-067
```

The node name here is `slurm-h100-206-067`.

**Step 3: Install dependencies on the cluster**

This only needs to be done once:

```bash
# Create a conda environment (on the GPU node or login node -- they share the filesystem)
conda create -n ariadne python=3.11 -y
conda activate ariadne

# Clone the repo
git clone https://github.com/LLM-VLM-GSL/AriadneMem.git
cd AriadneMem

# Install dependencies
pip install -r requirements.txt
pip install -r MCP/requirements.txt

# Set up config
cp config.py.example config.py
# Edit config.py with your API key and model settings
```

**Step 4: Configure Cursor MCP**

Edit `~/.cursor/mcp.json` on the remote machine (the login node):

```json
{
  "mcpServers": {
    "ariadnemem": {
      "command": "ssh",
      "args": [
        "-o", "StrictHostKeyChecking=no",
        "-o", "LogLevel=ERROR",
        "slurm-h100-206-067",
        "/home/YOUR_USER/.conda/envs/ariadne/bin/python",
        "/home/YOUR_USER/AriadneMem/MCP/server/stdio_server.py"
      ]
    }
  }
}
```

Replace:
- `slurm-h100-206-067` with your actual GPU node name from Step 2
- `/home/YOUR_USER/.conda/envs/ariadne/bin/python` with the full path to your conda Python (find it with `conda run -n ariadne which python`)
- `/home/YOUR_USER/AriadneMem/` with your actual repo path

> **When your GPU node changes** (e.g. after a new `srun`), just update the node name in `mcp.json` and reload Cursor.

**Step 5: Reload Cursor**

Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac) and run:

```
Developer: Reload Window
```

**Step 6: Verify**

Go to `Settings > MCP`. You should see:
- `ariadnemem` listed as a server
- Status indicator is green (not red)
- 7 tools and 2 resources detected

You can now use AriadneMem tools in any Cursor chat:
- "Use AriadneMem to remember that my favorite IDE is Cursor"
- "Query AriadneMem: what is my favorite IDE?"
- "Show me the AriadneMem graph structure for 'project deadlines'"

### Troubleshooting (Remote Setup)

**Red indicator in Settings > MCP?**

1. Verify SSH works from the login node:
   ```bash
   ssh slurm-h100-206-067 echo "OK"
   ```
2. Verify the server can start:
   ```bash
   echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | \
     ssh slurm-h100-206-067 /path/to/python /path/to/stdio_server.py
   ```
   You should see a JSON response with `"serverInfo"`.

3. Make sure `config.py` exists in the repo root (not just `config.py.example`).

4. Check that the GPU node is still allocated:
   ```bash
   squeue -u $USER
   ```

**Tools not showing up?**

Reload Cursor window after any `mcp.json` change. The server needs a fresh start.

---

## Cursor Integration (Local / Fixed Server)

If Cursor and AriadneMem run on the same machine (e.g. your laptop, a fixed GPU server), no SSH is needed:

### Configuration

Edit `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "ariadnemem": {
      "command": "/path/to/python",
      "args": ["/path/to/AriadneMem/MCP/server/stdio_server.py"]
    }
  }
}
```

Then reload Cursor (`Ctrl+Shift+P` > `Developer: Reload Window`).

---

## HTTP Server (Programmatic / Multi-Client)

The HTTP server is useful when:
- You want a persistent server that multiple clients connect to
- You need a REST API alongside MCP
- You are building your own client (not using Cursor/Claude Desktop)

### Start Server

```bash
cd MCP
python run.py
```

### Protocol

| Item | Value |
|------|-------|
| Protocol Version | 2025-03-26 |
| Transport | Streamable HTTP |
| Message Format | JSON-RPC 2.0 |
| Authentication | Bearer Token |

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/mcp` | POST | JSON-RPC messages (requests, notifications) |
| `/mcp` | GET | Server-to-client SSE stream |
| `/mcp` | DELETE | Terminate session |

### Programmatic Example

```python
import httpx

BASE = "http://localhost:8000/mcp"
HEADERS = {
    "Authorization": "Bearer ariadnemem-dev-token",
    "Accept": "application/json",
    "Content-Type": "application/json",
}

# 1. Initialize
resp = httpx.post(BASE, headers=HEADERS, json={
    "jsonrpc": "2.0", "id": 1, "method": "initialize",
    "params": {"protocolVersion": "2025-03-26", "capabilities": {},
               "clientInfo": {"name": "test", "version": "1.0"}}
})
session_id = resp.headers["Mcp-Session-Id"]
HEADERS["Mcp-Session-Id"] = session_id

# 2. Add dialogues
httpx.post(BASE, headers=HEADERS, json={
    "jsonrpc": "2.0", "id": 2, "method": "tools/call",
    "params": {"name": "memory_add_batch", "arguments": {
        "dialogues": [
            {"speaker": "Alice", "content": "Let's meet at 2pm tomorrow"},
            {"speaker": "Bob", "content": "Can we change to 3pm?"},
            {"speaker": "Alice", "content": "Sure, 3pm works"},
        ]
    }}
})

# 3. Query with graph reasoning
resp = httpx.post(BASE, headers=HEADERS, json={
    "jsonrpc": "2.0", "id": 3, "method": "tools/call",
    "params": {"name": "memory_query", "arguments": {
        "question": "What time (in hour) will Alice and Bob meet?"
    }}
})
print(resp.json())
# -> answer: "3pm" (with graph stats showing bridge discovery)
```

---

## MCP Tools

| Tool | Description | AriadneMem Feature |
|------|-------------|-------------------|
| `memory_add` | Add a single dialogue to memory | Phase I: entropy gating + coarsening |
| `memory_add_batch` | Add multiple dialogues at once | Phase I: batch processing |
| `memory_query` | Query and get graph-reasoned answer | Phase II: full pipeline (1 LLM call) |
| `memory_retrieve` | Retrieve raw memory entries | Hybrid retrieval (semantic + lexical) |
| `memory_graph_inspect` | Inspect graph structure for a query | Nodes, edges, bridge connections, DFS paths |
| `memory_stats` | Get memory statistics | Entry count, configuration |
| `memory_clear` | Clear all memories (irreversible) | Database reset |

### Tool Details

#### `memory_query` (Primary Tool)

Executes the full AriadneMem Phase II pipeline:

```
Query -> Fast Path Check -> Hybrid Retrieval -> Graph Construction
     -> Bridge Discovery -> Path Mining -> Topology-Aware Synthesis
```

Returns:
```json
{
  "question": "What time (in hour) will Alice and Bob meet?",
  "answer": "3pm at Starbucks",
  "graph_stats": {
    "nodes_retrieved": 5,
    "edges": 4,
    "reasoning_paths": 2,
    "bridge_nodes_discovered": 1,
    "llm_calls": 1
  }
}
```

#### `memory_graph_inspect` (Explainability Tool)

Returns the full graph structure for debugging/explainability:

```json
{
  "query": "What happened with Alice's trip?",
  "graph": {
    "nodes": [
      {"label": "F1", "content": "Alice planned Paris trip", "timestamp": "2024-01-10"},
      {"label": "F2", "content": "Flight was cancelled", "timestamp": "2024-01-12"},
      {"label": "F3", "content": "Alice booked London instead", "timestamp": "2024-01-13"}
    ],
    "edges": [
      {"source": "F1", "target": "F2", "type": "direct"},
      {"source": "F2", "target": "F3", "type": "inferred"}
    ],
    "reasoning_paths": [
      {"path_id": 1, "hops": 3, "chain": "F1 -> F2 -> F3"}
    ]
  },
  "summary": {
    "total_nodes": 3,
    "direct_edges": 1,
    "bridge_edges": 1,
    "reasoning_paths": 1
  }
}
```

---

## Configuration

The MCP server inherits all settings from the parent `config.py`, including:

| Setting | Description |
|---------|-------------|
| `REASONING_MODE` | `"eco"` (fast, 1-2 sentence reasoning), `"pro"` (thorough, 9-10 sentences), `"custom"` |
| `BUILDER_LLM_MODEL` | Phase I model override (extraction + coarsening) |
| `ANSWER_LLM_MODEL` | Phase II model override (answer generation) |
| `MAX_REASONING_PATHS` | Auto-set by mode: eco=10, pro=25 |

See the parent [README](../README.md) for full configuration reference.

---

## Project Structure

```
MCP/
├── README.md               # This file
├── run.py                  # HTTP server entry point
├── requirements.txt        # MCP-specific dependencies (fastapi, uvicorn)
├── mcp_config/
│   ├── __init__.py
│   └── settings.py         # Server settings (inherits from config.py)
└── server/
    ├── __init__.py
    ├── stdio_server.py     # stdio transport (recommended for Cursor)
    ├── http_server.py      # HTTP transport (FastAPI + Streamable HTTP)
    └── mcp_handler.py      # MCP protocol handler (7 tools, 2 resources)
```

---

## License

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) -- Non-commercial use only. See [../README.md](../README.md#license) for details.
