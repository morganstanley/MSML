# Tools

Agents act on the world through **tools**. Each [Agent](04_agents.md) is granted a
subset; a call is dispatched by name to the tool's implementation, which returns
output (and optionally an image) back into the agent loop.

## What's available

| Tool | Purpose |
|------|---------|
| `shell_exec` | Run a shell command in the workspace |
| `read_file`, `grep_file` | Read and search workspace files |
| `view_image` | Load a generated image (e.g. a plot) into the conversation |
| `ask_user`, `report_to_user` | Ask a question / finish and report back |
| `memory_store`, `memory_search`, `memory_read`, `memory_import` | Read, write, and import persistent [Memory](06_memory.md) |
| `spawn_sub_agent` | Delegate a subtask to a child agent |
| `read_board`, `propose_experiment`, `update_experiment`, `cancel_experiments` | Manage Phase 3 experiments |
| `reality_check` | System-side validation of an experiment |
| `update_playbook` | Append to the strategist's `playbook.md` |
| `read_adapter`, `write_adapter_file`, `patch_adapter_file`, `read_reference_adapter` | Inspect and edit the [Adapter](03_adapters.md) |
| `web_search` | Web search (a provider built-in, not a registry file) |

## Definitions

Tools mirror [Agents](04_agents.md): each is one markdown file at
`src/alpha_lab/tools/registry/<tool>.md` with the same frontmatter contract, and
its JSON-Schema parameters under `metadata.parameters`. `load_tool("shell_exec")`
reads a file into a `ToolDefinition`; an agent's `allowed-tools` resolve to a
tuple of these. Implementations live separately in `tools/execution.py` — the
definition files carry no code reference.

## Granting

Each [Agent](04_agents.md) is granted only the tools it needs, through the
`allowed-tools` list in its definition — that list is what bounds what the agent
can do. `load_tools` resolves those names into the tuple of `ToolDefinition`s the
agent's loop can call; filesystem confinement is handled separately by the
[Sandbox](04_agents.md).

## Reaching run state (`deps`)

A `ToolDefinition` is inert markdown; its implementation in `tools/execution.py`
is where the work happens. Any tool that touches live run state does so through
`deps` — a run-scoped container (`RunDeps`) holding the config, workspace,
executors, and memory store, published for the run's duration. This is the same
idea as [pydantic-ai](https://ai.pydantic.dev/dependencies/)'s dependency
injection — where a tool reads its run-scoped dependencies off `RunContext.deps` —
except Alpha Lab publishes them as a single module global rather than threading a
typed `deps` object through a run context. So a tool reads `deps.get()` for the
run's config/workspace/executors, or `deps.memory_store` directly; `execute_tool`
fails loud when there's no active run. This matters when
**adding a tool**: if it needs the workspace, an executor, or memory, that's the
seam to reach for — a purely computational tool needs none of it.
