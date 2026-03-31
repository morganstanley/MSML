"""Tool schemas and implementations for alpha-lab."""

from __future__ import annotations

import base64
import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Tool Schemas (Responses API format)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Tool Registry — each tool has a schema; get_tool_schemas() builds per-step lists
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[str, dict[str, Any]] = {
    "shell_exec": {
        "type": "function",
        "name": "shell_exec",
        "description": (
            "Execute a shell command in the workspace directory. "
            "Commands run inside the workspace directory."
            "Use this to run analysis scripts, install packages, etc. "
            "Write scripts to files first, then execute them."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default 120, max 180).",
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    },
    "view_image": {
        "type": "function",
        "name": "view_image",
        "description": (
            "View a PNG or JPG image file from the workspace. "
            "Use this after generating plots to analyze them visually. "
            "The image will be displayed in the conversation for you to reason about."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the image file (absolute or relative to workspace).",
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    "ask_user": {
        "type": "function",
        "name": "ask_user",
        "description": (
            "Ask the user a question and wait for their response. "
            "ONLY use this when you are completely blocked and cannot proceed "
            "without user input. Do NOT use for status updates or confirmations."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to ask the user.",
                },
            },
            "required": ["question"],
            "additionalProperties": False,
        },
    },
    "report_to_user": {
        "type": "function",
        "name": "report_to_user",
        "description": (
            "Call this ONLY when you have fully completed the entire analysis "
            "and have written all findings to the workspace files. This returns "
            "control to the user. Include a summary of everything you found."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": (
                        "A comprehensive summary of all findings, key insights, "
                        "data quality issues, and recommended next steps."
                    ),
                },
            },
            "required": ["summary"],
            "additionalProperties": False,
        },
    },
    "read_file": {
        "type": "function",
        "name": "read_file",
        "description": (
            "Read a file from the workspace. Returns numbered lines. "
            "Use offset and limit to read portions of large files."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file (absolute or relative to workspace).",
                },
                "offset": {
                    "type": "integer",
                    "description": "Line number to start from (0-based, default 0).",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max number of lines to return (default 500).",
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    "grep_file": {
        "type": "function",
        "name": "grep_file",
        "description": (
            "Search files in the workspace using grep. Returns matching lines "
            "with file paths and line numbers."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The search pattern (regex).",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search (relative to workspace, default '.').",
                },
                "include": {
                    "type": "string",
                    "description": "Glob pattern to filter files (e.g. '*.py').",
                },
            },
            "required": ["pattern"],
            "additionalProperties": False,
        },
    },
    # Phase 3 tools
    "propose_experiment": {
        "type": "function",
        "name": "propose_experiment",
        "description": (
            "Propose a new experiment. Creates an entry in the experiment board "
            "with status 'to_implement'. A worker will implement and run it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": (
                        "Short unique name for the experiment (used as directory name). "
                        "Use snake_case, e.g. 'xgboost_momentum_5d'."
                    ),
                },
                "description": {
                    "type": "string",
                    "description": "Detailed description of what the experiment should do.",
                },
                "hypothesis": {
                    "type": "string",
                    "description": "The hypothesis being tested.",
                },
                "config": {
                    "type": "string",
                    "description": (
                        "JSON string with experiment config: "
                        "{model_type, hyperparams, features, horizon, etc.}"
                    ),
                },
            },
            "required": ["name", "description", "hypothesis", "config"],
            "additionalProperties": False,
        },
    },
    "update_playbook": {
        "type": "function",
        "name": "update_playbook",
        "description": (
            "Write or update the playbook.md file in the workspace. "
            "The playbook contains accumulated strategic wisdom: "
            "what works, what doesn't, and what to try next."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Full text content for playbook.md.",
                },
            },
            "required": ["content"],
            "additionalProperties": False,
        },
    },
    "read_board": {
        "type": "function",
        "name": "read_board",
        "description": (
            "Read the experiment board: column counts, recent experiments, "
            "and the leaderboard (top experiments by Sharpe ratio)."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    "update_experiment": {
        "type": "function",
        "name": "update_experiment",
        "description": (
            "Update an experiment's status, results, or error message. "
            "Use this to transition experiments through kanban columns."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "experiment_id": {
                    "type": "integer",
                    "description": "The experiment ID to update.",
                },
                "status": {
                    "type": "string",
                    "description": (
                        "New kanban status. Valid: to_implement, implemented, "
                        "checked, queued, running, finished, analyzed, done."
                    ),
                },
                "results": {
                    "type": "string",
                    "description": "JSON string of result metrics (key-value pairs for the domain's metrics).",
                },
                "error": {
                    "type": "string",
                    "description": "Error message if the experiment failed.",
                },
                "debrief_path": {
                    "type": "string",
                    "description": "Path to the debrief markdown file (relative to workspace).",
                },
            },
            "required": ["experiment_id"],
            "additionalProperties": False,
        },
    },
    "reality_check": {
        "type": "function",
        "name": "reality_check",
        "description": (
            "Run validation reality check on a slice of real data BEFORE marking "
            "experiment as checked. This catches data leakage, missing data, short OOS "
            "windows, and timing issues that smoke tests on synthetic data miss. "
            "REQUIRED after smoke test, before updating to 'checked' status."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "experiment_name": {
                    "type": "string",
                    "description": "Name of the experiment directory (e.g. 'xgboost_momentum_5d').",
                },
            },
            "required": ["experiment_name"],
            "additionalProperties": False,
        },
    },
    "write_adapter_file": {
        "type": "function",
        "name": "write_adapter_file",
        "description": (
            "Write a file to the workspace adapter directory. "
            "Valid filenames: manifest.json, domain_knowledge.md, "
            "and the 9 prompt files (phase1.md, phase2_builder.md, etc.)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Filename to write (e.g. 'manifest.json', 'phase1.md').",
                },
                "content": {
                    "type": "string",
                    "description": "File content to write.",
                },
            },
            "required": ["filename", "content"],
            "additionalProperties": False,
        },
    },
    "read_reference_adapter": {
        "type": "function",
        "name": "read_reference_adapter",
        "description": (
            "Read a built-in reference adapter to understand the expected format. "
            "Returns all files concatenated. Available adapters: time_series, cuda_kernel, nanogpt."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Built-in adapter name: 'time_series', 'cuda_kernel', or 'nanogpt'.",
                },
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    },
    "read_adapter": {
        "type": "function",
        "name": "read_adapter",
        "description": (
            "Read the current workspace adapter files. "
            "Returns all adapter files concatenated."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    "patch_adapter_file": {
        "type": "function",
        "name": "patch_adapter_file",
        "description": (
            "Patch (overwrite) a file in the workspace adapter directory. "
            "Creates a git checkpoint in the workspace before writing. "
            "Valid filenames: manifest.json, domain_knowledge.md, and prompt .md files."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Filename to patch (e.g. 'phase3_strategist.md').",
                },
                "content": {
                    "type": "string",
                    "description": "New file content.",
                },
                "reason": {
                    "type": "string",
                    "description": "Reason for the patch.",
                },
            },
            "required": ["filename", "content", "reason"],
            "additionalProperties": False,
        },
    },
    "spawn_sub_agent": {
        "type": "function",
        "name": "spawn_sub_agent",
        "description": (
            "Spawn a sub-agent to work on a focused sub-task in its own conversation context. "
            "The sub-agent inherits your model, provider, and tools (except spawn_sub_agent). "
            "It runs to completion and returns its final report. Use this to delegate "
            "self-contained sub-problems that benefit from a fresh context window."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": (
                        "Clear description of what the sub-agent should accomplish. "
                        "Be specific about expected outputs and success criteria."
                    ),
                },
                "context": {
                    "type": "string",
                    "description": (
                        "Background information the sub-agent needs: data paths, "
                        "prior findings, constraints, relevant file locations."
                    ),
                },
            },
            "required": ["task"],
            "additionalProperties": False,
        },
    },
    "cancel_experiments": {
        "type": "function",
        "name": "cancel_experiments",
        "description": (
            "Cancel one or more queued experiments. Use this to prune experiments "
            "that are unlikely to beat current best based on learnings from completed runs. "
            "Can only cancel experiments in 'to_implement' status (not yet started). "
            "Provide a reason for the cancellation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "experiment_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "List of experiment IDs to cancel.",
                },
                "reason": {
                    "type": "string",
                    "description": (
                        "Why these experiments are being cancelled "
                        "(e.g. 'Similar approach already failed in experiment #42')."
                    ),
                },
            },
            "required": ["experiment_ids", "reason"],
            "additionalProperties": False,
        },
    },
}

# Backward-compat aliases
FUNCTION_TOOLS: list[dict[str, Any]] = [
    TOOL_REGISTRY[name]
    for name in ("shell_exec", "view_image", "ask_user", "report_to_user")
]

WEB_SEARCH_TOOL: dict[str, Any] = {"type": "web_search_preview"}

ALL_TOOL_SCHEMAS: list[dict[str, Any]] = FUNCTION_TOOLS + [WEB_SEARCH_TOOL]


def get_tool_schemas(
    tool_names: list[str],
    include_web_search: bool = False,
) -> list[dict[str, Any]]:
    """Build a tool schema list from named tools in the registry."""
    schemas = [TOOL_REGISTRY[name] for name in tool_names if name in TOOL_REGISTRY]
    if include_web_search:
        schemas.append(WEB_SEARCH_TOOL)
    return schemas


# ---------------------------------------------------------------------------
# Tool Implementations
# ---------------------------------------------------------------------------

MAX_OUTPUT_CHARS = 30_000
DEFAULT_TIMEOUT = 120
MAX_TIMEOUT = 180


def execute_shell(
    command: str,
    workspace: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    """Execute a shell command in the workspace directory."""
    timeout = min(max(timeout, 1), MAX_TIMEOUT)

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        output_parts = []
        if result.stdout:
            output_parts.append(result.stdout)
        if result.stderr:
            output_parts.append(f"[stderr]\n{result.stderr}")
        output_parts.append(f"[exit code: {result.returncode}]")

        output = "\n".join(output_parts)

    except subprocess.TimeoutExpired:
        output = f"[ERROR] Command timed out after {timeout}s"
    except Exception as e:
        output = f"[ERROR] {type(e).__name__}: {e}"

    return _truncate_output(output)


def _truncate_output(text: str) -> str:
    """Truncate output, keeping first and last portions."""
    if len(text) <= MAX_OUTPUT_CHARS:
        return text

    half = MAX_OUTPUT_CHARS // 2
    truncated_msg = (
        f"\n\n[... truncated {len(text) - MAX_OUTPUT_CHARS} chars ...]\n\n"
    )
    return text[:half] + truncated_msg + text[-half:]


def _resolve_in_workspace(path: str, workspace: str) -> Path | None:
    """Resolve a path ensuring it stays within the workspace. Returns None if outside."""
    p = Path(path)
    if not p.is_absolute():
        p = Path(workspace) / p
    resolved = p.resolve()
    ws_resolved = Path(workspace).resolve()
    try:
        resolved.relative_to(ws_resolved)
    except ValueError:
        return None
    return resolved


def read_file(
    path: str,
    workspace: str,
    offset: int = 0,
    limit: int = 500,
) -> str:
    """Read a file from workspace, returning numbered lines."""
    p = _resolve_in_workspace(path, workspace)
    if p is None:
        return f"[ERROR] Path outside workspace: {path}"

    if not p.exists():
        return f"[ERROR] File not found: {p}"
    if not p.is_file():
        return f"[ERROR] Not a file: {p}"

    try:
        lines = p.read_text(errors="replace").splitlines()
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"

    total = len(lines)
    selected = lines[offset : offset + limit]
    numbered = [
        f"{i + offset + 1:>5} | {line}" for i, line in enumerate(selected)
    ]

    header = f"[{p.name}] lines {offset + 1}-{offset + len(selected)} of {total}"
    return header + "\n" + "\n".join(numbered)


def grep_files(
    pattern: str,
    workspace: str,
    path: str = ".",
    include: str | None = None,
) -> str:
    """Search workspace files via grep -rn."""
    # Validate search path stays within workspace
    resolved = _resolve_in_workspace(path, workspace)
    if resolved is None:
        return f"[ERROR] Path outside workspace: {path}"
    # Use resolved path relative to workspace for grep cwd
    try:
        search_path = str(resolved.relative_to(Path(workspace).resolve()))
    except ValueError:
        search_path = "."

    cmd = ["grep", "-rn", "--color=never"]
    if include:
        cmd.extend(["--include", include])
    cmd.append("--")
    cmd.append(pattern)
    cmd.append(search_path)

    try:
        result = subprocess.run(
            cmd,
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout or ""
        if result.returncode == 1 and not output:
            return "No matches found."
        if result.stderr:
            output += f"\n[stderr] {result.stderr}"
        return _truncate_output(output) if output else "No matches found."
    except subprocess.TimeoutExpired:
        return "[ERROR] grep timed out after 30s"
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"


def read_image_base64(path: str, workspace: str) -> tuple[str, str]:
    """Read an image file and return (base64_data, media_type)."""
    p = _resolve_in_workspace(path, workspace)
    if p is None:
        raise ValueError(f"Path outside workspace: {path}")

    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")

    suffix = p.suffix.lower()
    media_type_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    media_type = media_type_map.get(suffix)
    if media_type is None:
        raise ValueError(f"Unsupported image format: {suffix}")

    data = p.read_bytes()
    return base64.b64encode(data).decode("ascii"), media_type


# ---------------------------------------------------------------------------
# Web Search Proxy (for Anthropic provider — no built-in web search)
# ---------------------------------------------------------------------------


def _proxy_web_search(query: str, openai_client: Any | None = None) -> str:
    """Proxy a web search through GPT with web_search_preview.

    Used when the provider doesn't have built-in web search (e.g. Anthropic).
    Falls back to an error message if no OpenAI client is available.
    """
    if openai_client is None:
        return "[ERROR] Web search requires an OpenAI client for proxy. Not available."

    try:
        response = openai_client.responses.create(
            model="gpt-4.1-mini",
            tools=[{"type": "web_search_preview"}],
            input=f"Search the web for: {query}\nReturn the key facts you find.",
        )
        return response.output_text or "(no results)"
    except Exception as e:
        return f"[ERROR] Web search proxy failed: {e}"


# ---------------------------------------------------------------------------
# Tool Dispatch
# ---------------------------------------------------------------------------


def parse_tool_args(arguments: str) -> dict[str, Any]:
    """Parse tool call arguments from JSON string."""
    try:
        return json.loads(arguments) if arguments else {}
    except json.JSONDecodeError:
        return {}


def execute_tool(
    name: str,
    arguments: dict[str, Any],
    workspace: str,
    ask_user_fn: Callable[[str], str] | None = None,
    db: Any | None = None,
    openai_client: Any | None = None,
    adapter: Any | None = None,
) -> dict[str, Any]:
    """Execute a tool and return the result.

    Returns a dict with:
      - "output": str result for the API
      - "image": optional (base64, media_type) tuple for view_image
      - "done": True if report_to_user was called
    """
    if name == "shell_exec":
        command = arguments.get("command", "")
        timeout = arguments.get("timeout", DEFAULT_TIMEOUT)
        # Log shell commands to a single global log file
        import datetime
        log_path = Path(__file__).resolve().parent.parent.parent / "tool_call_log.log"
        try:
            with open(log_path, "a") as log_f:
                ts = datetime.datetime.now().isoformat(timespec="seconds")
                log_f.write(f"[{ts}] shell_exec | workspace={workspace} | {command}\n")
        except Exception:
            pass  # Don't let logging failures break execution
        output = execute_shell(command, workspace, timeout)
        return {"output": output}

    elif name == "view_image":
        path = arguments.get("path", "")
        try:
            b64_data, media_type = read_image_base64(path, workspace)
            return {
                "output": f"Image loaded successfully: {path}",
                "image": (b64_data, media_type),
            }
        except (FileNotFoundError, ValueError) as e:
            return {"output": f"[ERROR] {e}"}

    elif name == "ask_user":
        question = arguments.get("question", "")
        if ask_user_fn is not None:
            answer = ask_user_fn(question)
            return {"output": answer}
        return {"output": "[ERROR] ask_user is not available in this mode."}

    elif name == "report_to_user":
        summary = arguments.get("summary", "")
        return {"output": "Report delivered to user.", "done": True, "summary": summary}

    elif name == "read_file":
        path = arguments.get("path", "")
        offset = arguments.get("offset", 0)
        limit = arguments.get("limit", 500)
        output = read_file(path, workspace, offset, limit)
        return {"output": output}

    elif name == "grep_file":
        pattern = arguments.get("pattern", "")
        search_path = arguments.get("path", ".")
        include = arguments.get("include")
        output = grep_files(pattern, workspace, search_path, include)
        return {"output": output}

    # Phase 3 tools
    elif name == "propose_experiment":
        if db is None:
            return {"output": "[ERROR] Experiment database not available."}
        import re as _re
        exp_name = arguments.get("name", "")
        # Sanitize: alphanumeric, underscores, hyphens only — no path traversal
        exp_name = _re.sub(r"[^a-zA-Z0-9_\-]", "_", exp_name)[:80]
        if not exp_name:
            return {"output": "[ERROR] Invalid experiment name."}
        description = arguments.get("description", "")
        hypothesis = arguments.get("hypothesis", "")
        config = arguments.get("config", "{}")
        try:
            exp_id = db.create(exp_name, description, hypothesis, config)
        except Exception as e:
            return {"output": f"[ERROR] Failed to create experiment: {e}"}
        return {"output": f"Experiment #{exp_id} '{exp_name}' created (to_implement)."}

    elif name == "update_playbook":
        content = arguments.get("content", "")
        playbook_path = Path(workspace) / "playbook.md"
        playbook_path.write_text(content)
        return {"output": f"playbook.md updated ({len(content)} chars)."}

    elif name == "read_board":
        if db is None:
            return {"output": "[ERROR] Experiment database not available."}
        # Use adapter metric if available, default to sharpe
        _metric = "sharpe"
        _metric_display = "Sharpe"
        if adapter is not None:
            _metric = adapter.metric.primary_metric
            _metric_display = adapter.metric.display_name

        summary = db.board_summary()
        recent = db.list_all()[-10:]
        leaders = db.leaderboard(_metric, 10)

        lines = ["## Board Summary"]
        for col, cnt in sorted(summary.items()):
            lines.append(f"  {col}: {cnt}")

        lines.append("\n## Recent Experiments (last 10)")
        for exp in recent:
            metrics_str = ""
            if exp.results_json:
                try:
                    m = json.loads(exp.results_json)
                    parts = [f"{k}={v}" for k, v in m.items()]
                    metrics_str = f" [{', '.join(parts[:5])}]"
                except (json.JSONDecodeError, TypeError):
                    pass
            err = f" ERROR: {exp.error}" if exp.error else ""
            lines.append(
                f"  #{exp.id} {exp.name} [{exp.status}]{metrics_str}{err}"
            )

        lines.append(f"\n## Leaderboard (by {_metric_display})")
        for i, exp in enumerate(leaders, 1):
            try:
                m = json.loads(exp.results_json or "{}")
                val = m.get(_metric, "?")
            except (json.JSONDecodeError, TypeError):
                val = "?"
            lines.append(f"  {i}. #{exp.id} {exp.name} — {_metric_display}: {val}")

        return {"output": "\n".join(lines)}

    elif name == "update_experiment":
        if db is None:
            return {"output": "[ERROR] Experiment database not available."}
        exp_id = arguments.get("experiment_id", 0)
        status = arguments.get("status")
        results = arguments.get("results")
        error = arguments.get("error")
        debrief_path = arguments.get("debrief_path")

        # Verify experiment exists
        exp = db.get(exp_id)
        if exp is None:
            return {"output": f"[ERROR] Experiment #{exp_id} not found."}

        updates: list[str] = []
        if results:
            db.set_results(exp_id, results)
            updates.append("results set")
        if error:
            db.set_error(exp_id, error)
            updates.append("error set")
        # Single update_status call with all kwargs
        status_kwargs = {}
        if debrief_path:
            status_kwargs["debrief_path"] = debrief_path
            updates.append(f"debrief_path={debrief_path}")
        if status or status_kwargs:
            db.update_status(exp_id, status or exp.status, **status_kwargs)
            if status:
                updates.append(f"status={status}")

        return {"output": f"Experiment #{exp_id} updated: {', '.join(updates) or 'no changes'}."}

    elif name == "reality_check":
        experiment_name = arguments.get("experiment_name", "")
        if not experiment_name:
            return {"output": "[ERROR] experiment_name is required."}

        try:
            from alpha_lab.validation import run_reality_check, save_validation_report
        except ImportError as e:
            return {"output": f"[ERROR] Could not import validation module: {e}"}

        # Load time limit from workspace config
        time_limit_seconds = None
        try:
            import yaml
            workspace_path = Path(workspace).resolve()

            config_paths = [
                workspace_path / "config.json",
                workspace_path.parent / "data" / "exchange_config.json",
                workspace_path.parent / "data" / "config.json",
                workspace_path.parent.parent / "data" / "exchange_config.json",
            ]

            for config_path in config_paths:
                if config_path.exists():
                    with open(config_path) as f:
                        if config_path.suffix == ".json":
                            config_data = json.load(f)
                        else:
                            config_data = yaml.safe_load(f)

                        pipeline_config = config_data.get("pipeline", {})
                        if isinstance(pipeline_config, dict):
                            phase3_config = pipeline_config.get("phase3", {})
                            if isinstance(phase3_config, dict):
                                time_limit_seconds = phase3_config.get("time_limit_seconds")
                                if time_limit_seconds:
                                    break
        except Exception:
            pass

        experiment_dir = Path(workspace) / "experiments" / experiment_name
        if not experiment_dir.exists():
            return {"output": f"[ERROR] Experiment directory not found: {experiment_dir}"}

        try:
            report = run_reality_check(
                experiment_dir=experiment_dir,
                workspace=Path(workspace),
                time_limit_seconds=time_limit_seconds,
            )

            save_validation_report(report, experiment_dir)

            return {"output": report.format()}
        except Exception as e:
            import traceback as tb_module
            tb = tb_module.format_exc()
            return {"output": f"[ERROR] Reality check failed: {e}\n\n{tb}"}

    elif name == "cancel_experiments":
        if db is None:
            return {"output": "[ERROR] Experiment database not available."}
        exp_ids = arguments.get("experiment_ids", [])
        reason = arguments.get("reason", "No reason provided")

        cancelled = []
        skipped = []
        for exp_id in exp_ids:
            exp = db.get(exp_id)
            if exp is None:
                skipped.append(f"#{exp_id} (not found)")
            elif exp.status != "to_implement":
                skipped.append(f"#{exp_id} {exp.name} (status={exp.status}, can only cancel to_implement)")
            else:
                db.update_status(exp_id, "cancelled")
                db.set_error(exp_id, f"Cancelled by strategist: {reason}")
                cancelled.append(f"#{exp_id} {exp.name}")

        lines = []
        if cancelled:
            lines.append(f"Cancelled {len(cancelled)} experiments: {', '.join(cancelled)}")
        if skipped:
            lines.append(f"Skipped {len(skipped)}: {', '.join(skipped)}")
        if not lines:
            lines.append("No experiments to cancel.")
        return {"output": "\n".join(lines)}

    elif name == "web_search":
        query = arguments.get("query", "")
        if not query:
            return {"output": "[ERROR] No search query provided."}
        output = _proxy_web_search(query, openai_client)
        return {"output": output}

    # Adapter tools (Phase 0 + Supervisor)
    elif name == "write_adapter_file":
        from alpha_lab.adapter import ADAPTER_FILES
        filename = arguments.get("filename", "")
        content = arguments.get("content", "")
        if filename not in ADAPTER_FILES:
            return {"output": f"[ERROR] Invalid adapter filename: {filename}. Allowed: {ADAPTER_FILES}"}
        adapter_dir = Path(workspace) / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        (adapter_dir / filename).write_text(content)
        return {"output": f"Wrote adapter/{filename} ({len(content)} chars)."}

    elif name == "read_reference_adapter":
        ref_name = arguments.get("name", "")
        try:
            from alpha_lab.adapter_loader import load_builtin_adapter
            ref = load_builtin_adapter(ref_name)
        except FileNotFoundError as e:
            return {"output": f"[ERROR] {e}"}
        parts = [f"# Reference adapter: {ref_name}\n"]
        parts.append(f"## manifest.json\ndomain_name: {ref.domain_name}")
        parts.append(f"domain_description: {ref.domain_description}")
        parts.append(f"metric: {ref.metric.primary_metric} ({ref.metric.direction})")
        parts.append(f"required_files: {ref.experiment.required_files}")
        parts.append(f"framework_dir: {ref.experiment.framework_dir}")
        for key, prompt_text in ref.prompts.items():
            # Truncate long prompts
            truncated = prompt_text[:3000] + "..." if len(prompt_text) > 3000 else prompt_text
            parts.append(f"\n## {key}.md\n{truncated}")
        if ref.domain_knowledge:
            dk = ref.domain_knowledge[:3000] + "..." if len(ref.domain_knowledge) > 3000 else ref.domain_knowledge
            parts.append(f"\n## domain_knowledge.md\n{dk}")
        return {"output": "\n".join(parts)}

    elif name == "read_adapter":
        adapter_dir = Path(workspace) / "adapter"
        if not adapter_dir.is_dir():
            return {"output": "[ERROR] No adapter directory in workspace."}
        parts = []
        for f in sorted(adapter_dir.iterdir()):
            if f.is_file():
                content = f.read_text()
                truncated = content[:3000] + "..." if len(content) > 3000 else content
                parts.append(f"## {f.name}\n{truncated}")
        return {"output": "\n".join(parts) if parts else "Adapter directory is empty."}

    elif name == "patch_adapter_file":
        from alpha_lab.adapter import ADAPTER_FILES
        filename = arguments.get("filename", "")
        content = arguments.get("content", "")
        reason = arguments.get("reason", "no reason")
        if filename not in ADAPTER_FILES:
            return {"output": f"[ERROR] Invalid adapter filename: {filename}. Allowed: {ADAPTER_FILES}"}
        adapter_dir = Path(workspace) / "adapter"
        if not adapter_dir.is_dir():
            return {"output": "[ERROR] No adapter directory to patch."}
        target = adapter_dir / filename
        old_size = target.stat().st_size if target.exists() else 0
        # Git checkpoint before patching
        import subprocess as _sp
        try:
            _sp.run(
                "git add -A && git commit -m 'checkpoint before supervisor patch' --allow-empty",
                shell=True, cwd=workspace, capture_output=True, timeout=30,
            )
        except Exception:
            pass  # Best-effort checkpoint
        target.write_text(content)
        return {
            "output": (
                f"Patched adapter/{filename}: {old_size} -> {len(content)} chars. "
                f"Reason: {reason}"
            )
        }

    else:
        return {"output": f"[ERROR] Unknown tool: {name}"}
