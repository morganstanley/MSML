"""Tool implementations and dispatch for alpha-lab."""

from __future__ import annotations

import base64
import json
import os
import signal
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from alpha_lab import deps, utils
from alpha_lab.adapter_loader import _resolve_adapter_path, load_adapter
from alpha_lab.experiment_db import KANBAN_COLUMNS
from alpha_lab.memory import MemoryStore

# Agent-allowed status transitions, keyed on current row status.
_AGENT_ALLOWED_TRANSITIONS: dict[str, tuple[str, ...]] = {
    "to_implement": ("to_implement", "implemented", "cancelled"),
    "implemented": ("implemented", "checked", "cancelled"),
    "finished": ("finished", "checked", "analyzed", "cancelled"),
    "analyzed": ("analyzed", "done", "cancelled"),
}


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

MAX_OUTPUT_CHARS = 30_000
DEFAULT_TIMEOUT = 300


# ---------------------------------------------------------------------------
# Tool Implementations
# ---------------------------------------------------------------------------


def execute_shell(
    command: str,
    workspace: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    """Execute a shell command in the workspace directory."""
    timeout = max(timeout, 1)

    try:
        proc = subprocess.Popen(
            command,
            shell=True,
            cwd=workspace,
            env={**os.environ, "ALPHALAB_WORKSPACE": str(Path(workspace).resolve())},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Kill entire process group (shell + all children). Process may
            # race to exit between timeout and kill — ProcessLookupError
            # (subclass of OSError) just means it's already gone, so treat
            # as success instead of bubbling up a confusing error at the
            # outer `except Exception` below.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
            # Bound the post-kill wait so an uninterruptible I/O child can't
            # hang the agent loop forever; after SIGKILL the reap should be
            # near-instant, so 5 s is more than enough.
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                return _truncate_output(
                    f"[ERROR] Command timed out after {timeout}s and did not exit after SIGKILL"
                )
            return _truncate_output(f"[ERROR] Command timed out after {timeout}s")

        output_parts = []
        if stdout:
            output_parts.append(stdout)
        if stderr:
            output_parts.append(f"[stderr]\n{stderr}")
        output_parts.append(f"[exit code: {proc.returncode}]")

        output = "\n".join(output_parts)

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
# Web Search Proxy (for Bedrock provider — no built-in web search)
# ---------------------------------------------------------------------------


def _proxy_web_search(query: str, openai_client: Any | None = None) -> str:
    """Proxy a web search through GPT with web_search_preview.

    Used when the provider doesn't have built-in web search (e.g. Bedrock).
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


def _open_memory_store(memory_store: Any | None, workspace: str) -> Any:
    """The injected memory store or a fresh in-process one when none was injected."""
    if memory_store is not None:
        return memory_store
    return MemoryStore(workspace)


def execute_tool(
    name: str,
    arguments: dict[str, Any],
    workspace: str,
    ask_user_fn: Callable[[str], str] | None = None,
    db: Any | None = None,
    openai_client: Any | None = None,
    adapter: Any | None = None,
    shell_timeout: int = DEFAULT_TIMEOUT,
    memory_store: Any | None = None,
) -> dict[str, Any]:
    """Execute a tool and return the result.

    Returns a dict with:
      - "output": str result for the API
      - "image": optional (base64, media_type) tuple for view_image
      - "done": True if report_to_user was called
    """
    if name == "shell_exec":
        command = arguments.get("command", "")
        # shell_timeout is the operator-configured ceiling. It's typed int
        # in Python but comes from config — JSON/YAML loads can produce
        # None or strings — so normalize defensively before min(...).
        try:
            shell_timeout = int(shell_timeout) if shell_timeout is not None else DEFAULT_TIMEOUT
        except (TypeError, ValueError):
            shell_timeout = DEFAULT_TIMEOUT
        if shell_timeout <= 0:
            shell_timeout = DEFAULT_TIMEOUT
        # The LLM may request a smaller timeout per-call but not exceed the cap.
        requested = arguments.get("timeout", shell_timeout)
        try:
            requested = int(requested)
        except (TypeError, ValueError):
            requested = shell_timeout
        timeout = min(requested, shell_timeout) if requested > 0 else shell_timeout
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
        # Resource validation (always). `config.resource` is required and must be an
        # exactly-matching enabled device type. In JIT mode it must also have a free slot.
        # Misuse fails loud back to the agent (the update_experiment pattern) so it can
        # re-propose. Runs before db.create (and MLflow attach); slot_states is recomputed
        # per call, so successive proposals in one turn tighten the count.
        d = deps.get()  # no active run ⇒ misuse; fail loud
        try:
            parsed = json.loads(config or "{}")
        except (json.JSONDecodeError, TypeError):
            return {"output": "[ERROR] propose_experiment: `config` must be valid JSON."}
        if not isinstance(parsed, dict):
            return {"output": "[ERROR] propose_experiment: `config` must be a JSON object."}

        states = utils.slot_states(db)
        enabled = list(states)
        resource = parsed.get("resource")
        if not resource:
            return {"output": (
                f"[ERROR] propose_experiment: `config.resource` is required; "
                f"must be exactly one of {enabled}."
            )}
        if resource not in enabled:
            return {"output": (
                f"[ERROR] propose_experiment: `config.resource` must be exactly "
                f"one of {enabled}; got {resource!r}."
            )}
        if d.config.pipeline.phase3.jit and states[resource]["free"] <= 0:
            free_types = [t for t in enabled if states[t]["free"] > 0]
            if free_types:
                return {"output": (
                    f"[ERROR] propose_experiment: no free {resource} slot. Propose for a "
                    f"resource with free slots instead: {free_types}."
                )}
            return {"output": (
                "[ERROR] propose_experiment: no free slots on any resource; "
                "stop proposing this turn."
            )}
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
        from alpha_lab.experiment_db import is_smoke_result
        if adapter is None:
            raise RuntimeError("adapter is required for leaderboard/metric resolution")
        _metric = adapter.metric.primary_metric
        _metric_display = adapter.metric.display_name
        _direction = adapter.metric.direction

        summary = db.board_summary()
        recent = db.list_all()[-10:]
        leaders = db.leaderboard(_metric, 10, _direction)

        lines = ["## Board Summary"]
        for col, cnt in sorted(summary.items()):
            lines.append(f"  {col}: {cnt}")

        lines.append("\n## Recent Experiments (last 10)")
        for exp in recent:
            smoke = is_smoke_result(exp.results_json)
            metrics_str = ""
            if smoke:
                metrics_str = " [SMOKE — metrics redacted]"
            elif exp.results_json:
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

        # Transition gate: agents may only transition out of a transitional
        # state they currently own. Other status writes are rejected.
        if status:
            if status not in KANBAN_COLUMNS:
                msg = f"update_experiment: '{status}' is not a valid status."
                return {"output": f"[ERROR] {msg}"}
            current = exp.status
            allowed = _AGENT_ALLOWED_TRANSITIONS.get(current, ())
            if status not in allowed:
                allowed_str = ", ".join(allowed) or "(none)"
                msg = (
                    f"update_experiment: cannot transition #{exp_id} from "
                    f"'{current}' to '{status}'. Allowed from "
                    f"'{current}': {allowed_str}."
                )
                return {"output": f"[ERROR] {msg}"}

        # Smoke-test gate: detect smoke_test flag in results JSON and tag it
        smoke_warning = ""
        if results:
            from alpha_lab.experiment_db import is_smoke_result
            if is_smoke_result(results):
                # Inject _smoke_flagged as audit metadata
                try:
                    parsed = json.loads(results)
                    if isinstance(parsed, dict):
                        parsed["_smoke_flagged"] = True
                        results = json.dumps(parsed)
                except (json.JSONDecodeError, TypeError):
                    pass
                smoke_warning = (
                    " [WARNING] These results contain smoke_test=true "
                    "and will be excluded from the leaderboard. "
                    "Metrics from smoke tests are not decision-grade — "
                    "only full GPU run metrics are comparable."
                )

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

        # MLflow side-effects on the experiment's sub-run. No-op when MLflow
        # is off. Re-fetch the experiment so we see the latest debrief_path
        # populated by the updates above.
        if results or error or status:
            terminal_set = {"analyzed", "done", "cancelled", "finished"}
            terminal = status if status in terminal_set else None
            _log_experiment_results_to_mlflow(
                db=db,
                exp=db.get(exp_id),
                results_json=results if results else None,
                error=error if error else None,
                workspace=workspace,
                terminal_status=terminal,
            )

        return {"output": f"Experiment #{exp_id} updated: {', '.join(updates) or 'no changes'}.{smoke_warning}"}

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
                if getattr(exp, "mlflow_run_uuid", None):
                    from alpha_lab import mlflow_logger
                    mlflow_logger.terminate_run(exp.mlflow_run_uuid, status="KILLED")
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
            ref = load_adapter(_resolve_adapter_path(ref_name))
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
        target.write_text(content)
        return {
            "output": (
                f"Patched adapter/{filename}: {old_size} -> {len(content)} chars. "
                f"Reason: {reason}"
            )
        }

    # ------------------------------------------------------------------
    # Memory tools
    # ------------------------------------------------------------------

    elif name == "memory_store":
        store = _open_memory_store(memory_store, workspace)
        entry_id = store.store(
            content=arguments.get("content", ""),
            tags=arguments.get("tags", []),
            summary=arguments.get("summary", ""),
            kind=arguments.get("kind"),
            phase=arguments.get("phase"),
            agent=arguments.get("agent"),
            run_id=arguments.get("run_id"),
            source_path=arguments.get("source_path"),
        )
        return {"output": f"Memory #{entry_id} stored."}

    elif name == "memory_search":
        store = _open_memory_store(memory_store, workspace)
        results = store.search(
            query=arguments.get("query", ""),
            tags=arguments.get("tags"),
            limit=arguments.get("limit", 10),
            kind=arguments.get("kind"),
            phase=arguments.get("phase"),
        )
        if not results:
            return {"output": "No matching memories found."}
        lines = [f"Found {len(results)} memories:"]
        for entry in results:
            meta_parts = []
            if entry.kind:
                meta_parts.append(f"kind={entry.kind}")
            if entry.phase:
                meta_parts.append(f"phase={entry.phase}")
            if entry.agent:
                meta_parts.append(f"agent={entry.agent}")
            if entry.run_id:
                meta_parts.append(f"run={entry.run_id}")
            if entry.source_path:
                meta_parts.append(f"source={entry.source_path}")
            label_parts = []
            if entry.tags:
                label_parts.append(", ".join(entry.tags))
            if meta_parts:
                label_parts.append("; ".join(meta_parts))
            label = f" [{'; '.join(label_parts)}]" if label_parts else ""
            lines.append(f"  #{entry.id}{label} {entry.summary}")
        return {"output": "\n".join(lines)}

    elif name == "memory_read":
        store = _open_memory_store(memory_store, workspace)
        content = store.read(arguments.get("memory_id", 0))
        return {"output": content}

    else:
        return {"output": f"[ERROR] Unknown tool: {name}"}


# ---------------------------------------------------------------------------
# MLflow integration helpers — no-op when MLflow is off
# ---------------------------------------------------------------------------


def _log_experiment_results_to_mlflow(
    *,
    db: Any,
    exp: Any,
    results_json: str | None,
    error: str | None,
    workspace: str,
    terminal_status: str | None = None,
) -> None:
    """Log metrics + artifacts to an experiment's MLflow sub-run.

    Called whenever ``update_experiment`` sets results/error/status. No-op
    when MLflow is disabled or the experiment has no associated sub-run.
    """
    if exp is None or not getattr(exp, "mlflow_run_uuid", None):
        return
    from alpha_lab import mlflow_logger

    run_uuid = exp.mlflow_run_uuid

    if results_json:
        try:
            parsed = json.loads(results_json)
        except (json.JSONDecodeError, TypeError):
            parsed = None
        if isinstance(parsed, dict):
            metrics = {
                k: float(v) for k, v in parsed.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
            if metrics:
                mlflow_logger.log_run_metrics(run_uuid, metrics)
            non_metric_params = {
                k: v for k, v in parsed.items()
                if not (isinstance(v, (int, float)) and not isinstance(v, bool))
            }
            if non_metric_params:
                mlflow_logger.log_run_params(run_uuid, non_metric_params)

    if error:
        mlflow_logger.log_run_params(run_uuid, {"error": error})

    exp_dir = Path(workspace) / "experiments" / exp.name
    if exp_dir.is_dir():
        mlflow_logger.log_run_artifacts_dir(run_uuid, exp_dir)
    if exp.debrief_path:
        debrief_p = Path(workspace) / exp.debrief_path
        if debrief_p.is_file():
            mlflow_logger.log_run_artifact(
                run_uuid, debrief_p,
                artifact_path=f"debrief/{debrief_p.name}",
            )

    if terminal_status:
        status_map = {
            "done": "FINISHED",
            "analyzed": "FINISHED",
            "cancelled": "KILLED",
            "finished": "FAILED" if error else "FINISHED",
        }
        if mlflow_status := status_map.get(terminal_status):
            mlflow_logger.terminate_run(run_uuid, status=mlflow_status)
