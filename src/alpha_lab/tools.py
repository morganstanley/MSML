"""Tool schemas and implementations for alpha-lab."""

from __future__ import annotations

import atexit
import base64
import ctypes
import ctypes.util
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Subprocess lifecycle: kill children when the parent dies.
#
# Two layers of belt-and-suspenders here, because `start_new_session=True`
# (set on every shell_exec subprocess so we can `killpg` on timeout) also
# decouples children from the parent's death — without intervention they
# become orphans that keep consuming CPU/GPU/writing to artifacts/ after
# `run.py` exits or is killed.
#
# Layer 1 (per-child, Linux): prctl(PR_SET_PDEATHSIG, SIGKILL) in the
# pre-exec hook so the kernel kills the child as soon as the parent dies,
# even on `kill -9` of the parent.
#
# Layer 2 (process-wide): a module-level registry of live subprocesses plus
# an atexit hook that killpg's anything still running on a clean exit.
# Belt-and-suspenders for non-Linux platforms or when prctl is unavailable.
# ---------------------------------------------------------------------------

_PR_SET_PDEATHSIG = 1  # from <linux/prctl.h>

_LIVE_SUBPROCESSES: set[subprocess.Popen] = set()
_LIVE_LOCK = threading.Lock()


def _preexec_setup() -> None:
    """Run in the child between fork and exec.

    - Put the child into a new process group so the parent can ``killpg`` on
      timeout without nuking its own group.
    - Ask the kernel to SIGKILL the child the moment the parent dies (Linux
      only; silently no-op elsewhere).
    """
    try:
        os.setpgrp()
    except Exception:
        pass
    if sys.platform.startswith("linux"):
        try:
            libc_name = ctypes.util.find_library("c") or "libc.so.6"
            libc = ctypes.CDLL(libc_name, use_errno=True)
            libc.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
        except Exception:
            pass


def _register_subprocess(proc: subprocess.Popen) -> None:
    with _LIVE_LOCK:
        _LIVE_SUBPROCESSES.add(proc)


def _unregister_subprocess(proc: subprocess.Popen) -> None:
    with _LIVE_LOCK:
        _LIVE_SUBPROCESSES.discard(proc)


def _kill_all_live_subprocesses() -> None:
    """Best-effort: SIGKILL every live shell_exec child's process group.

    Registered as an atexit hook so a clean `sys.exit` from `run.py` cleans
    up everything it spawned, even if a tool happens to be mid-`communicate`
    on another thread.
    """
    with _LIVE_LOCK:
        procs = list(_LIVE_SUBPROCESSES)
    for proc in procs:
        try:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass


atexit.register(_kill_all_live_subprocesses)


_TERM_HANDLERS_INSTALLED = False


def install_termination_handlers() -> None:
    """Install SIGTERM/SIGINT handlers that kill all live shell_exec children.

    Call this once from ``run.py`` at startup. Without it, SIGTERM bypasses
    Python's ``atexit`` machinery, leaving subprocess groups orphaned (still
    running, still holding GPUs, still writing to ``artifacts/``) after the
    parent dies. Idempotent.
    """
    global _TERM_HANDLERS_INSTALLED
    if _TERM_HANDLERS_INSTALLED:
        return
    _TERM_HANDLERS_INSTALLED = True

    def _handler(signum, frame):  # noqa: ARG001 — signal API
        try:
            _kill_all_live_subprocesses()
        finally:
            # 128 + signum is the POSIX convention for "killed by signal N".
            os._exit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            # Some signals can't be caught on some platforms / in some
            # threads — best-effort, skip silently.
            pass


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
                    "description": (
                        "Timeout in seconds. Capped at the task's "
                        "shell_timeout (TaskConfig.shell_timeout; "
                        "default 300). Raise this value (up to shell_timeout) "
                        "for data-heavy scripts; to exceed the cap, increase "
                        "shell_timeout in the task config."
                    ),
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
    "propose_variant": {
        "type": "function",
        "name": "propose_variant",
        "description": (
            "Spawn a VARIANT of an existing experiment by copying its entire "
            "directory (code, config) into a new experiment directory. "
            "Use this when an existing experiment is promising and you want "
            "to vary it (e.g. hyperparameter sweep, small architectural "
            "tweak) without re-implementing from scratch. The implementer "
            "for the variant edits only the files that need changing — "
            "everything else is inherited from the base. "
            "Variants are capped per base (max_variants_per_base in config). "
            "The base experiment's directory is preserved untouched; the "
            "variant's directory starts as an exact copy minus `results/` "
            "and `logs/` (those are the base's outputs, not the variant's). "
            "Write a clear `what_changes` describing the intended diff — "
            "the implementer reads it from `.variant_intent.md` and patches "
            "the base code accordingly. For genuinely novel experiments "
            "that don't share code with anything existing, use "
            "`propose_experiment` instead."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "base_experiment_id": {
                    "type": "integer",
                    "description": (
                        "The id of the existing experiment to clone. Must "
                        "have a directory at experiments/<base_name>/ with "
                        "the canonical code files in it."
                    ),
                },
                "name": {
                    "type": "string",
                    "description": (
                        "Short unique name for the variant (used as "
                        "directory name). Use snake_case. Convention: "
                        "include the base's name + the change "
                        "(e.g. 'tft_baseline_hidden_128')."
                    ),
                },
                "hypothesis": {
                    "type": "string",
                    "description": (
                        "Why this variant might do better than the base. "
                        "Cite the base by id and what the variant changes."
                    ),
                },
                "what_changes": {
                    "type": "string",
                    "description": (
                        "Concrete description of the diff the implementer "
                        "should apply: which file(s) to edit, which "
                        "hyperparameters or constants to change, from "
                        "what value to what value. Plain English is fine; "
                        "this is read by an LLM, not parsed."
                    ),
                },
            },
            "required": [
                "base_experiment_id", "name", "hypothesis", "what_changes",
            ],
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
                    "description": (
                        "Error message if the experiment failed. Pass an "
                        "empty string ('') to CLEAR a stale error after a "
                        "successful retry — without this the GUI keeps "
                        "showing the row as failed. A transition to "
                        "``analyzed`` or ``done`` with new ``results`` "
                        "auto-clears the error field, so the empty-string "
                        "form is only needed when you want to clear "
                        "without also moving status."
                    ),
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
    "memory_store": {
        "type": "function",
        "name": "memory_store",
        "description": (
            "Store a piece of knowledge in persistent memory. Use this to save "
            "important findings, data insights, experiment results, or decisions "
            "that future agents should know about. Include relevant tags for searchability."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The full content to store.",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags for categorization (e.g. ['data_quality', 'phase1']).",
                },
                "summary": {
                    "type": "string",
                    "description": "One-line summary for search results.",
                },
            },
            "required": ["content", "tags", "summary"],
            "additionalProperties": False,
        },
    },
    "memory_search": {
        "type": "function",
        "name": "memory_search",
        "description": (
            "Search persistent memory for relevant knowledge from previous agents/phases. "
            "Returns summaries of matching entries. Use memory_read to get full content."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keywords.",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional: filter by tags.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default 10).",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    "memory_read": {
        "type": "function",
        "name": "memory_read",
        "description": (
            "Read the full content of a specific memory entry by ID. "
            "Use memory_search first to find relevant entry IDs."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "integer",
                    "description": "The memory entry ID.",
                },
            },
            "required": ["memory_id"],
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

    # -----------------------------------------------------------------------
    # Conductor tools. Each mutation tool requires `reason` and `evidence`
    # strings — the agent's prompt teaches it that decisions without evidence
    # look bad in the audit log. We don't reject empty evidence at the tool
    # layer (per the design: prompt-level enforcement, not tool-level).
    # -----------------------------------------------------------------------

    "park_experiment": {
        "type": "function",
        "name": "park_experiment",
        "description": (
            "Soft-cancel an experiment so the dispatcher skips it. Reversible "
            "via unpark_experiment. The DB row is preserved; only parked_at is set."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "experiment_id": {"type": "integer"},
                "reason": {"type": "string"},
                "evidence": {"type": "string"},
            },
            "required": ["experiment_id", "reason", "evidence"],
            "additionalProperties": False,
        },
    },

    "unpark_experiment": {
        "type": "function",
        "name": "unpark_experiment",
        "description": "Restore a parked experiment to the active queue.",
        "parameters": {
            "type": "object",
            "properties": {
                "experiment_id": {"type": "integer"},
                "reason": {"type": "string"},
                "evidence": {"type": "string"},
            },
            "required": ["experiment_id", "reason", "evidence"],
            "additionalProperties": False,
        },
    },

    "set_priority": {
        "type": "function",
        "name": "set_priority",
        "description": (
            "Override an experiment's queue priority. Higher runs first; "
            "ties broken by created_at ASC. Use small numbers (typically -10 to 10)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "experiment_id": {"type": "integer"},
                "priority": {"type": "integer"},
                "reason": {"type": "string"},
                "evidence": {"type": "string"},
            },
            "required": ["experiment_id", "priority", "reason", "evidence"],
            "additionalProperties": False,
        },
    },

    "clear_experiment_block": {
        "type": "function",
        "name": "clear_experiment_block",
        "description": (
            "Clear a STALE `blocked:` error on a to_implement/implemented row so "
            "the dispatcher can assign it again — without cloning it or changing "
            "its economics. Use ONLY when a transient precondition a worker "
            "recorded as `blocked:` has since been resolved (e.g. a shared fixture "
            "now passes) and the row is self-stranded: the dispatcher never assigns "
            "a blocked row, so the flag never clears itself, and the worker that "
            "could clear it is never assigned. Do NOT use on a permanent block "
            "such as `blocked: superseded by #<id>`. No-op (and reports so) if the "
            "row is not currently in a blocked state. Record why in the meta log "
            "via the reason/evidence fields."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "experiment_id": {"type": "integer"},
                "reason": {"type": "string"},
                "evidence": {"type": "string"},
            },
            "required": ["experiment_id", "reason", "evidence"],
            "additionalProperties": False,
        },
    },

    "annotate_experiment": {
        "type": "function",
        "name": "annotate_experiment",
        "description": (
            "Apply a leaderboard label to an experiment. Labels: champion, "
            "control, challenger, exploration, exploitation, ensemble-candidate, "
            "home-run-attempt, quarantined, quarantined_leakage, "
            "quarantined_invalid_split, quarantined_zombie. Set label='' to clear."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "experiment_id": {"type": "integer"},
                "label": {"type": "string"},
                "reason": {"type": "string"},
                "evidence": {"type": "string"},
            },
            "required": ["experiment_id", "label", "reason", "evidence"],
            "additionalProperties": False,
        },
    },

    "issue_directive": {
        "type": "function",
        "name": "issue_directive",
        "description": (
            "Append a directive to meta/directives.md for a target role: "
            "strategist, worker, reporter, supervisor, or all. Other agents "
            "read this file at the top of their next turn. "
            "Scope controls how same-role agents share the directive: "
            "'standing' (default) applies to every action of the target role "
            "indefinitely; 'one-shot' is claimed by the first same-role "
            "agent that acks it and skipped by the rest; "
            "'per-experiment:<id>' targets one specific experiment row. "
            "Choose 'standing' for per-action policies ('include cold-client "
            "slice in every debrief'); 'one-shot' for tasks that should "
            "happen exactly once across all same-role actors ('propose 3 "
            "new experiments', 'write a leaderboard CSV')."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "target_role": {"type": "string"},
                "message": {"type": "string"},
                "reason": {"type": "string"},
                "evidence": {"type": "string"},
                "scope": {"type": "string"},
            },
            "required": ["target_role", "message", "reason", "evidence"],
            "additionalProperties": False,
        },
    },

    "retire_directive": {
        "type": "function",
        "name": "retire_directive",
        "description": (
            "Explicitly retire a previously-issued directive so it stops "
            "being injected into downstream agents' prompts. Use when a "
            "directive is no longer applicable (phase moved on, mechanism "
            "deprecated, superseded by a newer directive, payoff already "
            "realized). REQUIRED for any directive that becomes "
            "irrelevant — directives never expire on their own; the "
            "Conductor owns the lifecycle. To replace a directive with "
            "a revised version, issue the new one (different id) AND "
            "retire the old one in the same turn."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "directive_id": {"type": "string"},
                "reason": {
                    "type": "string",
                    "description": "Why this directive is retired. Required.",
                },
            },
            "required": ["directive_id", "reason"],
            "additionalProperties": False,
        },
    },

    "ack_directive": {
        "type": "function",
        "name": "ack_directive",
        "description": (
            "Acknowledge that you have acted on a one-shot or per-experiment "
            "Conductor directive. Appends an entry to "
            "meta/directive_acks.jsonl that future same-role agents will see, "
            "so they skip the directive instead of duplicating your work. "
            "Use this immediately after acting on a one-shot directive. "
            "STANDING-scope directives do NOT need to be acked — every "
            "same-role agent honors them independently for their own work."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "directive_id": {"type": "string"},
                "action_taken": {"type": "string"},
            },
            "required": ["directive_id", "action_taken"],
            "additionalProperties": False,
        },
    },

    "write_note_to_user": {
        "type": "function",
        "name": "write_note_to_user",
        "description": (
            "Append a note to meta/notes_to_user.md. The user is not expected "
            "to read these promptly — it is a record of what you observed and "
            "concluded over the run."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
            },
            "required": ["message"],
            "additionalProperties": False,
        },
    },

    "set_throttle": {
        "type": "function",
        "name": "set_throttle",
        "description": (
            "Set system throttle for cpu or gpu submissions. Levels: none "
            "(default), slow (halve new-submit capacity, rounded down to >=1), "
            "halt-new (let in-flight finish but stop new launches). Pass only "
            "the dimension you want to change; the other is preserved."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "gpu": {"type": "string"},
                "cpu": {"type": "string"},
                "reason": {"type": "string"},
                "evidence": {"type": "string"},
            },
            "required": ["reason", "evidence"],
            "additionalProperties": False,
        },
    },

    "read_meta_log": {
        "type": "function",
        "name": "read_meta_log",
        "description": (
            "Read your previous Conductor decisions from meta/meta_log.jsonl. "
            "Returns the most recent last_n entries; with sample_older=true, "
            "additionally returns up to 10 stratified-sampled older entries "
            "for retrospective audit."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "last_n": {"type": "integer"},
                "sample_older": {"type": "boolean"},
            },
            "required": [],
            "additionalProperties": False,
        },
    },

    "read_user_instructions": {
        "type": "function",
        "name": "read_user_instructions",
        "description": (
            "Read meta/instructions/from_user.md and indicate whether the "
            "content has changed since you last marked it seen. Returns "
            "{content, is_new}."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "mark_seen": {"type": "boolean"},
            },
            "required": [],
            "additionalProperties": False,
        },
    },

    "ack_user_instruction": {
        "type": "function",
        "name": "ack_user_instruction",
        "description": (
            "Append an acknowledgement to meta/instructions/ack.md after "
            "reading a (new) user instruction in from_user.md. The "
            "message should restate in your own words what the user asked "
            "for AND what directives/throttles/annotations/rewinds you "
            "applied in response. The user reads this file when they "
            "want to confirm that you understood their instruction; the "
            "system never blocks on it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
            },
            "required": ["message"],
            "additionalProperties": False,
        },
    },

    "read_system_load": {
        "type": "function",
        "name": "read_system_load",
        "description": (
            "Single-line system-load summary: CPU load avg, free disk, GPU "
            "utilization (best-effort), and current throttle state."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },

    "peek_experiment_log": {
        "type": "function",
        "name": "peek_experiment_log",
        "description": (
            "Read the last N lines of an in-flight experiment's subprocess "
            "output log. Use this to inspect learning curves, error tails, "
            "or runtime progress. Default 200 lines, max 2000."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "experiment_name": {"type": "string"},
                "last_n_lines": {"type": "integer"},
            },
            "required": ["experiment_name"],
            "additionalProperties": False,
        },
    },

    "kill_experiment": {
        "type": "function",
        "name": "kill_experiment",
        "description": (
            "Cancel an in-flight experiment. Implementation: parks the row "
            "(preventing reassignment) and writes a kill-request marker the "
            "dispatcher reads to terminate the subprocess. Reserved for cases "
            "with hard evidence — see your prompt's Rules section."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "experiment_id": {"type": "integer"},
                "reason": {"type": "string"},
                "evidence": {"type": "string"},
            },
            "required": ["experiment_id", "reason", "evidence"],
            "additionalProperties": False,
        },
    },

    "delete_path": {
        "type": "function",
        "name": "delete_path",
        "description": (
            "Delete a workspace path AFTER backing it up to meta/backups/<ts>/. "
            "The user can restore from the backup. Refuses to touch protected "
            "paths (meta/, adapter/, harness/, experiments.db, etc.)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "reason": {"type": "string"},
                "evidence": {"type": "string"},
            },
            "required": ["path", "reason", "evidence"],
            "additionalProperties": False,
        },
    },

    "backup_path": {
        "type": "function",
        "name": "backup_path",
        "description": (
            "Copy a workspace path to meta/backups/<ts>/ without deleting it. "
            "Useful before risky operations or as a defensive snapshot."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },

    "request_phase_rewind": {
        "type": "function",
        "name": "request_phase_rewind",
        "description": (
            "Request a rewind to phase0, phase1, or phase2. EXECUTES "
            "IMMEDIATELY: backs up the affected artifacts and writes a marker "
            "the dispatcher consumes to restart the target phase. Requires "
            "Python-verified evidence — see your prompt."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "target_phase": {"type": "string"},
                "reason": {"type": "string"},
                "evidence": {"type": "string"},
            },
            "required": ["target_phase", "reason", "evidence"],
            "additionalProperties": False,
        },
    },

    "request_verification": {
        "type": "function",
        "name": "request_verification",
        "description": (
            "Conductor-only. Commission the independent finding-verifier to re-implement and "
            "stress-test a result FROM SCRATCH (executed notebooks; see-everything/import-nothing) "
            "and report whether it survives a fair benchmark. You are BOTH the user and the system: "
            "`steering` is written to the verifier's from_user.md AS THE USER (what to verify and "
            "what you care about — e.g. a specific experiment name, or 'verify the strong benchmark "
            "itself'), and the verifier's findings are surfaced back into your context digest AS THE "
            "SYSTEM so you can fold them into directives/annotations. Naming a `candidate` PINS it as the "
            "verifier's NEXT pick (jumping its adaptive order); `priority` (e.g. 'high') flags urgency — the "
            "latest request wins (NOT FIFO), and the pin is verified after any in-flight candidate finishes. "
            "Runs in a background thread, "
            "concurrent with Phase 3 — does NOT block the run. Use when a result is actionable enough "
            "to be worth an independent check. (The system also AUTO-commissions a verification once "
            "`conductor_verify_after_n_strategies` experiments are analyzed if you have not.)"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "candidate": {"type": "string", "description": "experiment name to PIN as the verifier's NEXT candidate (jumps its adaptive order); or 'NONE' to let the verifier pick adaptively from your steering"},
                "steering": {"type": "string", "description": "what to verify + what you care about; becomes the verifier's from_user.md"},
                "priority": {"type": "string", "description": "urgency of the pin, e.g. 'high'/'urgent' (optional). Latest request wins (not FIFO); the pinned candidate is verified next, after any in-flight candidate finishes"},
                "reason": {"type": "string"},
            },
            "required": ["steering", "reason"],
            "additionalProperties": False,
        },
    },

    "request_run_end": {
        "type": "function",
        "name": "request_run_end",
        "description": (
            "Request a graceful end-of-run. Conductor-only. EXECUTES "
            "IMMEDIATELY (subject to floors): writes a marker the "
            "dispatcher reads at the top of its next main-loop iteration; "
            "the dispatcher then stops admitting new submissions, lets "
            "in-flight experiments finish, generates a final milestone "
            "report, and exits cleanly. Use ONLY when you have "
            "Python-verified evidence the run is exhausted — diminishing "
            "returns over many experiments, broad coverage of approaches "
            "per `research_state.md`, no improvement in the primary "
            "metric for a long stretch, AND the run's `description` / "
            "`target` goal has been substantively satisfied. The tool "
            "REFUSES if any floor is unmet: minimum runtime hours, "
            "minimum analyzed-experiment count, or "
            "`allow_conductor_end_run=false` in the task config. Same "
            "evidence discipline as request_phase_rewind: write a "
            "Python script under `meta/scratch/` that demonstrates the "
            "exhaustion (e.g. plateau analysis), run it, paste script "
            "and output into evidence."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {"type": "string"},
                "evidence": {"type": "string"},
            },
            "required": ["reason", "evidence"],
            "additionalProperties": False,
        },
    },

    "read_experiment": {
        "type": "function",
        "name": "read_experiment",
        "description": (
            "Fetch a bounded summary of one experiment by id: description, "
            "hypothesis, config keys, headline metrics, error excerpt, parked "
            "and annotation state, plus paths to its run_experiment.py and "
            "debrief. Does NOT include the full code or full debrief — use "
            "read_file on the returned paths if you need those."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "experiment_id": {"type": "integer"},
            },
            "required": ["experiment_id"],
            "additionalProperties": False,
        },
    },

    "note_to_conductor": {
        "type": "function",
        "name": "note_to_conductor",
        "description": (
            "Leave a note for the Conductor in meta/notes_inbox.md. The "
            "Conductor reads this at the top of every turn. Use it to flag "
            "things the Conductor should know but you cannot fix yourself "
            "(e.g. \"don't preempt experiment #181 — the LoRA approach "
            "needs a full run\")."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["message"],
            "additionalProperties": False,
        },
    },
}

# Backward-compat aliases. ``ack_directive`` + ``note_to_conductor`` are
# included so the Phase 1 generic agent (which uses ``ALL_TOOL_SCHEMAS``)
# can actually respond to Conductor directives written mid-run. Without
# these, the Conductor's directives sit unread/unacked even when the
# Phase 1 prompt tells the agent to consume them.
FUNCTION_TOOLS: list[dict[str, Any]] = [
    TOOL_REGISTRY[name]
    for name in ("shell_exec", "view_image", "ask_user", "report_to_user",
                 "memory_store", "memory_search", "memory_read",
                 "ack_directive", "note_to_conductor")
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

# Default fallback for the in-tool hard cap. The production path threads
# config.tool_output_hard_cap_chars into execute_tool, which forwards it down
# to execute_shell / grep_file / _truncate_output. Kept at module level so
# tests and ad-hoc callers that bypass execute_tool still have a sane bound.
MAX_OUTPUT_CHARS = 30_000
DEFAULT_TIMEOUT = 300


def execute_shell(
    command: str,
    workspace: str,
    timeout: int = DEFAULT_TIMEOUT,
    max_output_chars: int = MAX_OUTPUT_CHARS,
) -> str:
    """Execute a shell command in the workspace directory."""
    timeout = max(timeout, 1)

    # Agents occasionally paste binary-contaminated text into a command;
    # an embedded NUL makes Popen raise ValueError before the shell even
    # sees the command. Strip NULs rather than failing the whole call.
    if "\x00" in command:
        command = command.replace("\x00", "")

    # Prepend the workspace to PYTHONPATH so ``import backtest`` (or any
    # other local package the workspace ships) resolves to the local
    # copy rather than to whatever site-packages ``.pth`` files happen
    # to inject onto sys.path. Without this, an unrelated project's
    # ``backtest/`` on PYTHONPATH (via an editable-install ``.pth`` or
    # a stale workspace entry) silently shadows the workspace's own
    # framework module — causing ``ImportError: cannot import name
    # X from .../some-other-project/backtest/metrics.py``.
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        workspace + (os.pathsep + existing if existing else "")
    )

    proc: subprocess.Popen | None = None
    try:
        proc = subprocess.Popen(
            command,
            shell=True,
            cwd=workspace,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            # Replace undecodable bytes instead of crashing on commands
            # that emit binary (e.g. accidental ``cat`` of a parquet
            # file, or a script with non-UTF8 logging). Without this the
            # whole shell_exec call returns ``[ERROR] UnicodeDecodeError``
            # and the agent loses the actual command output.
            errors="replace",
            # ``preexec_fn`` runs in the child between fork and exec: it
            # puts the child in its own process group (so we can ``killpg``
            # on timeout) and asks the kernel to SIGKILL the child if the
            # parent dies. Replaces the previous ``start_new_session=True``,
            # which created orphaned subprocesses on parent kill.
            preexec_fn=_preexec_setup,
        )
        _register_subprocess(proc)
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
                    f"[ERROR] Command timed out after {timeout}s and did not exit after SIGKILL",
                    max_output_chars,
                )
            return _truncate_output(
                f"[ERROR] Command timed out after {timeout}s", max_output_chars,
            )

        output_parts = []
        if stdout:
            output_parts.append(stdout)
        if stderr:
            output_parts.append(f"[stderr]\n{stderr}")
        output_parts.append(f"[exit code: {proc.returncode}]")

        output = "\n".join(output_parts)

    except Exception as e:
        output = f"[ERROR] {type(e).__name__}: {e}"
    finally:
        if proc is not None:
            _unregister_subprocess(proc)

    return _truncate_output(output, max_output_chars)


def _truncate_output(text: str, max_output_chars: int = MAX_OUTPUT_CHARS) -> str:
    """Truncate output, keeping first and last portions."""
    if len(text) <= max_output_chars:
        return text

    half = max_output_chars // 2
    truncated_msg = (
        f"\n\n[... truncated {len(text) - max_output_chars} chars ...]\n\n"
    )
    return text[:half] + truncated_msg + text[-half:]


def _resolve_in_workspace(
    path: str,
    workspace: str,
    extra_roots: list[str] | tuple[str, ...] | None = None,
) -> Path | None:
    """Resolve a path ensuring it stays within the workspace OR any extra root.

    ``extra_roots`` typically holds the configured ``data_path`` so agents
    can read the dataset's own documentation (README, DECISIONS, etc.) —
    previously workers retried `etf_rfq_research_dataset/README.md` 259+
    times because the prompt mentions the dataset path but the resolver
    rejected it as "outside workspace". Read access to the dataset's own
    documentation is safe by design.
    """
    p = Path(path)
    if not p.is_absolute():
        p = Path(workspace) / p
    resolved = p.resolve()
    roots = [Path(workspace).resolve()]
    if extra_roots:
        for r in extra_roots:
            if r:
                try:
                    roots.append(Path(r).resolve())
                except Exception:
                    pass
    for root in roots:
        try:
            resolved.relative_to(root)
            return resolved
        except ValueError:
            continue
    return None


def read_file(
    path: str,
    workspace: str,
    offset: int = 0,
    limit: int = 500,
    extra_roots: list[str] | tuple[str, ...] | None = None,
) -> str:
    """Read a file from workspace, returning numbered lines."""
    p = _resolve_in_workspace(path, workspace, extra_roots=extra_roots)
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
    max_output_chars: int = MAX_OUTPUT_CHARS,
    extra_roots: list[str] | tuple[str, ...] | None = None,
) -> str:
    """Search workspace files via grep -rn."""
    # Validate search path stays within workspace (or extra roots like data_path)
    resolved = _resolve_in_workspace(path, workspace, extra_roots=extra_roots)
    if resolved is None:
        return f"[ERROR] Path outside workspace: {path}"
    # Use resolved path relative to workspace for grep cwd
    try:
        search_path = str(resolved.relative_to(Path(workspace).resolve()))
    except ValueError:
        # Resolved outside the workspace (must be under an extra_root). grep
        # from the resolved absolute path directly.
        search_path = str(resolved)

    cmd = ["grep", "-rn", "--color=never"]
    if include:
        cmd.extend(["--include", include])
    cmd.append("--")
    cmd.append(pattern)
    cmd.append(search_path)

    # 120s timeout — workspaces accumulate large `logs/` dirs (tens of GB),
    # and a wide `grep -rn` on the dir routinely needs 30-90s on NFS. The
    # previous 30s default rejected ~21% of agent grep calls. Logs are the
    # primary source of truth for diagnosis; we'd rather wait than push
    # agents toward shallower investigations.
    GREP_TIMEOUT_SECONDS = 120
    try:
        result = subprocess.run(
            cmd,
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=GREP_TIMEOUT_SECONDS,
        )
        output = result.stdout or ""
        if result.returncode == 1 and not output:
            return "No matches found."
        if result.stderr:
            output += f"\n[stderr] {result.stderr}"
        return _truncate_output(output, max_output_chars) if output else "No matches found."
    except subprocess.TimeoutExpired:
        return f"[ERROR] grep timed out after {GREP_TIMEOUT_SECONDS}s — narrow the path (e.g. logs/dispatcher.jsonl) or use --include to filter by filename"
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


# Model that performs proxied web searches. Benchmarked 2026-07-31 against
# registry-anchored ground truth (PyPI/endoflife.date): gpt-5.5 was the only
# model exact on 9/9 freshness queries and 6/6 repeat trials; gpt-4.1-mini
# returned stale versions in 3 of 8 trials under this prompt shape.
WEB_SEARCH_PROXY_MODEL = "gpt-5.5"


def _proxy_web_search(
    query: str,
    openai_client: Any | None = None,
    model: str = WEB_SEARCH_PROXY_MODEL,
) -> str:
    """Proxy a web search through an OpenAI model with the ``web_search`` tool.

    Used when the provider doesn't have built-in web search (e.g. Bedrock).
    Falls back to an error message if no OpenAI client is available.
    """
    if openai_client is None:
        return "[ERROR] Web search requires an OpenAI client for proxy. Not available."

    try:
        response = openai_client.responses.create(
            model=model,
            tools=[{"type": "web_search"}],
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


def _experiment_budget_refusal(db: Any) -> str | None:
    """Lifetime experiment budget enforced at the proposal tools themselves.

    The strategist's turn-entry gate skips a turn once the budget is spent,
    but it cannot stop a single in-flight turn from proposing past the cap
    (one glm-5.2 run proposed 72 experiments past the prompt banner before
    the turn gate existed; the tool accepted every row). Reads the config
    off the DB handle the way propose_variant's fan-out cap does; a bare
    test DB without config gets no cap, same as that path."""
    cfg = getattr(db, "_task_config", None)
    try:
        max_exps = int(cfg.pipeline.phase3.max_experiments) if cfg else 0
    except (AttributeError, TypeError, ValueError):
        max_exps = 0
    if max_exps <= 0:
        return None
    try:
        summary = db.board_summary()
    except Exception:
        return None
    total = sum(v for k, v in summary.items() if k != "cancelled")
    if total < max_exps:
        return None
    return (
        f"[REFUSED] Lifetime experiment budget reached ({total}/{max_exps} "
        "non-cancelled experiments). No further proposals are accepted "
        "this run — consolidate findings and report instead."
    )


def execute_tool(
    name: str,
    arguments: dict[str, Any],
    workspace: str,
    ask_user_fn: Callable[[str], str] | None = None,
    db: Any | None = None,
    openai_client: Any | None = None,
    adapter: Any | None = None,
    shell_timeout: int = DEFAULT_TIMEOUT,
    tool_output_hard_cap_chars: int = MAX_OUTPUT_CHARS,
    data_path: str | None = None,
    caller_role: str = "unknown",
    caller_id: str = "unknown",
) -> dict[str, Any]:
    """Execute a tool and return the result.

    Returns a dict with:
      - "output": str result for the API
      - "image": optional (base64, media_type) tuple for view_image
      - "done": True if report_to_user was called
    """
    if name == "shell_exec":
        command = arguments.get("command", "")
        # Guard: refuse an UNSCOPED `find` (searching from `/` or a bare top-level mount
        # like /ms, /v, /home). On networked filesystems these scan everything and hang for
        # many minutes to hours. Scoped finds (relative paths, or deep absolute paths under
        # the workspace) are allowed. System-wide — applies to every agent.
        import re as _re_find
        _bad_find = (_re_find.search(
            r'(?:^|[;&|\n])\s*find\s+(?:-[A-Za-z]+\s+)*'
            r'(?:/|/ms|/v|/u|/usr|/home|/proc|/sys|/dev|/etc|/opt|/var|/lib(?:64)?|/s?bin|/mnt|/net|~)'
            r'(?:\s|$)', command) if (command and "find" in command) else None)
        if _bad_find:
            return {"output": (
                "[ERROR] Refused: unscoped find `" + _bad_find.group(0).strip() + "` scans the "
                "whole (networked) filesystem and can hang for hours. Scope it to a directory — "
                "e.g. `find experiments/<name> -name ...` or a deep absolute path under the workspace.")}
        # shell_timeout is the operator-configured CEILING. It's typed int in Python but
        # comes from config — JSON/YAML can produce None/strings — normalize before min(...).
        try:
            shell_timeout = int(shell_timeout) if shell_timeout is not None else DEFAULT_TIMEOUT
        except (TypeError, ValueError):
            shell_timeout = DEFAULT_TIMEOUT
        if shell_timeout <= 0:
            shell_timeout = DEFAULT_TIMEOUT
        # When the LLM OMITS a timeout, default to DEFAULT_TIMEOUT — NEVER the (possibly
        # multi-day) ceiling, or a runaway command with no explicit timeout hangs for days.
        # The LLM may explicitly request more per-call, up to the ceiling.
        requested = arguments.get("timeout", DEFAULT_TIMEOUT)
        try:
            requested = int(requested)
        except (TypeError, ValueError):
            requested = DEFAULT_TIMEOUT
        timeout = min(requested, shell_timeout) if requested > 0 else DEFAULT_TIMEOUT
        # Log shell commands to a single global log file
        import datetime
        log_path = Path(__file__).resolve().parent.parent.parent / "tool_call_log.log"
        try:
            with open(log_path, "a") as log_f:
                ts = datetime.datetime.now().isoformat(timespec="seconds")
                log_f.write(f"[{ts}] shell_exec | workspace={workspace} | {command}\n")
        except Exception:
            pass  # Don't let logging failures break execution
        output = execute_shell(command, workspace, timeout, tool_output_hard_cap_chars)
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
        extra_roots = [data_path] if data_path else None
        output = read_file(path, workspace, offset, limit, extra_roots=extra_roots)
        return {"output": output}

    elif name == "grep_file":
        pattern = arguments.get("pattern", "")
        search_path = arguments.get("path", ".")
        include = arguments.get("include")
        extra_roots = [data_path] if data_path else None
        output = grep_files(pattern, workspace, search_path, include, tool_output_hard_cap_chars, extra_roots=extra_roots)
        return {"output": output}

    # Phase 3 tools
    elif name == "propose_experiment":
        if db is None:
            return {"output": "[ERROR] Experiment database not available."}
        refusal = _experiment_budget_refusal(db)
        if refusal:
            return {"output": refusal}
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

    elif name == "propose_variant":
        # Spawn a variant of an existing experiment. The variant's directory
        # starts as an exact copy of the base's experiments/<base.name>/
        # (excluding ``results/`` and ``logs/`` — those are the base's
        # outputs, not the variant's). The base is preserved untouched. A
        # ``.variant_intent.md`` file is written into the variant dir for
        # the implement-worker to read before editing.
        if db is None:
            return {"output": "[ERROR] Experiment database not available."}
        refusal = _experiment_budget_refusal(db)
        if refusal:
            return {"output": refusal}
        import re as _re
        import shutil
        try:
            base_id = int(arguments.get("base_experiment_id", 0))
        except (TypeError, ValueError):
            return {"output": "[ERROR] base_experiment_id must be an integer."}
        base = db.get(base_id) if base_id else None
        if base is None:
            return {"output": f"[ERROR] Base experiment #{base_id} not found."}
        # Cap variant fan-out per base. The cap lives on TaskConfig but is
        # not directly available here — we read it via the global config
        # callsite. Fall back to a conservative default (5) if not set.
        max_variants = 5
        try:
            # The dispatcher attaches the config-loaded TaskConfig to the
            # ExperimentDB via the strategist/worker call paths; if not
            # present we use the default. Reading via getattr keeps this
            # tolerant of test fixtures that build a bare DB.
            cfg = getattr(db, "_task_config", None)
            if cfg is not None:
                p3 = getattr(getattr(cfg, "pipeline", None), "phase3", None)
                if p3 is not None and hasattr(p3, "max_variants_per_base"):
                    max_variants = int(p3.max_variants_per_base)
        except Exception:  # pragma: no cover — defensive
            pass
        try:
            existing_variants = db.count_variants_of(base_id)
        except Exception as e:
            return {"output": f"[ERROR] Failed to count variants of #{base_id}: {e}"}
        if existing_variants >= max_variants:
            return {
                "output": (
                    f"[ERROR] Variant cap reached for base #{base_id}: "
                    f"{existing_variants} variants already exist, "
                    f"max_variants_per_base={max_variants}. Try a "
                    f"genuinely different approach via propose_experiment, "
                    f"or pick a different base."
                )
            }
        # Sanitize the variant name (mirrors propose_experiment).
        variant_name = _re.sub(
            r"[^a-zA-Z0-9_\-]", "_", arguments.get("name", "")
        )[:80]
        if not variant_name:
            return {"output": "[ERROR] Invalid variant name."}
        if variant_name == base.name:
            return {
                "output": (
                    f"[ERROR] Variant name '{variant_name}' is identical to "
                    f"the base experiment's name. Pick a different name."
                )
            }
        # Validate the base experiment directory actually exists on disk.
        exp_root = Path(workspace) / "experiments"
        base_dir = exp_root / base.name
        if not base_dir.is_dir():
            return {
                "output": (
                    f"[ERROR] Base experiment directory not found on disk: "
                    f"experiments/{base.name}/ — cannot variant from "
                    f"#{base_id}. Try a different base."
                )
            }
        variant_dir = exp_root / variant_name
        if variant_dir.exists():
            return {
                "output": (
                    f"[ERROR] experiments/{variant_name}/ already exists — "
                    f"pick a different variant name."
                )
            }
        # Copy the base's INPUTS (run_experiment.py, strategy.py, config) into
        # the variant dir, excluding the base's OUTPUTS — copying outputs makes
        # a not-yet-run variant look already-run/already-analyzed and lets the
        # parent's findings be misattributed to the variant.
        #   results/        : base's canonical metrics — would mislead the analyzer.
        #   logs/           : workspace-relative, exclude defensively.
        #   __pycache__/    : junk.  .base_snapshot/ : stale recursive snapshot.
        #   debrief.md / analysis.md : base's write-up (the variant has none yet).
        #   analysis/       : base's analysis scripts/outputs.
        #   run_status.json : base's run record (carries the base's exp_name/job).
        #   local_job*.out (+ symlink) : base's subprocess stdout.
        # run_experiment.py / strategy.py / config.* are INPUTS and DO copy;
        # .variant_intent.md is overwritten fresh below.
        excluded = {
            "results", "logs", "__pycache__", ".base_snapshot",
            "analysis", "debrief.md", "analysis.md", "run_status.json",
        }
        def _ignore(_src: str, names: list[str]) -> list[str]:
            return [
                n for n in names
                if n in excluded
                or n.endswith(".pyc")
                or n.startswith("local_job")  # local_job*.out + symlink
            ]
        try:
            shutil.copytree(base_dir, variant_dir, ignore=_ignore)
        except (OSError, shutil.Error) as e:
            return {
                "output": (
                    f"[ERROR] Failed to copy experiments/{base.name}/ -> "
                    f"experiments/{variant_name}/: {e}"
                )
            }
        # Write .variant_intent.md so the implement-worker reads what to
        # change before editing the inherited code.
        what_changes = (arguments.get("what_changes") or "").strip()
        hypothesis = (arguments.get("hypothesis") or "").strip()
        intent = (
            f"# Variant intent (read this before editing)\n\n"
            f"This experiment is a **variant of #{base_id} `{base.name}`**. "
            f"The directory was copied from `experiments/{base.name}/` at "
            f"creation time (excluding `results/`, `logs/`, "
            f"`__pycache__`). The strategist asks for ONE focused diff — "
            f"do not rewrite from scratch.\n\n"
            f"## Hypothesis (why this variant might do better)\n\n"
            f"{hypothesis or '(strategist left this empty)'}\n\n"
            f"## What changes (apply only these edits)\n\n"
            f"{what_changes or '(strategist left this empty)'}\n\n"
            f"## Implementer instructions\n\n"
            f"1. Read this file first, then read the inherited code "
            f"(`strategy.py`, `run_experiment.py`, `config.yaml`, etc.).\n"
            f"2. Edit ONLY the files needed for the change above. Do not "
            f"   rewrite `strategy.py` from scratch.\n"
            f"3. Smoke-test like a fresh experiment, run reality check, "
            f"   then update_experiment(status='checked').\n"
            f"4. Do NOT delete this file — the analyzer reads it to "
            f"   compare the variant to its base."
        )
        try:
            (variant_dir / ".variant_intent.md").write_text(intent)
        except OSError as e:
            # Best effort — if the write fails, the variant still has the
            # parent_id pointer and the implement-worker can recover from
            # there.
            logger = logging.getLogger("alpha_lab.tools")
            logger.warning("propose_variant: .variant_intent.md write failed: %s", e)
        # Description carried over: base description plus the intended change
        # so the strategist gets a useful one-liner in board listings.
        description = (
            f"Variant of #{base_id} ({base.name}): {what_changes[:200]}"
            if what_changes else f"Variant of #{base_id} ({base.name})"
        )
        # Reuse the base's config_json verbatim — the implement-worker edits
        # files on disk, not this JSON blob. (config_json is descriptive
        # metadata; the on-disk config.yaml is authoritative.)
        try:
            exp_id = db.create(
                variant_name, description, hypothesis,
                base.config_json or "{}",
                parent_id=base_id,
            )
        except Exception as e:
            # Roll back the copytree so we don't leave a half-baked dir.
            try:
                shutil.rmtree(variant_dir)
            except OSError:
                pass
            return {"output": f"[ERROR] Failed to create variant row: {e}"}
        return {
            "output": (
                f"Variant #{exp_id} '{variant_name}' created from #{base_id} "
                f"'{base.name}'. Directory copied "
                f"(excluding results/, logs/), .variant_intent.md written. "
                f"Status: to_implement. The implement-worker will read "
                f"the intent file and apply your requested changes."
            )
        }

    elif name == "update_playbook":
        # Back up the existing playbook before overwriting so an LLM
        # error (e.g. accidentally calling with an empty or truncated
        # content payload) cannot destroy accumulated strategist
        # context. Backups land under
        # ``meta/backups/playbook_<ts>.md`` so the audit trail is in
        # the same place as the Conductor's own backups.
        #
        # Race-safety / self-healing: the analyzer appends emerging
        # guardrails to playbook.md via shell O_APPEND while the
        # strategist may be mid-turn computing a new consolidated body.
        # To avoid losing those appends on the strategist's write, the
        # tool reads the CURRENT file just before writing and preserves
        # anything below a sentinel marker. The strategist's content
        # goes above the sentinel; analyzer appends always go to EOF
        # (i.e. below the sentinel). When the strategist consolidates
        # appends into the main body, it just writes content above the
        # sentinel and the appends below clear out (until the next
        # analyzer write).
        import re as _re
        _APPEND_SENTINEL = (
            "<!-- ANALYZER-APPENDS-BELOW (handled by update_playbook tool; "
            "appends below this line are preserved across strategist writes "
            "and folded into the main body on the next consolidation) -->"
        )
        # Match ANY ANALYZER-APPENDS-BELOW sentinel variant, not just the exact
        # canonical string above. Older playbooks (and adapter templates) carry
        # a legacy "<!-- ANALYZER-APPENDS-BELOW (do not delete this line) -->"
        # whose text differs; keying the dedup on the exact string lets that
        # legacy variant survive forever alongside the canonical one. The regex
        # collapses every variant to the single canonical sentinel on each write.
        _APPEND_SENTINEL_RE = _re.compile(r"<!--\s*ANALYZER-APPENDS-BELOW.*?-->")
        content = arguments.get("content", "")
        # The strategist typically reads playbook.md, edits the body, and
        # writes the whole thing back — which means the incoming ``content``
        # often INCLUDES a sentinel verbatim. Cut at the FIRST sentinel variant
        # (everything before it is the strategist's body) so the tool can
        # re-emit exactly one canonical sentinel + preserved appends. Without
        # this the sentinel accumulates by one per write (observed: 15 after 14
        # turns), and legacy-variant lines linger indefinitely.
        if _APPEND_SENTINEL_RE.search(content):
            content = _APPEND_SENTINEL_RE.split(content, 1)[0]
        playbook_path = Path(workspace) / "playbook.md"
        preserved_below = ""
        if playbook_path.exists():
            try:
                current = playbook_path.read_text()
                from alpha_lab import meta_layout as _ml
                _ml.ensure_meta_layout(workspace)
                ts = int(time.time())
                backup_path = _ml.backups_dir(workspace) / f"playbook_{ts}.md"
                backup_path.write_text(current)
                # Preserve any analyzer appends that landed since the last
                # strategist consolidation. We read once, right before
                # writing, so even if an append landed mid-strategist-turn
                # it's captured here. If the existing file already has
                # multiple sentinels accumulated (a prior version of this
                # tool had a bug where each write added one extra), keep
                # only the content after the LAST sentinel — that's where
                # the freshest analyzer appends live.
                if _APPEND_SENTINEL_RE.search(current):
                    # Keep only content after the LAST sentinel variant —
                    # that's where the freshest analyzer appends live.
                    after = _APPEND_SENTINEL_RE.split(current)[-1]
                    # Strip any stale duplicate sentinels (legacy or canonical)
                    # that might still hide in ``after`` from earlier corruption.
                    after = _APPEND_SENTINEL_RE.sub("", after)
                    preserved_below = after.lstrip("\n")
            except OSError as e:
                # Don't block the write on backup failure — log and
                # proceed. Losing the backup is preferable to refusing
                # the strategist's update.
                logger = logging.getLogger("alpha_lab.tools")
                logger.warning(f"update_playbook: backup failed ({e}); writing anyway")
        # Compose: strategist's consolidated body, then the sentinel,
        # then any analyzer appends that survived. Atomic write (tmp +
        # rename) so concurrent readers see either old or new, never
        # torn.
        composed_parts = [content.rstrip(), "", _APPEND_SENTINEL]
        if preserved_below.strip():
            composed_parts.extend(["", preserved_below.rstrip()])
        composed = "\n".join(composed_parts) + "\n"
        tmp = playbook_path.with_suffix(playbook_path.suffix + ".tmp")
        try:
            tmp.write_text(composed)
            tmp.replace(playbook_path)
        except OSError as e:
            # Fall back to direct overwrite if rename fails for any reason.
            playbook_path.write_text(composed)
            logger = logging.getLogger("alpha_lab.tools")
            logger.warning(f"update_playbook: atomic rename failed ({e}); wrote directly")
        msg = (
            f"playbook.md updated ({len(composed)} chars; "
            f"prior version backed up to meta/backups/"
        )
        if preserved_below.strip():
            msg += "; preserved analyzer appends below sentinel"
        msg += ")."
        return {"output": msg}

    elif name == "read_board":
        if db is None:
            return {"output": "[ERROR] Experiment database not available."}
        # Use adapter metric if available, default to sharpe
        _metric = "sharpe"
        _metric_display = "Sharpe"
        _direction = "maximize"
        if adapter is not None:
            _metric = adapter.metric.primary_metric
            _metric_display = adapter.metric.display_name
            _direction = adapter.metric.direction

        summary = db.board_summary()
        recent = db.list_all()[-10:]
        leaders = db.leaderboard(_metric, 10, direction=_direction)

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
        # Distinguish "field omitted" (None) from "explicit clear" (""):
        # the latter is how a worker un-marks a stale error after a
        # successful retry. Without this, a failed-then-recovered
        # experiment carries its old failure narrative all the way to
        # ``analyzed``, surfacing in the GUI as a red-error row even
        # though the run succeeded.
        error_arg = arguments.get("error")
        debrief_path = arguments.get("debrief_path")

        # Verify experiment exists
        exp = db.get(exp_id)
        if exp is None:
            return {"output": f"[ERROR] Experiment #{exp_id} not found."}

        updates: list[str] = []
        results_refused: str | None = None
        results_auto_promoted: bool = False
        if results:
            outcome = db.set_results(exp_id, results)
            if outcome == "applied":
                updates.append("results set")
            elif outcome == "applied:promoted":
                results_auto_promoted = True
                updates.append(
                    "results set; row auto-promoted to status='finished' "
                    "(canonical payload landed on a stuck implemented/checked "
                    "row — analyzer will pick it up next)."
                )
            elif outcome.startswith("refused:"):
                results_refused = outcome[len("refused:"):]
                updates.append(
                    f"results REFUSED ({results_refused}) — payload is "
                    "non-canonical (smoke/dry/partial). Re-run the experiment "
                    "until results/metrics.json is a full canonical artifact "
                    "before calling update_experiment(results=...)."
                )
            else:
                updates.append(f"results outcome={outcome}")
        if error_arg is not None:
            db.set_error(exp_id, error_arg)
            updates.append("error cleared" if error_arg == "" else "error set")
        # Safety net: a transition to ``analyzed`` or ``done`` carrying a
        # ``results`` payload is the worker asserting "this run
        # succeeded". A stale error field from an earlier failed attempt
        # would lie in the GUI. Auto-clear it unless the worker is
        # explicitly setting a new error this same call. Skipped when the
        # results payload was refused (non-canonical) — the error field is
        # the worker's primary signal that the canonical run still needs
        # to land.
        elif (
            status in ("analyzed", "done")
            and results
            and not results_refused
            and exp.error
        ):
            db.set_error(exp_id, "")
            updates.append("error auto-cleared (success transition with results)")
        # Single update_status call with all kwargs
        status_kwargs = {}
        if debrief_path:
            status_kwargs["debrief_path"] = debrief_path
            updates.append(f"debrief_path={debrief_path}")
        if status or status_kwargs:
            outcome = db.update_status(exp_id, status or exp.status, **status_kwargs)
            if status:
                # Surface the kanban guard's verdict so the LLM doesn't
                # silently retry a blocked transition. "applied" is the
                # normal case; "idempotent" means same-status (LLM may be
                # re-emitting); "blocked:<current>" means the move was
                # disallowed by the forward-only guard and the row stays
                # where it was.
                if outcome == "applied":
                    updates.append(f"status={status}")
                elif outcome == "idempotent":
                    updates.append(
                        f"status already {status} (idempotent — no need to retry)"
                    )
                elif outcome.startswith("blocked:"):
                    current = outcome.split(":", 1)[1]
                    updates.append(
                        f"status transition {current} -> {status} BLOCKED by "
                        f"forward-only kanban guard; row stays at {current}. "
                        f"Do not retry this transition."
                    )

        # MLflow side-effects on the experiment's sub-run. No-op when MLflow
        # is off. Re-fetch the experiment so we see the latest debrief_path
        # populated by the updates above.
        if results or error_arg is not None or status:
            terminal_set = {"analyzed", "done", "cancelled", "finished"}
            terminal = status if status in terminal_set else None
            _log_experiment_results_to_mlflow(
                db=db,
                exp=db.get(exp_id),
                results_json=results if results else None,
                error=error_arg if error_arg else None,
                workspace=workspace,
                terminal_status=terminal,
            )

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
        # Snapshot the current adapter file before overwriting, so a bad patch can be rolled
        # back. The adapter dir lives under the gitignored workspace, so a git checkpoint can
        # never track it — and the old `git add -A && git commit` instead swept the ENTIRE repo
        # working tree (source edits, configs, run logs) into a commit under the user's git
        # identity, polluting the branch while protecting nothing. A plain file backup gives the
        # rollback safety with zero git side effects.
        if target.exists():
            import shutil as _shutil, time as _time
            try:
                _bdir = adapter_dir / ".backups"
                _bdir.mkdir(exist_ok=True)
                _shutil.copy2(target, _bdir / f"{filename}.{int(_time.time())}.bak")
            except OSError:
                pass  # best-effort backup
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
        from alpha_lab.memory import MemoryStore
        store = MemoryStore(workspace)
        entry_id = store.store(
            content=arguments.get("content", ""),
            tags=arguments.get("tags", []),
            summary=arguments.get("summary", ""),
        )
        return {"output": f"Memory #{entry_id} stored."}

    elif name == "memory_search":
        from alpha_lab.memory import MemoryStore
        store = MemoryStore(workspace)
        results = store.search(
            query=arguments.get("query", ""),
            tags=arguments.get("tags"),
            limit=arguments.get("limit", 10),
        )
        if not results:
            return {"output": "No matching memories found."}
        lines = [f"Found {len(results)} memories:"]
        for entry in results:
            lines.append(f"  #{entry.id} [{', '.join(entry.tags)}] {entry.summary}")
        return {"output": "\n".join(lines)}

    elif name == "memory_read":
        from alpha_lab.memory import MemoryStore
        store = MemoryStore(workspace)
        content = store.read(arguments.get("memory_id", 0))
        return {"output": content}

    # -----------------------------------------------------------------------
    # Conductor tools. Each mutation tool also writes a meta_log entry so the
    # audit trail captures the decision regardless of which agent invoked the
    # call (in practice only the Conductor is granted these tools, but we
    # log defensively).
    # -----------------------------------------------------------------------
    elif name == "park_experiment":
        if db is None:
            return {"output": "[ERROR] Experiment database not available."}
        from alpha_lab import conductor_tools as _ct
        eid = int(arguments.get("experiment_id", 0))
        reason = arguments.get("reason", "")
        evidence = arguments.get("evidence", "")
        if db.get(eid) is None:
            return {"output": f"[ERROR] Experiment #{eid} not found."}
        status = db.park(eid)
        _ct.meta_log_append(
            workspace,
            _ct.meta_log_entry("park", target=eid, reason=reason, evidence=evidence),
        )
        _ct.meta_log_render_md(workspace)
        return {"output": f"parked #{eid} (status={status})"}

    elif name == "unpark_experiment":
        if db is None:
            return {"output": "[ERROR] Experiment database not available."}
        from alpha_lab import conductor_tools as _ct
        eid = int(arguments.get("experiment_id", 0))
        reason = arguments.get("reason", "")
        evidence = arguments.get("evidence", "")
        if db.get(eid) is None:
            return {"output": f"[ERROR] Experiment #{eid} not found."}
        if not db.unpark(eid):
            return {
                "output": (
                    f"[ERROR] Experiment #{eid} is still cancelling; "
                    "wait for executor confirmation before unparking."
                )
            }
        _ct.meta_log_append(
            workspace,
            _ct.meta_log_entry("unpark", target=eid, reason=reason, evidence=evidence),
        )
        _ct.meta_log_render_md(workspace)
        return {"output": f"unparked #{eid}"}

    elif name == "set_priority":
        if db is None:
            return {"output": "[ERROR] Experiment database not available."}
        from alpha_lab import conductor_tools as _ct
        eid = int(arguments.get("experiment_id", 0))
        priority = int(arguments.get("priority", 0))
        reason = arguments.get("reason", "")
        evidence = arguments.get("evidence", "")
        if db.get(eid) is None:
            return {"output": f"[ERROR] Experiment #{eid} not found."}
        db.set_priority(eid, priority)
        _ct.meta_log_append(
            workspace,
            _ct.meta_log_entry(
                "set_priority", target=eid, reason=f"priority={priority}: {reason}",
                evidence=evidence,
            ),
        )
        _ct.meta_log_render_md(workspace)
        return {"output": f"set priority of #{eid} to {priority}"}

    elif name == "clear_experiment_block":
        if db is None:
            return {"output": "[ERROR] Experiment database not available."}
        from alpha_lab import conductor_tools as _ct
        eid = int(arguments.get("experiment_id", 0))
        reason = arguments.get("reason", "")
        evidence = arguments.get("evidence", "")
        exp = db.get(eid)
        if exp is None:
            return {"output": f"[ERROR] Experiment #{eid} not found."}
        if not db.clear_block(eid):
            return {
                "output": (
                    f"#{eid} is not in a blocked state "
                    f"(status={exp.status}, error={(exp.error or '')[:60]!r}); "
                    "nothing cleared."
                )
            }
        _ct.meta_log_append(
            workspace,
            _ct.meta_log_entry(
                "clear_experiment_block", target=eid, reason=reason, evidence=evidence,
            ),
        )
        _ct.meta_log_render_md(workspace)
        return {
            "output": (
                f"cleared stale block on #{eid} (status={exp.status}); "
                "requeued in place — dispatcher can now assign it."
            )
        }

    elif name == "annotate_experiment":
        if db is None:
            return {"output": "[ERROR] Experiment database not available."}
        from alpha_lab import conductor_tools as _ct
        eid = int(arguments.get("experiment_id", 0))
        label = (arguments.get("label", "") or "").strip()
        reason = arguments.get("reason", "")
        evidence = arguments.get("evidence", "")
        if db.get(eid) is None:
            return {"output": f"[ERROR] Experiment #{eid} not found."}
        if label == "":
            _ct.clear_annotation(workspace, eid)
            descr = "cleared"
        else:
            _ct.set_annotation(workspace, eid, label, reason=reason)
            descr = f"set to {label!r}"
        _ct.meta_log_append(
            workspace,
            _ct.meta_log_entry(
                "annotate", target=eid, reason=f"{descr}: {reason}", evidence=evidence,
            ),
        )
        _ct.meta_log_render_md(workspace)
        return {"output": f"annotation for #{eid} {descr}"}

    elif name == "issue_directive":
        from alpha_lab import conductor_tools as _ct
        target_role = arguments.get("target_role", "all")
        message = arguments.get("message", "")
        reason = arguments.get("reason", "")
        evidence = arguments.get("evidence", "")
        scope = arguments.get("scope", "standing")
        directive_id = _ct.append_directive(
            workspace, target_role, message, reason=reason, scope=scope,
        )
        _ct.meta_log_append(
            workspace,
            _ct.meta_log_entry(
                "directive",
                target=target_role,
                reason=f"[{directive_id} scope={scope}] {reason}\n\n{message}",
                evidence=evidence,
            ),
        )
        _ct.meta_log_render_md(workspace)
        return {
            "output": (
                f"directive {directive_id} recorded for {target_role} "
                f"(scope={scope})"
            )
        }

    elif name == "retire_directive":
        from alpha_lab import conductor_tools as _ct
        directive_id = arguments.get("directive_id", "").strip()
        reason = arguments.get("reason", "").strip()
        if not directive_id:
            return {"output": "[ERROR] retire_directive requires a directive_id."}
        if not reason:
            return {"output": "[ERROR] retire_directive requires a reason."}
        # Sanity: only retire ids that actually exist in the directives file,
        # so a typo doesn't silently no-op.
        existing_ids = {d["id"] for d in _ct.parse_directives(workspace)}
        if directive_id not in existing_ids:
            return {
                "output": (
                    f"[ERROR] directive_id {directive_id!r} not found in "
                    f"meta/directives.md. Issue new directives via "
                    f"issue_directive; only retire ids that exist."
                )
            }
        already_retired = directive_id in _ct.retired_directive_ids(workspace)
        _ct.append_directive_retirement(workspace, directive_id, reason)
        _ct.meta_log_append(
            workspace,
            _ct.meta_log_entry(
                "retire_directive",
                target=directive_id,
                reason=reason,
                evidence="",
            ),
        )
        _ct.meta_log_render_md(workspace)
        descr = "re-retired" if already_retired else "retired"
        return {"output": f"directive {directive_id} {descr}"}

    elif name == "ack_directive":
        from alpha_lab import conductor_tools as _ct
        directive_id = arguments.get("directive_id", "")
        action_taken = arguments.get("action_taken", "")
        _ct.append_directive_ack(
            workspace,
            directive_id=directive_id,
            actor_role=caller_role,
            actor_id=caller_id,
            action=action_taken,
        )
        return {
            "output": (
                f"ack recorded for directive {directive_id} "
                f"by {caller_role}/{caller_id}"
            )
        }

    elif name == "write_note_to_user":
        from alpha_lab import conductor_tools as _ct
        message = arguments.get("message", "")
        _ct.append_note_to_user(workspace, message)
        _ct.meta_log_append(
            workspace,
            _ct.meta_log_entry("note_to_user", reason=message[:200]),
        )
        _ct.meta_log_render_md(workspace)
        return {"output": "note appended to meta/notes_to_user.md"}

    elif name == "set_throttle":
        from alpha_lab import conductor_tools as _ct
        gpu = arguments.get("gpu")
        cpu = arguments.get("cpu")
        reason = arguments.get("reason", "")
        evidence = arguments.get("evidence", "")
        new_state = _ct.set_throttle_state(workspace, gpu=gpu, cpu=cpu)
        _ct.meta_log_append(
            workspace,
            _ct.meta_log_entry(
                "throttle", target=new_state, reason=reason, evidence=evidence,
            ),
        )
        _ct.meta_log_render_md(workspace)
        return {"output": f"throttle now: gpu={new_state['gpu']} cpu={new_state['cpu']}"}

    elif name == "read_meta_log":
        from alpha_lab import conductor_tools as _ct
        last_n = int(arguments.get("last_n", _ct.DEFAULT_META_LOG_LAST_N))
        sample_older = bool(arguments.get("sample_older", False))
        entries = _ct.meta_log_read(workspace, last_n=last_n, sample_older=sample_older)
        return {"output": json.dumps(entries, indent=2, default=str)[:30_000]}

    elif name == "read_user_instructions":
        from alpha_lab import conductor_tools as _ct
        from alpha_lab.conductor import _strip_from_user_boilerplate
        raw, is_new = _ct.read_from_user_diff(workspace)
        if arguments.get("mark_seen", False):
            _ct.mark_user_instructions_seen(workspace)
        # Strip the bootstrap-time `#`-commented help block so the
        # Conductor doesn't treat its own help text as a user
        # instruction. ``is_new`` is preserved as-is — it compares
        # the raw file against ``.last_seen``, which is the right
        # signal for "user wrote something new" detection.
        content = _strip_from_user_boilerplate(raw)
        return {
            "output": json.dumps(
                {"content": content[:10_000], "is_new": is_new}, indent=2
            )
        }

    elif name == "ack_user_instruction":
        from alpha_lab import conductor_tools as _ct
        message = arguments.get("message", "")
        _ct.append_ack(workspace, message)
        _ct.meta_log_append(
            workspace,
            _ct.meta_log_entry("ack_user_instruction", reason=message[:300]),
        )
        _ct.meta_log_render_md(workspace)
        return {"output": "acknowledgement appended to meta/instructions/ack.md"}

    elif name == "read_system_load":
        from alpha_lab import conductor_tools as _ct
        return {"output": _ct.read_system_load(workspace)}

    elif name == "peek_experiment_log":
        from alpha_lab import conductor_tools as _ct
        exp_name = arguments.get("experiment_name", "")
        last_n = int(arguments.get("last_n_lines", _ct.DEFAULT_PEEK_LINES))
        return {"output": _ct.peek_experiment_log(workspace, exp_name, last_n)}

    elif name == "kill_experiment":
        # Implementation: park the row so the dispatcher does not reassign,
        # plus write a kill-request marker the dispatcher reads at the top of
        # its next loop iteration to terminate any subprocess. The actual
        # subprocess termination lives in the dispatcher (it owns the
        # executor); this tool only signals intent.
        if db is None:
            return {"output": "[ERROR] Experiment database not available."}
        from alpha_lab import conductor_tools as _ct
        eid = int(arguments.get("experiment_id", 0))
        reason = arguments.get("reason", "")
        evidence = arguments.get("evidence", "")
        exp = db.get(eid)
        if exp is None:
            return {"output": f"[ERROR] Experiment #{eid} not found."}
        db.park(eid)
        # Write kill marker (jsonl, append-only) the dispatcher consumes.
        from alpha_lab import meta_layout as _ml
        marker_path = _ml.meta_dir(workspace) / "kill_requests.jsonl"
        _ml.ensure_meta_layout(workspace)
        try:
            with open(marker_path, "a") as f:
                f.write(json.dumps({
                    "ts": __import__("time").time(),
                    "experiment_id": eid,
                    "reason": reason,
                }) + "\n")
        except OSError:
            pass
        _ct.meta_log_append(
            workspace,
            _ct.meta_log_entry("kill", target=eid, reason=reason, evidence=evidence),
        )
        _ct.meta_log_render_md(workspace)
        return {"output": f"kill requested for #{eid} (parked + marker written)"}

    elif name == "delete_path":
        from alpha_lab import conductor_tools as _ct
        rel_path = arguments.get("path", "")
        reason = arguments.get("reason", "")
        evidence = arguments.get("evidence", "")
        try:
            backup = _ct.safe_delete_with_backup(workspace, rel_path, adapter=adapter)
        except (FileNotFoundError, ValueError, PermissionError) as e:
            return {"output": f"[ERROR] {e}"}
        _ct.meta_log_append(
            workspace,
            _ct.meta_log_entry(
                "delete", target=rel_path,
                reason=f"{reason} (backup at {backup})", evidence=evidence,
            ),
        )
        _ct.meta_log_render_md(workspace)
        return {"output": f"deleted {rel_path}; backup at {backup}"}

    elif name == "backup_path":
        from alpha_lab import conductor_tools as _ct
        rel_path = arguments.get("path", "")
        reason = arguments.get("reason", "")
        try:
            backup = _ct.backup_workspace_path(workspace, rel_path)
        except (FileNotFoundError, ValueError) as e:
            return {"output": f"[ERROR] {e}"}
        _ct.meta_log_append(
            workspace,
            _ct.meta_log_entry("backup", target=rel_path, reason=reason),
        )
        _ct.meta_log_render_md(workspace)
        return {"output": f"backed up {rel_path} -> {backup}"}

    elif name == "request_phase_rewind":
        from alpha_lab import conductor_tools as _ct
        target_phase = arguments.get("target_phase", "")
        reason = arguments.get("reason", "")
        evidence = arguments.get("evidence", "")
        try:
            payload = _ct.request_phase_rewind(
                workspace, target_phase, reason, evidence, adapter=adapter,
            )
        except ValueError as e:
            return {"output": f"[ERROR] {e}"}
        _ct.meta_log_append(
            workspace,
            _ct.meta_log_entry(
                "phase_rewind", target=target_phase,
                reason=reason, evidence=evidence,
            ),
        )
        _ct.meta_log_render_md(workspace)
        return {
            "output": f"phase rewind requested: {target_phase}; "
                      f"backup at {payload['backup_dir']}; "
                      f"dispatcher will execute on next loop iteration"
        }

    elif name == "request_verification":
        # Conductor-only: commission the finding-verifier. Writes the verifier's from_user.md
        # (Conductor as the user) + a marker the dispatcher consumes to spawn it in a thread.
        from alpha_lab import conductor_tools as _ct
        try:
            payload = _ct.request_verification(
                workspace,
                candidate=arguments.get("candidate", ""),
                steering=arguments.get("steering", ""),
                reason=arguments.get("reason", ""),
                priority=arguments.get("priority", ""),
            )
            return {"output": (
                f"Verification commissioned (candidate={payload.get('candidate') or 'verifier-picks'}). "
                "The dispatcher will spawn the verifier in a background thread; its findings will "
                "appear in verify/feedback_stream.md and in your next context digest.")}
        except Exception as e:
            return {"output": f"[ERROR] request_verification failed: {type(e).__name__}: {e}"}

    elif name == "request_run_end":
        # Conductor-only: graceful end-of-run. Pull the floors from
        # TaskConfig (attached to the DB by run.py) and the wall-clock
        # start from meta/run_state.json. Refuse if any floor is unmet.
        from alpha_lab import conductor_tools as _ct
        from alpha_lab import meta_layout as _ml
        if db is None:
            return {"output": "[ERROR] Experiment database not available."}
        cfg = getattr(db, "_task_config", None)
        if cfg is None:
            return {
                "output": (
                    "[ERROR] TaskConfig not attached to DB — cannot enforce "
                    "min_runtime_hours / min_analyzed_before_end / "
                    "allow_conductor_end_run floors. Refusing for safety."
                )
            }
        start_ts = _ml.read_dispatcher_start_ts(workspace)
        if start_ts is None:
            return {
                "output": (
                    "[ERROR] Dispatcher start timestamp not found in "
                    "meta/run_state.json — refusing run end because the "
                    "min_runtime_hours floor cannot be evaluated."
                )
            }
        summary = db.board_summary()
        # All terminal work counts toward the floor. Counting only
        # analyzed/done livelocked real runs: rows stranded in checked/
        # finished (or deliberately cancelled) could never satisfy the
        # floor once the experiment cap was reached, so the run could
        # neither end gracefully nor produce more work.
        analyzed_count = sum(
            summary.get(status, 0)
            for status in ("analyzed", "done", "checked", "finished",
                           "cancelled")
        )
        min_analyzed = int(getattr(cfg, "min_analyzed_before_end", 100))
        # The floor must be satisfiable under the experiment cap: the
        # default floor (100) with a 20-experiment cap made request_run_end
        # permanently refuse in observed runs.
        try:
            max_exps = int(cfg.pipeline.phase3.max_experiments)
        except (AttributeError, TypeError, ValueError):
            max_exps = 0
        if max_exps > 0:
            min_analyzed = min(min_analyzed, max_exps)
        reason = arguments.get("reason", "")
        evidence = arguments.get("evidence", "")
        try:
            payload = _ct.request_run_end(
                workspace, reason, evidence,
                dispatcher_start_ts=start_ts,
                analyzed_count=analyzed_count,
                min_runtime_hours=float(getattr(cfg, "min_runtime_hours", 6.0)),
                min_analyzed_before_end=min_analyzed,
                allow_end=bool(getattr(cfg, "allow_conductor_end_run", True)),
            )
        except ValueError as e:
            return {"output": f"[ERROR] {e}"}
        _ct.meta_log_append(
            workspace,
            _ct.meta_log_entry(
                "run_end", target="dispatcher",
                reason=reason, evidence=evidence,
            ),
        )
        _ct.meta_log_render_md(workspace)
        return {
            "output": (
                f"run end requested: elapsed_hours="
                f"{payload['elapsed_hours_at_request']:.1f}, "
                f"analyzed_at_request={payload['analyzed_at_request']}. "
                f"The dispatcher will stop admitting new submissions on "
                f"its next loop iteration, let in-flight experiments "
                f"finish, generate one final milestone report, and exit."
            )
        }

    elif name == "read_experiment":
        if db is None:
            return {"output": "[ERROR] Experiment database not available."}
        from alpha_lab import conductor_tools as _ct
        eid = int(arguments.get("experiment_id", 0))
        summary = _ct.read_experiment_summary(workspace, db, eid)
        return {"output": json.dumps(summary, indent=2, default=str)}

    elif name == "note_to_conductor":
        from alpha_lab import conductor_tools as _ct
        # The sender role is inferred from the AgentLoop's log_name and
        # passed through `caller_role`. The Conductor's inbox file is
        # partitioned by sender header so it can address each note correctly.
        message = arguments.get("message", "")
        _ct.append_note_to_conductor(workspace, caller_role, message)
        return {"output": f"note recorded for the Conductor (from {caller_role})"}

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
