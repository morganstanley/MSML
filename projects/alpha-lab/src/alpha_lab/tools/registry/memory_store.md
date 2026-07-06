---
name: memory_store
description: Store a piece of knowledge in persistent memory. Use this to save important findings, data insights, experiment results, or decisions that future agents should know about. Include relevant tags for searchability. Memory is persisted in a lightweight, portable workspace format so it can be reused outside Alpha Lab as well.
metadata:
  workspace_access:
    ".memory": rw
  parameters:
    additionalProperties: false
    type: object
    properties:
      content:
        description: The full content to store.
        type: string
      summary:
        description: One-line summary for search results.
        type: string
      tags:
        description: "Tags for categorization (e.g. ['data_quality', 'phase1'])."
        items:
          type: string
        type: array
      kind:
        description: Optional memory type. Prefer one of finding, decision, failure, result, hypothesis, constraint, reference. Use reference for reusable institutional knowledge, after user consent when it came from intake/config/workspace context. Common aliases like error, experiment_result, or runbook are normalized.
        type: string
      phase:
        description: Optional pipeline phase (e.g. phase1, phase2, phase3).
        type: string
      agent:
        description: Optional agent role that learned this (e.g. strategist, worker).
        type: string
      run_id:
        description: Optional run identifier for tracing provenance.
        type: string
      source_path:
        description: Optional workspace file that this memory came from.
        type: string
    required:
      - content
      - tags
      - summary
---
