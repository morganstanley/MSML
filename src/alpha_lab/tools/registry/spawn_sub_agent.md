---
name: spawn_sub_agent
description: Spawn a sub-agent to work on a focused sub-task in its own conversation context. The sub-agent inherits your model, provider, and tools (except spawn_sub_agent). It runs to completion and returns its final report. Use this to delegate self-contained sub-problems that benefit from a fresh context window.
metadata:
  parameters:
    additionalProperties: false
    type: object
    properties:
      context:
        description: "Background information the sub-agent needs: data paths, prior findings, constraints, relevant file locations."
        type: string
      task:
        description: Clear description of what the sub-agent should accomplish. Be specific about expected outputs and success criteria.
        type: string
    required:
      - task
---
