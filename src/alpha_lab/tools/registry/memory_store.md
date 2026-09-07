---
name: memory_store
description: Store a piece of knowledge in persistent memory. Use this to save important findings, data insights, experiment results, or decisions that future agents should know about. Include relevant tags for searchability. Memory is persisted in a lightweight, portable workspace format so it can be reused outside Alpha Lab as well.
metadata:
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
        description: "Required kind: fact (taken as true), idea (possible direction), unknown (known gap), pattern (observed trend), constraint (requirement)."
        type: string
      agent:
        description: Optional agent role that learned this (e.g. strategist, worker).
        type: string
      run_id:
        description: Optional run identifier for tracing provenance.
        type: string
      sources:
        description: Optional source references supporting this memory.
        oneOf:
          - type: string
          - items:
              type: string
            type: array
      owner:
        description: Optional memory owner. Defaults to the current OS user.
        type: string
      frozen:
        description: Whether this memory record may be edited or removed later. Defaults to false.
        type: boolean
      max_similarity:
        description: Reject the write if its closest existing memory's cosine similarity exceeds this value. Defaults to 1.0 (no dedupe).
        type: number
    required:
      - content
      - tags
      - summary
      - kind
---
