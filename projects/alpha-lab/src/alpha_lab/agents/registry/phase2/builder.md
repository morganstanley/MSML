---
name: builder
description: Phase 2 Builder — constructs the domain-specific evaluation framework (e.g. backtesting code) in the workspace, reading Phase 1 outputs as context.
allowed-tools:
  - shell_exec
  - view_image
  - read_file
  - grep_file
  - report_to_user
  - memory_store
  - memory_search
  - memory_read
metadata:
  needs_gpu: true
  include_web_search: false
  reasoning_effort: low
  log_name: phase2_builder
  min_report_attempts: 1
  prompt_source: adapter:phase2_builder
---
