---
name: tester
description: Phase 2 Tester — writes pytest tests for the framework built by the Builder and runs them, fixing failures until tests pass.
allowed-tools:
  - read_file
  - grep_file
  - shell_exec
  - report_to_user
  - memory_store
  - memory_search
  - memory_read
metadata:
  needs_gpu: true
  include_web_search: false
  reasoning_effort: low
  log_name: phase2_tester
  min_report_attempts: 1
  prompt_source: adapter:phase2_tester
---
