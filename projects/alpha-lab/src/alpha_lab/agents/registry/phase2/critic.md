---
name: critic
description: Phase 2 Critic — reviews the framework built by the Builder for lookahead bias, data leakage, and other backtesting pitfalls; writes a verdict file.
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
  reasoning_effort: medium
  log_name: phase2_critic
  min_report_attempts: 1
  prompt_source: adapter:phase2_critic
---
