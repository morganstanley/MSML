---
name: explorer
description: Phase 1 single-agent dataset explorer — autonomously profiles the dataset, writes scripts and plots, and produces learnings.md and a data_report/.
allowed-tools:
  - shell_exec
  - view_image
  - ask_user
  - report_to_user
  - memory_store
  - memory_search
  - memory_read
metadata:
  needs_gpu: true
  include_web_search: true
  log_name: conversation
  prompt_source: adapter:phase1
---
