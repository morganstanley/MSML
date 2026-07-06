---
name: strategist
description: Periodic Phase 3 meta-agent that reviews the experiment board, prunes the queue, proposes new experiments, and maintains the playbook. Single instance, runs on a cadence; no shell access.
allowed-tools:
  - read_board
  - propose_experiment
  - cancel_experiments
  - update_playbook
  - read_file
  - grep_file
  - report_to_user
  - memory_store
  - memory_search
  - memory_read
  - web_search
metadata:
  include_web_search: true
  log_name: strategist
  min_report_attempts: 1
  prompt_source: adapter:phase3_strategist
---
