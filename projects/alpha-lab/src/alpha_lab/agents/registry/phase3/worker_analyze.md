---
name: worker_analyze
description: Phase 3 worker that analyzes a completed experiment — reads SLURM output, metrics, and model artifacts; compares against baselines; writes a debrief; and advances the experiment to `analyzed`.
allowed-tools:
  - read_file
  - grep_file
  - shell_exec
  - view_image
  - read_board
  - update_experiment
  - report_to_user
  - memory_store
  - memory_search
  - memory_read
metadata:
  needs_gpu: true
  include_web_search: false
  log_name: worker_{worker_id}_analyze_{experiment.name}
  min_report_attempts: 1
  prompt_source: adapter:phase3_worker_analyze
---
