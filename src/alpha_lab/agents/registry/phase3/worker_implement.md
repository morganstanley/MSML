---
name: worker_implement
description: Phase 3 worker that implements a single proposed experiment — writes strategy/config/run_experiment files, smoke-tests them, runs the framework's tests, and advances the experiment to `checked` for execution.
allowed-tools:
  - shell_exec
  - read_file
  - grep_file
  - view_image
  - update_experiment
  - report_to_user
  - memory_store
  - memory_search
  - memory_read
metadata:
  needs_gpu: true
  include_web_search: false
  log_name: worker_{worker_id}_implement_{experiment.name}
  min_report_attempts: 1
  prompt_source: adapter:phase3_worker_implement
---
