---
name: worker_fixer
description: Phase 3 worker that diagnoses and fixes a failed experiment — reads logs, applies a surgical edit, smoke-tests, and bumps the experiment back to `checked` so it can be resubmitted.
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
  log_name: worker_{worker_id}_fix_{experiment.name}
  min_report_attempts: 1
  prompt_source: adapter:phase3_fixer
---
