---
name: interactive
description: Interactive REPL agent driven by the alpha-lab CLI; shares the Phase 1 Explorer surface but runs out-of-pipeline with ask_user enabled for direct user dialogue.
allowed-tools:
  - shell_exec
  - view_image
  - ask_user
  - report_to_user
  - read_file
  - grep_file
  - propose_experiment
  - update_playbook
  - read_board
  - update_experiment
  - reality_check
  - write_adapter_file
  - read_reference_adapter
  - read_adapter
  - patch_adapter_file
  - spawn_sub_agent
  - memory_store
  - memory_search
  - memory_read
  - cancel_experiments
metadata:
  include_web_search: true
  log_name: agent
  prompt_source: adapter:phase1
---
