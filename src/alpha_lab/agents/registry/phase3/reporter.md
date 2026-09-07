---
name: reporter
description: Phase 3 milestone reporter that aggregates the leaderboard, generates publication-quality comparison plots, and writes a polished milestone report plus an entry in the running overview log.
allowed-tools:
  - shell_exec
  - read_file
  - grep_file
  - view_image
  - read_board
  - report_to_user
  - memory_store
  - memory_search
  - memory_read
metadata:
  include_web_search: false
  reasoning_effort: medium
  log_name: reporter_milestone_{milestone_number:03d}
  min_report_attempts: 1
  prompt_source: adapter:phase3_reporter
---
