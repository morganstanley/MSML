---
name: handoff
description: User's-proxy handoff agent — non-interactive turn after an experiment is analyzed; updates proxy_state and the public agenda. No ask_user.
allowed-tools:
  - read_file
  - grep_file
  - shell_exec
  - read_board
  - report_to_user
metadata:
  include_web_search: false
  log_name: worker_{worker_id}_handoff_{experiment}
  min_report_attempts: 1
  prompt_source: adapter:phase3_proxy_handoff
---
