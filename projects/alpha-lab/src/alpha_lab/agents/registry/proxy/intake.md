---
name: intake
description: User's-proxy intake agent — interactive session before a run starts; captures the user's goals/preferences into the agenda and proxy_state. ask_user enabled.
allowed-tools:
  - ask_user
  - read_file
  - grep_file
  - shell_exec
  - report_to_user
metadata:
  include_web_search: true
  log_name: intake
  min_report_attempts: 1
  prompt_source: adapter:phase0_proxy_intake
---
