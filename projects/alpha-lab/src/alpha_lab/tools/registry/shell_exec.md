---
name: shell_exec
description: Execute a shell command in the workspace directory. Commands run inside the workspace directory. Use this to run analysis scripts, install packages, etc. Write scripts to files first, then execute them.
metadata:
  workspace_access:
    ".": rw
  parameters:
    additionalProperties: false
    type: object
    properties:
      command:
        description: The shell command to execute.
        type: string
      timeout:
        description: Timeout in seconds. Capped at the task's shell_timeout (TaskConfig.shell_timeout; default 300). Raise this value (up to shell_timeout) for data-heavy scripts; to exceed the cap, increase
          shell_timeout in the task config.
        type: integer
    required:
      - command
---
