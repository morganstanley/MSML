---
name: ask_user
description: Ask the user a question and wait for their response. ONLY use this when you are completely blocked and cannot proceed without user input. Do NOT use for status updates or confirmations.
metadata:
  parameters:
    additionalProperties: false
    type: object
    properties:
      question:
        description: The question to ask the user.
        type: string
    required:
      - question
---
