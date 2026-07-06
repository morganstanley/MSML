---
name: read_reference_adapter
description: "Read a built-in reference adapter to understand the expected format. Returns all files concatenated. If the adapter name is invalid, the error message includes the list of available adapters."
metadata:
  parameters:
    additionalProperties: false
    type: object
    properties:
      name:
        description: "Built-in adapter name"
        type: string
    required:
      - name
---
