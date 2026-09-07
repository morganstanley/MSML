#!/bin/bash
# Fetch and cache auth token for Alpha-Lab.
# Run this after: suu -tr "<ticket>" <proid>
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${RUNCMP_PYTHON:-python3}"
exec "$PYTHON" "$SCRIPT_DIR/prefetch_token.py" "$@"
