#!/usr/bin/env python3
"""
AriadneMem MCP Server - stdio transport

Reads JSON-RPC 2.0 messages from stdin (one per line),
processes them via MCPHandler, writes responses to stdout.

Usage:
    python stdio_server.py
"""

import sys
import os
import json
import asyncio
import io

# Save original stdout for JSON-RPC output
_original_stdout = sys.stdout

# Redirect stdout and stderr to devnull to suppress ALL output
# (library imports, model loading, print statements, warnings, etc.)
_devnull = open(os.devnull, 'w')
sys.stdout = _devnull
sys.stderr = _devnull

# Also suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Add AriadneMem root to path
_mcp_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_root_dir = os.path.dirname(_mcp_dir)
sys.path.insert(0, _root_dir)
sys.path.insert(0, _mcp_dir)

from main import AriadneMemSystem
from server.mcp_handler import MCPHandler


def main():
    """Main stdio loop - completely silent except JSON-RPC on stdout"""
    # Initialize AriadneMem system (all output suppressed)
    system = AriadneMemSystem(clear_db=False)
    handler = MCPHandler(system)
    handler.initialized = True

    # Read from stdin line by line
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            # Parse to check if this is a notification (no "id" field)
            data = json.loads(line)
            is_notification = "id" not in data

            response = asyncio.run(handler.handle_message(line))

            # Per JSON-RPC 2.0: notifications MUST NOT receive a response
            if not is_notification:
                _original_stdout.write(response + "\n")
                _original_stdout.flush()
        except Exception as e:
            # Only send error response for requests (not notifications)
            try:
                data = json.loads(line)
                if "id" in data:
                    error_response = json.dumps({
                        "jsonrpc": "2.0",
                        "id": data["id"],
                        "error": {"code": -32603, "message": str(e)}
                    })
                    _original_stdout.write(error_response + "\n")
                    _original_stdout.flush()
            except:
                pass


if __name__ == "__main__":
    main()
