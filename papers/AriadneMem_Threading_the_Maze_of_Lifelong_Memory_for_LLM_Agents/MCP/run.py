#!/usr/bin/env python3
"""
AriadneMem MCP Server - Runner Script

Starts the Streamable HTTP server providing:
- MCP Protocol endpoint (/mcp)
- REST API (/api/*)
- Health check (/api/health)
"""

import argparse


def main():
    parser = argparse.ArgumentParser(
        description="AriadneMem MCP Server - Graph-Based Lifelong Memory for LLM Agents"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  AriadneMem MCP Server")
    print("  Graph-Based Lifelong Memory for LLM Agents")
    print("=" * 60)
    print()
    print(f"  MCP Endpoint:  http://localhost:{args.port}/mcp")
    print(f"  REST API:      http://localhost:{args.port}/api/")
    print(f"  Health Check:  http://localhost:{args.port}/api/health")
    print()
    print("  Protocol: Streamable HTTP (MCP 2025-03-26)")
    print()
    print("-" * 60)

    import uvicorn
    uvicorn.run(
        "server.http_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
