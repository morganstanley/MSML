"""
HTTP Server for AriadneMem MCP

Provides:
- MCP over Streamable HTTP (2025-03-26 spec) at /mcp
- REST API for direct access at /api/*
- Health check at /api/health

Transport: Streamable HTTP (single /mcp endpoint for POST, GET, DELETE)
Auth: Bearer token (configured via ARIADNEMEM_API_TOKEN env var)
"""

import asyncio
import json
import os
import sys
import secrets
from typing import Optional
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Header, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

# Add AriadneMem root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from main import AriadneMemSystem
from .mcp_handler import MCPHandler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mcp_config.settings import get_settings


# === Session Management ===

class SessionManager:
    """Manages MCP sessions"""

    def __init__(self):
        self._sessions: dict[str, dict] = {}
        self._lock = asyncio.Lock()

    async def create_session(self, handler: MCPHandler) -> str:
        """Create a new session and return session ID"""
        session_id = secrets.token_urlsafe(32)
        async with self._lock:
            self._sessions[session_id] = {
                "handler": handler,
                "created_at": datetime.utcnow(),
                "last_active": datetime.utcnow(),
                "message_queue": asyncio.Queue(),
            }
        return session_id

    async def get_handler(self, session_id: str) -> Optional[MCPHandler]:
        """Get handler for session"""
        async with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session["last_active"] = datetime.utcnow()
                return session["handler"]
        return None

    async def get_queue(self, session_id: str) -> Optional[asyncio.Queue]:
        """Get message queue for session"""
        async with self._lock:
            session = self._sessions.get(session_id)
            return session["message_queue"] if session else None

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        async with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
        return False

    async def cleanup_expired(self, max_age_minutes: int = 30):
        """Remove expired sessions"""
        async with self._lock:
            now = datetime.utcnow()
            expired = [
                sid for sid, s in self._sessions.items()
                if (now - s["last_active"]).total_seconds() > max_age_minutes * 60
            ]
            for sid in expired:
                del self._sessions[sid]


# === Global State ===

settings = get_settings()
session_manager = SessionManager()

# Shared AriadneMem system instance
_system: Optional[AriadneMemSystem] = None


def get_system() -> AriadneMemSystem:
    """Get or create the AriadneMem system"""
    global _system
    if _system is None:
        print("Initializing AriadneMem system...")
        _system = AriadneMemSystem(clear_db=False)
        print("AriadneMem system ready")
    return _system


# === Auth Helper ===

def verify_token(authorization: Optional[str]) -> bool:
    """Verify Bearer token"""
    if not settings.api_token:
        return True  # No auth configured

    if not authorization:
        return False

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return False

    return parts[1] == settings.api_token


# === Helper ===

def _is_initialize_request(data: dict | list) -> bool:
    """Check if message is an initialize request"""
    if isinstance(data, list):
        return any(isinstance(item, dict) and item.get("method") == "initialize" for item in data)
    return isinstance(data, dict) and data.get("method") == "initialize"


def _is_notification_only(data: dict | list) -> bool:
    """Check if message contains only notifications (no requests)"""
    items = data if isinstance(data, list) else [data]
    for item in items:
        if isinstance(item, dict) and "method" in item and "id" in item:
            return False
    return True


# === Lifecycle ===

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle"""
    print("AriadneMem MCP Server starting...")
    print(f"  LLM Model: {settings.llm_model}")
    print(f"  Embedding: {settings.embedding_model}")
    print(f"  Transport: Streamable HTTP (MCP 2025-03-26)")

    # Start session cleanup
    async def cleanup_loop():
        while True:
            await asyncio.sleep(60)
            await session_manager.cleanup_expired()

    cleanup_task = asyncio.create_task(cleanup_loop())

    yield

    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    print("AriadneMem MCP Server stopped")


# === FastAPI App ===

app = FastAPI(
    title="AriadneMem MCP Server",
    description="Graph-Based Lifelong Memory Service for LLM Agents",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === Health & Info ===

@app.get("/api/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "server": "AriadneMem MCP",
        "version": "1.0.0",
    }


@app.get("/api/server/info")
async def server_info():
    """Server information"""
    return {
        "name": "AriadneMem MCP Server",
        "version": "1.0.0",
        "protocol_version": "2025-03-26",
        "transport": "Streamable HTTP",
        "paper": "AriadneMem: Threading the Maze of Lifelong Memory for LLM Agents",
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
        "features": {
            "entropy_aware_gating": True,
            "conflict_aware_coarsening": True,
            "steiner_bridge_discovery": True,
            "multi_hop_path_mining": True,
            "topology_aware_synthesis": True,
        },
    }


# === MCP Protocol Endpoints (Streamable HTTP - 2025-03-26) ===

@app.post("/mcp")
async def mcp_post(
    request: Request,
    authorization: Optional[str] = Header(None),
    mcp_session_id: Optional[str] = Header(None, alias="Mcp-Session-Id"),
):
    """
    MCP POST endpoint - handles JSON-RPC 2.0 messages.

    For initialize: creates new session, returns Mcp-Session-Id header.
    For other requests: requires valid Mcp-Session-Id header.
    """
    # Validate Accept header
    accept = request.headers.get("accept", "")
    if "application/json" not in accept and "text/event-stream" not in accept:
        raise HTTPException(status_code=406, detail="Accept header must include application/json or text/event-stream")

    # Authenticate
    if not verify_token(authorization):
        raise HTTPException(status_code=401, detail="Invalid or missing Bearer token")

    # Parse body
    try:
        body = await request.body()
        data = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as e:
        return JSONResponse(status_code=400, content={
            "jsonrpc": "2.0",
            "error": {"code": -32700, "message": f"Parse error: {e}"},
        })

    # Handle initialization
    if _is_initialize_request(data):
        system = get_system()
        handler = MCPHandler(system)
        # Use client-provided session ID if present, otherwise generate one
        session_id = mcp_session_id or secrets.token_urlsafe(32)
        # Remove old session if client is re-initializing
        await session_manager.delete_session(session_id)
        async with session_manager._lock:
            session_manager._sessions[session_id] = {
                "handler": handler,
                "created_at": datetime.utcnow(),
                "last_active": datetime.utcnow(),
                "message_queue": asyncio.Queue(),
            }
        handler.initialized = True

        response_str = await handler.handle_message(json.dumps(data))
        response_data = json.loads(response_str)

        return JSONResponse(
            content=response_data,
            headers={"Mcp-Session-Id": session_id},
        )

    # Non-init requests require session ID
    if not mcp_session_id:
        raise HTTPException(status_code=400, detail="Mcp-Session-Id header required")

    handler = await session_manager.get_handler(mcp_session_id)
    if not handler:
        # Auto-create session for clients that skip initialize
        system = get_system()
        handler = MCPHandler(system)
        handler.initialized = True
        async with session_manager._lock:
            session_manager._sessions[mcp_session_id] = {
                "handler": handler,
                "created_at": datetime.utcnow(),
                "last_active": datetime.utcnow(),
                "message_queue": asyncio.Queue(),
            }


    # Notifications return 202
    if _is_notification_only(data):
        await handler.handle_message(json.dumps(data))
        return Response(status_code=202)

    # Process request
    response_str = await handler.handle_message(json.dumps(data))
    response_data = json.loads(response_str)

    return JSONResponse(
        content=response_data,
        headers={"Mcp-Session-Id": mcp_session_id},
    )


@app.get("/mcp")
async def mcp_get(
    request: Request,
    authorization: Optional[str] = Header(None),
    mcp_session_id: Optional[str] = Header(None, alias="Mcp-Session-Id"),
):
    """
    MCP GET endpoint - SSE stream for server-to-client messages.
    """
    accept = request.headers.get("accept", "")
    if "text/event-stream" not in accept:
        raise HTTPException(status_code=406, detail="Accept header must include text/event-stream")

    if not verify_token(authorization):
        raise HTTPException(status_code=401, detail="Invalid or missing Bearer token")

    if not mcp_session_id:
        raise HTTPException(status_code=400, detail="Mcp-Session-Id header required")

    queue = await session_manager.get_queue(mcp_session_id)
    if not queue:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    async def event_generator():
        yield ": keepalive\n\n"
        event_counter = 0
        while True:
            try:
                message = await asyncio.wait_for(queue.get(), timeout=15.0)
                event_counter += 1
                yield f"id: {mcp_session_id}-{event_counter}\n"
                yield f"event: message\n"
                yield f"data: {json.dumps(message)}\n\n"
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Mcp-Session-Id": mcp_session_id,
        },
    )


@app.delete("/mcp")
async def mcp_delete(
    authorization: Optional[str] = Header(None),
    mcp_session_id: Optional[str] = Header(None, alias="Mcp-Session-Id"),
):
    """Terminate an MCP session"""
    if not verify_token(authorization):
        raise HTTPException(status_code=401, detail="Invalid or missing Bearer token")

    if not mcp_session_id:
        raise HTTPException(status_code=400, detail="Mcp-Session-Id header required")

    deleted = await session_manager.delete_session(mcp_session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")

    return Response(status_code=204)


# === Entry Point ===

def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the HTTP server"""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
