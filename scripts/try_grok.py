"""Standalone smoke test for grok via MS AI Gateway.

Mirrors the auth + transport patterns used by ``src/alpha_lab/client.py``:
  - Bearer token loaded from ``.token_cache.json`` (looked up via -t or
    cwd-relative ``./.token_cache.json``).
  - ``httpx.Client`` with the MS CA bundle for cert verification.
  - Event-hook injects ``Authorization: Bearer <token>`` on every request
    so a refreshed token is picked up automatically.
  - OpenAI SDK pointed at ``https://ai-gateway-dev.example.com/xai/v1``.
  - ``X-Ms-Assert-Username`` header set from $USER (LDAP lookup skipped
    here to keep the script standalone — the main system does that via
    ms.directory, which isn't always available).

Usage:
  # From a directory that contains .token_cache.json (token must be valid):
  python scripts/try_grok.py

  # Or pass an explicit token cache path + model:
  python scripts/try_grok.py --token-cache /path/to/.token_cache.json \\
      --model grok-3 --endpoint chat

Exit codes:
  0 — at least one grok response was successfully received
  1 — token not found / invalid / expired
  2 — listing models failed
  3 — chat completion failed
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import httpx
from openai import OpenAI


# Matches src/alpha_lab/client.py — different gateway path (xai instead of openai).
_AIGW_XAI_URL = "https://ai-gateway-dev.example.com/xai/v1"


def _ca_cert() -> str:
    """Same logic as client._get_ca_cert_bundle."""
    if platform.system().casefold() == "windows":
        return "\\\\fileshare\\pki\\root-ca.crt"
    return "/etc/pki/ca-trust/certs/internal-ca-chain.crt"


def _load_token(cache_path: Path) -> str:
    """Load a non-expired bearer token from the cache. Exits on failure."""
    if not cache_path.exists():
        print(f"[ERROR] Token cache not found at {cache_path}", file=sys.stderr)
        print(
            "Run scripts/token_refresh.sh as proid first, or pass "
            "--token-cache <path> explicitly.",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        data = json.loads(cache_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print(f"[ERROR] Could not parse {cache_path}: {e}", file=sys.stderr)
        sys.exit(1)

    token = data.get("bearer_token")
    expires_at = data.get("expires_at", 0)
    if not token:
        print(f"[ERROR] {cache_path} has no bearer_token field", file=sys.stderr)
        sys.exit(1)
    if expires_at and time.time() >= expires_at:
        remaining = int(time.time() - expires_at)
        print(
            f"[ERROR] Token expired {remaining}s ago. "
            "Run scripts/token_refresh.sh to refresh.",
            file=sys.stderr,
        )
        sys.exit(1)
    remaining = int(expires_at - time.time()) if expires_at else -1
    print(
        f"[OK] Loaded bearer token from {cache_path} "
        f"(expires in {remaining // 60}m {remaining % 60}s)"
        if remaining > 0
        else f"[OK] Loaded bearer token from {cache_path} (no expiry recorded)"
    )
    return token


def _build_client(cache_path: Path) -> OpenAI:
    """Build an OpenAI SDK client pointed at the xai gateway endpoint.

    The token is re-read on every request via the event hook, so refreshes
    by a running ``token_refresh.sh`` daemon are picked up without restart.
    """
    # Validate the token up front so the script fails fast on a bad cache.
    _load_token(cache_path)

    def _inject_token(request: httpx.Request) -> None:
        token = _load_token(cache_path)  # re-read each time (cheap; file-system)
        request.headers["Authorization"] = f"Bearer {token}"

    http_client = httpx.Client(
        verify=_ca_cert(),
        event_hooks={"request": [_inject_token]},
        timeout=httpx.Timeout(60.0, read=120.0),
    )

    userid = os.environ.get("USER", "unknown")

    return OpenAI(
        base_url=_AIGW_XAI_URL,
        api_key="Dummy",  # gateway uses Bearer header, not api_key
        http_client=http_client,
        default_headers={"X-Ms-Assert-Username": userid},
        timeout=300,
    )


def _list_grok_models(client: OpenAI) -> list[str]:
    """Try ``GET /models`` and return the model ids that look like grok.

    The xai gateway path may not implement /models — in that case we
    return an empty list and let the caller fall through to a default
    model name. Models-list 404 is informational, not a fatal error.
    """
    try:
        resp = client.models.list()
    except Exception as e:
        print(
            f"[INFO] /xai/v1/models not available (skipping): "
            f"{str(e)[:200]}"
        )
        return []
    ids = sorted({m.id for m in resp.data})
    print(f"[OK] /xai/v1/models returned {len(ids)} models")
    grokish = [m for m in ids if "grok" in m.lower()]
    if grokish:
        print(f"     grok-like models: {grokish}")
    else:
        print("     (none of them have 'grok' in the name — printing all:)")
        for m in ids[:20]:
            print(f"       - {m}")
    return ids


def _try_chat(client: OpenAI, model: str) -> bool:
    """Make a chat.completions call and print the result."""
    print(f"\n[chat.completions] model={model}")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a concise assistant."},
                {
                    "role": "user",
                    "content": (
                        "In one sentence: who are you and which model are you "
                        "running on?"
                    ),
                },
            ],
            max_tokens=200,
        )
    except Exception as e:
        print(f"[ERROR] chat.completions.create failed: {e}", file=sys.stderr)
        return False
    if not resp.choices:
        print(f"[ERROR] empty choices in response: {resp}", file=sys.stderr)
        return False
    msg = resp.choices[0].message
    print(f"[OK]  response.choices[0].message.content: {msg.content!r}")
    if hasattr(resp, "model") and resp.model:
        print(f"[OK]  response.model: {resp.model}")
    if hasattr(resp, "usage") and resp.usage:
        u = resp.usage
        print(
            f"[OK]  usage: prompt={u.prompt_tokens} "
            f"completion={u.completion_tokens} total={u.total_tokens}"
        )
    return True


def _try_responses(client: OpenAI, model: str) -> bool:
    """Make a /responses call and print the result. Mirrors what the main
    OpenAIProvider uses internally."""
    print(f"\n[responses] model={model}")
    try:
        resp = client.responses.create(
            model=model,
            input=[
                {
                    "role": "user",
                    "content": (
                        "Reply with one short sentence acknowledging you "
                        "received this message."
                    ),
                }
            ],
            store=False,
        )
    except Exception as e:
        print(f"[ERROR] responses.create failed: {e}", file=sys.stderr)
        return False

    # The Responses API returns a structured object — pull the text out
    # the same way provider_openai.py does.
    text = ""
    try:
        for item in resp.output:
            if getattr(item, "type", "") == "message":
                for content in item.content:
                    if getattr(content, "type", "") == "output_text":
                        text += getattr(content, "text", "") or ""
    except Exception as e:
        print(f"[WARN] could not parse resp.output: {e}", file=sys.stderr)
    if not text:
        # Fall back to whatever the SDK exposes as a convenience accessor
        text = getattr(resp, "output_text", "") or ""
    if text:
        print(f"[OK]  responses output text: {text!r}")
    else:
        print(f"[WARN] no text extracted; raw repr (truncated): {str(resp)[:400]}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--token-cache",
        default="./.token_cache.json",
        help="Path to the bearer-token JSON cache (default: ./.token_cache.json)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "grok model id to test. If omitted, the script picks the first "
            "grok-like id returned by /models, or falls back to 'grok-3' as "
            "a guess."
        ),
    )
    parser.add_argument(
        "--endpoint",
        choices=("chat", "responses", "both"),
        default="both",
        help="Which endpoint(s) to exercise (default: both)",
    )
    args = parser.parse_args()

    cache_path = Path(args.token_cache).resolve()
    print(f"AIGW endpoint: {_AIGW_XAI_URL}")
    print(f"Token cache:   {cache_path}")
    print(f"USER:          {os.environ.get('USER', 'unknown')}\n")

    client = _build_client(cache_path)

    ids = _list_grok_models(client)
    if args.model:
        model = args.model
    else:
        grok_ids = [m for m in ids if "grok" in m.lower()]
        if grok_ids:
            model = grok_ids[0]
        else:
            model = "grok-3"  # last-ditch guess; user can override with --model
            print(f"[WARN] no grok-* in /models; trying {model!r} as a guess")

    ok_chat = ok_resp = True
    if args.endpoint in ("chat", "both"):
        ok_chat = _try_chat(client, model)
    if args.endpoint in ("responses", "both"):
        ok_resp = _try_responses(client, model)

    if not (ok_chat and ok_resp):
        return 3
    print("\nAll requested endpoints succeeded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
