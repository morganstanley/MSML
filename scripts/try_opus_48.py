"""Standalone smoke test for claude-opus-4-8 via Bedrock through the MS AI Gateway.

Mirrors the auth + transport patterns used by
``src/alpha_lab/client.py:get_bedrock_client``:
  - Bearer token loaded from ``.token_cache.json`` (looked up via -t or
    cwd-relative ``./.token_cache.json``).
  - boto3 ``bedrock-runtime`` client pointed at the AIGW Bedrock endpoint.
  - MS CA bundle for cert verification.
  - ``before-send`` event-hook injects ``Authorization: Bearer <token>``
    on every request so a refreshed token is picked up automatically.

What it exercises:
  1. ``converse`` (one-shot, no thinking) — confirms the model name resolves
     and basic round-trip works.
  2. ``converse`` with ``additionalModelRequestFields = {"thinking":
     {"type": "adaptive"}, "output_config": {"effort": "high"}}`` — confirms
     Opus 4.8 accepts the adaptive-thinking schema the provider sends.

Usage:
  python scripts/try_opus_48.py
  python scripts/try_opus_48.py --token-cache /path/to/.token_cache.json \\
      --model claude-opus-4-8

Exit codes:
  0 — both calls succeeded
  1 — token not found / invalid / expired
  3 — at least one converse call failed
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path


# Matches src/alpha_lab/client.py:_AIGW_BEDROCK_ENDPOINT
_AIGW_BEDROCK_ENDPOINT = "https://ai-gateway-dev.example.com/aws"
DEFAULT_MODEL = "claude-opus-4-8"


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
        elapsed = int(time.time() - expires_at)
        print(
            f"[ERROR] Token expired {elapsed}s ago. "
            "Run scripts/token_refresh.sh to refresh.",
            file=sys.stderr,
        )
        sys.exit(1)
    remaining = int(expires_at - time.time()) if expires_at else -1
    if remaining > 0:
        print(
            f"[OK] Loaded bearer token from {cache_path} "
            f"(expires in {remaining // 60}m {remaining % 60}s)"
        )
    else:
        print(f"[OK] Loaded bearer token from {cache_path} (no expiry recorded)")
    return token


def _build_client(cache_path: Path):
    """Build a boto3 bedrock-runtime client pointed at the AIGW endpoint.

    The token is re-read on every request via the before-send hook, so a
    refresher daemon's writes are picked up without restart.
    """
    import boto3
    from botocore.config import Config

    # Validate the token up front so the script fails fast on a bad cache.
    _load_token(cache_path)

    bedrock_config = Config(
        read_timeout=300,
        connect_timeout=30,
        retries={"max_attempts": 3, "mode": "adaptive"},
    )
    client = boto3.client(
        service_name="bedrock-runtime",
        endpoint_url=_AIGW_BEDROCK_ENDPOINT,
        aws_access_key_id="test",
        aws_secret_access_key="test",
        region_name="us-east-1",
        verify=_ca_cert(),
        config=bedrock_config,
    )

    def add_bearer_header(request, **kwargs):
        token = _load_token(cache_path)
        request.headers["Authorization"] = f"Bearer {token}"

    client.meta.events.register("before-send.*.*", add_bearer_header)
    return client


def _try_converse_basic(client, model: str) -> bool:
    """One-shot converse, no thinking. Confirms the model name resolves."""
    print(f"\n[converse — no thinking]  modelId={model!r}")
    try:
        resp = client.converse(
            modelId=model,
            system=[{"text": "You are a concise assistant."}],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "text": (
                                "In one sentence: who are you and which model "
                                "are you running on?"
                            )
                        }
                    ],
                }
            ],
            inferenceConfig={"maxTokens": 200},
        )
    except Exception as e:
        print(f"[ERROR] converse failed: {e}", file=sys.stderr)
        return False
    return _print_response(resp)


def _try_converse_adaptive(client, model: str) -> bool:
    """Converse with adaptive-thinking + output_config.effort='high'.

    This is the exact schema ``provider_bedrock.py`` sends for Opus 4.7/4.8.
    If Opus 4.8 rejects this payload, the main system will too.
    """
    print(f"\n[converse — adaptive thinking effort=high]  modelId={model!r}")
    try:
        resp = client.converse(
            modelId=model,
            system=[{"text": "You are a concise assistant."}],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "text": (
                                "Briefly think through this then reply: "
                                "what's 2+2? Show one short sentence of "
                                "reasoning, then the answer."
                            )
                        }
                    ],
                }
            ],
            inferenceConfig={"maxTokens": 64_000},
            additionalModelRequestFields={
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": "high"},
            },
        )
    except Exception as e:
        print(f"[ERROR] converse(adaptive) failed: {e}", file=sys.stderr)
        return False
    return _print_response(resp)


def _print_response(resp) -> bool:
    """Pretty-print whatever the model produced."""
    out = resp.get("output", {}) or {}
    msg = out.get("message", {}) or {}
    blocks = msg.get("content", []) or []
    found_text = False
    for i, block in enumerate(blocks):
        if "text" in block and block["text"]:
            print(f"[OK]  output.message.content[{i}].text: {block['text']!r}")
            found_text = True
        elif "reasoningContent" in block:
            rc = block["reasoningContent"]
            rt = rc.get("reasoningText", {}).get("text", "") if isinstance(rc, dict) else ""
            preview = (rt or "")[:200].replace("\n", " ")
            print(f"[OK]  output.message.content[{i}].reasoning (preview): {preview!r}")
    usage = resp.get("usage") or {}
    if usage:
        print(
            f"[OK]  usage: input={usage.get('inputTokens')} "
            f"output={usage.get('outputTokens')} "
            f"total={usage.get('totalTokens')}"
        )
    stop = resp.get("stopReason")
    if stop:
        print(f"[OK]  stopReason: {stop}")
    if not found_text:
        print(
            f"[WARN] no text block in response; raw repr (truncated): "
            f"{str(resp)[:500]}",
            file=sys.stderr,
        )
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--token-cache", default="./.token_cache.json",
        help="Path to the bearer-token JSON cache (default: ./.token_cache.json)",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Bedrock model id (default: {DEFAULT_MODEL!r}). "
             f"Try also 'anthropic.claude-opus-4-8-v1' or "
             f"'us.anthropic.claude-opus-4-8-v1' if the short form 404s.",
    )
    parser.add_argument(
        "--skip-adaptive", action="store_true",
        help="Skip the adaptive-thinking call (do the basic call only)",
    )
    args = parser.parse_args()

    cache_path = Path(args.token_cache).resolve()
    print(f"AIGW Bedrock endpoint: {_AIGW_BEDROCK_ENDPOINT}")
    print(f"Token cache:           {cache_path}")
    print(f"USER:                  {os.environ.get('USER', 'unknown')}")
    print(f"Target model:          {args.model}\n")

    client = _build_client(cache_path)

    ok_basic = _try_converse_basic(client, args.model)
    ok_adaptive = True
    if not args.skip_adaptive:
        ok_adaptive = _try_converse_adaptive(client, args.model)

    if not (ok_basic and ok_adaptive):
        return 3
    print("\nAll requested calls succeeded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
