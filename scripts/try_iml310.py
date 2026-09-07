"""Talk to an OpenAI-compatible lab endpoint and report tokens/sec.

Default target: gpu-host-1.example.com:8000  (api_key "dummy", serves zai-org/GLM-5.1).
Internal lab host, plain HTTP, no proxy (internal lab hosts are reached directly).
The served model is auto-discovered via /v1/models, so pointing at a different host
picks up whatever model lives there.

Usage:
  # Ask it something — everything else is handled by generous defaults:
  python scripts/try_iml310.py "what is the capital of France?"

  # Hit a different lab host (gpu-host-2 instead of gpu-host-1); model auto-discovered there:
  python scripts/try_iml310.py --host 39 "summarize relativity in one line"

  # No prompt → run a full connectivity smoke test instead:
  python scripts/try_iml310.py --host 39

Diagnostics (model, usage, elapsed, tok/s) go to stderr; the answer goes to stdout.

Overridable defaults (rarely needed):
  --host 310|39|<name>   --base-url http://host:port/v1   --model NAME   --api-key KEY
  --max-tokens 8192   --timeout 180   --system "system prompt"

Exit codes: 0 ok | 1 chat call failed | 2 can't reach / list models (smoke test)
"""
from __future__ import annotations

import argparse
import sys
import time

from openai import OpenAI

DEFAULT_HOST = "310"
DEFAULT_MODEL = "zai-org/GLM-5.1"  # gpu-host-1's model; only used if /v1/models can't be listed


def build_base_url(host: str) -> str:
    """Turn --host into a base URL. Accepts '310', '39', 'gpu-host-2', a hostname, or a full URL."""
    h = host.strip()
    if "://" in h:  # already a full URL
        h = h.rstrip("/")
        return h if h.endswith("/v1") else h + "/v1"
    if h.isdigit():  # bare lab number -> imlNNN
        h = f"iml{h}"
    if "." not in h:  # short name -> qualify with the lab domain
        h = f"{h}.lab.example.com"
    return f"http://{h}:8000/v1"


def _tps_line(usage, elapsed: float) -> str:
    """'[usage ...] [N.Ns, T tok/s]' — output tok/s is completion_tokens over wall time."""
    bits = ""
    if usage:
        bits = f"usage prompt={usage.prompt_tokens} completion={usage.completion_tokens} total={usage.total_tokens}"
    tps = ""
    if usage and elapsed > 0:
        tps = f", {usage.completion_tokens / elapsed:.1f} tok/s (output/wall)"
    return f"[{bits}] [{elapsed:.2f}s{tps}]"


def resolve_model(client: OpenAI, override: str | None) -> str:
    """Explicit --model wins; else first model the server lists; else the known default."""
    if override:
        return override
    try:
        ids = [m.id for m in client.models.list().data]
        if ids:
            return ids[0]
        reason = "server listed no models"
    except Exception as e:
        reason = f"/v1/models failed: {type(e).__name__}"
    print(f"[{reason}; falling back to default model {DEFAULT_MODEL!r}]", file=sys.stderr)
    return DEFAULT_MODEL


def ask(client: OpenAI, model: str, prompt: str, system: str | None, max_tokens: int) -> int:
    """Stream one chat completion. Answer -> stdout, tok/s diagnostics -> stderr.

    Streams (with include_usage) so total wall is measured request -> last chunk,
    matching scripts/bedrock_tps.py. Throughput = completion_tokens / total wall.
    completion_tokens is the server's FULL output count and INCLUDES thinking tokens
    for reasoning models (verified on GLM-5.1: a reply with ~190 visible tokens
    reported completion_tokens=1206), so this is apples-to-apples with Bedrock's
    outputTokens-based rate — all generated tokens over wall time, both sides.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    print(f"[model={model} max_tokens={max_tokens}]", file=sys.stderr)
    t0 = time.perf_counter()
    parts: list[str] = []
    usage = None
    finish = None
    try:
        stream = client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens,
            stream=True, stream_options={"include_usage": True},
        )
        for chunk in stream:
            if getattr(chunk, "usage", None) is not None:
                usage = chunk.usage
            if not chunk.choices:
                continue
            ch = chunk.choices[0]
            if ch.finish_reason:
                finish = ch.finish_reason
            piece = getattr(ch.delta, "content", None)
            if piece:
                parts.append(piece)
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {str(e)[:400]}", file=sys.stderr)
        return 1
    total = time.perf_counter() - t0

    answer = "".join(parts)
    if answer:
        print(answer)
    else:
        # GLM-5.1 is a reasoning model: a tight max_tokens gets spent thinking
        # before any visible content is emitted (finish_reason="length").
        print(
            f"(empty content; finish_reason={finish!r} — raise --max-tokens if it was cut off)",
            file=sys.stderr,
        )
    if usage and total > 0:
        out = usage.completion_tokens
        print(
            f"[out={out} (incl thinking) in={usage.prompt_tokens}  "
            f"total={total:.2f}s  overall={out / total:.1f} tok/s (output/wall)]",
            file=sys.stderr,
        )
    else:
        print(
            f"[total={total:.2f}s; no usage returned (server may not support include_usage)]",
            file=sys.stderr,
        )
    return 0


def smoke_test(client: OpenAI, model_override: str | None) -> int:
    """No-prompt path: prove the endpoint is reachable and OpenAI-compatible."""
    print("\n[1] GET /v1/models ...")
    model = model_override
    try:
        ids = [m.id for m in client.models.list().data]
        print(f"    OK — {len(ids)} model(s): {ids}")
        if model is None and ids:
            model = ids[0]
    except Exception as e:
        print(f"    FAILED: {type(e).__name__}: {str(e)[:300]}")
        if model is None:
            return 2

    print(f"\n[2] POST /v1/chat/completions  model={model!r} ...")
    chat_ok = False
    try:
        t0 = time.perf_counter()
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with exactly: HELLO WORLD"}],
            max_tokens=1024,
        )
        elapsed = time.perf_counter() - t0
        print(f"    OK — response: {r.choices[0].message.content!r}")
        print("    " + _tps_line(getattr(r, "usage", None), elapsed))
        chat_ok = True
    except Exception as e:
        print(f"    FAILED: {type(e).__name__}: {str(e)[:300]}")

    print(f"\n[3] POST /v1/completions  model={model!r} ...")
    comp_ok = False
    try:
        r = client.completions.create(model=model, prompt="Hello, ", max_tokens=16)
        print(f"    OK — text: {r.choices[0].text!r}")
        comp_ok = True
    except Exception as e:
        print(f"    FAILED: {type(e).__name__}: {str(e)[:200]}")

    print()
    if chat_ok or comp_ok:
        print("RESULT: endpoint is reachable and OpenAI-compatible.")
        return 0
    print("RESULT: reachable but neither chat nor completions worked.")
    return 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt", nargs="?", default=None, help="what to ask; omit to run a smoke test")
    ap.add_argument("--host", default=DEFAULT_HOST, help="lab host: 310, 39, a name, or full URL")
    ap.add_argument("--base-url", default=None, help="full override; else built from --host")
    ap.add_argument("--api-key", default="dummy")
    ap.add_argument("--model", default=None, help="override; else first served model, else the known default")
    ap.add_argument("--system", default=None, help="optional system prompt")
    ap.add_argument("--max-tokens", type=int, default=8192, help="generous by default (reasoning model)")
    ap.add_argument("--timeout", type=float, default=180.0)
    args = ap.parse_args()

    base_url = args.base_url or build_base_url(args.host)
    client = OpenAI(base_url=base_url, api_key=args.api_key, timeout=args.timeout)

    if args.prompt is None:
        print(f"base_url: {base_url}\napi_key:  {args.api_key!r}")
        return smoke_test(client, args.model)

    model = resolve_model(client, args.model)
    return ask(client, model, args.prompt, args.system, args.max_tokens)


if __name__ == "__main__":
    sys.exit(main())
