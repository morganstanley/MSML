"""One-off: measure tokens/sec from the Bedrock Anthropic endpoint the way Alpha Lab hits it.

Reproduces ``src/alpha_lab/provider_bedrock.py:stream_response`` and
``client.py:get_bedrock_client`` faithfully:
  - Bearer token from ``.token_cache.json`` (re-read per request via a
    boto3 ``before-send`` hook), boto3 ``bedrock-runtime`` pointed at the
    AIGW Bedrock endpoint, MS CA bundle for TLS.
  - ``converse_stream`` with the same request shape: system + 1h cachePoint,
    ``inferenceConfig.maxTokens`` scaled the same way, and the same
    extended-thinking block (adaptive ``output_config.effort`` for Opus 4.7/4.8,
    ``thinking.budget_tokens`` for older Claude).

What it measures (per run, then mean over --runs):
  - total     : request -> end of stream (wall clock).
  - overall   : outputTokens / total wall  == tokens/sec, the apples-to-apples metric.
outputTokens INCLUDES thinking tokens (Bedrock has no separate field), so this counts
ALL generated tokens, matching completion_tokens on the OpenAI-compatible side
(scripts/try_iml310.py). A streamed "decode window" rate was intentionally dropped:
reasoning streams as deltas on Bedrock but NOT on the vLLM lab servers (verified:
GLM-5.1 emits 0 reasoning deltas yet counts ~1000 thinking tokens), so a window-based
rate is not comparable across backends. Only all-tokens / total-wall is.

Usage:
  python scripts/bedrock_tps.py
  python scripts/bedrock_tps.py "write 200 words on attention" --effort high --runs 5
  python scripts/bedrock_tps.py --model claude-opus-4-7 --effort none

Exit codes: 0 ok | 1 token problem | 3 all runs failed
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

# --- mirrored from src/alpha_lab/client.py and provider_bedrock.py ---
_AIGW_BEDROCK_ENDPOINT = "https://ai-gateway-dev.example.com/aws"
THINKING_BUDGETS = {"low": 5000, "medium": 16000, "high": 32000}
OPUS_47_EFFORTS = {"low", "medium", "high", "xhigh", "max"}
DEFAULT_MAX_TOKENS = 8_192
OPUS_47_MAX_TOKENS = 64_000
OUTPUT_HEADROOM_TOKENS = 16_384
DEFAULT_MODEL = "claude-opus-4-8"
DEFAULT_PROMPT = (
    "Write a clear, self-contained ~250-word explanation of how the transformer "
    "self-attention mechanism works. No preamble; start with the explanation."
)


def _ca_cert() -> str:
    if platform.system().casefold() == "windows":
        return "\\\\fileshare\\pki\\root-ca.crt"
    return "/etc/pki/ca-trust/certs/internal-ca-chain.crt"


def _load_token(cache_path: Path) -> str:
    if not cache_path.exists():
        print(f"[ERROR] Token cache not found at {cache_path}", file=sys.stderr)
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
        print(f"[ERROR] Token expired {int(time.time() - expires_at)}s ago.", file=sys.stderr)
        sys.exit(1)
    return token


def _build_client(cache_path: Path):
    import boto3
    from botocore.config import Config

    _load_token(cache_path)  # fail fast on a bad cache
    cfg = Config(read_timeout=300, connect_timeout=30, retries={"max_attempts": 3, "mode": "adaptive"})
    client = boto3.client(
        service_name="bedrock-runtime",
        endpoint_url=_AIGW_BEDROCK_ENDPOINT,
        aws_access_key_id="test",
        aws_secret_access_key="test",
        region_name="us-east-1",
        verify=_ca_cert(),
        config=cfg,
    )

    def add_bearer_header(request, **kwargs):
        request.headers["Authorization"] = f"Bearer {_load_token(cache_path)}"

    client.meta.events.register("before-send.*.*", add_bearer_header)
    return client


def _build_kwargs(model: str, system: str, prompt: str, effort: str, max_tokens_override):
    """Exact request shape from provider_bedrock.stream_response."""
    effort_norm = effort.strip().lower() if effort else ""
    is_opus_4_7 = "claude-opus-4-7" in model or "claude-opus-4-8" in model

    if max_tokens_override is not None:
        max_tokens = max_tokens_override
    elif is_opus_4_7:
        max_tokens = OPUS_47_MAX_TOKENS
    elif effort_norm in THINKING_BUDGETS:
        max_tokens = THINKING_BUDGETS[effort_norm] + OUTPUT_HEADROOM_TOKENS
    else:
        max_tokens = DEFAULT_MAX_TOKENS

    kwargs = {
        "modelId": model,
        "system": [{"text": system}, {"cachePoint": {"type": "default", "ttl": "1h"}}],
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {"maxTokens": max_tokens},
    }
    if effort_norm and effort_norm != "none":
        if is_opus_4_7 and effort_norm in OPUS_47_EFFORTS:
            kwargs["additionalModelRequestFields"] = {
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": effort_norm},
            }
        elif effort_norm in THINKING_BUDGETS:
            kwargs["additionalModelRequestFields"] = {
                "thinking": {"type": "enabled", "budget_tokens": THINKING_BUDGETS[effort_norm]},
            }
    return kwargs, max_tokens


def measure_once(client, kwargs) -> dict:
    """Stream one converse_stream call; return timing + token counts."""
    t_req = time.perf_counter()
    resp = client.converse_stream(**kwargs)
    t_first = t_last = None
    text_parts: list[str] = []
    input_tokens = output_tokens = 0
    stop_reason = None
    for event in resp.get("stream"):
        if "contentBlockDelta" in event:
            delta = event["contentBlockDelta"]["delta"]
            if "text" in delta or "reasoningContent" in delta:
                now = time.perf_counter()
                if t_first is None:
                    t_first = now
                t_last = now
                if "text" in delta:
                    text_parts.append(delta["text"])
        elif "messageStop" in event:
            stop_reason = event["messageStop"].get("stopReason")
        elif "metadata" in event:
            usage = event["metadata"].get("usage", {})
            input_tokens = usage.get("inputTokens", 0)
            output_tokens = usage.get("outputTokens", 0)
    t_done = time.perf_counter()

    total = t_done - t_req
    ttft = (t_first - t_req) if t_first else None
    gen_window = (t_last - t_first) if (t_first and t_last and t_last > t_first) else None
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "stop_reason": stop_reason,
        "total": total,
        "ttft": ttft,
        "gen_window": gen_window,
        "overall_tps": (output_tokens / total) if total > 0 else 0.0,
        "decode_tps": (output_tokens / gen_window) if gen_window else None,
        "text_preview": "".join(text_parts)[:160].replace("\n", " "),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--effort", default="high", help="none|low|medium|high|xhigh|max")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=None, help="override; else mirror provider")
    ap.add_argument("--system", default="You are a concise, helpful assistant.")
    ap.add_argument("--token-cache", default="./.token_cache.json")
    args = ap.parse_args()

    cache_path = Path(args.token_cache).resolve()
    client = _build_client(cache_path)
    kwargs, max_tokens = _build_kwargs(args.model, args.system, args.prompt, args.effort, args.max_tokens)

    print(f"endpoint : {_AIGW_BEDROCK_ENDPOINT}")
    print(f"model    : {args.model}")
    print(f"effort   : {args.effort}   (thinking: {kwargs.get('additionalModelRequestFields', 'disabled')})")
    print(f"maxTokens: {max_tokens}   runs: {args.runs}")
    print(f"prompt   : {args.prompt[:80]!r}\n")

    results = []
    for i in range(1, args.runs + 1):
        try:
            r = measure_once(client, kwargs)
        except Exception as e:
            print(f"run {i}: FAILED {type(e).__name__}: {str(e)[:300]}", file=sys.stderr)
            continue
        results.append(r)
        print(
            f"run {i}: out={r['output_tokens']:>5} (incl thinking) in={r['input_tokens']:>4}  "
            f"total={r['total']:6.2f}s  overall={r['overall_tps']:6.1f} tok/s  "
            f"stop={r['stop_reason']}"
        )
        if i == 1:
            print(f"       text: {r['text_preview']!r}")

    if not results:
        print("\nAll runs failed.", file=sys.stderr)
        return 3

    def mean(key):
        vals = [r[key] for r in results if r[key] is not None]
        return statistics.mean(vals) if vals else None

    m_overall = mean("overall_tps")
    m_out = mean("output_tokens")
    print(f"\n=== mean over {len(results)} run(s) ===")
    print(f"out tokens : {m_out:.0f} (incl thinking)")
    print(f"overall    : {m_overall:.1f} tok/s  (output tokens / total wall)")
    print("note: outputTokens includes thinking tokens (Bedrock has no separate field);")
    print("      this all-tokens / total-wall rate matches scripts/try_iml310.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
