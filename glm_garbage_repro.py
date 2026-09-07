#!/usr/bin/env python3
"""Minimal reproduction: GLM-5.1 on gpu-host-1 returns incoherent garbage, intermittently.

Endpoint:  http://gpu-host-1.example.com:8000/v1   (model: zai-org/GLM-5.1, vLLM)

It intermittently returns incoherent garbage for even a trivial prompt, and it
FLAPS over minutes: a serial single caller sees windows that are fully coherent
(e.g. 5/5) and windows that are fully garbage (e.g. 0/3).

Why this is the server, not the caller
--------------------------------------
This sends the most VANILLA request possible — `model` + one user `messages`
entry + `max_tokens`, nothing else (no thinking params, no extra_body, no tools).
That is the canonical OpenAI chat-completions "hello world"; it cannot be
"communicated poorly". The decisive evidence: the *same identical request*
returns a clean "4" in some trials and gibberish in others. Poor communication
would fail every time, not intermittently. (Run with --thinking-off to confirm
it is also independent of the chat_template_kwargs thinking toggle; cross-check
with `curl` and `scripts/try_iml310.py` to rule out this SDK entirely.)

Run (needs only the `openai` SDK; internal lab hosts reached directly — ensure
no_proxy includes your lab domain):
    python glm_garbage_repro.py                 # 8 vanilla trials
    python glm_garbage_repro.py --n 20
    python glm_garbage_repro.py --host 39        # Kimi (gpu-host-2) for comparison
    python glm_garbage_repro.py --thinking-off   # add enable_thinking=False

Exit: 0 if all coherent, 1 if any garbage/failure.
"""
from __future__ import annotations

import argparse
import sys
import time

from openai import OpenAI

PROMPT = "What is 2+2? Reply with just the number."


def build_url(host: str) -> str:
    h = host.strip()
    if "://" in h:
        h = h.rstrip("/")
        return h if h.endswith("/v1") else h + "/v1"
    if h.isdigit():
        h = f"iml{h}"
    if "." not in h:
        h = f"{h}.lab.example.com"
    return f"http://{h}:8000/v1"


def is_coherent(text: str) -> bool:
    """Healthy answer is a clean, short '4'. Garbage is long and/or not '4'."""
    s = (text or "").strip()
    return s.startswith("4") and len(s) <= 5


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="310", help="lab host: 310 (GLM), 39 (Kimi), or full URL")
    ap.add_argument("--model", default=None, help="override; else auto-discovered from /v1/models")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=2048, help="generous so thinking has room")
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--thinking-off", action="store_true",
                    help="also send chat_template_kwargs.enable_thinking=False")
    args = ap.parse_args()

    base_url = build_url(args.host)
    client = OpenAI(base_url=base_url, api_key="dummy", timeout=60)

    model = args.model
    if model is None:
        try:
            model = client.models.list().data[0].id
        except Exception as e:
            print(f"FATAL: cannot list models at {base_url}: {type(e).__name__}: {e}")
            return 2

    # The request kwargs. Default = pure vanilla (canonical hello-world).
    extra = {}
    if args.thinking_off:
        extra = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}

    print(f"endpoint = {base_url}")
    print(f"model    = {model}")
    print(f"prompt   = {PROMPT!r}   (expect a clean '4')")
    print(f"request  = model + 1 user message + max_tokens={args.max_tokens}"
          f"{' + enable_thinking=False' if args.thinking_off else '  (VANILLA — no extra params)'}")
    print("-" * 74)

    good = 0
    for i in range(args.n):
        t0 = time.time()
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": PROMPT}],
                max_tokens=args.max_tokens,
                **extra,
            )
            text = r.choices[0].message.content or ""
            finish = r.choices[0].finish_reason
            ctok = r.usage.completion_tokens if r.usage else "?"
            ok = is_coherent(text)
            good += ok
            tag = "COHERENT" if ok else "GARBAGE "
            print(f"  [{i+1:2}/{args.n}] {time.time()-t0:5.1f}s  {tag}  finish={finish:>6} ctok={ctok}  {text[:60]!r}")
        except Exception as e:
            print(f"  [{i+1:2}/{args.n}] {time.time()-t0:5.1f}s  FAIL      {type(e).__name__}: {str(e)[:80]}")
        time.sleep(args.sleep)

    print("-" * 74)
    if good == args.n:
        verdict = "all coherent (server healthy right now)"
    elif good == 0:
        verdict = "all garbage (server degraded right now)"
    else:
        verdict = (f"INTERMITTENT — identical request gave {good} coherent and "
                   f"{args.n - good} garbage. Same input, different output => server-side defect.")
    print(f"coherent {good}/{args.n}   -> {verdict}")
    return 0 if good == args.n else 1


if __name__ == "__main__":
    sys.exit(main())
