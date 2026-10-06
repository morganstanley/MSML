"""Fixed isolated-worker entry point for both Python benchmarks."""

from __future__ import annotations

import json
import sys
from contextlib import redirect_stdout

from tools.throughput_bench import execute_worker_job
from tools.benchmark_suites import SUITES


def main() -> None:
    try:
        job = json.load(sys.stdin)
        if not isinstance(job, dict):
            raise ValueError("worker input must be a JSON object")
        suite_name = job.get("suite")
        if not isinstance(suite_name, str) or suite_name not in SUITES:
            raise ValueError(f"unknown benchmark suite: {suite_name!r}")

        # Keep incidental library output away from the machine-readable result.
        with redirect_stdout(sys.stderr):
            result = execute_worker_job(SUITES[suite_name], job)
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
