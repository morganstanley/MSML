"""Isolated worker for peak-RSS benchmark runs."""

from __future__ import annotations

import gc
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, TextIO

from tools.benchmark_common import current_rss_bytes, peak_rss_bytes
from tools.benchmark_suites import SUITES


def _event(protocol: TextIO, name: str, **fields: object) -> None:
    print(
        json.dumps({"event": name, **fields}, separators=(",", ":")),
        file=protocol,
        flush=True,
    )


def _required_string(job: dict[str, Any], key: str) -> str:
    value = job.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"RSS worker requires string field {key!r}")
    return value


def _run(protocol: TextIO, job: dict[str, Any]) -> None:
    suite_name = _required_string(job, "suite")
    try:
        suite = SUITES[suite_name]
    except KeyError as error:
        raise ValueError(f"unknown benchmark suite: {suite_name!r}") from error

    implementation = _required_string(job, "implementation")
    item = suite.item_named(_required_string(job, "item"))
    corpus = _required_string(job, "corpus")
    input_path = Path(_required_string(job, "path"))
    if not input_path.is_file():
        raise FileNotFoundError(f"corpus does not exist: {input_path}")
    if implementation not in suite.implementations:
        raise ValueError(f"unknown implementation: {implementation}")
    if not suite.supports(implementation, item):
        raise ValueError(f"{implementation} does not support {item.name}")

    output_value = job.get("output_path")
    output_path = Path(output_value) if isinstance(output_value, str) else None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    dfa_value = job.get("dfa", False)
    if not isinstance(dfa_value, bool):
        raise ValueError("RSS worker field 'dfa' must be boolean")
    dfa = dfa_value
    dump_value = job.get("dump", False)
    if not isinstance(dump_value, bool):
        raise ValueError("RSS worker field 'dump' must be boolean")
    dump = dump_value
    if dfa and dump:
        raise ValueError("RSS worker cannot combine dump and DFA modes")
    if implementation not in suite.rss_implementations_for(dfa=dfa):
        raise ValueError(
            f"{implementation} is not part of the selected "
            f"{'DFA' if dfa else 'fast'} comparison"
        )
    if suite.force_dfa and not dfa:
        raise ValueError(f"{suite_name} RSS benchmarks require the DFA path")
    output_extension = suite.rss_output_extension(
        implementation,
        dfa=dfa,
        dump=dump,
    )
    if (output_path is None) != (output_extension is None):
        expected = "array output" if output_extension is None else "file output"
        raise ValueError(
            f"{suite_name} RSS benchmarks require {expected} in this mode"
        )

    suite.preload(
        implementation,
        item,
        phase="rss",
        output_path=output_path,
    )
    encoder = suite.construct(
        implementation,
        item,
        dfa=dfa,
        profile=False,
    )
    gc.collect()
    model_rss = current_rss_bytes()
    initialization_peak_rss = peak_rss_bytes()

    # Python reference implementations receive a ready string, whereas
    # Gigatoken and Hiriluk receive the file path. Retain the prepared input so
    # the final high-water mark reflects each implementation's real API.
    input_value = suite.prepare_input(implementation, input_path)

    rss_run = suite.encode_rss(
        implementation,
        encoder,
        input_value,
        output_path,
    )

    # Read the kernel-maintained high-water mark while the encoder, prepared
    # input, and native output are all still live.
    retained = rss_run.retained
    _event(
        protocol,
        "done",
        implementation=implementation,
        item=item.name,
        corpus=corpus,
        model_rss=model_rss,
        initialization_peak_rss=initialization_peak_rss,
        peak_rss=peak_rss_bytes(),
        tokens=int(rss_run.tokens),
        output_mode=rss_run.output_mode,
    )
    _ = (encoder, input_value, retained)


def main() -> None:
    protocol = sys.stdout
    try:
        line = sys.stdin.readline()
        if not line:
            raise ValueError("RSS worker did not receive a job")
        job = json.loads(line)
        if not isinstance(job, dict):
            raise ValueError("RSS worker input must be a JSON object")
        # Libraries occasionally print progress messages.  Keep stdout as a
        # machine-readable protocol by redirecting those messages to stderr.
        with redirect_stdout(sys.stderr):
            _run(protocol, job)
    except Exception as error:
        _event(protocol, "error", message=str(error))
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
