"""Turn throughput tables from logs/hiriluk-throughput-*.out into LaTeX tables.

Each throughput log (see `tools/throughput_bench.py`) contains a plain-text
table printed by `benchmark_common.print_table`: a header row, a `---` rule,
then one data row per (item, corpus, impl) triple. This module extracts that
table and renders it as the two-column comparison table used in the paper.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"

CORPORA = ("github", "english", "chinese")
CORPUS_LABELS = {"github": "Github", "english": "English", "chinese": "Chinese"}
TTFT_ENCODING_BASE = {
    "r50k": "r50k_base",
    "p50k": "p50k_base",
    "cl100k": "cl100k_base",
    "o200k": "o200k_base",
}
MODEL_LABELS = {
    "gpt2": "GPT2",
    "roberta": "RoBERTa",
    "starcoder": "StarCoder",
    "mistral": "Mistral",
    "llama4": "Llama4",
    "qwen": "Qwen-3",
    "gptoss": "gpt-oss",
}


def _latest(pattern: str) -> Path:
    matches = sorted(glob.glob(str(LOGS_DIR / pattern)))
    if not matches:
        raise FileNotFoundError(f"no log matches {pattern!r} under {LOGS_DIR}")
    return Path(matches[-1])


def parse_table(path: Path) -> list[dict[str, str]]:
    """Parse the `impl ... output` table out of a throughput log."""

    lines = path.read_text().splitlines()
    header_index = next(i for i, line in enumerate(lines) if line.startswith("impl"))
    header = lines[header_index].split()
    rows = []
    for line in lines[header_index + 2 :]:
        if not line.strip():
            break
        rows.append(dict(zip(header, line.split())))
    return rows


def _group(rows: list[dict[str, str]], item_key: str) -> dict[str, dict[str, dict[str, dict[str, str]]]]:
    """Nest rows as {item: {corpus: {impl: row}}}, preserving item order."""

    grouped: dict[str, dict[str, dict[str, dict[str, str]]]] = {}
    for row in rows:
        by_corpus = grouped.setdefault(row[item_key], {})
        by_corpus.setdefault(row["corpus"], {})[row["impl"]] = row
    return grouped


def _cell(row: dict[str, str], *, bold: bool, quartiles: bool) -> str:
    value = f"{{\\bf {row['median_MiB/s']}}}" if bold else row["median_MiB/s"]
    if not quartiles:
        return value
    return f"{value} ({row['p25_MiB/s']}-{row['p75_MiB/s']})"


def _render(
    rows: list[dict[str, str]],
    *,
    item_key: str,
    item_header: str,
    baseline_impl: str,
    baseline_label: str,
    caption: str,
    label: str,
    quartiles: bool = True,
    speedup: bool = True,
    name_map: dict[str, str] | None = None,
) -> str:
    grouped = _group(rows, item_key)
    name_map = name_map or {}
    column_spec = "|l|l|c|c|c|c|" if speedup else "|l|l|c|c|"
    header_row = rf"{{\bf {item_header}}} & {{\bf Dataset}} & \python{{{baseline_label}}} & {{\bf Ours}}"
    if speedup:
        header_row += " & Speedup"
    header_row += r" \\"

    lines = [
        r"\begin{table}[ht]",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{center}",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\hline",
        header_row,
        r"\hline",
    ]
    for item, by_corpus in grouped.items():
        display_name = name_map.get(item, item)
        for corpus in CORPORA:
            pair = by_corpus.get(corpus)
            if pair is None:
                continue
            baseline_row = pair[baseline_impl]
            hiriluk_row = pair["hiriluk"]
            baseline_median = float(baseline_row["median_MiB/s"])
            hiriluk_median = float(hiriluk_row["median_MiB/s"])
            baseline_cell = _cell(
                baseline_row, bold=baseline_median > hiriluk_median, quartiles=quartiles
            )
            hiriluk_cell = _cell(
                hiriluk_row, bold=hiriluk_median >= baseline_median, quartiles=quartiles
            )
            row_line = rf"\python{{{display_name}}} & {CORPUS_LABELS[corpus]} & {baseline_cell} & {hiriluk_cell}"
            if speedup:
                row_line += rf" & {hiriluk_median / baseline_median:.1f}$\times$"
            row_line += r" \\"
            lines.append(row_line)
        lines.append(r"\hline")
    lines += [
        r"\end{tabular}",
        r"\end{center}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def dfa_table_to_latex(path: Path) -> str:
    """Render a `hiriluk-throughput-dfa-*.out` log as the tiktoken comparison table."""

    caption = (
        r"Throughput comparison of our tokenizer on \python{tiktoken} encodings "
        r"measured in MiB/s. \python{tiktoken} is provided with a string as "
        r"input and gives array as output, both in memory. Our algorithm is "
        r"provided with a file path as input and writes output to file. We "
        r"report the median throughput over 10 runs for each row with upper "
        r"and lower quartiles in the parentheses."
    )
    return _render(
        parse_table(path),
        item_key="encoding",
        item_header="Encoding",
        baseline_impl="tiktoken",
        baseline_label="tiktoken",
        caption=caption,
        label="tbl:dfa-tiktoken-throughput",
    )


def hf_table_to_latex(path: Path) -> str:
    """Render a `hiriluk-throughput-hf-*.out` log as the tokenizers comparison table."""

    caption = (
        r"Throughput comparison of our tokenizer on \python{tokenizers} models "
        r"measured in MiB/s. \python{tokenizers} is provided with a string as "
        r"input and gives an array as output, both in memory. Our "
        r"algorithm is provided with a file path as input and writes output to "
        r"file. We report the median throughput over 10 runs for each row with "
        r"upper and lower quartiles in the parentheses."
    )
    return _render(
        parse_table(path),
        item_key="model",
        item_header="Model",
        baseline_impl="huggingface",
        baseline_label="tokenizers",
        caption=caption,
        label="tbl:hf-tokenizers-throughput",
        name_map=MODEL_LABELS,
    )


def _speedup_rows(
    rows: list[dict[str, str]],
    *,
    item_key: str,
    baseline_impl: str,
    name_map: dict[str, str] | None = None,
) -> list[tuple[str, dict[str, float]]]:
    """Collect per-item speedups over `baseline_impl` as [(name, {corpus: speedup})]."""

    name_map = name_map or {}
    collected = []
    for item, by_corpus in _group(rows, item_key).items():
        speedups = {}
        for corpus in CORPORA:
            pair = by_corpus.get(corpus)
            if pair is None or baseline_impl not in pair or "hiriluk" not in pair:
                continue
            speedups[corpus] = float(pair["hiriluk"]["median_MiB/s"]) / float(
                pair[baseline_impl]["median_MiB/s"]
            )
        if speedups:
            collected.append((name_map.get(item, item), speedups))
    return collected


def combined_speedup_table_to_latex(dfa_path: Path, hf_path: Path) -> str:
    """Render the dfa and hf logs as one speedup table.

    Rows are encodings (all \\python{tiktoken} ones first, then the
    \\python{tokenizers} models after a rule), columns are the corpora, and
    each entry is our throughput divided by the corresponding baseline's.
    """

    blocks = (
        _speedup_rows(parse_table(dfa_path), item_key="encoding", baseline_impl="tiktoken"),
        _speedup_rows(
            parse_table(hf_path),
            item_key="model",
            baseline_impl="huggingface",
            name_map=MODEL_LABELS,
        ),
    )

    caption = (
        r"Throughput speedup of our tokenizer over \python{tiktoken} (top) and "
        r"\python{tokenizers} (bottom) on each corpus, computed from the median "
        r"throughput over 10 runs."
    )
    corpus_headers = " & ".join(rf"{{\bf {CORPUS_LABELS[corpus]}}}" for corpus in CORPORA)
    out = [
        r"\begin{table}[ht]",
        rf"\caption{{{caption}}}",
        r"\label{tbl:combined-speedup}",
        r"\begin{center}",
        rf"\begin{{tabular}}{{|l|{'c|' * len(CORPORA)}}}",
        r"\hline",
        rf"{{\bf Encoding}} & {corpus_headers} \\",
        r"\hline",
    ]
    for block in blocks:
        for name, speedups in block:
            cells = [
                rf"{speedups[corpus]:.1f}$\times$" if corpus in speedups else ""
                for corpus in CORPORA
            ]
            out.append(rf"\python{{{name}}} & " + " & ".join(cells) + r" \\")
        out.append(r"\hline")
    out += [
        r"\end{tabular}",
        r"\end{center}",
        r"\end{table}",
    ]
    return "\n".join(out)


def _group_fast(
    rows: list[dict[str, str]],
) -> dict[str, dict[str, dict[tuple[str, str], dict[str, str]]]]:
    """Nest rows as {encoding: {corpus: {(impl, mode): row}}}, preserving encoding order.

    The fast suite measures each implementation twice per corpus, so `impl`
    alone does not identify a row: `mode` is `stream` when the output was
    written to a temporary JSON file (`streamed-json-file`,
    `awkward-u32-array+json-file`) and `array` when it stayed in memory.
    """

    grouped: dict[str, dict[str, dict[tuple[str, str], dict[str, str]]]] = {}
    for row in rows:
        mode = "stream" if row["output"].endswith("json-file") else "array"
        by_corpus = grouped.setdefault(row["encoding"], {})
        by_corpus.setdefault(row["corpus"], {})[(row["impl"], mode)] = row
    return grouped


def fast_table_to_latex(path: Path) -> str:
    """Render a `hiriluk-throughput-fast-*.out` log as the gigatoken comparison table.

    Columns are gigatoken and ours, each in array and stream mode; a cell is
    left blank when the log has no row for that combination.
    """

    grouped = _group_fast(parse_table(path))
    columns = (("gigatoken", "array"), ("gigatoken", "stream"), ("hiriluk", "array"), ("hiriluk", "stream"))

    caption = (
        r"Throughput comparison of optimized tokenizer on \python{tiktoken} "
        r"encodings (MiB/s). \python{gigatoken} and our algorithm are provided "
        r"with a file path as input. The array column reports throughput when "
        r"output is collected as array in memory, the stream column reports "
        r"when output is written to a temporary file. We report the median of "
        r"10 runs (quartiles omitted for space)."
    )
    out = [
        r"\begin{table}[ht]",
        rf"\caption{{{caption}}}",
        r"\label{tbl:gigatoken-throughput-array}",
        r"\begin{center}",
        r"\begin{tabular}{|l|l|c|c|c|c|}",
        r"\hline",
        r"{\bf Encoding} & {\bf Dataset} & \python{gigatoken} (Array) & "
        r"\python{gigatoken} (Stream) & {\bf Ours} (Array) & {\bf Ours} (Stream) \\",
        r"\hline",
    ]
    for encoding, by_corpus in grouped.items():
        for corpus in CORPORA:
            by_column = by_corpus.get(corpus)
            if by_column is None:
                continue
            present = [by_column[key] for key in columns if key in by_column]
            if not present:
                continue
            best = max(float(row["median_MiB/s"]) for row in present)
            cells = [
                _cell(
                    by_column[key],
                    bold=float(by_column[key]["median_MiB/s"]) == best,
                    quartiles=False,
                )
                if key in by_column
                else ""
                for key in columns
            ]
            out.append(
                rf"\python{{{encoding}}} & {CORPUS_LABELS[corpus]} & " + " & ".join(cells) + r" \\"
            )
        out.append(r"\hline")
    out += [
        r"\end{tabular}",
        r"\end{center}",
        r"\end{table}",
    ]
    return "\n".join(out)


def _group_by_encoding(rows: list[dict[str, str]]) -> dict[str, dict[str, dict[str, str]]]:
    """Nest rows as {encoding: {corpus: row}}, preserving encoding order."""

    grouped: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["encoding"], {})[row["corpus"]] = row
    return grouped


def _parse_hiriluk_ttft_dfa(lines: list[str]) -> list[dict[str, str]]:
    """Parse the encoding-suite `DFA/lookup/MTC` TTFT table out of a ttft log.

    The same log also has a `SIMD/full-cache` (fast) encoding-suite table and
    a `Hiriluk DFA/lookup/MTC` model-suite table; both would also match a
    naive search for "DFA/lookup/MTC" or "impl", so the section header is
    matched precisely (no `Hiriluk` prefix) to pick the right one.
    """

    start = next(
        i
        for i, line in enumerate(lines)
        if line.startswith("===") and "DFA/lookup/MTC" in line and "Hiriluk" not in line
    )
    header_index = next(i for i in range(start, len(lines)) if lines[i].startswith("impl"))
    header = lines[header_index].split()
    rows = []
    for line in lines[header_index + 2 :]:
        if not line.strip():
            break
        rows.append(dict(zip(header, line.split())))
    return rows


def _parse_external_ttft(
    lines: list[str], marker_prefix: str, stop_prefixes: tuple[str, ...]
) -> list[dict[str, str]]:
    """Parse a `tiktoken`/`gigatoken` reference TTFT table out of a ttft log.

    These tables have no `impl` column: a `{marker_prefix}...` line introduces
    the run, followed by a header row and data rows interleaved with
    `[ttft] ...` progress lines.
    """

    start = next(i for i, line in enumerate(lines) if line.startswith(marker_prefix))
    header = lines[start + 1].split()
    rows = []
    for line in lines[start + 2 :]:
        if line.startswith("[ttft]"):
            continue
        if not line.strip() or line.startswith(stop_prefixes):
            break
        rows.append(dict(zip(header, line.split())))
    return rows


def _ttft_cell(median: str, p25: str, p75: str, *, bold: bool) -> str:
    median, p25, p75 = f"{float(median):.2f}", f"{float(p25):.2f}", f"{float(p75):.2f}"
    value = f"{{\\bf {median}}}" if bold else median
    return f"{value} ({p25}-{p75})"


def _parse_ttft_log(
    path: Path,
) -> tuple[
    dict[str, dict[str, dict[str, str]]],
    dict[str, dict[str, dict[str, str]]],
    dict[str, dict[str, dict[str, str]]],
]:
    """Parse a ttft log into (hiriluk, tiktoken, gigatoken) {encoding: {corpus: row}}.

    "Ours" is Hiriluk's DFA/lookup/MTC implementation (not the SIMD/fast
    one), and `gigatoken` is taken from its `resident_str` (in-memory string)
    measurement, matching how `tiktoken` is invoked.
    """

    lines = path.read_text().splitlines()
    hiriluk_rows = _group_by_encoding(_parse_hiriluk_ttft_dfa(lines))
    tiktoken_rows = _group_by_encoding(
        _parse_external_ttft(lines, "warmup=", ("gigatoken=", "finished="))
    )
    gigatoken_rows = _group_by_encoding(
        [
            row
            for row in _parse_external_ttft(lines, "gigatoken=", ("finished=",))
            if row["boundary"] == "resident_str"
        ]
    )
    return hiriluk_rows, tiktoken_rows, gigatoken_rows


def ttft_table_to_latex(path: Path) -> str:
    """Render a `hiriluk-ttft-dfa-*.out` log as the TTFT comparison table."""

    hiriluk_rows, tiktoken_rows, gigatoken_rows = _parse_ttft_log(path)

    caption = (
        r"Time to first token (TTFT) of all comparison of all tokenizers on "
        r"\python{tiktoken} encodings measured in milliseconds (ms). "
        r"We report the median TTFT over 30 runs for each row with upper "
        r"and lower quartiles in the parentheses."
    )
    out = [
        r"\begin{table}[ht]",
        rf"\caption{{{caption}}}",
        r"\label{tbl:ttft-results}",
        r"\begin{center}",
        r"\begin{tabular}{|l|l|c|c|c|c|}",
        r"\hline",
        r"{\bf Encoding} & {\bf Dataset} & \python{tiktoken} & \python{gigatoken} & {\bf Ours} & Speedup \\",
        r"\hline",
    ]
    for encoding, base_name in TTFT_ENCODING_BASE.items():
        for corpus in CORPORA:
            hiriluk_row = hiriluk_rows.get(encoding, {}).get(corpus)
            tiktoken_row = tiktoken_rows.get(base_name, {}).get(corpus)
            gigatoken_row = gigatoken_rows.get(base_name, {}).get(corpus)
            if hiriluk_row is None or tiktoken_row is None or gigatoken_row is None:
                continue
            ours_median = float(hiriluk_row["median_TTFT_ms"])
            tiktoken_median = float(tiktoken_row["median_ttft_ms"])
            gigatoken_median = float(gigatoken_row["median_ttft_ms"])
            best = min(tiktoken_median, gigatoken_median, ours_median)
            tiktoken_cell = _ttft_cell(
                tiktoken_row["median_ttft_ms"],
                tiktoken_row["ttft_p25_ms"],
                tiktoken_row["ttft_p75_ms"],
                bold=tiktoken_median == best,
            )
            gigatoken_cell = _ttft_cell(
                gigatoken_row["median_ttft_ms"],
                gigatoken_row["ttft_p25_ms"],
                gigatoken_row["ttft_p75_ms"],
                bold=gigatoken_median == best,
            )
            ours_cell = _ttft_cell(
                hiriluk_row["median_TTFT_ms"],
                hiriluk_row["TTFT_p25_ms"],
                hiriluk_row["TTFT_p75_ms"],
                bold=ours_median == best,
            )
            speedup = min(tiktoken_median, gigatoken_median) / ours_median
            out.append(
                rf"\python{{{encoding}}} & {CORPUS_LABELS[corpus]} & {tiktoken_cell} & "
                rf"{gigatoken_cell} & {ours_cell} & {speedup:.1f}$\times$ \\"
            )
        out.append(r"\hline")
    out += [
        r"\end{tabular}",
        r"\end{center}",
        r"\end{table}",
    ]
    return "\n".join(out)


def ttft_speedup_table_to_latex(path: Path) -> str:
    """Render a `hiriluk-ttft-dfa-*.out` log as a compact TTFT speedup table.

    Rows are the \\python{tiktoken} encodings and columns are the corpora; each
    entry is the best of \\python{tiktoken} and \\python{gigatoken} divided by
    our median TTFT, matching the speedup column of `ttft_table_to_latex`.
    """

    hiriluk_rows, tiktoken_rows, gigatoken_rows = _parse_ttft_log(path)

    caption = (
        r"Time to first token (TTFT) speedup of our tokenizer over the better "
        r"of \python{tiktoken} and \python{gigatoken} on each corpus, computed "
        r"from the median TTFT over 30 runs."
    )
    corpus_headers = " & ".join(rf"{{\bf {CORPUS_LABELS[corpus]}}}" for corpus in CORPORA)
    out = [
        r"\begin{table}[ht]",
        rf"\caption{{{caption}}}",
        r"\label{tbl:ttft-speedup}",
        r"\begin{center}",
        rf"\begin{{tabular}}{{|l|{'c|' * len(CORPORA)}}}",
        r"\hline",
        rf"{{\bf Encoding}} & {corpus_headers} \\",
        r"\hline",
    ]
    for encoding, base_name in TTFT_ENCODING_BASE.items():
        cells = []
        for corpus in CORPORA:
            hiriluk_row = hiriluk_rows.get(encoding, {}).get(corpus)
            tiktoken_row = tiktoken_rows.get(base_name, {}).get(corpus)
            gigatoken_row = gigatoken_rows.get(base_name, {}).get(corpus)
            if hiriluk_row is None or tiktoken_row is None or gigatoken_row is None:
                cells.append("")
                continue
            baseline = min(
                float(tiktoken_row["median_ttft_ms"]), float(gigatoken_row["median_ttft_ms"])
            )
            speedup = baseline / float(hiriluk_row["median_TTFT_ms"])
            cells.append(rf"{speedup:.1f}$\times$")
        if not any(cells):
            continue
        out.append(rf"\python{{{encoding}}} & " + " & ".join(cells) + r" \\")
    out += [
        r"\hline",
        r"\end{tabular}",
        r"\end{center}",
        r"\end{table}",
    ]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dfa", type=Path, default=None, help="path to a hiriluk-throughput-dfa-*.out log"
    )
    parser.add_argument(
        "--hf", type=Path, default=None, help="path to a hiriluk-throughput-hf-*.out log"
    )
    parser.add_argument(
        "--fast", type=Path, default=None, help="path to a hiriluk-throughput-fast-*.out log"
    )
    parser.add_argument(
        "--ttft", type=Path, default=None, help="path to a hiriluk-ttft-dfa-*.out log"
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help=(
            "render the combined dfa+hf speedup table and the TTFT speedup table "
            "(built from the --dfa/--hf/--ttft logs)"
        ),
    )
    args = parser.parse_args()

    dfa_pattern, hf_pattern = "hiriluk-throughput-dfa-*.out", "hiriluk-throughput-hf-*.out"
    ttft_pattern = "hiriluk-ttft-dfa-*.out"
    # Rendering is deferred so that _latest is only resolved for selected tables.
    tables = (
        (args.dfa is not None, lambda: dfa_table_to_latex(args.dfa or _latest(dfa_pattern))),
        (args.hf is not None, lambda: hf_table_to_latex(args.hf or _latest(hf_pattern))),
        (
            args.fast is not None,
            lambda: fast_table_to_latex(args.fast or _latest("hiriluk-throughput-fast-*.out")),
        ),
        (
            args.ttft is not None,
            lambda: ttft_table_to_latex(args.ttft or _latest(ttft_pattern)),
        ),
        (
            args.combined,
            lambda: combined_speedup_table_to_latex(
                args.dfa or _latest(dfa_pattern), args.hf or _latest(hf_pattern)
            ),
        ),
        (
            args.combined,
            lambda: ttft_speedup_table_to_latex(args.ttft or _latest(ttft_pattern)),
        ),
    )
    # Any explicit flag selects the tables to render; with no flags, render all
    # of them from the latest logs.
    selected = [render for chosen, render in tables if chosen] or [
        render for _, render in tables
    ]

    for index, render in enumerate(selected):
        if index:
            print()
        print(render())


if __name__ == "__main__":
    main()
