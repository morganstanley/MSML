"""Convert a percent-format .py (``# %%`` cells) into an EXECUTED .ipynb + .html.

No papermill / jupytext needed — uses only nbformat + nbconvert (both present in
the p312 venv). This is the verifier's notebook engine: an agent writes a plain
Python file with cell markers, this runs it on a fresh kernel and embeds all
outputs (charts, tables, prints) inline, so the artifact is a self-contained,
human-readable proof.

Cell markers:
  ``# %%``            -> start a code cell
  ``# %% [markdown]`` -> start a markdown cell (subsequent ``# `` comment lines
                         become the markdown body)

Usage:
  python scripts/nb_run.py input.py output.ipynb [--timeout 1800]

Exit codes: 0 ok | 2 a cell raised (partial notebook written for inspection) | 1 setup error
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def parse_cells(src: str):
    """Split a percent-format script into (kind, text) cells."""
    cells: list[tuple[str, str]] = []
    kind = "code"
    buf: list[str] = []

    def flush():
        text = "\n".join(buf).strip("\n")
        if text.strip():
            cells.append((kind, text))

    for line in src.splitlines():
        m = re.match(r"#\s*%%(.*)$", line)
        if m:
            flush()
            buf.clear()
            kind = "markdown" if "markdown" in m.group(1).strip().lower() else "code"
        else:
            buf.append(line)
    flush()
    return cells


def build_notebook(cells):
    nb = new_notebook()
    for kind, text in cells:
        if kind == "markdown":
            md = "\n".join(re.sub(r"^#\s?", "", ln) for ln in text.splitlines())
            nb.cells.append(new_markdown_cell(md))
        else:
            nb.cells.append(new_code_cell(text))
    return nb


def _ensure_kernelspec() -> str:
    """Register an ephemeral kernelspec for THIS interpreter and return its name.

    We deliberately do NOT reuse a pre-registered ``python3`` spec — it may point
    at a different interpreter that lacks torch/pandas. Binding the kernel to
    ``sys.executable`` guarantees the notebook runs in the same env that launched
    this script (the p312 venv, which has torch/pandas/sklearn/matplotlib). No
    global/user install, no pip — just a temp dir on JUPYTER_PATH.
    """
    import json
    base = Path(tempfile.gettempdir()) / "verifier_jupyter"
    spec_dir = base / "kernels" / "verifier_py"
    spec_dir.mkdir(parents=True, exist_ok=True)
    (spec_dir / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "verifier_py",
        "language": "python",
    }))
    existing = os.environ.get("JUPYTER_PATH", "")
    os.environ["JUPYTER_PATH"] = f"{base}{os.pathsep}{existing}" if existing else str(base)
    return "verifier_py"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--timeout", type=int, default=1800, help="per-cell timeout seconds")
    ap.add_argument("--no-html", action="store_true")
    args = ap.parse_args()

    from nbconvert.preprocessors import ExecutePreprocessor

    src = Path(args.input).read_text()
    nb = build_notebook(parse_cells(src))
    kernel = _ensure_kernelspec()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    ep = ExecutePreprocessor(timeout=args.timeout, kernel_name=kernel, allow_errors=False)
    try:
        ep.preprocess(nb, {"metadata": {"path": str(out.resolve().parent)}})
    except Exception as e:
        nbformat.write(nb, str(out))  # keep the partial run for inspection
        print(f"EXECUTION FAILED: {type(e).__name__}: {str(e)[:600]}", file=sys.stderr)
        return 2

    nbformat.write(nb, str(out))
    if not args.no_html:
        from nbconvert import HTMLExporter
        html, _ = HTMLExporter().from_notebook_node(nb)
        out.with_suffix(".html").write_text(html)
    print(f"OK: executed {len(nb.cells)} cells -> {out}"
          + ("" if args.no_html else f" (+ {out.with_suffix('.html').name})"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
