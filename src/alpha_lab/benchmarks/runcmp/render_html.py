"""Markdown -> standalone HTML for reviewer reports.

Reports are read by humans in a browser as often as in a terminal, so every
REPORT.md is written alongside a REPORT.html. The HTML is self-contained (no
network fetches, no local assets) so it survives being emailed or copied to a
share.

The reports are chart-heavy: their figures are ASCII/box-drawing blocks inside
fenced code. Those must render in a monospace font with preserved whitespace or
the charts collapse, which is why the stylesheet leaves `pre` alone and only
styles around it.
"""

from __future__ import annotations

import html as _html
import re as _re
from pathlib import Path

_CSS = """
:root { color-scheme: light dark; }
body { max-width: 62rem; margin: 2rem auto; padding: 0 1.5rem;
       font: 16px/1.6 -apple-system, "Segoe UI", Roboto, Helvetica, sans-serif;
       color: #1a1a1a; background: #fff; }
h1, h2, h3 { line-height: 1.25; margin-top: 2rem; }
h1 { font-size: 1.9rem; border-bottom: 2px solid #ddd; padding-bottom: .4rem; }
h2 { font-size: 1.4rem; border-bottom: 1px solid #eee; padding-bottom: .3rem; }
code { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
       font-size: .87em; background: #f4f4f4; padding: .1em .3em;
       border-radius: 3px; }
pre { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
      font-size: .82rem; line-height: 1.35; background: #f7f7f7;
      border: 1px solid #e2e2e2; border-radius: 5px; padding: .9rem 1.1rem;
      overflow-x: auto; white-space: pre; }
pre code { background: none; padding: 0; font-size: inherit; }
table { border-collapse: collapse; margin: 1.2rem 0; font-size: .93rem;
        display: block; overflow-x: auto; max-width: 100%; }
th, td { border: 1px solid #ddd; padding: .4rem .6rem; text-align: left; }
th { background: #f2f2f2; font-weight: 600; }
tr:nth-child(even) td { background: #fafafa; }
blockquote { border-left: 3px solid #ccc; margin-left: 0; padding-left: 1rem;
             color: #555; }
a { color: #0b5fff; }
.chart { margin: 1.2rem 0; }
.chart-title { font-weight: 600; margin-bottom: .5rem; font-size: .95rem; }
.chart-row { display: flex; align-items: center; gap: .6rem; margin: .22rem 0; }
.chart-label { flex: 0 0 16rem; text-align: right; font-size: .85rem;
               white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.chart-track { flex: 1 1 auto; background: #f0f0f0; border-radius: 3px;
               height: 1.05rem; overflow: hidden; }
.chart-bar { height: 100%; border-radius: 3px; }
.chart-value { flex: 0 0 11rem; font-size: .85rem;
               font-family: ui-monospace, Menlo, Consolas, monospace; }
.chart-note { color: #777; font-family: -apple-system, sans-serif; }
.chart-svg { width: 100%; max-width: 46rem; height: auto; display: block;
             margin: .3rem 0; }
.chart-svg .grid { stroke: #ececec; stroke-width: 1; }
.chart-svg .frame { fill: none; stroke: #ccc; stroke-width: 1; }
.chart-svg .tick { font: 11px ui-monospace, Menlo, monospace; fill: #666; }
.chart-svg .axis { font: 12px -apple-system, sans-serif; fill: #444; }
.chart-svg .legend { font: 12px -apple-system, sans-serif; }
.chart-svg .ptlabel { font: 10px -apple-system, sans-serif; fill: #555; }
.stack-track { flex: 1 1 auto; display: flex; height: 1.3rem;
               border-radius: 3px; overflow: hidden; }
.stack-seg { height: 100%; font: 10px ui-monospace, monospace; color: #fff;
             text-align: center; line-height: 1.3rem; overflow: hidden;
             white-space: nowrap; }
.stack-legend { margin: .35rem 0 0 16rem; font-size: .82rem; }
.stack-legend .legend { margin-right: .9rem; }
.chart-group { margin: .55rem 0; }
.chart-label.group { flex: none; text-align: left; font-weight: 600;
                     font-size: .88rem; margin-bottom: .15rem; }
.grouped-row .chart-sublabel { flex: 0 0 12rem; text-align: right;
                               font-size: .8rem; color: #666;
                               white-space: nowrap; overflow: hidden; }
table.heatmap { border-collapse: collapse; font-size: .85rem;
                display: table; }
table.heatmap th, table.heatmap td { border: 1px solid #e6e6e6;
                padding: .3rem .55rem; text-align: right;
                font-family: ui-monospace, Menlo, monospace; }
table.heatmap thead th { background: #f2f2f2;
                font-family: -apple-system, sans-serif; }
table.heatmap tbody th { text-align: left; background: #f9f9f9;
                font-family: -apple-system, sans-serif; font-weight: 500; }
@media (prefers-color-scheme: dark) {
  body { color: #e4e4e4; background: #1c1c1e; }
  h1 { border-bottom-color: #3a3a3c; } h2 { border-bottom-color: #2c2c2e; }
  code { background: #2c2c2e; } pre { background: #242426; border-color: #3a3a3c; }
  th { background: #2c2c2e; } th, td { border-color: #3a3a3c; }
  tr:nth-child(even) td { background: #202022; }
  blockquote { border-left-color: #48484a; color: #a1a1a6; }
  a { color: #5aa0ff; }
  .chart-track { background: #2c2c2e; }
  .chart-note { color: #98989d; }
}
"""

_SHELL = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{css}</style>
</head><body>
{body}
</body></html>
"""

# Colored charts. Reports emit fenced blocks with language "chart". Four
# types render as self-contained colored HTML/SVG; the .md stays readable
# text. Any parse failure leaves the whole fence untouched as <pre>, so a
# malformed chart degrades to exactly what the markdown shows.
#
#   ```chart
#   type: bars            (default when omitted — legacy blocks keep working)
#   title: referee RMSE by pair (lower is better)
#   cond GLM | 0.022312
#   msml GLM | 0.022218 | winner
#   ```
#
#   ```chart
#   type: line
#   title: best val_bpb so far (lower is better)
#   x: experiment number
#   y: val_bpb
#   series cond: 1,0.92; 3,0.87; 8,0.85; 17,0.796
#   series msml: 1,0.94; 2,0.88; 8,0.768
#   ```
#
#   ```chart
#   type: scatter
#   title: cost vs quality
#   x: dollars per scored experiment
#   y: referee RMSE
#   point msml Opus 5 | 9.80 | 0.021507 | msml
#   point cond Opus 4.8 | 3.99 | 0.021697 | cond
#   ```
#
#   ```chart
#   type: stacked
#   title: request payload composition (%)
#   row cond Opus 5 | instructions=17.5 | tools=22.2 | images=38.9 | other=21.4
#   row msml Opus 5 | instructions=9.6 | tools=20.5 | images=46.8 | other=23.1
#   ```

# Delimits the machine-generated production-cost footer appended to every
# report at publication (user order 2026-08-11: every report ends with what
# it cost to make). The fact-checker uses it to exempt those numbers from
# the body audit — they come from the session's own usage ledger, not probes.
PRODUCTION_COST_MARKER = "<!-- production-cost -->"

_CHART_FENCE_RE = _re.compile(r"```chart[ \t]*\n(.*?)```", _re.DOTALL)
_CHART_PALETTE = ("#4c78a8", "#f58518", "#54a24b", "#e45756",
                  "#72b7b2", "#eeca3b", "#b279a2", "#9d755d")

_SVG_W, _SVG_H = 640, 300
_ML, _MR, _MT, _MB = 64, 16, 14, 42  # plot margins inside the SVG


def _num(s: str) -> float:
    return float(s.strip().replace(",", "").rstrip("%$"))


def _parse_header(body: str) -> tuple[dict, list[str]]:
    """Split leading ``key: value`` lines from data lines."""
    meta: dict[str, str] = {}
    data: list[str] = []
    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            continue
        key = line.split(":", 1)[0].lower()
        if (not data and ":" in line
                and key in ("type", "title", "x", "y", "cols", "style",
                            "fill", "marginals")):
            meta[key] = line.split(":", 1)[1].strip()
        else:
            data.append(line)
    return meta, data


def _scale(vals: list[float]) -> tuple[float, float]:
    lo, hi = min(vals), max(vals)
    if lo == hi:
        pad = abs(lo) * 0.05 or 1.0
        return lo - pad, hi + pad
    pad = (hi - lo) * 0.06
    return lo - pad, hi + pad


def _ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    import math
    span = hi - lo
    step = 10 ** math.floor(math.log10(span / n))
    for mult in (1, 2, 2.5, 5, 10):
        if span / (step * mult) <= n:
            step *= mult
            break
    first = math.ceil(lo / step) * step
    out = []
    t = first
    while t <= hi + 1e-12:
        out.append(round(t, 10))
        t += step
    return out


def _fmt_tick(v: float) -> str:
    if v == int(v) and abs(v) < 1e6:
        return str(int(v))
    return f"{v:.6g}"


def _svg_frame(title: str, xlab: str, ylab: str,
               xlo: float, xhi: float, ylo: float, yhi: float) -> list[str]:
    """SVG opening, plot frame, gridlines, ticks, and axis labels."""
    def X(v: float) -> float:
        return _ML + (v - xlo) / (xhi - xlo) * (_SVG_W - _ML - _MR)

    def Y(v: float) -> float:
        return _SVG_H - _MB - (v - ylo) / (yhi - ylo) * (_SVG_H - _MT - _MB)

    out = [f'<svg viewBox="0 0 {_SVG_W} {_SVG_H}" class="chart-svg" '
           f'role="img" aria-label="{_html.escape(title)}">']
    for t in _ticks(ylo, yhi):
        y = Y(t)
        out.append(f'<line x1="{_ML}" y1="{y:.1f}" x2="{_SVG_W - _MR}" '
                   f'y2="{y:.1f}" class="grid"/>')
        out.append(f'<text x="{_ML - 6}" y="{y + 4:.1f}" class="tick" '
                   f'text-anchor="end">{_fmt_tick(t)}</text>')
    for t in _ticks(xlo, xhi):
        x = X(t)
        out.append(f'<line x1="{x:.1f}" y1="{_MT}" x2="{x:.1f}" '
                   f'y2="{_SVG_H - _MB}" class="grid"/>')
        out.append(f'<text x="{x:.1f}" y="{_SVG_H - _MB + 16}" class="tick" '
                   f'text-anchor="middle">{_fmt_tick(t)}</text>')
    out.append(f'<rect x="{_ML}" y="{_MT}" width="{_SVG_W - _ML - _MR}" '
               f'height="{_SVG_H - _MT - _MB}" class="frame"/>')
    if xlab:
        out.append(f'<text x="{(_ML + _SVG_W - _MR) / 2}" y="{_SVG_H - 6}" '
                   f'class="axis" text-anchor="middle">{_html.escape(xlab)}</text>')
    if ylab:
        out.append(f'<text x="14" y="{(_MT + _SVG_H - _MB) / 2}" class="axis" '
                   f'text-anchor="middle" transform="rotate(-90 14 '
                   f'{(_MT + _SVG_H - _MB) / 2})">{_html.escape(ylab)}</text>')
    return out


def _chart_bars(meta: dict, data: list[str]) -> str | None:
    rows: list[tuple[str, float, str]] = []
    for line in data:
        parts = [p.strip() for p in line.split("|")]
        if len(parts) not in (2, 3):
            return None
        try:
            value = _num(parts[1])
        except ValueError:
            return None
        rows.append((parts[0], value, parts[2] if len(parts) == 3 else ""))
    if not rows:
        return None
    vmax = max(abs(v) for _, v, _ in rows) or 1.0
    out = []
    for i, (label, value, note) in enumerate(rows):
        pct = max(1.0, 100.0 * abs(value) / vmax)
        color = _CHART_PALETTE[i % len(_CHART_PALETTE)]
        note_html = (f' <span class="chart-note">{_html.escape(note)}</span>'
                     if note else "")
        out.append(
            '<div class="chart-row">'
            f'<div class="chart-label">{_html.escape(label)}</div>'
            f'<div class="chart-track"><div class="chart-bar" '
            f'style="width:{pct:.1f}%;background:{color}"></div></div>'
            f'<div class="chart-value">{_html.escape(str(value))}{note_html}</div>'
            '</div>')
    return "\n".join(out)


def _chart_line(meta: dict, data: list[str]) -> str | None:
    series: list[tuple[str, list[tuple[float, float]]]] = []
    bands: list[tuple[str, list[tuple[float, float, float, float]]]] = []
    for line in data:
        low = line.lower()
        if low.startswith("band ") and ":" in line:
            name, rest = line[5:].split(":", 1)
            quads = []
            try:
                for grp in rest.split(";"):
                    if not grp.strip():
                        continue
                    x, lo_, mid, hi_ = (_num(v) for v in grp.split(","))
                    quads.append((x, lo_, mid, hi_))
            except ValueError:
                return None
            if quads:
                bands.append((name.strip(), sorted(quads)))
            continue
        if not low.startswith("series ") or ":" not in line:
            return None
        name, rest = line[7:].split(":", 1)
        pts = []
        try:
            for pair in rest.split(";"):
                if not pair.strip():
                    continue
                xs, ys = pair.split(",")
                pts.append((_num(xs), _num(ys)))
        except ValueError:
            return None
        if pts:
            series.append((name.strip(), sorted(pts)))
    if not series and not bands:
        return None
    ax = ([p[0] for _, pts in series for p in pts]
          + [q[0] for _, qs in bands for q in qs])
    ay = ([p[1] for _, pts in series for p in pts]
          + [v for _, qs in bands for q in qs for v in q[1:]])
    xlo, xhi = _scale(ax)
    ylo, yhi = _scale(ay)
    out = _svg_frame(meta.get("title", ""), meta.get("x", ""),
                     meta.get("y", ""), xlo, xhi, ylo, yhi)

    def X(v): return _ML + (v - xlo) / (xhi - xlo) * (_SVG_W - _ML - _MR)
    def Y(v): return _SVG_H - _MB - (v - ylo) / (yhi - ylo) * (_SVG_H - _MT - _MB)

    for i, (name, quads) in enumerate(bands):
        color = _CHART_PALETTE[(len(series) + i) % len(_CHART_PALETTE)]
        upper = " ".join(f"{X(q[0]):.1f},{Y(q[3]):.1f}" for q in quads)
        lower = " ".join(f"{X(q[0]):.1f},{Y(q[1]):.1f}"
                         for q in reversed(quads))
        out.append(f'<polygon points="{upper} {lower}" fill="{color}" '
                   f'fill-opacity="0.22"/>')
        mid = " ".join(f"{X(q[0]):.1f},{Y(q[2]):.1f}" for q in quads)
        out.append(f'<polyline points="{mid}" fill="none" stroke="{color}" '
                   f'stroke-width="2.2"/>')
        out.append(f'<text x="{_ML + 10}" y="{_MT + 16 + 16 * (len(series) + i)}" '
                   f'class="legend" fill="{color}">&#9632; '
                   f'{_html.escape(name)} (band lo/mid/hi)</text>')
    step = meta.get("style", "").lower() == "step"
    fill = meta.get("fill", "").lower() in ("true", "yes", "1")
    for i, (name, pts) in enumerate(series):
        color = _CHART_PALETTE[i % len(_CHART_PALETTE)]
        draw = pts
        if step:
            draw = []
            for j, (x, y) in enumerate(pts):
                if j:
                    draw.append((x, pts[j - 1][1]))
                draw.append((x, y))
        path = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in draw)
        if fill:
            base = Y(max(ylo, min(0, yhi)) if ylo <= 0 <= yhi else ylo)
            out.append(f'<polygon points="{X(draw[0][0]):.1f},{base:.1f} '
                       f'{path} {X(draw[-1][0]):.1f},{base:.1f}" '
                       f'fill="{color}" fill-opacity="0.18"/>')
        out.append(f'<polyline points="{path}" fill="none" stroke="{color}" '
                   f'stroke-width="2.2"/>')
        if len(pts) <= 40:
            for x, y in pts:
                out.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="3" '
                           f'fill="{color}"/>')
        out.append(f'<text x="{_ML + 10}" y="{_MT + 16 + 16 * i}" class="legend" '
                   f'fill="{color}">&#9632; {_html.escape(name)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def _chart_scatter(meta: dict, data: list[str]) -> str | None:
    """Scatter of labeled points; ``marginals: true`` adds the x/y marginal
    histograms (computed here from the points), turning it into a joint +
    marginal distribution view. Unlabeled mass points: ``point | x | y``."""
    pts: list[tuple[str, float, float, str]] = []
    for line in data:
        if not line.lower().startswith("point"):
            return None
        rest = line[5:].lstrip()
        if rest.startswith("|"):
            rest = rest[1:]
        parts = [p.strip() for p in rest.split("|")]
        if len(parts) == 2:
            parts = [""] + parts
        if len(parts) not in (3, 4):
            return None
        try:
            pts.append((parts[0], _num(parts[1]), _num(parts[2]),
                        parts[3] if len(parts) == 4 else ""))
        except ValueError:
            return None
    if not pts:
        return None
    marginals = meta.get("marginals", "").lower() in ("true", "yes", "1")
    mh = 44 if marginals else 0  # marginal strip height/width
    xlo, xhi = _scale([p[1] for p in pts])
    ylo, yhi = _scale([p[2] for p in pts])
    groups: list[str] = []
    for p in pts:
        if p[3] not in groups:
            groups.append(p[3])
    W, H = _SVG_W, _SVG_H + mh
    top, right = _MT + mh, _MR + mh
    out = [f'<svg viewBox="0 0 {W} {H}" class="chart-svg" role="img" '
           f'aria-label="{_html.escape(meta.get("title", ""))}">']

    def X(v): return _ML + (v - xlo) / (xhi - xlo) * (W - _ML - right)
    def Y(v): return H - _MB - (v - ylo) / (yhi - ylo) * (H - top - _MB)

    for t in _ticks(ylo, yhi):
        out.append(f'<line x1="{_ML}" y1="{Y(t):.1f}" x2="{W - right}" '
                   f'y2="{Y(t):.1f}" class="grid"/>')
        out.append(f'<text x="{_ML - 6}" y="{Y(t) + 4:.1f}" class="tick" '
                   f'text-anchor="end">{_fmt_tick(t)}</text>')
    for t in _ticks(xlo, xhi):
        out.append(f'<line x1="{X(t):.1f}" y1="{top}" x2="{X(t):.1f}" '
                   f'y2="{H - _MB}" class="grid"/>')
        out.append(f'<text x="{X(t):.1f}" y="{H - _MB + 16}" class="tick" '
                   f'text-anchor="middle">{_fmt_tick(t)}</text>')
    out.append(f'<rect x="{_ML}" y="{top}" width="{W - _ML - right}" '
               f'height="{H - top - _MB}" class="frame"/>')
    if meta.get("x"):
        out.append(f'<text x="{(_ML + W - right) / 2}" y="{H - 6}" '
                   f'class="axis" text-anchor="middle">'
                   f'{_html.escape(meta["x"])}</text>')
    if meta.get("y"):
        out.append(f'<text x="14" y="{(top + H - _MB) / 2}" class="axis" '
                   f'text-anchor="middle" transform="rotate(-90 14 '
                   f'{(top + H - _MB) / 2})">{_html.escape(meta["y"])}</text>')

    if marginals:
        nb = max(8, min(24, int(len(pts) ** 0.5) * 2))
        xc = _bin([p[1] for p in pts], xlo, xhi, nb)
        yc = _bin([p[2] for p in pts], ylo, yhi, nb)
        xpk, ypk = max(xc) or 1, max(yc) or 1
        bwx = (xhi - xlo) / nb
        bwy = (yhi - ylo) / nb
        for j, c in enumerate(xc):
            if not c:
                continue
            x0, x1 = X(xlo + j * bwx), X(xlo + (j + 1) * bwx)
            hgt = (mh - 6) * c / xpk
            out.append(f'<rect x="{x0 + 0.5:.1f}" y="{top - 4 - hgt:.1f}" '
                       f'width="{x1 - x0 - 1:.1f}" height="{hgt:.1f}" '
                       f'fill="#8ea8c5"/>')
        for j, c in enumerate(yc):
            if not c:
                continue
            y0, y1 = Y(ylo + (j + 1) * bwy), Y(ylo + j * bwy)
            wid = (mh - 6) * c / ypk
            out.append(f'<rect x="{W - right + 4:.1f}" y="{y0 + 0.5:.1f}" '
                       f'width="{wid:.1f}" height="{y1 - y0 - 1:.1f}" '
                       f'fill="#8ea8c5"/>')

    dense = len(pts) > 60
    for label, x, y, grp in pts:
        color = _CHART_PALETTE[groups.index(grp) % len(_CHART_PALETTE)]
        out.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" '
                   f'r="{3 if dense else 5}" fill="{color}" '
                   f'fill-opacity="{0.55 if dense else 0.85}"/>')
        if label and not dense:
            out.append(f'<text x="{X(x) + 7:.1f}" y="{Y(y) - 6:.1f}" '
                       f'class="ptlabel">{_html.escape(label)}</text>')
    if any(g for g in groups):
        for i, g in enumerate(groups):
            color = _CHART_PALETTE[i % len(_CHART_PALETTE)]
            out.append(f'<text x="{_ML + 10}" y="{top + 16 + 16 * i}" '
                       f'class="legend" fill="{color}">&#9679; '
                       f'{_html.escape(g or "(ungrouped)")}</text>')
    out.append("</svg>")
    return "\n".join(out)


def _chart_stacked(meta: dict, data: list[str]) -> str | None:
    rows: list[tuple[str, list[tuple[str, float]]]] = []
    seg_names: list[str] = []
    for line in data:
        if not line.lower().startswith("row "):
            return None
        parts = [p.strip() for p in line[4:].split("|")]
        if len(parts) < 2:
            return None
        segs = []
        try:
            for cell in parts[1:]:
                name, val = cell.split("=", 1)
                name = name.strip()
                segs.append((name, _num(val)))
                if name not in seg_names:
                    seg_names.append(name)
        except ValueError:
            return None
        rows.append((parts[0], segs))
    if not rows:
        return None
    out = []
    for label, segs in rows:
        total = sum(v for _, v in segs) or 1.0
        cells = []
        for name, val in segs:
            color = _CHART_PALETTE[seg_names.index(name) % len(_CHART_PALETTE)]
            w = 100.0 * val / total
            txt = f"{val:.4g}" if w >= 7 else ""
            cells.append(f'<div class="stack-seg" style="width:{w:.2f}%;'
                         f'background:{color}" title="{_html.escape(name)}='
                         f'{val:.6g}">{txt}</div>')
        out.append('<div class="chart-row">'
                   f'<div class="chart-label">{_html.escape(label)}</div>'
                   f'<div class="stack-track">{"".join(cells)}</div></div>')
    legend = " ".join(
        f'<span class="legend" style="color:'
        f'{_CHART_PALETTE[i % len(_CHART_PALETTE)]}">&#9632; '
        f'{_html.escape(n)}</span>'
        for i, n in enumerate(seg_names))
    out.append(f'<div class="stack-legend">{legend}</div>')
    return "\n".join(out)


def _chart_grouped(meta: dict, data: list[str]) -> str | None:
    """Side-by-side bars per labeled group: ``row LABEL | name=v | name=v``."""
    rows: list[tuple[str, list[tuple[str, float]]]] = []
    names: list[str] = []
    for line in data:
        if not line.lower().startswith("row "):
            return None
        parts = [p.strip() for p in line[4:].split("|")]
        if len(parts) < 2:
            return None
        segs = []
        try:
            for cell in parts[1:]:
                n, v = cell.split("=", 1)
                n = n.strip()
                segs.append((n, _num(v)))
                if n not in names:
                    names.append(n)
        except ValueError:
            return None
        rows.append((parts[0], segs))
    if not rows:
        return None
    vmax = max(abs(v) for _, segs in rows for _, v in segs) or 1.0
    out = []
    for label, segs in rows:
        bars = []
        for n, v in segs:
            color = _CHART_PALETTE[names.index(n) % len(_CHART_PALETTE)]
            pct = max(1.0, 100.0 * abs(v) / vmax)
            bars.append(
                '<div class="chart-row grouped-row">'
                f'<div class="chart-sublabel">{_html.escape(n)}</div>'
                f'<div class="chart-track"><div class="chart-bar" '
                f'style="width:{pct:.1f}%;background:{color}"></div></div>'
                f'<div class="chart-value">{v:.6g}</div></div>')
        out.append(f'<div class="chart-group"><div class="chart-label group">'
                   f'{_html.escape(label)}</div>{"".join(bars)}</div>')
    return "\n".join(out)


def _chart_dumbbell(meta: dict, data: list[str]) -> str | None:
    """Two dots joined by a line per label: ``row LABEL | name=v | name=v``."""
    rows: list[tuple[str, list[tuple[str, float]]]] = []
    names: list[str] = []
    for line in data:
        if not line.lower().startswith("row "):
            return None
        parts = [p.strip() for p in line[4:].split("|")]
        if len(parts) != 3:
            return None
        try:
            segs = []
            for cell in parts[1:]:
                n, v = cell.split("=", 1)
                n = n.strip()
                segs.append((n, _num(v)))
                if n not in names:
                    names.append(n)
        except ValueError:
            return None
        rows.append((parts[0], segs))
    if not rows or len(names) != 2:
        return None
    vals = [v for _, segs in rows for _, v in segs]
    lo, hi = _scale(vals)
    h = 30 * len(rows) + 40
    out = [f'<svg viewBox="0 0 {_SVG_W} {h}" class="chart-svg" role="img">']
    def X(v): return 210 + (v - lo) / (hi - lo) * (_SVG_W - 210 - 30)
    for t in _ticks(lo, hi):
        out.append(f'<line x1="{X(t):.1f}" y1="14" x2="{X(t):.1f}" '
                   f'y2="{h - 26}" class="grid"/>')
        out.append(f'<text x="{X(t):.1f}" y="{h - 12}" class="tick" '
                   f'text-anchor="middle">{_fmt_tick(t)}</text>')
    for i, (label, segs) in enumerate(rows):
        y = 26 + 30 * i
        (n1, v1), (n2, v2) = segs
        out.append(f'<text x="202" y="{y + 4}" class="tick" '
                   f'text-anchor="end">{_html.escape(label)}</text>')
        out.append(f'<line x1="{X(v1):.1f}" y1="{y}" x2="{X(v2):.1f}" '
                   f'y2="{y}" stroke="#bbb" stroke-width="2"/>')
        for n, v in segs:
            c = _CHART_PALETTE[names.index(n) % len(_CHART_PALETTE)]
            out.append(f'<circle cx="{X(v):.1f}" cy="{y}" r="6" fill="{c}"/>')
    for i, n in enumerate(names):
        c = _CHART_PALETTE[i % len(_CHART_PALETTE)]
        out.append(f'<text x="{210 + 120 * i}" y="12" class="legend" '
                   f'fill="{c}">&#9679; {_html.escape(n)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def _chart_slope(meta: dict, data: list[str]) -> str | None:
    """Two-state slope graph: ``slope LABEL | v_left | v_right``.

    ``x:`` may name the two states ("v2 -> v3")."""
    rows: list[tuple[str, float, float]] = []
    for line in data:
        if not line.lower().startswith("slope "):
            return None
        parts = [p.strip() for p in line[6:].split("|")]
        if len(parts) != 3:
            return None
        try:
            rows.append((parts[0], _num(parts[1]), _num(parts[2])))
        except ValueError:
            return None
    if not rows:
        return None
    vals = [v for _, a, b in rows for v in (a, b)]
    lo, hi = _scale(vals)
    xl, xr = 190, _SVG_W - 190
    out = [f'<svg viewBox="0 0 {_SVG_W} {_SVG_H}" class="chart-svg" role="img">']
    def Y(v): return _SVG_H - 30 - (v - lo) / (hi - lo) * (_SVG_H - 55)
    states = [s.strip() for s in (meta.get("x") or "before -> after").split("->")]
    out.append(f'<text x="{xl}" y="16" class="axis" text-anchor="middle">'
               f'{_html.escape(states[0])}</text>')
    out.append(f'<text x="{xr}" y="16" class="axis" text-anchor="middle">'
               f'{_html.escape(states[-1])}</text>')
    out.append(f'<line x1="{xl}" y1="24" x2="{xl}" y2="{_SVG_H - 24}" class="frame"/>')
    out.append(f'<line x1="{xr}" y1="24" x2="{xr}" y2="{_SVG_H - 24}" class="frame"/>')
    for i, (label, a, b) in enumerate(rows):
        c = _CHART_PALETTE[i % len(_CHART_PALETTE)]
        out.append(f'<line x1="{xl}" y1="{Y(a):.1f}" x2="{xr}" y2="{Y(b):.1f}" '
                   f'stroke="{c}" stroke-width="2.2"/>')
        out.append(f'<circle cx="{xl}" cy="{Y(a):.1f}" r="4" fill="{c}"/>')
        out.append(f'<circle cx="{xr}" cy="{Y(b):.1f}" r="4" fill="{c}"/>')
        out.append(f'<text x="{xl - 8}" y="{Y(a) + 4:.1f}" class="tick" '
                   f'text-anchor="end">{_html.escape(label)} {a:.6g}</text>')
        out.append(f'<text x="{xr + 8}" y="{Y(b) + 4:.1f}" class="tick">'
                   f'{b:.6g}</text>')
    out.append("</svg>")
    return "\n".join(out)


def _chart_heatmap(meta: dict, data: list[str]) -> str | None:
    """Matrix with color intensity: ``cols:`` meta + ``row LABEL | v | v``."""
    cols = [c.strip() for c in (meta.get("cols") or "").split("|") if c.strip()]
    rows: list[tuple[str, list[float]]] = []
    for line in data:
        if line.lower().startswith("cols:"):
            cols = [c.strip() for c in line[5:].split("|") if c.strip()]
            continue
        if not line.lower().startswith("row "):
            return None
        parts = [p.strip() for p in line[4:].split("|")]
        if len(parts) < 2:
            return None
        try:
            rows.append((parts[0], [_num(v) for v in parts[1:]]))
        except ValueError:
            return None
    if not rows or not cols or any(len(v) != len(cols) for _, v in rows):
        return None
    flat = [v for _, vs in rows for v in vs]
    lo, hi = min(flat), max(flat)
    span = (hi - lo) or 1.0
    head = "".join(f'<th>{_html.escape(c)}</th>' for c in cols)
    body = []
    for label, vs in rows:
        cells = []
        for v in vs:
            frac = (v - lo) / span
            # light -> saturated blue; text flips to white on dark cells
            bg = f"rgba(76,120,168,{0.12 + 0.78 * frac:.2f})"
            fg = "#fff" if frac > 0.55 else "#1a1a1a"
            cells.append(f'<td style="background:{bg};color:{fg}">{v:.6g}</td>')
        body.append(f'<tr><th>{_html.escape(label)}</th>{"".join(cells)}</tr>')
    return ('<table class="heatmap"><thead><tr><th></th>' + head
            + "</tr></thead><tbody>" + "".join(body) + "</tbody></table>")


def _chart_spans(meta: dict, data: list[str]) -> str | None:
    """Gantt-style spans: ``span LABEL | start | end [| group]`` (numeric x)."""
    rows: list[tuple[str, float, float, str]] = []
    for line in data:
        if not line.lower().startswith("span "):
            return None
        parts = [p.strip() for p in line[5:].split("|")]
        if len(parts) not in (3, 4):
            return None
        try:
            rows.append((parts[0], _num(parts[1]), _num(parts[2]),
                         parts[3] if len(parts) == 4 else ""))
        except ValueError:
            return None
    if not rows:
        return None
    lo = min(r[1] for r in rows)
    hi = max(r[2] for r in rows)
    lo, hi = _scale([lo, hi])
    groups: list[str] = []
    for r in rows:
        if r[3] not in groups:
            groups.append(r[3])
    h = 26 * len(rows) + 50
    out = [f'<svg viewBox="0 0 {_SVG_W} {h}" class="chart-svg" role="img">']
    def X(v): return 190 + (v - lo) / (hi - lo) * (_SVG_W - 190 - 20)
    for t in _ticks(lo, hi):
        out.append(f'<line x1="{X(t):.1f}" y1="10" x2="{X(t):.1f}" '
                   f'y2="{h - 32}" class="grid"/>')
        out.append(f'<text x="{X(t):.1f}" y="{h - 18}" class="tick" '
                   f'text-anchor="middle">{_fmt_tick(t)}</text>')
    for i, (label, a, b, grp) in enumerate(rows):
        y = 16 + 26 * i
        c = _CHART_PALETTE[groups.index(grp) % len(_CHART_PALETTE)]
        out.append(f'<text x="182" y="{y + 12}" class="tick" '
                   f'text-anchor="end">{_html.escape(label)}</text>')
        out.append(f'<rect x="{X(a):.1f}" y="{y}" '
                   f'width="{max(X(b) - X(a), 1.5):.1f}" height="16" rx="3" '
                   f'fill="{c}" fill-opacity="0.85"/>')
    if meta.get("x"):
        out.append(f'<text x="{(190 + _SVG_W - 20) / 2}" y="{h - 4}" '
                   f'class="axis" text-anchor="middle">'
                   f'{_html.escape(meta["x"])}</text>')
    out.append("</svg>")
    return "\n".join(out)


def _chart_box(meta: dict, data: list[str]) -> str | None:
    """Box plots: ``box LABEL | min | q1 | median | q3 | max``."""
    rows: list[tuple[str, list[float]]] = []
    for line in data:
        if not line.lower().startswith("box "):
            return None
        parts = [p.strip() for p in line[4:].split("|")]
        # Optional trailing note (e.g. "n=19") joins the label — reviewers
        # annotate sample sizes and a rigid 6-field rule silently degraded
        # both distribution charts of an otherwise-compliant report to text.
        if len(parts) == 7:
            parts = [f"{parts[0]} ({parts[6]})"] + parts[1:6]
        if len(parts) != 6:
            return None
        try:
            q = [_num(v) for v in parts[1:]]
        except ValueError:
            return None
        if sorted(q) != q:
            return None
        rows.append((parts[0], q))
    if not rows:
        return None
    flat = [v for _, q in rows for v in q]
    lo, hi = _scale(flat)
    h = 34 * len(rows) + 50
    out = [f'<svg viewBox="0 0 {_SVG_W} {h}" class="chart-svg" role="img">']
    def X(v): return 190 + (v - lo) / (hi - lo) * (_SVG_W - 190 - 20)
    for t in _ticks(lo, hi):
        out.append(f'<line x1="{X(t):.1f}" y1="10" x2="{X(t):.1f}" '
                   f'y2="{h - 32}" class="grid"/>')
        out.append(f'<text x="{X(t):.1f}" y="{h - 18}" class="tick" '
                   f'text-anchor="middle">{_fmt_tick(t)}</text>')
    for i, (label, (mn, q1, md, q3, mx)) in enumerate(rows):
        y = 18 + 34 * i
        c = _CHART_PALETTE[i % len(_CHART_PALETTE)]
        out.append(f'<text x="182" y="{y + 12}" class="tick" '
                   f'text-anchor="end">{_html.escape(label)}</text>')
        out.append(f'<line x1="{X(mn):.1f}" y1="{y + 8}" x2="{X(q1):.1f}" '
                   f'y2="{y + 8}" stroke="{c}" stroke-width="1.5"/>')
        out.append(f'<line x1="{X(q3):.1f}" y1="{y + 8}" x2="{X(mx):.1f}" '
                   f'y2="{y + 8}" stroke="{c}" stroke-width="1.5"/>')
        out.append(f'<rect x="{X(q1):.1f}" y="{y}" '
                   f'width="{max(X(q3) - X(q1), 1.5):.1f}" height="16" rx="2" '
                   f'fill="{c}" fill-opacity="0.45" stroke="{c}"/>')
        out.append(f'<line x1="{X(md):.1f}" y1="{y - 1}" x2="{X(md):.1f}" '
                   f'y2="{y + 17}" stroke="{c}" stroke-width="2.5"/>')
        for v in (mn, mx):
            out.append(f'<line x1="{X(v):.1f}" y1="{y + 3}" x2="{X(v):.1f}" '
                       f'y2="{y + 13}" stroke="{c}" stroke-width="1.5"/>')
    out.append("</svg>")
    return "\n".join(out)


def _parse_value_series(data: list[str]) -> list[tuple[str, list[float]]] | None:
    """Parse ``series NAME: v1, v2, v3, ...`` rows of raw sample values."""
    series: list[tuple[str, list[float]]] = []
    for line in data:
        if not line.lower().startswith("series ") or ":" not in line:
            return None
        name, rest = line[7:].split(":", 1)
        try:
            vals = [_num(v) for v in rest.replace(";", ",").split(",")
                    if v.strip()]
        except ValueError:
            return None
        if vals:
            series.append((name.strip(), vals))
    return series or None


def _bin(vals: list[float], lo: float, hi: float, n: int) -> list[int]:
    counts = [0] * n
    span = (hi - lo) or 1.0
    for v in vals:
        i = min(int((v - lo) / span * n), n - 1)
        counts[i] += 1
    return counts


def _chart_hist(meta: dict, data: list[str]) -> str | None:
    """Histogram from RAW values: ``series NAME: v1, v2, ...`` (the renderer
    bins). Multiple series overlay with transparency."""
    series = _parse_value_series(data)
    if series is None:
        return None
    allv = [v for _, vs in series for v in vs]
    lo, hi = _scale(allv)
    nbins = max(8, min(30, int(len(allv) ** 0.5)))
    peak = 0
    binned = []
    for name, vs in series:
        counts = _bin(vs, lo, hi, nbins)
        peak = max(peak, max(counts))
        binned.append((name, counts, len(vs)))
    out = _svg_frame(meta.get("title", ""), meta.get("x", ""),
                     meta.get("y", "count"), lo, hi, 0, peak * 1.05 or 1)
    def X(v): return _ML + (v - lo) / (hi - lo) * (_SVG_W - _ML - _MR)
    def Y(v): return _SVG_H - _MB - v / (peak * 1.05 or 1) * (_SVG_H - _MT - _MB)
    bw = (hi - lo) / nbins
    for i, (name, counts, n) in enumerate(binned):
        color = _CHART_PALETTE[i % len(_CHART_PALETTE)]
        for j, c in enumerate(counts):
            if not c:
                continue
            x0, x1 = X(lo + j * bw), X(lo + (j + 1) * bw)
            out.append(f'<rect x="{x0 + 0.5:.1f}" y="{Y(c):.1f}" '
                       f'width="{x1 - x0 - 1:.1f}" '
                       f'height="{Y(0) - Y(c):.1f}" fill="{color}" '
                       f'fill-opacity="{0.75 if len(binned) == 1 else 0.45}"/>')
        out.append(f'<text x="{_ML + 10}" y="{_MT + 16 + 16 * i}" '
                   f'class="legend" fill="{color}">&#9632; '
                   f'{_html.escape(name)} (n={n})</text>')
    out.append("</svg>")
    return "\n".join(out)


def _chart_density(meta: dict, data: list[str]) -> str | None:
    """Ridgeline density strips from RAW values, one ridge per series —
    compare many distributions at once (per-agent durations, per-run sizes)."""
    series = _parse_value_series(data)
    if series is None:
        return None
    allv = [v for _, vs in series for v in vs]
    lo, hi = _scale(allv)
    nbins = 60
    ridge_h, gap = 34, 10
    h = _MT + len(series) * (ridge_h + gap) + _MB
    out = [f'<svg viewBox="0 0 {_SVG_W} {h}" class="chart-svg" role="img">']
    def X(v): return 190 + (v - lo) / (hi - lo) * (_SVG_W - 190 - 20)
    for t in _ticks(lo, hi):
        out.append(f'<line x1="{X(t):.1f}" y1="{_MT}" x2="{X(t):.1f}" '
                   f'y2="{h - _MB + 6}" class="grid"/>')
        out.append(f'<text x="{X(t):.1f}" y="{h - _MB + 20}" class="tick" '
                   f'text-anchor="middle">{_fmt_tick(t)}</text>')
    for i, (name, vs) in enumerate(series):
        base = _MT + (i + 1) * (ridge_h + gap) - gap
        counts = _bin(vs, lo, hi, nbins)
        # 3-tap smoothing so sparse samples read as a shape, not a comb
        sm = [(counts[max(0, j - 1)] + 2 * counts[j]
               + counts[min(nbins - 1, j + 1)]) / 4 for j in range(nbins)]
        pk = max(sm) or 1.0
        bw = (hi - lo) / nbins
        pts = [f"{X(lo):.1f},{base:.1f}"]
        for j, c in enumerate(sm):
            pts.append(f"{X(lo + (j + 0.5) * bw):.1f},"
                       f"{base - c / pk * ridge_h:.1f}")
        pts.append(f"{X(hi):.1f},{base:.1f}")
        color = _CHART_PALETTE[i % len(_CHART_PALETTE)]
        out.append(f'<polygon points="{" ".join(pts)}" fill="{color}" '
                   f'fill-opacity="0.55" stroke="{color}" stroke-width="1.2"/>')
        med = sorted(vs)[len(vs) // 2]
        out.append(f'<line x1="{X(med):.1f}" y1="{base - ridge_h}" '
                   f'x2="{X(med):.1f}" y2="{base}" stroke="#333" '
                   f'stroke-width="1.4" stroke-dasharray="3,2"/>')
        out.append(f'<text x="182" y="{base - 6}" class="tick" '
                   f'text-anchor="end">{_html.escape(name)} (n={len(vs)})</text>')
    if meta.get("x"):
        out.append(f'<text x="{(190 + _SVG_W - 20) / 2}" y="{h - 4}" '
                   f'class="axis" text-anchor="middle">'
                   f'{_html.escape(meta["x"])}</text>')
    out.append("</svg>")
    return "\n".join(out)


_CHART_TYPES = {"bars": _chart_bars, "line": _chart_line,
                "scatter": _chart_scatter, "stacked": _chart_stacked,
                "grouped": _chart_grouped, "dumbbell": _chart_dumbbell,
                "slope": _chart_slope, "heatmap": _chart_heatmap,
                "spans": _chart_spans, "box": _chart_box,
                "hist": _chart_hist, "density": _chart_density}


# One place reads a block's requested type, so the renderer, the write gate and
# the strip pass can never disagree about what a block is. Deliberately strict:
# an unknown spelling is an error the writer is told to fix, not something this
# layer guesses at (a synonym table would let two spellings mean one chart and
# hide the mistake from the model that made it).
def canonical_kind(meta: dict) -> str:
    """The chart type this block asks for, lowercased and trimmed."""
    return str(meta.get("type", "bars")).strip().lower()


def _chart_to_html(body: str) -> str | None:
    """Render one chart block's inner text to HTML, or None if unparseable."""
    meta, data = _parse_header(body)
    kind = canonical_kind(meta)
    fn = _CHART_TYPES.get(kind)
    if fn is None:
        return None
    inner = fn(meta, data)
    if inner is None:
        return None
    title = meta.get("title", "")
    parts = ['<div class="chart">']
    if title and kind in ("bars", "stacked"):
        parts.append(f'<div class="chart-title">{_html.escape(title)}</div>')
    elif title:
        parts.append(f'<div class="chart-title">{_html.escape(title)}</div>')
    parts.append(inner)
    parts.append("</div>")
    return "\n".join(parts)


def _replace_chart_blocks(md_text: str) -> str:
    """Swap parseable ```chart fences for raw HTML before markdown rendering
    (the markdown library passes block-level raw HTML through unchanged)."""
    def _sub(m: "_re.Match[str]") -> str:
        rendered = _chart_to_html(m.group(1))
        return rendered if rendered is not None else m.group(0)
    return _CHART_FENCE_RE.sub(_sub, md_text)


def markdown_to_html(md_text: str, title: str = "Report") -> str:
    """Render markdown to a standalone HTML document.

    Falls back to a <pre> block when no markdown library is installed, so a
    report is never lost to a missing dependency.
    """
    try:
        import markdown as _md

        body = _md.markdown(
            _replace_chart_blocks(md_text),
            extensions=["tables", "fenced_code", "toc", "sane_lists"],
        )
    except ImportError:
        body = "<pre>" + _html.escape(md_text) + "</pre>"
    return _SHELL.format(title=_html.escape(title), css=_CSS, body=body)


def write_report_pair(out_dir: Path, md_text: str,
                      stem: str = "REPORT") -> tuple[Path, Path]:
    """Write ``<stem>.md`` and ``<stem>.html`` side by side; return both paths.

    The markdown is written first: it is the artifact factcheck.py audits, so
    an HTML rendering failure must never cost us the report itself.
    """
    out_dir = Path(out_dir)
    md_path = out_dir / f"{stem}.md"
    md_path.write_text(md_text)
    html_path = out_dir / f"{stem}.html"
    title = f"{stem} — {out_dir.name}"
    try:
        html_path.write_text(markdown_to_html(md_text, title=title))
    except Exception as exc:  # rendering must not break the run
        html_path.write_text(
            _SHELL.format(title=_html.escape(title), css=_CSS,
                          body=f"<p>HTML rendering failed: "
                               f"{_html.escape(str(exc))}</p><pre>"
                               + _html.escape(md_text) + "</pre>"))
    return md_path, html_path


def strip_unrenderable_charts(markdown: str) -> tuple[str, int]:
    """Remove every ```chart fence that would not render; return (md, n).

    Last line of defense for the warn-accept path: a page published after
    exhausted refusals must never show raw fence text where a chart should
    be (measured failure 2026-08-07: two batteries shipped pages with dead
    bars/spans blocks that readers hit before any factcheck did).
    """
    removed = 0

    def _keep_or_drop(m: "_re.Match[str]") -> str:
        nonlocal removed
        body = m.group(1)
        meta, data = _parse_header(body)
        fn = _CHART_TYPES.get(canonical_kind(meta))
        try:
            ok = fn is not None and fn(meta, data) is not None
        except Exception:
            ok = False
        if ok:
            return m.group(0)
        removed += 1
        return ("*[chart removed at publication: block did not parse; "
                "the numbers remain in the surrounding text]*")

    out = _re.sub(r"```chart[ \t]*\n(.*?)```", _keep_or_drop, markdown,
                  flags=_re.DOTALL)
    return out, removed


def chart_errors(markdown: str) -> list[str]:
    """Validate every ```chart block against the real renderer.

    Returns one message per block that would NOT render (unknown type, or
    the type's parser rejects the body). The write gate and factcheck call
    this so an unrenderable chart is refused at write time and flagged at
    verification time — never silently published as raw fence text
    (measured failure 2026-08-07: a writer invented a YAML chart dialect,
    every gate counted the blocks without parsing them, and the HTML
    shipped with zero rendered charts).
    """
    errors: list[str] = []
    blocks = _re.findall(r"```chart[ \t]*\n(.*?)```", markdown, _re.DOTALL)
    for i, body in enumerate(blocks, 1):
        meta, data = _parse_header(body)
        kind = canonical_kind(meta)
        fn = _CHART_TYPES.get(kind)
        if fn is None:
            errors.append(
                f"chart {i}: unknown type '{kind}' — supported types are "
                + ", ".join(sorted(_CHART_TYPES)))
            continue
        try:
            rendered = fn(meta, data)
        except Exception as exc:  # a parser crash is a body error
            rendered = None
            errors.append(f"chart {i} (type {kind}): body raised "
                          f"{type(exc).__name__}: {exc}")
            continue
        if rendered is None:
            errors.append(
                f"chart {i} (type {kind}): body did not parse — check the "
                "row grammar for this type (pipe-separated rows, not "
                "YAML/JSON lists)")
    return errors
