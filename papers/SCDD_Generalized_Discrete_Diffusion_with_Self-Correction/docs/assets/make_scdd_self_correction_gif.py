from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

try:
  from fontTools.ttLib import TTFont
except Exception:  # pragma: no cover - fontTools is expected to be present
  TTFont = None


WIDTH = 1900

DATA = Path(__file__).with_name("scdd_real_trajectory_sample8.json")
OUT = Path(__file__).with_name("scdd_self_correction_without_remasking.gif")

FONT_DIR = Path("/usr/share/fonts/dejavu-sans-fonts")
SANS = FONT_DIR / "DejaVuSans.ttf"
SANS_BOLD = FONT_DIR / "DejaVuSans-Bold.ttf"

# Palette --------------------------------------------------------------------
PAPER = "#ffffff"
INK = "#11161f"        # headings
MUTED = "#7b8491"      # secondary labels
DIVIDER = "#eef1f5"

MASK = "#cdd3dd"       # masked slots ([MASK] chips)
PENDING = "#9aa3b0"    # revealed but not yet equal to the final token (still self-correcting)
FINAL = "#1b2430"      # revealed and settled on its final token
FLASH = "#e8590c"      # a token that changed on this exact denoising step

# Layout geometry ------------------------------------------------------------
LEFT = 60
RIGHT = WIDTH - LEFT
BODY_TOP = 74
LINE_H = 19
GAP = 13               # constant spacing between every adjacent item on a line
HEADER_DIVIDER_Y = 58
MASK_PLACEHOLDER = "[MASK]"


def load_font(path: Path, size: int) -> ImageFont.FreeTypeFont:
  return ImageFont.truetype(str(path), size)


TOKEN_FONT = load_font(SANS, 12)
TOKEN_BOLD = load_font(SANS_BOLD, 12)
SMALL_FONT = load_font(SANS, 13)
LEGEND_FONT = load_font(SANS, 12)

_MEASURE = ImageDraw.Draw(Image.new("RGB", (4, 4)))


def _font_codepoints() -> set[int] | None:
  if TTFont is None:
    return None
  try:
    return set(TTFont(str(SANS)).getBestCmap().keys())
  except Exception:
    return None


RENDERABLE = _font_codepoints()


def sanitize(text: str) -> str:
  """Drop everything the figure should never show: broken byte-pair fragments
  (U+FFFD), glyphs the font cannot draw (CJK tofu, box-drawing, ...), and other
  non-printable noise. Internal spacing is preserved; the ends are trimmed."""
  text = text.replace("\n", " ").replace("\t", " ")
  out = []
  for ch in text:
    if ch == " ":
      out.append(ch)
      continue
    if not ch.isprintable() or ord(ch) == 0xFFFD:
      continue
    if RENDERABLE is not None and ord(ch) not in RENDERABLE:
      continue
    out.append(ch)
  # collapse runs of spaces created by the drops, then trim the ends
  return " ".join("".join(out).split())


def text_width(text: str, font: ImageFont.FreeTypeFont) -> int:
  if not text:
    return 0
  bbox = _MEASURE.textbbox((0, 0), text, font=font)
  return bbox[2] - bbox[0]


def is_break(final_text: str) -> bool:
  return "\n" in final_text


def cell_glyph(final_text: str, cell: dict, prev_cell: dict | None):
  """What to draw for one word at one state: (text, font, fill), or None to
  skip (a break, or a revealed-but-illegible byte fragment)."""
  if cell["masked"]:
    return MASK_PLACEHOLDER, TOKEN_FONT, MASK

  disp = sanitize(cell["text"])
  if not disp:
    return None

  settled = (not cell["partial"]) and cell["text"] == final_text
  changed = prev_cell is not None and (
      cell["text"] != prev_cell["text"] or cell["masked"] != prev_cell["masked"])
  if changed:
    return disp, TOKEN_BOLD, FLASH
  if settled:
    return disp, TOKEN_FONT, FINAL
  return disp, TOKEN_FONT, PENDING


def layout_frame(payload: dict, state_idx: int, prev_idx: int | None):
  """Flow the words for one state left-to-right with a constant gap between
  every adjacent item, wrapping at the right margin. Each word takes exactly
  its own rendered width, so spacing is uniform while widths adapt per step.
  Returns (items, bottom_y) where items are (x, y, text, font, fill)."""
  groups = payload["visualization"]["groups"]
  state = payload["visualization"]["states"][state_idx]
  prev = payload["visualization"]["states"][prev_idx] if prev_idx is not None else None

  items = []
  x = LEFT
  y = BODY_TOP
  for gi, group in enumerate(groups):
    final_text = group["final_text"]
    if is_break(final_text):
      x = LEFT
      y += LINE_H
      continue

    glyph = cell_glyph(final_text, state[gi], prev[gi] if prev is not None else None)
    if glyph is None:
      continue
    text, font, fill = glyph
    width = text_width(text, font)

    gap = GAP if x > LEFT else 0
    if x + gap + width > RIGHT:
      x = LEFT
      y += LINE_H
      gap = 0
    x += gap
    items.append((x, y, text, font, fill))
    x += width

  return items, y + LINE_H


def step_label(payload: dict, state_idx: int) -> str:
  num_steps = payload["num_steps"]
  if state_idx == 0:
    return "start  ·  all [MASK]"
  if state_idx == len(payload["states"]) - 1:
    return "final sample"
  return f"denoising step {state_idx:03d} / {num_steps}"


def draw_header(draw: ImageDraw.ImageDraw, payload: dict, state_idx: int) -> None:
  label = step_label(payload, state_idx)
  draw.text((LEFT, 28), "512-token trajectory, decoded as it denoises from all [MASK] to text",
            font=SMALL_FONT, fill=MUTED)
  draw.text((RIGHT - text_width(label, SMALL_FONT), 28), label, font=SMALL_FONT, fill=MUTED)
  draw.line((LEFT, HEADER_DIVIDER_Y, RIGHT, HEADER_DIVIDER_Y), fill=DIVIDER, width=2)


def draw_legend(draw: ImageDraw.ImageDraw, y: int) -> None:
  items = [
      (MASK, "masked"),
      (PENDING, "forming"),
      (FINAL, "settled"),
      (FLASH, "updated this step"),
  ]
  x = LEFT
  for color, label in items:
    draw.rectangle((x, y + 3, x + 10, y + 13), fill=color)
    x += 16
    draw.text((x, y), label, font=LEGEND_FONT, fill=MUTED)
    x += text_width(label, LEGEND_FONT) + 26


def draw_footer(draw: ImageDraw.ImageDraw, payload: dict, state_idx: int, y: int) -> None:
  state = payload["states"][state_idx]
  mask_index = payload["mask_index"]
  mask_count = sum(1 for token_id in state if token_id == mask_index)
  decoded = len(state) - mask_count

  draw.line((LEFT, y, RIGHT, y), fill=DIVIDER, width=2)
  draw_legend(draw, y + 12)
  footer = f"decoded {decoded:03d} / 512      remaining [MASK] {mask_count:03d}"
  draw.text((RIGHT - text_width(footer, SMALL_FONT), y + 12), footer,
            font=SMALL_FONT, fill=MUTED)


def make_frame(payload: dict, state_idx: int, prev_idx: int | None,
               footer_y: int, height: int) -> Image.Image:
  items, _ = layout_frame(payload, state_idx, prev_idx)
  image = Image.new("RGB", (WIDTH, height), PAPER)
  draw = ImageDraw.Draw(image)
  draw_header(draw, payload, state_idx)
  for x, y, text, font, fill in items:
    draw.text((x, y), text, font=font, fill=fill)
  draw_footer(draw, payload, state_idx, footer_y)
  return image


def _hex(color: str) -> tuple[int, int, int]:
  color = color.lstrip("#")
  return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))


def _ramp(stop: str, steps: int) -> list[tuple[int, int, int]]:
  r, g, b = _hex(stop)
  out = []
  for i in range(steps):
    t = i / (steps - 1)
    out.append((round(255 + (r - 255) * t),
                round(255 + (g - 255) * t),
                round(255 + (b - 255) * t)))
  return out


def build_palette() -> Image.Image:
  """A fixed palette so the orange "updated" flash always survives quantization,
  every frame shares one palette (no flicker), and frames delta-compress well.
  Anti-aliased text edges are covered by white->color ramps."""
  colors: list[tuple[int, int, int]] = []
  colors += _ramp(FINAL, 40)     # ink / settled text + neutral AA
  colors += _ramp(MASK, 14)      # masked chips
  colors += _ramp(PENDING, 14)   # forming text
  colors += _ramp(MUTED, 10)     # labels
  colors += _ramp(FLASH, 28)     # the orange update flash (kept generous)
  colors.append(_hex(DIVIDER))
  seen = {}
  for c in colors:
    seen.setdefault(c, None)
  uniq = list(seen.keys())[:255]
  uniq.append((255, 255, 255))
  flat: list[int] = []
  for c in uniq:
    flat += list(c)
  flat += [255, 255, 255] * (256 - len(uniq))
  pal = Image.new("P", (1, 1))
  pal.putpalette(flat)
  return pal


def main() -> None:
  payload = json.loads(DATA.read_text(encoding="utf-8"))
  palette = build_palette()
  num_states = len(payload["states"])
  last = num_states - 1

  # The animation: all-[MASK] start, the 128 denoising steps, then the final
  # sample. (prev_idx, this_idx, duration_ms)
  plan = [(None, 0, 900)]
  for idx in range(1, num_states):
    duration = 360 if idx == last else 70
    plan.append((idx - 1, idx, duration))
  # rest on a clean, fully-settled final frame (compared with itself, so nothing
  # is flagged as "updated this step")
  plan.append((last, last, 2400))

  # One fixed canvas height: tall enough for whichever frame wraps to the most
  # lines (typically the dense all-[MASK] start).
  bottoms = [layout_frame(payload, idx, prev)[1] for prev, idx, _ in plan]
  footer_y = max(bottoms) + 10
  height = footer_y + 44

  paletted = []
  durations = []
  for prev_idx, idx, duration in plan:
    frame = make_frame(payload, idx, prev_idx, footer_y, height)
    paletted.append(frame.quantize(palette=palette, dither=Image.Dither.NONE))
    durations.append(duration)

  paletted[0].save(
      OUT,
      save_all=True,
      append_images=paletted[1:],
      duration=durations,
      loop=0,
      optimize=True,
      disposal=1)
  size_mb = OUT.stat().st_size / 1e6
  print(f"Wrote {OUT}  ({len(paletted)} frames, {WIDTH}x{height}, {size_mb:.1f} MB)")


if __name__ == "__main__":
  main()
