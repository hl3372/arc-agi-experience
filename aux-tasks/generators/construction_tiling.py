"""Construction & tiling family generators: tasks 3, 14, 44, 45, 78, 86."""
from __future__ import annotations

import numpy as np

from common.grid import (
    COLOR_NAMES,
    NUM_COLORS,
    decorate_background_cells,
    generate_scene,
    grid_to_list,
    make_grid,
    pick_colors,
    populate_scene,
    random_background,
    random_grid_size,
)
from generators.core2_extra.utils import add_patterned_background
from registry import register

MAX_RETRIES = 20


# ---------------------------------------------------------------------------
# Task 3 — Place a cell at a coordinate (grid -> grid)
# ---------------------------------------------------------------------------

_T3_K_BANDS = ((1, 1, 0.25), (2, 4, 0.35), (5, 8, 0.25), (9, 12, 0.15))
_T3_NOOP_P = 0.15          # mixed mode: per-placement chance of repeating the cell's colour
_T3_SINGLE_REPEAT_P = 0.10  # single mode: chance the shared colour is copied from one target cell
_T3_K1_NOOP_P = 0.12        # k = 1: chance the listed colour is the cell's own colour
_T3_SHAPES = ("h_line", "v_line", "diagonal", "L", "rect_outline")


def _t3_sample_k(rng: np.random.Generator, area: int) -> int:
    """Placements per example: 1 / 2-4 / 5-8 / 9-12 at 25 / 35 / 25 / 15 %, capped at area // 3."""
    cap = max(1, area // 3)
    u = float(rng.random())
    acc = 0.0
    lo, hi = 1, 1
    for lo, hi, prob in _T3_K_BANDS:
        acc += prob
        if u < acc:
            break
    lo, hi = min(lo, cap), min(hi, cap)
    return int(rng.integers(lo, hi + 1)) if hi > lo else int(lo)


def _t3_structured_cells(rng: np.random.Generator, k: int, H: int, W: int):
    """k cells forming a line, a diagonal, an L or a rectangle outline; None when nothing fits."""
    options = []
    if W >= k:
        options.append("h_line")
    if H >= k:
        options.append("v_line")
    if H >= k and W >= k and k >= 2:
        options.append("diagonal")
    if k >= 3 and any(a <= H and (k + 1 - a) <= W for a in range(2, k)):
        options.append("L")
    rects = [(a, b) for a in range(3, H + 1) for b in range(3, W + 1) if 2 * (a + b) - 4 == k]
    if rects:
        options.append("rect_outline")
    if not options:
        return None, None
    shape = options[int(rng.integers(0, len(options)))]
    if shape == "h_line":
        r = int(rng.integers(0, H))
        c0 = int(rng.integers(0, W - k + 1))
        cells = [(r, c0 + i) for i in range(k)]
    elif shape == "v_line":
        c = int(rng.integers(0, W))
        r0 = int(rng.integers(0, H - k + 1))
        cells = [(r0 + i, c) for i in range(k)]
    elif shape == "diagonal":
        r0 = int(rng.integers(0, H - k + 1))
        c0 = int(rng.integers(0, W - k + 1))
        if rng.random() < 0.5:
            cells = [(r0 + i, c0 + i) for i in range(k)]
        else:
            cells = [(r0 + i, c0 + k - 1 - i) for i in range(k)]
    elif shape == "L":
        arms = [a for a in range(2, k) if a <= H and (k + 1 - a) <= W]
        a = arms[int(rng.integers(0, len(arms)))]
        b = k + 1 - a
        r0 = int(rng.integers(0, H - a + 1))
        c0 = int(rng.integers(0, W - b + 1))
        corner = int(rng.integers(0, 4))
        rows = list(range(r0, r0 + a))
        cols = list(range(c0, c0 + b))
        cr = rows[-1] if corner in (0, 1) else rows[0]
        cc = cols[0] if corner in (0, 2) else cols[-1]
        cells = sorted({(r, cc) for r in rows} | {(cr, c) for c in cols})
    else:
        a, b = rects[int(rng.integers(0, len(rects)))]
        r0 = int(rng.integers(0, H - a + 1))
        c0 = int(rng.integers(0, W - b + 1))
        cells = sorted({(r0 + r, c0 + c) for r in range(a) for c in range(b)
                        if r in (0, a - 1) or c in (0, b - 1)})
    return shape, cells


def _t3_validate(inp: np.ndarray, out: np.ndarray, rows, cols, colors, n_noop: int) -> None:
    """Independent check of the displayed pair against the listed placements; raises on any violation."""
    H, W = inp.shape
    if out.shape != inp.shape:
        raise RuntimeError("task 3: output shape differs")
    positions = list(zip(rows, cols))
    if len(set(positions)) != len(positions):
        raise RuntimeError("task 3: duplicate placement positions")
    expected = inp.copy()
    for r, c, col in zip(rows, cols, colors):
        if not (0 <= r < H and 0 <= c < W):
            raise RuntimeError("task 3: placement out of bounds")
        expected[r, c] = col
    if not np.array_equal(expected, out):
        raise RuntimeError("task 3: output is not the input with the placements applied")
    if sum(int(inp[r, c]) == int(col) for r, c, col in zip(rows, cols, colors)) != n_noop:
        raise RuntimeError("task 3: n_noop mismatch")


@register("3")
def generate_place_cell(rng: np.random.Generator) -> dict:
    """Task 3: paint k listed cells with listed colours (0-indexed coordinates).

    Sizes 1..30 per side, independent (the full 30x30 canvas is allowed).
    k = 1 (25%) is the original single-placement task; otherwise 2..12 cells.
    Colour mode: one shared colour or a colour per placement (50/50).  Layout:
    scattered (70%) or a structured line / diagonal / L / rectangle outline
    (30%), always listed as plain coordinates in row-major (60%) or shuffled
    (40%) order.  A placement may repeat the cell's current colour (a no-op),
    so "listed cells always change" is not a usable shortcut; when every
    placement is a no-op the output equals the input.  The pair is verified
    independently against the metadata and the generator raises on any
    violation.
    """
    h, w = random_grid_size(rng, min_dim=1, max_dim=30)
    bg = int(rng.integers(0, NUM_COLORS))
    grid = make_grid(h, w, fill=bg)
    # Palette of 2..9 object colours (small palettes common, the full 9 reachable) used
    # round-robin over 1..9 objects, so busy canvases can show all 10 colours.
    n_colors = int(rng.choice([2, 3, 4, 5, 6, 7, 8, 9], p=[0.22, 0.22, 0.16, 0.12, 0.10, 0.08, 0.06, 0.04]))
    palette = [int(c) for c in pick_colors(rng, n_colors, exclude=[bg])]
    n_objects = int(rng.integers(1, min(9, max(2, h * w // 40)) + 1))
    for i in range(n_objects):
        populate_scene(grid, rng, bg_color=bg, n_objects=1, palette=[palette[i % len(palette)]])

    # Display-only background variety; placing cells is unambiguous so no guard is needed.
    object_colors = {int(c) for c in np.unique(grid) if int(c) != bg}
    grid = decorate_background_cells(rng, grid, bg, object_colors)
    input_grid = grid.copy()

    # --- placements
    k = _t3_sample_k(rng, h * w)
    layout, shape = "scattered", None
    cells = None
    if k >= 2 and rng.random() < 0.30:
        shape, cells = _t3_structured_cells(rng, k, h, w)
        if cells is not None:
            layout = "structured"
    if cells is None:
        flat = rng.choice(h * w, size=k, replace=False)
        cells = [(int(i) // w, int(i) % w) for i in flat.tolist()]
    current = [int(input_grid[r, c]) for r, c in cells]

    colour_mode = "single" if (k == 1 or rng.random() < 0.5) else "mixed"
    if colour_mode == "single":
        if k == 1:
            # original task: a colour that differs from the cell, or (12%) the cell's own colour
            if rng.random() < _T3_K1_NOOP_P:
                shared = current[0]
            else:
                shared = int(rng.choice([c for c in range(NUM_COLORS) if c != current[0]]))
        elif rng.random() < _T3_SINGLE_REPEAT_P:
            shared = current[int(rng.integers(0, k))]
        else:
            shared = int(rng.integers(0, NUM_COLORS))
        colors = [shared] * k
    else:
        colors = []
        for cur in current:
            if rng.random() < _T3_NOOP_P:
                colors.append(cur)
            else:
                colors.append(int(rng.choice([c for c in range(NUM_COLORS) if c != cur])))

    order_mode = "row_major"
    idx = list(range(k))
    if k >= 2 and rng.random() < 0.40:
        order_mode = "shuffled"
        idx = rng.permutation(k).tolist()
    else:
        idx.sort(key=lambda i: cells[i])
    rows = [int(cells[i][0]) for i in idx]
    cols = [int(cells[i][1]) for i in idx]
    colors = [int(colors[i]) for i in idx]
    changed = [int(input_grid[r, c]) != col for r, c, col in zip(rows, cols, colors)]
    n_noop = int(k - sum(changed))

    output_grid = input_grid.copy()
    for r, c, col in zip(rows, cols, colors):
        output_grid[r, c] = col
    _t3_validate(input_grid, output_grid, rows, cols, colors, n_noop)

    metadata = {
        "n_placements": int(k),
        "colour_mode": colour_mode,
        "layout_mode": layout,
        "shape": shape,
        "order_mode": order_mode,
        "n_noop": n_noop,
        "output_equals_input": bool(n_noop == k),
        "placement_rows": rows,
        "placement_cols": cols,
        "placement_colors": colors,
        "placement_changed": [bool(x) for x in changed],
        "n_distinct_placed_colors": int(len(set(colors))),
    }
    if k == 1:
        metadata.update({"target_row": rows[0], "target_col": cols[0], "target_color": colors[0]})
    return {
        "input": grid_to_list(input_grid),
        "output": grid_to_list(output_grid),
        "metadata": metadata,
    }


def _format_task14_lattice_text(
    rng: np.random.Generator,
    n_rows: int,
    n_cols: int,
    interval: int,
    divider_color: int,
    bg_color: int,
) -> str:
    """Legacy wording helper (kept for reference; task 14 now renders through _t14_text)."""
    p = {"form": "full", "kind": "solid", "H": n_rows, "W": n_cols, "sr": interval, "sc": interval,
         "L": divider_color, "F": bg_color, "Z": bg_color, "region": (0, 0, n_rows - 1, n_cols - 1),
         "sr2": None, "sc2": None, "L2": None}
    return _t14_text(rng, p)[0]


# ---------------------------------------------------------------------------
# Task 14 — Generate lattice grid from text (text -> grid)
#
# Design (September 2026 review): the task builds a lattice from a text specification, so
# every variant must be fully determined by the text and rebuildable from it alone.
#   forms  full     the classic lattice: H = nr*sr + 1 rows, W = nc*sc + 1 columns, so the
#                   last row and column are lines (the border); nr, nc drawn independently
#                   (no forced squares); row spacing sr and column spacing sc independent
#                   (non-square cells) -- the original four wordings are kept when sr == sc.
#          partial  any H x W canvas (sr+2..30 by sc+2..30): lines at every multiple of the
#                   spacing from row 0 / column 0, the last cells cut off at the far edge.
#          region   the lattice occupies an inclusive sub-rectangle of a larger canvas in a
#                   third colour; spacings count from the rectangle's top-left corner.
#          overlay  a second lattice (its own spacings and colour) painted over the first;
#                   the text says the second colour wins where they meet.
#   kinds  solid | thick (two cells: the multiple and the next index) | dotted (only even
#          offsets along each line are painted) -- kinds apply to the full and partial forms.
#   Colours are part of the spec (line, fill, canvas, second line) and are named by
#   necessity; indices are 0-based and every wording says so where it matters.
#   Every wording is a template used both to render the text and, as a regular expression,
#   to parse it back; the generator rebuilds the grid from the parsed text and raises on
#   any difference.  Shares: full 30%, partial 25%, region 15%, overlay 15%, thick 8%,
#   dotted 7%.  Sizes: the full form keeps its structural n*spacing+1 ceiling, every other
#   form reaches any 1..30 canvas the spacing allows.
# ---------------------------------------------------------------------------

import re as _re

_T14_FORM_P = {"full": 0.30, "partial": 0.25, "region": 0.15, "overlay": 0.15, "thick": 0.08, "dotted": 0.07}

# Base wordings.  Placeholders render as digits and parse back as integers; a repeated
# placeholder must repeat the same value.  {sr2x} = 2*sr, {sc2x} = 2*sc, {Hm1} = H-1, {Wm1} = W-1.
_T14_BASE = {
    # --- full, square spacing (the original four wordings)
    "full_sq_1": "Create a {H} by {W} grid (rows by columns), with row and column indices starting at 0. Paint color {L} on every row and column whose index is a multiple of {sr}. Keep the outer border in color {L}. Fill every other cell with color {F}.",
    "full_sq_2": "Construct a {H}x{W} lattice ({H} rows by {W} columns). Use color {L} for divider lines every {sr} cells horizontally and vertically, starting from the top and left edges. The outer border is part of the divider lines. All remaining cells are color {F}.",
    "full_sq_3": "Make a {H} by {W} canvas (rows by columns) with background color {F}. Then color rows 0, {sr}, {sr2x}, ... and columns 0, {sr}, {sr2x}, ... with color {L}. Also color row {Hm1} and column {Wm1} with color {L}.",
    "full_sq_4": "Build a {H}x{W} grid ({H} rows by {W} columns). Draw horizontal and vertical lattice lines in color {L} at spacing {sr}, beginning at row 0 and column 0, and include the border lines. Everything that is not a divider line should be color {F}.",
    # --- full, rectangular cells
    "full_rect_1": "Create a {H} by {W} grid (rows by columns), with row and column indices starting at 0. Paint color {L} on every row whose index is a multiple of {sr} and on every column whose index is a multiple of {sc}. Keep the outer border in color {L}. Fill every other cell with color {F}.",
    "full_rect_2": "Construct a {H}x{W} lattice ({H} rows by {W} columns). Use color {L} for horizontal divider lines every {sr} rows and vertical divider lines every {sc} columns, starting from the top and left edges. The outer border is part of the divider lines. All remaining cells are color {F}.",
    "full_rect_3": "Make a {H} by {W} canvas (rows by columns) with background color {F}. Then color rows 0, {sr}, {sr2x}, ... and columns 0, {sc}, {sc2x}, ... with color {L}. Also color row {Hm1} and column {Wm1} with color {L}.",
    "full_rect_4": "Build a {H}x{W} grid ({H} rows by {W} columns). Draw horizontal lattice lines in color {L} every {sr} rows and vertical lattice lines every {sc} columns, beginning at row 0 and column 0, and include the border lines. Everything that is not a divider line should be color {F}.",
    # --- partial (cut off at the far edge)
    "partial_1": "Create a {H} by {W} grid (rows by columns), with row and column indices starting at 0. Paint color {L} on every row whose index is a multiple of {sr} and on every column whose index is a multiple of {sc}; the last line in each direction may fall short of the far edge, leaving a partial cell. Fill every other cell with color {F}.",
    "partial_2": "Make a {H}x{W} canvas ({H} rows by {W} columns) filled with color {F}. Starting from the top-left corner, draw horizontal lines in color {L} every {sr} rows (rows 0, {sr}, {sr2x}, ...) and vertical lines every {sc} columns (columns 0, {sc}, {sc2x}, ...); do not add any border line.",
    # --- region (sub-rectangle of a larger canvas)
    "region_1": "Create a {H} by {W} grid (rows by columns) filled with color {Z}, with indices starting at 0. Inside the rectangle from row {r0} to row {r1} and from column {c0} to column {c1} (inclusive), fill with color {F} and draw a lattice in color {L}: horizontal lines on every row whose offset from row {r0} is a multiple of {sr}, and vertical lines on every column whose offset from column {c0} is a multiple of {sc}. Leave everything outside the rectangle as color {Z}.",
    "region_2": "Make a {H}x{W} canvas ({H} rows by {W} columns) in color {Z}. A lattice occupies rows {r0} to {r1} and columns {c0} to {c1} (inclusive, 0-indexed): its cells are color {F}, with lines in color {L} every {sr} rows and every {sc} columns counted from the rectangle's top-left corner, which is itself on a line.",
}
_T14_SUFFIX = {
    "overlay2": " Then draw a second lattice in color {L2}: every row whose index is a multiple of {sr2} and every column whose index is a multiple of {sc2}, painted over the first lattice, so color {L2} wins wherever the two overlap.",
    "overlay3": " Then draw a third lattice in color {L3}: every row whose index is a multiple of {sr3} and every column whose index is a multiple of {sc3}, painted over both earlier lattices, so color {L3} wins wherever they overlap.",
    "thick": " Every lattice line is two cells thick: it covers its own index and the next index as well, skipping the second cell where it would leave the grid.",
    "dotted": " The lines are dotted: on each horizontal line only the cells at even column offsets from the left edge of the lattice are painted, and on each vertical line only the cells at even row offsets from its top edge; a cell is painted if either rule paints it.",
}
_T14_FIELD_RE = _re.compile(r"\{([A-Za-z0-9]+)\}")


def _t14_template_regex(template: str) -> str:
    """Turn a wording template into a regular expression with named integer groups
    (repeated placeholders become back-references)."""
    out = []
    seen: set[str] = set()
    pos = 0
    for m in _T14_FIELD_RE.finditer(template):
        out.append(_re.escape(template[pos:m.start()]))
        name = m.group(1)
        out.append(f"(?P={name})" if name in seen else f"(?P<{name}>\\d+)")
        seen.add(name)
        pos = m.end()
    out.append(_re.escape(template[pos:]))
    return "".join(out)


_T14_BASE_RE = {k: _re.compile(_t14_template_regex(v)) for k, v in _T14_BASE.items()}
_T14_SUFFIX_RE = {k: _re.compile(_t14_template_regex(v)) for k, v in _T14_SUFFIX.items()}


def _t14_lattice_mask(H: int, W: int, region, sr: int, sc: int, kind: str) -> np.ndarray:
    """Cells of one lattice inside the inclusive region (spacings count from its top-left)."""
    r0, c0, r1, c1 = region
    mask = np.zeros((H, W), dtype=bool)
    rows = [r for r in range(r0, r1 + 1) if (r - r0) % sr == 0]
    cols = [c for c in range(c0, c1 + 1) if (c - c0) % sc == 0]
    if kind == "thick":
        rows = sorted({rr for r in rows for rr in (r, r + 1) if rr <= r1})
        cols = sorted({cc for c in cols for cc in (c, c + 1) if cc <= c1})
    if kind == "dotted":
        for r in rows:
            for c in range(c0, c1 + 1):
                if (c - c0) % 2 == 0:
                    mask[r, c] = True
        for c in cols:
            for r in range(r0, r1 + 1):
                if (r - r0) % 2 == 0:
                    mask[r, c] = True
        return mask
    for r in rows:
        mask[r, c0:c1 + 1] = True
    for c in cols:
        mask[r0:r1 + 1, c] = True
    return mask


def _t14_build(p: dict) -> np.ndarray:
    """Build the grid a parameter set describes."""
    H, W = p["H"], p["W"]
    r0, c0, r1, c1 = p["region"]
    grid = make_grid(H, W, fill=p["Z"] if p["form"] == "region" else p["F"])
    grid[r0:r1 + 1, c0:c1 + 1] = p["F"]
    grid[_t14_lattice_mask(H, W, p["region"], p["sr"], p["sc"], p["kind"])] = p["L"]
    if p["form"] == "overlay":
        grid[_t14_lattice_mask(H, W, p["region"], p["sr2"], p["sc2"], "solid")] = p["L2"]
        if p.get("L3") is not None:
            grid[_t14_lattice_mask(H, W, p["region"], p["sr3"], p["sc3"], "solid")] = p["L3"]
    return grid


def _t14_fields(p: dict) -> dict:
    r0, c0, r1, c1 = p["region"]
    return {"H": p["H"], "W": p["W"], "sr": p["sr"], "sc": p["sc"], "L": p["L"], "F": p["F"], "Z": p["Z"],
            "sr2x": 2 * p["sr"], "sc2x": 2 * p["sc"], "Hm1": p["H"] - 1, "Wm1": p["W"] - 1,
            "r0": r0, "c0": c0, "r1": r1, "c1": c1, "sr2": p["sr2"], "sc2": p["sc2"], "L2": p["L2"],
            "sr3": p.get("sr3"), "sc3": p.get("sc3"), "L3": p.get("L3")}


def _t14_text(rng: np.random.Generator, p: dict) -> tuple[str, str]:
    """Render the text specification for a parameter set; returns (text, wording id)."""
    form, kind = p["form"], p["kind"]
    base_form = "full" if form in ("full", "thick", "dotted", "overlay") and p["base"] == "full" else \
        "partial" if form in ("partial", "thick", "dotted", "overlay") else "region"
    if base_form == "full":
        ids = [k for k in _T14_BASE if k.startswith("full_sq_" if p["sr"] == p["sc"] else "full_rect_")]
    elif base_form == "partial":
        ids = ["partial_1", "partial_2"]
    else:
        ids = ["region_1", "region_2"]
    wid = ids[int(rng.integers(0, len(ids)))]
    fields = _t14_fields(p)
    text = _T14_BASE[wid].format(**fields)
    if form == "overlay":
        text += _T14_SUFFIX["overlay2"].format(**fields)
        if p.get("L3") is not None:
            text += _T14_SUFFIX["overlay3"].format(**fields)
    elif kind in ("thick", "dotted"):
        text += _T14_SUFFIX[kind].format(**fields)
    return text, wid


def _parse_task14_text(text: str) -> dict:
    """Recover the parameter set from the text alone (the validator's reading)."""
    for wid, rx in _T14_BASE_RE.items():
        m = rx.match(text)
        if m is None:
            continue
        g = {k: int(v) for k, v in m.groupdict().items()}
        rest = text[m.end():]
        form = "region" if wid.startswith("region") else "partial" if wid.startswith("partial") else "full"
        kind = "solid"
        sr2 = sc2 = L2 = sr3 = sc3 = L3 = None
        while rest:
            for sid, srx in _T14_SUFFIX_RE.items():
                sm = srx.match(rest)
                if sm is None:
                    continue
                sg = {k: int(v) for k, v in sm.groupdict().items()}
                if sid == "overlay2":
                    form, sr2, sc2, L2 = "overlay", sg["sr2"], sg["sc2"], sg["L2"]
                elif sid == "overlay3":
                    if form != "overlay":
                        raise ValueError("task 14: third lattice without a second")
                    sr3, sc3, L3 = sg["sr3"], sg["sc3"], sg["L3"]
                else:
                    kind = sid
                rest = rest[sm.end():]
                break
            else:
                raise ValueError("task 14: unparsed suffix " + rest[:40])
        H, W = g["H"], g["W"]
        sr = g["sr"]
        sc = g.get("sc", sr)
        if wid.startswith("full_sq_") and "sr2x" in g and g["sr2x"] != 2 * sr:
            raise ValueError("task 14: inconsistent multiples")
        if "sr2x" in g and g["sr2x"] != 2 * sr or "sc2x" in g and g["sc2x"] != 2 * sc:
            raise ValueError("task 14: inconsistent multiples")
        if "Hm1" in g and (g["Hm1"] != H - 1 or g["Wm1"] != W - 1):
            raise ValueError("task 14: inconsistent border indices")
        region = (g["r0"], g["c0"], g["r1"], g["c1"]) if form == "region" else (0, 0, H - 1, W - 1)
        base = "partial" if wid.startswith("partial") else "full" if wid.startswith("full") else "region"
        return {"form": form, "kind": kind, "H": H, "W": W, "sr": sr, "sc": sc, "L": g["L"], "F": g["F"],
                "Z": g.get("Z", g["F"]), "region": region, "sr2": sr2, "sc2": sc2, "L2": L2,
                "sr3": sr3, "sc3": sc3, "L3": L3, "base": base, "wording": wid}
    raise ValueError("task 14: no wording matched")


def _t14_weighted(rng: np.random.Generator, table: dict) -> str:
    keys = list(table)
    probs = np.array([table[k] for k in keys], dtype=float)
    return keys[int(rng.choice(len(keys), p=probs / probs.sum()))]


def _t14_rint(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(lo) if hi <= lo else int(rng.integers(lo, hi + 1))


def _t14_divisor_spacings(n: int) -> list[int]:
    """Spacings 2..8 that divide ``n`` with at least two blocks (the full form's sides are
    blocks * spacing + 1)."""
    return [s for s in range(2, 9) if n % s == 0 and n // s >= 2]


def _t14_sample(rng: np.random.Generator, H: int, W: int) -> dict | None:
    """Draw one parameter set (form, kind, spacings, colours, region, overlay) for a canvas
    drawn FIRST (size-first keeps the per-value side density flat: the full form used to
    draw its own sides as blocks * spacing + 1, which piled up on sizes with many divisors).
    The full form takes its spacings from the divisors of H-1 / W-1 and is skipped where none
    fits; the dotted kind uses the partial wordings only (the full wordings assert a solid
    border, which a dotted line does not have).  None when no form fits (retried by the
    caller at the same size)."""
    full_rows, full_cols = _t14_divisor_spacings(H - 1), _t14_divisor_spacings(W - 1)
    full_ok = bool(full_rows and full_cols)
    region_ok = H >= 8 and W >= 8
    for _ in range(12):
        draw = _t14_weighted(rng, _T14_FORM_P)
        kind = draw if draw in ("thick", "dotted") else "solid"
        if draw == "dotted":
            form = base = "partial"
        elif draw == "thick":
            form = base = "full" if (full_ok and rng.random() < 0.5) else "partial"
        elif draw == "overlay":
            form, base = "overlay", ("full" if (full_ok and rng.random() < 0.5) else "partial")
        elif draw == "region":
            if not region_ok:
                continue
            form = base = "region"
        elif draw == "full":
            if not full_ok:
                continue
            form = base = "full"
        else:
            form = base = "partial"
        break
    else:
        return None
    if base == "full":
        sr = full_rows[int(rng.integers(0, len(full_rows)))]
        sc = sr if (sr in full_cols and rng.random() < 0.5) else full_cols[int(rng.integers(0, len(full_cols)))]
        region = (0, 0, H - 1, W - 1)
    elif base == "partial":
        sr = _t14_rint(rng, 2, min(8, H - 2))
        sc = _t14_rint(rng, 2, min(8, W - 2))
        region = (0, 0, H - 1, W - 1)
    else:  # region: the lattice rectangle is strictly smaller than the canvas in at least one dimension
        sr = _t14_rint(rng, 2, min(8, H - 4))
        sc = _t14_rint(rng, 2, min(8, W - 4))
        rh = _t14_rint(rng, sr + 2, H)
        rw = _t14_rint(rng, sc + 2, W)
        if rh == H and rw == W:
            if rng.random() < 0.5:
                rh = _t14_rint(rng, sr + 2, H - 1)
            else:
                rw = _t14_rint(rng, sc + 2, W - 1)
        r0 = _t14_rint(rng, 0, H - rh)
        c0 = _t14_rint(rng, 0, W - rw)
        region = (r0, c0, r0 + rh - 1, c0 + rw - 1)
    n_extra = 0 if form != "overlay" else (1 if rng.random() < 0.7 else 2)
    n_colors = 2 + n_extra + (1 if form == "region" else 0)
    cols = [int(c) for c in pick_colors(rng, n_colors)]
    p = {"form": form, "kind": kind, "base": base, "H": H, "W": W, "sr": sr, "sc": sc,
         "L": cols[0], "F": cols[1], "Z": cols[2] if form == "region" else cols[1],
         "region": region, "sr2": None, "sc2": None, "L2": None, "sr3": None, "sc3": None, "L3": None}
    if form == "overlay":
        used = {(sr, sc)}
        extra = []
        for _ in range(n_extra):
            for _try in range(30):
                cand = (_t14_rint(rng, 2, 8), _t14_rint(rng, 2, 8))
                if cand not in used:
                    used.add(cand)
                    extra.append(cand)
                    break
        p.update(sr2=extra[0][0], sc2=extra[0][1], L2=cols[2])
        if len(extra) > 1:
            p.update(sr3=extra[1][0], sc3=extra[1][1], L3=cols[3])
    return p


@register("14")
def generate_lattice_from_text(rng: np.random.Generator) -> dict:
    """Task 14: construct a lattice grid from a text specification (see the design note above).

    The canvas (H, W independent and uniform over 4..30) is drawn first and the parameter
    set is retried at that size, so the side density is flat and squares only happen by
    coincidence.  Every example is validated by parsing the text back with the template
    regular expressions and rebuilding the grid from the parsed parameters alone; the
    generator raises on any difference.
    """
    H = _t14_rint(rng, 4, 30)
    W = _t14_rint(rng, 4, 30)
    for _ in range(MAX_RETRIES):
        p = _t14_sample(rng, H, W)
        if p is None:
            continue
        grid = _t14_build(p)
        named = {p["L"], p["F"]} | ({p["Z"]} if p["form"] == "region" else set()) \
            | ({p["L2"]} if p["L2"] is not None else set()) | ({p["L3"]} if p.get("L3") is not None else set())
        present = {int(v) for v in np.unique(grid)}
        if len(present) < 2 or not named <= present:
            continue   # a denser overlay or a thick lattice swallowed a named colour: redraw
        text, wid = _t14_text(rng, p)
        parsed = _parse_task14_text(text)
        rebuilt = _t14_build(parsed)
        if not np.array_equal(rebuilt, grid):
            raise RuntimeError("task 14: the text does not rebuild the grid")
        r0, c0, r1, c1 = p["region"]
        n_row_lines = len([r for r in range(r0, r1 + 1) if (r - r0) % p["sr"] == 0])
        n_col_lines = len([c for c in range(c0, c1 + 1) if (c - c0) % p["sc"] == 0])
        metadata = {
            "form": p["form"], "kind": p["kind"], "wording": wid, "base": p["base"],
            "grid_h": int(p["H"]), "grid_w": int(p["W"]),
            "row_spacing": int(p["sr"]), "col_spacing": int(p["sc"]),
            "line_color": int(p["L"]), "fill_color": int(p["F"]),
            "canvas_color": int(p["Z"]) if p["form"] == "region" else None,
            "region": [int(v) for v in p["region"]],
            "row_spacing_2": p["sr2"], "col_spacing_2": p["sc2"], "line_color_2": p["L2"],
            "row_spacing_3": p.get("sr3"), "col_spacing_3": p.get("sc3"), "line_color_3": p.get("L3"),
            "n_lattices": 1 + (p["L2"] is not None) + (p.get("L3") is not None),
            "n_row_lines": int(n_row_lines), "n_col_lines": int(n_col_lines),
            "n_colors": int(len(np.unique(grid))),
        }
        return {"input": text, "output": grid_to_list(grid), "metadata": metadata}
    raise RuntimeError("task 14: no lattice after the retry budget")


# ---------------------------------------------------------------------------
# Task 44 — Tile grid in MxN arrangement (grid -> grid)
# ---------------------------------------------------------------------------


@register("44")
def generate_tile_grid(rng: np.random.Generator) -> dict:
    from generators.tiling_44_45 import generate_repeat
    return generate_repeat(rng)


@register("45")
def generate_infer_tile_scale(rng: np.random.Generator) -> dict:
    from generators.tiling_44_45 import generate_scale
    return generate_scale(rng)


# ---------------------------------------------------------------------------
# Task 78 — Stamp the shown pattern at named marker cells (grid-pair -> grid)
# ---------------------------------------------------------------------------

@register("78")
def generate_stamp_pattern(rng: np.random.Generator) -> dict:
    from generators.stamp_78 import generate_stamp
    return generate_stamp(rng)


