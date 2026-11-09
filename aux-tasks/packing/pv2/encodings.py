"""Grid encodings: how a grid is written as text, and how that text is read back.

Every encoding is a renderer plus a strict parser (malformed text raises), round-trip exact on any
rectangular grid of colours 0-9.  Episodes pick an encoding per grid and validate every rendered
grid by parsing it back (DESIGN.md, V1).

Flags say what an encoding hands over, so an episode never pairs an encoding with a question whose
skill it would give away:

* ``reveals_coords``      row / column numbers are printed (never with coordinate-reading tasks)
* ``reveals_dims``        the size can be read off without counting: a size line, or row / column
                          labels whose last number gives the height / width (never with a size question)
* ``reveals_background``  a default fill colour is printed (never with background-inference tasks)
* ``uses_names``          colours are words, so colour answers are words too

``describe`` holds interchangeable phrases used when a prompt must say how a grid (or a grid
answer) is written.  Symbol maps are deliberately absent (the user deferred them, 2026-09-29).
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable

Grid = list[list[int]]

COLOR_NAMES = ("black", "blue", "red", "green", "yellow", "grey", "magenta", "orange", "cyan", "maroon")
_NAME_INDEX = {name: i for i, name in enumerate(COLOR_NAMES)}


@dataclass(frozen=True)
class Encoding:
    name: str
    render: Callable[[Grid], str]
    parse: Callable[[str], Grid]
    describe: tuple[str, ...]
    reveals_coords: bool = False
    reveals_dims: bool = False
    reveals_background: bool = False
    uses_names: bool = False


def check_grid(grid) -> Grid:
    """A non-empty rectangular list of lists of ints 0-9 (raises otherwise)."""
    if not isinstance(grid, list) or not grid or not all(isinstance(r, list) and r for r in grid):
        raise ValueError("a grid is a non-empty list of non-empty rows")
    width = len(grid[0])
    for row in grid:
        if len(row) != width:
            raise ValueError("ragged grid")
        for v in row:
            if isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= 9:
                raise ValueError(f"cell value {v!r} is not a colour 0-9")
    return grid


def _lines(text: str) -> list[str]:
    lines = text.strip("\n").split("\n")
    if not lines or any(not line.strip() for line in lines):
        raise ValueError("empty line inside a grid")
    return [line.rstrip() for line in lines]


def _digit(ch: str) -> int:
    if len(ch) != 1 or ch not in "0123456789":
        raise ValueError(f"{ch!r} is not a digit colour")
    return int(ch)


# --- digit rows -------------------------------------------------------------

def _render_compact(g: Grid) -> str:
    return "\n".join("".join(str(v) for v in row) for row in check_grid(g))


def _parse_compact(text: str) -> Grid:
    return check_grid([[_digit(ch) for ch in line] for line in _lines(text)])


def _sep_codec(sep: str):
    def render(g: Grid) -> str:
        return "\n".join(sep.join(str(v) for v in row) for row in check_grid(g))

    def parse(text: str) -> Grid:
        rows = []
        for line in _lines(text):
            parts = line.split(sep) if sep.strip() else line.split(" ")
            rows.append([_digit(p) for p in parts])
        return check_grid(rows)
    return render, parse


_render_spaced, _parse_spaced = _sep_codec(" ")
_render_comma, _parse_comma = _sep_codec(",")


def _render_brackets(g: Grid) -> str:
    return "\n".join("[" + " ".join(str(v) for v in row) + "]" for row in check_grid(g))


def _parse_brackets(text: str) -> Grid:
    rows = []
    for line in _lines(text):
        if not (line.startswith("[") and line.endswith("]")):
            raise ValueError("bracketed row expected")
        rows.append([_digit(p) for p in line[1:-1].split(" ")])
    return check_grid(rows)


# --- JSON ---------------------------------------------------------------------

def _render_json_flat(g: Grid) -> str:
    return json.dumps(check_grid(g), separators=(",", ":"))


def _render_json_rows(g: Grid) -> str:
    rows = [json.dumps(row, separators=(",", ":")) for row in check_grid(g)]
    return "[" + ",\n ".join(rows) + "]"


def _parse_json(text: str) -> Grid:
    value = json.loads(text)
    return check_grid(value)


# --- colour names -------------------------------------------------------------

def _render_names(g: Grid) -> str:
    return "\n".join(" ".join(COLOR_NAMES[v] for v in row) for row in check_grid(g))


def _parse_names(text: str) -> Grid:
    rows = []
    for line in _lines(text):
        row = []
        for word in line.split(" "):
            if word not in _NAME_INDEX:
                raise ValueError(f"{word!r} is not a colour name")
            row.append(_NAME_INDEX[word])
        rows.append(row)
    return check_grid(rows)


# --- sparse listing -------------------------------------------------------------

_SPARSE_HEAD = re.compile(r"(\d+) rows x (\d+) columns, cells written \(row,col\)=color with rows and "
                          r"columns 0-indexed from the top-left; every cell is ([0-9])(?: except (.*))?\.")
_SPARSE_CELL = re.compile(r"\((\d+),(\d+)\)=([0-9])")


def _render_sparse(g: Grid) -> str:
    check_grid(g)
    fill = Counter(v for row in g for v in row).most_common(1)[0][0]
    cells = [f"({r},{c})={v}" for r, row in enumerate(g) for c, v in enumerate(row) if v != fill]
    # the index base is stated in the listing itself (presentation verifier, 2026-09-29): a reflect or
    # rotate answer read from a 1-based listing would come out shifted
    head = (f"{len(g)} rows x {len(g[0])} columns, cells written (row,col)=color with rows and columns "
            f"0-indexed from the top-left; every cell is {fill}")
    return head + (" except " + ", ".join(cells) if cells else "") + "."


def _parse_sparse(text: str) -> Grid:
    m = _SPARSE_HEAD.fullmatch(text.strip())
    if not m:
        raise ValueError("not a sparse grid listing")
    h, w, fill = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not (1 <= h <= 30 and 1 <= w <= 30):
        raise ValueError("sparse grid size out of range")
    grid = [[fill] * w for _ in range(h)]
    if m.group(4):
        parts = m.group(4).split(", ")
        seen = set()
        for part in parts:
            cm = _SPARSE_CELL.fullmatch(part)
            if not cm:
                raise ValueError(f"{part!r} is not a (row,col)=colour cell")
            r, c, v = int(cm.group(1)), int(cm.group(2)), int(cm.group(3))
            if not (0 <= r < h and 0 <= c < w) or (r, c) in seen or v == fill:
                raise ValueError(f"bad sparse cell {part!r}")
            seen.add((r, c))
            grid[r][c] = v
    return check_grid(grid)


# --- pipe tables ----------------------------------------------------------------

def _render_pipes(g: Grid) -> str:
    return "\n".join("|" + "|".join(str(v) for v in row) + "|" for row in check_grid(g))


def _parse_pipes(text: str) -> Grid:
    rows = []
    for line in _lines(text):
        if not (line.startswith("|") and line.endswith("|")):
            raise ValueError("pipe row expected")
        rows.append([_digit(p) for p in line[1:-1].split("|")])
    return check_grid(rows)


def _render_indexed_table(g: Grid) -> str:
    check_grid(g)
    w = len(g[0])
    head = "|   | " + " | ".join(f"c{c}" for c in range(w)) + " |"
    rule = "|---|" + "|".join("---" for _ in range(w)) + "|"
    body = ["| r" + str(r) + " | " + " | ".join(str(v) for v in row) + " |" for r, row in enumerate(g)]
    return "\n".join([head, rule, *body])


def _parse_indexed_table(text: str) -> Grid:
    lines = _lines(text)
    if len(lines) < 3:
        raise ValueError("indexed table too short")
    head = [p.strip() for p in lines[0].strip("|").split("|")]
    w = len(head) - 1
    if head[0] != "" or head[1:] != [f"c{c}" for c in range(w)]:
        raise ValueError("bad column header")
    rows = []
    for r, line in enumerate(lines[2:]):
        parts = [p.strip() for p in line.strip("|").split("|")]
        if parts[0] != f"r{r}" or len(parts) != w + 1:
            raise ValueError("bad indexed row")
        rows.append([_digit(p) for p in parts[1:]])
    return check_grid(rows)


# --- row-labelled -----------------------------------------------------------------

def _render_row_labelled(g: Grid) -> str:
    return "\n".join(f"row {r}: " + "".join(str(v) for v in row) for r, row in enumerate(check_grid(g)))


def _parse_row_labelled(text: str) -> Grid:
    rows = []
    for r, line in enumerate(_lines(text)):
        prefix = f"row {r}: "
        if not line.startswith(prefix):
            raise ValueError("row label missing or out of order")
        rows.append([_digit(ch) for ch in line[len(prefix):]])
    return check_grid(rows)


# --- the stage-2 ARC rendering (indexed rows + runs) ---------------------------------
# Copied byte-for-byte from ``stage2/arc_render.py`` (``_indexed_rows`` / ``_run_rows``, grid format
# ``indexed_runs_compact``), the format the ARC solver reads in stage 2.  The block header with the
# role label and "(H rows x W cols)" is layout (pv2.layouts), not part of the encoding.

STAGE2_RUNS_TITLE = "Runs of the same grid (color@columns):"


def _stage2_row_labels(count: int) -> list[str]:
    width = len(f"r{count - 1}")
    return [f"r{index}".ljust(width) for index in range(count)]


def _stage2_runs(values, axis: str) -> str:
    parts, start = [], 0
    for index in range(1, len(values) + 1):
        if index == len(values) or values[index] != values[start]:
            parts.append(f"{values[start]}@{axis}{start}" if index - start == 1
                         else f"{values[start]}@{axis}{start}-{axis}{index - 1}")
            start = index
    return " ".join(parts)


def _render_stage2(g: Grid) -> str:
    check_grid(g)
    rows = [f"{label} | " + "".join(str(cell) for cell in row)
            for label, row in zip(_stage2_row_labels(len(g)), g)]
    runs = [STAGE2_RUNS_TITLE] + [f"r{index}: {_stage2_runs(row, 'c')}" for index, row in enumerate(g)]
    return "\n".join(rows + runs)


def _parse_stage2(text: str) -> Grid:
    lines = _lines(text)
    if STAGE2_RUNS_TITLE not in lines:
        raise ValueError("stage-2 runs listing missing")
    cut = lines.index(STAGE2_RUNS_TITLE)
    body, runs = lines[:cut], lines[cut + 1:]
    labels = _stage2_row_labels(len(body)) if body else []
    grid = []
    for label, line in zip(labels, body):
        prefix = f"{label} | "
        if not line.startswith(prefix):
            raise ValueError("stage-2 row label missing or out of order")
        grid.append([_digit(ch) for ch in line[len(prefix):]])
    check_grid(grid)
    # the runs listing must describe exactly the same rows
    if runs != [f"r{i}: {_stage2_runs(row, 'c')}" for i, row in enumerate(grid)]:
        raise ValueError("stage-2 runs listing disagrees with the rows")
    return grid


def _render_runs(g: Grid) -> str:
    return "\n".join(f"r{index}: {_stage2_runs(row, 'c')}" for index, row in enumerate(check_grid(g)))


_RUN_TOKEN = re.compile(r"([0-9])@c(\d+)(?:-c(\d+))?")


def _parse_runs(text: str) -> Grid:
    rows = []
    for index, line in enumerate(_lines(text)):
        prefix = f"r{index}: "
        if not line.startswith(prefix):
            raise ValueError("runs row label missing or out of order")
        row = []
        for token in line[len(prefix):].split(" "):
            m = _RUN_TOKEN.fullmatch(token)
            if not m:
                raise ValueError(f"{token!r} is not a color@columns run")
            a = int(m.group(2))
            b = int(m.group(3)) if m.group(3) is not None else a
            if a != len(row) or b < a:
                raise ValueError("runs do not tile the row from the left")
            row.extend([int(m.group(1))] * (b - a + 1))
        rows.append(row)
    return check_grid(rows)


ENCODINGS: dict[str, Encoding] = {e.name: e for e in (
    Encoding("compact", _render_compact, _parse_compact,
             ("as rows of digits with no spaces, one row per line",
              "one line per row, each cell a single digit, nothing between them",
              "as digit rows without separators")),
    Encoding("spaced", _render_spaced, _parse_spaced,
             ("as rows of digits separated by single spaces, one row per line",
              "one line per row, cells separated by spaces")),
    Encoding("comma", _render_comma, _parse_comma,
             ("as rows of comma-separated digits, one row per line",
              "one line per row, cells separated by commas")),
    Encoding("json_flat", _render_json_flat, _parse_json,
             ("as a JSON list of rows on one line",
              "as a single-line JSON array of arrays")),
    Encoding("json_rows", _render_json_rows, _parse_json,
             ("as a JSON list of rows, one row per line",
              "as a JSON array of arrays with each row on its own line")),
    Encoding("brackets", _render_brackets, _parse_brackets,
             ("as bracketed rows of space-separated digits, one row per line",
              "one row per line, each row in square brackets with its digits separated by spaces")),
    Encoding("names", _render_names, _parse_names,
             ("as rows of colour names separated by spaces, one row per line",
              "one line per row, each cell written as its colour name"),
             uses_names=True),
    Encoding("sparse", _render_sparse, _parse_sparse,
             ("as a size line with a fill colour and the list of cells that differ from it",),
             reveals_coords=True, reveals_dims=True, reveals_background=True),
    Encoding("pipes", _render_pipes, _parse_pipes,
             ("as rows of digits between vertical bars, one row per line",
              "one line per row, cells separated by | characters, with a | at both ends of each row")),
    Encoding("indexed_table", _render_indexed_table, _parse_indexed_table,
             ("as a table with row labels r0, r1, ... and column labels c0, c1, ...",),
             reveals_coords=True, reveals_dims=True),
    Encoding("stage2", _render_stage2, _parse_stage2,
             ("with row labels r0, r1, ... and a listing of the runs in each row (color@columns)",
              "as labelled digit rows followed by each row's runs of one color"),
             reveals_coords=True, reveals_dims=True),
    Encoding("runs", _render_runs, _parse_runs,
             ("as each row's runs of one color, written color@columns (r0: 0@c0-c3 5@c4 ...)",
              "one line per row listing its runs of equal color from left to right"),
             reveals_coords=True, reveals_dims=True),
    Encoding("row_labelled", _render_row_labelled, _parse_row_labelled,
             ("as digit rows, each prefixed with its row number",),
             reveals_coords=True, reveals_dims=True),
)}


def render(grid: Grid, encoding: str) -> str:
    return ENCODINGS[encoding].render(grid)


def parse(text: str, encoding: str) -> Grid:
    """Read a grid back; only the canonical rendering is accepted (re-rendering must reproduce the
    text exactly), so stray blank lines or spacing that a lenient reader would forgive -- and that
    would break the blank-line separation of grids in a gallery -- raise instead."""
    grid = ENCODINGS[encoding].parse(text)
    if ENCODINGS[encoding].render(grid) != text.strip("\n"):
        raise ValueError(f"not the canonical {encoding} rendering")
    return grid


SPARSE_MAX_SHARE = 0.20   # the sparse listing is offered only when few cells differ from the fill


def suits(grid: Grid, encoding: str) -> bool:
    """Whether the encoding is sensible for this grid (the sparse listing needs a sparse grid)."""
    if encoding != "sparse":
        return True
    cells = [v for row in check_grid(grid) for v in row]
    fill_count = Counter(cells).most_common(1)[0][1]
    return (len(cells) - fill_count) <= SPARSE_MAX_SHARE * len(cells)


def colour_word(value: int, encoding: str) -> str:
    """How a colour is named in prose next to a grid in this encoding."""
    if not 0 <= int(value) <= 9:
        raise ValueError(value)
    return COLOR_NAMES[int(value)] if ENCODINGS[encoding].uses_names else f"color {int(value)}"


# Default encoding weights (DESIGN.md 13.8): compact and the stage-2 rendering dominate; the rest
# share the remainder.  Config may override.  ``sparse`` is offered only when it is shorter than
# compact on the grid at hand.
DEFAULT_WEIGHTS = {
    "compact": 0.42, "stage2": 0.18, "spaced": 0.07, "comma": 0.06, "json_flat": 0.04,
    "json_rows": 0.04, "names": 0.04, "brackets": 0.03, "pipes": 0.03, "runs": 0.03,
    "row_labelled": 0.02, "indexed_table": 0.01, "sparse": 0.02,
}


def allowed(grid: Grid, name: str, *, names_ok: bool, coords_ok: bool, dims_ok: bool,
            background_ok: bool) -> bool:
    e = ENCODINGS[name]
    if e.uses_names and not names_ok:
        return False
    if e.reveals_coords and not coords_ok:
        return False
    if e.reveals_dims and not dims_ok:
        return False
    if e.reveals_background and not background_ok:
        return False
    if name == "sparse" and (not suits(grid, "sparse") or len(render(grid, "sparse")) >= len(render(grid, "compact"))):
        return False
    return True


def choose(rng, grid: Grid, *, names_ok: bool = False, coords_ok: bool = True, dims_ok: bool = True,
           background_ok: bool = False, weights: dict | None = None) -> str:
    """Draw an encoding for one grid among those its questions allow (weighted)."""
    weights = DEFAULT_WEIGHTS if weights is None else weights
    names = [n for n, w in weights.items() if w > 0 and allowed(grid, n, names_ok=names_ok, coords_ok=coords_ok,
                                                             dims_ok=dims_ok, background_ok=background_ok)]
    if not names:
        return "compact"
    p = [weights[n] for n in names]
    total = sum(p)
    return names[int(rng.choice(len(names), p=[x / total for x in p]))]
