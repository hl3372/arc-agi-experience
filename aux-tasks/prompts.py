"""Prompt templates for each auxiliary task.

Each function takes metadata dict and returns the prompt string
that would be shown to the model alongside the input grid.
"""
from __future__ import annotations
import re
from typing import Dict, Any, Optional

from symbol_remap import (
    COLOR_PALETTE_META_KEY,
    render_color_label,
    render_pattern_grid,
)


def _m(meta: dict, key: str, default: str = "") -> str:
    """Get metadata value as string."""
    v = meta.get(key, default)
    return str(v) if v is not None else default


def _ci(meta: dict, key: str) -> str:
    """Get color as a symbolic label. Handles both name strings and ints."""
    v = meta.get(key, "")
    return render_color_label(v, meta.get(COLOR_PALETTE_META_KEY))


def _m_any(meta: dict, *keys: str, default: str = "") -> str:
    """Get the first present metadata value as a string."""
    for key in keys:
        if key in meta and meta[key] is not None:
            return str(meta[key])
    return default


def _ci_list(meta: dict, key: str) -> list:
    """Render every colour of a list-valued metadata key through the palette."""
    palette = meta.get(COLOR_PALETTE_META_KEY)
    return [render_color_label(v, palette) for v in (meta.get(key) or [])]


def _join_and(items: list) -> str:
    """'a', 'a and b', 'a, b and c'."""
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


# ---------------------------------------------------------------------------
# Shared background texture wording (common/backgrounds.py policy)
# ---------------------------------------------------------------------------

# Object rule fragment for tasks that use sparse (inferred) textures: the
# solver must recognise isolated cells as background instead of being told the
# texture colour.
ISOLATED_CELL_RULE = "isolated single cells, whatever their color, are background texture and are ignored"

# House background sentence.  The background may be a solid colour or a texture and neither is
# ever named or defined, so every object rule opens with this instead of a bare "Ignoring the
# background," clause (2026-09-17 sweep: the weak opener let a literal reader take the
# isolated-cell clause as the only texture the prompt admits).
_BG_IGNORE = "Ignore the background, whether it is a solid color or a pattern."


def background_sentence(meta: dict) -> str:
    """Name the texture colour(s) of a solid / structured background texture.

    Reads ``texture_group``, ``texture_colors`` and ``texture_named`` (as written
    by ``common.backgrounds.apply_background`` consumers).  Returns "" when
    there is no texture or when the texture is sparse (inferred through
    ``ISOLATED_CELL_RULE``); otherwise "cells of color Y are background
    texture, not objects" / "cells of colors Y and Z are background texture,
    not objects", every colour rendered through the palette.
    """
    colors = meta.get("texture_colors") or []
    if not colors:
        return ""
    named = meta.get("texture_named")
    if named is None:
        named = meta.get("texture_group") not in (None, "sparse")
    if not named:
        return ""
    labels = _ci_list(meta, "texture_colors")
    noun = "color" if len(labels) == 1 else "colors"
    return f"cells of {noun} {_join_and(labels)} are background texture, not objects"


def _ci_any(meta: dict, *keys: str, default: str = "") -> str:
    """Get the first present color metadata value as a color index string."""
    for key in keys:
        if key in meta and meta[key] is not None:
            return _ci(meta, key)
    return default


def _template_desc(meta: dict) -> str:
    """Describe a template from offset metadata inside its minimal bounding box."""
    explicit = _m_any(
        meta,
        "template_description",
        "template_desc",
        "template_shape_desc",
        "query_shape_desc",
        "query_shape_pattern_desc",
    )
    if explicit:
        lowered = explicit.lower()
        if lowered.startswith("a color ") or lowered.startswith("the color "):
            return explicit
        color = _ci(meta, "template_color")
        return f"the color {color} {explicit}" if color else explicit
    shape = meta.get("template_shape", [])
    if not shape:
        return "the template described in the metadata"
    cells = [tuple(cell) for cell in shape]
    max_r = max(r for r, _ in cells)
    max_c = max(c for _, c in cells)
    cell_set = set(cells)
    rows = []
    for r in range(max_r + 1):
        rows.append(" ".join("1" if (r, c) in cell_set else "0" for c in range(max_c + 1)))
    pattern = "//".join(rows)
    return f"the color {_ci(meta, 'template_color')} shape with pattern {pattern}"


def _symbol_mode(meta: dict) -> bool:
    return meta.get(COLOR_PALETTE_META_KEY) is not None


def _color_answer_sentence(meta: dict) -> str:
    """House answer-format sentence for the tasks whose answer is one colour (25/57/75).

    Under a symbol palette the answer is one of the palette's symbols, otherwise a colour
    label; the wording follows the palette instead of being written out in one of the two
    forms (2026-09-17 sweep)."""
    if _symbol_mode(meta):
        return "Answer as a single color symbol."
    return "Answer as a single color label."


def _rows(n) -> str:
    """'1 row' / 'k rows'."""
    n = int(n)
    return f"{n} row" if n == 1 else f"{n} rows"


def _cols(n) -> str:
    """'1 column' / 'k columns'."""
    n = int(n)
    return f"{n} column" if n == 1 else f"{n} columns"


def _relative_side(direction: str) -> str:
    """Render a direction token into a natural relative-position phrase."""
    mapping = {
        "left": "to the left of",
        "right": "to the right of",
        "above": "above",
        "below": "below",
        "northwest": "to the northwest of",
        "northeast": "to the northeast of",
        "southwest": "to the southwest of",
        "southeast": "to the southeast of",
        "up-left": "up-left of",
        "up-right": "up-right of",
        "down-left": "down-left of",
        "down-right": "down-right of",
    }
    return mapping.get(direction, direction)


def _pattern_literal(meta: dict, *keys: str) -> str:
    """Render a small grid metadata field in compact // notation."""
    palette = meta.get(COLOR_PALETTE_META_KEY)
    for key in keys:
        value = meta.get(key)
        if not isinstance(value, list) or not value or not isinstance(value[0], list):
            continue
        try:
            return render_pattern_grid(value, color_palette=palette, row_sep="//", compact=True)
        except Exception:
            continue
    return ""


def _normalize_pattern_text(pattern: str) -> str:
    """Normalize legacy single-slash row separators to // notation."""
    if "//" in pattern:
        return pattern
    return pattern.replace("/", "//")


_SHAPE_CATEGORY_PLURALS = {
    "rectangle": "solid rectangles (squares and 1-cell-wide bars count)",
    "non_square_rectangle": "solid non-square rectangles (1-cell-wide bars count, squares do not)",
    "square": "solid squares",
    "line": "lines (straight horizontal or vertical bars 1 cell wide, at least 2 cells long)",
    "L": "L-shapes",
    "plus": "plus signs",
    "X": "X-shapes",
    "frame": "hollow rectangles",
}


def _shape_query_phrase(meta: dict) -> str:
    """Describe the queried shape category for tasks 4/5.

    The generator writes the full self-contained definition into
    ``query_shape_desc``; the fallback below only covers metadata without it.
    Pattern literals use ``#`` = cell and ``.`` = empty (never color digits).
    """
    explicit = _m_any(meta, "query_shape_desc", "query_shape_pattern_desc")
    if explicit:
        return explicit
    query_shape = _m(meta, "query_shape")
    if query_shape.startswith("pattern:"):
        pattern = _normalize_pattern_text(query_shape.split(':', 1)[1])
        return (
            f"shaped exactly like the pattern {pattern} (# = object cell, . = empty, rows "
            "separated by //, and rotated or mirrored copies count)"
        )
    return _SHAPE_CATEGORY_PLURALS.get(query_shape, f"{query_shape}s")


def _shape_size_filter_note(meta: dict) -> str:
    """Optional bounding-box restriction for tasks 4/5 (``size_filter`` = [a, b])."""
    size_filter = meta.get("size_filter")
    if not size_filter:
        return ""
    a, b = size_filter[0], size_filter[1]
    if a == b:
        return f" Count only those whose bounding box is {a}x{b}."
    return f" Count only those whose bounding box is {a}x{b} (either orientation)."


def _shape_background_sentence(meta: dict) -> str:
    """Tasks 4/5: nothing about the background is named or defined (user policy, 2026-09-03:
    "ignoring the background" only -- neither the background colour nor any texture colour;
    solvers infer what is background from the grid).  Kept as a hook for a task that
    genuinely needs a texture pointed out; always "" here."""
    return ""


_SHAPE_OBJECT_RULE = (
    f"{_BG_IGNORE} Same-colored cells that touch, including diagonally, form one "
    f"object, and every object has at least two cells; {ISOLATED_CELL_RULE}; "
    "objects of different colors are always separate, even where they touch."
)


def _pattern_family_phrase(meta: dict) -> str:
    """Describe the pattern family used by completion/occlusion tasks."""
    family = _m(meta, "pattern_family", "periodic")
    symmetry_type = _m(meta, "symmetry_type", family)
    if family != "periodic":
        family = symmetry_type
    if family == "periodic":
        tile_h = _m(meta, "tile_h")
        tile_w = _m(meta, "tile_w")
        if tile_h and tile_w:
            return (f"a repeating rectangular-tile color pattern with tile size "
                    f"{_rows(tile_h)} by {_cols(tile_w)}")
        return "a repeating rectangular-tile color pattern"
    if family == "horizontal_mirror":
        return "horizontal mirror symmetry: each row matches the row the same distance from the opposite edge"
    if family == "vertical_mirror":
        return "vertical mirror symmetry: each column matches the column the same distance from the opposite edge"
    if family == "both_axis_mirror":
        return "both horizontal and vertical mirror symmetry"
    if family == "rotational_180":
        return "180-degree rotational symmetry: each cell matches the cell opposite it through the grid center"
    if family == "main_diagonal_mirror":
        return "main-diagonal mirror symmetry in a square grid: each cell matches its reflection across the top-left to bottom-right diagonal"
    if family == "anti_diagonal_mirror":
        return "anti-diagonal mirror symmetry in a square grid: each cell matches its reflection across the top-right to bottom-left diagonal"
    if family == "full_square_symmetry":
        return "full square symmetry in a square grid: every cell matches its images under all 8 square symmetries, including 90/180/270-degree rotations, horizontal and vertical mirrors, and both diagonal mirrors"
    return "the stated repeating or symmetric color pattern"


def _binary_layout_phrase(meta: dict) -> str:
    """Describe how the two binary subgrids are arranged."""
    layout = _m(meta, "layout")
    if layout == "vertical":
        return "stacked vertically"
    if layout == "horizontal":
        return "arranged side by side"
    return "separated by a divider"


def _link_rule(conn) -> str:
    """The ONE link rule that applies at this connectivity, for stored wordings ({fn:_link_rule}).

    Replaces the schema "a shared edge at either number, a shared corner at 8 alone", which read as if
    both numbers applied (training-side blind solve, 2026-09-30)."""
    if str(conn) == "4":
        return "a shared edge links two cells, and cells that touch only at a corner are not linked"
    return "a shared edge or a shared corner links two cells"


def _contact_rule(contact_mode) -> str:
    """The one contact rule for task 56's {m:contact_mode} (4 or 8), for stored wordings."""
    if str(contact_mode) == "4":
        return "cells side by side along an edge are in contact, and cells that meet only at a corner are not"
    return "cells side by side along an edge or meeting corner to corner are in contact"


def _neighbor_rule(contact_mode) -> str:
    """Task 56's {m:contact_mode}-neighbors of a cell, for stored wordings ({fn:_neighbor_rule})."""
    if str(contact_mode) == "4":
        return "the cells sharing an edge with it (cells meeting it only at a corner are not neighbors)"
    return "the cells sharing an edge or a corner with it"


def _connectivity_phrase(meta: dict, default: str = "4") -> str:
    """Render a connectivity metadata field into a human-readable phrase."""
    value = _m_any(meta, "connectivity", "object_connectivity", default=default)
    return value if value else default


def get_prompt(
    task_id: str,
    meta: Optional[Dict[str, Any]] = None,
    color_palette=None,
    prompt_variant: Optional[str] = None,
) -> str:
    """Return the prompt string for a given task and its metadata."""
    if prompt_variant is not None:
        return render_prompt_variant_template(
            task_id,
            prompt_variant,
            meta=meta,
            color_palette=color_palette,
        )
    m = dict(meta or {})
    if color_palette is not None:
        m[COLOR_PALETTE_META_KEY] = color_palette
    fn = PROMPT_TEMPLATES.get(task_id)
    if fn is None:
        return f"[No prompt template for task {task_id}]"
    return fn(m)


_VARIANT_TOKEN_RE = re.compile(r"\{([^{}]+)\}")
_LBRACE_SENTINEL = "\u0000LBRACE\u0000"
_RBRACE_SENTINEL = "\u0000RBRACE\u0000"


def _split_variant_args(value: str) -> list[str]:
    return [part.strip() for part in value.split("|") if part.strip()]


def _render_variant_token(token: str, meta: dict) -> str:
    """Render one token from the prompt-variant mini-template language."""
    token = token.strip()
    if not token:
        return ""

    if token == "template_desc":
        return _template_desc(meta)
    if token == "shape_query_phrase":
        return _shape_query_phrase(meta)
    if token == "binary_layout_phrase":
        return _binary_layout_phrase(meta)
    if token == "connectivity_phrase":
        return _connectivity_phrase(meta)
    if token == "symbol_color_order":
        if _symbol_mode(meta):
            return " ".join(str(s) for s in meta[COLOR_PALETTE_META_KEY])
        return "0 1 2 3 4 5 6 7 8 9"

    if token.startswith("m:"):
        return _m(meta, token[2:].strip())
    if token.startswith("raw:"):
        return _m(meta, token[4:].strip())
    if token.startswith("ci:"):
        return _ci(meta, token[3:].strip())
    if token.startswith("m_any:"):
        parts = _split_variant_args(token[6:])
        default = ""
        keys = []
        for part in parts:
            if part.startswith("default="):
                default = part.split("=", 1)[1]
            else:
                keys.append(part)
        return _m_any(meta, *keys, default=default)
    if token.startswith("ci_any:"):
        parts = _split_variant_args(token[7:])
        default = ""
        keys = []
        for part in parts:
            if part.startswith("default="):
                default = part.split("=", 1)[1]
            else:
                keys.append(part)
        return _ci_any(meta, *keys, default=default)
    if token.startswith("relative_side:"):
        return _relative_side(_m(meta, token[14:].strip()))
    if token.startswith("pattern_literal:"):
        return _pattern_literal(meta, *_split_variant_args(token[16:]))

    if ":" in token:
        raise ValueError(f"Unknown prompt-variant token: {token!r}")
    return _m(meta, token)


def render_prompt_variant_template(
    task_id: str,
    template: str,
    meta: Optional[Dict[str, Any]] = None,
    color_palette=None,
) -> str:
    """Render a reviewed prompt-variant template for a task.

    The mini-template language intentionally exposes only the local prompt
    helpers needed by the existing canonical prompts:
      - `{m:key}` or `{key}` for raw metadata
      - `{ci:key}` for color metadata rendered through the active palette
      - `{m_any:key1|key2}` / `{ci_any:key1|key2}`
      - `{template_desc}`, `{shape_query_phrase}`, `{pattern_literal:key1|key2}`
      - `{relative_side:direction}`, `{symbol_color_order}`

    Use `{{` and `}}` for literal braces.
    """
    del task_id  # Kept for a stable public API and future task-specific hooks.
    m = dict(meta or {})
    if color_palette is not None:
        m[COLOR_PALETTE_META_KEY] = color_palette
    text = str(template).replace("{{", _LBRACE_SENTINEL).replace("}}", _RBRACE_SENTINEL)
    text = _VARIANT_TOKEN_RE.sub(lambda match: _render_variant_token(match.group(1), m), text)
    return text.replace(_LBRACE_SENTINEL, "{").replace(_RBRACE_SENTINEL, "}")


# ---------------------------------------------------------------------------
# Prompt template functions
# ---------------------------------------------------------------------------

def _p1(m): return "What are the dimensions of this grid? Answer as HxW."

def _p2(m): return f"List all coordinates of color {_ci(m, 'query_color_index')} cells in row-major order (scanning top to bottom, then left to right) as [(r,c),...]. Rows and columns are 0-indexed from the top-left corner."

_P3_TAIL = " Rows and columns are 0-indexed from the top-left corner. Leave every other cell unchanged."


def _p3(m):
    rows, cols = m.get("placement_rows"), m.get("placement_cols")
    if not rows or len(rows) == 1:
        # single placement (also the legacy metadata shape: target_row / target_col / target_color)
        r = _m(m, "target_row") if "target_row" in m else str(rows[0])
        c = _m(m, "target_col") if "target_col" in m else str(cols[0])
        col = _ci(m, "target_color") if "target_color" in m else _ci_list(m, "placement_colors")[0]
        return f"Place a color {col} cell at position ({r}, {c})." + _P3_TAIL
    labels = _ci_list(m, "placement_colors")
    positions = [f"({r}, {c})" for r, c in zip(rows, cols)]
    if m.get("colour_mode") == "single":
        return f"Place color {labels[0]} cells at positions {_join_and(positions)}." + _P3_TAIL
    entries = [f"{pos} in color {lab}" for pos, lab in zip(positions, labels)]
    return "Place cells at the following positions: " + ", ".join(entries) + "." + _P3_TAIL

def _p4(m): return f"{_shape_background_sentence(m)}{_SHAPE_OBJECT_RULE} How many objects are {_shape_query_phrase(m)}?{_shape_size_filter_note(m)} Answer with the number."

def _p5(m): return f"{_shape_background_sentence(m)}{_SHAPE_OBJECT_RULE} How many objects of color {_ci_any(m, 'query_color', 'query_color_index')} are {_shape_query_phrase(m)}?{_shape_size_filter_note(m)} Answer with the number."

_P5A_ACTION = (
    " Remove every object: replace each of its cells with what the background would show there, "
    "continuing the background's pattern through the removed cells. Every background cell, "
    "including any isolated single cell, stays exactly as it is. Output the whole grid."
)


def _p5a(m): return f"{_SHAPE_OBJECT_RULE}{_P5A_ACTION}"

def _p6(m): return f"Magnify this grid by a factor of {_m(m, 'scale')}. Replace each cell with a block of {_m(m, 'scale')}x{_m(m, 'scale')} cells of the same color."



def _pair_object_rule(m):
    conn = _connectivity_phrase(m, default="4")
    adjacency = "horizontally, vertically, or diagonally" if conn == "8" else "horizontally or vertically"
    corner = "" if conn == "8" else " (cells that touch only at a corner are not connected)"
    # Policy (user, 2026-09-03): nothing about the background is named -- not its colour and
    # not the texture colours; the solver infers what is background from the grid.
    return (
        f"{_BG_IGNORE} Treat each remaining {conn}-connected same-color group of "
        f"at least two cells, connected {adjacency}{corner}, as one object; {ISOLATED_CELL_RULE}; "
        "cells of different colors are always separate, even where they touch."
    )



def _p7(m):
    match_rule = (
        "Two objects match when they have identical shape up to rotation, i.e. one becomes the other after a "
        "rotation by 0, 90, 180 or 270 degrees (mirror images do not count); their colors do not matter."
    )
    recolor = _ci_any(m, "highlight_color", "target_color", "match_color")
    return (
        f"{_pair_object_rule(m)} {match_rule} Recolor every object that matches at least one other object to "
        f"color {recolor}. Leave every other cell unchanged."
    )

def _p8(m):
    match_rule = (
        "Two objects match when they have the same color and identical shape up to rotation, i.e. one becomes "
        "the other after a rotation by 0, 90, 180 or 270 degrees (mirror images do not count); objects of "
        "different colors never match, whatever their shape."
    )
    recolor = _ci_any(m, "highlight_color", "target_color", "match_color")
    return (
        f"{_pair_object_rule(m)} {match_rule} Recolor every object that matches at least one other object to "
        f"color {recolor}. Leave every other cell unchanged."
    )

def _p9(m):
    direction = _m_any(m, "rotation_direction", "direction", default="clockwise")
    divider = "The divider is a single straight line in a color that appears nowhere else in the grid."
    if _m(m, "layout") == "vertical":
        return f"The grid shows a top image and a bottom image separated by a divider. {divider} The bottom image is a rotation of the top image. What {direction} rotation angle maps the top image to the bottom image? Answer as 90, 180, or 270."
    return f"The grid shows a left image and a right image separated by a divider. {divider} The right image is a rotation of the left image. What {direction} rotation angle maps the left image to the right image? Answer as 90, 180, or 270."

def _p10(m):
    # For a 180-degree rotation the direction is meaningless (clockwise and
    # anticlockwise give the same result), so the generator omits the direction
    # key in that case and the prompt drops the word entirely.
    direction = _m(m, 'direction')
    if direction:
        return f"Rotate this grid {_m(m, 'angle')} degrees {direction}."
    return f"Rotate this grid {_m(m, 'angle')} degrees."

def _p11(m):
    axis = _m(m, 'axis')
    if axis == 'horizontal':
        return "Reflect this grid across the horizontal axis (flip up-down)."
    elif axis == 'vertical':
        return "Reflect this grid across the vertical axis (flip left-right)."
    return f"Reflect this grid across the {axis} axis."

def _p12(m): return f"Rotate only the color {_ci(m, 'target_color')} object ({_connectivity_phrase(m)}-connected) by {_m(m, 'angle')} degrees clockwise around the center of its square bounding box; the cells it vacates become background. Leave every other cell unchanged."

_P13_PIECE_RULE = (
    "Each piece is one connected block of a single color: a group of at least two same-colored cells connected "
    "horizontally or vertically (cells that touch only at a corner are not connected). Pieces are separated "
    "from each other by background cells, and "
    f"{ISOLATED_CELL_RULE}."
)


_P13_FIT = (
    "Translate each loose piece (do not rotate or reflect it) so that it seats flush against a jagged side of "
    "the fixed piece it complements, arriving from the left, from the right, from above or from below. "
    "Seating flush means that every cell of the loose piece that faces the fixed piece (every cell with no "
    "cell of the same piece directly beyond it in the direction of approach) has a cell of that fixed piece "
    "directly beyond it, so no gap remains anywhere along the seam; and the seam is jagged: those facing cells "
    "do not all lie in one row when arriving from above or below, or in one column when "
    "arriving from the left or right."
)
_P13_SIDES_WORD = {1: "one of its sides", 2: "two of its sides", 3: "three of its sides", 4: "all four of its sides"}


def _p13(m):
    # Policy: nothing about the background is named or defined (no colour, no texture colour).
    # The wording is side-agnostic: cavities may lie on any side of a fixed piece, and a loose
    # piece may arrive from any of the four directions.
    return (
        f"{_P13_PIECE_RULE} {_p13_anchor_rule(m)} {_P13_FIT} "
        f"{_p13_receive_rule(m)} {_P13_LEFTOVER} {_P13_TAIL}"
    )


# Leftovers (2026-09-07): some loose pieces seat nowhere and some cavities receive nothing.  The
# prompt never says whether a given example has either; this standing sentence covers both.
_P13_LEFTOVER = ("A loose piece that seats nowhere stays exactly where it is, and a cavity that no loose piece "
                 "fits stays open.")
_P13_TAIL = ("A moved piece may not overlap any other piece. The cells a loose piece leaves behind become plain "
             "background; the fixed pieces and every background cell stay unchanged.")


def _p13_anchor_rule(m) -> str:
    """Task 13: which pieces are fixed (by colour or by touching the border)."""
    if _m(m, 'anchor_rule') == 'color':
        return f"The pieces of color {_ci(m, 'anchor_color')} are fixed and never move; every piece of any other color is loose."
    return "The pieces that touch the border of the grid are fixed and never move; every other piece is loose."


def _p13_receive_rule(m) -> str:
    """Task 13: how many loose pieces a fixed piece may receive (host / mixed / one-to-one).  Never
    says that every cavity is filled or that every loose piece is used: some are not."""
    variant = _m(m, 'variant')
    if variant == 'host':
        n_sides = m.get("max_sides", 1)
        n_cav = _m(m, 'n_sites') if 'n_sites' in m else _m(m, 'n_loose')
        return (f"The fixed piece has {n_cav} jagged cavities in total, spread over "
                f"{_P13_SIDES_WORD.get(int(n_sides), 'several of its sides')}, and each cavity receives at most one loose piece.")
    if variant == 'mixed':
        return ("A fixed piece may have several jagged cavities, possibly on different sides, and each cavity "
                "receives at most one loose piece.")
    return "Each fixed piece receives at most one loose piece."

def _p14(m): return "Read the text specification and construct the described lattice grid exactly."

_LINE_RULE = ("Each line is a straight, one-cell-wide horizontal, vertical, or 45-degree diagonal segment, and every line "
              "continues straight through and at least two cells past each point where it meets another line.")
_CROSSING_SHOWN_RULE = ("Where lines cross, the shared cell is shown in one of the crossing lines' colors but still belongs "
                        "to every line passing through it.")
_DISTRACTOR_NOTE = "Ignore isolated non-line distractor objects."
_PAIRWISE_COUNT_RULE = ("Count the line intersections: each place where two lines cross is one intersection, and where three "
                        "or more lines pass through the same cell, count every pair of them separately, so three lines "
                        "through a single cell are three intersections.")

def _p15(m): return f"{_LINE_RULE} {_CROSSING_SHOWN_RULE} {_DISTRACTOR_NOTE} Count the cells that lie on two or more lines. Answer with the number."

def _p16(m): return f"{_LINE_RULE} {_CROSSING_SHOWN_RULE} Paint every cell that lies on two or more lines with color {_ci(m, 'highlight_color')}. {_DISTRACTOR_NOTE} Leave every other cell unchanged."

def _p15a(m): return f"{_LINE_RULE} {_PAIRWISE_COUNT_RULE} {_DISTRACTOR_NOTE} Answer with the number."


def _largest_object_rule(m, conn: str) -> str:
    """Tasks 17-19 object rule.  Policy: nothing about the background is named or
    defined; objects have at least two cells; isolated single cells are background
    texture (sparse textures are inferred, never named)."""
    adjacency = "horizontally, vertically, or diagonally" if conn == "8" else "horizontally or vertically"
    corner = "" if conn == "8" else " (cells that touch only at a corner are not connected)"
    return (
        f"{_BG_IGNORE} Treat each {conn}-connected same-color group of at least two cells, "
        f"connected {adjacency}{corner}, as one object; {ISOLATED_CELL_RULE}; cells of different colors are "
        f"always separate, even where they touch."
    )

def _p17(m):
    conn = _connectivity_phrase(m)
    return (
        f"{_largest_object_rule(m, conn)} Find the object with the largest bounding-box area, i.e. the height "
        f"times the width of the smallest axis-aligned rectangle that encloses it (exactly one object has the "
        f"largest box), and recolor every cell of that object to color {_ci(m, 'target_color')}. "
        f"Leave every other cell unchanged."
    )

def _p18(m):
    # Tasks 18 and 19 were merged on 2026-09-07: the connectivity comes from the example.
    return (
        f"{_largest_object_rule(m, _connectivity_phrase(m))} Find the object with the most cells (exactly one object has the most) "
        f"and recolor every cell of that object to color {_ci(m, 'target_color')}. Leave every other cell unchanged."
    )

def _cells(n):
    """'1 cell' / 'k cells'."""
    n = int(n)
    return f"{n} cell" if n == 1 else f"{n} cells"


def _exact_object_rule(conn: str) -> str:
    """Tasks 20/21 object rule (no isolated-cell clause: n = 1 is a legitimate object size)."""
    if conn == "8":
        return ("Treat each remaining foreground 8-connected same-color component, connected horizontally, vertically, "
                "or diagonally, as one object; cells of different colors are always separate, even where they touch.")
    return ("Treat each remaining foreground 4-connected same-color component, connected horizontally or vertically, "
            "as one object (cells that touch only at a corner are not connected); cells of different colors are always "
            "separate, even where they touch.")

def _p20(m):
    # Tasks 20 and 21 were merged on 2026-09-07: the connectivity comes from the example.
    return f"{_BG_IGNORE} {_exact_object_rule(_connectivity_phrase(m))} Recolor every object with exactly {_cells(_m(m, 'n'))} to color {_ci_any(m, 'target_color', 'target_color_index')}. Leave every other cell unchanged."

def _p22(m):
    obj = _ci(m, "obj_color")
    return (
        f"The object is made of every cell of color {obj}; every cell of any other color is empty, whatever its color. "
        f"Count the empty regions that are completely enclosed by the object. An empty region is enclosed when it cannot "
        f"reach the edge of the grid by moving up, down, left, or right through empty cells; diagonal contact does not "
        f"connect cells. Answer with the number (0 if nothing is enclosed)."
    )

# Task 23 (2026-09-07): the marker colour is named (a per-example given, as on task 24), the
# reflection count is worded so the final wall cell can never be read as a reflection, and
# the edit-task note closes the prompt.
_P23_TIMES = {1: "once", 2: "twice", 3: "three times", 4: "four times", 5: "five times", 6: "six times",
              7: "seven times", 8: "eight times"}
_P23_ORDINAL = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh", 8: "eighth"}
_P23_GLOSS = {"NE": "up-right", "NW": "up-left", "SE": "down-right", "SW": "down-left"}
_P23_OVERLAP = ("The trajectory may overlap itself and is drawn on top of any existing cells except the marked "
                "start cell, which keeps its color.")
_P23_OVERLAP_ZERO = ("The line is drawn on top of any existing cells except the marked start cell, which keeps "
                     "its color.")
# The turn happens ON the wall cell (which is drawn once); a mirror at the outer edge of the
# wall cell would double it, so the sentence says where the reflection takes place.
_P23_ON_WALL = ("each reflection happens on the wall cell the line reaches, which is drawn, and the line turns "
                "there and continues")


def _p23_reflection_rule(n: int) -> str:
    if n == 0:
        return ("The line does not reflect: it runs straight to the wall and stops on the wall cell it reaches, "
                "which is drawn.")
    return (f"The line reflects off the grid walls exactly {_P23_TIMES[n]}: {_P23_ON_WALL}; after the "
            f"{_P23_ORDINAL[n]} reflection it continues straight and stops on the wall cell it reaches, "
            f"which is drawn.")


def _p23(m):
    d = _m(m, 'direction')
    n = int(_m(m, 'n_bounces'))
    overlap = _P23_OVERLAP_ZERO if n == 0 else _P23_OVERLAP
    return (f"Starting from the single cell of color {_ci(m, 'start_color')}, draw a diagonal line in color "
            f"{_ci(m, 'trace_color')} in the {d} ({_P23_GLOSS.get(d, d)}) direction. "
            f"{_p23_reflection_rule(n)} {overlap} Leave every other cell unchanged.")

# Task 24 (2026-09-07): a bounce is defined, so the cell where the trace stops (often a wall
# cell) can never be counted as one.
_P24_BOUNCE_RULE = ("A bounce is a change of direction at a wall cell; the cell where the trace stops is not a "
                    "bounce, even when it lies on a wall.")

def _p24(m): return f"A marked cell of color {_ci(m, 'start_color')} shows where the diagonal trace started. The diagonal trace is the line of color {_ci(m, 'trace_color')} cells together with the marked start cell; cells of any other color are background detail to ignore. {_P24_BOUNCE_RULE} How many times does the trace bounce off the grid walls before it stops? Answer with the number."

def _p25(m):
    return f"Which color is the background? {_color_answer_sentence(m)}"

def _p26(m):
    from generators.crop_frame_26 import FRAME_DEFINITION
    return FRAME_DEFINITION

def _p27(m):
    from generators.packing_27 import prompt27
    return prompt27(m)

def _p28(m):
    from generators.fit_holes_28 import prompt28
    return prompt28(m)

def _p29(m): return f"Replace every color {_ci(m, 'source_color')} cell with color {_ci(m, 'target_color')}. Leave every other cell unchanged."

def _p30(m): return f"How many color {_ci_any(m, 'query_color', 'query_color_index')} cells are in this grid? Answer with the number."

def _p31(m):
    from generators.gravity_31 import prompt31
    return prompt31(m)

def _p31_action(m):
    """Task 31's per-example action sentence, reachable by prompt variants as {fn:_p31_action}."""
    from generators.gravity_31 import prompt31_action
    return prompt31_action(m)

def _p32(m): return f"Cells of color {_ci(m, 'mask_color')} are masked. The complete grid follows {_pattern_family_phrase(m)}. Restore the masked cells and output the full grid."

def _p33(m): return f"A solid rectangle of color {_ci(m, 'occluder_color')} hides part of a grid whose complete pattern follows {_pattern_family_phrase(m)}. Output exactly the hidden rectangle patch."

def _p34(m):
    from generators.mirror_divider_34 import PROMPT34
    return PROMPT34

def _p35(m):
    return (f"How many color {_ci(m, 'query_color')} cells are strictly inside the unique rectangular frame, excluding its border? "
            "The frame is a complete one-cell-thick outline of one color, with horizontal and vertical sides. "
            "Answer with the number.")

def _p36(m):
    obj = _ci(m, "obj_color")
    return (
        f"The shape is made of every cell of color {obj}; every cell of any other color is empty, whatever its color. "
        f"Is the shape closed, meaning at least one empty region is completely enclosed by it, or open, meaning every "
        f"empty cell can reach the edge of the grid? Empty cells connect only up, down, left, and right; diagonal "
        f"contact between empty cells does not connect them, so a wall that continues diagonally still encloses. "
        f"Answer 'closed' or 'open'."
    )


def _p37(m):
    from generators.clean_topology_36_37 import PROMPT37
    return PROMPT37.format(color=_ci(m, "obj_color"))

def _p38(m):
    from generators.mirror_check_38 import PROMPT38
    return PROMPT38

def _p39(m):
    from generators.anchor_translate_39_40 import PROMPT39
    return PROMPT39.format(anchor=_ci(m, 'anchor_color'), angle=_m(m, 'angle'))

def _p40(m):
    from generators.anchor_translate_39_40 import PROMPT40
    return PROMPT40.format(direction=_m(m, 'direction'))

def _p41(m):
    from generators.infer_translation_41 import PROMPT41
    return PROMPT41

def _p42(m):
    if m.get("encoding") == "different":
        return _binary_different_colors(m)
    return (
        f"The input shows two binary subgrids {_binary_layout_phrase(m)}. "
        f"In both subgrids, color {_ci_any(m, 'fg_color')} means true and color "
        f"{_ci_any(m, 'bg_color')} means false. Compute their {_m(m, 'operation').upper()} "
        f"and output the result using the same true/false colors."
    )

def _binary_different_colors(m):
    return (
        f"The input shows two binary subgrids {_binary_layout_phrase(m)}. In the first subgrid, "
        f"color {_ci_any(m, 'fg_color_left')} means true; in the second subgrid, color "
        f"{_ci_any(m, 'fg_color_right')} means true; and color {_ci_any(m, 'bg_color')} "
        f"means false in both. Compute their {_m(m, 'operation').upper()} and output the "
        f"result using color {_ci_any(m, 'output_true_color')} for true cells and color "
        f"{_ci_any(m, 'bg_color')} for false cells."
    )

def _tile_times_phrase(value: str, axis: str) -> str:
    return f"once {axis}" if value == "1" else f"{value} times {axis}"


def _copies(n) -> str:
    n = str(n)
    return f"{n} copy" if n == "1" else f"{n} copies"

def _p44(m):
    # 2026-09-16: "N times" alone left open whether the input counts as one of the N copies.
    rows, cols = _m(m, 'tile_rows'), _m(m, 'tile_cols')
    return (f"Tile this grid {_tile_times_phrase(rows, 'vertically')} and {_tile_times_phrase(cols, 'horizontally')}. "
            f"The output is {_copies(rows)} tall and {_copies(cols)} wide, counting the input as one copy.")

def _p45(m): return "Identify the smallest repeating pattern that tiles this grid exactly, then determine how many copies appear vertically and horizontally. Answer as MxN."

def _isolation_neighbors(m):
    conn = _connectivity_phrase(m)
    directions = "horizontally or vertically" if conn == "4" else "horizontally, vertically, or diagonally"
    return f"{conn}-connected neighbor ({directions})"

def _p46a(m): return f"Ignoring the background, count every cell that has no same-color {_isolation_neighbors(m)}. Answer with the number."

def _p47(m): return f"Ignoring the background, in one simultaneous pass replace every cell that has no same-color {_isolation_neighbors(m)} with the background color. Leave every other cell unchanged."

def _p49(m): return f"Hollow out every solid single-color rectangular object, leaving a {_m_any(m, 'border_thickness', default='1')}-cell-thick border. Cleared cells become the plain background color. A rectangle with no interior beyond that border stays unchanged. Rectangles have horizontal and vertical sides, both at least two cells long; squares count. Leave every other cell unchanged."

def _p50(m): return "Ignoring the background, fill the interiors of all closed rectangular frames with their wall color. Each frame has horizontal and vertical sides and a uniform wall thickness, and its walls form one connected single-color object. Squares count as rectangles. Leave every other cell unchanged."

_FRAME_MEASURE_RULE = "Ignoring the background, consider the closed rectangular frames. Each frame has horizontal and vertical sides, and its walls form one connected single-color object with a uniform thickness within that frame. Squares count as rectangles."

def _p51(m): return f"{_FRAME_MEASURE_RULE} What is the greatest wall thickness, in cells, among the frames? Answer with the number."

def _p52(m):
    what = f"cells of color {_ci(m, 'target_color')}" if m.get('count_mode') == 'color' else "cells, regardless of color,"
    return (f"{_FRAME_MEASURE_RULE} Select the {_m(m, 'extreme_kind')} frame by outer area "
            f"(outer height times outer width); exactly one frame has that extreme area. "
            f"How many {what} lie strictly inside its inner boundary? Do not count wall cells. Answer with the number.")

def _p54(m): return "Ignoring the background, the grid shows a small pattern and an enlarged copy of it, made by replacing every cell of the small pattern with an n x n block of the same color. The two are separated by background cells. What is n? Answer with the number."

def _p55(m):
    n = _m(m, 'scale')
    return (
        f"Demagnify this grid: split it into non-overlapping {n}x{n} blocks starting at the "
        f"top-left corner (every block is filled with a single color) and replace each block "
        f"with one cell of that block's color."
    )

def _p56(m):
    obj = _ci(m, "object_color")
    tgt = _ci_any(m, "target_color", "target_object_color", "target_cell_color")
    what = "target object" if _m(m, "target_kind") == "object" else "target cell"
    if _m(m, "contact_mode") == "8":
        rule = "8-neighbor adjacent (sharing an edge or a corner)"
        test = "share an edge or a corner"
    else:
        rule = "4-neighbor adjacent (sharing an edge)"
        test = "share an edge"
    return (
        f"The grid contains exactly one color {obj} object and exactly one color {tgt} {what}. "
        f"Is the color {obj} object {rule} to the color {tgt} {what}, i.e. does some color {obj} cell "
        f"{test} with some color {tgt} cell? Answer 'yes' or 'no'."
    )

def _p57(m):
    wall = _ci(m, "enclosure_color")
    tail = _color_answer_sentence(m)
    return (
        f"A closed wall of color {wall} encloses exactly one region: the cells that cannot reach the grid border "
        f"by up/down/left/right moves without crossing a color {wall} cell. Among the cells strictly inside that "
        f"wall, ignoring the background, one color occurs more often than any other. {tail}"
    )

def _p58(m):
    clause = _m(m, "orientation_text")
    clause = f", {clause}" if clause else ""
    return (
        f"Find every separate 4-connected group (connected horizontally or vertically; cells that touch only "
        f"at a corner are not connected) of color "
        f"{_ci(m, 'template_color')} cells whose shape is exactly the template {_m(m, 'template_pattern')} "
        f"(rows top to bottom separated by //, # = a cell of the shape, . = a position outside the shape; the group must "
        f"consist of exactly these cells and nothing more){clause}. Recolor every cell of each such group to "
        f"color {_ci(m, 'highlight_color')}. Leave every other cell unchanged."
    )

def _p59(m):
    lead = (
        f"How many separate 4-connected groups (connected horizontally or vertically; cells that touch only "
        f"at a corner are not connected) of color {_ci(m, 'template_color')} cells "
    )
    if _m(m, "query_mode") == "generic":
        return (
            lead + f"form {_m(m, 'family_description')}, of any size and orientation? "
            "Count only whole groups whose cells form exactly that shape, with no extra cells attached. "
            "Answer with the number."
        )
    clause = _m(m, "orientation_text")
    return (
        lead + f"have exactly the template {_m(m, 'template_pattern')} "
        "(rows top to bottom separated by //, # = a cell of the shape, . = a position outside the shape; "
        "the group must consist of exactly these cells and nothing more)"
        + (f", {clause}" if clause else "") + "? Answer with the number."
    )

def _p60(m):
    count = _m(m, "n")
    cells = "cell" if count == "1" else "cells"
    return f"The path is the single one-cell-wide 4-connected line of color {_ci(m, 'path_color')} cells (connected horizontally or vertically; cells that touch only at a corner are not connected); cells of any other color are background detail to ignore. Return the last {count} {cells} from both path ends. List each run from the endpoint inward; the two runs do not overlap. Put first the run whose endpoint comes first in row-major order (smaller row, then smaller column), the other run second, and give each cell as a (row, col) pair with 0-indexed row and column. Format the answer exactly as first:[(row,col),(row,col),...], second:[(row,col),(row,col),...]."

def _p61(m):
    count = _m(m, "n")
    cells = "cell" if count == "1" else "cells"
    return (
        f"The path is the single one-cell-wide 4-connected line of color {_ci(m, 'path_color')} cells "
        "(connected horizontally or vertically; cells that touch only at a corner are not connected); ignore cells of every other color. "
        f"Starting at each endpoint and following the path inward, recolor the first {count} {cells} "
        f"with color {_ci(m, 'highlight_color')}. The two runs do not overlap. Leave every other cell unchanged."
    )


def _p62(m):
    count = _m(m, "n")
    cells = "cell" if count == "1" else "cells"
    return (
        f"The path is the one-cell-wide 4-connected line of color {_ci(m, 'path_color')} cells "
        f"together with the single start cell of color {_ci(m, 'start_color')} at one endpoint "
        "(connected horizontally or vertically; cells that touch only at a corner are not connected); ignore cells of every other color. "
        f"Starting at that marked endpoint, traverse the path and return the coordinates of the last {count} {cells} "
        "at the far end, listed in traversal order toward the far endpoint. The start cell is not included. "
        "Rows and columns are 0-indexed from the top-left corner. "
        "Format the answer exactly as [(row,col),(row,col),...]."
    )


def _p63(m):
    count = _m(m, "n")
    cells = "cell" if count == "1" else "cells"
    return (
        f"The path is the one-cell-wide 4-connected line of color {_ci(m, 'path_color')} cells "
        f"together with the single start cell of color {_ci(m, 'start_color')} at one endpoint "
        "(connected horizontally or vertically; cells that touch only at a corner are not connected); ignore cells of every other color. "
        f"Starting at that marked endpoint, traverse the path and recolor the last {count} {cells} "
        f"at the far end with color {_ci(m, 'highlight_color')}. The start cell is not included. "
        "Leave every other cell unchanged."
    )


def _p64(m): return f"Extract the single edge-connected foreground object from the patterned background. The object may have several colors; none of its colors is used in the background. The object does not span the full height or the full width of the grid. Preserve the object's colors and position on a canvas of the same size, and fill every other output cell with color {_ci(m, 'canvas_color')}."

def _p65(m): return "Find the smallest rectangular tile whose unchanged copies, placed side by side and above one another starting at the input's top-left corner, reproduce the whole input. Copies may be cut off at the bottom or right edge. Output that top-left tile as a grid."

def _p66(m):
    # Branch on the canonical short key ('main'/'anti'); the fallback handles any
    # unexpected value. Previously this read diagonal_desc first, so the long
    # description always won and the named branches were dead code.
    diagonal = _m_any(m, 'diagonal', 'diagonal_desc')
    if diagonal == "main":
        return "Reflect this grid across its main diagonal (the top-left to bottom-right diagonal of its bounding box). A height-by-width grid becomes width-by-height."
    if diagonal == "anti":
        return "Reflect this grid across its anti-diagonal (the top-right to bottom-left diagonal of its bounding box). A height-by-width grid becomes width-by-height."
    return f"Reflect this grid across the {diagonal} of its bounding box."

def _p67(m): return "Reverse the layering order inside each overlapping rectangle group. Each rectangle has its own color. Leave non-overlapping rectangles unchanged."

def _object_rule_68(m) -> str:
    """Tasks 68a/68b object rule.  The singleton rule is drawn per example: either
    a lone cell is an object, or it is background texture (``ISOLATED_CELL_RULE``).
    The background sentence in front of this one is byte-identical on plain and
    patterned grids -- nothing about the background is named or defined.

    Both variants are load-bearing and neither is a default: metadata that does
    not say which rule and which connectivity the example was drawn under has no
    prompt, and asking for one raises rather than quietly counting lone cells."""
    conn = _m_any(m, "connectivity", "object_connectivity")
    if conn not in ("4", "8"):
        raise ValueError("Tasks 68a/68b state the connectivity of the example: 4 or 8")
    rule = _m_any(m, "singleton_rule")
    if rule not in ("count", "ignore"):
        raise ValueError("Tasks 68a/68b state the singleton rule of the example: "
                         "'count' or 'ignore'")
    adjacency = "horizontally or vertically" if conn == "4" else "horizontally, vertically, or diagonally"
    corner = " (cells that touch only at a corner are not connected)" if conn == "4" else ""
    if rule == "ignore":
        return (f"An object is a maximal group of at least two same-color cells connected "
                f"{adjacency}{corner}; {ISOLATED_CELL_RULE}; cells of different colors are "
                f"separate objects even when they touch.")
    return (f"An object is a maximal group of same-color cells connected {adjacency}{corner}. "
            f"A single cell also counts as an object; cells of different colors are separate "
            f"objects even when they touch.")

def _p68a(m):
    return (f"{_BG_IGNORE} How many {_connectivity_phrase(m)}-connected objects are in this grid? "
            f"{_object_rule_68(m)} Answer with the number.")

def _p68b(m):
    return (f"{_BG_IGNORE} How many {_connectivity_phrase(m)}-connected objects are entirely inside "
            f"the marked rectangle? {_object_rule_68(m)} Do not count the rectangle border. Count an "
            f"object only if all its cells are inside the border; it may be adjacent to the border. "
            f"Answer with the number.")



def _p70(m): return f"Swap every color {_ci(m, 'color_a')} cell with color {_ci(m, 'color_b')} and vice versa. Leave every other cell unchanged."

def _p71(m):
    return (f"Draw a filled diamond in color {_ci(m, 'diamond_color')} centered at EACH cell of color {_ci(m, 'marker_color')}; "
            f"every diamond has the same inclusive top-corner to bottom-corner span of {_m_any(m, 'tip_to_tip_span', 'length')} cells. "
            "The diamonds' sides follow the grid diagonals. Clip the diamonds at the grid edges. "
            "Paint every cell inside any diamond, including its center and boundary; "
            "overlapping diamonds form their combined filled area. Preserve every input cell outside the diamonds.")

def _p72(m):
    return (f"{_BG_IGNORE} Identify the diamond. Its sides follow the grid diagonals and its center is a cell. "
            "It is either filled or drawn as a one-cell-thick outline. "
            "What is its full outer top-corner to bottom-corner span, counting both corner cells? "
            "If it is clipped by the grid, infer the full span. Answer with the number.")

def _p73(m):
    return (f"{_BG_IGNORE} What is the gap between the two parallel one-cell-wide lines or line segments? "
            "For horizontal or vertical lines, count intervening rows or columns. "
            "For diagonal lines, count intervening parallel diagonal bands. "
            "Measure the gap between the lines' supporting rows, columns or diagonal bands, regardless of their endpoints. "
            "Answer with the number.")

def _p74(m):
    direction = _relative_side(_m(m, 'direction'))
    return (f"{_BG_IGNORE} Draw a new color {_ci(m, 'new_color')} line parallel to the existing one-cell-wide line, "
            f"with a gap of {_m(m, 'distance')} {direction} the existing line. "
            "Draw the entire new row, column or diagonal band, clipped at the grid edges. "
            "For horizontal or vertical lines, the gap counts intervening rows or columns; "
            "for diagonal lines, it counts intervening parallel diagonal bands. "
            "Paint every cell along the new line. Leave every other cell unchanged.")

def _p75(m):
    return f"What color is the center cell of this grid? {_color_answer_sentence(m)}"

# task 76 answers in color NAMES over digit grids: the prompt gives the names (parity review 2026-09-30:
# nothing said names rather than digits, in the old and the new packing alike).  Under a symbol palette
# the answer uses the grid's own symbols, so the names are not given.
COLOR_NAME_KEY = ("Write each color as its name: 0 black, 1 blue, 2 red, 3 green, 4 yellow, 5 grey, 6 magenta, "
                  "7 orange, 8 cyan, 9 maroon.")


def _color_name_key(meta: dict) -> str:
    """The names key plus a trailing space, or nothing under a symbol palette (stored wordings put
    ``{fn:_color_name_key}`` right before "Format:")."""
    return "" if _symbol_mode(meta) else COLOR_NAME_KEY + " "


def _p76(m):
    return ("Count the cells of each color present; each present color has a different count, so no tie needs to "
            "be resolved. Sort from least frequent to most frequent. " + _color_name_key(m)
            + "Format: color:count, color:count, ...")

def _p78(m):
    return (
        "The input contains a stamp grid and a canvas grid. "
        f"The canvas cells of color {_ci(m, 'marker_color')} are markers. "
        "Place one copy of the stamp at each marker so the center cell of the stamp grid lands on that marker. "
        "Copy every stamp cell, including its background-colored cells: the whole stamp is opaque and "
        "overwrites every covered canvas cell. Clip parts that fall outside the canvas. "
        "Leave all canvas cells outside the stamped footprints unchanged."
    )

def _p80(m): return f"Return the {_m(m, 'which_half')} half of this grid."

def _p81(m):
    connectivity = _connectivity_phrase(m)
    return (
        "The cells of the same color as the uniform outer frame are walls. "
        f"Treat each {connectivity}-connected group of the other color as a compartment: "
        f"compartment cells are {_adjacency_phrase(connectivity)}. "
        f"Color the largest compartment with color {_ci(m, 'color_large')} and the smallest with color {_ci(m, 'color_small')}. "
        "Measure compartment size by the number of cells. If several compartments tie for largest or smallest, "
        "choose the one whose first cell comes earliest in row-major order (scanning top to bottom, then left to right). "
        "Leave every other cell unchanged."
    )

def _steps(n) -> str:
    """'1 horizontal, vertical, or diagonal step' / 'N horizontal, vertical, or diagonal steps' (82)."""
    n = str(n)
    return f"{n} horizontal, vertical, or diagonal step" + ("" if n == "1" else "s")

def _p82(m):
    thickness = _m(m, 'outline_thickness')
    return (
        f"Draw a {thickness}-cell-thick external outline in color {_ci(m, 'outline_color')} "
        "around every object, including diagonal corner cells (an 8-connected band). "
        f"Paint each background cell reachable from an original object cell in at most {_steps(thickness)}, "
        "including background cells in holes and gaps. "
        "Clip the outline at the grid edge; overlapping bands use the same color. "
        "Leave the original objects unchanged. Leave every other cell unchanged."
    )

def _p83(m):
    contact = "a shared edge" if str(_m(m, "connectivity", 4)) == "4" else "a shared edge or corner"
    return (
        # The house sentence: the same wording on a plain ground and on the sparse
        # textures generators/interior_83.py paints, so the prompt never varies with
        # the picture.  A texture dot is a one-cell group with no interior cell, so
        # both readings of it (background, or an object the rule leaves alone) give
        # the same output.
        f"{_BG_IGNORE} Treat each same-color group connected through shared edges as one object. "
        f"Fill its interior cells with color {_ci(m, 'fill_color')}. "
        f"A cell is interior if it does not touch anything outside its object through {contact}; "
        "the area beyond the grid also counts as outside. "
        "Determine interiors from the original input. Leave every other cell unchanged."
    )

def _p84(m):
    return (
        "Ignoring the background, treat each 4-connected same-color component, connected horizontally or "
        "vertically (cells that touch only at a corner are not connected), as one object; cells of different "
        "colors are separate even where they touch. "
        "Two objects match if their shapes are identical after a rotation by 0, 90, 180 or 270 degrees; "
        "their colors do not matter, and mirror images do not count. Remove every object that matches "
        "at least one other object, including every copy in a matching group. Removed cells become "
        "background. Leave every other cell unchanged."
    )

_P85_SIZE_PHRASE = {"size=": "exactly", "size<": "fewer than", "size>": "more than"}


def _p85_filter_phrase(m) -> str:
    """Task 85's removal criterion as a prose phrase, one per filter kind: "of color X" /
    "with fewer than k cells" / "with exactly k cells" / "with more than k cells" (2026-09-17
    sweep: every other prompt in 1-102 states its criterion in prose, not as a filter
    expression).  Stored wordings reach it as {fn:_p85_filter_phrase}."""
    kind = _m_any(m, "filter_kind", "filter_description")
    value = _m(m, "filter_value")
    if kind != "color" and kind not in _P85_SIZE_PHRASE:
        # legacy metadata: the description carries the comparison ("size<13")
        kind, value = kind[:5], kind[5:]
    if kind == "color":
        return f"of color {_ci(m, 'filter_color')}"
    phrase = _P85_SIZE_PHRASE.get(kind)
    if phrase is None:
        raise ValueError(f"task 85: unknown filter kind {kind!r}")
    return f"with {phrase} {_cells(value)}"


def _p85_filter_sentence(m) -> str:
    """Task 85's removal instruction."""
    return f"Remove every object {_p85_filter_phrase(m)}."


def _p85(m):
    return (
        "Ignoring the background, treat each 4-connected same-color component, connected horizontally or "
        "vertically (cells that touch only at a corner are not connected), as one object; cells of different "
        "colors are separate even where they touch. "
        f"{_p85_filter_sentence(m)} Removed cells become background. Leave every other cell unchanged."
    )


# --- tasks 87-89: one lattice family (generators/lattice_blocks_87_89.py) ----
# Complete rows and complete columns of one color are the lattice lines; the blocks are
# the rectangular areas between them, clipped edge blocks included.  The lines have a
# phase per axis and are periodic only most of the time, so a block can be cut short on
# ANY side (top and left included) and nothing may be read off cell (0, 0): "a block
# that the edge of the grid cuts short still counts as a block" covers all four sides
# and the sentence promises no periodicity.  The background and any texture inside the
# blocks are never named or defined.
_LATTICE_BLOCKS = (
    "Some complete rows and complete columns of this grid are filled with one single color; "
    "these lattice lines divide the rest of the grid into rectangular blocks. The lattice lines "
    "are not part of any block, and a block that the edge of the grid cuts short still counts as "
    "a block."
)
_LATTICE_INDEX_NOTE = (
    "Block rows are counted from the top and block columns from the left, both 0-indexed, so the "
    "block in the top-left corner is block (0, 0)."
)

def _p87(m): return (
    f"{_LATTICE_BLOCKS} {_LATTICE_INDEX_NOTE} Paint every cell of block "
    f"({_m(m, 'target_row')}, {_m(m, 'target_col')}) with color {_ci(m, 'paint_color')}, where the "
    f"first index is the block row and the second is the block column. Leave every other cell unchanged."
)

def _p88(m): return (
    f"{_LATTICE_BLOCKS} {_BG_IGNORE} A marked block is a block that is completely filled with one "
    "single color other than the background. For each color separately: whenever two or more marked "
    "blocks of that color sit in the same row of blocks, fill every block strictly between them with "
    "that same color, and whenever two or more sit in the same column of blocks, fill every block "
    "strictly between them with that same color. A marked block that shares no block row and no block "
    "column with another marked block of its color changes nothing, and neither do two marked blocks "
    "that are already neighbors. Leave every other cell unchanged."
)

def _p89(m): return (
    f"{_LATTICE_BLOCKS} How many blocks are completely filled with color {_ci(m, 'target_color')}? "
    "A block counts only when every one of its cells is that color. Answer with the number."
)

_P90_QUESTION = (
    "How many objects have at least one cell on the outer border of the grid? "
    "Answer with the number."
)

def _p90(m):
    # The connectivity is drawn per example (generators/border_objects_90.py).  The
    # standard object rule carries the isolated-cell clause, so sparse background
    # texture on the border is inferred, and the background itself is never named.
    return f"{_largest_object_rule(m, _connectivity_phrase(m, default='4'))} {_P90_QUESTION}"

_P91_PACK_RULE = (
    "Pack every object against the {edge} edge. Each object keeps its shape and slides "
    "straight in that direction as far as possible without leaving the grid or overlapping "
    "another object. Objects closest to the {edge} edge settle first; later objects come to "
    "rest against objects that have already settled (gravity stacking). The cells an object "
    "leaves behind become background."
)

def _p91(m):
    # Rebuilt on 2026-09-15 (generators/pack_edge_91.py).  The standard object rule
    # replaces the bare "rigid 4-connected object": it says same-color, it says
    # "ignoring the background" without naming it, and its isolated-cell clause
    # disposes of the sparse texture, which stays where it is.  The settle-order
    # sentence is kept verbatim -- it decides the answer on most grids.
    edge = _m(m, 'direction')
    return (
        f"{_largest_object_rule(m, '4')} {_P91_PACK_RULE.format(edge=edge)} "
        f"Leave every other cell unchanged."
    )

_P92_QUESTION = (
    "How many objects have at least one cell that is {adjacency} to a cell of a different "
    "object? Count objects, not touching pairs. Answer with the number."
)

def _p92(m):
    # Tasks 92 and 92b were merged on 2026-09-15: the connectivity comes from the
    # example (generators/touching_92.py) and is drawn after the scene.  The standard
    # object rule supplies "ignoring the background", the same-color clause and the
    # isolated-cell clause, none of which the old one-sentence prompt had.
    conn = _connectivity_phrase(m, default="4")
    adjacency = "side-or-corner-adjacent" if conn == "8" else "side-adjacent"
    return f"{_largest_object_rule(m, conn)} {_P92_QUESTION.format(adjacency=adjacency)}"

def _adjacency_phrase(conn: str) -> str:
    """The standard adjacency wording (tasks 81, 93/94)."""
    if conn == "8":
        return "connected horizontally, vertically, or diagonally"
    return "connected horizontally or vertically (cells that touch only at a corner are not connected)"

def _p93(m):
    # The connectivity is drawn per example; a corner-joined target is the case
    # where the clause changes the crop.  The background is never mentioned:
    # the object is defined by its colour, so nothing has to be ignored.
    conn = _connectivity_phrase(m, default="4")
    color = _ci(m, 'target_color')
    return (
        f"Tight-crop the unique color {color} object using {conn}-connectivity: connected cells of color {color} "
        f"form one object, {_adjacency_phrase(conn)}, and exactly one such object is present. "
        f"Output the smallest rectangle containing all cells in that connected object, copying every input cell "
        f"in that rectangle exactly as it appears, including any non-object cells inside the box."
    )

def _p94(m):
    # The connectivity is drawn per example, after the whole scene exists.
    # Policy: the background is never named or described, so the standing
    # "Ignore the background" sentence is the whole of it; a lone color X cell
    # is an object, which is why there is no isolated-cell rule here.
    conn = _connectivity_phrase(m, default="4")
    color = _ci(m, 'target_color')
    box_rule = _m(m, 'box_rule', 'smallest rectangle enclosing all cells of that object')
    ordering = _m(m, 'ordering', 'by top row ascending, then by left column ascending')
    return (
        f"Find every color {color} object using {conn}-connectivity. {_BG_IGNORE} "
        f"Only connected cells of color {color} form objects, {_adjacency_phrase(conn)}; "
        f"a single cell of color {color} on its own is an object; cells of any other color are not objects. "
        f"Each bounding box is the {box_rule}. "
        f"Return one box per object as [(top_row,left_column,bottom_row,right_column),...], where the bottom row "
        f"and right column are inclusive 0-indexed coordinates, ordering the boxes {ordering}. "
        "Rows and columns are 0-indexed from the top-left corner."
    )

_P95_OUTLINE_RULE = (
    "The tight bounding box of an object is the smallest rectangle that contains all of its cells. "
    "Every object is found in the input grid, before any outline is drawn: draw the one-cell outline "
    "just outside each object's tight bounding box (the ring of cells surrounding the box, corner cells included) in color {h}, writing color {h} onto those cells "
    "over whatever is there, another object's cells included, and dropping any outline cell that "
    "falls outside the grid."
)

def _p95(m):
    # Rebuilt on 2026-09-15 (generators/bbox_outline_95.py).  The connectivity is
    # drawn per example, after the scene, and decides how many rings a corner-joined
    # target group gets.  The standard object rule replaces the old sentence, which
    # described the background ("patterned background cells, if present") against
    # policy; its isolated-cell clause is load-bearing here because lone target-colour
    # cells really occur.  The three ring policies (clip at the border, write over
    # whatever is there, leave everything else) are stated because each of them
    # decides cells on a real share of the grids.
    conn = _connectivity_phrase(m, default="4")
    outline = _ci(m, 'outline_color')
    return (
        f"{_largest_object_rule(m, conn)} Find every color {_ci(m, 'target_color')} object. "
        f"{_P95_OUTLINE_RULE.format(h=outline)} Leave every other cell unchanged."
    )

_P96_FRAME_RULE = (
    "A rectangular frame of color {f} is a complete one-cell-thick rectangular ring of color {f}, "
    "at least 3 rows by 3 columns, whose interior contains no cell of color {f}."
)

_P96_ROW = (
    "Arrange the interiors in that order side by side, from left to right, separated by a gap of "
    "{gap} {unit} of color {s}. The output is exactly as tall as the tallest interior and exactly as wide as "
    "the interiors and the gaps between them together, with no gap before the first interior or after the last; each interior "
    "sits with its top row on the output's top row, and every cell no interior covers is color {s}."
)

_P96_COLUMN = (
    "Arrange the interiors in that order one below another, from top to bottom, separated by a gap of "
    "{gap} {unit} of color {s}. The output is exactly as wide as the widest interior and exactly as tall as "
    "the interiors and the gaps between them together, with no gap before the first interior or after the last; each interior "
    "sits with its left column on the output's left column, and every cell no interior covers is color {s}."
)

def _p96(m):
    # Rebuilt on 2026-09-15 (generators/frame_extract_96.py).  "Rectangular frame"
    # is defined, and every clause of the definition is load-bearing (broken rings,
    # solid color F bodies and rings with color F content inside them all occur).
    # The gap colour is a task given drawn independently of the background, which
    # is never named: the old wording said "with 1 cells of background color 4
    # between adjacent interiors".
    frame = _ci(m, 'frame_color')
    spacer = _ci(m, 'spacer_color')
    gap = int(_m(m, 'gap', '1'))
    vertical = _m(m, 'arrangement') == "vertical"
    unit = ("row" if gap == 1 else "rows") if vertical else ("column" if gap == 1 else "columns")
    layout = (_P96_COLUMN if vertical else _P96_ROW).format(gap=gap, unit=unit, s=spacer)
    return (
        f"{_P96_FRAME_RULE.format(f=frame)} Find every rectangular frame of color {frame} and sort "
        f"them by top row, then by left column. Each frame's interior is the block of cells strictly "
        f"inside its ring, copied exactly. {layout}"
    )

def _p118(m):
    rule = _m(m, 'summary_rule')
    if rule == "top_left_color":
        detail = "use the top-left cell color of each block"
    elif rule == "majority_color":
        detail = f"use the single most frequent color in each block; if no color is uniquely most frequent, use background color {_ci(m, 'background_color')}"
    else:
        detail = f"output color {_ci(m, 'true_color')} if a block contains color {_ci(m, 'target_color')}, otherwise color {_ci(m, 'false_color')}"
    return f"Divide the grid into non-overlapping {_m(m, 'block_size')}x{_m(m, 'block_size')} blocks and compress each block to one output cell: {detail}."

def _p133(m):
    if _m(m, "predicate") == "non_background":
        predicate = f"cell color is not background color {_ci(m, 'background_color')}"
    else:
        predicate = f"cell color equals {_ci(m, 'target_color')}"
    return f"Output a same-size binary mask for this predicate: {predicate}. Use color {_ci(m, 'true_color')} for matching cells and color {_ci(m, 'false_color')} for all other cells."

def _p134(m):
    axis = _m(m, 'axis')
    policy = _m(m, 'marker_policy')
    if policy == "preserve_target_marker":
        suffix = f" After copying, keep the original target marker cell color {_ci(m, 'target_marker_color')} at its original position."
    else:
        suffix = " Copy every cell from the source line, including marker cells."
    return f"Copy the {axis} containing source marker color {_ci(m, 'source_marker_color')} to replace the {axis} containing target marker color {_ci(m, 'target_marker_color')}.{suffix}"

def _p135(m):
    condition = _m(m, 'condition')
    if condition == "all_background":
        cond = f"are entirely background color {_ci(m, 'background_color')}"
    else:
        cond = f"contain color {_ci(m, 'condition_color')}"
    axis = _m(m, 'axis')
    axis_phrase = "row and column" if axis == "both" else axis
    return (
        f"Using the original input grid, identify every {axis_phrase} that {cond}. "
        f"Then remove all of the identified lines at once (judging each row and column by "
        f"the original grid, not by the grid after other lines are removed) and compact the "
        f"remaining grid while preserving the original order."
    )

def _p136(m): return f"Mark every cell at the intersection of a row containing color {_ci(m, 'row_marker_color')} and a column containing color {_ci(m, 'column_marker_color')} with color {_ci(m, 'intersection_color')}. Intersection cells are recolored even if they currently contain another marker or distractor color. Leave non-intersection cells unchanged."

def _p137(m): return f"Background fill and any regular repeating texture pattern (for example a checkerboard or evenly spaced stripes) are not objects; consider only the irregular solid blobs. Among those objects, find the {_m(m, 'position_rule')} one using {_m(m, 'connectivity', '4')}-connectivity. Recolor all of its cells to color {_ci(m, 'target_color')}. If tied, use row-major order by top-left bounding-box corner."

def _p138(m): return f"For each color {_ci(m, 'target_color')} object, ignoring other-color objects and patterned background cells if present, recolor boundary cells to color {_ci(m, 'boundary_color')} and interior cells to color {_ci(m, 'interior_color')}. A boundary cell has at least one 4-neighbor outside its object."

def _p145(m): return f"Within each {_m(m, 'axis')}, compact all non-background cells toward the {_m(m, 'direction')}, preserving their order, and fill vacated cells with background color {_ci(m, 'background_color')}."

def _p146(m):
    # The target cell is the first or last cell encountered while scanning each
    # line in the given direction. State that endpoint concretely so "first" /
    # "last" is never read in plain reading order instead of scan order.
    axis = _m(m, 'axis')
    occ = _m(m, 'occurrence')
    scan = _m(m, 'scan_direction')
    endpoint = {
        ('left_to_right', 'first'): 'leftmost',
        ('left_to_right', 'last'): 'rightmost',
        ('right_to_left', 'first'): 'rightmost',
        ('right_to_left', 'last'): 'leftmost',
        ('top_to_bottom', 'first'): 'topmost',
        ('top_to_bottom', 'last'): 'bottommost',
        ('bottom_to_top', 'first'): 'bottommost',
        ('bottom_to_top', 'last'): 'topmost',
    }[(scan, occ)]
    return (
        f"In each {axis} that contains color {_ci(m, 'target_color')}, recolor only the "
        f"{endpoint} such cell in that {axis} to color {_ci(m, 'marker_color')}; leave every "
        f"other cell unchanged. Lines with no cell of color {_ci(m, 'target_color')} are left unchanged."
    )

def _p147(m):
    condition = _m(m, 'condition')
    if condition == "at_least":
        cond = f"length at least {_m(m, 'length')}"
    else:
        cond = f"length exactly {_m(m, 'length')}"
    return f"Find every maximal {_m(m, 'axis')} run of contiguous color {_ci(m, 'target_color')} cells with {cond} (any other color, including the background, ends a run), and recolor all cells in those runs to color {_ci(m, 'marker_color')}. Leave other-color cells unchanged."

def _p153(m): return f"Swap the two {_m(m, 'axis')}s containing marker colors {_ci(m, 'marker_color_a')} and {_ci(m, 'marker_color_b')}. Preserve all other rows and columns unchanged."

def _p154(m): return f"Treat any color that forms a regular repeating pattern tiled across the whole grid (for example a checkerboard or evenly spaced stripes) as patterned background, not as an object. For every other non-background single-color object using {_m(m, 'connectivity', '4')}-connectivity, replace the object with the filled rectangle of its bounding box using that object's original color."

def _p157(m):
    direction = _m(m, 'read_direction').replace('_', ' ')
    start_corner = {
        "top_left_to_bottom_right": "top-left corner, stepping down and to the right",
        "bottom_right_to_top_left": "bottom-right corner, stepping up and to the left",
        "top_right_to_bottom_left": "top-right corner, stepping down and to the left",
        "bottom_left_to_top_right": "bottom-left corner, stepping up and to the right",
    }.get(_m(m, 'read_direction'), direction)
    diag_name = {"main": "main", "anti": "anti-"}.get(_m(m, 'diagonal'), _m(m, 'diagonal'))
    return (
        f"Extract the {diag_name}diagonal, reading from the {start_corner} one cell at a "
        f"time until you leave the grid (so on a non-square grid the diagonal has the length of the "
        f"shorter side). Output it as a single {_m(m, 'output_orientation')}."
    )

def _p158(m):
    alignment = {
        "anti_diagonal": "anti-diagonal",
        "main_diagonal": "main diagonal",
    }.get(_m(m, 'alignment'), _m(m, 'alignment'))
    return f"Exactly two cells have marker color {_ci(m, 'marker_color')} and are aligned {alignment}. Ignoring patterned background cells if present, mark the unique midpoint cell whose row and column are the integer averages of the two marker coordinates with color {_ci(m, 'midpoint_color')}; leave all other cells unchanged."

def _p162(m): return f"Delete duplicate {_m(m, 'axis')} with identical full patterns, keeping the {_m(m, 'keep_occurrence')} occurrence of each pattern. Preserve the original order of retained rows or columns."

def _p164(m):
    sides = _m(m, 'sides')
    if sides == "all":
        return f"Pad all four sides of the grid with a {_m(m, 'thickness')}-cell-thick border of color {_ci(m, 'padding_color')}."
    sides_phrase = {
        "left_right": "left and right",
        "top_bottom": "top and bottom",
    }.get(sides, sides)
    return f"Pad the {sides_phrase} side(s) of the grid with a {_m(m, 'thickness')}-cell-thick border of color {_ci(m, 'padding_color')}."


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
PROMPT_TEMPLATES = {
    "1": _p1, "2": _p2, "3": _p3, "4": _p4, "5": _p5, "5a": _p5a,
    "6": _p6, "7": _p7, "8": _p8, "9": _p9, "10": _p10,
    "11": _p11, "12": _p12, "13": _p13, "14": _p14, "15": _p15,
    "15a": _p15a,
    "16": _p16, "17": _p17, "18": _p18, "20": _p20,
    "22": _p22, "23": _p23, "24": _p24, "25": _p25,
    "26": _p26, "27": _p27, "28": _p28, "29": _p29, "30": _p30,
    "31": _p31, "32": _p32, "33": _p33, "34": _p34, "35": _p35,
    "36": _p36, "37": _p37, "38": _p38, "39": _p39, "40": _p40,
    "41": _p41, "42": _p42, "44": _p44, "45": _p45,
    "46a": _p46a, "47": _p47,
    "49": _p49, "50": _p50, "51": _p51, "52": _p52,
    "54": _p54, "55": _p55, "56": _p56, "57": _p57, "58": _p58,
    "59": _p59, "60": _p60, "61": _p61, "62": _p62, "63": _p63,
    "64": _p64, "65": _p65, "66": _p66, "67": _p67,
    "68a": _p68a, "68b": _p68b,
    "70": _p70, "71": _p71, "72": _p72, "73": _p73, "74": _p74,
    "75": _p75, "76": _p76, "78": _p78,
    "80": _p80, "81": _p81, "82": _p82, "83": _p83,
    "84": _p84, "85": _p85, "87": _p87, "88": _p88,
    "89": _p89, "90": _p90, "91": _p91, "92": _p92,
    "93": _p93, "94": _p94, "95": _p95, "96": _p96, "118": _p118,
    "133": _p133, "134": _p134, "135": _p135, "136": _p136,
    "137": _p137, "138": _p138, "145": _p145, "146": _p146,
    "147": _p147, "153": _p153, "154": _p154, "157": _p157,
    "158": _p158, "162": _p162, "164": _p164,
}


def register_prompt(task_id: str):
    """Decorator for prompt modules that extend PROMPT_TEMPLATES."""
    def decorator(fn):
        if task_id in PROMPT_TEMPLATES:
            raise ValueError(f"Duplicate prompt template for task {task_id}")
        PROMPT_TEMPLATES[task_id] = fn
        return fn
    return decorator


def _load_external_prompt_modules() -> None:
    """Import optional prompt_modules/* files so they can register templates."""
    try:
        import importlib
        import pkgutil
        import prompt_modules
    except Exception:
        return
    for module_info in pkgutil.iter_modules(prompt_modules.__path__):
        if module_info.name.startswith("_"):
            continue
        importlib.import_module(f"prompt_modules.{module_info.name}")


_load_external_prompt_modules()
