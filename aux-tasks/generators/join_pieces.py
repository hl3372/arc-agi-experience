"""Task 13 — join the matching jagged pieces (shape assembly).

Scene model, built in a *canonical frame* where every fixed piece's seam faces
right and every loose piece's seam faces left:

* ``anchors`` are the fixed pieces.  Each anchor owns 1..4 *docking sites* on
  its right side.  A site is a run of rows whose right edge is a jagged seam
  profile; an anchor with several sites is a host with several cavities, an
  anchor with one site and no wall rows is a plain jagged half.
* every site owns exactly one loose piece whose left edge is the complement of
  the seam.  Loose pieces are scattered anywhere on the canvas (any side of
  their anchor) with a one-cell clearance from everything else.
* which pieces are fixed is stated by an explicit rule recorded in metadata
  (``anchor_rule``): ``border`` (fixed pieces touch the border opposite the
  docking edge, loose pieces never do) or ``color`` (all fixed pieces share
  ``anchor_color``; loose pieces use other colors).

Every example is re-solved from the *visible* input grid by an exhaustive
row-profile solver before it is accepted: each loose piece must have exactly
one legal docking translation over all fixed pieces, and the resulting
assembly must equal the constructed output.  The validated scene is finally
rotated/flipped into one of four orientations, so the docking edge is global
and named in the prompt; a second, independent re-solver written from the
prompt's wording then re-derives the answer on the oriented grids (background
= most frequent colour, pieces = 4-connected groups of >= 2 cells, fixed
pieces by the stated rule on the named side, brute-force docking) and must
agree.

September-2026 campaign (user: "as good and right as possible, with variety"):

* sizes: H and W independent from non-overlapping width-proportional bands
  [6,11] / [12,20] / [21,30] (0.28 / 0.36 / 0.36), so per-value density is
  near-flat; no bundle-derived cap (H*W <= 900).
* colours: a near-flat target for the number of piece colours; anchors and
  loose pieces draw from disjoint colour sets (the colour rule needs it, the
  border rule keeps it for readability), so a grid shows 2..10 colours.
* pieces scale with the canvas: span up to 18 rows, loose thickness up to 8,
  anchor thickness up to 10, relief up to 6, with flatter draws inside the
  caps, so large canvases carry large, deeply jagged pieces.
* backgrounds: ~28% of examples carry a SPARSE texture from
  ``common/backgrounds.py`` (isolated single cells in a colour no piece uses),
  painted only on free cells at least one cell away from every piece and from
  every docking footprint, identical in input and output.  Solid / structured
  kinds are deliberately NOT used: a band, a block or a ring is a 4-connected
  group of >= 2 cells of one colour, i.e. indistinguishable from a piece under
  the prompt's own rule (and a border ring would even count as a fixed piece
  under the border rule).  Nothing about the background is named in the prompt:
  the piece rule says pieces have >= 2 cells and isolated single cells are
  background texture, so the solver infers the texture.

Follow-up after the independent verification (same day):

* identifiability: nothing names the background, so ``_identify_background`` requires, after
  the isolated texture cells are set aside, that the background beats every other colour by
  ``BG_MARGIN`` (1.25x) and owns the largest 4-connected region; ``_validate`` and the
  prompt-side re-solver both use it (the verifier had found two exact ties and 134 margins
  under 1.25).  ``background_margin`` is recorded in metadata.
* the fixed pieces together cover at most ``FIXED_SHARE_CAP`` (36%) of the canvas: the anchor
  body is drawn inside a per-unit cell budget and the spans are redrawn narrower when even a
  one-cell body would overflow it (``_build_unit`` / ``_build_unit_once``).
* hosts with three or four cavities appear only where wide sites fit: the sites one anchor may
  carry follow the canvas height (about seven rows per site) and its area (three cavities from
  550 cells, four from 800); such hosts keep shallow cavities and slim loose pieces, and their
  spans lean on the upper half of the allowed range (``_sample_span(..., wide=True)``).
* a failed build is retried on the same canvas a few times before the size is resampled, so
  rejections do not tilt the size field; ``_REJECTS`` counts the retry reasons (diagnostics).
* prompt: "one connected block of a single color" (was "solid block"); the parenthetical
  "(do not rotate or reflect it)" replaces spaced hyphens.

Leftovers (2026-09-07; user: "every time all pieces are used. We should have variants with extra
pieces, and variants where there are holes that would not be filled"):

* about 30% of the examples carry one or two EXTRA loose pieces that seat flush-and-jagged
  nowhere (against no fixed piece, from no direction, anywhere in free space: exhaustive
  ``_all_fits`` per anchor); they stay exactly where they are.  Kinds: a near-miss of a real
  loose piece (one cell added or removed; preferably of an open cavity's piece), a fresh jagged
  piece from the seam vocabulary; at least four cells, never a plain rectangle (``_build_extras``).
* about 30% carry one or two OPEN cavities: a site whose loose piece is simply not in the scene;
  the cavity's footprint stays reserved (nothing is placed in it, no texture cell either) and
  the fixed piece is unchanged.  At least one piece always docks.
* both re-solvers treat a loose piece with no legal seat as staying put (``_assemble``); the
  generator additionally requires that exactly the extras stay.  Uniqueness is unchanged: every
  loose piece has exactly one seat (docked) or none (extra), over all fixed pieces and all four
  directions, and open cavities admit no loose piece (else the piece would have two seats or the
  extra one).
* the prompt never says whether an example has leftovers: the receive rules say "at most one
  loose piece" per cavity / fixed piece and a standing sentence covers both cases ("A loose
  piece that seats nowhere stays exactly where it is, and a cavity that no loose piece fits
  stays open.").  Metadata: ``n_docked``, ``n_open_cavities``, ``site_open``, ``open_sites``,
  ``open_cavity_sides``, ``open_cavity_cells``, ``n_extra``, ``extra_kinds``, ``extra_sizes``,
  ``extra_piece_colors``; ``n_loose`` counts the loose pieces visible in the input.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

import common.backgrounds as bglib
from common.grid import NUM_COLORS, grid_to_list
from common.objects import connected_components
from registry import register

MAX_CANVAS_ATTEMPTS = 400
_PARAM_TRIES = 6      # size draws per canvas; each draw shrinks the size caps a little
_PLACE_TRIES = 5      # random placements per size draw (later tries pack pieces together)
_UNIT_TRIES = 6       # rebuilds of one anchor unit (seam clashes with earlier anchors)

_STYLES = ("random", "teeth", "staircase", "notch", "vshape", "plateau", "walk")
_STYLE_WEIGHTS = (0.18, 0.14, 0.14, 0.13, 0.13, 0.15, 0.13)
_RELIEF_WEIGHTS = {1: 0.10, 2: 0.18, 3: 0.22, 4: 0.20, 5: 0.13, 6: 0.09, 7: 0.05, 8: 0.03}
_ANCHOR_THICK_WEIGHTS = {1: 0.10, 2: 0.22, 3: 0.20, 4: 0.14, 5: 0.10, 6: 0.08, 7: 0.06, 8: 0.04, 9: 0.03, 10: 0.03}
_LOOSE_THICK_WEIGHTS = {1: 0.18, 2: 0.24, 3: 0.20, 4: 0.13, 5: 0.10, 6: 0.07, 7: 0.05, 8: 0.03}
_TEXTURE_P = 0.28          # share of examples that carry a sparse background texture
_EXTRA_P = 0.42            # draw rate for one or two loose pieces that seat nowhere (they stay);
_OPEN_P = 0.33             # ... and for a cavity that no loose piece fits (it stays open).  Builds and
#                            placements fail more often with leftovers, so ~30% of the examples end
#                            up with each (drawn independently: ~10% both, ~50% neither)
_EXTRA_KINDS = ("near_miss", "jagged")   # no plain blocks: a rectangle can never seat under the
_EXTRA_KIND_WEIGHTS = (0.60, 0.40)       # jaggedness rule, so it would be a giveaway, not a trap
_EXTRA_MIN_CELLS = 4                     # an extra must look like a candidate, not a crumb
_EXTRA_TRIES = 40
_4DIRS = ((0, 1), (0, -1), (1, 0), (-1, 0))
BG_MARGIN = 1.25           # background count must exceed every other colour's by this factor
FIXED_SHARE_CAP = 0.36     # fixed pieces may cover at most this share of the canvas (a host with
                           # cavities needs room; the background margin is the real guard)
_REJECTS: Counter = Counter()   # diagnostics only: why builds were retried
_NO_TEXTURE = {"texture_group": None, "texture_kind": None, "texture_colors": [],
               "texture_named": False, "texture_coverage": 0.0}
_PROTRUSION_WEIGHTS = {-1: 0.08, 0: 0.30, 1: 0.22, 2: 0.16, 3: 0.12, 4: 0.12}
_OUTER_KINDS = ("straight", "stepped", "walk")
_OUTER_WEIGHTS = (0.45, 0.35, 0.20)
_OUTER_WEIGHTS_MULTI = (0.60, 0.28, 0.12)   # multi-side pieces: mostly straight outer edges (fewer
                                            # accidental seatings in the perpendicular directions)
class _CapsInfeasible(Exception):
    """A multi-side plan cannot fit its body caps at all (later, smaller draws cannot help)."""


_MULTI_TRIES = 6      # rebuilds of one multi-side unit (local uniqueness over four directions)
_SITE_COUNT_WEIGHTS = (0.10, 0.30, 0.27, 0.17, 0.10, 0.06)  # n_sites = 1..6 (before canvas feasibility)
_SITE_COUNT_WEIGHTS_LARGE = (0.08, 0.24, 0.26, 0.20, 0.13, 0.09)  # canvases of 400+ cells: more pieces, more colours

# Compositions of n docking sites into anchors (sites per anchor, each <= 4) with weights.
_COMPOSITIONS: dict[int, list[tuple[tuple[int, ...], float]]] = {
    1: [((1,), 1.0)],
    2: [((1, 1), 0.55), ((2,), 0.45)],
    3: [((1, 1, 1), 0.40), ((3,), 0.35), ((2, 1), 0.25)],
    4: [((1, 1, 1, 1), 0.30), ((4,), 0.30), ((2, 2), 0.15), ((3, 1), 0.13), ((2, 1, 1), 0.12)],
    5: [((1, 1, 1, 1, 1), 0.25), ((4, 1), 0.15), ((3, 2), 0.18), ((3, 1, 1), 0.14),
        ((2, 2, 1), 0.16), ((2, 1, 1, 1), 0.12)],
    6: [((1, 1, 1, 1, 1, 1), 0.20), ((4, 2), 0.15), ((3, 3), 0.18), ((2, 2, 2), 0.12),
        ((4, 1, 1), 0.10), ((3, 2, 1), 0.12), ((2, 2, 1, 1), 0.08), ((3, 1, 1, 1), 0.05)],
}

# canonical dock edge is "right"; the transform maps it to the named global edge
_OPPOSITE = {"right": "left", "left": "right", "top": "bottom", "bottom": "top"}


# ---------------------------------------------------------------------------
# small sampling helpers
# ---------------------------------------------------------------------------

def _wchoice(rng: np.random.Generator, values, weights):
    """Weighted choice over a small list (weights need not be normalised)."""
    total = float(sum(weights))
    u = rng.random() * total
    acc = 0.0
    for v, w in zip(values, weights):
        acc += w
        if u < acc:
            return v
    return values[-1]


def _wchoice_capped(rng: np.random.Generator, table: dict[int, float], lo: int, hi: int) -> int:
    """Weighted choice from *table* restricted to [lo, hi] (hi >= lo assumed)."""
    vals = [v for v in table if lo <= v <= hi]
    if not vals:
        return lo
    return _wchoice(rng, vals, [table[v] for v in vals])


def _sample_side(rng: np.random.Generator) -> int:
    """One side in 6..30 from non-overlapping bands [6,11] / [12,20] / [21,30] weighted
    0.28 / 0.36 / 0.36 (about width-proportional with a mild small lean): per-value
    density is near-flat, with no spike where bands used to overlap."""
    u = rng.random()
    if u < 0.28:
        return int(rng.integers(6, 12))
    if u < 0.64:
        return int(rng.integers(12, 21))
    return int(rng.integers(21, 31))


def _thick_draw(rng: np.random.Generator, table: dict[int, float], lo: int, hi: int) -> int:
    """Body thickness: half the time uniform over [lo, hi] (so large canvases get thick
    pieces), half the time from the small-leaning table."""
    if hi <= lo:
        return int(lo)
    if rng.random() < 0.5:
        return int(rng.integers(lo, hi + 1))
    return _wchoice_capped(rng, table, lo, hi)


def _flat_count(rng: np.random.Generator, lo: int, hi: int) -> int:
    """Integer in [lo, hi], near-flat with a mild lean toward lo."""
    if hi <= lo:
        return int(lo)
    vals = list(range(lo, hi + 1))
    weights = [1.0 - 0.05 * (v - lo) for v in vals]
    return int(_wchoice(rng, vals, weights))


def _cycle_colors(rng: np.random.Generator, colors: list[int], n: int) -> list[int]:
    """Assign *n* pieces colours from *colors* so every colour is used at least once
    when n >= len(colors); order shuffled."""
    order = [colors[i % len(colors)] for i in range(n)]
    perm = rng.permutation(n)
    return [order[int(i)] for i in perm]


def _sample_canvas(rng: np.random.Generator) -> tuple[int, int]:
    """H, W independent with an ARC-like small bias; any 1..30 x 1..30 canvas (the former
    H*W <= 700 rejection came from the SFT bundle budget, which is not a task-design input)."""
    return _sample_side(rng), _sample_side(rng)


def _sample_span(rng: np.random.Generator, cap: int, lo: int = 2, wide: bool = False) -> int:
    """Site span in [lo, cap] (cap <= 18): 45% uniform over the upper half, 25% uniform over
    the whole range, 30% small-leaning, so big canvases get long seams and small ones stay
    feasible.  ``wide`` (multi-site hosts) leans harder on the upper half (70 / 15 / 15) so the
    sites of one host are not all the minimum width."""
    cap = max(lo, min(18, cap))
    if cap == lo:
        return lo
    u = rng.random()
    p_upper, p_flat = (0.70, 0.85) if wide else (0.45, 0.70)
    if u < p_upper:
        return int(rng.integers(max(lo, (cap + 1) // 2), cap + 1))
    if u < p_flat:
        return int(rng.integers(lo, cap + 1))
    return lo + int(round((cap - lo) * float(rng.beta(1.4, 2.2))))


# ---------------------------------------------------------------------------
# seam vocabulary
# ---------------------------------------------------------------------------

def _seam_profile(rng: np.random.Generator, span: int, d: int, style: str) -> list[int] | None:
    """Per-row seam heights in [0, d] (min 0, max d) in one of several jagged styles.

    Returns None when the style cannot be realised for this span."""
    if style == "random":
        p = [int(v) for v in rng.integers(0, d + 1, size=span)]
        if max(p) - min(p) != d:
            i, j = (0, 1) if span == 2 else [int(v) for v in rng.choice(span, size=2, replace=False)]
            p[i], p[j] = 0, d
        return p
    if style == "teeth":
        w = int(rng.integers(1, max(1, min(3, span // 2)) + 1))
        phase = int(rng.integers(0, 2 * w))
        return [d if ((j + phase) // w) % 2 == 0 else 0 for j in range(span)]
    if style == "staircase":
        if span >= 3 and rng.random() < 0.4:  # triangle (rise then fall)
            m = int(rng.integers(1, span - 1))
            return [int(round(d * j / m)) if j <= m else int(round(d * (span - 1 - j) / (span - 1 - m)))
                    for j in range(span)]
        if rng.random() < 0.5:  # even steps
            p = [int(round(d * j / (span - 1))) for j in range(span)]
        else:  # irregular monotone steps
            inner = sorted(int(v) for v in rng.integers(0, d + 1, size=span - 2))
            p = [0] + inner + [d]
        if rng.random() < 0.5:
            p.reverse()
        return p
    if style == "vshape":
        if span < 3:
            return None
        m = int(rng.integers(1, span - 1))
        return [d - (int(round(d * j / m)) if j <= m else int(round(d * (span - 1 - j) / (span - 1 - m))))
                for j in range(span)]
    if style == "notch":
        w = int(rng.integers(1, span))
        start = int(rng.integers(0, span - w + 1))
        p = [0] * span
        for j in range(start, start + w):
            p[j] = d
        if span >= 5 and rng.random() < 0.3:  # second notch of width 1..2 at a level in (0, d]
            w2 = int(rng.integers(1, 3))
            s2 = int(rng.integers(0, span - w2 + 1))
            lvl = int(rng.integers(1, d + 1))
            for j in range(s2, s2 + w2):
                p[j] = max(p[j], lvl) if p[j] == 0 else p[j]
        if rng.random() < 0.5:  # dip instead of bump
            p = [d - v for v in p]
        if max(p) - min(p) != d:
            return None
        return p
    if style == "plateau":
        levels = [0, d] if d < 2 or rng.random() < 0.65 else [0, int(rng.integers(1, d)), d]
        p: list[int] = []
        lvl_idx = int(rng.integers(0, len(levels)))
        while len(p) < span:
            run = int(rng.integers(1, 4))
            p.extend([levels[lvl_idx]] * run)
            nxt = int(rng.integers(0, len(levels) - 1))
            lvl_idx = nxt if nxt < lvl_idx else nxt + 1
        p = p[:span]
        if max(p) - min(p) != d:
            return None
        return p
    if style == "walk":
        d_eff = min(d, span - 1)
        if d_eff < 1:
            return None
        for _ in range(40):
            p = [int(rng.integers(0, d_eff + 1))]
            for _j in range(1, span):
                u = rng.random()
                step = -1 if u < 0.4 else (1 if u >= 0.6 else 0)
                p.append(min(d_eff, max(0, p[-1] + step)))
            if max(p) - min(p) == d_eff:
                base = min(p)
                return [v - base for v in p]
        return None
    raise ValueError(style)


def _make_seam(rng, span: int, relief_cap: int, min_relief: int, used_keys: set) -> dict | None:
    """Pick a style + relief and realise a seam profile not used elsewhere in this scene."""
    if relief_cap < min_relief:
        return None
    relief = _thick_draw(rng, _RELIEF_WEIGHTS, min_relief, relief_cap)
    for attempt in range(14):
        style = _wchoice(rng, _STYLES, _STYLE_WEIGHTS)
        if attempt >= 8:  # only after several style failures fall back to a fresh relief draw
            relief = _thick_draw(rng, _RELIEF_WEIGHTS, min_relief, relief_cap)
        p = _seam_profile(rng, span, relief, style)
        if p is None:
            continue
        actual = max(p) - min(p)
        if actual < min_relief or actual < 1:
            continue
        key = (span, tuple(p))
        if key in used_keys:
            continue
        used_keys.add(key)
        return {"span": span, "relief": actual, "style": style, "p": p}
    return None


def _outer_profile(rng, n: int, kind: str, max_inset: int, trim: bool) -> list[int]:
    """Inset (>= 0, min 0) of a piece's outer side per row: straight / stepped / random-walk,
    optionally with trimmed corner cells on the first/last row."""
    if kind == "straight" or n < 2 or max_inset < 1:
        vals = [0] * n
    elif kind == "stepped":
        vals = [0] * n
        for _ in range(int(rng.integers(1, 3))):
            seg = int(rng.integers(1, min(3, n) + 1))
            start = int(rng.integers(0, n - seg + 1))
            extra = int(rng.integers(1, max_inset + 1))
            for r in range(start, start + seg):
                vals[r] = max(vals[r], extra)
    else:  # walk
        vals = [int(rng.integers(0, max_inset + 1))]
        for _ in range(n - 1):
            u = rng.random()
            step = -1 if u < 0.35 else (1 if u >= 0.65 else 0)
            vals.append(min(max_inset, max(0, vals[-1] + step)))
    base = min(vals)
    vals = [v - base for v in vals]
    if trim and n >= 2 and max_inset >= 1 and rng.random() < 0.3:
        if rng.random() < 0.6:
            vals[0] += 1
        if rng.random() < 0.6:
            vals[-1] += 1
        base = min(vals)
        vals = [v - base for v in vals]
    return vals


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------

def _cells_of(rows: list[tuple[int, int, int]]) -> set[tuple[int, int]]:
    return {(r, c) for r, lo, hi in rows for c in range(lo, hi + 1)}


def _intervals(cells: set) -> list[tuple[int, int, int]]:
    """Row-interval list (r, lo, hi) of a cell set; a row may yield several intervals."""
    by_row: dict[int, list[int]] = {}
    for r, c in cells:
        by_row.setdefault(r, []).append(c)
    out: list[tuple[int, int, int]] = []
    for r in sorted(by_row):
        cols = sorted(by_row[r])
        lo = prev = cols[0]
        for c in cols[1:]:
            if c != prev + 1:
                out.append((r, lo, prev))
                lo = c
            prev = c
        out.append((r, lo, prev))
    return out


def _bbox(cells: set) -> tuple[int, int, int, int]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _shift(cells: set, dr: int, dc: int) -> set:
    return {(r + dr, c + dc) for r, c in cells}


# the four directions a loose piece may arrive from: "beyond" = the step from a loose cell
# toward the fixed piece it seats against
_BEYOND = {"from_left": (0, 1), "from_right": (0, -1), "from_above": (1, 0), "from_below": (-1, 0)}


def _edge_cells(loose: set, beyond: tuple[int, int]) -> list[tuple[int, int]]:
    """Cells of the loose piece that face the fixed piece: no cell of the piece lies directly
    beyond them in the direction of approach."""
    dr, dc = beyond
    return [(r, c) for (r, c) in loose if (r + dr, c + dc) not in loose]


def _is_jagged(edge: list[tuple[int, int]], beyond: tuple[int, int]) -> bool:
    """The seam is jagged when the facing cells do not all lie on one straight line
    perpendicular to the direction of approach."""
    if beyond[0] == 0:          # arriving from the left / right: compare columns
        return len({c for _, c in edge}) > 1
    return len({r for r, _ in edge}) > 1


def _fits_dir(loose: set, anchor: set, all_anchor: set, h: int | None, w: int | None,
              beyond: tuple[int, int]) -> dict[tuple[int, int], frozenset]:
    """Every translation of *loose* that seats it flush against *anchor* arriving in direction
    *beyond*: each facing cell has a cell of *anchor* directly beyond it, the seam is jagged, the
    piece stays inside the grid and overlaps no fixed cell.  Exhaustive: the first facing cell
    must sit directly before some anchor cell, which fixes the translation.  Returns
    {translation: placed cells}."""
    edge = _edge_cells(loose, beyond)
    if not _is_jagged(edge, beyond):
        return {}
    dr, dc = beyond
    r0, c0 = edge[0]
    # the other facing cells' "beyond" cells relative to the first one's; farthest first so a
    # wrong candidate fails on the first comparison most of the time
    rest = sorted(((r - r0, c - c0) for r, c in edge[1:]), key=lambda t: -(abs(t[0]) + abs(t[1])))
    fits: dict[tuple[int, int], frozenset] = {}
    for ar, ac in anchor:          # every anchor cell is a candidate for the first facing cell's beyond cell
        for er, ec in rest:
            if (ar + er, ac + ec) not in anchor:
                break
        else:
            dy, dx = ar - dr - r0, ac - dc - c0
            placed = frozenset((r + dy, c + dx) for r, c in loose)
            if h is not None and any(r < 0 or c < 0 or r >= h or c >= w for r, c in placed):
                continue
            if placed & all_anchor:
                continue
            fits[(dy, dx)] = placed
    return fits


def _all_fits(loose: set, anchors: list[set], all_anchor: set, h: int | None, w: int | None) -> set:
    """Distinct placements (as frozensets) of *loose* seating flush against any of *anchors*
    from any of the four directions (a placement that seats from two directions counts once)."""
    placements: set[frozenset] = set()
    for a in anchors:
        for beyond in _BEYOND.values():
            placements.update(_fits_dir(loose, a, all_anchor, h, w, beyond).values())
    return placements


def _dilate(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _pick_position(rng, blocked: np.ndarray, rows: list[tuple[int, int, int]], h: int, w: int,
                   fixed_left: int | None = None, min_left: int = 0,
                   hug: bool = False) -> tuple[int, int] | None:
    """Uniformly pick (top, left) so that no cell of the row-interval piece is blocked.

    ``hug`` prefers positions touching a grid border (keeps the free space in one block)."""
    hc, wc = blocked.shape
    nh, nw = hc - h + 1, wc - w + 1
    if nh <= 0 or nw <= 0:
        return None
    cs = np.zeros((hc, wc + 1), dtype=np.int32)
    np.cumsum(blocked, axis=1, out=cs[:, 1:])
    bad = np.zeros((nh, nw), dtype=bool)
    for r, lo, hi in rows:
        bad |= (cs[r:r + nh, hi + 1:hi + 1 + nw] - cs[r:r + nh, lo:lo + nw]) > 0
    if fixed_left is not None:
        bad[:, :fixed_left] = True
        bad[:, fixed_left + 1:] = True
    elif min_left > 0:
        bad[:, :min_left] = True
    idx = np.flatnonzero(~bad)
    if idx.size == 0:
        return None
    if hug:
        # packing heuristic: prefer positions whose bounding box touches a border or whose
        # one-cell ring overlaps the clearance zone of something already placed
        pad = np.zeros((hc + 2, wc + 2), dtype=np.int32)
        pad[1:-1, 1:-1] = blocked
        ii = np.zeros((hc + 3, wc + 3), dtype=np.int32)
        ii[1:, 1:] = pad.cumsum(axis=0).cumsum(axis=1)
        ring = (ii[h + 2:h + 2 + nh, w + 2:w + 2 + nw] - ii[0:nh, w + 2:w + 2 + nw]
                - ii[h + 2:h + 2 + nh, 0:nw] + ii[0:nh, 0:nw])
        touch = ring > 0
        touch[0, :] = True
        touch[-1, :] = True
        touch[:, 0] = True
        touch[:, -1] = True
        idx_h = np.flatnonzero(~bad & touch)
        if idx_h.size:
            idx = idx_h
    pick = int(idx[int(rng.integers(0, idx.size))])
    return pick // nw, pick % nw


def _paint(grid: np.ndarray, rows, top: int, left: int, color: int) -> None:
    for r, lo, hi in rows:
        grid[top + r, left + lo:left + hi + 1] = color


# ---------------------------------------------------------------------------
# one anchor with its docking sites and loose pieces (local coordinates)
# ---------------------------------------------------------------------------

def _build_unit(rng, k: int, h_cap: int, w_cap: int, span_cap: int, min_span: int,
                min_relief: int, rect_union: bool, used_keys: set,
                ta_cap: int = 10, tl_cap: int = 8, relief_hi: int = 6,
                cell_budget: int | None = None) -> dict | None:
    """Build one fixed piece with *k* docking sites plus its k docked loose pieces.

    ``h_cap``/``w_cap`` bound the unit (anchor + docked pieces), ``span_cap``/``min_span`` bound
    each site's row span, ``ta_cap``/``tl_cap``/``relief_hi`` bound body thicknesses and seam
    relief, ``cell_budget`` (when given) bounds the anchor's cell count so the fixed pieces stay
    a minority of the canvas: the anchor body is drawn inside the budget the spans and seam
    depths leave, and when even a 1-thick body would overflow it the spans are redrawn smaller.
    Returns a dict with row-interval lists in local coordinates (anchor's min column = 0):
    ``anchor_rows``, ``loose`` (dicts with ``rows`` in docked position and seam info), or
    None when the caps cannot be met or a seam is not locally unique."""
    if w_cap < 5 or h_cap < k * min_span + (k - 1):
        return None
    if k >= 3:
        # a host with several cavities: keep its cavities shallow and its loose pieces slim so
        # the width stays inside the cell budget and the rows can go to the sites' spans
        tl_cap = min(tl_cap, 3)
        relief_hi = min(relief_hi, 5)
    for _span_try in range(3):
        unit = _build_unit_once(rng, k, h_cap, w_cap, span_cap, min_span, min_relief, rect_union,
                                used_keys, ta_cap, tl_cap, relief_hi, cell_budget)
        if unit is not None:
            return unit
        span_cap = max(min_span, int(round(span_cap * 0.7)))   # the budget was missed: narrower sites
    return None


def _build_unit_once(rng, k: int, h_cap: int, w_cap: int, span_cap: int, min_span: int,
                     min_relief: int, rect_union: bool, used_keys: set,
                     ta_cap: int, tl_cap: int, relief_hi: int, cell_budget: int | None) -> dict | None:
    # --- wall rows (rows of the anchor that are not part of a site)
    if k == 1:
        top = bot = 0
        if rng.random() < 0.28:
            top, bot = int(rng.integers(0, 3)), int(rng.integers(0, 3))
            if top == 0 and bot == 0:
                top = 1
        between: list[int] = []
    else:
        between = [_wchoice(rng, [1, 2, 3], [0.55, 0.30, 0.15]) for _ in range(k - 1)]
        top = _wchoice(rng, [0, 1, 2, 3], [0.30, 0.40, 0.20, 0.10])
        bot = _wchoice(rng, [0, 1, 2, 3], [0.30, 0.40, 0.20, 0.10])
    walls = top + bot + sum(between)
    if h_cap - walls < k * min_span:
        between, top, bot = [1] * (k - 1), 0, 0
        walls = k - 1
        if h_cap - walls < k * min_span:
            return None
    # --- spans
    budget = h_cap - walls
    spans: list[int] = []
    fair = max(min_span, int(round(1.6 * budget / k)))
    for i in range(k):
        cap = min(span_cap, fair, budget - sum(spans) - min_span * (k - i - 1))
        if cap < min_span:
            return None
        spans.append(_sample_span(rng, cap, min_span, wide=k >= 3))
    host_like = k >= 2 or walls > 0
    n_rows = walls + sum(spans)
    segments = [top] + between + [bot]  # wall rows before site 0, between sites, after the last
    # --- anchor outer (left) side
    outer_a = "straight" if rect_union else _wchoice(rng, _OUTER_KINDS, _OUTER_WEIGHTS)
    max_inset = 2 if w_cap >= 9 else 1
    L = _outer_profile(rng, n_rows, outer_a, max_inset, trim=not rect_union)
    max_l = max(L)
    relief_cap = min(relief_hi, w_cap - 4)

    for _seam_try in range(6):
        keys = set(used_keys)
        seams = []
        for span in spans:
            seam = _make_seam(rng, span, relief_cap, min_relief, keys)
            if seam is None:
                break
            seams.append(seam)
        if len(seams) != k:
            continue
        # cavity depth beyond the mouth (host-like) / plain seam base (pair)
        site_plan: list[dict] = []
        if host_like:
            # A cavity's loose piece is exactly as thick as the cavity's extra depth plus its
            # protrusion minus its outer inset, so draw the thickness first (from the room the
            # site has) and derive the extra depth from it.
            t_lo = 2 if tl_cap >= 4 else 1
            for s in seams:
                outer_l = "straight" if rect_union else _wchoice(rng, _OUTER_KINDS, _OUTER_WEIGHTS)
                o = _outer_profile(rng, s["span"], outer_l, max_inset, trim=not rect_union)
                protrusion = 0 if rect_union else _wchoice_capped(rng, _PROTRUSION_WEIGHTS, -1, 4)
                t_target = _thick_draw(rng, _LOOSE_THICK_WEIGHTS, t_lo, max(t_lo, min(8, tl_cap)))
                extra = max(1, t_target - protrusion + max(o))
                site_plan.append({"outer": outer_l, "o": o, "protrusion": protrusion, "extra": extra})
            depths = [s["relief"] + sp["extra"] for s, sp in zip(seams, site_plan)]
        else:
            depths = [seams[0]["relief"]]
        d_max = max(depths)
        t_cap = w_cap - max_l - d_max - 2
        if cell_budget is not None:
            # every anchor row is at most (mouth + 1) cells wide: keep n_rows * (mouth + 1) inside the budget
            t_cap = min(t_cap, cell_budget // n_rows - max_l - d_max)
        if t_cap < 1:
            continue
        t_anchor = _thick_draw(rng, _ANCHOR_THICK_WEIGHTS, 2 if min(t_cap, ta_cap) >= 4 else 1,
                               min(10, t_cap, ta_cap))
        mouth = max_l + t_anchor - 1 + d_max
        # --- right edge per row
        R = [0] * n_rows
        stepped_mouth = host_like and not rect_union and rng.random() < 0.2
        r = 0
        site_start: list[int] = []
        for i in range(k):
            for _ in range(segments[i]):
                R[r] = mouth - (1 if stepped_mouth and rng.random() < 0.3 else 0)
                r += 1
            site_start.append(r)
            for j in range(spans[i]):
                R[r] = mouth - depths[i] + seams[i]["p"][j]
                r += 1
        for _ in range(segments[k]):
            R[r] = mouth - (1 if stepped_mouth and rng.random() < 0.3 else 0)
            r += 1
        anchor_rows = [(rr, L[rr], R[rr]) for rr in range(n_rows)]
        # --- loose pieces (docked position)
        loose: list[dict] = []
        for i, seam in enumerate(seams):
            span = seam["span"]
            rows_i = list(range(site_start[i], site_start[i] + span))
            max_r = max(R[rr] for rr in rows_i)
            if host_like:
                outer_l, o = site_plan[i]["outer"], site_plan[i]["o"]
            else:
                outer_l = "straight" if rect_union else _wchoice(rng, _OUTER_KINDS, _OUTER_WEIGHTS)
                o = _outer_profile(rng, span, outer_l, max_inset, trim=not rect_union)
            max_o = max(o)
            protrusion: int | None = None
            if host_like:
                p_cap = w_cap - 1 - mouth
                protrusion = min(site_plan[i]["protrusion"], p_cap)
                t = min(tl_cap, protrusion + mouth - max_r - max_o)
                if t < 1:
                    t = 1
                    if max_r + t + max_o > w_cap - 1:
                        o, max_o = [0] * span, 0
                    if max_r + t + max_o > w_cap - 1:
                        loose = []
                        break
            else:
                cap = w_cap - 1 - max_r - max_o
                if cap < 1:
                    o, max_o = [0] * span, 0
                    cap = w_cap - 1 - max_r
                    if cap < 1:
                        loose = []
                        break
                t = _thick_draw(rng, _LOOSE_THICK_WEIGHTS, 2 if min(cap, tl_cap) >= 4 else 1,
                                min(8, cap, tl_cap))
            q_max = max_r + t + max_o
            if host_like:
                protrusion = q_max - mouth
            rows = [(rr, R[rr] + 1, q_max - o[j]) for j, rr in enumerate(rows_i)]
            loose.append({
                "rows": rows, "span": span, "relief": seam["relief"], "style": seam["style"],
                "outer": outer_l, "thickness": t, "protrusion": protrusion,
            })
        if len(loose) != k:
            continue
        # --- local uniqueness: each loose piece must seat against this anchor in exactly one
        # way over all four directions of approach (the seam is jagged by construction)
        anchor_cells = _cells_of(anchor_rows)
        loose_cells = [_cells_of(p["rows"]) for p in loose]
        if any(len(_all_fits(lc, [anchor_cells], anchor_cells, None, None)) != 1 for lc in loose_cells):
            continue
        used_keys |= keys
        for p, lc in zip(loose, loose_cells):
            p["cells"] = lc
            p["side"] = "right"
        anchor_w = max(hi for _, _, hi in anchor_rows) + 1
        return {
            "anchor_rows": anchor_rows, "anchor_cells": anchor_cells, "loose": loose,
            "loose_cells": loose_cells, "k": k, "outer": outer_a, "thickness": t_anchor,
            "walls": walls, "h": n_rows, "w": anchor_w,
            "unit_w": max(anchor_w, max(max(hi for _, _, hi in p["rows"]) for p in loose) + 1),
            "sides": ["right"], "free_sides": ["left", "top", "bottom"],
            "body": (n_rows, anchor_w),
        }
    return None


# ---------------------------------------------------------------------------
# multi-side anchors: cavities on two, three or four sides of one fixed piece
# ---------------------------------------------------------------------------

_SIDES = ("right", "left", "top", "bottom")
_ADJACENT = {"right": ("top", "bottom"), "left": ("top", "bottom"),
             "top": ("left", "right"), "bottom": ("left", "right")}


def _side_map(side: str, u: int, r: int, L: int) -> tuple[int, int]:
    """Map a cell given along a side (r = position along the side, u = position along the
    outward axis with the body's far edge at u = L - 1; u >= L protrudes) to body coordinates."""
    if side == "right":
        return r, u
    if side == "left":
        return r, L - 1 - u
    if side == "bottom":
        return u, r
    return L - 1 - u, r          # top


def _build_multi_unit(rng, sides: list[str], k_per_side: list[int], h_cap: int, w_cap: int,
                      span_cap: int, min_span: int, min_relief: int, used_keys: set,
                      tl_cap: int, relief_hi: int, cell_budget: int | None) -> dict | None:
    """One fixed piece with jagged cavities on several sides plus its docked loose pieces.

    The anchor starts as a solid body; on each docking side ``k`` cavities (runs of ``span``
    rows or columns, separated by wall cells, kept away from the corners) are carved to a
    jagged profile, and the loose piece of a cavity is its exact complement plus an optional
    protrusion beyond the body edge.  Corner clearance: a side's runs start at least one cell
    beyond the deepest cavity of each adjacent docking side, so cavities never meet and docked
    pieces never touch around a corner.  Every loose piece must seat against the anchor in
    exactly one way over all four directions.  Coordinates are body-relative (the body's top-left
    is (0, 0)); loose cells may be negative (protruding left / above) and are normalised by
    the caller.  Returns None when the caps cannot be met.

    The build is need-driven: cavity depths are drawn first and shrunk until the corner margins
    and the core fit ``h_cap`` x ``w_cap``, then the spans are drawn to fit the run lengths that
    remain, so a unit fails only on uniqueness or clash grounds."""
    if len(sides) < 2 or any(k < 1 for k in k_per_side):
        return None
    tl_cap = min(tl_cap, 3)
    min_relief = max(min_relief, 2)
    relief_hi = max(min_relief, min(relief_hi, 4))
    span_cap = max(min_span, min(span_cap, 9))
    vertical = ("left", "right")   # runs along rows; margins from top/bottom depths
    for _try in range(_MULTI_TRIES):
        keys = set(used_keys)
        # --- 1. per-site depth specs (span-free): relief cap, thickness, protrusion, outer kind
        specs: dict[str, list[dict]] = {}
        for side, k in zip(sides, k_per_side):
            lst = []
            for _ in range(k):
                outer = _wchoice(rng, _OUTER_KINDS, _OUTER_WEIGHTS_MULTI)
                lst.append({"relief": _thick_draw(rng, _RELIEF_WEIGHTS, min_relief, relief_hi),
                            "t": _thick_draw(rng, _LOOSE_THICK_WEIGHTS, 1, max(1, tl_cap)),
                            "protrusion": _wchoice_capped(rng, _PROTRUSION_WEIGHTS, 0, 2),
                            "outer": outer})
            specs[side] = lst

        def depth_of(sp):
            return sp["relief"] + max(1, sp["t"] - sp["protrusion"] + (0 if sp["outer"] == "straight" else 1))

        def margins():
            mr = max([depth_of(sp) + 1 for s_ in ("top", "bottom") if s_ in specs for sp in specs[s_]] + [1])
            mc = max([depth_of(sp) + 1 for s_ in ("left", "right") if s_ in specs for sp in specs[s_]] + [1])
            return mr, mc

        def needs():
            mr, mc = margins()
            nh = max([2 * mr + len(specs[s_]) * min_span + len(specs[s_]) - 1 for s_ in vertical if s_ in specs] + [3])
            nw = max([2 * mc + len(specs[s_]) * min_span + len(specs[s_]) - 1 for s_ in ("top", "bottom") if s_ in specs] + [3])
            core_w = sum(max(depth_of(sp) for sp in specs[s_]) for s_ in ("left", "right") if s_ in specs) + 2
            core_h = sum(max(depth_of(sp) for sp in specs[s_]) for s_ in ("top", "bottom") if s_ in specs) + 2
            return max(nh, core_h), max(nw, core_w)

        # --- 2. shrink the deepest specs until the minimal body fits the caps
        feasible = True
        budget = cell_budget if cell_budget is not None else 10 ** 6
        for _ in range(40):
            nh, nw = needs()
            if nh <= h_cap and nw <= w_cap and nh * nw <= budget:
                break
            if nh > h_cap:
                axis_sides = ("top", "bottom")
            elif nw > w_cap:
                axis_sides = ("left", "right")
            else:   # over the cell budget: shrink the margins of the longer axis first
                axis_sides = ("top", "bottom") if nh >= nw else ("left", "right")
                if not any(s_ in specs for s_ in axis_sides):
                    axis_sides = ("left", "right") if nh >= nw else ("top", "bottom")
            cand = [sp for s_ in axis_sides if s_ in specs for sp in specs[s_]]
            cand.sort(key=depth_of, reverse=True)
            if not cand:                 # the runs alone overflow the axis: no depth to shrink
                feasible = False
                break
            sp = cand[0]
            if sp["relief"] > min_relief:
                sp["relief"] -= 1
            elif sp["outer"] != "straight":
                sp["outer"] = "straight"
            elif sp["t"] > 1:
                sp["t"] -= 1
            elif sp["protrusion"] < 1:
                sp["protrusion"] = 1
            else:
                feasible = False
                break
        else:
            feasible = False
        if not feasible:
            _REJECTS["multi-side body caps"] += 1
            raise _CapsInfeasible   # no smaller draw can help: the caller moves to a new plan
        m_rows, m_cols = margins()
        need_h, need_w = needs()

        # --- 3. spans that fit the remaining run length, then seams and outer profiles
        ok = True
        cur_h, cur_w = need_h, need_w
        for side, lst in sorted(specs.items(), key=lambda kv: kv[0] not in vertical):
            length_cap = h_cap if side in vertical else w_cap
            other = cur_w if side in vertical else cur_h
            length_cap = min(length_cap, budget // max(1, other))   # keep the body under budget
            m = m_rows if side in vertical else m_cols
            avail = length_cap - 2 * m - (len(lst) - 1)          # cells available for the spans
            for i, sp in enumerate(lst):
                rest = (len(lst) - 1 - i) * min_span
                cap = min(span_cap, avail - rest)
                seam = None
                for _draw in range(3):      # a fresh span when the seam profiles of this span are used up
                    lo = min_span if cap < 3 else max(min_span, 3)   # two-cell seams only when forced
                    span = _sample_span(rng, cap, lo, wide=False)
                    seam = _make_seam(rng, span, sp["relief"], min_relief, keys)
                    if seam is not None:
                        break
                if seam is None:
                    ok = False
                    break
                avail -= span
                o = _outer_profile(rng, span, sp["outer"], 1, trim=True)
                sp["span"], sp["seam"], sp["o"] = span, seam, o
                sp["relief"] = seam["relief"]
                sp["extra"] = max(1, sp["t"] - sp["protrusion"] + max(o))
                sp["depth"] = sp["relief"] + sp["extra"]
            if not ok:
                break
            used_len = 2 * m + sum(sp["span"] for sp in lst) + (len(lst) - 1)
            if side in vertical:
                cur_h = max(cur_h, used_len)
            else:
                cur_w = max(cur_w, used_len)
        if not ok:
            _REJECTS["multi-side seam"] += 1
            continue
        max_depth = {s_: max(sp["depth"] for sp in specs[s_]) for s_ in sides}

        def run_need(side):
            return sum(sp["span"] for sp in specs[side]) + (len(specs[side]) - 1)

        # exact margins (a seam may have come out shallower than its cap)
        m_rows = max([max_depth[s_] + 1 for s_ in ("top", "bottom") if s_ in specs] + [1])
        m_cols = max([max_depth[s_] + 1 for s_ in ("left", "right") if s_ in specs] + [1])
        need_h = max([2 * m_rows + run_need(s_) for s_ in vertical if s_ in specs] + [3])
        need_w = max([2 * m_cols + run_need(s_) for s_ in ("top", "bottom") if s_ in specs] + [3])
        need_h = max(need_h, max_depth.get("top", 0) + max_depth.get("bottom", 0) + 2)
        need_w = max(need_w, max_depth.get("left", 0) + max_depth.get("right", 0) + 2)
        if need_h > h_cap or need_w > w_cap:      # cannot happen by construction; keep the guard
            _REJECTS["multi-side body caps"] += 1
            continue
        body_h = min(h_cap, need_h + int(rng.integers(0, 3)))
        body_w = min(w_cap, need_w + int(rng.integers(0, 3)))
        if cell_budget is not None and body_h * body_w > cell_budget:
            body_h, body_w = need_h, need_w
        # --- 4. carve the runs along each side (random wall gaps, away from the corners)
        anchor = {(r, c) for r in range(body_h) for c in range(body_w)}
        loose: list[dict] = []
        for side, lst in specs.items():
            L = body_w if side in vertical else body_h            # outward extent
            length = body_h if side in vertical else body_w       # run axis length
            m = m_rows if side in vertical else m_cols
            free = length - 2 * m - run_need(side)
            gaps = [0] * (len(lst) + 1)
            for _ in range(free):
                gaps[int(rng.integers(0, len(gaps)))] += 1
            pos = m + gaps[0]
            for i, sp in enumerate(lst):
                p, o = sp["seam"]["p"], sp["o"]
                max_o = max(o)
                cells = set()
                for j in range(sp["span"]):
                    d = sp["depth"] - p[j]
                    for u in range(L - d, L):                     # carve the cavity
                        anchor.discard(_side_map(side, u, pos + j, L))
                    top_u = L - 1 - sp["extra"] + sp["t"] + max_o - o[j]
                    for u in range(L - d, top_u + 1):             # the loose piece
                        cells.add(_side_map(side, u, pos + j, L))
                loose.append({"cells": cells, "span": sp["span"], "relief": sp["relief"],
                              "style": sp["seam"]["style"], "outer": sp["outer"],
                              "thickness": sp["t"], "protrusion": sp["protrusion"], "side": side})
                pos += sp["span"] + 1 + gaps[i + 1]
        if cell_budget is not None and len(anchor) > cell_budget:
            _REJECTS["multi-side cell budget"] += 1
            return None
        if any(len(p["cells"]) < 2 for p in loose):
            _REJECTS["multi-side tiny piece"] += 1
            continue
        comps = connected_components(_mask_of(anchor), connectivity=4, background=0)
        if len(comps) != 1:
            _REJECTS["multi-side anchor split"] += 1
            continue
        # --- 5. local uniqueness over all four directions
        if any(len(_all_fits(p["cells"], [anchor], anchor, None, None)) != 1 for p in loose):
            _REJECTS["multi-side local non-unique"] += 1
            continue
        # docked pieces must not touch each other (8-neighbourhood)
        occ = {}
        clash = False
        for idx, p in enumerate(loose):
            for r, c in p["cells"]:
                occ[(r, c)] = idx
        for (r, c), idx in occ.items():
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    j = occ.get((r + dr, c + dc))
                    if j is not None and j != idx:
                        clash = True
        if clash:
            _REJECTS["multi-side docked touch"] += 1
            continue
        used_keys |= keys
        all_cells = set(anchor)
        for p in loose:
            all_cells |= p["cells"]
        r0, c0, r1, c1 = _bbox(all_cells)
        anchor_n = _shift(anchor, -r0, -c0)
        for p in loose:
            p["cells"] = _shift(p["cells"], -r0, -c0)
        return {
            "anchor_rows": _intervals(anchor_n), "anchor_cells": anchor_n, "loose": loose,
            "loose_cells": [p["cells"] for p in loose], "k": sum(k_per_side),
            "outer": "straight", "thickness": min(body_h, body_w),
            "walls": 0, "h": r1 - r0 + 1, "w": c1 - c0 + 1, "unit_w": c1 - c0 + 1,
            "sides": list(sides), "free_sides": [s_ for s_ in _SIDES if s_ not in sides],
            "body": (body_h, body_w),
        }
    return None


def _mask_of(cells: set) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(cells)
    m = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=np.int64)
    for r, c in cells:
        m[r - r0, c - c0] = 1
    return m


def _build_units(rng, comp: list[int], h_caps: list[int], w_cap: int, span_cap: int, min_span: int,
                 min_relief: int, rect_union: bool, ta_cap: int, tl_cap: int,
                 relief_hi: int, budgets: list[int] | None = None,
                 side_plans: list[tuple[list[str], list[int]] | None] | None = None,
                 w_cap_multi: int | None = None) -> list[dict] | None:
    """Build every anchor unit; a loose piece must not seat (in free space, from any direction)
    against any other anchor, otherwise the offending unit is rebuilt.  ``budgets`` = per-unit
    anchor cell budgets; ``side_plans[i]`` = (sides, sites per side) for a multi-side anchor,
    None for the single-side builder."""
    used_keys: set = set()
    units: list[dict] = []
    for idx, (k, h_cap) in enumerate(zip(comp, h_caps)):
        budget = budgets[idx] if budgets is not None else None
        plan = side_plans[idx] if side_plans is not None else None
        for _ in range(_UNIT_TRIES):
            if plan is None:
                unit = _build_unit(rng, k, h_cap, w_cap, span_cap, min_span, min_relief, rect_union,
                                   used_keys, ta_cap, tl_cap, relief_hi, budget)
            else:
                unit = _build_multi_unit(rng, plan[0], plan[1], h_cap, w_cap_multi or w_cap, span_cap,
                                         min_span, min_relief, used_keys, tl_cap, relief_hi, budget)
            if unit is None:
                continue
            clash = False
            for prev in units:
                if any(_all_fits(lc, [prev["anchor_cells"]], prev["anchor_cells"], None, None)
                       for lc in unit["loose_cells"]):
                    clash = True
                    break
                if any(_all_fits(lc, [unit["anchor_cells"]], unit["anchor_cells"], None, None)
                       for lc in prev["loose_cells"]):
                    clash = True
                    break
            if not clash:
                units.append(unit)
                break
        else:
            return None
    return units


# ---------------------------------------------------------------------------
# extra loose pieces (seat nowhere) — the "leftovers" of the September-2026 follow-up
# ---------------------------------------------------------------------------

def _normalise(cells: set) -> set:
    r0, c0, _, _ = _bbox(cells)
    return _shift(cells, -r0, -c0)


def _connected(cells: set) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in _4DIRS:
            n = (r + dr, c + dc)
            if n in cells and n not in seen:
                seen.add(n)
                stack.append(n)
    return len(seen) == len(cells)


def _perturb(rng, cells: set) -> set | None:
    """A near-miss of a real loose piece: one boundary cell removed (piece stays connected) or
    one cell added on its outline."""
    cells = set(cells)
    if rng.random() < 0.5 and len(cells) > 2:
        cand = sorted(x for x in cells if any((x[0] + dr, x[1] + dc) not in cells for dr, dc in _4DIRS))
        for i in rng.permutation(len(cand)):
            trial = cells - {cand[int(i)]}
            if len(trial) >= 2 and _connected(trial):
                return _normalise(trial)
        return None
    ring = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in _4DIRS} - cells)
    return _normalise(cells | {ring[int(rng.integers(0, len(ring)))]})


def _jagged_extra(rng, span_cap: int, tl_cap: int, relief_hi: int, used_keys: set) -> set | None:
    """A fresh jagged piece drawn from the seam vocabulary (its seam matches no cavity)."""
    span = _sample_span(rng, max(2, min(span_cap, 9)), 2, wide=False)
    seam = _make_seam(rng, span, max(1, min(relief_hi, 4)), 1, used_keys)
    if seam is None:
        return None
    t = _thick_draw(rng, _LOOSE_THICK_WEIGHTS, 1, max(1, min(tl_cap, 4)))
    p = seam["p"]
    right = max(p) + t - 1
    if rng.random() < 0.5:
        o = _outer_profile(rng, span, _wchoice(rng, _OUTER_KINDS, _OUTER_WEIGHTS), 1, trim=True)
    else:
        o = [0] * span
    rows = [(j, p[j], right + max(o) - o[j]) for j in range(span)]
    return _normalise(_cells_of(rows))


def _build_extras(rng, units: list[dict], n_extra: int, open_sites: set, span_cap: int, tl_cap: int,
                  relief_hi: int, hc: int, wc: int, used_keys: set) -> list[dict] | None:
    """*n_extra* loose pieces that seat flush-and-jagged NOWHERE: against no fixed piece, from no
    direction, anywhere in free space (exhaustive ``_all_fits`` per anchor, unbounded canvas).
    Kinds: a near-miss of a real loose piece (preferably of an open cavity's piece) or a fresh
    jagged piece; at least four cells and never a plain rectangle (a rectangle cannot seat under
    the jaggedness rule, so it would give itself away).  None when no candidate survives."""
    anchors = [u["anchor_cells"] for u in units]
    sources = [(ui, si) for ui, u in enumerate(units) for si in range(len(u["loose"]))]
    open_sources = [src for src in sources if src in open_sites]
    extras: list[dict] = []
    for _ in range(n_extra):
        for _try in range(_EXTRA_TRIES):
            kind = _wchoice(rng, _EXTRA_KINDS, _EXTRA_KIND_WEIGHTS)
            if kind == "near_miss":
                pool = open_sources if (open_sources and rng.random() < 0.6) else sources
                ui, si = pool[int(rng.integers(0, len(pool)))]
                cells = _perturb(rng, units[ui]["loose"][si]["cells"])
            else:
                cells = _jagged_extra(rng, span_cap, tl_cap, relief_hi, used_keys)
            if cells is None or len(cells) < _EXTRA_MIN_CELLS:
                continue
            r0, c0, r1, c1 = _bbox(cells)
            if len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1):
                continue                                   # a plain rectangle: never seats, cheap
            if r1 - r0 + 1 > hc - 2 or c1 - c0 + 1 > wc - 2:
                continue
            if any(_all_fits(cells, [a], a, None, None) for a in anchors):
                _REJECTS["extra seats somewhere"] += 1
                continue
            if any(cells == e["cells"] for e in extras):
                continue
            extras.append({"cells": cells, "kind": kind, "rows": _intervals(cells),
                           "h": r1 - r0 + 1, "w": c1 - c0 + 1})
            break
        else:
            return None
    return extras


# ---------------------------------------------------------------------------
# scene assembly + validation
# ---------------------------------------------------------------------------

def _place_scene(rng, units: list[dict], hc: int, wc: int, border_rule: bool,
                 hug: bool, open_sites: frozenset = frozenset(), extras: list[dict] | None = None) -> dict | None:
    """Place anchors (with reserved docking footprints) then scattered loose pieces, all with a
    one-cell 8-neighbourhood clearance.  Under the border rule every anchor is flush with the
    grid border on one of its free (non-docking) sides and no loose piece touches any border.
    ``hug`` = prefer border-touching positions.  ``open_sites`` (anchor index, site index) keep
    their docking footprint reserved but get no scattered piece; ``extras`` are scattered like
    loose pieces (positions in ``extra_pos``)."""
    blocked = np.zeros((hc, wc), dtype=bool)
    anchor_pos: list[tuple[int, int]] = []
    flush_sides: list[str | None] = []
    for unit in units:
        footprint = set(unit["anchor_cells"])
        for piece in unit["loose"]:
            footprint |= piece["cells"]
        rows = _intervals(footprint)
        h, w = unit["h"], unit["unit_w"]
        flush = None
        if border_rule:
            free = list(unit["free_sides"])
            flush = "left" if "left" in free else free[int(rng.integers(0, len(free)))]
            if flush == "left":
                pos = _pick_position(rng, blocked, rows, h, w, fixed_left=0, hug=hug)
            else:
                pos = _pick_flush(rng, blocked, rows, h, w, flush, hug)
        else:
            pos = _pick_position(rng, blocked, rows, h, w, hug=hug)
        if pos is None:
            return None
        anchor_pos.append(pos)
        flush_sides.append(flush)
        mask = np.zeros((hc, wc), dtype=bool)
        for r, lo, hi in rows:
            mask[pos[0] + r, pos[1] + lo:pos[1] + hi + 1] = True
        blocked |= _dilate(mask)
    if border_rule:
        # loose pieces never touch any border
        blocked[0, :] = True
        blocked[-1, :] = True
        blocked[:, 0] = True
        blocked[:, -1] = True
    order = [(ui, si) for ui, unit in enumerate(units) for si in range(len(unit["loose"]))
             if (ui, si) not in open_sites]
    order += [(-1, i) for i in range(len(extras or []))]
    rng.shuffle(order)

    def _size(k):
        return len(extras[k[1]]["cells"]) if k[0] < 0 else len(units[k[0]]["loose_cells"][k[1]])
    order.sort(key=lambda k: -_size(k))  # largest first (stable)
    placed: dict[tuple[int, int], tuple[int, int]] = {}
    extra_pos: dict[int, tuple[int, int]] = {}
    for ui, si in order:
        if ui < 0:
            norm, h, w = extras[si]["rows"], extras[si]["h"], extras[si]["w"]
        else:
            piece = units[ui]["loose"][si]
            r0, c0, r1, c1 = _bbox(piece["cells"])
            norm = _intervals(_shift(piece["cells"], -r0, -c0))
            h, w = r1 - r0 + 1, c1 - c0 + 1
        pos = _pick_position(rng, blocked, norm, h, w, hug=hug)
        if pos is None:
            return None
        if ui < 0:
            extra_pos[si] = pos
        else:
            placed[(ui, si)] = pos
        mask = np.zeros((hc, wc), dtype=bool)
        for r, lo, hi in norm:
            mask[pos[0] + r, pos[1] + lo:pos[1] + hi + 1] = True
        blocked |= _dilate(mask)
    return {"anchor_pos": anchor_pos, "loose_pos": placed, "flush_sides": flush_sides, "extra_pos": extra_pos}


def _pick_flush(rng, blocked: np.ndarray, rows, h: int, w: int, side: str, hug: bool):
    """Position with the footprint's bounding box flush against the named grid border."""
    hc, wc = blocked.shape
    if side == "right":
        col = wc - w
        if col < 0:
            return None
        return _pick_position(rng, blocked, rows, h, w, fixed_left=col, hug=hug)
    nh, nw = hc - h + 1, wc - w + 1
    if nh <= 0 or nw <= 0:
        return None
    row = 0 if side == "top" else hc - h
    # try every column at that row, in random order
    cols = rng.permutation(nw)
    for c in cols:
        c = int(c)
        ok = True
        for r, lo, hi in rows:
            if blocked[row + r, c + lo:c + hi + 1].any():
                ok = False
                break
        if ok:
            return row, c
    return None


def _identify_background(grid: np.ndarray) -> tuple[int | None, float]:
    """The background as a solver must find it (nothing names it): after setting aside isolated
    single cells (texture), the most frequent colour, which must beat every other colour by
    ``BG_MARGIN`` and own the largest 4-connected region.  Returns (colour or None, margin)."""
    keep = ~bglib.isolated_cell_mask(grid)
    vals, counts = np.unique(grid[keep], return_counts=True)
    if len(vals) == 0:
        return None, 0.0
    order = np.argsort(-counts)
    bg = int(vals[order[0]])
    top = int(counts[order[0]])
    runner = int(counts[order[1]]) if len(vals) > 1 else 0
    margin = top / runner if runner else float("inf")
    if runner and top < BG_MARGIN * runner:
        return None, margin
    # the background must own the largest 4-connected same-colour region
    largest_bg = max((len(c["cells"]) for c in connected_components(grid, connectivity=4, background=-1)
                      if c["color"] == bg), default=0)
    largest_other = max((len(c["cells"]) for c in connected_components(grid, connectivity=4, background=bg)),
                        default=0)
    if largest_bg <= largest_other:
        return None, margin
    return bg, margin


def _touching_pieces(inp: np.ndarray, bg: int, comps: list[dict]) -> bool:
    """True when two different pieces are 8-adjacent in the input."""
    ids = np.full(inp.shape, -1, dtype=np.int32)
    for idx, comp in enumerate(comps):
        for r, c in comp["cells"]:
            ids[r, c] = idx
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a = ids[max(0, -dr):inp.shape[0] - max(0, dr), max(0, -dc):inp.shape[1] - max(0, dc)]
        b = ids[max(0, dr):, max(0, dc):]
        b = b[:a.shape[0], :a.shape[1]]
        if np.any((a >= 0) & (b >= 0) & (a != b)):
            return True
    return False


def _validate(inp: np.ndarray, out: np.ndarray, bg: int, rule: str, anchor_color: int | None,
              n_anchors: int, n_loose: int, n_stay: int = 0) -> bool:
    """Independent re-solve from the visible canonical input: identify pieces and fixed pieces
    by the stated rule, enumerate every legal docking of every loose piece, require exactly
    one per loose piece (none for exactly *n_stay* of them, which stay where they are) and
    that the unique assembly equals *out*."""
    found, _margin = _identify_background(inp)
    if found != bg:
        _REJECTS["background not identifiable"] += 1
        return False
    # isolated single cells are background texture (the prompt's isolated-cell rule)
    comps = [c for c in connected_components(inp, connectivity=4, background=bg) if len(c["cells"]) >= 2]
    if _touching_pieces(inp, bg, comps):
        return False
    h, w = inp.shape
    if rule == "border":
        anchors = [c for c in comps if _touches_border(c["cells"], h, w)]
    else:
        anchors = [c for c in comps if c["color"] == anchor_color]
    loose = [c for c in comps if not any(c is a for a in anchors)]
    if len(anchors) != n_anchors or len(loose) != n_loose:
        return False
    return _assemble(inp, out, bg, anchors, loose, n_stay=n_stay)


def _touches_border(cells: set, h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in cells)


def _assemble(inp: np.ndarray, out: np.ndarray, bg: int, anchors: list[dict], loose: list[dict],
              n_stay: int | None = None) -> bool:
    """Seat every loose piece: over all four directions of approach and all fixed pieces there
    must be at most one flush, jagged placement; a piece with none stays exactly where it is
    (when *n_stay* is given, exactly that many must stay); placements may not overlap each
    other, a fixed piece or a staying piece; the result (texture cells stay, vacated cells become
    plain background) must equal *out*."""
    h, w = inp.shape
    all_anchor: set = set()
    for a in anchors:
        all_anchor |= a["cells"]
    anchor_sets = [a["cells"] for a in anchors]
    seats: list[frozenset | None] = []
    for piece in loose:
        placements = _all_fits(piece["cells"], anchor_sets, all_anchor, h, w)
        if len(placements) > 1:
            return False
        seats.append(next(iter(placements)) if placements else None)
    stays = [piece for piece, seat in zip(loose, seats) if seat is None]
    if n_stay is not None and len(stays) != n_stay:
        return False
    if len(stays) == len(loose):
        return False                     # nothing moves: not a join-the-pieces example
    result = inp.copy()
    occupied: set = set(all_anchor)
    for piece in stays:
        occupied |= set(piece["cells"])
    for piece, seat in zip(loose, seats):
        if seat is None:
            continue
        for r, c in piece["cells"]:
            result[r, c] = bg
    for piece, seat in zip(loose, seats):
        if seat is None:
            continue
        if seat & occupied:
            return False
        occupied |= seat
        for r, c in seat:
            result[r, c] = piece["color"]
    return bool(np.array_equal(result, out))


def resolve_from_prompt(inp: np.ndarray, out: np.ndarray, dock: str | None, rule: str,
                        anchor_color: int | None) -> bool:
    """Independent re-solve written from the prompt's wording and nothing else (``dock`` is
    accepted for API compatibility and ignored: the wording is side-agnostic).  The background
    is the colour a solver must infer (most frequent after the isolated texture cells are set
    aside, by a clear margin, owning the largest region); pieces are 4-connected same-colour
    groups of at least two cells (isolated cells are texture); fixed pieces touch the grid
    border, or carry the named colour; a loose piece seats flush against a fixed piece arriving
    from the left, the right, above or below: every facing cell has a fixed cell directly
    beyond it, the seam is jagged, the piece stays inside the grid and overlaps no fixed cell;
    a loose piece that seats nowhere stays exactly where it is; exactly one legal assembly must
    exist and equal *out*."""
    del dock
    H, W = inp.shape
    bg, _margin = _identify_background(inp)
    if bg is None:
        return False
    comps = [c for c in connected_components(inp, connectivity=4, background=bg) if len(c["cells"]) >= 2]
    if rule == "border":
        anchors = [c for c in comps if _touches_border(c["cells"], H, W)]
    else:
        anchors = [c for c in comps if c["color"] == anchor_color]
    anchor_ids = {id(a) for a in anchors}
    loose = [c for c in comps if id(c) not in anchor_ids]
    if not anchors or not loose:
        return False
    return _assemble(inp, out, bg, anchors, loose)


def _orient(grid: np.ndarray, dock: str) -> np.ndarray:
    if dock == "right":
        return grid.copy()
    if dock == "left":
        return np.fliplr(grid).copy()
    if dock == "top":
        return np.rot90(grid, k=1).copy()
    return np.rot90(grid, k=3).copy()


# where each canonical side ends up under _orient(grid, dock)
_SIDE_MAP = {
    "right": {"right": "right", "left": "left", "top": "top", "bottom": "bottom"},
    "left": {"right": "left", "left": "right", "top": "top", "bottom": "bottom"},
    "top": {"right": "top", "top": "left", "left": "bottom", "bottom": "right"},
    "bottom": {"right": "bottom", "bottom": "left", "left": "top", "top": "right"},
}


def _paint_cells(grid: np.ndarray, cells: set, top: int, left: int, color: int) -> None:
    for r, c in cells:
        grid[top + r, left + c] = color


def _choose_sides(rng, k: int, area: int, rule: str, k_cap_side: int) -> tuple[list[str], list[int]] | None:
    """For an anchor with *k* sites: the docking sides and the sites per side, or None for the
    single-side builder.  Two-side anchors from ~200 cells, three- and four-side from ~500;
    four sides need the colour rule (no free side can touch the border)."""
    if k < 2 or area < 200:
        return None
    if area < 500:
        n_sides = 2 if rng.random() < 0.45 else 1
    else:
        n_sides = (_wchoice(rng, [1, 2, 3, 4], [0.12, 0.26, 0.14, 0.48]) if k >= 4
                   else _wchoice(rng, [1, 2, 3], [0.18, 0.44, 0.38]) if k == 3
                   else _wchoice(rng, [1, 2], [0.45, 0.55]))
    n_sides = min(n_sides, k)
    # more sites than one side may carry: spread them over enough sides
    n_sides = max(n_sides, -(-k // k_cap_side))
    if rule == "border" and n_sides == 4:
        n_sides = 3
    if n_sides < 2 or n_sides > 4:
        return None
    if n_sides == 2 and rng.random() < (0.85 if area < 350 else 0.5 if area < 500 else 0.3):
        # opposite sides need no corner clearance, so they are what small canvases can hold
        pair = ["left", "right"] if rng.random() < 0.5 else ["top", "bottom"]
        sides = pair if rng.random() < 0.5 else pair[::-1]
    else:
        sides = [str(s) for s in rng.choice(list(_SIDES), size=n_sides, replace=False)]
    # distribute the sites: every side at least one, none above the per-side cap
    if k > n_sides * k_cap_side:
        return None
    per = [1] * n_sides
    for _ in range(k - n_sides):
        options = [i for i in range(n_sides) if per[i] < k_cap_side]
        per[int(rng.choice(options))] += 1
    return sides, per


# ---------------------------------------------------------------------------
# generator
# ---------------------------------------------------------------------------

@register("13")
def generate_join_pieces(rng: np.random.Generator) -> dict:
    """Task 13: translate each loose jagged piece onto the fixed piece / cavity it complements."""
    canvas: tuple[int, int] | None = None
    canvas_uses = 0
    for _attempt in range(MAX_CANVAS_ATTEMPTS):
        # a failed build is retried on the SAME canvas a few times before the size is
        # resampled, so rejections (small canvases fail more) do not tilt the size field
        if canvas is None or canvas_uses >= 3:
            canvas = _sample_canvas(rng)
            canvas_uses = 0
        canvas_uses += 1
        H, W = canvas
        dock = str(rng.choice(["right", "left", "top", "bottom"]))
        hc, wc = (H, W) if dock in ("right", "left") else (W, H)
        rule = "border" if rng.random() < 0.5 else "color"
        area = hc * wc
        n_max = min(6, max(1, area // 44))
        if rule == "border":  # fixed pieces stack in one column: ~4 rows per site
            n_max = min(n_max, max(1, (hc + 1) // 4))
        site_weights = _SITE_COUNT_WEIGHTS_LARGE if area >= 400 else _SITE_COUNT_WEIGHTS
        n_sites = _wchoice(rng, list(range(1, n_max + 1)), site_weights[:n_max])
        anchor_cap = (hc + 1) // 4 if rule == "border" else max(1, min(6, area // 55))
        # sites need rows: a host with several cavities on a short canvas can only hold 3-row
        # sites, so the sites one anchor may carry follow the canvas height (about seven rows
        # per site; two sites from eight rows); pairs of single-site anchors are not limited
        k_cap = max(1, (hc + 2) // 7)
        if k_cap == 1 and hc >= 8:
            k_cap = 2
        # ... and area: a host with three or four cavities of decent width cannot stay a
        # minority of a canvas under ~550 / ~800 cells
        k_cap = min(k_cap, 2 if area < 550 else 3 if area < 800 else 4)
        # cavities may be spread over several sides of one anchor (from ~200 cells), so an
        # anchor may carry up to twice the per-side cap, four at most
        k_cap_total = min(4, 2 * k_cap) if area >= 200 else k_cap
        options = [(c, w_) for c, w_ in _COMPOSITIONS[n_sites] if len(c) <= anchor_cap and max(c) <= k_cap_total]
        if area >= 500:   # large canvases: lean to anchors with three or four cavities (multi-side hosts)
            options = [(c, w_ * (1.6 if max(c) >= 3 else 1.0)) for c, w_ in options]
        if not options:
            continue
        comp = list(_wchoice(rng, [c for c, _ in options], [w_ for _, w_ in options]))
        rng.shuffle(comp)
        n_anchors = len(comp)
        side_plans = [_choose_sides(rng, k, area, rule, k_cap) for k in comp]
        if any(k > k_cap and plan is None for k, plan in zip(comp, side_plans)):
            continue
        rect_union = rng.random() < 0.25
        min_relief = 2 if n_sites == 1 else 1
        min_span = 3 if n_sites >= 3 and all(p is None for p in side_plans) else 2

        # size caps scale with the canvas area each site can claim; every failed draw shrinks them
        scene = None
        units: list[dict] | None = None
        site_scale = (area / n_sites) ** 0.5
        for draw in range(_PARAM_TRIES):
            scale = site_scale * (1.0 - 0.12 * draw)
            span_cap = max(min_span, min(18, int(round(1.0 * scale))))
            w_cap = max(5, min(wc - 1, int(round(1.7 * scale))))
            ta_cap = max(1, int(round(0.4 * scale)))
            tl_cap = max(1, int(round(0.4 * scale)))
            relief_hi = max(min_relief, min(8, int(round(0.6 * scale))))
            if rule == "border":
                total = hc - (n_anchors - 1)
                h_caps = [max(k * min_span + k - 1, int(total * k / n_sites * (1.0 - 0.1 * draw))) for k in comp]
            else:
                h_caps = [hc if n_anchors == 1 else max(k * min_span + k - 1, min(hc, int(round(1.3 * scale * k ** 0.6))))
                          for k in comp]
            # multi-side anchors: body caps leave room for protrusions on every side
            for i, plan in enumerate(side_plans):
                if plan is not None:
                    k = comp[i]
                    h_caps[i] = min(hc - 4, max(6, int(round(1.4 * scale * k ** 0.5))))
            w_cap_multi = min(wc - 4, max(6, int(round(1.4 * scale))))
            # anchor cell budgets: the fixed pieces together stay under FIXED_SHARE_CAP of the
            # canvas (with a little headroom), shared between anchors by their site counts
            budgets = [max(6, int(0.9 * FIXED_SHARE_CAP * area * k / n_sites)) for k in comp]
            try:
                units = _build_units(rng, comp, h_caps, w_cap, span_cap, min_span, min_relief, rect_union,
                                     ta_cap, tl_cap, relief_hi, budgets, side_plans=side_plans,
                                     w_cap_multi=w_cap_multi)
            except _CapsInfeasible:
                units = None
                break
            if units is None:
                _REJECTS[f"units draw {draw}"] += 1
                continue
            # the fixed pieces may not dominate the canvas (the background must stay the
            # obvious majority for a solver who is told nothing about it)
            if sum(len(u["anchor_cells"]) for u in units) > FIXED_SHARE_CAP * area:
                _REJECTS[f"fixed share draw {draw}"] += 1
                continue
            # --- leftovers: cavities that stay open and loose pieces that seat nowhere
            all_sites = [(ui, si) for ui, unit in enumerate(units) for si in range(unit["k"])]
            open_sites: set = set()
            if rng.random() < _OPEN_P and n_sites >= 2:
                n_open = 2 if (n_sites >= 4 and rng.random() < 0.3) else 1
                for i in rng.permutation(len(all_sites))[:n_open]:
                    open_sites.add(all_sites[int(i)])
            extras: list[dict] = []
            if rng.random() < _EXTRA_P:
                n_extra = 2 if (area >= 150 and rng.random() < 0.3) else 1
                built = _build_extras(rng, units, n_extra, open_sites, span_cap, tl_cap, relief_hi,
                                      hc, wc, set())
                if built is None:
                    _REJECTS[f"extras draw {draw}"] += 1
                    continue
                extras = built
            for q in range(_PLACE_TRIES):
                scene = _place_scene(rng, units, hc, wc, rule == "border", hug=q >= 1,
                                     open_sites=frozenset(open_sites), extras=extras)
                if scene is not None:
                    break
            if scene is not None:
                break
            _REJECTS[f"scene draw {draw}"] += 1
        if scene is None or units is None:
            continue
        n_open = len(open_sites)
        n_extra = len(extras)
        n_docked = n_sites - n_open

        # --- colours: near-flat targets for the number of piece colours; anchors and loose
        # pieces draw from disjoint sets (the colour rule needs it; the border rule keeps it)
        bg = int(rng.integers(0, NUM_COLORS))
        pool = [c for c in range(NUM_COLORS) if c != bg]
        rng.shuffle(pool)
        n_loose = n_docked + n_extra          # loose pieces visible in the input
        n_slots = n_sites + n_extra           # colour slots: one per cavity (open ones unused) + extras
        anchor_color: int | None = None
        if rule == "color":
            anchor_color = pool[0]
            k = _flat_count(rng, 1, min(n_loose, 8))
            anchor_colors = [anchor_color] * n_anchors
            loose_colors = _cycle_colors(rng, pool[1:1 + k], n_slots)
            scheme = "uniform_loose" if k == 1 else "distinct"
        else:
            a = _flat_count(rng, 1, min(n_anchors, 8))
            l_ = _flat_count(rng, 1, min(n_loose, 9 - a))
            anchor_colors = _cycle_colors(rng, pool[:a], n_anchors)
            loose_colors = _cycle_colors(rng, pool[a:a + l_], n_slots)
            scheme = "two_tone" if (a == 1 and l_ == 1) else "distinct"
        extra_colors = [int(c) for c in loose_colors[n_sites:]]

        # --- paint canonical input / output
        inp = np.full((hc, wc), bg, dtype=np.int64)
        out = np.full((hc, wc), bg, dtype=np.int64)
        site_idx = 0
        site_color: dict[tuple[int, int], int] = {}
        for ui, unit in enumerate(units):
            top, left = scene["anchor_pos"][ui]
            _paint_cells(inp, unit["anchor_cells"], top, left, anchor_colors[ui])
            _paint_cells(out, unit["anchor_cells"], top, left, anchor_colors[ui])
            for si, piece in enumerate(unit["loose"]):
                site_color[(ui, si)] = loose_colors[site_idx]
                if (ui, si) not in open_sites:
                    _paint_cells(out, piece["cells"], top, left, loose_colors[site_idx])
                site_idx += 1
        for (ui, si), (top, left) in scene["loose_pos"].items():
            piece = units[ui]["loose"][si]
            r0_, c0_, _, _ = _bbox(piece["cells"])
            _paint_cells(inp, _shift(piece["cells"], -r0_, -c0_), top, left, site_color[(ui, si)])
        for i, extra in enumerate(extras):          # extras stay: same cells in input and output
            top, left = scene["extra_pos"][i]
            _paint_cells(inp, extra["cells"], top, left, extra_colors[i])
            _paint_cells(out, extra["cells"], top, left, extra_colors[i])
        open_mask = np.zeros((hc, wc), dtype=bool)   # the reserved footprint of every open cavity
        for (ui, si) in open_sites:
            top, left = scene["anchor_pos"][ui]
            for r, c in units[ui]["loose"][si]["cells"]:
                open_mask[top + r, left + c] = True
        used_loose = sorted({site_color[k] for k in scene["loose_pos"]} | set(extra_colors))

        # --- optional sparse texture (isolated cells) on free cells at least one cell away
        # from every piece and every docking footprint; identical in input and output
        texture = dict(_NO_TEXTURE)
        if min(hc, wc) >= 8 and rng.random() < _TEXTURE_P:
            occupied = _dilate((inp != bg) | (out != bg) | open_mask)
            painted = bglib.apply_background(rng, inp, bg, occupied, group="sparse")
            if painted is not None:
                display, info = painted
                tex_mask = display != inp
                inp = display
                out = out.copy()
                out[tex_mask] = display[tex_mask]
                texture = {"texture_group": info["group"], "texture_kind": info["kind"],
                           "texture_colors": [int(c) for c in info["texture_colors"]],
                           "texture_named": False, "texture_coverage": float(info["coverage"])}

        if not _validate(inp, out, bg, rule, anchor_color, n_anchors, n_loose, n_stay=n_extra):
            continue

        # --- orientation + metadata
        inp_o, out_o = _orient(inp, dock), _orient(out, dock)
        assert inp_o.shape == (H, W)
        if not resolve_from_prompt(inp_o, out_o, dock, rule, anchor_color):
            raise RuntimeError("Task 13: the prompt-side re-solver disagrees with the constructed answer")
        _bg_found, bg_margin = _identify_background(inp_o)
        sites_per_anchor = [unit["k"] for unit in units]
        if n_anchors == 1 and sites_per_anchor[0] >= 2:
            variant = "host"
        elif max(sites_per_anchor) == 1:
            variant = "pairs"
        else:
            variant = "mixed"
        pieces = [p for unit in units for p in unit["loose"]]
        smap = _SIDE_MAP[dock]
        sides_per_anchor = [sorted(smap[s] for s in unit["sides"]) for unit in units]
        site_keys = [(ui, si) for ui, unit in enumerate(units) for si in range(len(unit["loose"]))]
        site_open = [key in open_sites for key in site_keys]
        open_cells = _orient(open_mask.astype(np.int64), dock)
        metadata: dict = {
            "variant": variant,
            "anchor_rule": rule,
            "bg_color": bg,
            "background_margin": float(round(bg_margin, 3)),
            "dock_edge": dock,
            "loose_edge": _OPPOSITE[dock],
            "fixed_side": _OPPOSITE[dock],
            "orientation": dock,
            "dock_sides_per_anchor": sides_per_anchor,
            "n_sides_per_anchor": [len(s) for s in sides_per_anchor],
            "max_sides": max(len(s) for s in sides_per_anchor),
            "multi_side": any(len(s) >= 2 for s in sides_per_anchor),
            "piece_sides": [smap[p["side"]] for p in pieces],
            "n_anchors": n_anchors,
            "n_loose": n_loose,
            "n_sites": n_sites,
            "n_docked": n_docked,
            "n_open_cavities": n_open,
            "site_open": site_open,
            "open_sites": [list(key) for key in sorted(open_sites)],
            "open_cavity_sides": [smap[pieces[i]["side"]] for i, is_open in enumerate(site_open) if is_open],
            "open_cavity_cells": [[int(r), int(c)] for r, c in np.argwhere(open_cells == 1).tolist()],
            "n_extra": n_extra,
            "extra_kinds": [e["kind"] for e in extras],
            "extra_sizes": [len(e["cells"]) for e in extras],
            "extra_piece_colors": extra_colors,
            "sites_per_anchor": sites_per_anchor,
            "spans": [p["span"] for p in pieces],
            "reliefs": [p["relief"] for p in pieces],
            "styles": [p["style"] for p in pieces],
            "loose_outer": [p["outer"] for p in pieces],
            "loose_thickness": [p["thickness"] for p in pieces],
            "protrusions": [p["protrusion"] for p in pieces],
            "anchor_outer": [unit["outer"] for unit in units],
            "anchor_thickness": [unit["thickness"] for unit in units],
            "anchor_walls": [unit["walls"] for unit in units],
            "rect_union": rect_union,
            "color_scheme": scheme,
            "anchor_palette": sorted(set(anchor_colors)),
            "loose_palette": used_loose,
            "grid_size": [H, W],
            "size_draw": int(draw),
            "caps": [int(span_cap), int(w_cap), int(ta_cap), int(tl_cap), int(relief_hi)],
            "bg_style": texture["texture_kind"] or "uniform",
            "n_colors": int(len(np.unique(inp_o))),
            "piece_colors": sorted(set(anchor_colors) | set(used_loose)),
            **texture,
        }
        if rule == "color":
            metadata["anchor_color"] = anchor_color
        return {
            "input": grid_to_list(inp_o),
            "output": grid_to_list(out_o),
            "metadata": metadata,
        }
    raise RuntimeError("Task 13: failed to generate a valid example")
