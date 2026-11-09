"""Regression checks for the largest-object family (tasks 17 and 18).

Task 19 (most cells under 8-connectivity) was merged into task 18 on 2026-09-07: task 18 now
draws the connectivity per example, like task 17, and the scene is drawn without looking at
it (see ``test_task18_scene_independent_of_connectivity``).

The oracle here is a plain BFS written independently of the generator and of
``common.topology`` / ``common.backgrounds``: it re-derives the winner from
the visible input grid and the prompted connectivity under BOTH readings of
the prompt (every non-background cell is an object cell; isolated single
cells are set aside first), and re-derives the background without being told
it (strict plurality by 1.25x after isolated cells are set aside, largest
4-connected region).  Everything else (size, colour and object-count
distributions, rule-matters shares, deliberate-only touching, sparse
textures, prompt policy) is measured on the same 2000-seed sweep per task.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, deque

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from registry import generate  # noqa: E402
from prompts import get_prompt, ISOLATED_CELL_RULE  # noqa: E402
from common.backgrounds import SPARSE_KINDS  # noqa: E402

SEEDS = 2000
TASKS = ("17", "18")
BG_MARGIN = 1.25
LETTERS = list("ABCDEFGHIJ")

# Byte-identity of other tasks is checked at pass level (every registered task
# is hashed against the pre-pass baseline), so there is no frozen sibling here.
SIBLING_HASHES_300: dict[str, str] = {}

CLASS_OF = {
    "polyomino": "polyomino", "scaled": "polyomino",
    "bar": "bar",
    "rect": "rect", "square": "rect",
    "frame": "frame",
    "L": "letter", "T": "letter", "U": "letter", "plus": "letter", "H": "letter", "staircase": "letter",
    "blob": "blob", "big_blob": "big_blob",
    "chain": "chain", "diagonal": "diagonal",
}
MAIN_CLASSES = ("polyomino", "bar", "rect", "frame", "letter", "blob")


# ---------------------------------------------------------------------------
# Independent oracle
# ---------------------------------------------------------------------------

def _components(grid: np.ndarray, conn: int, bg: int):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    d4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
    deltas = d4 if conn == 4 else d4 + ((1, 1), (1, -1), (-1, 1), (-1, -1))
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] == bg:
                continue
            col = int(grid[r, c])
            cells = []
            dq = deque([(r, c)])
            seen[r, c] = True
            while dq:
                a, b = dq.popleft()
                cells.append((a, b))
                for dr, dc in deltas:
                    na, nb = a + dr, b + dc
                    if 0 <= na < h and 0 <= nb < w and not seen[na, nb] and grid[na, nb] == col:
                        seen[na, nb] = True
                        dq.append((na, nb))
            comps.append((col, frozenset(cells)))
    return comps


def _isolated(grid: np.ndarray) -> np.ndarray:
    """Cells with no 8-neighbour of the same colour (the prompt's texture rule)."""
    h, w = grid.shape
    out = np.ones((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            v = grid[r, c]
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == v:
                        out[r, c] = False
    return out


def _infer_background(grid: np.ndarray):
    """(colour, margin, largest_region_colour) from the visible grid alone: isolated
    cells are set aside, then the plurality colour must beat the runner-up by 1.25x
    and own the largest 4-connected region."""
    iso = _isolated(grid)
    counts = Counter(int(v) for v in grid[~iso].ravel())
    ranked = counts.most_common()
    color, count = ranked[0]
    runner = ranked[1][1] if len(ranked) > 1 else 0
    margin = count / max(1, runner)
    best_size, best_col = 0, None
    for col in counts:
        for _, cells in _components(np.where(grid == col, 1, 0), 4, 0):
            if len(cells) > best_size:
                best_size, best_col = len(cells), col
    return color, margin, best_col


def _bbox_area(cells) -> int:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)


def _score_fn(task: str):
    return _bbox_area if task == "17" else len


def _top(comps, score):
    """(winner cells, top score, second score); winner is None when tied."""
    ranked = sorted(comps, key=lambda cc: -score(cc[1]))
    top = score(ranked[0][1])
    second = score(ranked[1][1]) if len(ranked) > 1 else 0
    return (None if top == second else ranked[0][1]), top, second


def _strip(grid: np.ndarray, bg: int, accents) -> np.ndarray:
    g = grid.copy()
    for a in accents:
        g[g == a] = bg
    return g


def _plain_reading(grid: np.ndarray, bg: int) -> np.ndarray:
    """Isolated single cells of any colour set aside (the prompt's reading)."""
    g = grid.copy()
    g[_isolated(grid) & (grid != bg)] = bg
    return g


def _touching_other_color(grid: np.ndarray, bg: int, skip) -> bool:
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            v = int(grid[r, c])
            if v == bg or v in skip:
                continue
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        u = int(grid[nr, nc])
                        if u != bg and u != v and u not in skip:
                            return True
    return False


def _area_band(area: int) -> str:
    return "small" if area < 100 else "mid" if area < 400 else "large"


# ---------------------------------------------------------------------------
# One sweep per task, shared by every test
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sweep():
    data = {}
    for tid in TASKS:
        rows = []
        for seed in range(SEEDS):
            t0 = time.perf_counter()
            ex = generate(tid, seed)
            ms = (time.perf_counter() - t0) * 1000.0
            rows.append((seed, ex, ms))
        data[tid] = rows
    return data


def _grid(ex):
    return np.array(ex["input"], dtype=int), np.array(ex["output"], dtype=int), ex["metadata"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tid", TASKS)
def test_oracle_resolves_every_example_under_both_readings(sweep, tid):
    for seed, ex, _ in sweep[tid]:
        g, o, m = _grid(ex)
        bg = int(m["background_color"])
        conn = int(m["connectivity"])
        assert conn in (4, 8) and str(conn) == m["connectivity"], seed
        assert g.shape == o.shape and 5 <= g.shape[0] <= 30 and 5 <= g.shape[1] <= 30, (seed, g.shape)
        assert g.min() >= 0 and g.max() <= 9 and o.min() >= 0 and o.max() <= 9, seed
        target = int(m["target_color"])
        assert target not in set(np.unique(g).tolist()), seed
        changed = frozenset((int(r), int(c)) for r, c in np.argwhere(o != g))
        # reading (a): every non-background cell is an object cell (texture ignored by nobody)
        # reading (b): isolated single cells of any colour are background texture
        for view in (g, _plain_reading(g, bg)):
            comps = _components(view, conn, bg)
            winner, top, second = _top(comps, _score_fn(tid))
            assert winner is not None and top - second >= 2, (seed, top, second)
            assert changed == winner, seed
        assert all(int(o[r, c]) == target for r, c in changed), seed
        # under the prompt's reading every object has at least two cells
        assert all(len(c) >= 2 for _, c in _components(_plain_reading(g, bg), conn, bg)), seed
        assert m["winner_cells"] == len(changed) and m["margin"] == top - second, seed
        assert m["runner_up_score"] == second, seed


@pytest.mark.parametrize("tid", TASKS)
def test_background_identifiable_without_being_named(sweep, tid):
    margins = []
    for seed, ex, _ in sweep[tid]:
        g, o, m = _grid(ex)
        bg = int(m["background_color"])
        # strict plurality on the raw grid too
        vals, counts = np.unique(g, return_counts=True)
        order = np.argsort(-counts)
        assert int(vals[order[0]]) == bg and (len(counts) == 1 or counts[order[0]] > counts[order[1]]), seed
        color, margin, region_color = _infer_background(g)
        assert color == bg and margin >= BG_MARGIN and region_color == bg, (seed, color, margin, region_color)
        assert abs(m["background_margin"] - margin) < 0.01 + 1e-9 or m["background_margin"] >= BG_MARGIN, seed
        margins.append(margin)
    assert min(margins) >= BG_MARGIN


@pytest.mark.parametrize("tid", TASKS)
def test_prompt_policy(sweep, tid):
    """Nothing about the background is named or defined; connectivity, the two-cell
    minimum, the isolated-cell rule and the target colour are stated."""
    for seed, ex, _ in sweep[tid]:
        m = ex["metadata"]
        conn = int(m["connectivity"])
        prompt = get_prompt(tid, m, color_palette=LETTERS)
        assert ("Ignore the background, whether it is a solid color or a pattern." in prompt
                and ISOLATED_CELL_RULE in prompt), seed
        assert "at least two cells" in prompt and f"{conn}-connected" in prompt, seed
        assert ("diagonally" in prompt) == (conn == 8), seed
        assert f"color {LETTERS[int(m['target_color'])]}" in prompt, seed
        # the only colour mentioned is the target (letters: no digit can leak in)
        named = re.findall(r"color ([A-J])", prompt)
        assert named == [LETTERS[int(m["target_color"])]], (seed, named)
        assert not re.search(r"background is|background color|background colour", prompt), seed
        assert not re.search(r"\b(dots|dashes|lattice|speckle|checker|decorat)", prompt), seed
        # the default palette names the same target with a digit
        assert f"color {m['target_color']}" in get_prompt(tid, m), seed


@pytest.mark.parametrize("tid", TASKS)
def test_deterministic(tid):
    for seed in (0, 1, 7, 42, 123, 999):
        a = generate(tid, seed)
        b = generate(tid, seed)
        assert a["input"] == b["input"] and a["output"] == b["output"] and a["metadata"] == b["metadata"], seed


@pytest.mark.parametrize("tid", TASKS)
def test_performance(sweep, tid):
    ms = np.array([row[2] for row in sweep[tid]])
    assert ms.mean() <= 8.0, ms.mean()
    assert np.percentile(ms, 99) <= 40.0, np.percentile(ms, 99)


@pytest.mark.parametrize("tid", TASKS)
def test_sizes_independent_and_near_flat(sweep, tid):
    hs = np.array([len(ex["input"]) for _, ex, _ in sweep[tid]])
    ws = np.array([len(ex["input"][0]) for _, ex, _ in sweep[tid]])
    areas = hs * ws
    assert hs.min() == 5 and ws.min() == 5 and hs.max() == 30 and ws.max() == 30
    # near-flat per-value density over 5..30 (expected 1/26 ~ 0.038), both axes
    for side in (hs, ws):
        shares = np.array([np.mean(side == v) for v in range(5, 31)])
        assert shares.min() >= 0.015 and shares.max() <= 0.07, shares
    # independent axes: no forced square share and no correlation
    assert np.mean(hs == ws) <= 0.08
    assert np.mean(np.abs(hs - ws) <= 2) <= 0.30
    assert abs(np.corrcoef(hs, ws)[0, 1]) <= 0.10
    # no area cap: 30x30 reachable, a fair share above the old 600 cap
    assert areas.max() == 900 and np.mean(areas > 600) >= 0.06  # uniform sides: 64/676 ~ 0.095
    assert np.mean(areas < 100) >= 0.08


@pytest.mark.parametrize("tid", TASKS)
def test_colours_near_flat_none_dominant(sweep, tid):
    n_in = Counter()
    n_out = Counter()
    for seed, ex, _ in sweep[tid]:
        g, o, m = _grid(ex)
        n_in[len(np.unique(g))] += 1
        n_out[len(np.unique(o))] += 1
        assert m["n_colors"] == len(np.unique(g)), seed
        assert set(m["object_colors"]) <= set(np.unique(g).tolist()), seed
        assert int(m["background_color"]) not in m["object_colors"], seed
    total = sum(n_in.values())
    shares = {k: v / total for k, v in n_in.items()}
    assert max(shares.values()) <= 0.30, shares
    assert all(shares.get(k, 0) >= 0.03 for k in range(3, 10)), shares
    assert n_in[9] > 0 and n_out[10] > 0, (n_in, n_out)


@pytest.mark.parametrize("tid", TASKS)
def test_scenes_scale_with_canvas(sweep, tid):
    n_by_band = {"small": [], "mid": [], "large": []}
    winner_by_band = {"small": [], "mid": [], "large": []}
    for seed, ex, _ in sweep[tid]:
        g, o, m = _grid(ex)
        band = _area_band(g.size)
        n_by_band[band].append(m["n_objects"])
        winner_by_band[band].append(m["winner_cells"])
        assert m["n_objects"] >= 2, seed
    assert np.median(n_by_band["small"]) <= 4
    assert np.median(n_by_band["mid"]) >= 5
    assert np.median(n_by_band["large"]) >= 9 and max(n_by_band["large"]) >= 14
    assert np.median(winner_by_band["large"]) >= 2 * np.median(winner_by_band["small"])
    assert max(winner_by_band["large"]) >= 40


@pytest.mark.parametrize("tid", TASKS)
def test_shape_class_coverage(sweep, tid):
    kinds = Counter()
    winner_kinds = Counter()
    for _, ex, _ in sweep[tid]:
        m = ex["metadata"]
        for k in m["shape_kinds"]:
            kinds[CLASS_OF[k]] += 1
        winner_kinds[m["winner_kind"]] += 1
    total = sum(kinds.values())
    for cls in MAIN_CLASSES:
        assert kinds[cls] / total >= 0.04, (cls, kinds)
    assert kinds["big_blob"] > 0 and kinds["chain"] > 0, kinds
    if tid == "18":
        # the scene never hints the connectivity: no diagonal-line family, chains only
        assert kinds["diagonal"] == 0, kinds
    else:
        assert kinds["diagonal"] > 0, kinds
    assert max(winner_kinds.values()) / SEEDS <= 0.45, winner_kinds
    assert len(winner_kinds) >= 10, winner_kinds


@pytest.mark.parametrize("tid", TASKS)
def test_rule_matters(sweep, tid):
    """Share of scenes where the prompted rule changes the answer, recomputed
    independently on the object-only grid, and consistent with metadata."""
    n_rule = 0
    n_conn = 0
    for seed, ex, _ in sweep[tid]:
        g, o, m = _grid(ex)
        bg = int(m["background_color"])
        conn = int(m["connectivity"])
        plain = _strip(g, bg, m["accent_colors"])
        winner = frozenset((int(r), int(c)) for r, c in np.argwhere(o != g))
        alt_winner, _, _ = _top(_components(plain, 8 if conn == 4 else 4, bg), _score_fn(tid))
        conn_matters = alt_winner != winner
        cells_winner, _, _ = _top(_components(plain, conn, bg), len)
        bbox_matters = cells_winner != winner
        rule = bbox_matters if tid == "17" else conn_matters
        assert m["rule_matters"] == rule and m["connectivity_matters"] == conn_matters, seed
        n_rule += rule
        n_conn += conn_matters
    share = n_rule / SEEDS
    if tid == "17":
        assert share >= 0.45, share
    else:
        assert share >= 0.40, share


@pytest.mark.parametrize("tid", TASKS)
def test_touching_only_deliberate(sweep, tid):
    touching = 0
    for seed, ex, _ in sweep[tid]:
        g, o, m = _grid(ex)
        bg = int(m["background_color"])
        accents = set(m["accent_colors"])
        touch = _touching_other_color(g, bg, accents)
        assert touch == bool(m["touch_feature"]), seed
        touching += touch
        # same-colour diagonal-only contacts happen only inside declared chains
        plain = _strip(g, bg, m["accent_colors"])
        comps8 = _components(plain, 8, bg)
        comps4 = _components(plain, 4, bg)
        n_split = sum(1 for _, c8 in comps8 if sum(1 for _, c4 in comps4 if c4 <= c8) > 1)
        assert n_split == m["chain_features"], (seed, n_split, m["chain_features"])
    assert touching / SEEDS <= 0.15, touching / SEEDS


@pytest.mark.parametrize("tid", TASKS)
def test_backgrounds_sparse_library_only(sweep, tid):
    styles = Counter()
    for seed, ex, _ in sweep[tid]:
        g, o, m = _grid(ex)
        style = m["bg_style"]
        styles[style] += 1
        accents = set(m["accent_colors"])
        assert m["texture_named"] is False, seed
        if style == "plain":
            assert not accents and m["texture_kind"] is None and m["texture_coverage"] == 0.0, seed
            continue
        assert style in SPARSE_KINDS and m["texture_group"] == "sparse" and m["texture_kind"] == style, seed
        assert set(m["texture_colors"]) == accents and m["texture_coverage"] > 0, seed
        assert min(g.shape) >= 8, seed
        assert accents and int(m["background_color"]) not in accents and int(m["target_color"]) not in accents, seed
        assert not (accents & set(m["object_colors"])), seed
        # sparse means isolated: no texture cell has an 8-neighbour of any texture colour
        tex = np.isin(g, sorted(accents))
        iso = _isolated(np.where(tex, -1, g))
        assert bool(iso[tex].all()), seed
    textured = 1 - styles["plain"] / SEEDS
    assert 0.12 <= textured <= 0.5, styles
    assert len(set(styles) - {"plain"}) >= 5, styles
    assert not (set(styles) & {"dots", "dashes", "speckle", "checker"}), styles


def test_task18_margins_loose_and_tight(sweep):
    """``margin_mode`` is the realized gap on the emitted grid: tight = the runner-up is 2 or 3
    behind, loose = 4 or more.  Both happen often."""
    modes = Counter()
    ratio = 0
    for _, ex, _ in sweep["18"]:
        m = ex["metadata"]
        assert m["margin"] >= 2
        modes[m["margin_mode"]] += 1
        if m["margin_mode"] == "loose":
            assert m["margin"] >= 4, m
        else:
            assert m["margin"] in (2, 3), m
        ratio += m["winner_score"] >= 1.25 * m["runner_up_score"]
    assert 0.20 <= modes["loose"] / SEEDS <= 0.60, modes
    assert 0.40 <= modes["tight"] / SEEDS <= 0.80, modes
    assert ratio / SEEDS >= 0.40, ratio


@pytest.mark.parametrize("tid", TASKS)
def test_both_connectivities_drawn(sweep, tid):
    conns = Counter(ex["metadata"]["connectivity"] for _, ex, _ in sweep[tid])
    assert set(conns) == {"4", "8"}, conns
    assert 0.42 <= conns["4"] / SEEDS <= 0.58, conns


def _shares(rows, key_fn) -> dict:
    c = Counter(key_fn(m) for m in rows)
    n = len(rows)
    return {k: v / n for k, v in c.items()}


def test_task18_scene_independent_of_connectivity(sweep):
    """The picture never hints the connectivity: scene recipes, shape classes, textures,
    object counts, canvas sizes and the touching feature have the same rates under both
    readings (only the winner role differs, by design), and the rule is load-bearing under
    each reading on its own."""
    by = {4: [], 8: []}
    for _, ex, _ in sweep["18"]:
        m = ex["metadata"]
        by[int(m["connectivity"])].append(m)
    a, b = by[4], by[8]
    assert min(len(a), len(b)) >= 700
    for key_fn, tol in (
        (lambda m: m["scene_mode"], 0.07),
        (lambda m: m["bg_style"] != "plain", 0.06),
        (lambda m: bool(m["touch_feature"]), 0.05),
        (lambda m: m["chain_features"] > 0, 0.07),
        (lambda m: min(4, m["n_colors"]), 0.06),
    ):
        sa, sb = _shares(a, key_fn), _shares(b, key_fn)
        for k in set(sa) | set(sb):
            assert abs(sa.get(k, 0) - sb.get(k, 0)) <= tol, (k, sa.get(k, 0), sb.get(k, 0))
    # shape classes over all placed objects
    ca = Counter(CLASS_OF[k] for m in a for k in m["shape_kinds"])
    cb = Counter(CLASS_OF[k] for m in b for k in m["shape_kinds"])
    na, nb = sum(ca.values()), sum(cb.values())
    for cls in set(ca) | set(cb):
        assert abs(ca[cls] / na - cb[cls] / nb) <= 0.05, (cls, ca[cls] / na, cb[cls] / nb)
    assert abs(np.mean([m["n_objects"] for m in a]) - np.mean([m["n_objects"] for m in b])) <= 1.0
    assert abs(np.mean([m["grid_h"] * m["grid_w"] for m in a]) - np.mean([m["grid_h"] * m["grid_w"] for m in b])) <= 40
    # the rule is load-bearing under each reading, and the chain scenes are where it bites
    for rows in (a, b):
        assert np.mean([m["rule_matters"] for m in rows]) >= 0.40
        chain = [m for m in rows if m["scene_mode"].startswith("chain_scene")]
        assert len(chain) / len(rows) >= 0.35
        assert all(m["rule_matters"] for m in chain)
    # who wins depends on the reading: chains win only under 8-conn
    assert not any(m["winner_kind"] == "chain" for m in a)
    assert np.mean([m["winner_kind"] == "chain" for m in b]) >= 0.30


@pytest.mark.parametrize("tid", TASKS)
def test_metadata_keys(sweep, tid):
    keys = {"target_color", "background_color", "connectivity", "winner_cells", "winner_bbox",
            "runner_up_score", "rule_matters", "shape_kinds", "bg_style", "winner_kind",
            "scene_mode", "n_objects", "density", "accent_colors", "object_colors",
            "background_margin", "n_colors", "texture_group", "texture_kind", "texture_colors",
            "texture_named", "texture_coverage"}
    for _, ex, _ in sweep[tid]:
        assert keys <= set(ex["metadata"]), keys - set(ex["metadata"])


@pytest.mark.parametrize("tid", sorted(SIBLING_HASHES_300))
def test_siblings_byte_identical(tid):
    hs = hashlib.sha1()
    for seed in range(300):
        hs.update(json.dumps(generate(tid, seed), sort_keys=True).encode())
    assert hs.hexdigest() == SIBLING_HASHES_300[tid], tid
