"""Tasks 95 (outline every color X object's box) and 96 (extract frame interiors).

The oracles below are written from the prompt text alone -- the connectivity,
the colours, the arrangement and the gap are parsed out of the sentences, the
objects come from a union-find labelling and the frames from a corner walk, and
nothing here imports the generators' readers -- so a bug shared with a module
cannot hide itself.
"""
from collections import Counter
from copy import deepcopy
import json
import re
import time

import numpy as np
import pytest

from generators import bbox_outline_95 as module95
from generators import frame_extract_96 as module96
from prompts import get_prompt
from registry import generate
from symbol_remap import palette_for_example

ORACLE_SEEDS = 400
SWEEP_SEEDS = 700


def _ex(task, seed):
    return generate(task, seed)


@pytest.fixture(scope="module")
def sweep95():
    return [_ex("95", seed) for seed in range(SWEEP_SEEDS)]


@pytest.fixture(scope="module")
def sweep96():
    return [_ex("96", seed) for seed in range(SWEEP_SEEDS)]


def _meta(sweep):
    return [ex["metadata"] for ex in sweep]


# ---------------------------------------------------------------------------
# task 95: an independent oracle read off the prompt
# ---------------------------------------------------------------------------

def _connectivity_from_prompt(prompt):
    match = re.search(r"[Tt]reat each (\d)-connected same-color group", prompt)
    assert match, prompt
    conn = int(match.group(1))
    if conn == 4:
        assert "connected horizontally or vertically" in prompt
        assert "cells that touch only at a corner are not connected" in prompt
    else:
        assert "connected horizontally, vertically, or diagonally" in prompt
    return conn


def _colors_from_95_prompt(prompt):
    target = re.search(r"Find every color (\d) object", prompt)
    outline = re.search(r"tight bounding box (?:\([^)]*\) )?in color (\d)", prompt)
    assert target and outline, prompt
    return int(target.group(1)), int(outline.group(1))


def _label(mask, connectivity):
    """Row-major union-find labelling (a second implementation)."""
    h, w = mask.shape
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    steps = [(-1, 0), (0, -1)] + ([(-1, -1), (-1, 1)] if connectivity == 8 else [])
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            parent.setdefault(r * w + c, r * w + c)
            for dr, dc in steps:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc]:
                    parent.setdefault(rr * w + cc, rr * w + cc)
                    union(r * w + c, rr * w + cc)
    groups = {}
    for key in parent:
        groups.setdefault(find(key), []).append((key // w, key % w))
    return list(groups.values())


def _isolated(arr, r, c):
    h, w = arr.shape
    color = arr[r, c]
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and arr[rr, cc] == color:
                return False
    return True


def _oracle_95_objects(grid, prompt, *, outline_singletons=False):
    conn = _connectivity_from_prompt(prompt)
    target, _ = _colors_from_95_prompt(prompt)
    arr = np.asarray(grid)
    objects = []
    for cells in _label(arr == target, conn):
        if len(cells) >= 2 or outline_singletons:
            objects.append(cells)
            continue
        r, c = cells[0]
        assert _isolated(arr, r, c), (r, c)      # neither an object nor texture
    return objects


def _oracle_95(grid, prompt, *, outline_singletons=False):
    _, outline = _colors_from_95_prompt(prompt)
    arr = np.asarray(grid)
    h, w = arr.shape
    out = arr.copy()
    for cells in _oracle_95_objects(grid, prompt, outline_singletons=outline_singletons):
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, c0, r1, c1 = min(rs), min(cs), max(rs), max(cs)
        for r in range(r0 - 1, r1 + 2):
            for c in range(c0 - 1, c1 + 2):
                if (r in (r0 - 1, r1 + 1) or c in (c0 - 1, c1 + 1)) and 0 <= r < h and 0 <= c < w:
                    out[r, c] = outline
    return out.tolist()


def test_95_independent_oracle():
    for seed in range(ORACLE_SEEDS):
        ex = _ex("95", seed)
        prompt = get_prompt("95", ex["metadata"])
        assert _oracle_95(ex["input"], prompt) == ex["output"], seed


def test_95_isolated_cell_rule_is_load_bearing():
    """Lone target cells really occur, and outlining them changes the answer."""
    changed = 0
    strays = 0
    for seed in range(ORACLE_SEEDS):
        ex = _ex("95", seed)
        prompt = get_prompt("95", ex["metadata"])
        if ex["metadata"]["stray_target_cells"]:
            strays += 1
            naive = _oracle_95(ex["input"], prompt, outline_singletons=True)
            if naive != ex["output"]:
                changed += 1
    assert strays >= 0.20 * ORACLE_SEEDS, strays
    assert changed >= 0.95 * strays, (changed, strays)


# ---------------------------------------------------------------------------
# task 96: an independent oracle read off the prompt
# ---------------------------------------------------------------------------

def _p96_facts(prompt):
    frame = re.search(r"A rectangular frame of color (\d) is", prompt)
    spacer = re.search(r"of color (\d)\. The output is", prompt)
    gap = re.search(r"a gap of (\d+) (column|columns|row|rows) of color", prompt)
    assert frame and spacer and gap, prompt
    unit = gap.group(2)
    count = int(gap.group(1))
    assert (count == 1) == unit.endswith(("column", "row")), prompt   # pluralisation
    arrangement = "vertical" if unit.startswith("row") else "horizontal"
    if arrangement == "horizontal":
        assert "side by side, from left to right" in prompt
        assert "as tall as the tallest interior" in prompt
        assert "its top row on the output's top row" in prompt
    else:
        assert "one below another, from top to bottom" in prompt
        assert "as wide as the widest interior" in prompt
        assert "its left column on the output's left column" in prompt
    return int(frame.group(1)), int(spacer.group(1)), count, arrangement


def _oracle_96_frames(grid, frame_color, *, ignore_interior=False, any_ring=False):
    """Corner-walk frame finder: a genuinely different derivation from the
    module's component test.  ``ignore_interior`` / ``any_ring`` switch off one
    clause of the definition so the tests can measure what it decides."""
    arr = np.asarray(grid)
    h, w = arr.shape
    found = []
    for r0 in range(h):
        for c0 in range(w):
            if arr[r0, c0] != frame_color:
                continue
            if r0 and arr[r0 - 1, c0] == frame_color:
                continue
            if c0 and arr[r0, c0 - 1] == frame_color:
                continue
            c1 = c0
            while c1 + 1 < w and arr[r0, c1 + 1] == frame_color:
                c1 += 1
            r1 = r0
            while r1 + 1 < h and arr[r1 + 1, c0] == frame_color:
                r1 += 1
            if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
                continue
            ring = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                    if r in (r0, r1) or c in (c0, c1)]
            if any(arr[r, c] != frame_color for r, c in ring):
                continue
            if not ignore_interior and any(
                    arr[r, c] == frame_color
                    for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)):
                continue
            if not any_ring:
                # the ring must be a whole component: nothing of the frame
                # colour may hang off it
                leaks = False
                for r, c in ring:
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        rr, cc = r + dr, c + dc
                        if not (0 <= rr < h and 0 <= cc < w):
                            continue
                        inside = r0 <= rr <= r1 and c0 <= cc <= c1
                        if not inside and arr[rr, cc] == frame_color:
                            leaks = True
                if leaks:
                    continue
            found.append((r0, c0, r1, c1))
    return sorted(found, key=lambda b: (b[0], b[1]))


def _oracle_96(grid, prompt, **kwargs):
    frame, spacer, gap, arrangement = _p96_facts(prompt)
    arr = np.asarray(grid)
    boxes = _oracle_96_frames(grid, frame, **kwargs)
    parts = [arr[r0 + 1:r1, c0 + 1:c1] for r0, c0, r1, c1 in boxes]
    assert parts, prompt
    if arrangement == "horizontal":
        out = np.full((max(p.shape[0] for p in parts),
                       sum(p.shape[1] for p in parts) + gap * (len(parts) - 1)), spacer)
        col = 0
        for part in parts:
            out[:part.shape[0], col:col + part.shape[1]] = part
            col += part.shape[1] + gap
    else:
        out = np.full((sum(p.shape[0] for p in parts) + gap * (len(parts) - 1),
                       max(p.shape[1] for p in parts)), spacer)
        row = 0
        for part in parts:
            out[row:row + part.shape[0], :part.shape[1]] = part
            row += part.shape[0] + gap
    return out.tolist()


def test_96_independent_oracle():
    for seed in range(ORACLE_SEEDS):
        ex = _ex("96", seed)
        prompt = get_prompt("96", ex["metadata"])
        assert _oracle_96(ex["input"], prompt) == ex["output"], seed


def test_96_frames_are_found_without_the_metadata(sweep96):
    for ex in sweep96:
        meta = ex["metadata"]
        prompt = get_prompt("96", meta)
        frame, _, _, _ = _p96_facts(prompt)
        boxes = _oracle_96_frames(ex["input"], frame)
        assert [list(b) for b in boxes] == [list(b) for b in meta["frame_boxes"]], meta
        assert len(boxes) >= 2
        # the stated order is a total order, and the frames never touch
        assert len({(b[0], b[1]) for b in boxes}) == len(boxes)
        for i, a in enumerate(boxes):
            for b in boxes[i + 1:]:
                gap_r = max(a[0] - b[2], b[0] - a[2])
                gap_c = max(a[1] - b[3], b[1] - a[3])
                assert max(gap_r, gap_c) >= 2, (a, b)


def test_96_definition_clauses_are_load_bearing(sweep96):
    """Broken rings, solid bodies and rings with frame-colour content inside
    them all occur, and reading a clause away changes the answer."""
    meta = _meta(sweep96)
    broken = np.mean([m["n_broken_rings"] > 0 for m in meta])
    spoiled = np.mean([m["n_spoiled_rings"] > 0 for m in meta])
    solid = np.mean([m["n_solid_frame_color_bodies"] > 0 for m in meta])
    assert broken >= 0.15, broken
    assert spoiled >= 0.07, spoiled
    assert solid >= 0.22, solid
    ignored_interior = 0
    seen_spoiled = 0
    for ex in sweep96[:ORACLE_SEEDS]:
        meta_ex = ex["metadata"]
        if not meta_ex["n_spoiled_rings"]:
            continue
        seen_spoiled += 1
        prompt = get_prompt("96", meta_ex)
        if _oracle_96(ex["input"], prompt, ignore_interior=True) != ex["output"]:
            ignored_interior += 1
    assert seen_spoiled >= 10, seen_spoiled
    assert ignored_interior == seen_spoiled, (ignored_interior, seen_spoiled)


def test_96_padding_rule_is_load_bearing(sweep96):
    """The short interiors really are padded, so "as tall as the tallest" and
    the alignment sentence decide cells."""
    meta = _meta(sweep96)
    assert np.mean([m["ragged"] for m in meta]) >= 0.50
    assert np.mean([m["padding_cells"] > 0 for m in meta]) >= 0.60
    for ex in sweep96[:120]:
        m = ex["metadata"]
        if not m["ragged"]:
            continue
        out = np.asarray(ex["output"])
        assert int((out == m["spacer_color"]).sum()) >= m["padding_cells"]


def test_determinism_and_cell_range():
    for task in ("95", "96"):
        for seed in range(50):
            a, b = _ex(task, seed), _ex(task, seed)
            assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True), (task, seed)
            for grid in (a["input"], a["output"]):
                arr = np.asarray(grid)
                assert arr.min() >= 0 and arr.max() <= 9
                assert arr.ndim == 2


# ---------------------------------------------------------------------------
# sizes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task", ["95", "96"])
def test_sides_are_flat_and_independent(task, sweep95, sweep96):
    meta = _meta(sweep95 if task == "95" else sweep96)
    hs = [m["grid_h"] for m in meta]
    ws = [m["grid_w"] for m in meta]
    assert min(hs) <= 6 and max(hs) >= 29, (min(hs), max(hs))
    assert min(ws) <= 6 and max(ws) >= 29, (min(ws), max(ws))
    assert max(hs) <= 30 and max(ws) <= 30
    assert abs(float(np.corrcoef(hs, ws)[0, 1])) <= 0.15
    squares = np.mean([h == w for h, w in zip(hs, ws)])
    assert squares <= 0.10, squares
    big = np.mean([h * w >= 700 for h, w in zip(hs, ws)])
    assert 0.01 <= big <= 0.09, big
    counts = Counter(hs + ws)
    fair = len(hs + ws) / 26
    assert max(counts.values()) <= 2.2 * fair, counts.most_common(3)
    assert min(counts.get(v, 0) for v in range(5, 31)) >= 0.35 * fair, counts
    if task == "96":
        # the only structural floor: two non-touching 3x3 frames need a side of 7
        assert all(max(h, w) >= 7 for h, w in zip(hs, ws))


def test_96_output_is_an_arc_canvas(sweep96):
    for m in _meta(sweep96):
        assert 1 <= m["output_h"] <= 30 and 1 <= m["output_w"] <= 30, m
        assert m["output_h"] * m["output_w"] <= 900
    sides = [m["output_w"] for m in _meta(sweep96) if m["arrangement"] == "horizontal"]
    assert max(sides) >= 20, max(sides)          # the cap is reached, not avoided


# ---------------------------------------------------------------------------
# colours
# ---------------------------------------------------------------------------

def test_95_colours_and_the_reserved_outline(sweep95):
    meta = _meta(sweep95)
    counts = [m["n_colors"] for m in meta]
    out_counts = [m["n_colors_output"] for m in meta]
    assert min(counts) == 2 and max(counts) == 9, (min(counts), max(counts))
    assert max(out_counts) == 10
    assert np.mean([c >= 8 for c in counts]) >= 0.15
    assert np.mean([c >= 8 for c in out_counts]) >= 0.25
    assert np.mean([c <= 3 for c in counts]) >= 0.08
    for ex, m in zip(sweep95, meta):
        arr = np.asarray(ex["input"])
        assert m["outline_color"] not in set(int(v) for v in np.unique(arr))
        assert m["outline_color"] not in (m["target_color"], m["background_color"])
        assert m["target_color"] not in (m["texture_colors"] or [])
        assert m["outline_color"] not in (m["texture_colors"] or [])


def test_96_colours_and_the_spacer(sweep96):
    meta = _meta(sweep96)
    counts = [m["n_colors"] for m in meta]
    assert min(counts) == 2 and max(counts) == 10, (min(counts), max(counts))
    assert np.mean([c >= 8 for c in counts]) >= 0.15
    assert np.mean([c <= 3 for c in counts]) >= 0.08
    for m in meta:
        assert m["spacer_color"] != m["background_color"]
        assert m["spacer_color"] != m["frame_color"]
        assert m["frame_color"] not in (m["texture_colors"] or [])
    # the spacer is drawn independently of the scene: it is in the input about
    # as often as a free colour would be, and it is not a background alias
    in_input = np.mean([m["spacer_in_input"] for m in meta])
    assert 0.15 <= in_input <= 0.75, in_input
    spread = Counter(m["spacer_color"] for m in meta)
    assert len(spread) == 10, spread
    assert max(spread.values()) <= 2.0 * len(meta) / 10, spread.most_common(3)


# ---------------------------------------------------------------------------
# textures, objects, scenes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task", ["95", "96"])
def test_textures_come_from_the_library(task, sweep95, sweep96):
    from common.backgrounds import isolated_cell_mask
    sweep = sweep95 if task == "95" else sweep96
    groups = Counter(ex["metadata"]["texture_group"] for ex in sweep)
    assert groups["sparse"] >= 0.05 * len(sweep), groups
    assert groups["solid"] >= 0.03 * len(sweep), groups
    assert groups[None] >= 0.4 * len(sweep), groups
    for ex in sweep:
        meta = ex["metadata"]
        colors = meta["texture_colors"]
        if not colors:
            continue
        assert meta["texture_named"] is False
        arr = np.asarray(ex["input"])
        if meta["texture_group"] == "sparse":
            assert bool(isolated_cell_mask(arr)[np.isin(arr, colors)].all())


@pytest.mark.parametrize("task", ["95", "96"])
def test_scenes_scale_with_the_canvas(task, sweep95, sweep96):
    meta = _meta(sweep95 if task == "95" else sweep96)
    bands = {}
    for m in meta:
        area = m["grid_h"] * m["grid_w"]
        key = "<150" if area < 150 else ("150-399" if area < 400 else ("400-699" if area < 700 else "700+"))
        bands.setdefault(key, []).append(m["occupancy"])
    for key in ("<150", "150-399", "400-699"):
        assert np.mean(bands[key]) >= 0.12, (key, np.mean(bands[key]))
    if bands.get("700+"):
        assert np.mean(bands["700+"]) >= 0.6 * np.mean(bands["<150"]), (
            np.mean(bands["700+"]), np.mean(bands["<150"]))


def test_95_bodies_come_from_the_shared_vocabulary(sweep95):
    meta = _meta(sweep95)
    kinds = Counter(k for m in meta for k in m["object_kinds"] + m["distractor_kinds"])
    assert len(kinds) >= 8, kinds
    assert max(kinds.values()) <= 0.5 * sum(kinds.values()), kinds.most_common(3)
    sizes = [s for m in meta for s in m["target_object_sizes"]]
    assert min(sizes) == 2 and max(sizes) >= 60, (min(sizes), max(sizes))


def test_96_interiors_are_varied(sweep96):
    meta = _meta(sweep96)
    sizes = [tuple(s) for m in meta for s in m["interior_sizes"]]
    assert min(s[0] for s in sizes) == 1 and max(s[0] for s in sizes) == 8
    assert min(s[1] for s in sizes) == 1 and max(s[1] for s in sizes) == 8
    kinds = Counter(k for m in meta for k in m["interior_kinds"])
    assert set(kinds) >= {"empty", "fill", "body", "detail", "bands", "mixed"}, kinds
    assert max(kinds.values()) <= 0.45 * sum(kinds.values()), kinds.most_common(3)
    blank = np.mean([m["blank_interiors"] > 0 for m in meta])
    all_blank = np.mean([m["blank_interiors"] == m["n_frames"] for m in meta])
    assert blank >= 0.20, blank
    assert all_blank <= 0.15, all_blank
    frames = Counter(m["n_frames"] for m in meta)
    assert min(frames) == 2 and max(frames) >= 6, frames
    assert Counter(m["gap"] for m in meta).keys() == {1, 2, 3}
    assert set(m["arrangement"] for m in meta) == {"horizontal", "vertical"}


# ---------------------------------------------------------------------------
# task 95: the connectivity and the three ring policies
# ---------------------------------------------------------------------------

def test_95_connectivity_is_stated_and_load_bearing(sweep95):
    meta = _meta(sweep95)
    share4 = np.mean([m["connectivity"] == 4 for m in meta])
    assert 0.35 <= share4 <= 0.65, share4
    differs = np.mean([m["differs_under_other_connectivity"] for m in meta])
    assert differs >= 0.30, differs
    for ex, m in zip(sweep95[:150], meta[:150]):
        prompt = get_prompt("95", m)
        assert _connectivity_from_prompt(prompt) == m["connectivity"]
        other = 4 if m["connectivity"] == 8 else 8
        alt = _oracle_95(ex["input"], get_prompt("95", dict(m, connectivity=other)))
        assert (alt != ex["output"]) == m["differs_under_other_connectivity"], m


def test_95_the_connectivity_cannot_be_read_off_the_scene(sweep95):
    """The coin is flipped after the scene exists, so input-only features are
    at chance."""
    meta = _meta(sweep95)
    feats = [(m["grid_h"], m["grid_w"], m["n_colors"], m["target_cells"]) for m in meta]
    labels = [m["connectivity"] for m in meta]
    half = len(feats) // 2
    table = {}
    for f, t in zip(feats[:half], labels[:half]):
        table.setdefault(f, Counter())[t] += 1
    fallback = Counter(labels[:half]).most_common(1)[0][0]
    hits = sum(1 for f, t in zip(feats[half:], labels[half:])
               if (table[f].most_common(1)[0][0] if f in table else fallback) == t)
    assert hits / (len(feats) - half) <= 0.62


def test_95_ring_policies_are_true_and_load_bearing(sweep95):
    meta = _meta(sweep95)
    clipped = np.mean([m["rings_clipped"] for m in meta])
    overlap = np.mean([m["rings_touch"] for m in meta])
    over = np.mean([m["rings_overwrite_object"] for m in meta])
    covered = np.mean([m["target_cells_covered"] > 0 for m in meta])
    assert 0.30 <= clipped <= 0.95, clipped
    assert overlap >= 0.15, overlap
    assert over >= 0.30, over
    assert 0.05 <= covered <= 0.45, covered
    for ex, m in zip(sweep95, meta):
        h, w = m["grid_h"], m["grid_w"]
        arr = np.asarray(ex["input"])
        out = np.asarray(ex["output"])
        # the clip is real: a ring is short exactly when its box touches an edge
        touching = any(r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1
                       for r0, c0, r1, c1 in m["target_boxes"])
        assert (m["ring_cells_clipped"] > 0) == touching, m
        # nothing outside the rings moved
        changed = np.argwhere(arr != out)
        assert len(changed) <= m["ring_cells"]
        assert all(int(out[r, c]) == m["outline_color"] for r, c in changed)
        # a covered target cell only ever happens under the 4-reading of a
        # corner-joined figure
        if m["target_cells_covered"]:
            assert m["connectivity"] == 4 and m["n_targets"] > m["n_figures"], m


def test_95_every_target_cell_reads_the_same_way(sweep95):
    """No color X cell is a 4-group of one that still has a diagonal neighbour:
    such a cell would be neither an object nor an isolated single cell."""
    for ex in sweep95:
        m = ex["metadata"]
        arr = np.asarray(ex["input"])
        for cells in _label(arr == m["target_color"], 4):
            if len(cells) == 1:
                r, c = cells[0]
                assert _isolated(arr, r, c), (m, r, c)


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------

def test_prompt_policy(sweep95, sweep96):
    wordings = {"95": set(), "96": set()}
    for ex in sweep95[:250]:
        meta = ex["metadata"]
        prompt = get_prompt("95", meta)
        wordings["95"].add(re.sub(r"color \d", "color X", prompt))
        assert "Ignore the background, whether it is a solid color or a pattern." in prompt
        assert not re.search(r"background (is|color|colour)\b", prompt)
        # the only word about the background is the canonical isolated-cell clause
        assert not re.search(r"\bpatterned\b|\bnoise\b|background cells", prompt)
        assert prompt.count("texture") == 1
        assert "isolated single cells, whatever their color, are background texture" in prompt
        assert "at least two cells" in prompt
        assert prompt.endswith("Leave every other cell unchanged.")
        assert "dropping any outline cell that falls outside the grid" in prompt
        assert "over whatever is there, another object's cells included" in prompt
        assert "before any outline is drawn" in prompt
    for ex in sweep96[:250]:
        meta = ex["metadata"]
        prompt = get_prompt("96", meta)
        wordings["96"].add(re.sub(r"color \d", "color X", prompt))
        assert "background" not in prompt         # the gap colour is a given
        assert not re.search(r"\btexture\b|\bpatterned\b|\bnoise\b", prompt)
        assert "complete one-cell-thick rectangular ring" in prompt
        assert "at least 3 rows by 3 columns" in prompt
        assert "whose interior contains no cell of color" in prompt
        assert "sort them by top row, then by left column" in prompt
        assert "strictly inside its ring, copied exactly" in prompt
        _p96_facts(prompt)
    assert len(wordings["95"]) == 2, wordings["95"]
    # 96: two arrangements x three gaps, each with its own singular / plural
    assert len(wordings["96"]) == 6, sorted(wordings["96"])


def test_prompts_survive_the_symbol_palette():
    letters = list("ABCDEFGHIJ")
    for task in ("95", "96"):
        for seed in range(40):
            meta = generate(task, seed)["metadata"]
            prompt = get_prompt(task, meta, color_palette=letters)
            assert not re.search(r"\bcolor \d\b", prompt), (task, seed, prompt)
            assert re.search(r"\bcolor [A-J]\b", prompt), (task, seed, prompt)
            remapped = get_prompt(task, meta, color_palette=palette_for_example(task, seed))
            assert not re.search(r"\bcolor \d\b", remapped), (task, seed, remapped)


# ---------------------------------------------------------------------------
# metadata contract and the validators
# ---------------------------------------------------------------------------

def test_metadata_contract(sweep95, sweep96):
    shared = {"grid_h", "grid_w", "n_colors", "background_color", "occupancy",
              "texture_group", "texture_kind", "texture_colors", "texture_named",
              "texture_coverage", "attempts"}
    for ex in sweep95[:200]:
        meta = ex["metadata"]
        assert shared <= set(meta), shared - set(meta)
        assert {"connectivity", "target_color", "outline_color", "n_targets",
                "n_figures", "figure_cap", "ring_cells", "ring_cells_clipped",
                "rings_clipped", "ring_overlap_cells", "rings_touch",
                "ring_cells_over_objects", "target_cells_covered",
                "rings_overwrite_object", "stray_target_cells", "object_kinds",
                "distractor_kinds", "differs_under_other_connectivity"} <= set(meta)
        assert meta["connectivity"] in (4, 8)
        assert meta["grid_h"] == len(ex["input"]) and meta["grid_w"] == len(ex["input"][0])
        assert len(ex["output"]) == meta["grid_h"] and len(ex["output"][0]) == meta["grid_w"]
        assert 1 <= meta["n_targets"]
    for ex in sweep96[:200]:
        meta = ex["metadata"]
        assert shared <= set(meta), shared - set(meta)
        assert {"frame_color", "spacer_color", "arrangement", "gap", "n_frames",
                "frame_boxes", "interior_sizes", "interior_kinds", "ragged",
                "padding_cells", "output_h", "output_w", "n_broken_rings",
                "n_spoiled_rings", "n_solid_frame_color_bodies",
                "n_other_color_frames", "blank_interiors",
                "uniform_interiors"} <= set(meta)
        assert meta["grid_h"] == len(ex["input"]) and meta["grid_w"] == len(ex["input"][0])
        assert meta["output_h"] == len(ex["output"]) and meta["output_w"] == len(ex["output"][0])
        assert meta["n_frames"] == len(meta["frame_boxes"]) == len(meta["interior_sizes"])
    assert np.mean([ex["metadata"]["attempts"] for ex in sweep95]) <= 2.0
    assert np.mean([ex["metadata"]["attempts"] for ex in sweep96]) <= 2.0


def test_95_validator_raises_on_a_tampered_example():
    ex = _ex("95", 5)
    module95.validate_95(deepcopy(ex))

    wrong = deepcopy(ex)
    wrong["output"][0][0] = (wrong["output"][0][0] + 1) % 10
    with pytest.raises(ValueError):
        module95.validate_95(wrong)

    flip_seed = next(s for s in range(80)
                     if _ex("95", s)["metadata"]["differs_under_other_connectivity"])
    flipped = _ex("95", flip_seed)
    flipped["metadata"]["connectivity"] = 4 if flipped["metadata"]["connectivity"] == 8 else 8
    with pytest.raises(ValueError):
        module95.validate_95(flipped)

    painted = deepcopy(ex)
    grid = [list(row) for row in painted["input"]]
    grid[0][0] = painted["metadata"]["outline_color"]
    painted["input"] = grid
    with pytest.raises(ValueError):
        module95.validate_95(painted)

    counted = deepcopy(ex)
    counted["metadata"]["n_targets"] += 1
    with pytest.raises(ValueError):
        module95.validate_95(counted)

    ringed = deepcopy(ex)
    ringed["metadata"]["ring_cells"] += 1
    with pytest.raises(ValueError):
        module95.validate_95(ringed)


def test_96_validator_raises_on_a_tampered_example():
    ex = _ex("96", 5)
    module96.validate_96(deepcopy(ex))

    wrong = deepcopy(ex)
    wrong["output"][0][0] = (wrong["output"][0][0] + 1) % 10
    with pytest.raises(ValueError):
        module96.validate_96(wrong)

    swapped = deepcopy(ex)
    swapped["metadata"]["arrangement"] = (
        "vertical" if swapped["metadata"]["arrangement"] == "horizontal" else "horizontal")
    with pytest.raises(ValueError):
        module96.validate_96(swapped)

    wider = deepcopy(ex)
    wider["metadata"]["gap"] += 1
    with pytest.raises(ValueError):
        module96.validate_96(wider)

    spacer = deepcopy(ex)
    spacer["metadata"]["spacer_color"] = spacer["metadata"]["background_color"]
    with pytest.raises(ValueError):
        module96.validate_96(spacer)

    boxes = deepcopy(ex)
    boxes["metadata"]["n_frames"] += 1
    with pytest.raises(ValueError):
        module96.validate_96(boxes)

    # a frame erased from the picture is detected, not read from the plan
    erased = deepcopy(ex)
    grid = [list(row) for row in erased["input"]]
    r0, c0, r1, c1 = erased["metadata"]["frame_boxes"][0]
    grid[r0][c0] = erased["metadata"]["background_color"]
    erased["input"] = grid
    with pytest.raises(ValueError):
        module96.validate_96(erased)


def test_generation_speed():
    for task in ("95", "96"):
        times = []
        for seed in range(300):
            t0 = time.perf_counter()
            generate(task, seed)
            times.append((time.perf_counter() - t0) * 1000.0)
        assert np.mean(times) <= 8.0, (task, np.mean(times))
        assert np.percentile(times, 99) <= 60.0, (task, np.percentile(times, 99))
