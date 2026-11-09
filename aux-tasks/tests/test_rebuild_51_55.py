"""Independent prompt-only checks for frame measurement and block scaling.

The readers below consume displayed cells and instructions, never construction
metadata or production recognition helpers. They also run with letter symbols.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from functools import lru_cache
import re

import numpy as np
import pytest

from prompts import get_prompt, PROMPT_TEMPLATES
from registry import generate, list_tasks

TASKS = ("51", "52", "54", "55")


def visible_frames(grid):
    a = np.asarray(grid)
    h, w = a.shape
    unseen = {(r, c) for r in range(h) for c in range(w)}
    result = []
    while unseen:
        first = min(unseen)
        color = a[first].item()
        unseen.remove(first)
        stack, cells = [first], {first}
        while stack:
            r, c = stack.pop()
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in unseen and a[p] == color:
                    unseen.remove(p)
                    cells.add(p)
                    stack.append(p)
        r0, r1 = min(r for r, _ in cells), max(r for r, _ in cells)+1
        c0, c1 = min(c for _, c in cells), max(c for _, c in cells)+1
        hh, ww = r1-r0, c1-c0
        for t in range(1, (min(hh, ww)-1)//2+1):
            wall = {(r, c) for r in range(r0, r1) for c in range(c0, c1)
                    if not (r0+t <= r < r1-t and c0+t <= c < c1-t)}
            if cells == wall:
                result.append((r0, c0, hh, ww, color, t))
                break
    return sorted(result)


def pair_candidates(grid):
    """Try all colors and every possible separating row/column, including
    one-cell gaps and nonprimitive reference patterns."""
    a = np.asarray(grid)
    candidates = {}

    def cropped(part, bg, origin):
        pts = np.argwhere(part != bg)
        if not len(pts):
            return None
        lo, hi = pts.min(0), pts.max(0)+1
        return part[lo[0]:hi[0], lo[1]:hi[1]], tuple((lo+origin).tolist()+(hi+origin).tolist())

    for bg in np.unique(a):
        for axis in (0, 1):
            for cut in range(1, a.shape[axis]-1):
                band = a[cut, :] if axis == 0 else a[:, cut]
                if not np.all(band == bg):
                    continue
                parts = ((a[:cut], (0, 0)), (a[cut+1:], (cut+1, 0))) if axis == 0 else (
                    (a[:, :cut], (0, 0)), (a[:, cut+1:], (0, cut+1)))
                pieces = [cropped(p, bg, origin) for p, origin in parts]
                if any(p is None for p in pieces):
                    continue
                (s, sb), (b, bb) = sorted(pieces, key=lambda p: p[0].size)
                sh, sw = s.shape
                bh, bw = b.shape
                if bh % sh or bw % sw or bh//sh < 2 or bh//sh != bw//sw:
                    continue
                n = bh//sh
                if all(np.all(b[r*n:(r+1)*n, c*n:(c+1)*n] == s[r, c])
                       for r in range(sh) for c in range(sw)):
                    candidates[(bg.item(), sb, bb)] = n
    return candidates


def read_blocks(grid, n):
    a = np.asarray(grid)
    h, w = a.shape
    assert h % n == w % n == 0
    out = []
    for r in range(0, h, n):
        row = []
        for c in range(0, w, n):
            block = a[r:r+n, c:c+n]
            assert np.all(block == block[0, 0])
            row.append(block[0, 0].item())
        out.append(row)
    return out


def solve(tid, grid, prompt):
    if tid in ("51", "52"):
        found = visible_frames(grid)
        assert found
        if tid == "51":
            return str(max(f[5] for f in found))
        extreme = re.search(r"Select the (largest|smallest) frame by outer area", prompt).group(1)
        area = (max if extreme == "largest" else min)(f[2]*f[3] for f in found)
        winners = [f for f in found if f[2]*f[3] == area]
        assert len(winners) == 1
        r, c, h, w, _, t = winners[0]
        inner = np.asarray(grid)[r+t:r+h-t, c+t:c+w-t]
        target = re.search(r"cells of color (\w+) lie", prompt)
        if target:
            color = target[1] if inner.dtype.kind in "US" else int(target[1])
            return str(int((inner == color).sum()))
        assert "regardless of color" in prompt
        return str(inner.size)
    if tid == "54":
        candidates = pair_candidates(grid)
        assert len(candidates) == 1, candidates
        return str(next(iter(candidates.values())))
    n, n2 = map(int, re.search(r"non-overlapping (\d+)x(\d+) blocks", prompt).groups())
    assert n == n2
    return read_blocks(grid, n)


@lru_cache(maxsize=None)
def sweep(tid):
    return [generate(tid, seed) for seed in range(1000)]


@pytest.mark.parametrize("tid", TASKS)
def test_prompt_only_oracle(tid):
    for seed, ex in enumerate(sweep(tid)):
        assert solve(tid, ex["input"], get_prompt(tid, ex["metadata"])) == ex["output"], (tid, seed)


@pytest.mark.parametrize("tid", TASKS)
def test_letters_and_arbitrary_symbol_order(tid):
    rng = np.random.default_rng(6551)
    for seed in range(80):
        ex = generate(tid, seed)
        palette = list(np.asarray(list("ABCDEFGHIJ"))[rng.permutation(10)])
        inp = np.asarray(palette)[np.asarray(ex["input"])].tolist()
        expected = np.asarray(palette)[np.asarray(ex["output"])].tolist() if tid == "55" else ex["output"]
        assert solve(tid, inp, get_prompt(tid, ex["metadata"], color_palette=palette)) == expected


def test_merge_and_question_chosen_after_scene(monkeypatch):
    import generators.frame_measurement_51_52 as module
    assert "53" not in list_tasks() and "53" not in PROMPT_TEMPLATES
    with pytest.raises(KeyError, match="merged into task 52"):
        generate("53", 0)
    for seed in range(120):
        inputs = []
        for extreme, mode in (("largest", "all"), ("smallest", "all"), ("largest", "color"), ("smallest", "color")):
            monkeypatch.setattr(module, "choose_question", lambda rng, e=extreme, m=mode: (e, m))
            ex = generate("52", seed)
            inputs.append(ex["input"])
            assert solve("52", ex["input"], get_prompt("52", ex["metadata"])) == ex["output"]
        assert all(a == inputs[0] for a in inputs)


@pytest.mark.parametrize("tid", TASKS)
def test_determinism_palette_and_canvas(tid):
    examples = sweep(tid)
    shapes = [np.asarray(ex["input"]).shape for ex in examples]
    assert max(h for h, _ in shapes) == max(w for _, w in shapes) == 30
    assert np.mean([h*w >= 700 for h, w in shapes]) >= .02
    assert np.mean([h == w for h, w in shapes]) < .15
    counts = Counter(ex["metadata"]["n_colors"] for ex in examples)
    assert set(counts) == set(range(2, 11))
    assert sum(counts[k] for k in (8, 9, 10))/len(examples) > .16
    for seed, ex in enumerate(examples[:40]):
        assert ex == generate(tid, seed)
        assert len(np.unique(ex["input"])) == ex["metadata"]["n_colors"]
        assert "background of color" not in get_prompt(tid, ex["metadata"])


def test_frames_variety_and_preserved_counts():
    for tid in ("51", "52"):
        ms = [e["metadata"] for e in sweep(tid)]
        assert max(m["n_rectangles"] for m in ms) >= 10
        assert all(m["n_rectangles"] == m["n_requested"] for m in ms)
        assert all(m["background_margin"] > 1.25 for m in ms)
        assert max(max(f[2:4]) for m in ms for f in m["rectangles"]) > 24
        assert np.mean([len(set(m["rect_colors"])) < m["n_rectangles"] for m in ms]) > .30
        styles = Counter(m["bg_style"] for m in ms)
        assert len(styles) >= 6 and .12 < 1-styles["plain"]/len(ms) < .40


def test_crowded_layout_regressions():
    for tid, seeds in (("51", (53, 57, 112, 352, 772, 4459)), ("52", (42, 410))):
        for seed in seeds:
            ex = generate(tid, seed)
            assert solve(tid, ex["input"], get_prompt(tid, ex["metadata"])) == ex["output"]


def test_thickness_contrasts_are_a_balanced_share():
    ms = [e["metadata"] for e in sweep("51")]
    modes = Counter(m["layout_mode"] for m in ms)
    assert .10 < modes["contrast"]/len(ms) < .35
    assert all(modes[k] > 40 for k in ("equal", "comparable", "varied"))
    assert max(m["max_thickness"] for m in ms) >= 5
    assert np.mean([m["max_thickness"] == 1 for m in ms]) < .50
    hits = []
    for m in ms:
        winner = max(m["rectangles"], key=lambda f: 2*f[5]*(f[2]+f[3])-4*f[5]**2)
        hits.append(winner[5] == m["max_thickness"])
    assert np.mean(hits) < .88


def test_extreme_count_variants_and_fills():
    examples = sweep("52")
    modes = Counter((e["metadata"]["extreme_kind"], e["metadata"]["count_mode"]) for e in examples)
    assert len(modes) == 4 and min(modes.values()) > 180
    families = Counter(f for e in examples for f in e["metadata"]["interior_families"])
    assert all(families[f] >= 80 for f in ("empty", "solid", "bands", "blocks", "objects", "checker"))
    assert .02 < np.mean([e["output"] == "0" for e in examples]) < .08
    assert np.mean([e["metadata"]["wall_color_islands"] > 0 for e in examples]) > .15
    assert sum(e["metadata"].get("target_color") == e["metadata"]["extreme_rect"][4]
               and e["metadata"]["count_mode"] == "color" and int(e["output"]) > 0
               for e in examples) > 5
    assert .10 < np.mean([e["metadata"]["layout_mode"] == "aspect_contrast" for e in examples]) < .35
    for extreme in ("largest", "smallest"):
        answers = [int(e["output"]) for e in examples
                   if e["metadata"]["count_mode"] == "all" and e["metadata"]["extreme_kind"] == extreme]
        assert max(answers) > 100 and len(set(answers)) > 40
        assert np.mean(np.asarray(answers) <= 2) < .20


def test_reference_is_needed_and_scale_range_expands():
    ms = [e["metadata"] for e in sweep("54")]
    assert np.mean([bool(m["reference_compression_factors"]) for m in ms]) > .25
    assert max(m["scale"] for m in ms) >= 10
    assert np.mean([m["base_h"]*m["base_w"] <= 6 for m in ms]) < .45
    for ex in sweep("54"):
        m = ex["metadata"]
        if m["base_family"].startswith("silhouette_"):
            r, c, rr, cc = m["small_bbox"]
            assert np.any(np.asarray(ex["input"])[r:rr, c:cc] == m["bg"]), "silhouette collapsed into a rectangle"
    # A twice-expanded reference and six-times-expanded big patch imply 3.
    root = np.array([[2, 7], [7, 2]])
    small = np.kron(root, np.ones((2, 2), int))
    big = np.kron(root, np.ones((6, 6), int))
    a = np.zeros((18, 12), int)
    a[:4, :4], a[6:18, :12] = small, big
    assert set(pair_candidates(a).values()) == {3}
    from generators.block_scaling_54_55 import pair_readings
    assert {v[0] for v in pair_readings(a).values()} == {3}


def test_requested_compression_matters():
    ms = [e["metadata"] for e in sweep("55")]
    assert .35 < np.mean([m["retains_blocks"] for m in ms]) < .65
    assert np.mean([m["scale"] == min(m["valid_factors"]) for m in ms]) < .75
    assert max(m["scale"] for m in ms) >= 10
    assert len({m["content_style"] for m in ms}) >= 7
    assert any(m["out_h"] == 1 for m in ms) and any(m["out_w"] == 1 for m in ms)
    for ex in sweep("55"):
        a = np.asarray(ex["input"])
        for n in ex["metadata"]["valid_factors"]:
            prompt = get_prompt("55", {"scale": n})
            out = solve("55", a, prompt)
            assert np.array_equal(np.kron(out, np.ones((n, n), int)), a)


@pytest.mark.parametrize("tid", TASKS)
def test_production_validators_reject_tampered_answers(tid):
    from generators.frame_measurement_51_52 import validate_example
    from generators.block_scaling_54_55 import validate_pair, validate_compression
    ex = deepcopy(generate(tid, 0))
    if tid == "55":
        ex["output"][0][0] = (ex["output"][0][0]+1) % 10
        validator = validate_compression
    elif tid == "54":
        ex["output"] = str(int(ex["output"])+1)
        validator = validate_pair
    else:
        ex["output"] = str(int(ex["output"])+1)
        validator = lambda ex: validate_example(ex, thickness=tid == "51")
    with pytest.raises(ValueError):
        validator(ex)
