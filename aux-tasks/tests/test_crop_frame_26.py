"""Task26: unnamed frame determinability, crop range, coherent scene coverage.

The oracle enumerates rectangles using integral images.  It does not call the
generator's run/corner frame detector or its component checker.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from functools import lru_cache

import numpy as np
import pytest

from generators import crop_frame_26 as G


@lru_cache(maxsize=100)
def _rectangles(h, w):
    rows = [(r, b) for r in range(h - 2) for b in range(r + 2, h)]
    cols = [(c, d) for c in range(w - 2) for d in range(c + 2, w)]
    return np.array([(r, c, b, d) for r, b in rows for c, d in cols], dtype=int).T


def _oracle_frames(grid):
    a = np.asarray(grid)
    h, w = a.shape
    r, c, b, d = _rectangles(h, w)
    candidates = []
    for color in np.unique(a):
        m = a == color
        p = np.pad(m.cumsum(0).cumsum(1), ((1, 0), (1, 0)))

        def area(r0, c0, r1, c1):
            return p[r1, c1] - p[r0, c1] - p[r1, c0] + p[r0, c0]

        inside = area(r + 1, c + 1, b, d)
        boundary = area(r, c, b + 1, d + 1) - inside
        core = area(r + 2, c + 2, np.maximum(r + 2, b - 1), np.maximum(c + 2, d - 1))
        thin = inside - core == 0
        hit = np.flatnonzero((boundary == 2 * ((b - r + 1) + (d - c + 1)) - 4) & thin)
        for i in hit:
            top, left, bot, right = int(r[i]), int(c[i]), int(b[i]), int(d[i])
            extra_wall = any((
                top > 0 and bool(m[top - 1, left:right + 1].all()),
                bot < h - 1 and bool(m[bot + 1, left:right + 1].all()),
                left > 0 and bool(m[top:bot + 1, left - 1].all()),
                right < w - 1 and bool(m[top:bot + 1, right + 1].all()),
            ))
            if not extra_wall:
                candidates.append((top, left, bot - top + 1, right - left + 1, int(color)))
    return sorted(candidates)


@pytest.fixture(scope="module")
def examples():
    return [G.generate_extract_subgrid(np.random.default_rng(s)) for s in range(400)]


def test_task26_matches_independent_visible_grid_oracle(examples):
    for seed, ex in enumerate(examples):
        found = _oracle_frames(ex["input"])
        assert len(found) == 1, (seed, found)
        r, c, h, w, color = found[0]
        expected = np.asarray(ex["input"])[r + 1:r + h - 1, c + 1:c + w - 1].tolist()
        assert expected == ex["output"], seed
        assert list(G.find_frames(ex["input"])[0]) == list(found[0]), seed
        assert len(np.unique(ex["input"])) == ex["metadata"]["n_colors"]
        assert color == ex["metadata"]["roi_color"]
        G.validate_example(ex)


def test_task26_frame_color_stays_unnamed_and_rule_is_constant():
    text = G.FRAME_DEFINITION
    assert "unique rectangular frame" in text
    assert "one-cell-thick" in text
    assert "strictly inside" in text and "excluding its border" in text
    assert not any(str(c) in text for c in range(10))
    assert "background" not in text and "largest" not in text and "outermost" not in text


def test_task26_input_output_palette_and_family_coverage(examples):
    ms = [e["metadata"] for e in examples]
    h, w = np.array([m["grid_h"] for m in ms]), np.array([m["grid_w"] for m in ms])
    ih, iw = np.array([m["crop_h"] for m in ms]), np.array([m["crop_w"] for m in ms])
    assert set(h) == set(range(3, 31)) and set(w) == set(range(3, 31))
    assert abs(np.corrcoef(h, w)[0, 1]) < .16
    assert (h == w).mean() < .09
    assert (h * w >= 700).mean() >= .025
    assert max(Counter(h).values()) < .09 * len(ms)
    assert min(ih) == min(iw) == 1 and max(ih) == max(iw) == 28
    assert .08 <= ((ih == 1) | (iw == 1)).mean() <= .40
    frac = ih * iw / (h * w)
    assert (frac >= .5).mean() >= .12
    assert (frac >= .75).mean() >= .025
    colors = Counter(m["n_colors"] for m in ms)
    assert set(colors) == set(range(2, 11))
    assert max(colors.values()) < .20 * len(ms), colors
    assert .18 <= sum(colors[k] for k in (8, 9, 10)) / len(ms) <= .42
    assert .05 <= sum(m["roi_color"] == 0 for m in ms) / len(ms) <= .16
    assert set(m["family"] for m in ms) == set(G.SCENE_FAMILIES)
    assert set(m["crop_mode"] for m in ms) == {"full", "large", "varied", "strip"}
    assert np.mean([m["attempts"] for m in ms]) < 3
    assert sum(bool(m["object_kinds"]) for m in ms) >= .10 * len(ms)
    assert .20 <= sum(m["frame_color_reused"] for m in ms) / len(ms) <= .45
    assert any(len(np.unique(e["output"])) == 10 for e in examples)


def _frame(a, r, c, h, w, color):
    a[r, c:c + w] = a[r + h - 1, c:c + w] = color
    a[r:r + h, c] = a[r:r + h, c + w - 1] = color


def test_task26_prior_dual_exclusive_frame_seed57291_is_rejected():
    grid = [[int(x) for x in row] for row in
            ("0999999", "8966629", "8968639", "8966609", "0999999", "0800330", "0000300")]
    assert len(_oracle_frames(grid)) == 2
    assert len(G.find_frames(grid)) == 2
    with pytest.raises(ValueError, match="exactly one"):
        G.solve_extract_subgrid(grid)


def test_task26_optional_content_never_forces_an_extra_frame_seed6124():
    ex = G.generate_extract_subgrid(np.random.default_rng(6124))
    assert ex["metadata"]["n_colors"] == 2
    assert (ex["metadata"]["crop_h"], ex["metadata"]["crop_w"]) == (4, 3)
    assert not ex["metadata"]["frame_color_reused"]
    assert len(_oracle_frames(ex["input"])) == 1
    G.validate_example(ex)


def test_task26_another_ring_counts_when_its_color_recurs_or_has_a_tail():
    a = np.zeros((12, 16), dtype=int)
    _frame(a, 1, 1, 5, 5, 1)
    _frame(a, 5, 9, 5, 5, 2)
    a[1, 13:15] = 2  # The alternate frame colour is not globally exclusive.
    a[9:12, 11] = 2  # A same-colour outward tail attaches to that ring.
    assert len(G._component_frames(a)) == 1
    assert len(_oracle_frames(a)) == len(G.find_frames(a)) == 2
    with pytest.raises(ValueError, match="exactly one"):
        G.solve_extract_subgrid(a)


def test_task26_filled_and_thick_rectangles_are_not_frames():
    a = np.zeros((12, 16), dtype=int)
    a[1:6, 1:6] = 1
    a[3:10, 8:15] = 2
    a[5:8, 10:13] = 0  # A two-cell-thick hollow block, not a thin frame.
    assert G.find_frames(a) == []
    assert _oracle_frames(a) == []


@pytest.mark.parametrize("interior", [np.array([[8]]), np.array([[2, 4, 2, 3]]), np.array([[1], [3], [1]])])
def test_task26_one_cell_wide_interiors_are_valid(interior):
    h, w = interior.shape
    a = np.full((h + 2, w + 2), 7, dtype=int)
    a[1:-1, 1:-1] = interior
    assert G.solve_extract_subgrid(a) == interior.tolist()


def test_task26_validator_raises_when_visible_answer_or_frame_is_tampered(examples):
    ex = deepcopy(examples[0])
    ex["output"][0][0] = (ex["output"][0][0] + 1) % 10
    with pytest.raises(ValueError, match="output"):
        G.validate_example(ex)
    ex = deepcopy(examples[0])
    r, c, _, _ = ex["metadata"]["frame_bbox"]
    ex["input"][r][c] = (ex["input"][r][c] + 1) % 10
    with pytest.raises(ValueError):
        G.validate_example(ex)
    ex = deepcopy(examples[0])
    ex["metadata"]["frame_bbox"][0] += 1
    with pytest.raises(ValueError, match="planned frame"):
        G.validate_example(ex)


def test_task26_retry_preserves_size_palette_crop_and_family(monkeypatch):
    original = G._render_scene
    calls = []

    def fail_first_three(rng, free, colors, family):
        calls.append((free.shape, int(free.sum()), tuple(colors), family))
        if len(calls) <= 3:
            return None
        return original(rng, free, colors, family)

    monkeypatch.setattr(G, "_render_scene", fail_first_three)
    ex = G.generate_extract_subgrid(np.random.default_rng(42))
    assert ex["metadata"]["attempts"] >= 4
    assert len(set(calls)) == 1


def test_task26_exhaustion_raises_instead_of_substituting_a_case(monkeypatch):
    monkeypatch.setattr(G, "_render_scene", lambda *args: None)
    with pytest.raises(RuntimeError, match="fixed"):
        G.generate_extract_subgrid(np.random.default_rng(0))
