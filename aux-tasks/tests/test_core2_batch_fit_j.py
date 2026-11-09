"""Verifier tests for fit and candidate-selection batch J tasks."""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, deque

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prompts import get_prompt
from registry import generate
from sft_wrapper import format_single
from symbol_remap import palette_for_example


TASK_IDS = ["230", "231", "249", "250", "251", "252", "253", "254", "255", "256"]
N_SEEDS = 30


def _shape(meta: dict, key: str = "source_shape") -> frozenset[tuple[int, int]]:
    return frozenset(tuple(p) for p in meta[key])


def _normalize(cells) -> frozenset[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return frozenset((r - min_r, c - min_c) for r, c in cells)


def _transform(shape, name: str) -> frozenset[tuple[int, int]]:
    if name == "identity":
        cells = set(shape)
    elif name == "rotate90":
        cells = {(c, -r) for r, c in shape}
    elif name == "rotate180":
        cells = {(-r, -c) for r, c in shape}
    elif name == "rotate270":
        cells = {(-c, r) for r, c in shape}
    elif name == "reflect_horizontal":
        cells = {(r, -c) for r, c in shape}
    elif name == "reflect_vertical":
        cells = {(-r, c) for r, c in shape}
    else:
        raise AssertionError(name)
    return _normalize(cells)


def _orientations(shape, allowed: list[str]) -> list[frozenset[tuple[int, int]]]:
    out = []
    seen = set()
    for name in allowed:
        orient = _transform(shape, name)
        if orient not in seen:
            seen.add(orient)
            out.append(orient)
    return out


def _abs(shape, anchor):
    ar, ac = anchor
    return {(ar + r, ac + c) for r, c in shape}


def _in_bounds(cells, h, w):
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _components(grid: np.ndarray, color: int) -> list[set[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = set()
            while q:
                cr, cc = q.popleft()
                comp.add((cr, cc))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _valid_anchor(grid, shape, anchor, meta):
    h, w = grid.shape
    bg = meta["background_color"]
    anchor_color = meta["anchor_color"]
    for orient in _orientations(shape, meta["allowed_transforms"]):
        cells = _abs(orient, anchor)
        if _in_bounds(cells, h, w) and all(int(grid[r, c]) in (bg, anchor_color) for r, c in cells):
            return orient
    return None


def _apply_placement(grid, shape, source_cells, anchor, meta):
    out = grid.copy()
    if meta["copy_move_mode"] == "move":
        for r, c in source_cells:
            out[r, c] = meta["background_color"]
    orient = _valid_anchor(grid, shape, anchor, meta)
    assert orient is not None
    for r, c in _abs(orient, anchor):
        out[r, c] = meta["object_color"]
    return out


def _verify(task_id: str, fn):
    for seed in range(N_SEEDS):
        ex = generate(task_id, seed)
        json.dumps(ex)
        prompt = get_prompt(task_id, ex["metadata"])
        assert "[No prompt" not in prompt
        if seed in {0, 7, 13}:
            palette = palette_for_example(task_id, seed)
            rendered_prompt, _answer = format_single(task_id, seed, color_palette=palette)
            assert "[No prompt" not in rendered_prompt
        fn(seed, ex)
        assert generate(task_id, seed) == ex


def test_task_230():
    def check(seed, ex):
        grid = np.array(ex["input"], dtype=np.uint8)
        out = np.array(ex["output"], dtype=np.uint8)
        meta = ex["metadata"]
        count_a = int(np.count_nonzero(grid == meta["set_a_color"]))
        count_b = int(np.count_nonzero(grid == meta["set_b_color"]))
        if count_a > count_b:
            categories = {"greater_count": [meta["set_a_color"]], "lesser_count": [meta["set_b_color"]], "tied": []}
        elif count_b > count_a:
            categories = {"greater_count": [meta["set_b_color"]], "lesser_count": [meta["set_a_color"]], "tied": []}
        else:
            categories = {"greater_count": [], "lesser_count": [], "tied": [meta["set_a_color"], meta["set_b_color"]]}
        fallback = {
            "set_A": [meta["set_a_color"]],
            "set_B": [meta["set_b_color"]],
            "both_sets": [meta["set_a_color"], meta["set_b_color"]],
            "neither_set": [],
        }[meta["fallback_category"]]
        selected = categories[meta["target_category"]] or fallback
        expected = grid.copy()
        replacement = meta["action_color"] if meta["action_type"] == "recolor" else meta["background_color"]
        for color in selected:
            expected[grid == color] = replacement
        assert selected, seed
        assert expected.tolist() != grid.tolist(), seed
        assert out.tolist() == expected.tolist(), seed
    _verify("230", check)


def test_task_231():
    def check(seed, ex):
        grid = np.array(ex["input"], dtype=np.uint8)
        out = np.array(ex["output"], dtype=np.uint8)
        meta = ex["metadata"]
        outer = grid.shape[0] if meta["axis"] == "row" else grid.shape[1]
        colors = []
        for i in range(outer):
            line = grid[i, :] if meta["axis"] == "row" else grid[:, i]
            ca = int(np.count_nonzero(line == meta["set_a_color"]))
            cb = int(np.count_nonzero(line == meta["set_b_color"]))
            colors.append(meta["greater_color"] if ca > cb else meta["less_color"] if ca < cb else meta["equal_color"])
        if meta["output_form"] == "summary_strip":
            expected = np.array(colors, dtype=np.uint8).reshape((outer, 1) if meta["axis"] == "row" else (1, outer))
        elif meta["output_form"] == "paint_full_lines":
            expected = np.full_like(grid, meta["background_color"])
            for i, color in enumerate(colors):
                if meta["axis"] == "row":
                    expected[i, :] = color
                else:
                    expected[:, i] = color
        else:
            expected = np.full_like(grid, meta["background_color"])
            for i, color in enumerate(colors):
                if meta["axis"] == "row":
                    expected[i, 0] = color
                else:
                    expected[0, i] = color
        assert out.tolist() == expected.tolist(), seed
    _verify("231", check)


def _check_anchor_placement(seed, ex, extra_ok=None):
    grid = np.array(ex["input"], dtype=np.uint8)
    out = np.array(ex["output"], dtype=np.uint8)
    meta = ex["metadata"]
    shape = _shape(meta)
    source_cells = _abs(shape, tuple(meta["source_anchor"]))
    valid = []
    for anchor in [tuple(p) for p in meta["candidate_anchors"]]:
        orient = _valid_anchor(grid, shape, anchor, meta)
        if orient is not None and (extra_ok is None or extra_ok(grid, orient, anchor, meta)):
            valid.append(anchor)
    assert len(valid) == 1, (seed, valid)
    expected = _apply_placement(grid, shape, source_cells, valid[0], meta)
    assert out.tolist() == expected.tolist(), seed


def _count_fitting_orientations(grid, shape, anchor, meta):
    h, w = grid.shape
    bg = meta["background_color"]
    anchor_color = meta["anchor_color"]
    count = 0
    for orient in _orientations(shape, meta["allowed_transforms"]):
        cells = _abs(orient, anchor)
        if _in_bounds(cells, h, w) and all(int(grid[r, c]) in (bg, anchor_color) for r, c in cells):
            count += 1
    return count


def test_task_249():
    def check(seed, ex):
        _check_anchor_placement(seed, ex)
        # The placed orientation must be uniquely determined: exactly one allowed
        # transform may fit at the selected anchor, otherwise the answer is ambiguous.
        grid = np.array(ex["input"], dtype=np.uint8)
        meta = ex["metadata"]
        shape = _shape(meta)
        good_anchor = tuple(meta["candidate_anchors"][0])
        assert _count_fitting_orientations(grid, shape, good_anchor, meta) == 1, seed
    _verify("249", check)


def test_task_250():
    def check(seed, ex):
        grid = np.array(ex["input"], dtype=np.uint8)
        out = np.array(ex["output"], dtype=np.uint8)
        meta = ex["metadata"]
        shape = _shape(meta)
        source_cells = _abs(shape, tuple(meta["source_anchor"]))
        matches = [
            comp for comp in _components(grid, meta["mask_color"])
            if _normalize(comp) in _orientations(shape, meta["allowed_transforms"])
        ]
        assert len(matches) == 1, (seed, matches)
        assert meta["region_type"] in {"mask_region", "hole", "frame_interior"}
        expected = grid.copy()
        if meta["copy_move_mode"] == "move":
            for r, c in source_cells:
                expected[r, c] = meta["background_color"]
        for r, c in matches[0]:
            expected[r, c] = meta["object_color"]
        assert out.tolist() == expected.tolist(), seed
    _verify("250", check)


def test_task_251():
    def aligned(_grid, _orient, anchor, meta):
        guide = tuple(meta["guide_position"])
        relation = meta["alignment_constraint"]
        return (
            (relation == "same_row_as_guide" and anchor[0] == guide[0])
            or (relation == "same_column_as_guide" and anchor[1] == guide[1])
            or (relation == "above_guide_same_column" and anchor[1] == guide[1] and anchor[0] < guide[0])
            or (relation == "left_of_guide_same_row" and anchor[0] == guide[0] and anchor[1] < guide[1])
        )
    _verify("251", lambda seed, ex: _check_anchor_placement(seed, ex, aligned))


def test_task_252():
    def contact(grid, orient, anchor, meta):
        dr, dc = {
            "north": (-1, 0),
            "south": (1, 0),
            "west": (0, -1),
            "east": (0, 1),
        }[meta["contact_side"]]
        h, w = grid.shape
        return any(
            0 <= r + dr < h and 0 <= c + dc < w and int(grid[r + dr, c + dc]) == meta["contact_color"]
            for r, c in _abs(orient, anchor)
        )
    _verify("252", lambda seed, ex: _check_anchor_placement(seed, ex, contact))


def _component_at(grid, color, origin):
    for comp in _components(grid, color):
        if origin in comp:
            return comp
    raise AssertionError(origin)


def _object_match(ref_shape, cand_shape, relation):
    if relation == "same_shape":
        return cand_shape == ref_shape
    if relation == "same_shape_under_rotation":
        return cand_shape in _orientations(ref_shape, ["identity", "rotate90", "rotate180", "rotate270"])
    if relation == "same_shape_under_reflection":
        return cand_shape in _orientations(ref_shape, ["identity", "reflect_horizontal", "reflect_vertical"])
    if relation == "same_cell_count":
        return len(cand_shape) == len(ref_shape)
    raise AssertionError(relation)


def test_task_253():
    def check(seed, ex):
        grid = np.array(ex["input"], dtype=np.uint8)
        meta = ex["metadata"]
        ref_shape = _shape(meta, "reference_shape")
        winners = []
        for item in meta["candidate_labels"]:
            comp = _component_at(grid, meta["candidate_color"], tuple(item["origin"]))
            if _object_match(ref_shape, _normalize(comp), meta["equivalence_relation"]):
                winners.append(item["label"])
        assert winners == [ex["output"]], (seed, winners)
    _verify("253", check)


def test_task_254():
    def check(seed, ex):
        grid = np.array(ex["input"], dtype=np.uint8)
        meta = ex["metadata"]
        r0, c0, r1, c1 = meta["reference_bbox"]
        ref = grid[r0:r1 + 1, c0:c1 + 1]
        winners = []
        for item, box in zip(meta["candidate_labels"], meta["candidate_bboxes"]):
            br0, bc0, br1, bc1 = box
            cand = grid[br0:br1 + 1, bc0:bc1 + 1]
            if meta["match_relation"] == "exact_contents":
                ok = np.array_equal(cand, ref)
            elif meta["match_relation"] == "rotate_180_contents":
                ok = np.array_equal(cand, np.rot90(ref, 2))
            else:
                ok = sorted(int(v) for v in cand.ravel()) == sorted(int(v) for v in ref.ravel())
            if ok:
                winners.append(item["label"])
        assert winners == [ex["output"]], (seed, winners)
    _verify("254", check)


def test_task_255():
    def check(seed, ex):
        meta = ex["metadata"]
        ref = tuple(meta["reference_position"])
        positions = [tuple(item["position"]) for item in meta["candidate_labels"]]
        labels = [item["label"] for item in meta["candidate_labels"]]
        relation = meta["spatial_relation"]
        if relation == "nearest_to_reference":
            scores = [abs(r - ref[0]) + abs(c - ref[1]) for r, c in positions]
            winner = labels[int(np.argmin(scores))]
        elif relation == "farthest_from_reference":
            scores = [abs(r - ref[0]) + abs(c - ref[1]) for r, c in positions]
            winner = labels[int(np.argmax(scores))]
        elif relation == "same_row_right_of_reference":
            hits = [label for label, (r, c) in zip(labels, positions) if r == ref[0] and c > ref[1]]
            assert len(hits) == 1, seed
            winner = hits[0]
        else:
            hits = [label for label, (r, c) in zip(labels, positions) if c == ref[1] and r > ref[0]]
            assert len(hits) == 1, seed
            winner = hits[0]
        assert ex["output"] == winner, seed
    _verify("255", check)
    counts = Counter(generate("255", seed)["output"] for seed in range(200))
    assert set(counts) == {"A", "B", "C"}, counts


def test_task_256():
    def check(seed, ex):
        grid = np.array(ex["input"], dtype=np.uint8)
        meta = ex["metadata"]
        target = _shape(meta, "target_shape")
        winners = []
        for item in meta["candidate_labels"]:
            comp = _component_at(grid, meta["piece_color"], tuple(item["origin"]))
            cand_shape = _normalize(comp)
            if target in _orientations(cand_shape, meta["allowed_transforms"]):
                winners.append(item["label"])
        assert winners == [ex["output"]], (seed, winners)
    _verify("256", check)


if __name__ == "__main__":
    for fn in [
        test_task_230,
        test_task_231,
        test_task_249,
        test_task_250,
        test_task_251,
        test_task_252,
        test_task_253,
        test_task_254,
        test_task_255,
        test_task_256,
    ]:
        fn()
        print(f"  PASS  {fn.__name__}")
