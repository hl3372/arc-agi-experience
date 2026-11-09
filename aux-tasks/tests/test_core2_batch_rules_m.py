"""Verifier tests for finite rule-inference and priority batch M tasks."""
from __future__ import annotations

import json
import os
import sys
from collections import deque

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prompts import get_prompt
from registry import generate
from sft_wrapper import format_single
from symbol_remap import palette_for_example


TASK_IDS = ["221", "222", "244", "245", "246", "247", "248", "257", "258", "259"]
N_SEEDS = 30

MOTION_RULES = {
    "motion_shift_north_2": (-2, 0),
    "motion_shift_south_2": (2, 0),
    "motion_shift_west_2": (0, -2),
    "motion_shift_east_2": (0, 2),
    "motion_shift_northwest_2": (-2, -2),
    "motion_shift_northeast_2": (-2, 2),
    "motion_shift_southwest_2": (2, -2),
    "motion_shift_southeast_2": (2, 2),
}

BORDER_LABELS = {
    "border_width_1": 1,
    "border_width_2": 2,
    "border_width_3": 3,
}

CANDIDATE_RULE_LABELS = [
    "rule_flip_horizontal",
    "rule_flip_vertical",
    "rule_rotate_180",
    "rule_transpose_main",
]

OBJECT_SELECTION_RULES = {
    "largest_object_recolor": ("largest_object", "recolor"),
    "smallest_object_recolor": ("smallest_object", "recolor"),
    "topmost_object_erase": ("topmost_object", "erase"),
    "leftmost_object_erase": ("leftmost_object", "erase"),
}

NEIGHBOR_PREDICATES = {
    "predicate_north_is_target": "north",
    "predicate_south_is_target": "south",
    "predicate_west_is_target": "west",
    "predicate_east_is_target": "east",
    "predicate_at_least_two_cardinal_targets": "at_least_two",
}


def _components(grid: np.ndarray, bg: int, color: int | None = None) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            value = int(grid[r, c])
            if value == bg or (color is not None and value != color):
                seen[r, c] = True
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = set()
            while q:
                cr, cc = q.popleft()
                cells.add((cr, cc))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == value
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rows = [p[0] for p in cells]
            cols = [p[1] for p in cells]
            comps.append({
                "color": value,
                "cells": cells,
                "bbox": (min(rows), min(cols), max(rows), max(cols)),
            })
    return sorted(comps, key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["color"]))


def _grid_rule(label: str, grid: np.ndarray) -> np.ndarray:
    if label == "rule_flip_horizontal":
        return np.fliplr(grid).copy()
    if label == "rule_flip_vertical":
        return np.flipud(grid).copy()
    if label == "rule_rotate_180":
        return np.rot90(grid, 2).copy()
    if label == "rule_transpose_main":
        return grid.T.copy()
    raise AssertionError(label)


def _predicate_mask(grid: np.ndarray, target_color: int, true_color: int, false_color: int, kind: str) -> np.ndarray:
    h, w = grid.shape
    out = np.full((h, w), false_color, dtype=np.uint8)
    for r in range(h):
        for c in range(w):
            flags = {
                "north": r > 0 and int(grid[r - 1, c]) == target_color,
                "south": r + 1 < h and int(grid[r + 1, c]) == target_color,
                "west": c > 0 and int(grid[r, c - 1]) == target_color,
                "east": c + 1 < w and int(grid[r, c + 1]) == target_color,
            }
            ok = sum(flags.values()) >= 2 if kind == "at_least_two" else flags[kind]
            if ok:
                out[r, c] = true_color
    return out


def _cell_rule_matches(grid: np.ndarray, r: int, c: int, rule: dict, bg: int, seed_color: int) -> bool:
    if rule["label"] == "cell_rule_seed_cell":
        return int(grid[r, c]) == seed_color
    if rule["label"] == "cell_rule_next_to_seed":
        h, w = grid.shape
        return any(
            0 <= r + dr < h and 0 <= c + dc < w and int(grid[r + dr, c + dc]) == seed_color
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
        )
    if rule["label"] == "cell_rule_foreground_cell":
        return int(grid[r, c]) != bg
    raise AssertionError(rule["label"])


def _apply_cell_rules(grid: np.ndarray, bg: int, seed_color: int, ordered_rules: list[dict]) -> np.ndarray:
    expected = grid.copy()
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            for rule in ordered_rules:
                if _cell_rule_matches(grid, r, c, rule, bg, seed_color):
                    expected[r, c] = rule["output_color"]
                    break
    return expected


def _object_rule_matches(comp: dict, rule: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if rule["label"] == "object_rule_large":
        return len(comp["cells"]) >= int(rule["threshold"])
    if rule["label"] == "object_rule_wide":
        return width > height
    if rule["label"] == "object_rule_tall":
        return height > width
    raise AssertionError(rule["label"])


def _apply_object_rules(grid: np.ndarray, bg: int, ordered_rules: list[dict]) -> np.ndarray:
    expected = grid.copy()
    for comp in _components(grid, bg):
        for rule in ordered_rules:
            if _object_rule_matches(comp, rule):
                for cell in comp["cells"]:
                    expected[cell] = rule["output_color"]
                break
    return expected


def _verify(task_id: str, fn):
    for seed in range(N_SEEDS):
        ex = generate(task_id, seed)
        json.dumps(ex)
        prompt = get_prompt(task_id, ex["metadata"])
        assert "[No prompt" not in prompt
        if seed in {0, 7, 13, 23}:
            palette = palette_for_example(task_id, seed)
            rendered_prompt, answer = format_single(task_id, seed, color_palette=palette)
            assert "[No prompt" not in rendered_prompt
            assert "Input:" in rendered_prompt
            assert isinstance(answer, str) and answer
        fn(seed, ex)
        assert generate(task_id, seed) == ex


def test_all_prompts_exist():
    for task_id in TASK_IDS:
        ex = generate(task_id, 0)
        assert "[No prompt" not in get_prompt(task_id, ex["metadata"])


def test_task_221():
    def check(seed, ex):
        before = np.array(ex["input"]["before"], dtype=np.uint8)
        after = np.array(ex["input"]["after"], dtype=np.uint8)
        bg = ex["metadata"]["background_color"]
        source = {
            (r, c, int(before[r, c]))
            for r in range(before.shape[0])
            for c in range(before.shape[1])
            if int(before[r, c]) != bg
        }
        target = {
            (r, c, int(after[r, c]))
            for r in range(after.shape[0])
            for c in range(after.shape[1])
            if int(after[r, c]) != bg
        }
        matches = []
        for label, (dr, dc) in MOTION_RULES.items():
            shifted = {(r + dr, c + dc, color) for r, c, color in source}
            if shifted == target:
                matches.append(label)
        assert matches == [ex["output"]], (seed, matches)
        assert before.tolist() != after.tolist(), seed
        assert len(source) >= 9, seed

    _verify("221", check)


def _matching_mapping_labels(ex: dict) -> list[str]:
    pairs = []
    if "examples" in ex["input"]:
        pairs = ex["input"]["examples"]
    else:
        pairs = [{"input": ex["input"]["before"], "output": ex["input"]["after"]}]
    observed = {}
    bg = ex["metadata"]["background_color"]
    for pair in pairs:
        before = np.array(pair["input"], dtype=np.uint8)
        after = np.array(pair["output"], dtype=np.uint8)
        for src, dst in zip(before.ravel(), after.ravel()):
            src_i = int(src)
            if src_i == bg:
                assert int(dst) == bg
                continue
            observed.setdefault(src_i, int(dst))
            assert observed[src_i] == int(dst)
    matches = []
    for table in ex["metadata"]["mapping_tables"]:
        table_map = {int(entry["from_color"]): int(entry["to_color"]) for entry in table["entries"]}
        if table_map == observed:
            matches.append(table["label"])
    return matches


def test_task_222():
    _verify("222", lambda seed, ex: assert_mapping(seed, ex))


def assert_mapping(seed, ex):
    matches = _matching_mapping_labels(ex)
    assert matches == [ex["output"]], (seed, matches)
    assert ex["output"].startswith("mapping_")


def test_task_244():
    def check(seed, ex):
        labels = []
        for pair in ex["input"]["examples"]:
            inp = np.array(pair["input"], dtype=np.uint8)
            out = np.array(pair["output"], dtype=np.uint8)
            dh = inp.shape[0] - out.shape[0]
            dw = inp.shape[1] - out.shape[1]
            assert dh == dw and dh % 2 == 0, seed
            k = dh // 2
            assert out.tolist() == inp[k : inp.shape[0] - k, k : inp.shape[1] - k].tolist(), seed
            labels.append(next(label for label, width in BORDER_LABELS.items() if width == k))
        assert len(set(labels)) == 1
        assert labels[0] == ex["output"], seed

    _verify("244", check)


def test_task_245():
    def check(seed, ex):
        matches = []
        for label in CANDIDATE_RULE_LABELS:
            if all(
                _grid_rule(label, np.array(pair["input"], dtype=np.uint8)).tolist() == pair["output"]
                for pair in ex["input"]["examples"]
            ):
                matches.append(label)
        assert matches == [ex["output"]], (seed, matches)
        assert set(ex["input"]["rule_labels"]) == set(CANDIDATE_RULE_LABELS)

    _verify("245", check)


def _apply_selection_rule(grid: np.ndarray, meta: dict, label: str) -> np.ndarray:
    selector, action = OBJECT_SELECTION_RULES[label]
    comps = _components(grid, meta["background_color"], color=meta["object_color"])
    if selector == "largest_object":
        selected = max(comps, key=lambda comp: (len(comp["cells"]), -comp["bbox"][0], -comp["bbox"][1]))
    elif selector == "smallest_object":
        selected = min(comps, key=lambda comp: (len(comp["cells"]), comp["bbox"][0], comp["bbox"][1]))
    elif selector == "topmost_object":
        selected = min(comps, key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))
    elif selector == "leftmost_object":
        selected = min(comps, key=lambda comp: (comp["bbox"][1], comp["bbox"][0]))
    else:
        raise AssertionError(selector)
    expected = grid.copy()
    replacement = meta["background_color"] if action == "erase" else meta["action_color"]
    for cell in selected["cells"]:
        expected[cell] = replacement
    return expected


def test_task_246():
    def check(seed, ex):
        matches = []
        for label in OBJECT_SELECTION_RULES:
            ok = True
            for pair in ex["input"]["examples"]:
                grid = np.array(pair["input"], dtype=np.uint8)
                expected = _apply_selection_rule(grid, ex["metadata"], label)
                ok = ok and expected.tolist() == pair["output"]
            if ok:
                matches.append(label)
        assert matches == [ex["output"]], (seed, matches)
        for pair in ex["input"]["examples"]:
            comps = _components(np.array(pair["input"], dtype=np.uint8), ex["metadata"]["background_color"], color=ex["metadata"]["object_color"])
            assert 3 <= len(comps) <= 5
            # distinct cell-counts so largest/smallest are unambiguous singletons
            assert len({len(comp["cells"]) for comp in comps}) == len(comps)

    _verify("246", check)
    ex = generate("246", 0)
    ex["metadata"]["object_selection_rules"][0]["label"] = "__bad_label__"
    assert generate("246", 0)["metadata"]["object_selection_rules"][0]["label"] != "__bad_label__"


def test_task_247():
    _verify("247", lambda seed, ex: assert_mapping(seed, ex))


def test_task_248():
    def check(seed, ex):
        grid = np.array(ex["input"]["example_input"], dtype=np.uint8)
        mask = np.array(ex["input"]["example_output"], dtype=np.uint8)
        meta = ex["metadata"]
        matches = []
        rendered_masks = []
        for label, kind in NEIGHBOR_PREDICATES.items():
            expected = _predicate_mask(grid, meta["target_color"], meta["true_color"], meta["false_color"], kind)
            rendered_masks.append(expected.tobytes())
            if expected.tolist() == mask.tolist():
                matches.append(label)
        assert matches == [ex["output"]], (seed, matches)
        assert len(set(rendered_masks)) == len(rendered_masks)
        assert np.any(mask == meta["true_color"]) and np.any(mask == meta["false_color"])

    _verify("248", check)


def test_task_257():
    def check(seed, ex):
        grid = np.array(ex["input"], dtype=np.uint8)
        out = np.array(ex["output"], dtype=np.uint8)
        meta = ex["metadata"]
        expected = _apply_cell_rules(grid, meta["background_color"], meta["seed_color"], meta["ordered_cell_rules"])
        assert out.tolist() == expected.tolist(), seed
        conflict_cells = []
        priority_matters = False
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                matches = [
                    rule
                    for rule in meta["ordered_cell_rules"]
                    if _cell_rule_matches(grid, r, c, rule, meta["background_color"], meta["seed_color"])
                ]
                if len(matches) > 1:
                    conflict_cells.append((r, c))
                    first_color = matches[0]["output_color"]
                    priority_matters |= any(rule["output_color"] != first_color for rule in matches[1:])
                    assert int(out[r, c]) == first_color
        assert conflict_cells, seed
        assert priority_matters, seed

    _verify("257", check)


def test_task_258():
    def check(seed, ex):
        before = np.array(ex["input"]["input"], dtype=np.uint8)
        after = np.array(ex["input"]["output"], dtype=np.uint8)
        meta = ex["metadata"]
        by_label = {rule["label"]: rule for rule in meta["cell_rules"]}
        matches = []
        outputs = []
        for candidate in meta["candidate_priority_orders"]:
            ordered = [by_label[label] for label in candidate["order"]]
            expected = _apply_cell_rules(before, meta["background_color"], meta["seed_color"], ordered)
            outputs.append(expected.tobytes())
            if expected.tolist() == after.tolist():
                matches.append(candidate["label"])
        assert matches == [ex["output"]], (seed, matches)
        assert len(set(outputs)) == len(outputs), seed
        conflict_cells = 0
        for r in range(before.shape[0]):
            for c in range(before.shape[1]):
                count = sum(
                    _cell_rule_matches(before, r, c, rule, meta["background_color"], meta["seed_color"])
                    for rule in meta["cell_rules"]
                )
                conflict_cells += int(count > 1)
        assert conflict_cells >= 3, seed

    _verify("258", check)
    ex = generate("258", 0)
    ex["metadata"]["candidate_priority_orders"][0]["label"] = "__bad_label__"
    assert generate("258", 0)["metadata"]["candidate_priority_orders"][0]["label"] != "__bad_label__"


def test_task_259():
    def check(seed, ex):
        grid = np.array(ex["input"], dtype=np.uint8)
        out = np.array(ex["output"], dtype=np.uint8)
        meta = ex["metadata"]
        expected = _apply_object_rules(grid, meta["background_color"], meta["ordered_object_rules"])
        assert out.tolist() == expected.tolist(), seed
        comps = _components(grid, meta["background_color"])
        assert 4 <= len(comps) <= 7, seed
        conflict_objects = 0
        priority_matters = False
        for comp in comps:
            matches = [rule for rule in meta["ordered_object_rules"] if _object_rule_matches(comp, rule)]
            if len(matches) > 1:
                conflict_objects += 1
                first_color = matches[0]["output_color"]
                priority_matters |= any(rule["output_color"] != first_color for rule in matches[1:])
                for cell in comp["cells"]:
                    assert int(out[cell]) == first_color
        assert conflict_objects >= 2, seed
        assert priority_matters, seed

    _verify("259", check)


if __name__ == "__main__":
    tests = [
        test_all_prompts_exist,
        test_task_221,
        test_task_222,
        test_task_244,
        test_task_245,
        test_task_246,
        test_task_247,
        test_task_248,
        test_task_257,
        test_task_258,
        test_task_259,
    ]
    for fn in tests:
        fn()
        print(f"  PASS  {fn.__name__}")
