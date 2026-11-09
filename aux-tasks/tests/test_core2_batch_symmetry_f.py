"""Verifier tests for symmetry/comparison batch F second-batch tasks."""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.grid import NUM_COLORS
from prompts import get_prompt
from registry import generate
from sft_wrapper import format_single
from symbol_remap import palette_for_example


TASK_IDS = ["103", "104", "107", "120", "127", "130", "281"]
N_SEEDS = 30


def _is_symmetric(grid: np.ndarray, label: str) -> bool:
    if label == "horizontal_reflection":
        return np.array_equal(grid, np.flipud(grid))
    if label == "vertical_reflection":
        return np.array_equal(grid, np.fliplr(grid))
    if label == "main_diagonal_reflection":
        return grid.shape[0] == grid.shape[1] and np.array_equal(grid, grid.T)
    if label == "anti_diagonal_reflection":
        return grid.shape[0] == grid.shape[1] and np.array_equal(grid, np.fliplr(np.flipud(grid)).T)
    if label == "rotational_180":
        return np.array_equal(grid, np.rot90(grid, 2))
    raise AssertionError(f"unknown label {label}")


def _oriented_shape_and_decoration(turns: int, base=None, decoration=None):
    shape = set(base) if base is not None else {(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)}
    deco = set(decoration) if decoration is not None else {(0, 0), (2, 2)}
    for _ in range((turns // 90) % 4):
        shape = {(c, -r) for r, c in shape}
        deco = {(c, -r) for r, c in deco}
    min_r = min(r for r, _ in shape)
    min_c = min(c for _, c in shape)
    return (
        {(r - min_r, c - min_c) for r, c in shape},
        {(r - min_r, c - min_c) for r, c in deco},
    )


def _verify(task_id: str, fn):
    for seed in range(N_SEEDS):
        ex = generate(task_id, seed)
        json.dumps(ex)
        prompt = get_prompt(task_id, ex["metadata"])
        assert "[No prompt" not in prompt
        if seed < 4:
            formatted, answer = format_single(
                task_id,
                seed,
                color_palette=palette_for_example(task_id, seed),
            )
            assert formatted and answer
        fn(seed, ex)
        assert generate(task_id, seed) == ex


def test_task_103():
    def in_scene(meta, r: int, c: int) -> bool:
        if meta["separator_axis"] == "horizontal":
            if meta["legend_side"] == "top":
                return r > meta["separator_row"]
            return r < meta["separator_row"]
        if meta["legend_side"] == "left":
            return c > meta["separator_col"]
        return c < meta["separator_col"]

    def in_legend(meta, r: int, c: int) -> bool:
        if meta["separator_axis"] == "horizontal":
            if meta["legend_side"] == "top":
                return r < meta["separator_row"]
            return r > meta["separator_row"]
        if meta["legend_side"] == "left":
            return c < meta["separator_col"]
        return c > meta["separator_col"]

    def measure_value(meta, cells) -> int:
        rows = [p[0] for p in cells]
        cols = [p[1] for p in cells]
        if meta["measure_dimension"] == "width":
            return max(cols) - min(cols) + 1
        if meta["measure_dimension"] == "cell_count":
            return len(cells)
        return max(rows) - min(rows) + 1

    def normalized_shape(cells) -> frozenset[tuple[int, int]]:
        min_r = min(r for r, _ in cells)
        min_c = min(c for _, c in cells)
        return frozenset((r - min_r, c - min_c) for r, c in cells)

    def is_rectangle(shape: frozenset[tuple[int, int]]) -> bool:
        max_r = max(r for r, _ in shape)
        max_c = max(c for _, c in shape)
        return len(shape) == (max_r + 1) * (max_c + 1)

    def check(seed, ex):
        grid = np.array(ex["input"], dtype=np.uint8)
        out = np.array(ex["output"], dtype=np.uint8)
        meta = ex["metadata"]
        mapping = {
            int(item["measure_value"]): int(item["output_color"])
            for item in meta["legend_entries"]
        }
        assert set(mapping.keys()) == {int(item["measure_value"]) for item in meta["legend_entries"]}, seed
        seen_legend = np.zeros(grid.shape, dtype=bool)
        legend_components = 0
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if (
                    seen_legend[r, c]
                    or not in_legend(meta, r, c)
                    or int(grid[r, c]) != meta["object_color"]
                ):
                    continue
                legend_components += 1
                stack = [(r, c)]
                seen_legend[r, c] = True
                while stack:
                    cr, cc = stack.pop()
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if (
                            0 <= nr < grid.shape[0]
                            and 0 <= nc < grid.shape[1]
                            and not seen_legend[nr, nc]
                            and in_legend(meta, nr, nc)
                            and int(grid[nr, nc]) == meta["object_color"]
                        ):
                            seen_legend[nr, nc] = True
                            stack.append((nr, nc))
        assert legend_components == 3, seed
        expected = grid.copy()
        seen = np.zeros(grid.shape, dtype=bool)
        scene_objects = []
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if seen[r, c] or not in_scene(meta, r, c) or int(grid[r, c]) != meta["object_color"]:
                    continue
                stack = [(r, c)]
                seen[r, c] = True
                cells = []
                while stack:
                    cr, cc = stack.pop()
                    cells.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if (
                            0 <= nr < grid.shape[0]
                            and 0 <= nc < grid.shape[1]
                            and not seen[nr, nc]
                            and in_scene(meta, nr, nc)
                            and int(grid[nr, nc]) == meta["object_color"]
                        ):
                            seen[nr, nc] = True
                            stack.append((nr, nc))
                measure = measure_value(meta, cells)
                scene_objects.append({
                    "measure": measure,
                    "shape": normalized_shape(cells),
                })
                for rr, cc in cells:
                    expected[rr, cc] = mapping[measure]
        assert out.tolist() == expected.tolist(), seed
        assert len(scene_objects) == 4, seed
        measures = {obj["measure"] for obj in scene_objects}
        assert measures == set(mapping.keys()), seed
        repeated = {
            measure: [obj["shape"] for obj in scene_objects if obj["measure"] == measure]
            for measure in measures
        }
        repeated = {
            measure: shapes
            for measure, shapes in repeated.items()
            if len(shapes) > 1
        }
        assert len(repeated) == 1, seed
        duplicate_shapes = next(iter(repeated.values()))
        assert len(set(duplicate_shapes)) == len(duplicate_shapes), seed
        assert any(not is_rectangle(obj["shape"]) for obj in scene_objects), seed
        distractor = meta["distractor_color"]
        mask = grid == distractor
        assert np.any(mask), seed
        assert np.array_equal(out[mask], grid[mask]), seed
    _verify("103", check)
    metas = [generate("103", seed)["metadata"] for seed in range(200)]
    combos = {
        (meta["separator_axis"], meta["legend_side"], meta["measure_dimension"])
        for meta in metas
    }
    assert combos == {
        ("horizontal", "top", "height"),
        ("horizontal", "top", "width"),
        ("horizontal", "top", "cell_count"),
        ("horizontal", "bottom", "height"),
        ("horizontal", "bottom", "width"),
        ("horizontal", "bottom", "cell_count"),
        ("vertical", "left", "height"),
        ("vertical", "left", "width"),
        ("vertical", "left", "cell_count"),
        ("vertical", "right", "height"),
        ("vertical", "right", "width"),
        ("vertical", "right", "cell_count"),
    }
    prompts_by_combo = {}
    for meta in metas:
        key = (meta["separator_axis"], meta["legend_side"], meta["measure_dimension"])
        prompts_by_combo.setdefault(key, get_prompt("103", meta).lower())
    assert "legend above and a scene below" in prompts_by_combo[("horizontal", "top", "height")]
    assert "scene above and a legend below" in prompts_by_combo[("horizontal", "bottom", "height")]
    assert "legend on the left and a scene on the right" in prompts_by_combo[("vertical", "left", "height")]
    assert "scene on the left and a legend on the right" in prompts_by_combo[("vertical", "right", "height")]
    assert "recolor each one by its height using that legend" in prompts_by_combo[("horizontal", "top", "height")]
    assert "height means the number of rows" in prompts_by_combo[("vertical", "left", "height")]
    assert "recolor each one by its width using that legend" in prompts_by_combo[("horizontal", "top", "width")]
    assert "width means the number of columns" in prompts_by_combo[("vertical", "right", "width")]
    assert "cell count 2 -> color" in prompts_by_combo[("horizontal", "top", "cell_count")]
    assert "recolor each one by its number of cells using that legend" in prompts_by_combo[("vertical", "left", "cell_count")]
    assert "cell count means the total number of cells" in prompts_by_combo[("vertical", "right", "cell_count")]
    assert "look only at objects of color" in prompts_by_combo[("horizontal", "top", "height")]
    assert "rectangle" not in prompts_by_combo[("horizontal", "top", "height")]
    assert "zero-indexed row" not in prompts_by_combo[("horizontal", "top", "height")]


def test_task_104():
    def check(seed, ex):
        inp = ex["input"]
        before = np.array(inp["before"], dtype=np.uint8)
        after = np.array(inp["after"], dtype=np.uint8)
        query = np.array(inp["query"], dtype=np.uint8)
        mapping = {}
        for b, a in zip(before.flat, after.flat):
            if int(b) in mapping:
                assert mapping[int(b)] == int(a)
            mapping[int(b)] = int(a)
        for item in ex["metadata"]["color_mapping"]:
            assert item["from_color"] != item["to_color"], seed
            assert item["from_color"] in set(int(v) for v in query.flat), seed
        expected = np.vectorize(mapping.__getitem__, otypes=[np.uint8])(query)
        assert ex["output"] == expected.astype(int).tolist(), seed
    _verify("104", check)
    for seed in range(12):
        prompt = get_prompt("104", generate("104", seed)["metadata"]).lower()
        assert "one-to-one color replacement rule" in prompt
        assert "preserving the query grid dimensions" in prompt
        assert "return only the transformed query grid" in prompt


def test_task_107():
    banned_rank_terms = [
        "color" + "_order",
        "ear" + "lier_in_" + "color" + "_order",
        "lat" + "er_in_" + "color" + "_order",
        "ear" + "lier color",
        "lat" + "er color",
        "lower" + " color",
        "greater" + " color",
        "explicit " + "color " + "order",
    ]

    def check(seed, ex):
        grid = np.array(ex["input"], dtype=np.uint8)
        out = np.array(ex["output"], dtype=np.uint8)
        meta = ex["metadata"]
        expected = grid.copy()
        assert meta["combine_rule"] == "row_column_parity", seed
        assert set(meta) >= {
            "blank_color",
            "row_header_colors",
            "column_header_colors",
            "combine_rule",
            "even_parity_source",
            "odd_parity_source",
        }, seed
        assert {
            meta["even_parity_source"],
            meta["odd_parity_source"],
        } == {"row_header", "column_header"}, seed
        assert np.all(grid[1:, 1:] == meta["blank_color"]), seed
        assert out[0, :].tolist() == grid[0, :].tolist(), seed
        assert out[:, 0].tolist() == grid[:, 0].tolist(), seed
        source_counts = {"row_header": 0, "column_header": 0}
        for r in range(1, grid.shape[0]):
            for c in range(1, grid.shape[1]):
                source = (
                    meta["even_parity_source"]
                    if (r + c) % 2 == 0
                    else meta["odd_parity_source"]
                )
                source_counts[source] += 1
                expected[r, c] = grid[r, 0] if source == "row_header" else grid[0, c]
        assert all(source_counts.values()), seed
        assert out.tolist() == expected.tolist(), seed

        serialized_meta = json.dumps(meta).lower()
        assert not any(term in serialized_meta for term in banned_rank_terms), seed

    _verify("107", check)
    even_sources = {
        generate("107", seed)["metadata"]["even_parity_source"] for seed in range(100)
    }
    assert even_sources == {"row_header", "column_header"}
    for seed in range(12):
        meta = generate("107", seed)["metadata"]
        prompt = get_prompt("107", meta).lower()
        assert "row 0 contains column headers" in prompt
        assert "column 0 contains row headers" in prompt
        assert "parity of r + c" in prompt
        assert "when r + c is even" in prompt
        assert "when r + c is odd" in prompt
        assert "row-header color at row r, column 0" in prompt
        assert "column-header color at row 0, column c" in prompt
        assert not any(term in prompt for term in banned_rank_terms)
        assert not any(term in json.dumps(meta).lower() for term in banned_rank_terms)
        assert "return the completed grid" in prompt


def test_task_120():
    def check(seed, ex):
        left = np.array(ex["input"]["left"], dtype=np.uint8)
        right = np.array(ex["input"]["right"], dtype=np.uint8)
        out = np.array(ex["output"], dtype=np.uint8)
        meta = ex["metadata"]
        expected = np.full(left.shape, meta["difference_color"], dtype=np.uint8)
        expected[left == right] = meta["match_color"]
        assert out.tolist() == expected.tolist(), seed
    _verify("120", check)


def test_task_127():
    def check(seed, ex):
        grid = np.array(ex["input"], dtype=np.uint8)
        out = np.array(ex["output"], dtype=np.uint8)
        meta = ex["metadata"]
        base = {tuple(p) for p in meta["base_shape_offsets"]}
        decoration = {tuple(p) for p in meta["decoration_offsets"]}
        expected = grid.copy()
        roles = {obj["role"] for obj in meta["objects"]}
        assert "decorated_example" in roles, seed
        assert "plain_target" in roles, seed
        for obj in meta["objects"]:
            _shape, deco = _oriented_shape_and_decoration(
                obj["rotation_degrees_clockwise"], base, decoration
            )
            ar, ac = obj["anchor"]
            if obj["role"] == "plain_target":
                for dr, dc in deco:
                    expected[ar + dr, ac + dc] = meta["decoration_color"]
        assert out.tolist() == expected.tolist(), seed
    _verify("127", check)
    sizes = {tuple(np.array(generate("127", s)["input"]).shape) for s in range(200)}
    n_objs = {len(generate("127", s)["metadata"]["objects"]) for s in range(200)}
    shapes = {
        tuple(map(tuple, generate("127", s)["metadata"]["base_shape_offsets"]))
        for s in range(200)
    }
    assert len(sizes) > 1
    assert len(n_objs) > 1
    assert len(shapes) > 1


def test_task_130():
    def connected(mask):
        cells = {tuple(p) for p in np.argwhere(mask)}
        if not cells:
            return False
        todo = [next(iter(cells))]
        seen = {todo[0]}
        while todo:
            r, c = todo.pop()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nb = (r + dr, c + dc)
                if nb in cells and nb not in seen:
                    seen.add(nb)
                    todo.append(nb)
        return seen == cells

    def check(seed, ex):
        grid = np.array(ex["input"], dtype=np.uint8)
        out = np.array(ex["output"], dtype=np.uint8)
        meta = ex["metadata"]
        w = meta["panel_width"]
        source = grid[:, :w] == meta["source_color"]
        target = grid[:, w + 1:] == meta["target_color"]
        assert connected(source), seed
        assert connected(target), seed
        expected = np.full((grid.shape[0], w), meta["background_color"], dtype=np.uint8)
        expected[target & ~source] = meta["add_color"]
        expected[source & ~target] = meta["remove_color"]
        assert out.tolist() == expected.tolist(), seed
    _verify("130", check)


def test_task_281():
    def expected_report(labels, grid):
        return "\n".join(
            f"{label}: {'yes' if _is_symmetric(grid, label) else 'no'}"
            for label in labels
        )

    def check(seed, ex):
        grid = np.array(ex["input"], dtype=np.uint8)
        meta = ex["metadata"]
        top, left, bottom, right = meta["frame_bbox"]
        labels = meta["allowed_symmetries"]
        assert 5 <= grid.shape[0] <= 30, seed
        assert 5 <= grid.shape[1] <= 30, seed
        assert grid[top, left:right + 1].tolist() == [meta["frame_color"]] * (right - left + 1), seed
        assert grid[bottom, left:right + 1].tolist() == [meta["frame_color"]] * (right - left + 1), seed
        assert grid[top:bottom + 1, left].tolist() == [meta["frame_color"]] * (bottom - top + 1), seed
        assert grid[top:bottom + 1, right].tolist() == [meta["frame_color"]] * (bottom - top + 1), seed
        inner = grid[top + 1:bottom, left + 1:right]
        assert inner.shape == (meta["interior_height"], meta["interior_width"]), seed
        assert ex["output"] == expected_report(labels, inner), seed

    _verify("281", check)
    seen_yes = {label: False for label in generate("281", 0)["metadata"]["allowed_symmetries"]}
    seen_no = {label: False for label in generate("281", 0)["metadata"]["allowed_symmetries"]}
    present_counts = set()
    heights = []
    widths = []
    for seed in range(300):
        ex = generate("281", seed)
        grid = np.array(ex["input"], dtype=np.uint8)
        meta = ex["metadata"]
        top, left, bottom, right = meta["frame_bbox"]
        inner = grid[top + 1:bottom, left + 1:right]
        labels = meta["allowed_symmetries"]
        heights.append(grid.shape[0])
        widths.append(grid.shape[1])
        present_counts.add(sum(_is_symmetric(inner, label) for label in labels))
        lines = ex["output"].splitlines()
        assert len(lines) == len(labels), seed
        for label, line in zip(labels, lines):
            assert line in {f"{label}: yes", f"{label}: no"}, seed
            if line.endswith(": yes"):
                seen_yes[label] = True
            else:
                seen_no[label] = True
    assert all(seen_yes.values())
    assert all(seen_no.values())
    assert 0 in present_counts
    assert any(count > 1 for count in present_counts)
    assert min(heights) <= 7
    assert max(heights) >= 24
    assert min(widths) <= 7
    assert max(widths) >= 24
    prompt = get_prompt("281", generate("281", 0)["metadata"]).lower()
    assert "frame of color" in prompt
    assert "ignore the frame itself and everything outside it" in prompt
    assert "strictly inside the frame" in prompt
    assert "return exactly one line for each label" in prompt
    assert "anti_diagonal_reflection means the framed interior is square" in prompt


if __name__ == "__main__":
    funcs = [v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    print(f"Running {len(funcs)} symmetry batch F tests...")
    failed = 0
    for func in funcs:
        try:
            func()
            print(f"  PASS  {func.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {func.__name__}: {exc}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL  {func.__name__}: {type(exc).__name__}: {exc}")
    print(f"\nResult: {len(funcs) - failed}/{len(funcs)} passed")
    sys.exit(0 if failed == 0 else 1)
