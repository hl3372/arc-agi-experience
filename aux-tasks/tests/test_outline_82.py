"""Prompt readers and geometry, query independence and tamper checks for 82."""
from __future__ import annotations

from collections import Counter, deque
from copy import deepcopy
import re

import numpy as np
import pytest

from generators import outline_82 as module
from prompts import get_prompt
from registry import generate


def _read(grid, prompt, *, diagonal=True):
    """Follow the actual prompt via graph distances, without task metadata."""
    match = re.search(r"Draw a (\d+)-cell-thick external outline in color (\S+) around", prompt)
    assert match and "diagonal corner cells" in prompt
    assert "original object cell" in prompt and "holes and gaps" in prompt
    assert "Clip the outline at the grid edge" in prompt
    assert "overlapping bands use the same color" in prompt
    assert prompt.endswith("Leave every other cell unchanged.")
    thickness = int(match[1])
    color = match[2] if isinstance(grid[0][0], str) else int(match[2])
    h, w = len(grid), len(grid[0])
    border = [grid[r][c] for r in range(h) for c in range(w)
              if r in (0, h - 1) or c in (0, w - 1)]
    bg, frequency = Counter(border).most_common(1)[0]
    assert frequency > len(border) / 2
    distance = {(r, c): 0 for r in range(h) for c in range(w) if grid[r][c] != bg}
    queue = deque(distance)
    result = deepcopy(grid)
    while queue:
        r, c = queue.popleft()
        if distance[r, c] == thickness:
            continue
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (dr == dc == 0) or (not diagonal and abs(dr) + abs(dc) != 1):
                    continue
                point = r + dr, c + dc
                if 0 <= point[0] < h and 0 <= point[1] < w and point not in distance:
                    distance[point] = distance[r, c] + 1
                    queue.append(point)
                    if grid[point[0]][point[1]] == bg:
                        result[point[0]][point[1]] = color
    return result


@pytest.fixture(scope="module")
def examples():
    return [generate("82", seed) for seed in range(1000)]


def test_independent_actual_prompt_reader(examples):
    other_reading = 0
    for ex in examples[:400]:
        prompt = get_prompt("82", ex["metadata"])
        assert _read(ex["input"], prompt) == ex["output"]
        other_reading += _read(ex["input"], prompt, diagonal=False) != ex["output"]
    assert other_reading > 350


def test_sizes_counts_and_scaled_shapes(examples):
    sizes = np.asarray([np.shape(ex["input"]) for ex in examples])
    assert set(sizes[:, 0]) == set(sizes[:, 1]) == set(range(5, 31))
    assert abs(np.corrcoef(sizes.T)[0, 1]) < .1
    assert np.mean(sizes[:, 0] == sizes[:, 1]) < .07
    assert np.mean(np.prod(sizes, axis=1) >= 700) > .03
    counts = Counter(ex["metadata"]["n_objects"] for ex in examples)
    assert set(counts) == set(range(1, 7))
    assert .30 < counts[1] / len(examples) < .60
    assert sum(counts[n] for n in (1, 2, 3)) > 820
    assert sum(counts[n] for n in (4, 5, 6)) > 60
    families = Counter(o["family"] for ex in examples for o in ex["metadata"]["objects"])
    assert set(families) == {"rectangle", "bar", "L", "T", "U", "H", "plus", "frame", "staircase", "diamond", "chamfer"}
    assert min(families.values()) > 25
    assert max(o["area"] for ex in examples for o in ex["metadata"]["objects"]) > 250
    small = [e for e in examples if np.size(e["input"]) < 150]
    large = [e for e in examples if np.size(e["input"]) >= 700]
    occupancy = lambda group: np.mean([1 - e["metadata"]["background_share"] for e in group])
    assert occupancy(large) > .18
    assert occupancy(large) > .85 * occupancy(small)


def test_clear_fields_and_modest_colors(examples):
    palettes = Counter(ex["metadata"]["n_colors"] for ex in examples)
    assert min(palettes) == 2 and max(palettes) >= 6
    assert .45 < palettes[2] / len(examples) < .7
    assert palettes[2] + palettes[3] > 800
    for key in ("background_color", "outline_color"):
        colors = Counter(ex["metadata"][key] for ex in examples)
        assert set(colors) == set(range(10)) and min(colors.values()) > 60
    for ex in examples:
        m = ex["metadata"]
        assert m["background_style"] == "solid"
        assert m["background_share"] > .5 and m["background_border_share"] > .5
        assert m["outline_color"] not in np.asarray(ex["input"])
        assert len(np.unique(ex["output"])) <= m["n_colors"] + 1


def test_widths_clipping_meeting_and_fixed_plans(examples):
    widths = Counter(ex["metadata"]["outline_thickness"] for ex in examples)
    assert .48 < widths[1] / len(examples) < .64
    assert .28 < widths[2] / len(examples) < .42
    assert .06 < widths[3] / len(examples) < .14
    clipped = meeting = overlap = freely_moved = 0
    for ex in examples:
        m = ex["metadata"]
        assert m["eligible_outline_thicknesses"] == [1, 2, 3]
        assert m["query_drawn_after_input"]
        assert m["planned_n_objects"] == m["n_objects"] <= m["witnessed_capacity"]
        assert m["attempts"] == 1 and m["free_placement_passes"] == 3
        assert m["planned_families"] == sorted(o["family"] for o in m["objects"])
        assert m["planned_object_colors"] == sorted(o["color"] for o in m["objects"])
        assert m["planned_object_areas"] == sorted(o["area"] for o in m["objects"])
        assert m["output_changed_cells"] > 0 and m["original_objects_preserved"]
        freely_moved += m["free_relocations"] > 0
        clipped += m["n_clipped_outlines"] > 0
        meeting += m["meeting_or_overlapping_band_pairs"] > 0
        overlap += m["overlapping_band_pairs"] > 0
    assert freely_moved > 800  # Tiny canvases can have only their witnessed location.
    assert 60 < clipped < 300
    assert 120 < meeting < 370
    assert 40 < overlap < 230


def test_each_width_on_identical_finished_inputs():
    class RequestedWidth:
        def __init__(self, seed, width):
            self.rng = np.random.default_rng(seed)
            self.width = width

        def __getattr__(self, name):
            return getattr(self.rng, name)

        def choice(self, options, *args, **kwargs):
            if options is module.WIDTHS:
                return self.width
            return self.rng.choice(options, *args, **kwargs)

    changed = 0
    for seed in range(50):
        scenes = [module.generate(RequestedWidth(seed, t)) for t in module.WIDTHS]
        assert scenes[0]["input"] == scenes[1]["input"] == scenes[2]["input"]
        for key in ("objects", "planned_families", "placement_gap", "placement_margins", "capacity_boxes"):
            assert scenes[0]["metadata"][key] == scenes[1]["metadata"][key] == scenes[2]["metadata"][key]
        changed += scenes[0]["output"] != scenes[1]["output"] != scenes[2]["output"]
    assert changed > 45


def test_symbols_and_repeatability(examples):
    for seed, ex in enumerate(examples[:100]):
        palette = np.random.default_rng(82000 + seed).permutation(list("QWERTYUIOP")).tolist()
        inp = [[palette[c] for c in row] for row in ex["input"]]
        out = [[palette[c] for c in row] for row in ex["output"]]
        prompt = get_prompt("82", ex["metadata"], color_palette=palette)
        assert _read(inp, prompt) == out
        assert re.search(r"color \d\b", prompt) is None
        assert ex == generate("82", seed)


@pytest.mark.parametrize("fault", ("output", "original", "background", "width", "count", "palette", "family", "area", "bounds", "clipping", "overlap", "queries", "attempts", "capacity", "witness", "gap"))
def test_validator_rejects_tampering(fault):
    ex = deepcopy(generate("82", 0))
    m = ex["metadata"]
    if fault == "output":
        r, c = next((r, c) for r, row in enumerate(ex["output"]) for c, value in enumerate(row)
                    if value == m["outline_color"])
        ex["output"][r][c] = m["background_color"]
    elif fault == "original":
        r, c = m["objects"][0]["first_cell"]
        ex["output"][r][c] = m["outline_color"]
    elif fault == "background":
        m["background_color"] = m["outline_color"]
    elif fault == "width":
        m["outline_thickness"] = 0
    elif fault == "count":
        m["planned_n_objects"] += 1
    elif fault == "palette":
        m["planned_object_colors"][0] = m["outline_color"]
    elif fault == "family":
        m["planned_families"][0] = "random_walk"
    elif fault == "area":
        m["objects"][0]["area"] += 1
    elif fault == "bounds":
        m["objects"][0]["bounds"][0] += 1
    elif fault == "clipping":
        m["n_clipped_outlines"] += 1
    elif fault == "overlap":
        m["overlap_cells"] += 1
    elif fault == "queries":
        m["eligible_outline_thicknesses"] = [m["outline_thickness"]]
    elif fault == "attempts":
        m["attempts"] += 1
    elif fault == "capacity":
        m["witnessed_capacity"] -= 1
    elif fault == "witness":
        m["capacity_witness_positions"][0] = [-1, -1]
    else:
        m["placement_gap"] = 0
    with pytest.raises(ValueError):
        module.validate_outline(ex)


def test_direct_chebyshev_holes_clipping_and_preservation():
    inp = np.zeros((11, 13), dtype=int)
    inp[0:3, 0:3] = 4
    inp[4:9, 6:11] = 7
    inp[5:8, 7:10] = 0
    inp[6:8, 3:5] = 2
    points = np.argwhere(inp != 0)
    for width in module.WIDTHS:
        out = module.solve_outline(inp, width, 3)
        expected = inp.copy()
        for r, c in np.argwhere(inp == 0):
            if min(max(abs(r - y), abs(c - x)) for y, x in points) <= width:
                expected[r, c] = 3
        assert np.array_equal(out, expected)
        assert np.array_equal(out[inp != 0], inp[inp != 0])
    assert module.solve_outline(inp, 2, 3)[6, 8] == 3
