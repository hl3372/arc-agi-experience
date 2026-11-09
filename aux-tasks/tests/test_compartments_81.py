"""Task 81: literal prompt reading, drawn rooms, and independent rule draws.

The reader below only sees the displayed colours and the rendered instruction.
Everything a scene claims -- room shapes, nested rings, doorways, wall
thicknesses -- is re-measured from the visible grid here, so the generator's
metadata is checked rather than trusted.
"""
from __future__ import annotations

from collections import Counter, deque
from copy import deepcopy
import re

import numpy as np
import pytest

from generators import compartments_81 as module
from prompts import get_prompt
from registry import generate

N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
SWEEP = 1000


def read_compartments(grid, prompt):
    """Own reader using only the displayed colors and the actual instruction."""
    match = re.search(r"each ([48])-connected group", prompt)
    assert match and "uniform outer frame" in prompt and "row-major order" in prompt
    rule = int(match.group(1))
    # the handbook's canonical adjacency wording, verbatim
    assert ("connected horizontally or vertically (cells that touch only at a corner are not connected)"
            if rule == 4 else "connected horizontally, vertically, or diagonally") in prompt
    assert "size by the number of cells" in prompt
    match = re.search(r"largest compartment with color (\S+) and the smallest with color (\S+)\.", prompt)
    paint_large, paint_small = match.groups()
    if not isinstance(grid[0][0], str):
        paint_large, paint_small = int(paint_large), int(paint_small)
    height, width = len(grid), len(grid[0])
    wall = grid[0][0]
    assert all(grid[r][c] == wall for r in range(height) for c in range(width)
               if r in (0, height - 1) or c in (0, width - 1))
    seen = set()
    rooms = []
    for row in range(height):
        for col in range(width):
            if grid[row][col] == wall or (row, col) in seen:
                continue
            todo, room = [(row, col)], []
            seen.add((row, col))
            while todo:
                r, c = todo.pop()
                room.append((r, c))
                for nr in range(max(0, r - 1), min(height, r + 2)):
                    for nc in range(max(0, c - 1), min(width, c + 2)):
                        if (nr, nc) == (r, c) or (rule == 4 and nr != r and nc != c):
                            continue
                        if grid[nr][nc] != wall and (nr, nc) not in seen:
                            seen.add((nr, nc))
                            todo.append((nr, nc))
            rooms.append(sorted(room))
    ranked = sorted(rooms, key=lambda room: (len(room), room[0]))
    assert len(ranked) >= 2 and len(ranked[0]) < len(ranked[-1])
    smallest = ranked[0]
    largest = next(room for room in ranked if len(room) == len(ranked[-1]))
    result = [list(row) for row in grid]
    for cells, color in ((largest, paint_large), (smallest, paint_small)):
        for r, c in cells:
            result[r][c] = color
    return result, rooms


def components(cells, steps=N4):
    """Connected pieces of a cell set, written without the shared libraries."""
    rest = set(cells)
    out = []
    while rest:
        first = min(rest)
        rest.remove(first)
        queue, piece = deque([first]), {first}
        while queue:
            r, c = queue.popleft()
            for dr, dc in steps:
                cell = (r + dr, c + dc)
                if cell in rest:
                    rest.remove(cell)
                    queue.append(cell)
                    piece.add(cell)
        out.append(piece)
    return out


def drawn_rooms(example):
    """The rooms behind a picture: compartments cut back at their doorways."""
    grid, meta = example["input"], example["metadata"]
    wall = grid[0][0]
    doors = {tuple(cell) for run in meta["doorway_cells"] for cell in run}
    floor = {(r, c) for r in range(len(grid)) for c in range(len(grid[0]))
             if grid[r][c] != wall} - doors
    return components(floor)


def outline_corners(cells):
    """Corners of the region outline: 4 for a rectangle, 6 for an L, 12 for a plus."""
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    total = 0
    for r in range(min(rows), max(rows) + 2):
        for c in range(min(cols), max(cols) + 2):
            quad = ((r - 1, c - 1) in cells, (r - 1, c) in cells,
                    (r, c - 1) in cells, (r, c) in cells)
            filled = sum(quad)
            if filled in (1, 3):
                total += 1
            elif filled == 2 and quad[0] == quad[3]:
                total += 2
    return total


def example_from_picture(picture, doorways=(), kinds=(), connectivity=4,
                         family="rectangular_partition", paints=(3, 4)):
    """A complete example drawn by hand: ``#`` is wall, ``.`` is floor.

    Every metadata field is re-derived from the picture the way the prompt
    reads it, so a test can break exactly one thing and watch the validator
    catch that one thing.
    """
    wall, background = 9, 0
    grid = [[wall if ch == "#" else background for ch in row] for row in picture]
    height, width = len(grid), len(grid[0])
    floor = {(r, c) for r in range(height) for c in range(width)
             if grid[r][c] == background}
    eight = N4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))
    outputs, meta = {}, {}
    for rule, steps in ((4, N4), (8, eight)):
        found = sorted(components(floor, steps), key=min)
        areas = sorted(len(region) for region in found)
        large = next(region for region in found if len(region) == areas[-1])
        small = next(region for region in found if len(region) == areas[0])
        painted = [list(row) for row in grid]
        for cells, color in ((large, paints[0]), (small, paints[1])):
            for r, c in cells:
                painted[r][c] = color
        outputs[rule] = painted
        meta[f"room_count_{rule}"] = len(found)
        meta[f"areas_{rule}"] = areas
        meta[f"ties_large_{rule}"] = areas.count(areas[-1]) > 1
        meta[f"ties_small_{rule}"] = areas.count(areas[0]) > 1
        if rule == connectivity:
            meta.update({"room_count": len(found), "largest_area": areas[-1],
                         "smallest_area": areas[0],
                         "ties_large": meta[f"ties_large_{rule}"],
                         "ties_small": meta[f"ties_small_{rule}"],
                         "largest_is_first": large is found[0],
                         "smallest_is_first": small is found[0]})
    depth = 0
    while 2 * depth < min(height, width) and all(
            grid[r][c] == wall
            for r in range(depth, height - depth)
            for c in range(depth, width - depth)
            if r in (depth, height - 1 - depth) or c in (depth, width - 1 - depth)):
        depth += 1
    meta.update({"grid_h": height, "grid_w": width, "n_colors": 2,
                 "background_color": background, "wall_color": wall,
                 "color_large": paints[0], "color_small": paints[1],
                 "layout_family": family, "planned_family": family,
                 "frame_thickness": depth, "wall_thicknesses": [1],
                 "varying_wall_thickness": False, "n_rooms": len(kinds),
                 "room_kinds": list(kinds), "n_nested": 0,
                 "n_doorways": len(doorways),
                 "doorway_cells": [[list(cell) for cell in run] for run in doorways],
                 "corner_contacts": 0, "attempts": 1, "connectivity": connectivity,
                 "connectivity_matters": outputs[4] != outputs[8],
                 "planned_room_count": meta["room_count_4"],
                 "compartment_cells": len(floor)})
    return {"input": grid, "output": outputs[connectivity], "metadata": meta}


THREE_ROOMS = ("################",
               "#....#.....#...#",
               "#....#.....#...#",
               "#..........#...#",
               "#....#.....#...#",
               "#....#.....#...#",
               "################")


@pytest.fixture(scope="module")
def examples():
    return [generate("81", seed) for seed in range(SWEEP)]


def test_independent_visible_prompt_reader(examples):
    for example in examples[:400]:
        output, _ = read_compartments(example["input"], get_prompt("81", example["metadata"]))
        assert output == example["output"]


def test_arbitrary_symbols_preserve_both_rules(examples):
    for seed, example in enumerate(examples[:300]):
        palette = np.random.default_rng(seed + 419).permutation(list("QWERTYUIOP")).tolist()
        meta = example["metadata"]
        grid = [[palette[c] for c in row] for row in example["input"]]
        expected = [[palette[c] for c in row] for row in example["output"]]
        prompt = get_prompt("81", meta, color_palette=palette)
        assert read_compartments(grid, prompt)[0] == expected
        assert f"color {palette[meta['color_large']]}" in prompt
        assert f"color {palette[meta['color_small']]}" in prompt
        assert "background color" not in prompt


def test_size_and_color_coverage(examples):
    sides = np.array([np.shape(ex["input"]) for ex in examples])
    assert set(sides[:, 0]) == set(sides[:, 1]) == set(range(5, 31))
    # 667 of the 676 canvases are drawable: the two-cell floor rules out the
    # nine tiny near-squares from 5x5 to 7x7, where a wall plus two chambers
    # of two cells each does not fit or leaves two equal halves.
    assert len(set(map(tuple, sides))) > 470
    assert abs(np.corrcoef(sides.T)[0, 1]) < .10
    assert .02 < np.mean(sides[:, 0] == sides[:, 1]) < .07
    assert .025 < np.mean(np.prod(sides, axis=1) >= 700) < .07
    for ex in examples:
        assert len(np.unique(ex["input"])) == 2
        assert len(np.unique(ex["output"])) in (3, 4)
    for key in ("wall_color", "background_color", "color_large", "color_small"):
        counts = Counter(ex["metadata"][key] for ex in examples)
        assert set(counts) == set(range(10))
        assert all(.06 < n / len(examples) < .14 for n in counts.values())


def test_layout_families_are_all_drawn(examples):
    families = Counter(ex["metadata"]["layout_family"] for ex in examples)
    assert set(families) == set(module.FAMILIES)
    share = {name: count / len(examples) for name, count in families.items()}
    assert .36 < share["corner_suite"] < .52
    assert .05 < share["double_corner"] < .16
    assert .12 < share["rectangular_partition"] < .28
    assert .06 < share["cutout_rooms"] < .19
    assert .04 < share["stepped_partition"] < .13
    assert .04 < share["paired_rooms"] < .12
    for ex in examples:
        meta = ex["metadata"]
        assert meta["planned_family"] == meta["layout_family"]
        assert meta["planned_room_count"] == meta["room_count_4"]
        assert len(meta["room_kinds"]) == meta["n_rooms"] >= 2
        assert meta["attempts"] <= module.MAX_ATTEMPTS
        assert meta["compartment_cells"] == int(
            np.count_nonzero(np.array(ex["input"]) != meta["wall_color"]))


def test_room_shapes_reach_well_beyond_rectangles(examples):
    kinds = Counter()
    for ex in examples:
        kinds.update(ex["metadata"]["room_kinds"])
    total = sum(kinds.values())
    assert set(kinds) <= set(module.ROOM_KINDS)
    share = {kind: count / total for kind, count in kinds.items()}
    assert share["rectangle"] <= .45
    assert sum(1 for value in share.values() if value >= .05) >= 5
    for kind in ("L", "U", "T", "alcove", "stepped", "irregular", "plus", "ring"):
        assert share.get(kind, 0) > .005, kind
    # The kind labels describe the picture: a named room is not a rectangle.
    for ex in examples[:200]:
        meta = ex["metadata"]
        if meta["n_doorways"] or meta["n_nested"]:
            continue
        rooms = drawn_rooms(ex)
        assert len(rooms) == meta["n_rooms"]
        boxy = sum(len(room) == np.prod(module._sides(module._bounds(room)))
                   for room in rooms)
        assert boxy == meta["room_kinds"].count("rectangle")


def test_nested_rooms_are_separate_compartments(examples):
    nested = [ex for ex in examples if ex["metadata"]["n_nested"]]
    assert .10 < len(nested) / len(examples) < .24
    assert any(ex["metadata"]["n_nested"] > 1 for ex in nested)
    for ex in nested[:120]:
        meta = ex["metadata"]
        grid = np.array(ex["input"])
        walls = {tuple(map(int, rc)) for rc in np.argwhere(grid == meta["wall_color"])}
        pieces = components(walls)
        loose = [piece for piece in pieces if (0, 0) not in piece]
        assert len(loose) == meta["n_nested"]
        for piece in loose:
            # a closed rectangular ring with a compartment shut inside it
            r0, c0, r1, c1 = module._bounds(piece)
            thickness = module._ring_thickness(piece)
            assert thickness >= 1
            inside = module._cells((r0 + thickness, c0 + thickness,
                                    r1 - thickness, c1 - thickness))
            assert all(grid[r, c] != meta["wall_color"] for r, c in inside)
            _, regions = module._visible_regions(grid, 8)
            assert inside in regions


def test_doorways_merge_two_rooms_into_one_compartment(examples):
    doored = [ex for ex in examples if ex["metadata"]["n_doorways"]]
    assert .14 < len(doored) / len(examples) < .29
    assert any(ex["metadata"]["n_doorways"] > 1 for ex in doored)
    for ex in doored[:120]:
        meta = ex["metadata"]
        grid = np.array(ex["input"])
        assert len(drawn_rooms(ex)) == meta["n_rooms"]
        for run in meta["doorway_cells"]:
            assert 1 <= len(run) <= 3
            for r, c in run:
                assert grid[r, c] != meta["wall_color"]
            closed = grid.copy()
            for r, c in run:
                closed[r, c] = meta["wall_color"]
            # closing a doorway splits exactly one compartment under either rule
            for rule in (4, 8):
                before = len(module._visible_regions(grid, rule)[1])
                after = len(module._visible_regions(closed, rule)[1])
                assert after == before + 1, (rule, run)
        assert meta["room_count_4"] < meta["n_rooms"]


def test_the_rule_changes_the_answer_on_most_of_the_scenes(examples):
    changed, rules = 0, Counter()
    for ex in examples:
        outputs = []
        for rule in (4, 8):
            meta = dict(ex["metadata"], connectivity=rule)
            output, rooms = read_compartments(ex["input"], get_prompt("81", meta))
            outputs.append(output)
            assert meta[f"room_count_{rule}"] == len(rooms)
            assert meta[f"areas_{rule}"] == sorted(map(len, rooms))
        assert ex["metadata"]["connectivity_matters"] == (outputs[0] != outputs[1])
        changed += outputs[0] != outputs[1]
        rules[ex["metadata"]["connectivity"]] += 1
    # 50.4% over 3000 seeds: the two-cell chamber floor takes corner joins off
    # the narrowest canvases, where four chambers no longer fit.
    assert .46 <= changed / len(examples) < .62
    assert .45 < rules[4] / len(examples) < .55
    contacts = Counter(ex["metadata"]["corner_contacts"] for ex in examples)
    assert contacts[1] + contacts[2] >= changed


def test_connectivity_is_not_predictable_from_the_input(examples):
    """The coin is drawn after the picture, so no visible feature leans on it."""
    buckets = Counter()
    fours = Counter()
    for ex in examples:
        meta = ex["metadata"]
        area = meta["grid_h"] * meta["grid_w"]
        keys = [("area", area // 200),
                ("rooms", min(6, meta["room_count_4"])),
                ("splits", meta["room_count_4"] != meta["room_count_8"]),
                ("floor", meta["compartment_cells"] // 120)]
        for key in keys:
            buckets[key] += 1
            fours[key] += meta["connectivity"] == 4
    for key, total in buckets.items():
        if total >= 150:
            assert abs(fours[key] / total - .5) < .10, (key, total, fours[key] / total)


def test_completed_input_does_not_depend_on_connectivity():
    class RuleRng:
        def __init__(self, seed, rule):
            self.base, self.rule = np.random.default_rng(seed), rule

        def __getattr__(self, name):
            return getattr(self.base, name)

        def choice(self, a, *args, **kwargs):
            if isinstance(a, tuple) and a == (4, 8):
                return self.rule
            return self.base.choice(a, *args, **kwargs)

    matters = 0
    for seed in range(120):
        four = module.generate_color_compartments(RuleRng(seed, 4))
        eight = module.generate_color_compartments(RuleRng(seed, 8))
        assert four["input"] == eight["input"]
        assert four["metadata"]["room_kinds"] == eight["metadata"]["room_kinds"]
        matters += four["output"] != eight["output"]
    assert matters > 50


def test_wall_thickness_varies_within_and_between_grids(examples):
    frames = Counter(ex["metadata"]["frame_thickness"] for ex in examples)
    assert set(frames) == {1, 2, 3}
    assert frames[2] / len(examples) > .10 and frames[3] / len(examples) > .02
    varying = [ex for ex in examples if ex["metadata"]["varying_wall_thickness"]]
    assert .15 < len(varying) / len(examples) < .34
    seen = set()
    for ex in examples:
        seen.update(ex["metadata"]["wall_thicknesses"])
    assert seen == {1, 2, 3}
    # The outer frame is equally deep on all four sides, so "uniform outer
    # frame" is visibly true: no room may leave an extra solid wall row or
    # column lying against one edge.
    for ex in examples:
        meta, grid = ex["metadata"], ex["input"]
        height, width = len(grid), len(grid[0])
        wall = meta["wall_color"]
        solid_rows = [all(value == wall for value in row) for row in grid]
        solid_cols = [all(grid[r][c] == wall for r in range(height)) for c in range(width)]

        def lead(flags):
            depth = 0
            while depth < len(flags) and flags[depth]:
                depth += 1
            return depth

        depths = (lead(solid_rows), lead(solid_rows[::-1]),
                  lead(solid_cols), lead(solid_cols[::-1]))
        assert set(depths) == {meta["frame_thickness"]}, (depths, meta["frame_thickness"])
        assert 1 <= meta["frame_thickness"] <= 3
    for ex in examples[:200]:
        meta, grid = ex["metadata"], ex["input"]
        height, width = len(grid), len(grid[0])
        ring = lambda k: ([grid[k][c] for c in range(k, width - k)]
                          + [grid[height - 1 - k][c] for c in range(k, width - k)]
                          + [grid[r][k] for r in range(k, height - k)]
                          + [grid[r][width - 1 - k] for r in range(k, height - k)])
        depth = 0
        while 2 * depth < min(height, width) and set(ring(depth)) == {meta["wall_color"]}:
            depth += 1
        assert depth == meta["frame_thickness"]


def test_the_extrema_are_not_always_the_first_room(examples):
    large_first = sum(ex["metadata"]["largest_is_first"] for ex in examples)
    small_first = sum(ex["metadata"]["smallest_is_first"] for ex in examples)
    assert .20 < large_first / len(examples) < .52
    assert .10 < small_first / len(examples) < .38
    spread, ties = [], 0
    for ex in examples:
        meta = ex["metadata"]
        areas = meta[f"areas_{meta['connectivity']}"]
        assert areas[0] >= 2 and areas[-1] > areas[0]
        spread.append(areas[-1] / areas[0])
        ties += meta["ties_large"] or meta["ties_small"]
    assert np.median(spread) > 2.5
    assert .12 < ties / len(examples) < .30
    # a tie is decided by the row-major rule, not by which room was drawn first
    tied = [ex for ex in examples if ex["metadata"]["ties_small"]][:80]
    assert tied
    for ex in tied:
        meta = ex["metadata"]
        _, regions = module._visible_regions(np.array(ex["input"]), meta["connectivity"])
        floor = min((region for region in regions
                     if len(region) == meta["smallest_area"]), key=min)
        painted = {(r, c) for r, row in enumerate(ex["output"])
                   for c, value in enumerate(row) if value == meta["color_small"]}
        assert painted == floor


def test_room_counts_and_occupancy_scale_with_the_canvas(examples):
    counts = Counter(ex["metadata"]["room_count"] for ex in examples)
    assert counts[2] / len(examples) > .08
    assert max(counts) >= 12
    big = [ex for ex in examples if ex["metadata"]["grid_h"] * ex["metadata"]["grid_w"] >= 500]
    small = [ex for ex in examples if ex["metadata"]["grid_h"] * ex["metadata"]["grid_w"] <= 120]
    assert np.mean([ex["metadata"]["n_rooms"] for ex in big]) > 5
    assert max(ex["metadata"]["n_rooms"] for ex in big) >= 12
    assert np.mean([ex["metadata"]["n_rooms"] for ex in big]) > np.mean(
        [ex["metadata"]["n_rooms"] for ex in small])

    def occupancy(pool):
        return np.mean([ex["metadata"]["compartment_cells"]
                        / (ex["metadata"]["grid_h"] * ex["metadata"]["grid_w"]) for ex in pool])

    assert occupancy(big) > .42
    assert occupancy(big) > occupancy(small) - .12


def test_walls_read_as_walls(examples):
    """One frame structure plus closed rings; a wall only stops at a doorway."""
    for ex in examples:
        meta = ex["metadata"]
        grid = ex["input"]
        wall = grid[0][0]
        cells = {(r, c) for r in range(len(grid)) for c in range(len(grid[0]))
                 if grid[r][c] == wall}
        doors = {tuple(cell) for run in meta["doorway_cells"] for cell in run}
        pieces = components(cells)
        assert sum((0, 0) in piece for piece in pieces) == 1
        assert len(pieces) == meta["n_nested"] + 1
        for piece in pieces:
            if (0, 0) not in piece:
                assert module._ring_thickness(piece) in (1, 2, 3)
        for r, c in cells:
            neighbours = sum((r + dr, c + dc) in cells for dr, dc in N4)
            if neighbours < 2:
                assert any((r + dr, c + dc) in doors for dr, dc in N4), (r, c)


def test_rooms_look_drawn(examples):
    """Rooms are unions of a few rectangles: blocky outlines, no carved noise."""
    corners = Counter()
    for ex in examples:
        for room in drawn_rooms(ex):
            assert len(room) >= 2
            corners[outline_corners(room)] += 1
    total = sum(corners.values())
    assert max(corners) <= 16
    assert sum(count for turns, count in corners.items() if turns <= 8) / total > .80
    assert sum(count for turns, count in corners.items() if turns <= 12) / total > .96


def test_no_room_is_one_cell_wide(examples):
    """Chambers are two cells wide in both directions, on every canvas.

    Narrow canvases used to fall back to a one-cell floor: half of the grids
    with a short side of seven or less were mostly one-wide slots (a barcode),
    and a third of them tied for smallest.  Only a nested room's wall ring is
    one cell thick now, and that is a wall, not a compartment.
    """
    def box(cells):
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        return min(max(rows) - min(rows) + 1, max(cols) - min(cols) + 1)

    narrow = [ex for ex in examples
              if min(ex["metadata"]["grid_h"], ex["metadata"]["grid_w"]) <= 7]
    assert .10 < len(narrow) / len(examples) < .25
    slotted = blocky = total = 0
    for ex in examples:
        grid = ex["input"]
        wall = grid[0][0]
        floor = {(r, c) for r in range(len(grid)) for c in range(len(grid[0]))
                 if grid[r][c] != wall}
        for room in drawn_rooms(ex):
            assert box(room) >= 2, (ex["metadata"]["grid_h"], ex["metadata"]["grid_w"])
            total += 1
            # a chamber is a body, not a line: it holds a 2x2 block
            blocky += any({(r + 1, c), (r, c + 1), (r + 1, c + 1)} <= room for r, c in room)
        slotted += sum(box(region) == 1 for region in components(floor)) >= 2
    assert slotted == 0
    assert blocky / total > .97
    # ties for smallest no longer pile up on the narrow canvases (32% before)
    tied = sum(ex["metadata"]["ties_small"] for ex in narrow) / len(narrow)
    assert tied < .26


def test_corner_contacts_and_row_major_ties_have_literal_meaning():
    grid = [[9, 9, 9, 9, 9], [9, 0, 0, 9, 9], [9, 9, 9, 0, 9],
            [9, 0, 9, 9, 9], [9, 9, 9, 9, 9]]
    four, rooms4 = read_compartments(grid, get_prompt("81", {"connectivity": 4, "color_large": 3, "color_small": 4}))
    eight, rooms8 = read_compartments(grid, get_prompt("81", {"connectivity": 8, "color_large": 3, "color_small": 4}))
    assert sorted(map(len, rooms4)) == [1, 1, 2]
    assert sorted(map(len, rooms8)) == [1, 3]
    assert four[2][3] == 4 and four[3][1] == 0
    assert eight[2][3] == 3 and eight[3][1] == 4


@pytest.mark.parametrize("tamper", ("output", "frame", "same_paint", "areas", "count",
                                    "rule", "color", "family", "kinds", "nested",
                                    "doorway", "fake_doorway", "thickness", "first"))
def test_validator_rejects_tampering(examples, tamper):
    ex = deepcopy(next(e for e in examples
                       if e["metadata"]["n_doorways"] and e["metadata"]["n_nested"] == 0))
    if tamper == "output":
        ex["output"][0][0] = ex["metadata"]["color_large"]
    elif tamper == "frame":
        ex["input"][0][1] = ex["metadata"]["background_color"]
    elif tamper == "same_paint":
        ex["metadata"]["color_small"] = ex["metadata"]["color_large"]
    elif tamper == "areas":
        ex["metadata"]["areas_8"][0] += 1
    elif tamper == "count":
        ex["metadata"]["planned_room_count"] += 1
    elif tamper == "color":
        ex["metadata"]["wall_color"] = ex["metadata"]["background_color"]
    elif tamper == "family":
        ex["metadata"]["planned_family"] = "discarded-plan"
    elif tamper == "kinds":
        ex["metadata"]["room_kinds"] = ex["metadata"]["room_kinds"][1:]
    elif tamper == "nested":
        ex["metadata"]["n_nested"] += 1
    elif tamper == "doorway":
        ex["metadata"]["doorway_cells"] = ex["metadata"]["doorway_cells"][1:]
    elif tamper == "fake_doorway":
        cell = next((r, c) for r, row in enumerate(ex["input"])
                    for c, value in enumerate(row)
                    if value == ex["metadata"]["background_color"]
                    and [r, c] not in [list(x) for run in ex["metadata"]["doorway_cells"] for x in run])
        ex["metadata"]["doorway_cells"] = [[list(cell)]]
        ex["metadata"]["n_doorways"] = 1
    elif tamper == "thickness":
        ex["metadata"]["wall_thicknesses"] = [3, 1]
    elif tamper == "first":
        ex["metadata"]["largest_is_first"] = not ex["metadata"]["largest_is_first"]
    else:
        ex["metadata"]["connectivity"] = 6
    with pytest.raises(ValueError):
        module.validate_compartments(ex)


def test_validator_wants_the_frame_equally_deep_on_all_four_sides():
    """A solid wall row against one edge makes 'uniform outer frame' a lie."""
    control = example_from_picture(THREE_ROOMS, doorways=[[(3, 5)]],
                                   kinds=("rectangle", "rectangle", "rectangle"))
    module.validate_compartments(control)  # the hand-drawn control is sound
    deeper = example_from_picture(THREE_ROOMS[:1] + THREE_ROOMS,  # a second solid top row
                                  doorways=[[(4, 5)]],
                                  kinds=("rectangle", "rectangle", "rectangle"))
    assert deeper["metadata"]["frame_thickness"] == 1  # the ring reading still says one
    with pytest.raises(ValueError, match="all four sides"):
        module.validate_compartments(deeper)


def test_validator_wants_doorways_to_split_under_both_rules(monkeypatch):
    """Closing a doorway must split a compartment under the 8-rule as well.

    The picture below is a staircase wall: the two chambers already touch at a
    corner, so closing the gap splits them under the 4-rule only.  The wall
    rule would reject this drawing for its loose ends before the doorway clause
    is reached, so it is switched off to leave that clause alone under test.
    """
    picture = ("##############",
               "#........#...#",
               "#...##.###...#",
               "####.....#...#",
               "#........#...#",
               "##############")
    ex = example_from_picture(picture, doorways=[[(2, 6)]], kinds=("L", "L", "rectangle"))
    grid = np.array(ex["input"])
    closed = grid.copy()
    closed[2, 6] = ex["metadata"]["wall_color"]
    assert (len(module._visible_regions(closed, 4)[1])
            == len(module._visible_regions(grid, 4)[1]) + 1)
    assert (len(module._visible_regions(closed, 8)[1])
            == len(module._visible_regions(grid, 8)[1]))
    monkeypatch.setattr(module, "_wall_fault", lambda *args: None)
    with pytest.raises(ValueError, match="merge exactly two compartments"):
        module.validate_compartments(ex)


def test_construction_failure_is_not_replaced(monkeypatch):
    def fail(*_):
        raise ValueError("fixed room plan failed")
    monkeypatch.setattr(module, "_draw_scene", fail)
    with pytest.raises(ValueError, match="fixed room plan failed"):
        generate("81", 0)


def test_generation_stays_cheap(examples):
    assert np.mean([ex["metadata"]["attempts"] for ex in examples]) < 2.0
