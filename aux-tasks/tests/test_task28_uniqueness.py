"""Visible-input regressions for task28; no planted roles or local-cover cap."""
from __future__ import annotations

from collections import Counter
from itertools import combinations
import numpy as np
import pytest

from generators.fit_holes_28 import (
    enumerate_final_outputs,
    generate_fit_holes,
    parse_scene,
    validate_fit_holes,
)


# This oracle deliberately uses complete per-hole cover lists and a separate
# global join, unlike the production solver's joint bit-mask cell search.
def _flood(points):
    remaining = set(points)
    groups = []
    while remaining:
        frontier = [remaining.pop()]
        group = set(frontier)
        while frontier:
            r, c = frontier.pop()
            for neighbor in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    group.add(neighbor)
                    frontier.append(neighbor)
        groups.append(frozenset(group))
    return groups


def visible_scene(grid):
    grid = np.asarray(grid)
    background = Counter(grid.flat).most_common(1)[0][0]
    foreground = _flood(zip(*np.where(grid != background)))
    objects = sorted((cells for cells in foreground if len(cells) >= 2), key=min)
    ignored = set().union(*(cells for cells in foreground if len(cells) == 1))
    holes, pieces, hosts = [], [], []
    h, w = grid.shape
    universe = {(r, c) for r in range(-1, h+1) for c in range(-1, w+1)}
    for cells in objects:
        voids = [region for region in _flood(universe-cells) if (-1, -1) not in region]
        if voids:
            assert all(all(grid[p] == background for p in hole) for hole in voids)
            hosts.append(cells)
            holes.extend(voids)
        else:
            pieces.append(cells)
    assert holes and pieces
    return background, holes, pieces, hosts, ignored


def local_covers(hole, pieces):
    """ALL local covers, with no truncation even if many cannot join globally."""
    placements = []
    for i, piece in enumerate(pieces):
        anchor = min(piece)
        for cell in hole:
            dr, dc = cell[0]-anchor[0], cell[1]-anchor[1]
            placed = frozenset((r+dr, c+dc) for r, c in piece)
            if placed <= hole:
                placements.append((i, dr, dc, placed))
    covers = set()

    def visit(empty, used, chosen):
        if not empty:
            covers.add(tuple(sorted(chosen)))
            return
        first = min(empty)
        for i, dr, dc, placed in placements:
            if i not in used and first in placed and placed <= empty:
                visit(empty-placed, used | {i}, chosen+[(i, dr, dc)])

    visit(hole, set(), [])
    return sorted(covers, key=lambda cover: (len(cover), cover))


def oracle_outputs(grid, limit=2, local_limit=None):
    grid = np.asarray(grid)
    background, holes, pieces, _, _ = visible_scene(grid)
    options = [local_covers(hole, pieces) for hole in holes]
    # A hole with no exact local tiling remains empty. All the others must be
    # filled simultaneously, without sharing a physical loose piece.
    options = [covers for covers in options if covers]
    if local_limit is not None:
        options = [covers[:local_limit] for covers in options]
    outputs = {}

    def join(index, used, selected):
        if len(outputs) >= limit:
            return
        if index == len(options):
            result = grid.copy()
            for i in used:
                for p in pieces[i]:
                    result[p] = background
            for i, dr, dc in selected:
                for r, c in pieces[i]:
                    result[r+dr, c+dc] = grid[r, c]
            outputs.setdefault(result.tobytes(), result)
            return
        for cover in options[index]:
            physical_ids = {entry[0] for entry in cover}
            if not used & physical_ids:
                join(index+1, used | physical_ids, selected+list(cover))

    join(0, set(), [])
    return list(outputs.values())


def compressed_oracle_outputs(grid, limit=2):
    """Complete per-hole covers joined by inventories, independently of cell DFS.

    Local covers are first enumerated with physical IDs by the literal oracle.
    Only afterwards are interchangeable inventories compressed for the global
    join. Destination paint and all erased physical subsets remain observable.
    """
    grid = np.asarray(grid)
    background, holes, pieces, _, _ = visible_scene(grid)
    classes, members, piece_class = [], [], []
    for index, cells in enumerate(pieces):
        r0, c0 = min(r for r, _ in cells), min(c for _, c in cells)
        shape = tuple(sorted((r-r0, c-c0) for r, c in cells))
        key = shape, grid[min(cells)].item()
        if key not in classes:
            classes.append(key)
            members.append([])
        kind = classes.index(key)
        piece_class.append(kind)
        members[kind].append(index)
    counts = tuple(map(len, members))
    options = []
    for hole in holes:
        covers = local_covers(hole, pieces)
        if not covers:
            continue
        unique = set()
        for cover in covers:
            used = Counter(piece_class[i] for i, _, _ in cover)
            paint = tuple(sorted((r+dr, c+dc, grid[r, c].item())
                                 for i, dr, dc in cover for r, c in pieces[i]))
            unique.add((tuple(used[i] for i in range(len(classes))), paint))
        options.append(sorted(unique))
    outputs = {}

    def erase_sources(output, used, kind):
        if len(outputs) >= limit:
            return
        if kind == len(classes):
            outputs.setdefault(output.tobytes(), output.copy())
            return
        for selected in combinations(members[kind], used[kind]):
            for index in selected:
                for p in pieces[index]:
                    output[p] = background
            erase_sources(output, used, kind+1)
            for index in selected:
                for p in pieces[index]:
                    output[p] = grid[p]
            if len(outputs) >= limit:
                return

    seen = set()

    def join(index, used, paint):
        if len(outputs) >= limit or (index, used, paint) in seen:
            return
        seen.add((index, used, paint))
        if index == len(options):
            result = grid.copy()
            for r, c, color in paint:
                result[r, c] = color
            erase_sources(result, used, 0)
            return
        for local, colored in options[index]:
            after = tuple(x+y for x, y in zip(used, local))
            if all(x <= cap for x, cap in zip(after, counts)):
                join(index+1, after, paint+colored)

    join(0, (0,)*len(classes), ())
    return list(outputs.values())


def _frame(grid, top, left, height, width, color=2):
    grid[top:top+height, left:left+width] = color
    grid[top+1:top+height-1, left+1:left+width-1] = 0


def test_distinct_identical_shape_pieces_have_two_colored_completions():
    grid = np.zeros((12, 14), dtype=np.uint8)
    _frame(grid, 1, 1, 4, 4)
    grid[1:3, 8:10] = 3
    grid[6:8, 8:10] = 4
    answers = enumerate_final_outputs(grid)
    assert len(answers) == len(oracle_outputs(grid)) == 2
    with pytest.raises(ValueError, match="one final colored output"):
        validate_fit_holes(grid, answers[0])


def test_more_than_two_local_covers_are_needed_to_reject_ambiguity():
    grid = np.zeros((18, 24), dtype=np.uint8)
    _frame(grid, 1, 1, 3, 8)  # 1x6 hole: length6, length2+4, or length3+3.
    _frame(grid, 6, 1, 3, 4)  # Reserves the only length2 piece.
    _frame(grid, 11, 1, 3, 6)  # Reserves the only length4 piece.
    for row, width, color in ((1, 6, 3), (4, 2, 4), (7, 4, 5), (10, 3, 6), (13, 3, 7)):
        grid[row, 14:14+width] = color
    _, holes, pieces, _, _ = visible_scene(grid)
    largest_hole = max(holes, key=len)
    assert len(local_covers(largest_hole, pieces)) > 2
    # The former per-hole cap falsely accepted the full-length piece alone.
    assert len(oracle_outputs(grid, local_limit=2)) == 1
    assert len(oracle_outputs(grid)) == len(enumerate_final_outputs(grid)) == 2


def test_assignment_permutations_do_not_count_as_different_outputs():
    grid = np.zeros((12, 14), dtype=np.uint8)
    _frame(grid, 1, 1, 4, 4)
    grid[1, 8:10] = 3
    grid[5, 8:10] = 3
    expected = grid.copy()
    expected[1, 8:10] = expected[5, 8:10] = 0
    expected[2:4, 2:4] = 3
    assert len(oracle_outputs(grid)) == 1
    assert validate_fit_holes(grid, expected)['distinct_outputs'] == 1


def test_monochrome_roles_and_same_piece_color_texture():
    grid = np.zeros((11, 15), dtype=np.uint8)
    _frame(grid, 1, 1, 4, 4, color=2)
    grid[2:4, 9:11] = 4
    grid[8, 13] = 4  # Same-color isolated texture is ignored.
    expected = grid.copy()
    expected[2:4, 2:4] = grid[2:4, 9:11]
    expected[2:4, 9:11] = 0
    scene = parse_scene(grid)
    assert len(scene.hosts) == len(scene.pieces) == 1
    assert len(scene.texture) == 1
    assert np.array_equal(oracle_outputs(grid)[0], expected)
    validate_fit_holes(grid, expected)


@pytest.mark.parametrize('violation', ['host', 'piece', 'role_color'])
def test_multicolor_objects_and_host_colored_pieces_are_rejected(violation):
    grid = np.zeros((11, 15), dtype=np.uint8)
    _frame(grid, 1, 1, 4, 4, color=2)
    grid[2:4, 9:11] = 4
    if violation == 'host':
        grid[1, 2] = 5
    elif violation == 'piece':
        grid[2, 9] = 5
    else:
        grid[2:4, 9:11] = 2
    with pytest.raises(ValueError, match='monochrome|different colors'):
        parse_scene(grid)


def test_tampered_output_is_rejected():
    sample = generate_fit_holes(np.random.default_rng(175))
    grid, output = np.array(sample['input']), np.array(sample['output'])
    hole = parse_scene(grid).holes[0]
    position = min(hole)
    output[position] = (int(output[position])+1) % 10
    with pytest.raises(ValueError, match="output disagrees"):
        validate_fit_holes(grid, output)


def test_tampered_input_creates_a_second_possible_final_grid():
    grid = np.zeros((12, 16), dtype=np.uint8)
    _frame(grid, 1, 1, 4, 4)
    grid[1:3, 10:12] = 3
    output = enumerate_final_outputs(grid)[0]
    validate_fit_holes(grid, output)
    grid[7:9, 10:12] = 4
    with pytest.raises(ValueError, match="one final colored output"):
        validate_fit_holes(grid, output)


def test_rotated_piece_is_not_a_legal_translation():
    grid = np.zeros((12, 15), dtype=np.uint8)
    grid[1:5, 1:6] = 2
    hole = {(2, 2), (3, 2), (3, 3), (3, 4)}
    for p in hole:
        grid[p] = 0
    # Same L tetromino rotated a quarter turn; no translated placement fits.
    for r, c in ((0, 0), (0, 1), (1, 0), (2, 0)):
        grid[2+r, 10+c] = 3
    for solve in (enumerate_final_outputs, oracle_outputs):
        answers = solve(grid)
        assert len(answers) == 1 and np.array_equal(answers[0], grid)


def test_unmatched_hole_stays_empty_even_if_a_piece_fits_partly():
    grid = np.zeros((14, 22), dtype=np.uint8)
    _frame(grid, 1, 1, 4, 4)   # Two horizontal dominoes fill this 2x2 hole.
    _frame(grid, 7, 1, 4, 5)   # 2x3 cannot be completed with only two dominoes.
    grid[2, 14:16] = grid[7, 14:16] = 3
    expected = grid.copy()
    expected[2, 14:16] = expected[7, 14:16] = 0
    expected[2:4, 2:4] = 3
    for solve in (enumerate_final_outputs, oracle_outputs):
        answers = solve(grid)
        assert len(answers) == 1 and np.array_equal(answers[0], expected)
    validate_fit_holes(grid, expected)
    expected[8, 2:4] = 3
    with pytest.raises(ValueError, match='output disagrees'):
        validate_fit_holes(grid, expected)


def test_competing_fillable_holes_cannot_be_arbitrarily_left_empty():
    grid = np.zeros((14, 22), dtype=np.uint8)
    _frame(grid, 1, 1, 4, 4)
    _frame(grid, 7, 1, 4, 4)
    grid[2:4, 14:16] = 3  # Either hole fits individually; both cannot be filled.
    assert enumerate_final_outputs(grid) == oracle_outputs(grid) == []
    with pytest.raises(ValueError, match='one final colored output'):
        validate_fit_holes(grid, grid)


def test_generated_scenes_match_an_independent_complete_cover_oracle():
    seeds = list(range(400)) + [175, 185, 357, 366, 390, 524, 853, 1094, 6558, 8579, 9689, 9999]
    for seed in seeds:
        example = generate_fit_holes(np.random.default_rng(seed))
        legal = compressed_oracle_outputs(example['input'])
        assert len(legal) == 1, seed
        assert np.array_equal(legal[0], example['output']), seed


def test_former_two_color_seed_respects_new_role_palette_without_changing_size():
    example = generate_fit_holes(np.random.default_rng(9295))
    assert np.shape(example['input']) == (9, 7)
    assert len(np.unique(example['input'])) == example['metadata']['requested_palette_size'] >= 3
    assert example['metadata']['multicolor_objects'] == 0
    assert example['metadata']['size_attempts'] == 1
    legal = compressed_oracle_outputs(example['input'])
    assert len(legal) == 1 and np.array_equal(legal[0], example['output'])


@pytest.mark.parametrize('seed,dimensions', [(10618, (19, 14)), (102913, (22, 20)), (493, (29, 30))])
def test_fixed_budgets_survive_former_capacity_and_dense_color_regressions(seed, dimensions):
    example = generate_fit_holes(np.random.default_rng(seed))
    assert np.shape(example['input']) == dimensions
    metadata = example['metadata']
    assert metadata['requested_hosts'] == metadata['n_hosts']
    assert metadata['requested_holes'] == metadata['n_holes']
    assert metadata['requested_pieces'] == metadata['n_loose_pieces']
    assert metadata['requested_palette_size'] == metadata['n_colors']
    legal = compressed_oracle_outputs(example['input'])
    assert len(legal) == 1 and np.array_equal(legal[0], example['output'])


def test_partial_identical_class_keeps_every_distinct_source_erasure():
    grid = np.zeros((14, 19), dtype=np.uint8)
    _frame(grid, 1, 1, 4, 4)
    for top in (1, 6, 11):
        grid[top:top+2, 12:14] = 3
    literal = oracle_outputs(grid, limit=10)
    assert len(literal) == 3
    assert len(enumerate_final_outputs(grid, limit=2)) == 2
    for solve in (enumerate_final_outputs, compressed_oracle_outputs):
        actual = solve(grid, limit=10)
        assert {x.tobytes() for x in actual} == {x.tobytes() for x in literal}


def test_two_partial_classes_enumerate_the_product_of_source_subsets():
    grid = np.zeros((22, 24), dtype=np.uint8)
    _frame(grid, 1, 1, 4, 4)
    _frame(grid, 7, 1, 3, 5)
    for top in (1, 6, 11):
        grid[top:top+2, 12:14] = 3
        grid[top, 18:21] = 4
    literal = oracle_outputs(grid, limit=30)
    assert len(literal) == 9
    actual = enumerate_final_outputs(grid, limit=30)
    assert {x.tobytes() for x in actual} == {x.tobytes() for x in literal}


def _many_small_cavities(holes=12, source_count=24, split=True):
    grid = np.zeros((30, 30), dtype=np.uint8)
    for index in range(holes):
        top, left = 1+5*(index//3), 1+5*(index % 3)
        _frame(grid, top, left, 4, 4)
    for index in range(source_count):
        top, left = 1+2*(index//3), 18+4*(index % 3)
        grid[top:top+(1 if split else 2), left:left+2] = 3
    return grid


def test_twelve_cavities_and_twenty_four_identical_parts_are_not_factorial():
    grid = _many_small_cavities()
    stats = {}
    outputs = enumerate_final_outputs(grid, stats=stats)
    # Independent closed-form answer: all 24 dominoes are needed by area, all
    # destination cells are the same color, and every source must disappear.
    expected = grid.copy()
    expected[expected == 3] = 0
    for index in range(12):
        top, left = 1+5*(index//3), 1+5*(index % 3)
        expected[top+1:top+3, left+1:left+3] = 3
    assert len(outputs) == 1 and np.array_equal(outputs[0], expected)
    assert stats['piece_classes'] == 1 and stats['search_nodes'] < 100
    assert stats['fillable_holes'] == 12
    independent = compressed_oracle_outputs(grid)
    assert len(independent) == 1 and np.array_equal(independent[0], expected)


def test_many_individually_fillable_holes_with_shared_shortage_have_no_output():
    grid = _many_small_cavities(holes=8, source_count=7, split=False)
    # Square sources need an extra blank row for spatial separation.
    grid[:, 18:] = 0
    for index in range(7):
        top, left = 1+4*(index//3), 18+4*(index % 3)
        grid[top:top+2, left:left+2] = 3
    stats = {}
    assert enumerate_final_outputs(grid, stats=stats) == []
    assert stats['fillable_holes'] == 8
    assert compressed_oracle_outputs(grid) == []


def test_grouped_search_matches_physical_id_oracle_on_small_random_inventories():
    rng = np.random.default_rng(280912)
    shapes = [((0, 0), (0, 1)), ((0, 0), (1, 0)),
              ((0, 0), (0, 1), (0, 2)),
              ((0, 0), (0, 1), (1, 0)),
              ((0, 0), (0, 1), (1, 0), (1, 1))]
    for _ in range(200):
        grid = np.zeros((24, 26), dtype=np.uint8)
        for left in (1, 9):
            grid[1:6, left:left+6] = 2
            shape = shapes[int(rng.integers(len(shapes)))]
            for r, c in shape:
                grid[r+2, c+left+1] = 0
        for index in range(int(rng.integers(1, 6))):
            shape = shapes[int(rng.integers(len(shapes)))]
            top, left = 10+5*(index//3), 2+7*(index % 3)
            color = int(rng.integers(3, 5))
            for r, c in shape:
                grid[top+r, left+c] = color
        expected = {x.tobytes() for x in oracle_outputs(grid, limit=10000)}
        for solve in (enumerate_final_outputs, compressed_oracle_outputs):
            actual = {x.tobytes() for x in solve(grid, limit=10000)}
            assert actual == expected


def test_complete_cover_reader_is_independent_of_numeric_color_symbols():
    alphabet = np.array(list('KQAFMZLTVB'))
    for seed in [0, 4, 5, 28, 41, 44, 136, 197, 493, 551, 825, 997]:
        example = generate_fit_holes(np.random.default_rng(seed))
        remapped_input = alphabet[np.asarray(example['input'])]
        expected = alphabet[np.asarray(example['output'])]
        outputs = compressed_oracle_outputs(remapped_input)
        assert len(outputs) == 1 and np.array_equal(outputs[0], expected)
