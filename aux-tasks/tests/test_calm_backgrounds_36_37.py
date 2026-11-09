"""Visible topology and calm background coverage for tasks 36 and 37."""
from collections import Counter
from itertools import combinations
import re

import numpy as np
import pytest

from generators import clean_topology_36_37 as G
from prompts import get_prompt
from registry import generate


def _color(prompt):
    label = re.search(r'color (\S+?)[; ,.]', prompt).group(1)
    return int(label) if label.isdigit() else label


def _closed(grid, color):
    """Flood from the canvas boundary through every non-wall color alike."""
    a = np.asarray(grid)
    h, w = a.shape
    reached = {(r, c) for r in range(h) for c in range(w)
               if (r in (0, h-1) or c in (0, w-1)) and a[r, c] != color}
    pending = list(reached)
    for r, c in pending:
        for y, x in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if 0 <= y < h and 0 <= x < w and (y, x) not in reached and a[y, x] != color:
                reached.add((y, x))
                pending.append((y, x))
    return "closed" if np.count_nonzero(a != color) > len(reached) else "open"


def _repair(grid, color):
    """Try pairs of ends, independently of the generator's outward-ray scan."""
    a = np.asarray(grid)
    cells = set(map(tuple, np.argwhere(a == color)))
    adjacent = {p: [q for q in cells if max(abs(p[0]-q[0]), abs(p[1]-q[1])) == 1]
                for p in cells}
    ends = {p for p, neighbors in adjacent.items() if len(neighbors) == 1}
    partners = {p: [] for p in ends}
    runs = []
    for first, last in combinations(ends, 2):
        dy, dx = last[0]-first[0], last[1]-first[1]
        steps = max(abs(dy), abs(dx))
        if steps < 2 or not (dy == 0 or dx == 0 or abs(dy) == abs(dx)):
            continue
        dr, dc = dy//steps, dx//steps
        if adjacent[first] != [(first[0]-dr, first[1]-dc)] or adjacent[last] != [(last[0]+dr, last[1]+dc)]:
            continue
        run = {(first[0]+i*dr, first[1]+i*dc) for i in range(1, steps)}
        if run & cells:
            continue
        partners[first].append(last)
        partners[last].append(first)
        runs.append(run)
    assert partners and all(len(matches) == 1 for matches in partners.values())
    assert len(set.union(*runs)) == sum(map(len, runs))
    out = a.copy()
    for run in runs:
        for point in run:
            out[point] = color
    assert _closed(out, color) == "closed"
    return out.tolist()


@pytest.fixture(scope="module", params=("36", "37"))
def examples(request):
    return request.param, [generate(request.param, seed) for seed in range(1000)]


def test_every_nonwall_color_is_passable_and_repair_changes_only_gaps(examples):
    task, cases = examples
    for seed, ex in enumerate(cases):
        prompt = get_prompt(task, ex['metadata'])
        color = _color(prompt)
        expected = _closed(ex['input'], color) if task == '36' else _repair(ex['input'], color)
        assert expected == ex['output'], (task, seed)
        if task == '37':
            a, b = np.asarray(ex['input']), np.asarray(ex['output'])
            assert np.all(b[a != b] == color)
            assert np.all(a[a == color] == b[a == color])
        if seed < 100:
            palette = np.random.default_rng(5183+seed).permutation(list("ABCDEFGHIJ"))
            mapped = palette[np.asarray(ex['input'])].tolist()
            symbol = _color(get_prompt(task, ex['metadata'], color_palette=palette.tolist()))
            answer = _closed(mapped, symbol) if task == '36' else _repair(mapped, symbol)
            expected = ex['output'] if task == '36' else palette[np.asarray(ex['output'])].tolist()
            assert answer == expected


def test_background_expansion_keeps_palette_and_geometry_coverage(examples):
    task, cases = examples
    m = [ex['metadata'] for ex in cases]
    styles = Counter(x['background'] for x in m)
    assert set(styles) == {'solid', *G.FILLS}
    assert .40 < (styles['solid']+styles['accents'])/len(m) < .57
    assert all(styles[style] > .045*len(m) for style in G.FILLS if style != 'accents')
    assert .50 < np.mean([x['n_colors'] <= 4 for x in m]) < .68
    assert .02 < np.mean([x['n_colors'] == 10 for x in m]) < .07
    assert set(x['silhouette'] for x in m) == set(G.FAMILIES)
    assert max(x['attempts'] for x in m) < 40
    assert all(len(np.unique(ex['input'])) == ex['metadata']['n_colors'] for ex in cases)
    if task == '36':
        assert .42 < np.mean([ex['output'] == 'closed' for ex in cases]) < .58
        assert .065 < np.mean([x['paired_local_statistics'] for x in m]) < .14
        assert .08 < np.mean([ex['output'] == 'closed' and bool(ex['metadata']['gap_widths']) for ex in cases]) < .18
    else:
        assert .20 < np.mean([x['n_gaps'] == 2 for x in m]) < .34
        assert .13 < np.mean([x['n_holes'] > 1 for x in m]) < .27
        assert max(v for x in m for v in x['gap_lengths']) >= 16
        assert {d for x in m for d in x['gap_directions']} == {'horizontal', 'vertical', 'main_diagonal', 'anti_diagonal'}


@pytest.mark.parametrize('style', G.FILLS)
def test_repair_gaps_show_the_underlying_field_without_added_markers(style):
    outline = np.zeros((22, 26), bool)
    outline[3, 3:23] = outline[18, 3:23] = True
    outline[3:19, 3] = outline[3:19, 22] = True
    gaps = np.zeros_like(outline)
    gaps[3, 8:13] = gaps[18, 15:18] = True
    wall = outline & ~gaps
    single_cell = np.zeros_like(wall)
    single_cell[3, 3] = True
    for seed in range(24):
        colors = [9, 2, 4, 7, 0]
        actual = G._paint(np.random.default_rng(seed), wall, colors, style, protected=outline)
        empty = G._paint(np.random.default_rng(seed), single_cell, colors, style, protected=outline)
        assert actual is not None and empty is not None
        assert np.array_equal(actual == colors[0], wall)
        # Reducing the wall to one unrelated cell leaves exactly the same
        # field at the repair positions: no marker or pattern defect.
        assert np.array_equal(actual[gaps], empty[gaps])


def test_pattern_phases_and_scales_vary_within_the_same_style(examples):
    _, cases = examples
    for style in ('bands', 'diagonal_bands', 'repeated_motifs', 'sparse_pattern', 'large_patches', 'local_patches'):
        selected = [ex for ex in cases if ex['metadata']['background'] == style
                    and min(np.shape(ex['input'])) >= 12]
        scales = {repr(ex['metadata']['background_scale']) for ex in selected}
        assert len(scales) >= 5, style
    ordinary = [ex['metadata']['background_base_share'] for ex in cases
                if ex['metadata']['background'] in ('accents', 'repeated_motifs', 'sparse_pattern', 'large_patches', 'local_patches')]
    assert np.mean(np.asarray(ordinary) >= .50) > .95
