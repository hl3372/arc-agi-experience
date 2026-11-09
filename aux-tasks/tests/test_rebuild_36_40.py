"""Prompt-based independent oracles and regressions for tasks 36--40."""
from collections import Counter, deque
from itertools import combinations
import copy
import re

import numpy as np
import pytest

from registry import generate
from prompts import get_prompt
from generators import clean_topology_36_37 as topology
from generators import mirror_check_38 as mirror
from generators import anchor_translate_39_40 as motion

NEIGHBORS4 = ((1, 0), (0, 1), (-1, 0), (0, -1))


def enclosed(grid, color, diagonal=False):
    a = np.asarray(grid)
    h, w = a.shape
    neighbors = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c] if diagonal else NEIGHBORS4
    seen = set()
    holes = []
    for r in range(h):
        for c in range(w):
            if (r, c) in seen or a[r, c] == color:
                continue
            q, cells, edge = deque([(r, c)]), set(), False
            seen.add((r, c))
            while q:
                y, x = q.popleft()
                cells.add((y, x))
                edge |= y == 0 or x == 0 or y == h - 1 or x == w - 1
                for dy, dx in neighbors:
                    p = y + dy, x + dx
                    if 0 <= p[0] < h and 0 <= p[1] < w and p not in seen and a[p] != color:
                        seen.add(p)
                        q.append(p)
            if not edge:
                holes.append(cells)
    return holes


def repair_oracle(grid, color):
    """Enumerate all pairs of visible ends, independently of the generator's ray scan."""
    a = np.asarray(grid)
    wall = a == color
    padded = np.pad(wall, 1)
    degree = sum(padded[y:y + a.shape[0], x:x + a.shape[1]] for y in range(3) for x in range(3) if (y, x) != (1, 1))
    ends = [tuple(p) for p in np.argwhere(wall & (degree == 1))]
    partners = {p: [] for p in ends}
    fills = {}
    for i, p in enumerate(ends):
        for q in ends[i + 1:]:
            dy, dx = q[0] - p[0], q[1] - p[1]
            n = max(abs(dy), abs(dx))
            if n <= 1 or not (dy == 0 or dx == 0 or abs(dy) == abs(dx)):
                continue
            sy, sx = dy // n, dx // n
            before, after = (p[0] - sy, p[1] - sx), (q[0] + sy, q[1] + sx)
            if not all(0 <= y < a.shape[0] and 0 <= x < a.shape[1] and wall[y, x] for y, x in (before, after)):
                continue
            run = {(p[0] + k * sy, p[1] + k * sx) for k in range(1, n)}
            if any(wall[y, x] for y, x in run):
                continue
            partners[p].append(q)
            partners[q].append(p)
            fills[p, q] = run
    if not ends or any(len(v) != 1 for v in partners.values()):
        raise ValueError("Unpaired or multiply paired visible wall ends")
    used = set()
    for run in fills.values():
        if used & run:
            raise ValueError("Crossing repair cells")
        used |= run
    # Two diagonals can cross halfway between cells without sharing a pixel.
    # Solve their segment equations independently of production's orientation test.
    for (p, q), (r, s) in combinations(fills, 2):
        p, q, r, s = (np.asarray(point, dtype=float) for point in (p, q, r, s))
        matrix = np.column_stack((q - p, r - s))
        if abs(np.linalg.det(matrix)) > 1e-8:
            t, u = np.linalg.solve(matrix, r - p)
            if 0 < t < 1 and 0 < u < 1:
                raise ValueError("Crossing repair segments")
    out = a.copy()
    for p in used:
        out[p] = color
    if not enclosed(out, color):
        raise ValueError("The repair must enclose an empty region")
    return out


def test_repair_rejects_diagonal_crossings_between_cell_centers():
    a = np.zeros((16, 16), int)
    a[2, 2:14] = a[13, 2:14] = 1
    for t in (3, 4, 5, 10, 11, 12):
        a[t, t] = a[t, 15-t] = 1
    for solve in (repair_oracle, topology.solve_repair):
        with pytest.raises(ValueError, match="[Cc]ross"):
            solve(a, 1)


def reflection_oracle(grid):
    a = np.asarray(grid)
    candidates = []
    for color in np.unique(a):
        coords = np.argwhere(a == color)
        if len(set(coords[:, 0])) == 1 and len(coords) == a.shape[1]:
            candidates.append((0, int(coords[0, 0])))
        if len(set(coords[:, 1])) == 1 and len(coords) == a.shape[0]:
            candidates.append((1, int(coords[0, 1])))
    if len(candidates) != 1:
        raise ValueError("Ambiguous divider")
    axis, index = candidates[0]
    first = a[:index] if axis == 0 else a[:, :index]
    second = a[index + 1:] if axis == 0 else a[:, index + 1:]
    return "true" if np.array_equal(first, np.flip(second, axis)) else "false"


def ground(grid):
    counts = Counter(np.asarray(grid).ravel().tolist()).most_common()
    assert counts[0][1] > 1.25 * counts[1][1]
    return counts[0][0]


def rotation_oracle(grid, anchor, angle):
    a = np.asarray(grid)
    bg = ground(a)
    (ar, ac), = np.argwhere(a == anchor)
    unseen = set(map(tuple, np.argwhere((a != bg) & (a != anchor))))
    shapes = []
    while unseen:
        part = {unseen.pop()}
        pending = list(part)
        for r, c in pending:
            for point in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if point in unseen:
                    unseen.remove(point)
                    part.add(point)
                    pending.append(point)
        if len(part) >= 2:
            shapes.append(part)
    assert len(shapes) == 1
    moving = shapes[0]
    out = a.copy()
    for point in moving:
        out[point] = bg
    for r, c in moving:
        vector = complex(int(c - ac), int(r - ar)) * (1j ** (angle // 90))
        y, x = int(vector.imag) + int(ar), int(vector.real) + int(ac)
        if 0 <= y < a.shape[0] and 0 <= x < a.shape[1]:
            out[y, x] = a[r, c]
    out[ar, ac] = anchor
    return out


def translation_oracle(grid, dr, dc):
    a = np.asarray(grid)
    bg = ground(a)
    # On a same-color component, a cell has a matching direct neighbor iff
    # the component has at least two cells. No component routine is reused.
    moving = np.zeros(a.shape, dtype=bool)
    moving[1:] |= (a[1:] == a[:-1]) & (a[1:] != bg)
    moving[:-1] |= (a[:-1] == a[1:]) & (a[:-1] != bg)
    moving[:, 1:] |= (a[:, 1:] == a[:, :-1]) & (a[:, 1:] != bg)
    moving[:, :-1] |= (a[:, :-1] == a[:, 1:]) & (a[:, :-1] != bg)
    out = a.copy()
    out[moving] = bg
    for r, c in np.argwhere(moving):
        y, x = r + dr, c + dc
        if 0 <= y < a.shape[0] and 0 <= x < a.shape[1]:
            out[y, x] = a[r, c]
    return out


def oracle(task, grid, prompt):
    if task in (36, 37):
        color = re.search(r'color (\S+?)[; ,.]', prompt).group(1)
        color = int(color) if color.isdigit() else color
        return ("closed" if enclosed(grid, color) else "open") if task == 36 else repair_oracle(grid, color).tolist()
    if task == 38:
        return reflection_oracle(grid)
    if task == 39:
        anchor = re.search(r'color (\S+) anchor', prompt).group(1)
        anchor = int(anchor) if anchor.isdigit() else anchor
        angle = int(re.search(r'object (\d+) degrees', prompt).group(1))
        return rotation_oracle(grid, anchor, angle).tolist()
    directions = {direction: int(count) for direction, count in re.findall(r'(up|down|left|right) (\d+)', prompt)}
    return translation_oracle(grid, directions.get('down', 0) - directions.get('up', 0), directions.get('right', 0) - directions.get('left', 0)).tolist()


@pytest.mark.parametrize('task', [36, 37, 38, 39, 40])
def test_visible_prompt_oracle_and_symbol_permutation(task):
    for seed in range(400):
        ex = generate(str(task), seed)
        assert oracle(task, ex['input'], get_prompt(str(task), ex['metadata'])) == ex['output'], seed
        if seed < 60:
            palette = list(np.random.default_rng(seed + 917).permutation(list('ABCDEFGHIJ')))
            a = np.asarray(palette)[np.asarray(ex['input'])].tolist()
            expected = np.asarray(palette)[np.asarray(ex['output'])].tolist() if isinstance(ex['output'], list) else ex['output']
            assert oracle(task, a, get_prompt(str(task), ex['metadata'], color_palette=palette)) == expected, seed


@pytest.mark.parametrize('task', [36, 37, 38, 39, 40])
def test_validator_rejects_corrupt_output(task):
    ex = copy.deepcopy(generate(str(task), 10))
    if isinstance(ex['output'], str):
        ex['output'] = 'wrong'
    else:
        ex['output'][0][0] = (ex['output'][0][0] + 1) % 10
    with pytest.raises(ValueError):
        if task in (36, 37):
            topology.validate_example(ex, task)
        elif task == 38:
            mirror.validate_example(ex)
        else:
            motion.validate_example(ex, task)


def test_motion_input_does_not_depend_on_variant(monkeypatch):
    for seed in range(60):
        monkeypatch.setattr(motion, '_rotation_angle', lambda rng: 90)
        first = generate('39', seed)['input']
        monkeypatch.setattr(motion, '_rotation_angle', lambda rng: 270)
        assert generate('39', seed)['input'] == first
        monkeypatch.setattr(motion, '_translation', lambda rng, h, w: (0, 0, 'identity'))
        first = generate('40', seed)['input']
        monkeypatch.setattr(motion, '_translation', lambda rng, h, w: (h, -w, 'clear'))
        assert generate('40', seed)['input'] == first


def test_balanced_reflection_foils_keep_row_and_column_counts():
    rng = np.random.default_rng(71)
    source = np.tile(np.arange(5), (12, 4))
    for mode in ('balanced_cells', 'balanced_blocks'):
        for _ in range(30):
            a, b = mirror._foil(rng, source, list(range(5)), mode)
            assert not np.array_equal(a, b)
            for color in range(5):
                assert np.array_equal((a == color).sum(axis=0), (b == color).sum(axis=0))
                assert np.array_equal((a == color).sum(axis=1), (b == color).sum(axis=1))


def test_background_details_stay_fixed_unless_covered():
    a = np.zeros((9, 12), dtype=int)
    a[2:4, 2:4] = 1
    a[2, 7] = 2
    a[7, 9] = 2
    expected = a.copy()
    expected[2:4, 2:4] = 0
    expected[2:4, 6:8] = 1
    assert np.array_equal(motion.solve_translation(a, 0, 4), expected)
    assert np.array_equal(translation_oracle(a, 0, 4), expected)


def test_fully_clipped_and_zero_motion_have_defined_outputs():
    a = np.zeros((8, 9), dtype=int)
    a[2:4, 2:4] = 1
    a[5, 7] = 2
    assert np.array_equal(motion.solve_translation(a, 0, 0), a)
    out = motion.solve_translation(a, 8, 0)
    assert (out == 1).sum() == 0 and out[5, 7] == 2


def test_diagonal_contact_does_not_connect_empty_space():
    a = [[0, 1, 0], [1, 2, 1], [0, 1, 0]]
    assert topology.solve_closed(a, 1) == 'closed'
    assert len(enclosed(a, 1)) == 1 and not enclosed(a, 1, diagonal=True)


@pytest.mark.parametrize('orientation', ['horizontal', 'vertical', 'main_diagonal', 'anti_diagonal'])
def test_single_cell_repairs_in_every_orientation(orientation):
    a = np.zeros((15, 15), dtype=int)
    if orientation in ('horizontal', 'vertical'):
        a[2, 2:13] = a[12, 2:13] = 1
        a[2:13, 2] = a[2:13, 12] = 1
        gap = (2, 7) if orientation == 'horizontal' else (7, 2)
    else:
        r, c = np.mgrid[:15, :15]
        a[abs(r - 7) + abs(c - 7) == 6] = 1
        gap = (4, 10) if orientation == 'main_diagonal' else (4, 4)
    expected = a.copy()
    a[gap] = 0
    assert np.array_equal(topology.solve_repair(a, 1), expected)
    assert np.array_equal(repair_oracle(a, 1), expected)


def test_unpaired_wall_end_is_rejected():
    a = np.zeros((9, 9), dtype=int)
    a[2, 2:5] = 1
    with pytest.raises(ValueError):
        topology.solve_repair(a, 1)


def test_paired_enclosures_have_equal_local_cues_but_different_answers():
    a = np.zeros((15, 21), dtype=bool)
    a[1, 1:20] = a[13, 1:20] = True
    a[1:14, 1] = a[1:14, 10] = a[1:14, 19] = True
    pair = topology._paired_breaches(np.random.default_rng(11), a)
    assert pair is not None
    (closed, _), (opened, _), lengths = pair
    assert enclosed(closed.astype(int), 1) and not enclosed(opened.astype(int), 1)
    assert closed.sum() == opened.sum() and len(lengths) == 2
    signatures = []
    for wall in (closed, opened):
        padded = np.pad(wall, 1)
        degree = sum(padded[y:y + 15, x:x + 21].astype(int) for y in range(3) for x in range(3) if (y, x) != (1, 1))
        signatures.append(Counter(degree[wall]))
    assert signatures[0] == signatures[1]


@pytest.mark.parametrize('task', [36, 37, 38, 39, 40])
def test_diversity_and_nontrivial_cases_remain_present(task):
    examples = [generate(str(task), seed) for seed in range(1000)]
    metas = [ex['metadata'] for ex in examples]
    sizes = np.array([np.asarray(ex['input']).shape for ex in examples])
    colors = Counter(m['n_colors'] for m in metas)
    limit = 320 if task in (36, 37) else 550 if task == 39 else 200
    assert max(colors) == 10 and max(colors.values()) < limit, colors
    assert (sizes.max(axis=0) == 30).all()
    assert .015 < np.mean(sizes.prod(axis=1) >= 700) < .08
    assert max(m['attempts'] for m in metas) < 100
    if task in (36, 37):
        assert .35 < np.mean([m['background'] in ('solid', 'accents') for m in metas]) < .65
        assert .05 < np.mean([m['n_colors'] >= 9 for m in metas]) < .15
        assert all(m['background'] != 'striped_panels' for m in metas)
        assert .48 < np.mean([m['silhouette'] in ('rectangle', 'chamfer', 'ellipse') for m in metas]) < .80
        assert len({m['silhouette'] for m in metas}) == len(topology.FAMILIES)
        assert set(sizes[:, 0]) == set(sizes[:, 1]) == set(range(8, 31))
        assert sum(m['n_holes'] > 1 for m in metas) > 40
        if task == 36:
            assert .42 < np.mean([ex['output'] == 'closed' for ex in examples]) < .58
            assert sum(m['paired_local_statistics'] for m in metas) > 30
            assert max(v for m in metas for v in m['gap_widths']) > 8
        else:
            assert max(v for m in metas for v in m['gap_lengths']) > 10
            assert {d for m in metas for d in m['gap_directions']} == {'horizontal', 'vertical', 'main_diagonal', 'anti_diagonal'}
            assert .18 < np.mean([m['n_gaps'] == 2 for m in metas]) < .38
    elif task == 38:
        assert .42 < np.mean([ex['output'] == 'true' for ex in examples]) < .58
        assert len({m['content_family'] for m in metas}) == len(mirror.CONTENT)
        negatives = [ex['metadata'] for ex in examples if ex['output'] == 'false']
        assert .55 < np.mean([m['balanced_foil'] for m in negatives]) < .85
        # Previously 87--173 retries: the balanced blocks must preserve room
        # for the full palette even on a very small, colorful half-grid.
        for seed, expected_colors in ((6545, 10), (7155, 8), (1612, 9), (3472, 9)):
            ex = generate('38', seed)
            assert ex['metadata']['n_colors'] == expected_colors
            assert ex['metadata']['attempts'] < 15
    else:
        assert min(sizes.ravel()) == (4 if task == 40 else 5)
        assert .02 < np.mean([ex['input'] == ex['output'] for ex in examples]) < .10
        assert .03 < np.mean([m['fully_clipped'] for m in metas]) < .22
        if task == 39:
            assert .30 < np.mean([m['object_color_count'] == 1 for m in metas]) < .50
            assert .30 < np.mean([2 <= m['object_color_count'] <= 3 for m in metas]) < .50
            assert .10 < np.mean([m['object_color_count'] >= 4 for m in metas]) < .30
            assert .20 < np.mean([m['texture_cells'] > 0 for m in metas]) < .40
            assert max(m['object_area'] for m in metas) > 200
            assert len({m['shape_family'] for m in metas}) == 10
            assert {m['angle'] for m in metas} == {0, 90, 180, 270}
        else:
            assert .20 < np.mean([m['texture_cells'] > 0 for m in metas]) < .45
            assert len({m['background'] for m in metas}) == 7
            assert max(max(abs(m['dr']), abs(m['dc'])) for m in metas) >= 25
