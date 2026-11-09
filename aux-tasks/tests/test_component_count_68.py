"""Input/prompt oracle and scene contracts for the merged counts 68a/68b.

The oracle reads the prompt and nothing else: it works out what the background
is (a solid colour or a pattern), reduces the grid to the object layer, and
counts same-colour groups with its own union-find under the stated
connectivity, with or without lone cells, optionally inside the marked
rectangle.  Its field renderers are written here from the task contract, so a
wrong parameter in the generator cannot be hidden by a matching bug in the
renderer; the search for those parameters is cross-checked against the
assumption-free parameter sweep in ``pattern_fields.brute`` on a sample, and
that sweep is what proves the reading unique.
"""
from collections import Counter
import re
import time

import numpy as np
import pytest

from registry import generate
from prompts import get_prompt
from generators import component_count_68 as subject
from generators import pattern_fields as pf

SEEDS = 800
SWEEP_SEEDS = 40          # grids given the full brute-force parameter sweep


# ---------------------------------------------------------------------------
# An independent reader
# ---------------------------------------------------------------------------

def groups(grid, connectivity, background):
    """Union-find over the cells that are not background; ``background`` is a
    per-cell mask, so it works for a plain field and a patterned one alike."""
    h, w = len(grid), len(grid[0])
    parent = {}

    def find(p):
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p

    offsets = [(-1, 0), (0, -1)] + ([(-1, -1), (-1, 1)] if connectivity == 8 else [])
    for r in range(h):
        for c in range(w):
            if background[r][c]:
                continue
            p = (r, c)
            parent[p] = p
            for dr, dc in offsets:
                q = (r + dr, c + dc)
                if q in parent and grid[q[0]][q[1]] == grid[r][c]:
                    parent[find(p)] = find(q)
    out = {}
    for p in parent:
        out.setdefault(find(p), set()).add(p)
    return list(out.values())


def render_field(family, p, h, w):
    """The six background fields, written out from the task contract."""
    out = [[0] * w for _ in range(h)]
    cuts = list(p.get("cuts", []))
    for r in range(h):
        for c in range(w):
            if family == "bands":
                key = r if p["axis"] == 0 else c
                v = int((key + p["phase"]) % p["period"] < p["thickness"])
            elif family == "diagonal_bands":
                key = (r + c) if p["sign"] > 0 else (r - c + w)
                v = int((key + p["phase"]) % p["period"] < p["thickness"])
            elif family == "tiles":
                a = (r - p["phase_r"]) // p["bh"]
                b = (c - p["phase_c"]) // p["bw"]
                v = (a + b + p["offset"]) % p["colors"]
            elif family == "grid_lines":
                v = int((r + p["phase_r"]) % p["k"] < p["tr"]
                        or (c + p["phase_c"]) % p["k"] < p["tc"])
            elif family == "concentric":
                depth = min(r, c, h - 1 - r, w - 1 - c)
                v = int((depth // p["t"]) % 2 == p["parity"])
            elif family == "split":
                key = r if p["axis"] == 0 else c
                v = p["labels"][sum(1 for cut in cuts if cut <= key)]
            else:
                raise AssertionError(family)
            out[r][c] = v
    return out


def background_mask(grid, readings=None):
    """The background of the picture, as the prompt describes it."""
    h, w = len(grid), len(grid[0])
    readings = pf.fit(grid) if readings is None else readings
    assert len(readings) <= 1, "more than one admissible background pattern"
    if readings:
        reading = readings[0]
        layers = render_field(reading.family, reading.params, h, w)
        colors = list(reading.colors)
        field = [[colors[layers[r][c]] for c in range(w)] for r in range(h)]
        assert field == reading.array().tolist(), "parameters do not render the field"
        return [[grid[r][c] == field[r][c] for c in range(w)] for r in range(h)], field
    counts = Counter(v for row in grid for v in row)
    bg, n = counts.most_common(1)[0]
    assert n > .50 * h * w, "no plain field and no pattern"
    return [[grid[r][c] == bg for c in range(w)] for r in range(h)], [[bg] * w for _ in range(h)]


def oracle(grid, prompt, readings=None):
    conn = int(re.search(r'([48])-connected', prompt).group(1))
    lone_cells_count = "A single cell also counts as an object" in prompt
    background, _field = background_mask(grid, readings)
    comps = [c for c in groups(grid, conn, background) if lone_cells_count or len(c) > 1]
    if "marked rectangle" not in prompt:
        return len(comps)
    frames = []
    for cells in groups(grid, 4, background):
        rr, cc = zip(*cells)
        r0, r1, c0, c1 = min(rr), max(rr), min(cc), max(cc)
        if min(r1 - r0, c1 - c0) < 2:
            continue
        outline = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                   if r in (r0, r1) or c in (c0, c1)}
        if cells == outline:
            frames.append((r0, c0, r1, c1))
    answers = {sum(all(r0 < r < r1 and c0 < c < c1 for r, c in cells) for cells in comps)
               for r0, c0, r1, c1 in frames}
    assert len(answers) == 1, (frames, answers)
    return answers.pop()


@pytest.fixture(scope='module', params=['68a', '68b'])
def examples(request):
    return request.param, [generate(request.param, s) for s in range(SEEDS)]


def _patterned(examples):
    return [e for e in examples if e['metadata']['background_style'] == 'pattern']


def _plain(examples):
    return [e for e in examples if e['metadata']['background_style'] == 'plain']


# ---------------------------------------------------------------------------
# The answer
# ---------------------------------------------------------------------------

def test_independent_input_prompt_oracle(examples):
    tid, examples = examples
    for example in examples:
        assert oracle(example['input'], get_prompt(tid, example['metadata'])) == int(example['output'])


def test_all_four_plans_and_exact_scene(examples):
    tid, examples = examples
    for e in examples:
        m = e['metadata']
        for rule in ('count', 'ignore'):
            for conn in (4, 8):
                prompt = get_prompt(tid, {**m, 'connectivity': conn, 'singleton_rule': rule})
                key = f"planned_count_{'A' if rule == 'count' else 'B'}_{conn}"
                assert oracle(e['input'], prompt) == m[key]
                assert subject.solve(e['input'], rule, conn, framed=tid == '68b') == m[key]
        assert len(set(v for row in e['input'] for v in row)) == m['planned_palette_count']
        assert m['planned_count_A_4'] >= m['planned_count_B_4']
        assert m['planned_count_A_8'] >= m['planned_count_B_8']
        assert m['planned_count_A_4'] <= 24 and m['planned_count_A_8'] <= 24
        assert m['planned_count_A_4'] - m['planned_count_A_8'] == m['target_diagonal_pairs']
        assert all(5 <= n <= 30 for n in [len(e['input']), len(e['input'][0])])


def test_the_planned_scene_is_the_one_that_ships(monkeypatch):
    """No canned fallback: every attempt on a canvas draws the SAME plan.

    The generator used to answer an exhausted retry budget with an empty plain
    canvas, which reported a zero class it had not drawn and put a fifth of the
    zeros on the narrowest canvases.
    """
    plans = []
    original = subject._build

    def spy(rng, h, w, **kw):
        plans.append((kw['n_iso'], kw['zero_mode'], kw['families'] is not None))
        return original(rng, h, w, **kw)

    monkeypatch.setattr(subject, '_build', spy)
    for tid in ('68a', '68b'):
        for seed in range(400):
            plans.clear()
            generate(tid, seed)
            assert len({(n, z) for n, z, _p in plans}) == 1, (tid, seed, plans)


@pytest.mark.parametrize('tid', ['68a', '68b'])
def test_no_canvas_needs_a_fallback_and_zeros_do_not_follow_the_size(tid):
    """Every canvas carries its plan, and the zero answers are spread over the
    size buckets: the marker used to eat the narrow canvases, leaving a fifth
    of the five-cell grids with nothing to count against a twelfth overall."""
    answers, sides = [], []
    for seed in range(2000):
        e = generate(tid, seed)                       # a canvas that cannot carry
        answers.append(int(e['output']))              # its plan raises here
        sides.append(min(len(e['input']), len(e['input'][0])))
    # At 8000 seeds the buckets sit at .067 / .075 / .070 / .074 around .073
    # (68b) and .046 / .063 / .044 / .071 around .067 (68a, whose empty class
    # needs a canvas that can carry a three-class field).  The bound here has
    # room for the noise of a 130-grid bucket.
    overall = np.mean([a == 0 for a in answers])
    for lo, hi in ((5, 5), (6, 6), (7, 7), (8, 30)):
        bucket = [a for a, s in zip(answers, sides) if lo <= s <= hi]
        share = np.mean([a == 0 for a in bucket])
        assert abs(share - overall) <= .05, (lo, hi, share, overall, len(bucket))


def test_the_zero_classes_are_what_the_grid_shows(examples):
    tid, examples = examples
    for e in examples:
        m = e['metadata']
        counts = [m[k] for k in ('planned_count_A_4', 'planned_count_A_8',
                                 'planned_count_B_4', 'planned_count_B_8')]
        if m['zero_class'] == 'empty':
            assert max(counts) == 0
        elif m['zero_class'] == 'dots':
            assert m['planned_count_A_4'] > 0
            assert m['planned_count_B_4'] == 0 and m['planned_count_B_8'] == 0
        else:
            assert m['planned_count_B_4'] > 0
    # An empty grid is still a picture: a field of three classes carries it
    # where the canvas can hold one, and a flat canvas is the rare exception.
    flat = [e for e in examples if len(set(v for row in e['input'] for v in row)) == 1]
    assert len(flat) <= .015 * len(examples), len(flat)
    for e in examples:
        m = e['metadata']
        if m['zero_class'] == 'empty' and m['background_style'] == 'pattern':
            assert m['max_color_share'] <= pf.MAX_COLOR_SHARE


def test_the_scene_is_drawn_before_the_rule_and_the_connectivity(examples):
    tid, examples = examples
    for seed in range(30):
        grid, meta = subject.scene(np.random.default_rng(seed), framed=tid == '68b')
        e = generate(tid, seed)
        assert grid.tolist() == e['input']
        assert 'connectivity' not in meta and 'singleton_rule' not in meta


# ---------------------------------------------------------------------------
# Both instruction variants carry their weight, and neither is visible
# ---------------------------------------------------------------------------

def test_rule_is_load_bearing_and_balanced(examples):
    tid, examples = examples
    flips = [e for e in examples
             if e['metadata']['planned_count_A_4'] != e['metadata']['planned_count_B_4']
             or e['metadata']['planned_count_A_8'] != e['metadata']['planned_count_B_8']]
    assert .50 < len(flips) / len(examples) < .85
    subgroups = [examples, flips, [e for e in examples if e not in flips],
                 _patterned(examples), _plain(examples)]
    for lo, hi in ((0, 0), (1, 3), (4, 99)):
        subgroups.append([e for e in examples if lo <= e['metadata']['n_isolated_counted'] <= hi])
    for group in subgroups:
        if len(group) < 40:
            continue
        share = sum(e['metadata']['singleton_rule'] == 'count' for e in group) / len(group)
        assert .40 < share < .60, len(group)


def test_connectivity_is_load_bearing_and_balanced(examples):
    tid, examples = examples
    flips = [e for e in examples
             if e['metadata']['planned_count_A_4'] != e['metadata']['planned_count_A_8']
             or e['metadata']['planned_count_B_4'] != e['metadata']['planned_count_B_8']]
    assert .40 < len(flips) / len(examples) < .80
    for group in (examples, flips, [e for e in examples if e not in flips],
                  _patterned(examples), _plain(examples)):
        if len(group) < 40:
            continue
        share = sum(e['metadata']['connectivity'] == 4 for e in group) / len(group)
        assert .40 < share < .60, len(group)


def _features(example):
    m = example['metadata']
    g = np.array(example['input'])
    counts = np.bincount(g.ravel(), minlength=10)
    return [g.shape[0], g.shape[1], g.size, int((counts > 0).sum()),
            float(counts.max() / g.size), m['n_isolated_cells'], m['object_area_share'],
            float(m['background_style'] == 'pattern'),
            float(pf.FAMILIES.index(m['pattern_family']) if m['pattern_family'] else -1),
            m['naive_plurality_count'], m['target_primitives'], m['planned_palette_count']]


def _best_threshold_accuracy(rows, labels):
    """The strongest single-feature rule, fitted on half and scored on the rest."""
    rows, labels = np.array(rows, float), np.array(labels, float)
    half = len(rows) // 2
    best = 0.0
    for j in range(rows.shape[1]):
        column = rows[:half, j]
        for t in np.unique(column):
            for sign in (1, -1):
                train = np.mean((sign * (column - t) >= 0) == labels[:half])
                if train > best:
                    best = train
                    pick = (j, t, sign)
    column = rows[half:, pick[0]]
    return float(np.mean((pick[2] * (column - pick[1]) >= 0) == labels[half:]))


def test_no_input_only_predictor_of_the_rule_or_the_connectivity(examples):
    tid, examples = examples
    rows = [_features(e) for e in examples]
    for label in ('rule', 'conn'):
        if label == 'rule':
            labels = [e['metadata']['singleton_rule'] == 'count' for e in examples]
        else:
            labels = [e['metadata']['connectivity'] == 4 for e in examples]
        assert _best_threshold_accuracy(rows, labels) <= .58
    # The isolated-cell budget is drawn from ONE distribution: its histogram
    # must not shift with the rule.
    table = np.zeros((2, 3))
    for e in examples:
        n = e['metadata']['n_isolated_counted']
        table[int(e['metadata']['singleton_rule'] == 'count'), 0 if n == 0 else (1 if n < 4 else 2)] += 1
    expected = np.outer(table.sum(1), table.sum(0)) / table.sum()
    chi2 = float(((table - expected) ** 2 / np.maximum(expected, 1e-9)).sum())
    assert chi2 < 9.21, chi2          # two degrees of freedom, p = 0.01


# ---------------------------------------------------------------------------
# The background
# ---------------------------------------------------------------------------

def test_background_shares_and_families(examples):
    tid, examples = examples
    patterned = _patterned(examples)
    assert .18 < len(patterned) / len(examples) < .32
    families = Counter(e['metadata']['pattern_family'] for e in patterned)
    expected = set(pf.FAMILIES) - ({'concentric'} if tid == '68b' else set())
    assert set(families) == expected
    for family in expected:
        assert families[family] / len(patterned) >= .02, families
    iso = [e['metadata']['n_isolated_counted'] for e in examples]
    assert .35 < np.mean([n >= 1 for n in iso]) < .70
    assert .09 < np.mean([n >= 4 for n in iso]) < .30
    assert max(iso) <= subject.MAX_ISOLATED
    kinds = {e['metadata']['isolated_kind'] for e in examples if e['metadata']['n_isolated_counted'] >= 4}
    assert len(kinds & set(bg_kind for bg_kind in __import__('common.backgrounds',
                                                             fromlist=['x']).SPARSE_KINDS)) >= 3
    for e in patterned:
        assert e['metadata']['max_color_share'] <= pf.MAX_COLOR_SHARE
    for e in _plain(examples):
        assert e['metadata']['max_color_share'] > .50


def test_the_pattern_is_recoverable_and_unique(examples):
    tid, examples = examples
    for e in _patterned(examples):
        readings = pf.fit(e['input'])
        assert len(readings) == 1
        m = e['metadata']
        assert readings[0].family == m['pattern_family']
        # The reading is unique as a FIELD; an equivalent parameter tuple with
        # the classes relabelled is the same background, not a second reading.
        drawn = np.array(m['pattern_colors'], int)[
            pf.render(m['pattern_family'], m['pattern_params'], *np.array(e['input']).shape)]
        assert np.array_equal(readings[0].array(), drawn)
        assert sorted(readings[0].colors) == sorted(m['pattern_colors'])
    for e in _plain(examples):
        assert pf.fit(e['input']) == []


@pytest.mark.parametrize('tid', ['68a', '68b'])
def test_the_parameter_sweep_finds_the_same_single_reading(tid):
    """The derivation may only ever propose; this is the assumption-free search."""
    patterned = plain = 0
    for seed in range(SEEDS):
        e = generate(tid, seed)
        grid = e['input']
        style = e['metadata']['background_style']
        if style == 'pattern' and patterned < SWEEP_SEEDS:
            patterned += 1
        elif style == 'plain' and plain < SWEEP_SEEDS:
            plain += 1
        else:
            continue
        swept = pf.brute(grid)
        assert {r.field for r in swept} == {r.field for r in pf.fit(grid)}
        assert len(swept) == (1 if style == 'pattern' else 0)
    assert patterned == plain == SWEEP_SEEDS


def test_the_plurality_shortcut_never_reaches_the_answer_on_a_pattern(examples):
    tid, examples = examples
    patterned = _patterned(examples)
    for e in patterned:
        m = e['metadata']
        box = None if tid == '68a' else (m['frame_bounds'][0] + 1, m['frame_bounds'][1] + 1,
                                         m['frame_bounds'][2] - 1, m['frame_bounds'][3] - 1)
        naive = subject.naive_plurality_counts(np.array(e['input']), box=box)
        for key in subject.KEYS:
            assert naive[key] != m[subject.PLAN_KEY[key]]
    plain_right = sum(subject.naive_plurality_counts(np.array(e['input']))[('count', 4)]
                      == e['metadata']['planned_count_A_4'] for e in _plain(examples) if tid == '68a')
    if tid == '68a':
        assert plain_right == len(_plain(examples))     # on a plain field the shortcut is the rule


# ---------------------------------------------------------------------------
# Figures on a pattern, not part of it
# ---------------------------------------------------------------------------

def test_anti_blending_bounds(examples):
    tid, examples = examples
    for e in _patterned(examples):
        m = e['metadata']
        g = np.array(e['input'])
        layers = pf.render(m['pattern_family'], m['pattern_params'], *g.shape)
        field = np.array(m['pattern_colors'], int)[layers]
        mask = g != field
        assert not set(m['object_colors']) & set(m['pattern_colors'])            # AB1
        assert not set(m['isolated_colors']) & set(m['pattern_colors'])
        if tid == '68b':
            assert m['roi_color'] not in m['pattern_colors']
        assert np.array_equal(mask, ~np.isin(g, m['pattern_colors']))            # AB4
        dots = {(r, c) for r, c in m['isolated_cells']}
        marker = subject._outline_cells(m['frame_bounds']) if tid == '68b' else set()
        lab = pf.label4(layers)
        sizes = np.bincount(lab.ravel())
        assert sizes.min() >= 2                                                  # no lone field cell
        for _color, cells in subject.components(g, 4, mask=mask):
            if cells & marker or (len(cells) == 1 and next(iter(cells)) in dots):
                continue
            classes = {int(layers[q]) for q in cells}
            assert len(classes) >= 2                                             # AB2
            rr = [r for r, _ in cells]
            cc = [c for _, c in cells]
            assert max(rr) - min(rr) + 1 >= 2 and max(cc) - min(cc) + 1 >= 2     # AB3a
            assert len(cells) >= 3
            assert 0 < min(rr) and max(rr) < g.shape[0] - 1
            assert 0 < min(cc) and max(cc) < g.shape[1] - 1
            assert max(rr) - min(rr) + 1 <= subject.MAX_BBOX_EXTENT * g.shape[0]  # AB6
            assert max(cc) - min(cc) + 1 <= subject.MAX_BBOX_EXTENT * g.shape[1]
            region_min = int(sizes[[int(lab[q]) for q in cells]].min())
            if len(cells) > subject.SCALE_FRACTION * region_min:
                per = min(sum(1 for q in cells if layers[q] == k) for k in classes)
                assert per >= subject._straddle_need(len(cells))
            if len(cells) >= 6:
                assert subject._edge_alignment(cells, layers, g.shape) <= subject.EDGE_ALIGNMENT_MAX
        assert m['straddle_min_colors'] >= 2 or m['planned_count_A_4'] == m['n_isolated_counted']
        assert m['edge_alignment_max'] <= subject.EDGE_ALIGNMENT_MAX


def test_visibility_of_the_field(examples):
    tid, examples = examples
    for e in _patterned(examples):
        m = e['metadata']
        g = np.array(e['input'])
        layers = pf.render(m['pattern_family'], m['pattern_params'], *g.shape)
        mask = g != np.array(m['pattern_colors'], int)[layers]
        metrics = pf.visibility(m['pattern_family'], m['pattern_params'], layers, mask,
                                annotations=pf.outline_components(g) & mask)
        assert pf.visibility_ok(m['pattern_family'], metrics)                    # V1-V3
        assert min(metrics['visible_share']) >= pf.MIN_UNCOVERED
        assert metrics['covered_share'] <= subject.COVER_CAP[tid == '68b']       # V4
        shares = pf.class_shares(layers)
        assert pf.SHARE_LO <= shares.min() and shares.max() <= pf.SHARE_HI       # V5
        assert m['max_color_share'] <= pf.MAX_COLOR_SHARE                        # V6
        if tid == '68b':
            assert m['frame_sides_on_pattern_boundary'] <= 1
            assert subject.read(g, framed=True)['frame_candidates'] == 1


def test_the_shipped_visibility_thresholds(examples):
    """V2 region by region, as the module documents it.

    The numbers differ between families because the regions do: AB2 makes every
    body straddle a colour change, so a class with three stripes cannot spare
    two of them, a lattice line is crossed by every body that touches it, and
    concentric rings are nested (see ``pf.V2_DOC``).
    """
    tid, examples = examples
    seen = Counter()
    for e in _patterned(examples):
        m = e['metadata']
        g = np.array(e['input'])
        family = m['pattern_family']
        layers = pf.render(family, m['pattern_params'], *g.shape)
        mask = g != np.array(m['pattern_colors'], int)[layers]
        metrics = pf.visibility(family, m['pattern_params'], layers, mask,
                                annotations=pf.outline_components(g) & mask)
        untouched, total = metrics['untouched_regions'], metrics['region_counts']
        seen[family] += 1
        if family in ('bands', 'diagonal_bands'):
            for u, t in zip(untouched, total):
                assert u >= (2 if t >= 4 else 1), (family, u, t)
        elif family == 'tiles':
            for u, t in zip(untouched, total):
                assert u >= (6 if t >= 8 else 4), (u, t)
            assert min(metrics['untouched_tile_rows']) >= 2
            assert min(metrics['untouched_tile_cols']) >= 2
        elif family == 'grid_lines':
            assert untouched[0] >= 2
            assert min(metrics['clean_line_rows'], metrics['clean_line_cols']) >= 1
            assert metrics['clean_line_rows'] + metrics['clean_line_cols'] >= 3
        elif family == 'concentric':
            assert min(untouched) >= 1 and sum(untouched) >= 2
        else:
            assert min(metrics['panel_clean_lines']) >= 1
        assert list(m['pattern_regions_uncovered']) == list(untouched)
    assert len(seen) >= 5


def test_the_flush_edge_bound_holds_under_either_connectivity(examples):
    """AB5 under the reading the prompt states.

    Under 8-connectivity a diagonal pair of one colour is a single object, and
    its boundary runs further along the field's edges than either half does;
    the rule and the connectivity are drawn after the scene, so the bound is
    measured under both.
    """
    tid, examples = examples
    for e in _patterned(examples):
        m = e['metadata']
        g = np.array(e['input'])
        layers = pf.render(m['pattern_family'], m['pattern_params'], *g.shape)
        mask = g != np.array(m['pattern_colors'], int)[layers]
        marker = subject._outline_cells(m['frame_bounds']) if tid == '68b' else set()
        worst = 0.0
        for conn in (4, 8):
            for _color, cells in subject.components(g, conn, mask=mask):
                if (marker and cells & marker) or len(cells) < 6:
                    continue
                worst = max(worst, subject._edge_alignment(cells, layers, g.shape))
        assert worst <= subject.EDGE_ALIGNMENT_MAX, (worst, e['metadata']['pattern_family'])
        assert abs(worst - m['edge_alignment_max']) < 1e-3


def test_phase_and_parameter_spread(examples):
    tid, examples = examples
    patterned = _patterned(examples)
    seen = Counter()
    for e in patterned:
        p = e['metadata']['pattern_params']
        seen[(e['metadata']['pattern_family'], tuple(sorted(str(v) for v in p.values())))] += 1
    assert max(seen.values()) / len(patterned) < .20, seen.most_common(3)
    phases = {p['phase'] for p in (e['metadata']['pattern_params'] for e in patterned)
              if 'phase' in p}
    assert len(phases) >= 4


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def test_coverage_and_counts(examples):
    tid, examples = examples
    sides = {len(e['input']) for e in examples} | {len(e['input'][0]) for e in examples}
    assert sides == set(range(5, 31))
    answers = [int(e['output']) for e in examples]
    assert .035 < np.mean([a == 0 for a in answers]) < .12
    for key in ('planned_count_A_4', 'planned_count_A_8', 'planned_count_B_4', 'planned_count_B_8'):
        zeros = np.mean([e['metadata'][key] == 0 for e in examples])
        assert .035 < zeros < .12, (key, zeros)
    assert max(answers) >= 22
    assert np.mean([a >= 15 for a in answers]) > .01
    assert np.mean([a <= 8 for a in answers]) > .60
    assert np.mean([e['metadata']['planned_count_A_4'] <= 8 for e in examples]) > .60
    palette = {e['metadata']['planned_palette_count'] for e in examples}
    assert 10 in palette
    # 68b always draws a marker, so its smallest palette is the ground, the
    # marker and one object colour; an empty 68a canvas is one colour.
    assert min(palette) <= (1 if tid == '68a' else 3)
    assert len([e for e in examples if len(e['input']) * len(e['input'][0]) >= 700]) >= 20
    families = {f for e in examples for f in e['metadata']['target_families']}
    assert {'rectangle', 'bar', 'L', 'T', 'U', 'H', 'plus', 'staircase', 'singleton'} <= families
    assert ('frame' in families) == (tid == '68a')
    assert np.mean([e['metadata']['placement_method'] == 'free_placement' for e in examples]) > .80
    # Scene scale is measured on the drawn bodies: the marker is an annotation
    # of its own, and on a small canvas it owns most of the covered cells
    # (68b's covered share is .33 under 300 cells against .11 of bodies).
    def body_share(e):
        return e['metadata']['planned_object_cells'] / (len(e['input']) * len(e['input'][0]))
    big = [body_share(e) for e in examples if len(e['input']) * len(e['input'][0]) >= 700]
    small = [body_share(e) for e in examples if len(e['input']) * len(e['input'][0]) < 300]
    assert np.mean(big) > .60 * np.mean(small)        # scenes scale with the canvas


def test_singletons_and_shared_colors(examples):
    tid, examples = examples
    with_singletons = with_reused_color = 0
    for e in examples:
        m = e['metadata']
        g = e['input']
        background, _f = background_mask(g)
        cs = groups(g, m['connectivity'], background)
        if tid == '68b':
            r0, c0, r1, c1 = m['frame_bounds']
            cs = [c for c in cs if all(r0 < r < r1 - 1 and c0 < col < c1 - 1 for r, col in c)]
        with_singletons += any(len(c) == 1 for c in cs)
        cols = [g[next(iter(c))[0]][next(iter(c))[1]] for c in cs]
        with_reused_color += len(cols) > len(set(cols))
    assert with_singletons > .20 * len(examples)
    assert with_reused_color > .25 * len(examples)


def test_shuffled_letters(examples):
    tid, examples = examples
    for seed, e in enumerate(examples[:300]):
        palette = list('QWERTYUIOP')
        np.random.default_rng(seed + 731).shuffle(palette)
        g = [[palette[v] for v in row] for row in e['input']]
        p = get_prompt(tid, e['metadata'], color_palette=palette)
        assert 'same-color' in p and 'single cell' in p
        assert 'background is' not in p


def test_timing():
    for tid in ('68a', '68b'):
        times = []
        for seed in range(400):
            start = time.time()
            generate(tid, seed)
            times.append(time.time() - start)
        assert np.mean(times) < .030, (tid, np.mean(times))
        assert np.percentile(times, 95) < .150, (tid, np.percentile(times, 95))


# ---------------------------------------------------------------------------
# The prompt
# ---------------------------------------------------------------------------

BG = "Ignore the background, whether it is a solid color or a pattern."


def test_prompt_wordings():
    base = {'connectivity': 4, 'singleton_rule': 'count'}
    assert get_prompt('68a', base) == (
        f"{BG} How many 4-connected objects are in this grid? An object is a maximal group of "
        "same-color cells connected horizontally or vertically (cells that touch only at a corner "
        "are not connected). A single cell also counts as an object; cells of different colors are "
        "separate objects even when they touch. Answer with the number.")
    assert get_prompt('68a', {**base, 'connectivity': 8}) == (
        f"{BG} How many 8-connected objects are in this grid? An object is a maximal group of "
        "same-color cells connected horizontally, vertically, or diagonally. A single cell also "
        "counts as an object; cells of different colors are separate objects even when they touch. "
        "Answer with the number.")
    assert get_prompt('68a', {**base, 'singleton_rule': 'ignore'}) == (
        f"{BG} How many 4-connected objects are in this grid? An object is a maximal group of at "
        "least two same-color cells connected horizontally or vertically (cells that touch only at "
        "a corner are not connected); isolated single cells, whatever their color, are background "
        "texture and are ignored; cells of different colors are separate objects even when they "
        "touch. Answer with the number.")
    assert get_prompt('68a', {'connectivity': 8, 'singleton_rule': 'ignore'}) == (
        f"{BG} How many 8-connected objects are in this grid? An object is a maximal group of at "
        "least two same-color cells connected horizontally, vertically, or diagonally; isolated "
        "single cells, whatever their color, are background texture and are ignored; cells of "
        "different colors are separate objects even when they touch. Answer with the number.")
    for rule in ('count', 'ignore'):
        text = get_prompt('68b', {'connectivity': 4, 'singleton_rule': rule})
        assert text.startswith(f"{BG} How many 4-connected objects are entirely inside the "
                               "marked rectangle?")
        assert text.endswith("Do not count the rectangle border. Count an object only if all its "
                             "cells are inside the border; it may be adjacent to the border. "
                             "Answer with the number.")


def test_the_prompt_has_no_default_rule():
    """Neither variant is a default: metadata that does not say which rule and
    which connectivity an example was drawn under has no prompt at all."""
    for tid in ('68a', '68b'):
        for meta in ({}, {'connectivity': 4}, {'connectivity': 4, 'singleton_rule': None},
                     {'connectivity': 4, 'singleton_rule': 'ignore_singletons'},
                     {'connectivity': 4, 'singleton_rule': ''},
                     {'singleton_rule': 'count'}, {'connectivity': 6, 'singleton_rule': 'count'}):
            with pytest.raises(ValueError):
                get_prompt(tid, dict(meta))
        for rule in ('count', 'ignore'):
            for conn in (4, 8):
                assert get_prompt(tid, {'connectivity': conn, 'singleton_rule': rule})


@pytest.mark.parametrize('tid', ['68a', '68b'])
def test_the_background_sentence_does_not_move_with_the_background(tid):
    """Byte-identical on plain and patterned grids: nothing is named, nothing leaks."""
    seen = {}
    for seed in range(400):
        e = generate(tid, seed)
        m = e['metadata']
        prompt = get_prompt(tid, m)
        assert prompt.startswith(BG)
        seen.setdefault((m['singleton_rule'], m['connectivity'], m['background_style']), prompt)
    for rule in ('count', 'ignore'):
        for conn in (4, 8):
            texts = {seen[key] for key in seen if key[0] == rule and key[1] == conn}
            assert len(texts) == 1, (rule, conn)
    assert len(seen) >= 8
    for prompt in seen.values():
        assert 'pattern_family' not in prompt and 'background is' not in prompt
        for word in ('band', 'stripe', 'tile', 'ring', 'lattice', 'panel', 'checker'):
            assert word not in prompt


# ---------------------------------------------------------------------------
# The validator raises; it never falls back
# ---------------------------------------------------------------------------

def _first(tid, style):
    for seed in range(400):
        e = generate(tid, seed)
        if e['metadata']['background_style'] == style:
            return e
    raise AssertionError(style)


@pytest.mark.parametrize('tid', ['68a', '68b'])
def test_validator_rejects_six_tampers(tid):
    framed = tid == '68b'

    e = _first(tid, 'plain')
    e['output'] = str(int(e['output']) + 1)
    with pytest.raises(ValueError, match='Wrong visible'):
        subject.validate(e, framed=framed)

    e = _first(tid, 'plain')
    e['metadata']['planned_count_A_4'] += 1
    with pytest.raises(ValueError, match='fixed scene plan'):
        subject.validate(e, framed=framed)

    e = _first(tid, 'pattern')
    e['metadata']['singleton_rule'] = 'count' if e['metadata']['singleton_rule'] == 'ignore' else 'ignore'
    if e['metadata']['planned_count_A_4'] != e['metadata']['planned_count_B_4']:
        with pytest.raises(ValueError, match='Wrong visible'):
            subject.validate(e, framed=framed)

    e = _first(tid, 'pattern')
    m = e['metadata']
    g = np.array(e['input'])
    layers = pf.render(m['pattern_family'], m['pattern_params'], *g.shape)
    field = np.array(m['pattern_colors'], int)[layers]
    spot = np.argwhere(g == field)[0]
    g[spot[0], spot[1]] = [c for c in range(10) if c not in m['pattern_colors']][0]
    e['input'] = g.tolist()
    with pytest.raises(ValueError):
        subject.validate(e, framed=framed)

    e = _first(tid, 'pattern')
    e['metadata']['pattern_params'] = dict(e['metadata']['pattern_params'])
    key = 'phase' if 'phase' in e['metadata']['pattern_params'] else \
        next(iter(e['metadata']['pattern_params']))
    e['metadata']['background_style'] = 'plain'
    with pytest.raises(ValueError, match='recovered background'):
        subject.validate(e, framed=framed)

    e = next(generate(tid, seed) for seed in range(400)
             if generate(tid, seed)['metadata']['background_style'] == 'pattern'
             and generate(tid, seed)['metadata']['straddle_min_colors'] >= 2)
    e['metadata']['straddle_min_colors'] = 9
    with pytest.raises(ValueError, match='straddle'):
        subject.validate(e, framed=framed)


@pytest.mark.parametrize('tid', ['68a', '68b'])
def test_validator_rejects_an_added_visible_object(tid):
    for seed in range(40):
        e = generate(tid, seed)
        g = np.array(e['input'])
        m = e['metadata']
        if m['background_style'] != 'plain':
            continue
        h, w = g.shape
        r0, c0, r1, c1 = m['frame_bounds'] if tid == '68b' else (0, 0, h, w)
        for r in range(r0 + 1, r1 - 1):
            for c in range(c0 + 1, c1 - 1):
                if np.all(g[max(0, r-1):r+2, max(0, c-1):c+2] == m['background']):
                    g[r, c] = next(v for v in range(10)
                                   if v != m['background'] and v != m['roi_color'])
                    e['input'] = g.tolist()
                    with pytest.raises(ValueError):
                        subject.validate(e, framed=tid == '68b')
                    return
    pytest.fail('No seed with a clear place for an added object')


# ---------------------------------------------------------------------------
# Hand-built readings
# ---------------------------------------------------------------------------

def _stripes(h, w, period, thickness, colors, axis=0):
    a = np.zeros((h, w), int)
    for r in range(h):
        for c in range(w):
            key = r if axis == 0 else c
            a[r, c] = colors[1] if key % period < thickness else colors[0]
    return a


def test_a_single_stripe_is_not_a_pattern():
    a = np.zeros((16, 16), int)
    a[6:10, :] = 1
    assert pf.fit(a) == []


def test_a_one_by_one_checkerboard_is_inadmissible():
    rr, cc = np.indices((20, 20))
    a = ((rr + cc) % 2).astype(int)
    assert pf.fit(a) == []
    assert not pf.family_ok('tiles', {'bh': 1, 'bw': 1, 'colors': 2, 'phase_r': 0,
                                      'phase_c': 0, 'offset': 0}, 20, 20)


def test_a_period_two_diagonal_is_inadmissible():
    assert not pf.family_ok('diagonal_bands', {'sign': 1, 'period': 2, 'thickness': 1,
                                               'phase': 0}, 20, 20)


def test_two_fields_that_both_fit_are_refused():
    """A grid whose covered cells hide the difference between two readings."""
    a = _stripes(20, 20, 4, 2, [1, 2])
    b = _stripes(20, 20, 4, 2, [1, 2])
    b[:, :] = np.roll(b, 1, axis=0)
    different = np.argwhere(a != b)
    a[different[:, 0], different[:, 1]] = 3       # paint over every difference
    readings = pf.fit(a)
    assert len(readings) != 1 or readings[0].family in pf.FAMILIES
    if len(readings) > 1:
        with pytest.raises(ValueError, match='More than one'):
            subject.read_field(a)


def test_an_object_filling_one_tile_is_refused():
    rr, cc = np.indices((20, 20))
    a = (((rr // 4) + (cc // 4)) % 2).astype(int) + 1
    a[4:8, 4:8] = 5                                # exactly one tile
    assert not subject._object_ok(
        {'family': 'tiles', 'params': {'bh': 4, 'bw': 4, 'colors': 2, 'phase_r': 0,
                                       'phase_c': 0, 'offset': 0},
         'layers': (((rr // 4) + (cc // 4)) % 2).astype(np.int8),
         'lab': pf.label4((((rr // 4) + (cc // 4)) % 2).astype(np.int8)),
         'sizes': np.bincount(pf.label4((((rr // 4) + (cc // 4)) % 2).astype(np.int8)).ravel()),
         'depth': np.minimum(np.minimum(rr, cc), np.minimum(19 - rr, 19 - cc))},
        {'labels': np.ones((4, 4), int), 'families': ['rectangle']}, (4, 4), 20, 20)


def test_an_object_inside_one_stripe_is_refused():
    layers = (np.indices((20, 20))[0] % 6 < 3).astype(np.int8)
    lab = pf.label4(layers)
    pattern = {'family': 'bands', 'params': {'axis': 0, 'period': 6, 'thickness': 3, 'phase': 0},
               'layers': layers, 'lab': lab, 'sizes': np.bincount(lab.ravel()), 'depth': None}
    part = {'labels': np.ones((2, 2), int), 'families': ['rectangle']}
    assert not subject._object_ok(pattern, part, (0, 5), 20, 20)     # inside one stripe
    assert subject._object_ok(pattern, part, (2, 5), 20, 20)         # across the seam


def test_a_frame_on_two_pattern_boundaries_is_refused():
    layers = (np.indices((24, 24))[0] % 6 < 3).astype(np.int8)
    assert subject._frame_sides_on_boundary(layers, (1, 2, 23, 21)) == 0
    assert subject._frame_sides_on_boundary(layers, (2, 2, 21, 21)) == 2


def test_wall_adjacent_singletons_count_and_outside_objects_do_not():
    a = np.zeros((12, 14), int)
    a[2:10, [3, 11]] = 6
    a[[2, 9], 3:12] = 6
    a[3, 4] = 1
    a[4, 5] = 1
    a[8, 10] = 2
    a[0, 1] = 1
    assert subject.solve(a, 'count', 4, framed=True) == 3
    assert subject.solve(a, 'count', 8, framed=True) == 2
    assert subject.solve(a, 'ignore', 4, framed=True) == 0
    assert subject.solve(a, 'ignore', 8, framed=True) == 1


def test_two_marked_rectangle_readings_with_different_answers_raise():
    a = np.zeros((15, 25), int)
    for box, color in (((2, 2, 12, 11), 1), ((2, 14, 12, 23), 2)):
        r0, c0, r1, c1 = box
        a[r0:r1, [c0, c1 - 1]] = color
        a[[r0, r1 - 1], c0:c1] = color
    a[4, 4] = 3
    a[4, 16] = 4
    a[7, 18] = 4
    with pytest.raises(ValueError, match='uniquely determined'):
        subject.read(a, framed=True)


def test_background_band_counterreading_is_rejected():
    a = np.zeros((9, 9), int)
    a[3:6, :] = 1
    with pytest.raises(ValueError):
        subject.read(a)


def test_empty_frame_and_smallest_canvas():
    a = np.zeros((5, 5), int)
    a[1:4, [1, 3]] = 8
    a[[1, 3], 1:4] = 8
    assert subject.solve(a, 'count', 4, framed=True) == 0
    a[2, 2] = 2
    assert subject.solve(a, 'count', 8, framed=True) == 1
    assert subject.solve(a, 'ignore', 8, framed=True) == 0


@pytest.mark.parametrize('seed', [1044, 1153, 2200, 2270, 3143, 3443, 3807, 4358, 4713, 4868])
def test_narrow_canvas_reserves_frame_and_field_before_count(seed):
    e = generate('68b', seed)
    assert oracle(e['input'], get_prompt('68b', e['metadata'])) == int(e['output'])
