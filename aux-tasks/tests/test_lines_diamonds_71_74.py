"""Independent prompt/input readers and capacity checks for tasks 71–74."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import re

import numpy as np
import pytest

from prompts import get_prompt
from registry import generate
from generators import diamond_geometry_71_72 as diamonds
from generators import parallel_geometry_73_74 as parallel


def _symbol(text):
    return int(text) if text.isdigit() else text


def read_71(grid, prompt):
    color = _symbol(re.search(r"diamond in color (\S+)", prompt)[1])
    marker = _symbol(re.search(r"EACH cell of color (\S+);", prompt)[1])
    radius = int(re.search(r"span of (\d+) cells", prompt)[1]) // 2
    a = np.asarray(grid)
    rr, cc = np.indices(a.shape)
    where = np.zeros(a.shape, bool)
    for r, c in np.argwhere(a == marker):
        where |= abs(rr - r) + abs(cc - c) <= radius
    out = a.copy()
    out[where] = color
    return out.tolist()


def read_72(grid, prompt):
    assert "one-cell-thick outline" in prompt and "counting both corner cells" in prompt
    a = np.asarray(grid)
    h, w = a.shape
    # Brute distance table over the canvas plus an exterior rim. This differs
    # from the generator's extrema, distance transforms and perimeter counting.
    centers = np.array([(r, c) for r in range(-1, h + 1) for c in range(-1, w + 1)])
    exterior = (centers[:, 0] < 0) | (centers[:, 0] >= h) | (centers[:, 1] < 0) | (centers[:, 1] >= w)
    points = np.array([(r, c) for r in range(h) for c in range(w)])
    distances = np.abs(centers[:, None, :] - points[None, :, :]).sum(axis=2)
    lengths = set()
    for color in np.unique(a):
        mask = a.ravel() == color
        if mask.all():
            raise ValueError("A monochrome canvas has no unique diamond")
        smallest = distances[:, mask].max(axis=1)
        largest = distances[:, ~mask].min(axis=1) - 1
        for i in np.flatnonzero(np.maximum(smallest, 1) <= largest):
            lengths.update(range(2 * max(1, int(smallest[i])) + 1, 2 * int(largest[i]) + 2, 2))
            if exterior[i]:
                # Pushing an exterior center another cell outward and adding
                # one to the radius gives another span with the identical mask.
                lengths.add(2 * int(largest[i]) + 3)
        ring = (smallest >= 1) & (distances[:, mask].min(axis=1) == smallest)
        ring &= (distances == smallest[:, None]).sum(axis=1) == int(mask.sum())
        lengths.update(2 * int(smallest[i]) + 1 for i in np.flatnonzero(ring))
        lengths.update(2 * int(smallest[i]) + 3 for i in np.flatnonzero(ring & exterior))
    assert len(lengths) == 1, lengths
    return str(next(iter(lengths)))


def _observed_lines(grid, full=False):
    a = np.asarray(grid)
    rr, cc = np.indices(a.shape)
    singles, doubles = [], []
    for color in np.unique(a):
        mask = a == color
        for orientation, projection, along in [('h', rr, cc), ('v', cc, rr), ('diag_down', rr - cc, rr), ('diag_up', rr + cc, rr)]:
            keys = np.unique(projection[mask])
            if len(keys) not in (1, 2):
                continue
            valid = True
            for key in keys:
                on = mask & (projection == key)
                coordinates = np.sort(along[on])
                if len(coordinates) < 4 or not np.array_equal(coordinates, np.arange(coordinates[0], coordinates[0] + len(coordinates))):
                    valid = False
                    break
                if full and not np.array_equal(on, projection == key):
                    valid = False
                    break
            if valid:
                item = (orientation, tuple(map(int, keys)))
                (singles if len(keys) == 1 else doubles).append(item)
    return singles, doubles


def read_73(grid, prompt):
    assert "supporting rows, columns or diagonal bands" in prompt
    singles, doubles = _observed_lines(grid)
    gaps = {keys[1] - keys[0] - 1 for _, keys in doubles if keys[1] - keys[0] > 1}
    for i, (orientation, (key,)) in enumerate(singles):
        for other_orientation, (other_key,) in singles[i + 1:]:
            if orientation == other_orientation and abs(other_key - key) > 1:
                gaps.add(abs(other_key - key) - 1)
    assert len(gaps) == 1, gaps
    return str(next(iter(gaps)))


def read_74(grid, prompt):
    new = _symbol(re.search(r"new color (\S+) line", prompt)[1])
    gap = int(re.search(r"with a gap of (\d+)", prompt)[1])
    direction = re.search(r"with a gap of \d+ (.+?) the existing line\.", prompt)[1]
    singles, _ = _observed_lines(grid, full=True)
    assert len(singles) == 1, singles
    orientation, (key,) = singles[0]
    positive = direction in ('below', 'to the right of', 'down-left of', 'down-right of')
    key += (gap + 1) * (1 if positive else -1)
    a = np.asarray(grid).copy()
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            value = r if orientation == 'h' else c if orientation == 'v' else r - c if orientation == 'diag_down' else r + c
            if value == key:
                a[r, c] = new
    return a.tolist()


READERS = {'71': read_71, '72': read_72, '73': read_73, '74': read_74}


@pytest.fixture(scope='module')
def examples():
    return {tid: [generate(tid, seed) for seed in range(400)] for tid in READERS}


@pytest.mark.parametrize('tid', READERS)
def test_literal_visible_reader_400(examples, tid):
    for seed, ex in enumerate(examples[tid]):
        assert READERS[tid](ex['input'], get_prompt(tid, ex['metadata'])) == ex['output'], (tid, seed)
        json.dumps(ex)


@pytest.mark.parametrize('tid', READERS)
def test_shuffled_symbols_and_determinism(examples, tid):
    for seed, ex in enumerate(examples[tid][:30]):
        assert generate(tid, seed) == ex
        palette = list('QWERTYUIOP')
        np.random.default_rng(seed + 8800).shuffle(palette)
        inp = [[palette[x] for x in row] for row in ex['input']]
        expected = [[palette[x] for x in row] for row in ex['output']] if isinstance(ex['output'], list) else ex['output']
        prompt = get_prompt(tid, ex['metadata'], color_palette=palette)
        assert not re.search(r'\bcolor\s+\d\b', prompt)
        assert READERS[tid](inp, prompt) == expected


@pytest.mark.parametrize('tid', READERS)
def test_size_palette_and_field_coverage(examples, tid):
    exs = examples[tid]
    sizes = [(len(ex['input']), len(ex['input'][0])) for ex in exs]
    assert {h for h, _ in sizes} == set(range(5, 31))
    assert {w for _, w in sizes} == set(range(5, 31))
    assert sum(h * w >= 700 for h, w in sizes) >= 8
    assert abs(np.corrcoef(np.array(sizes).T)[0, 1]) < .15
    palettes = Counter(ex['metadata']['n_colors'] for ex in exs)
    assert min(palettes) == 2 and max(palettes) >= 5
    textured = sum(ex['metadata']['texture_kind'] != 'solid' for ex in exs)
    assert (100 if tid == '71' else 55) < textured < (260 if tid == '71' else 155)
    for ex in exs:
        m = ex['metadata']
        assert m['n_colors'] == len(set(x for row in ex['input'] for x in row))
        assert m['n_colors'] == m['planned_n_colors']
        assert set(m['texture_colors']).issubset(set(x for row in ex['input'] for x in row))
        if m['texture_kind'] == 'panels':
            for color in m['texture_colors']:
                rows, cols = np.nonzero(np.asarray(ex['input']) == color)
                height, width = int(np.ptp(rows)) + 1, int(np.ptp(cols)) + 1
                assert height >= 2 and width >= 2
                assert height * width == len(rows)
        assert not m['texture_named']
        assert 'background is' not in get_prompt(tid, m).lower()


def test_diamond_drawing_variety_and_preservation(examples):
    exs = examples['71']
    assert 130 < sum(not e['metadata']['is_clipped_example'] for e in exs) < 230
    assert 30 < sum(e['metadata']['overlap_cells'] > 0 for e in exs) < 100
    assert sum(e['metadata']['n_diamonds'] >= 5 for e in exs) >= 25
    assert max(e['metadata']['n_diamonds'] for e in exs) == 10
    assert max(e['metadata']['length'] for e in exs) >= 29
    assert sum(e['metadata']['length'] >= 19 for e in exs) >= 40
    for ex in exs:
        m = ex['metadata'];a = np.asarray(ex['input']);out = np.asarray(ex['output'])
        assert int((a == m['marker_color']).sum()) == m['requested_count'] == m['n_diamonds']
        assert m['op_overlap'] == (m['overlap_cells'] > 0)
        assert m['op_clipped'] == m['is_clipped_example']
        assert m['background_outside_preserved']
        assert not np.array_equal(a, out)


def test_diamond_inference_is_meaningful_minority(examples):
    exs = examples['72']
    assert 70 < sum(e['metadata']['requires_span_inference'] for e in exs) < 130
    assert 60 < sum(e['metadata']['family'] == 'outline' for e in exs) < 135
    assert max(int(e['output']) for e in exs) >= 29
    assert sum(int(e['output']) >= 19 for e in exs) >= 40
    for ex in exs:
        m = ex['metadata']
        assert m['requires_span_inference'] == (max(m['bbox_h'], m['bbox_w']) < int(ex['output']))
        assert m['requires_span_inference'] == (m['clip_mode'] == 'both_axes')
        assert m['unique_full_span'] and m['off_grid_alternatives'] == 0


def test_parallel_variety_positive_gaps_and_legibility(examples):
    for tid in ('73', '74'):
        exs = examples[tid]
        assert {e['metadata']['orientation'] for e in exs} == set(parallel.ORIENTATIONS)
        assert max(e['metadata']['distance'] for e in exs) >= 30
        assert sum(e['metadata']['distance'] > 10 for e in exs) >= 65
        for ex in exs:
            assert ex['metadata']['distance'] >= 1
            assert min(ex['metadata']['line_lengths']) >= 4
    assert 125 < sum(e['metadata']['family'] == 'segments' for e in examples['73']) < 235
    assert 80 < sum(e['metadata']['same_color'] for e in examples['73']) < 160
    assert 30 < sum(e['metadata']['endpoint_separation_matters'] for e in examples['73']) < 95
    for ex in examples['73']:
        if ex['metadata']['op_endpoint_separation']:
            assert ex['metadata']['endpoint_separation_matters']
    both = [e for e in examples['74'] if e['metadata']['two_way_feasible']]
    assert 175 <= len(both) <= 270
    center_guesses = []
    for ex in both:
        m=ex['metadata'];h,w=len(ex['input']),len(ex['input'][0]);key=m['old_band'];new=m['new_band']
        middle=(h-1)/2 if m['orientation']=='h' else (w-1)/2 if m['orientation']=='v' else (h-w)/2 if m['orientation']=='diag_down' else (h+w-2)/2
        center_guesses.append((new > key) == (key <= middle))
        reverse = next(name for name, sign in parallel.SIGNS[m['orientation']].items() if sign == (-1 if new > key else 1))
        alternative = parallel.solve_draw_parallel(ex['input'],m['new_color'],m['distance'],reverse)
        assert alternative != ex['output']
    assert .37 < np.mean(center_guesses) < .63


def test_tampered_answers_raise(examples):
    ex=examples['71'][0];m=ex['metadata'];bad=np.array(ex['output']);bad[0,0]=(int(bad[0,0])+1)%10
    with pytest.raises(ValueError):diamonds.validate_draw_diamonds(ex['input'],bad.tolist(),m['marker_color'],m['diamond_color'],m['length'])
    ex=examples['72'][0]
    with pytest.raises(ValueError):diamonds.validate_diamond_length(ex['input'],int(ex['output'])+2)
    ex=examples['73'][0]
    with pytest.raises(ValueError):parallel.validate_parallel_gap(ex['input'],int(ex['output'])+1)
    ex=examples['74'][0];m=ex['metadata'];bad=np.array(ex['output']);bad[0,0]=(int(bad[0,0])+1)%10
    with pytest.raises(ValueError):parallel.validate_draw_parallel(ex['input'],bad.tolist(),m['new_color'],m['distance'],m['direction'])


def test_off_grid_diamond_alternatives_rejected_and_hard_case_accepted():
    rr,cc=np.indices((7,7))
    for style in ('filled','outline'):
        a=np.zeros((7,7),int);d=abs(rr)+abs(cc-3);a[d<=3 if style=='filled' else d==3]=7
        with pytest.raises(ValueError):diamonds.solve_diamond_length(a)
    rr,cc=np.indices((18,20));a=np.zeros((18,20),int);a[abs(rr-3)+abs(cc-4)<=6]=7
    assert diamonds.solve_diamond_length(a)=='13'
    # The colored boundary singleton is a radius-zero clamped witness of a
    # radius-one diamond centered just outside the canvas, despite radius zero
    # being excluded from the generated diamond family itself.
    rr,cc=np.indices((5,5));a=np.zeros((5,5),int);a[abs(rr-2)+abs(cc-2)<=2]=7;a[0,1]=3
    with pytest.raises(ValueError):diamonds.solve_diamond_length(a)
    a=np.zeros((5,5),int);a[abs(rr-2)+abs(cc-2)<=1]=7;a[0,1]=3
    with pytest.raises(AssertionError):read_72(a.tolist(),get_prompt('72'))


def test_every_size_has_all_diamond_clipping_modes():
    for h in range(5,31):
        for w in range(5,31):
            for mode in ('complete','one_axis','both_axes'):
                assert diamonds._length_plans(h,w,mode), (h,w,mode)


def test_competing_parallel_lines_rejected():
    a=np.zeros((12,12),int);a[1,:]=1;a[5,:]=2;a[10,:]=3
    with pytest.raises(ValueError):parallel.solve_parallel_gap(a)
    with pytest.raises(ValueError):parallel.solve_draw_parallel(a,4,2,'below')


def test_sibling_generators_keep_baseline_hashes():
    expected = SIBLING_HASHES
    for tid, digest in expected.items():
        raw=json.dumps([generate(tid,s) for s in range(20)],sort_keys=True,separators=(',',':')).encode()
        assert hashlib.sha256(raw).hexdigest()==digest


SIBLING_HASHES = {'15': 'afd58e11a70fac9cc99b29b310b9e09ac7d44a7d92213701119c655844ed450e', '15a': '8d421ece1f476799074ebe8d39e4c5f53033aff7e91843eb7830bb0b4e025ce2', '16': '368c08b3ba58baad05f11673fdb7482608ebfebc68dd0211a561e8c73d5633a1'}
