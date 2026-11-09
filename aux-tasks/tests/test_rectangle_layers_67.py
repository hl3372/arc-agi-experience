"""Independent scene/order checks and diversity guards for task 67."""
from collections import Counter
from functools import lru_cache
import itertools
import re

import numpy as np
import pytest

from generators.rectangle_layers_67 import InvalidScene, read_scene, validate
from prompts import get_prompt
from registry import generate


def oracle(grid):
    """Enumerate each group's possible orders, keeping distinct reverse images."""
    g = np.array(grid)
    colors, counts = np.unique(g, return_counts=True)
    bg = colors[counts.argmax()]
    colors = [c for c in colors if c != bg]
    masks = []
    for color in colors:
        rr, cc = np.where(g == color)
        mask = np.zeros(g.shape, bool)
        mask[rr.min():rr.max()+1, cc.min():cc.max()+1] = True
        masks.append(mask)
    unseen = set(range(len(colors)))
    answer = np.full(g.shape, bg, dtype=g.dtype)
    while unseen:
        group = {unseen.pop()}
        todo = list(group)
        while todo:
            i = todo.pop()
            for j in list(unseen):
                if (masks[i] & masks[j]).any():
                    unseen.remove(j); group.add(j); todo.append(j)
        if len(group) == 1:
            i = next(iter(group)); answer[masks[i]] = colors[i]
            continue
        region = np.any([masks[i] for i in group], axis=0)
        outputs = {}
        for order in itertools.permutations(group):
            drawn = np.full(g.shape, bg, dtype=g.dtype)
            for i in order: drawn[masks[i]] = colors[i]
            if not np.array_equal(drawn[region], g[region]): continue
            rev = np.full(g.shape, bg, dtype=g.dtype)
            for i in order[::-1]: rev[masks[i]] = colors[i]
            outputs[rev.tobytes()] = rev
        assert len(outputs) == 1
        answer[region] = next(iter(outputs.values()))[region]
    return answer.tolist()


@lru_cache(None)
def examples():
    return [generate('67', seed) for seed in range(1200)]


def test_independent_order_enumeration_and_metadata():
    for ex in examples():
        assert oracle(ex['input']) == ex['output']
        g = np.array(ex['input']); m = ex['metadata']
        reading = validate(ex)
        assert m['n_colors'] == len(np.unique(g)) == m['n_rectangles']+1
        assert sorted(m['planned_group_sizes']) == m['overlap_group_sizes']
        assert sorted(len(x) for x in reading.groups if len(x)>1) == m['overlap_group_sizes']
        assert (g != m['background']).mean() == m['occupancy']
        assert all(len(r['box']) == 4 for r in m['rectangles'])


def test_sizes_and_scene_scaling():
    exs = examples(); sizes = np.array([np.array(e['input']).shape for e in exs])
    assert sizes.min() == 5 and sizes.max() == 30
    assert len(set(map(tuple, sizes))) > 520
    assert abs(np.corrcoef(sizes.T)[0,1]) < .12
    assert np.mean(sizes[:,0] == sizes[:,1]) < .07
    for axis in range(2):
        counts = Counter(sizes[:,axis]); assert len(counts) == 26
        assert min(counts.values()) >= .35*max(counts.values())
    large = [e['metadata'] for e in exs if np.array(e['input']).size >= 700]
    assert len(large) >= 25
    assert np.mean([m['occupancy'] for m in large]) > .23
    assert max(m['max_rectangle_side'] for m in large) >= 23
    assert max(m['max_rectangle_area'] for m in large) >= 300


def test_layouts_layers_palette_and_isolates():
    ms = [e['metadata'] for e in examples()]
    n = len(ms)
    families = Counter(f for m in ms for f in m['planned_families'])
    assert set(families) == {'offset','cross','nested','chain','stack'}
    assert min(families.values()) > .09*n
    assert .14 < np.mean([m['non_common_core_groups'] > 0 for m in ms]) < .45
    assert .12 < np.mean([m['n_overlap_groups'] > 1 for m in ms]) < .4
    assert .1 < np.mean([m['max_overlap_group_size'] == 4 for m in ms]) < .3
    assert .25 < np.mean([m['n_extra'] == 0 for m in ms]) < .55
    assert .15 < np.mean([m['edge_contact'] for m in ms]) < .4
    assert set(m['n_colors'] for m in ms) == set(range(3,11))
    assert .08 < np.mean([m['n_colors'] >= 8 for m in ms]) < .3
    assert set(m['background'] for m in ms) == set(range(10))


def test_visible_area_cannot_replace_layer_inference():
    hits = 0
    for ex in examples():
        g = np.array(ex['input']); colors, counts = np.unique(g, return_counts=True)
        bg = colors[counts.argmax()]
        out = np.full(g.shape,bg)
        for color in sorted((c for c in colors if c != bg),key=lambda c:int((g==c).sum()),reverse=True):
            rr,cc = np.where(g==color)
            out[rr.min():rr.max()+1,cc.min():cc.max()+1] = color
        hits += int(np.array_equal(out, ex['output']))
    assert hits/len(examples()) < .62  # old distribution: .7375 over 10k


def test_prompt_and_shuffled_symbols():
    for seed, ex in enumerate(examples()[:300]):
        palette = np.random.default_rng(seed+874).permutation(list('QWERTYUIOP'))
        grid = [[str(palette[x]) for x in row] for row in ex['input']]
        expected = [[str(palette[x]) for x in row] for row in ex['output']]
        assert oracle(grid) == expected
        assert read_scene(grid).output.tolist() == expected
        prompt = get_prompt('67',ex['metadata'],color_palette=palette.tolist())
        assert 'Each rectangle has its own color.' in prompt
        assert 'background' not in prompt and not re.search(r'color \d',prompt)


def test_validator_rejects_changed_answer():
    ex = generate('67',19)
    ex['output'][0][0] = (ex['output'][0][0]+1)%10
    with pytest.raises(InvalidScene,match='output disagrees'):
        validate(ex)


def test_reject_hidden_extension():
    g=np.zeros((7,9),int)
    g[1:5,1:6]=1; g[0:6,4:8]=2
    with pytest.raises(InvalidScene): read_scene(g)


def test_reject_order_hidden_by_third_rectangle():
    g=np.zeros((8,8),int)
    g[1:5,1:4]=1; g[3:6,3:6]=2; g[2:5,2:5]=3
    other=np.zeros_like(g)
    other[3:6,3:6]=2; other[1:5,1:4]=1; other[2:5,2:5]=3
    assert np.array_equal(g,other)
    with pytest.raises(InvalidScene): read_scene(g)


@pytest.mark.parametrize('seed',[222,585,10001,10007,15255,20418])
def test_constrained_scene_plans_remain_feasible(seed):
    ex=generate('67',seed)
    assert oracle(ex['input']) == ex['output']
    assert validate(ex).background == ex['metadata']['background']
