"""The ARC transform library: size, declared properties checked against the functions, no
synonyms, one phrase per instance, and ``rivals.solve`` exact against brute force."""
import collections
import re

import numpy as np
import pytest

from _arc_battery import battery
from pv2.arc import gen
from pv2.arc.grids import Info, fits, neighbour_count_same
from pv2.arc.rivals import solve
from pv2.arc.transforms import FAMILIES, Rule, instance_count

INSTANCES = [(n, p) for n, f in FAMILIES.items() for p in f.instances]


def test_library_size():
    assert len(FAMILIES) >= 50
    assert instance_count() >= 1500
    for g in gen.GEN:
        assert g.fam in FAMILIES


def _results():
    out = {}
    for g_i, g in enumerate(battery()):
        info = Info(g)
        for name, p in INSTANCES:
            out[(name, p, g_i)] = FAMILIES[name].fn(info, *p)
    return out


RESULTS = None


def results():
    global RESULTS
    if RESULTS is None:
        RESULTS = _results()
    return RESULTS


def test_declared_properties_hold_on_the_battery():
    """shape / same_shape / noop / new_colors / sig / bg flags agree with what the functions do."""
    res = results()
    bad = collections.Counter()
    for g_i, g in enumerate(battery()):
        info = Info(g)
        cols = set(np.unique(g).tolist())
        for name, p in INSTANCES:
            fam = FAMILIES[name]
            o = res[(name, p, g_i)]
            s = fam.out_shape(info.h, info.w, p)
            if o is None:
                continue
            assert o.dtype == np.int8 and o.ndim == 2 and o.size > 0, (name, p)
            assert fits(*o.shape), (name, p, o.shape)
            if s is False:
                bad[("shape_false_but_defined", name)] += 1
            elif s is not None and s != o.shape:
                bad[("shape", name)] += 1
            if fam.same_shape and o.shape != g.shape:
                bad[("same_shape", name)] += 1
            if fam.noop is not None and fam.noop(info, *p) and not np.array_equal(o, g):
                bad[("noop", name)] += 1
            if not fam.new_colors and not set(np.unique(o).tolist()) <= cols:
                bad[("new_colors", name)] += 1
            if fam.bg and info.bg is None:
                bad[("bg_undefined_but_result", name)] += 1
            if fam.same_shape and o.shape == g.shape:
                d = o != g
                if not d.any():
                    continue
                if fam.sig == "paint" and set(np.unique(o[d]).tolist()) != {p[fam.paint_idx]}:
                    bad[("paint", name)] += 1
                if fam.sig == "erase" and set(np.unique(o[d]).tolist()) != {info.bg}:
                    bad[("erase", name)] += 1
                if (fam.sig == "fill_bg" or fam.from_bg) and set(np.unique(g[d]).tolist()) != {info.bg}:
                    bad[("fill_bg", name)] += 1
                if fam.sig == "hist" and not np.array_equal(np.bincount(o.ravel(), minlength=10), info.hist):
                    bad[("hist", name)] += 1
            if fam.sig == "hist" and not fam.same_shape and \
                    not np.array_equal(np.bincount(o.ravel(), minlength=10), info.hist):
                bad[("hist", name)] += 1
    assert not bad, dict(bad)


def test_every_family_acts_somewhere_and_is_deterministic():
    res = results()
    acted = collections.Counter()
    for (name, p, g_i), o in res.items():
        g = battery()[g_i]
        if o is not None and not (o.shape == g.shape and np.array_equal(o, g)):
            acted[name] += 1
    assert set(acted) == set(FAMILIES), set(FAMILIES) - set(acted)
    info = Info(battery()[3])
    for name, p in INSTANCES[::7]:
        a = FAMILIES[name].fn(info, *p)
        b = FAMILIES[name].fn(Info(battery()[3].copy()), *p)
        assert (a is None and b is None) or np.array_equal(a, b)


def test_no_two_instances_are_the_same_function():
    """Menus never show two phrases for one function: every instance behaves differently from
    every other on the battery (the battery holds objects of every size 1-20 both ways, corner
    contacts, block-scaled grids for factors 2-10 ...)."""
    res = results()
    sig = collections.defaultdict(list)
    n = len(battery())
    for name, p in INSTANCES:
        key = tuple(None if res[(name, p, i)] is None else (res[(name, p, i)].shape, res[(name, p, i)].tobytes())
                    for i in range(n))
        sig[key].append((name, p))
    groups = [v for v in sig.values() if len(v) > 1]
    assert not groups, groups[:5]


def test_descriptions_are_unique_and_clean():
    seen = {}
    for name, p in INSTANCES:
        d = FAMILIES[name].describe(*p)
        assert isinstance(d, str) and 10 < len(d) < 260, (name, p, d)
        assert "⟪" not in d and "{" not in d and not d.endswith("."), (name, p, d)
        assert not re.search(r"\b(above|below|inputs?|outputs?)\b", d), (name, p, d)
        assert d not in seen, (name, p, seen.get(d))
        seen[d] = (name, p)
    r = Rule.chain(("rot", (1,)), ("recolor", (3, 5)))
    assert r.describe() == "rotate the grid 90 degrees clockwise, then replace every cell of color 3 with color 5"
    assert r.token == "rot(1)+recolor(3,5)"


def test_background_rules_need_a_background():
    tie = np.array([[1, 2], [2, 1]], dtype=np.int8)
    info = Info(tie)
    assert info.bg is None
    for name, p in INSTANCES:
        fam = FAMILIES[name]
        if fam.bg:
            assert fam.fn(info, *p) is None, (name, p)


def test_no_effect_is_not_undefined():
    """A rule that has nothing to act on returns the grid unchanged (never None), otherwise a rival
    would be dropped as 'undefined' where it is really consistent."""
    g = np.full((6, 6), 4, dtype=np.int8)
    g[1:3, 1:4] = 2
    info = Info(g)
    for name, p in [("fill_holes", (3,)), ("remove_isolated", (4,)), ("remove_size", ("eq", 5, 4)),
                    ("recolor_size", (5, 1, 4)), ("lines", (7, "row")), ("rays", (7, "up")),
                    ("hollow", (4,)), ("hollow_rects", (1,)), ("connect", (7, "row")), ("delete", (9,))]:
        o = FAMILIES[name].fn(info, *p)
        assert o is not None and np.array_equal(o, g), (name, p)


def _brute(z: Info, y: np.ndarray):
    out = set()
    for name, p in INSTANCES:
        r = FAMILIES[name].fn(z, *p)
        if r is not None and r.shape == y.shape and np.array_equal(r, y):
            out.add((name, p))
    return out


@pytest.mark.parametrize("seed", range(3))
def test_solve_is_exact(seed):
    """solve(z, y) returns exactly the instances mapping z to y, for y made by a random instance
    (positive cases, incl. two-step results), by another battery grid, or by a small edit."""
    r = np.random.default_rng(seed)
    bat = battery()
    checked = 0
    for _ in range(120):
        z = bat[int(r.integers(len(bat)))]
        zi = Info(z)
        kind = r.random()
        if kind < 0.7:
            name, p = INSTANCES[int(r.integers(len(INSTANCES)))]
            y = FAMILIES[name].fn(zi, *p)
            if kind < 0.2 and y is not None:
                n2, p2 = INSTANCES[int(r.integers(len(INSTANCES)))]
                y = FAMILIES[n2].fn(Info(y), *p2)
        elif kind < 0.85:
            y = bat[int(r.integers(len(bat)))]
        else:
            y = z.copy()
            y[int(r.integers(z.shape[0])), int(r.integers(z.shape[1]))] = int(r.integers(10))
        if y is None or not fits(*y.shape) or (y.shape == z.shape and np.array_equal(y, z)):
            continue
        got = set(solve(zi, Info(y), fixed_maps=True))
        assert got == _brute(zi, y), (sorted(got ^ _brute(zi, y))[:5])
        checked += 1
    assert checked >= 60


def test_neighbour_counts():
    a = np.array([[1, 1, 0], [0, 1, 0], [2, 0, 2]], dtype=np.int8)
    n4 = neighbour_count_same(a, 4)
    n8 = neighbour_count_same(a, 8)
    assert n4[0, 0] == 1 and n4[1, 1] == 1 and n4[2, 0] == 0
    assert n8[2, 0] == 0 and n8[0, 2] == 1 and n8[1, 0] == 1


def test_solve_is_exact_for_object_size_rules():
    """remove_size / recolor_size candidates are derived from the changed cells; check every
    comparison, size and connectivity that acts on a battery grid."""
    checked = 0
    for g in battery():
        zi = Info(g)
        if zi.bg is None:
            continue
        for name in ("remove_size", "recolor_size", "remove_isolated", "keep_extreme", "recolor_extreme",
                     "recolor_size_cmp", "recolor_all_extreme", "remove_all_extreme", "crop_blob", "extract_fg",
                     "rays_skip", "rays_over"):
            for p in FAMILIES[name].instances[::(9 if name == "recolor_size_cmp" else 5)]:
                y = FAMILIES[name].fn(zi, *p)
                if y is None or np.array_equal(y, g):
                    continue
                got = set(solve(zi, Info(y), fixed_maps=True))
                assert (name, p) in got
                assert got == _brute(zi, y), (name, p, sorted(got ^ _brute(zi, y))[:4])
                checked += 1
    assert checked > 200
