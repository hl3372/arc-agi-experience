"""Prompt-only solvers frozen before reading the generators on 2026-09-11."""
import re
import copy
from collections import Counter
import pytest
from registry import generate, list_tasks
from prompts import get_prompt, PROMPT_TEMPLATES
from generators import isolation_46_48 as isolation
from generators import rectangle_edits_49_50 as rectedit
import numpy as np


def named_ground(prompt):
    value = re.search(r'Treat color (\S+) as background', prompt).group(1)
    return int(value) if value.isdigit() else value


def isolated(grid, ground, connectivity):
    a = np.asarray(grid)
    coords = {}
    for r in range(a.shape[0]):
        for c in range(a.shape[1]):
            if a[r,c] != ground:
                coords.setdefault(a[r,c].item(), []).append((r,c))
    found = set()
    # Pairwise distances, rather than a generator-style neighbor traversal.
    for points in coords.values():
        for p in points:
            adjacent = any(p != q and (abs(p[0]-q[0]) + abs(p[1]-q[1]) == 1
                           if connectivity == 4 else max(abs(p[0]-q[0]),abs(p[1]-q[1])) == 1)
                           for q in points)
            if not adjacent:
                found.add(p)
    return found


def components(grid):
    a=np.asarray(grid)
    unseen={(r,c) for r in range(a.shape[0]) for c in range(a.shape[1])}
    groups=[]
    while unseen:
        p=min(unseen); unseen.remove(p); color=a[p].item(); stack=[p]; cells={p}
        while stack:
            r,c=stack.pop()
            for q in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                if q in unseen and a[q]==color:
                    unseen.remove(q);stack.append(q);cells.add(q)
        ps=np.asarray(sorted(cells));lo=ps.min(axis=0);hi=ps.max(axis=0)
        groups.append((color,cells,tuple(map(int,(*lo,*hi)))))
    return groups


def filled_rectangles(grid):
    groups=[]
    for color,cells,(r0,c0,r1,c1) in components(grid):
        h,w=r1-r0+1,c1-c0+1
        if min(h,w)>=2 and len(cells)==h*w:
            groups.append((color,cells,(r0,c0,r1,c1)))
    return groups


def hollow_rectangles(grid):
    rings=[]
    for color,cells,(r0,c0,r1,c1) in components(grid):
        h,w=r1-r0+1,c1-c0+1
        for t in range(1,(min(h,w)-1)//2+1):
            ring={(r,c) for r in range(r0,r1+1) for c in range(c0,c1+1)
                  if min(r-r0,r1-r,c-c0,c1-c)<t}
            if cells==ring:
                rings.append((color,cells,(r0,c0,r1,c1),t));break
    return rings


def infer_ground(grid):
    colors,counts=np.unique(grid,return_counts=True)
    winners=colors[counts==counts.max()]
    if len(winners)!=1:
        raise ValueError('No unique visible plurality background')
    return winners[0].item()


def solve(task, grid, prompt):
    a=np.asarray(grid)
    if task in ('46a','46b','47','48'):
        conn=int(re.search(r'([48])-connected neighbor',prompt).group(1))
        ground=infer_ground(a)
        cells=isolated(a,ground,conn)
        if task.startswith('46'):
            return str(len(cells))
        out=a.copy()
        for p in cells:out[p]=ground
        return out.tolist()
    out=a.copy()
    if task=='49':
        ground=infer_ground(a)
        t=int(re.search(r'(\d+)-cell-thick border',prompt).group(1))
        for color,_,(r0,c0,r1,c1) in filled_rectangles(a):
            if color!=ground and r1-r0+1>2*t and c1-c0+1>2*t:
                out[r0+t:r1-t+1,c0+t:c1-t+1]=ground
    elif task=='50':
        ground=infer_ground(a)
        for color,_,(r0,c0,r1,c1),t in hollow_rectangles(a):
            if color!=ground:
                out[r0:r1+1,c0:c1+1]=color
    else:
        raise ValueError(task)
    return out.tolist()


@pytest.fixture(scope="module")
def sweep():
    return {task: [generate(task, seed) for seed in range(2000)] for task in ("46a", "47", "49", "50")}


@pytest.mark.parametrize("task", ["46a", "47", "49", "50"])
def test_independent_prompt_oracle(sweep, task):
    for ex in sweep[task][:400]:
        assert solve(task, ex["input"], get_prompt(task, ex["metadata"])) == ex["output"]


@pytest.mark.parametrize("task", ["46a", "47", "49", "50"])
def test_symbol_permutations(sweep, task):
    for seed, ex in enumerate(sweep[task][:100]):
        palette = np.random.default_rng(seed + 9851).permutation(list("ABCDEFGHIJ"))
        inp = palette[np.asarray(ex["input"])].tolist()
        expected = palette[np.asarray(ex["output"])].tolist() if isinstance(ex["output"], list) else ex["output"]
        assert solve(task, inp, get_prompt(task, ex["metadata"], color_palette=palette.tolist())) == expected


@pytest.mark.parametrize("task", ["46a", "47", "49", "50"])
def test_dimensions_palette_and_background(sweep, task):
    meta = [e["metadata"] for e in sweep[task]]
    # 49/50 moved from 4/3 to MIN_SIDE = 5 on 2026-09-15 (user: small sides are ambiguous).
    floor = 1 if task in ("46a", "47") else 5
    for key in ("grid_h", "grid_w"):
        assert set(m[key] for m in meta) == set(range(floor, 31))
    palette = Counter(m["n_colors"] for m in meta)
    assert set(palette) == set(range(2, 11)) and min(palette.values()) > 100
    assert abs(np.corrcoef([m["grid_h"] for m in meta], [m["grid_w"] for m in meta])[0, 1]) < .12
    assert 40 < sum(m["grid_h"] * m["grid_w"] >= 700 for m in meta) < 180
    assert all(m["background_margin"] > 1.25 for m in meta)
    assert np.mean([m["attempts"] for m in meta]) < 5
    if task in ("46a", "47"):
        assert 60 < sum(min(m["grid_h"], m["grid_w"]) == 1 for m in meta) < 200
    else:
        assert 300 < sum(m["texture_cells"] > 0 for m in meta) < 750


def test_isolation_diversity_and_count_range(sweep):
    meta = [e["metadata"] for e in sweep["46a"]]
    assert 75 < sum(m["zero"] for m in meta) < 190
    assert sum(m["connectivity_matters"] for m in meta) > 1200
    assert 850 < sum(m["connectivity"] == 4 for m in meta) < 1150
    families = {f for m in meta for f in m["shape_families"]}
    assert {"frame", "L", "T", "U", "H", "plus", "stair", "ellipse", "diamond", "diagonal", "mixed_color_pairs"} <= families
    assert max(max(m["shape_areas"]) for m in meta) > 100
    counts = {int(e["output"]) for e in sweep["46a"]}
    assert 0 in counts and max(counts) > 100 and len(counts) > 65
    large = [m for m in meta if m["grid_h"] * m["grid_w"] >= 700]
    assert np.mean([m["foreground_cells"] / (m["grid_h"] * m["grid_w"]) for m in large]) > .15
    for a, b in zip(sweep["46a"], sweep["47"]):
        assert a["input"] == b["input"]
        assert int(a["output"]) == int(np.count_nonzero(np.asarray(b["input"]) != np.asarray(b["output"])))


def test_rectangle_diversity(sweep):
    for task in ("49", "50"):
        meta = [e["metadata"] for e in sweep[task]]
        counts = {m["rectangle_count"] for m in meta}
        assert set(range(0, 10)) <= counts
        assert sum(m["object_count"] > m["rectangle_count"] for m in meta) > 650
        assert sum(len({r[4] for r in m["rectangles"]}) < len(m["rectangles"]) for m in meta) > 120
        assert max(max(r[2:4]) for m in meta for r in m["rectangles"]) >= 25
        assert {m["background"] for m in meta} == {"solid", *rectedit.BACKGROUND_KINDS}
    meta = [e["metadata"] for e in sweep["50"]]
    assert 60 < sum(m["identity"] for m in meta) < 170
    assert sum(len({r[5] for r in m["rectangles"]}) > 1 for m in meta) > 200
    assert 220 < sum("chamfered_ring" in m["shape_families"] for m in meta) < 500
    assert max(r[5] for m in meta for r in m["rectangles"]) >= 4
    assert {"filled_rect", "L", "T", "U", "open_frame", "chamfered_ring"} <= {f for m in meta for f in m["shape_families"]}


def test_rule_selection_cannot_change_the_input(monkeypatch):
    for seed in range(60):
        monkeypatch.setattr(isolation, "choose_connectivity", lambda rng: 4)
        a = generate("46a", seed)
        monkeypatch.setattr(isolation, "choose_connectivity", lambda rng: 8)
        b = generate("46a", seed)
        assert a["input"] == b["input"]
        grids = []
        for thickness in (1, 2, 3):
            monkeypatch.setattr(rectedit, "choose_thickness", lambda rng, t=thickness: t)
            grids.append(generate("49", seed)["input"])
        assert grids[0] == grids[1] == grids[2]


@pytest.mark.parametrize("task", ["46a", "47", "49", "50"])
def test_full_output_validator_detects_tampering(sweep, task):
    for ex in sweep[task][:20]:
        bad = copy.deepcopy(ex)
        if task == "46a":
            bad["output"] = str(int(bad["output"]) + 1)
        else:
            bad["output"][0][0] = (bad["output"][0][0] + 1) % 10
        with pytest.raises(ValueError):
            if task in ("46a", "47"):
                isolation.validate_example(bad, remove=task == "47")
            else:
                rectedit.validate_example(bad, fill=task == "50")


def test_small_rectangles_have_empty_interiors():
    a = np.zeros((12, 17), int)
    a[1:4, 1:4] = 2
    a[4:11, 8:16] = 3
    expected = a.copy()
    expected[7:8, 11:13] = 0
    assert rectedit.solve(a, fill=False, thickness=3) == expected.tolist()


@pytest.mark.parametrize("seed", [450, 647, 900, 1344, 1785, 9656])
def test_tiny_zero_isolation_cases_have_an_identifiable_field(seed):
    ex = generate("46a", seed)
    assert ex["output"] == solve("46a", ex["input"], get_prompt("46a", ex["metadata"])) == "0"
    assert ex["metadata"]["background_margin"] > 1.25


def test_fill_recognizes_frames_and_keeps_other_shapes():
    a = np.zeros((23, 29), int)
    expected = a.copy()
    for r, c, h, w, t in ((1, 1, 9, 8, 2), (12, 16, 10, 12, 3)):
        for y in range(h):
            for x in range(w):
                if min(y, x, h - 1 - y, w - 1 - x) < t:
                    a[r + y, c + x] = 4
        expected[r:r + h, c:c + w] = 4
    a[14:19, 2:7] = expected[14:19, 2:7] = 6  # Already filled.
    a[2:9, 16] = a[2:9, 22] = a[8, 16:23] = 7  # Open U.
    expected[a == 7] = 7
    a[5, 5] = 6  # Interior decoration is overwritten.
    a[0, 28] = expected[0, 28] = 4  # Same-color exterior dot is preserved.
    for y in range(7):
        for x in range(7):
            if min(y, x, 6 - y, 6 - x) < 2 and not (y in (0, 6) and x in (0, 6)):
                a[12 + y, 8 + x] = expected[12 + y, 8 + x] = 8  # Closed, nonrectangular outline.
    assert rectedit.solve(a, fill=True) == expected.tolist()
    assert solve("50", a, get_prompt("50")) == expected.tolist()


def test_retirement_and_prompt_policy(sweep):
    for old, new in (("46b", "46a"), ("48", "47")):
        assert old not in list_tasks() and old not in PROMPT_TEMPLATES
        with pytest.raises(KeyError, match=f"merged into task {new}"):
            generate(old, 0)
    for task, examples in sweep.items():
        for ex in examples[:100]:
            p = get_prompt(task, ex["metadata"], color_palette=list("ABCDEFGHIJ"))
            assert "Treat color" not in p and "most frequent" not in p and "background is" not in p
            if task == "46a":
                assert p.endswith("Answer with the number.")
            else:
                assert p.endswith("Leave every other cell unchanged.")
            if task in ("46a", "47"):
                assert "isolated single cells" not in p
                assert ("diagonally" in p) == (ex["metadata"]["connectivity"] == 8)
