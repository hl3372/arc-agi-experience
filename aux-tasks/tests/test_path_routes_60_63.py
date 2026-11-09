"""The irregular routes of tasks 60-63 (2026-09-15: "we also need more random paths").

Every check reads the drawn picture, never the construction plan: the line is
recovered from the grid the way a solver sees it, so a route that is not one
readable stroke fails here even if the builder thought it was.
"""
from collections import Counter
from functools import lru_cache

import numpy as np
import pytest

from generators import path_routes
from registry import generate

TASKS = ("60", "61", "62", "63")
IRREGULAR = path_routes.IRREGULAR_FAMILIES
N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


@lru_cache(maxsize=None)
def sweep(tid):
    return [generate(tid, seed) for seed in range(400)]


def drawn_path(ex):
    """The path cells as they appear: path colour plus the start marker."""
    a = np.asarray(ex["input"])
    m = ex["metadata"]
    mask = a == m["path_color"]
    if "start_color" in m:
        mask |= a == m["start_color"]
    return set(map(tuple, np.argwhere(mask)))


def order(cells):
    """Walk the drawn cells from one end; raises if they are not one line."""
    neighbours = {p: [(p[0]+dr, p[1]+dc) for dr, dc in N4
                      if (p[0]+dr, p[1]+dc) in cells] for p in cells}
    ends = sorted(p for p, ns in neighbours.items() if len(ns) == 1)
    assert len(ends) == 2, f"a drawn path must have exactly two ends, got {len(ends)}"
    assert all(len(ns) <= 2 for ns in neighbours.values()), "the drawn path branches"
    walk, seen = [ends[0]], {ends[0]}
    while True:
        following = [q for q in neighbours[walk[-1]] if q not in seen]
        if not following:
            break
        walk.append(following[0])
        seen.add(following[0])
    assert seen == cells, "the drawn path is disconnected"
    return walk


# ---------------------------------------------------------------------------
# shares
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tid", TASKS)
def test_irregular_routes_are_a_large_minority(tid):
    families = Counter(ex["metadata"]["path_family"] for ex in sweep(tid))
    total = sum(families.values())
    assert set(families) == set(path_routes.FAMILIES)
    share = sum(families[f] for f in IRREGULAR)/total
    assert .35 <= share <= .45, (tid, share)
    # None of the four is a token presence, and no templated mode was dropped.
    assert all(families[f]/total > .04 for f in IRREGULAR), families
    assert all(families[f]/total > .05 for f in path_routes.TEMPLATE_FAMILIES), families
    assert families["serpentine"] > families["straight"], families


@pytest.mark.parametrize("tid", TASKS)
def test_route_kind_metadata_matches_the_family(tid):
    for ex in sweep(tid):
        m = ex["metadata"]
        assert m["path_route_kind"] == ("irregular" if m["path_family"] in IRREGULAR
                                        else "templated")


# ---------------------------------------------------------------------------
# invariants on the drawn grid
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tid", TASKS)
def test_drawn_line_is_one_cell_wide_with_two_ends(tid):
    for seed, ex in enumerate(sweep(tid)):
        cells = drawn_path(ex)
        walk = order(cells)
        assert len(walk) == ex["metadata"]["path_length"], (tid, seed)
        # No two non-consecutive cells are 4-adjacent: that is what makes the
        # line one cell wide and its traversal unique.
        assert path_routes.is_simple_line(walk), (tid, seed)
        index = {cell: i for i, cell in enumerate(walk)}
        for i, (r, c) in enumerate(walk):
            for dr, dc in N4:
                j = index.get((r+dr, c+dc))
                assert j is None or abs(j-i) == 1, (tid, seed, (r, c))


@pytest.mark.parametrize("tid", TASKS)
def test_endpoints_are_unique_and_the_marker_is_one_of_them(tid):
    for seed, ex in enumerate(sweep(tid)):
        cells = drawn_path(ex)
        ends = [p for p in cells
                if sum((p[0]+dr, p[1]+dc) in cells for dr, dc in N4) == 1]
        assert len(ends) == 2, (tid, seed)
        marker = ex["metadata"].get("start_color")
        if marker is not None:
            spots = np.argwhere(np.asarray(ex["input"]) == marker)
            assert len(spots) == 1, (tid, seed)
            assert tuple(spots[0]) in ends, (tid, seed)


@pytest.mark.parametrize("tid", TASKS)
def test_the_line_rarely_touches_itself_diagonally(tid):
    shares = []
    for seed, ex in enumerate(sweep(tid)):
        share = path_routes.diagonal_contact_share(order(drawn_path(ex)))
        assert share <= path_routes.MAX_DIAGONAL_CONTACT, (tid, seed, share)
        assert abs(share-ex["metadata"]["path_diagonal_contact"]) < 1e-9, (tid, seed)
        shares.append(share)
    # A bound that only a handful of routes approach: the line is a stroke,
    # not a patch.
    assert np.mean(shares) < .05
    assert np.mean(np.asarray(shares) > .10) < .04


@pytest.mark.parametrize("tid", TASKS)
def test_irregular_routes_actually_bend(tid):
    rows = [ex["metadata"] for ex in sweep(tid)]
    for m in rows:
        if m["path_family"] in IRREGULAR:
            assert m["path_turns"] >= path_routes.MIN_TURNS[m["path_family"]], m
    irregular = [m["path_turns"]/m["path_length"] for m in rows
                 if m["path_family"] in IRREGULAR]
    straightish = [m["path_turns"]/m["path_length"] for m in rows
                   if m["path_family"] in ("straight", "elbow", "U")]
    assert np.mean(irregular) > 3*np.mean(straightish)


@pytest.mark.parametrize("tid", TASKS)
def test_occupancy_scales_with_the_canvas(tid):
    rows = [(np.asarray(ex["input"]).size, ex["metadata"]) for ex in sweep(tid)]
    for low, high, floor in ((25, 150, .16), (150, 400, .12), (400, 901, .10)):
        band = [m["path_occupancy"] for size, m in rows if low <= size < high
                if m["path_family"] in IRREGULAR]
        assert len(band) > 10, (tid, low)
        assert np.mean(band) > floor, (tid, low, np.mean(band))
    # Irregular routes fill their canvas at least as well as the templates do.
    irregular = [m["path_occupancy"] for _, m in rows if m["path_family"] in IRREGULAR]
    templated = [m["path_occupancy"] for _, m in rows if m["path_family"] not in IRREGULAR]
    assert np.mean(irregular) > .7*np.mean(templated)
    big = [m["path_length"] for size, m in rows if size >= 700 and m["path_family"] in IRREGULAR]
    small = [m["path_length"] for size, m in rows if size < 150 and m["path_family"] in IRREGULAR]
    assert np.mean(big) > 4*np.mean(small)


# ---------------------------------------------------------------------------
# the family must not leak the answer
# ---------------------------------------------------------------------------

def _accuracy(features, labels):
    """Nearest-centroid guess on standardized features: a cheap leak detector."""
    features = np.asarray(features, float)
    labels = np.asarray(labels)
    spread = features.std(axis=0)
    spread[spread == 0] = 1
    features = features/spread
    names = sorted(set(labels.tolist()))
    centroids = np.array([features[labels == name].mean(axis=0) for name in names])
    distance = ((features[:, None, :]-centroids[None])**2).sum(axis=2)
    guess = distance.argmin(axis=1)
    return float(np.mean([names[g] == label for g, label in zip(guess, labels.tolist())]))


@pytest.mark.parametrize("tid", TASKS)
def test_family_is_drawn_independently_of_the_answer(tid):
    rows = [ex["metadata"] for ex in sweep(tid)]
    kinds = [m["path_route_kind"] for m in rows]
    base = max(Counter(kinds).values())/len(kinds)
    # The answer-relevant quantities: n, and for task 60 the endpoints as well.
    answer = []
    for ex, m in zip(sweep(tid), rows):
        walk = order(drawn_path(ex))
        answer.append([m["n"], walk[0][0], walk[0][1], walk[-1][0], walk[-1][1]])
    assert _accuracy(answer, kinds) < base+.12, (tid, _accuracy(answer, kinds), base)
    # And the other direction: knowing the family must not pin down n.
    for kind in set(kinds):
        ns = [m["n"] for m in rows if m["path_route_kind"] == kind]
        assert min(ns) == 1 and max(ns) > 20, (tid, kind)


# ---------------------------------------------------------------------------
# eyeballing
# ---------------------------------------------------------------------------

def render(ex):
    a = np.asarray(ex["input"])
    m = ex["metadata"]
    cells = drawn_path(ex)
    marker = m.get("start_color")
    rows = []
    for r in range(a.shape[0]):
        rows.append("".join("#" if (r, c) in cells else
                            ("." if a[r, c] == m["bg_color"] else ",")
                            for c in range(a.shape[1])))
    if marker is not None:
        spot = tuple(np.argwhere(a == marker)[0])
        row = list(rows[spot[0]])
        row[spot[1]] = "S"
        rows[spot[0]] = "".join(row)
    return rows


def test_print_one_scene_per_family():
    """Six drawn routes per irregular family, for a human to judge."""
    shown = Counter()
    for seed in range(4000):
        if sum(shown[f] for f in IRREGULAR) >= 6*len(IRREGULAR):
            break
        ex = generate("62", seed)
        family = ex["metadata"]["path_family"]
        if family not in IRREGULAR or shown[family] >= 6:
            continue
        shown[family] += 1
        m = ex["metadata"]
        print(f"\n-- {family} seed={seed} {m['grid_h']}x{m['grid_w']} "
              f"len={m['path_length']} turns={m['path_turns']} "
              f"occupancy={m['path_occupancy']:.2f} diag={m['path_diagonal_contact']:.3f}")
        print("\n".join(render(ex)))
    assert all(shown[f] == 6 for f in IRREGULAR), shown
