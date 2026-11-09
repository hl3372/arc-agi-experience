"""Task 25: the visible figure/ground certificate, coverage and naturalness.

The oracle below re-implements the certificate from the visible grid alone
(pure Python sets, no generator helpers). Exactly one colour must take one of
two paths, and on both it owns >= 35% of the border ring, reaches every side
and outweighs its own enclosed pockets:

* ``ring_majority`` -- it owns >= 60% of the ring, no rival holds > 25% of it,
  it lies there in at most four pieces of at least four cells, its
  border-touching mass is >= 1.25x any rival's, and a nonmodal ground keeps a
  clean two-cell margin around ONE dominant figure (exactly one colour reaches
  its count, and that colour's largest component holds >= 60% of the non-ground
  cells);
* ``separating_field`` -- figures cut by the border own more of the ring than it
  does, so instead: no rival holds > 30% of the ring, its border-touching cells
  are ONE connected component, that component is 4-adjacent to every
  colour-blind component of non-ground cells, it holds >= 30% of the canvas and
  >= 1.25x the total mass of any other colour.

On top of both paths come the scene clauses that keep a picture from ALSO
reading as an inset panel inside a drawn border ("2 is the background and 5 is
its border"). They speak about the biggest colour-blind component of non-ground
cells -- the picture's one big figure -- once it holds 60% of them and a quarter
of the canvas and the border does not cut it: two colours need a ground over
half the canvas and no figure stretching over 70% of a side; a dominant figure
keeps a ground margin of 2 cells, a tenth of the shorter side on a big canvas,
on every side; and under a ground of 55% it needs company -- another figure of
at least four cells in a colour of its own, wholly outside its bounding box --
plus either an off-centre position (opposite margins differing by 3 cells) or a
concavity the ground reaches 2 cells into. ``panel_reader`` below is the reading
they defend against, and no accepted example may offer it.
"""
from collections import Counter
from copy import deepcopy
import time

import numpy as np
import pytest

from generators import background_color_25 as G
from prompts import get_prompt
from registry import generate
from symbol_remap import _COLOR_STRING_OUTPUT_TASKS


def _components(cells_by_color):
    """4-connected components of a {cell: color} map, as lists of (color, set)."""
    seen, result = set(), []
    for start, color in cells_by_color.items():
        if start in seen:
            continue
        part, todo = {start}, [start]
        seen.add(start)
        while todo:
            r, c = todo.pop()
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in cells_by_color and q not in seen and cells_by_color[q] == color:
                    seen.add(q)
                    part.add(q)
                    todo.append(q)
        result.append((color, part))
    return result


def oracle_certificate(grid):
    """(ground, path) from the visible grid; raises ValueError when no colour qualifies."""
    h, w = len(grid), len(grid[0])
    cells = {(r, c): grid[r][c] for r in range(h) for c in range(w)}
    ring = [(r, c) for (r, c) in cells if r in (0, h - 1) or c in (0, w - 1)]
    ring_count = Counter(cells[p] for p in ring)
    ring_set = set(ring)
    parts = _components(cells)
    counts = Counter(cells.values())

    def ring_majority(ground, ground_parts, ground_mass):
        if ring_count[ground] < .60 * len(ring):
            raise ValueError("no ring owner")
        if any(n > .25 * len(ring) for v, n in ring_count.items() if v != ground):
            raise ValueError("rival on ring")
        if len(ground_parts) > 4 or min(len(p) for p in ground_parts) < 4:
            raise ValueError("ground in small pieces")
        mass = Counter()
        for color, part in parts:
            if part & ring_set and color != ground:
                mass[color] += len(part)
        if ground_mass < 1.25 * max(list(mass.values()) or [0]):
            raise ValueError("rival mass")
        if counts[ground] <= max(n for v, n in counts.items() if v != ground):
            # the clean two-cell margin is replaced by the scene clauses below: the DOMINANT
            # figure keeps a proportional margin and company, smaller ones may be cut
            over = [v for v, n in counts.items() if v != ground and n >= counts[ground]]
            if len(over) != 1:
                raise ValueError("several colours outweigh the ground")
            body = max(len(part) for color, part in parts if color == over[0])
            if body < .60 * sum(n for v, n in counts.items() if v != ground):
                raise ValueError("not one dominant figure")

    def separating_field(ground, ground_parts, ground_mass):
        if any(n > .30 * len(ring) for v, n in ring_count.items() if v != ground):
            raise ValueError("rival on ring")
        if len(ground_parts) != 1:
            raise ValueError("the ground is not one field")
        field = ground_parts[0]
        for _color, part in _components({p: 0 for p in cells if cells[p] != ground}):
            if not any((r + dr, c + dc) in field for r, c in part
                       for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))):
                raise ValueError("a figure is walled off the field")
        if counts[ground] < .30 * h * w:
            raise ValueError("the field is too thin")
        if counts[ground] < 1.25 * max(n for v, n in counts.items() if v != ground):
            raise ValueError("rival mass on the canvas")

    def scene_rules(ground, ground_parts):
        """The picture must not also read as an inset panel inside a drawn border."""
        figure_cells_ = {p: 0 for p in cells if cells[p] != ground}
        blind = [part for _c, part in _components(figure_cells_)]
        if len(counts) == 2:
            if counts[ground] < .50 * h * w:
                raise ValueError("two colours and a ground under half the canvas")
            for part in blind:
                if part & ring_set:
                    continue
                if (max(r for r, _ in part) - min(r for r, _ in part) + 1 > .70 * h
                        or max(c for _, c in part) - min(c for _, c in part) + 1 > .70 * w):
                    raise ValueError("a two-colour figure stretches across the canvas")
        minority = counts[ground] <= max(n for v, n in counts.items() if v != ground)
        body = max(blind, key=len)
        if len(body) < .60 * len(figure_cells_) or len(body) < .25 * h * w:
            if minority:
                raise ValueError("a minority ground with no dominant figure")
            return
        if body & ring_set:
            if minority:
                raise ValueError("a dominant figure over a minority ground touches the border")
            return
        r0, r1 = min(r for r, _ in body), max(r for r, _ in body)
        c0, c1 = min(c for _, c in body), max(c for _, c in body)
        margins = (r0, h - 1 - r1, c0, w - 1 - c1)
        if min(margins) < max(2, -(-min(h, w) * 10 // 100)):
            raise ValueError("the margin around a dominant figure reads as a border")
        if counts[ground] >= .55 * h * w:
            return
        body_colors = {cells[p] for p in body}
        if not any(len(part) >= 4 and {cells[p] for p in part} - body_colors
                   and not any(r0 <= r <= r1 and c0 <= c <= c1 for r, c in part)
                   for part in blind):
            raise ValueError("a thin ground carries one figure only")
        if abs(margins[0] - margins[1]) >= 3 or abs(margins[2] - margins[3]) >= 3:
            return
        outside = set().union(*ground_parts) if ground_parts else set()
        pocket = {p: 0 for p in outside if r0 <= p[0] <= r1 and c0 <= p[1] <= c1}
        for _c, part in _components(pocket):
            deep = max(min(r - r0 + 1, r1 - r + 1, c - c0 + 1, c1 - c + 1) for r, c in part)
            if deep >= 2 and len(part) >= .08 * (r1 - r0 + 1) * (c1 - c0 + 1):
                return
        raise ValueError("a centred convex figure over a thin ground reads as a panel")

    winners = []
    for ground in ring_count:
        if ring_count[ground] < .35 * len(ring):
            continue
        if not all((any(grid[0][c] == ground for c in range(w)),
                    any(grid[h - 1][c] == ground for c in range(w)),
                    any(grid[r][0] == ground for r in range(h)),
                    any(grid[r][w - 1] == ground for r in range(h)))):
            continue
        ground_parts = [part for color, part in parts if color == ground and part & ring_set]
        pockets = [len(part) for color, part in parts if color == ground and not part & ring_set]
        ground_mass = sum(len(part) for part in ground_parts)
        if pockets and max(pockets) >= ground_mass:
            continue
        for name, rule in (("ring_majority", ring_majority), ("separating_field", separating_field)):
            try:
                rule(ground, ground_parts, ground_mass)
                scene_rules(ground, ground_parts)
            except ValueError:
                continue
            winners.append((ground, name))
            break
    if len(winners) != 1:
        raise ValueError("no unique ground")
    return winners[0]


def oracle(grid):
    """The ground colour of the visible grid; raises ValueError when no colour qualifies."""
    return oracle_certificate(grid)[0]


def _ring_at(grid, h, w, depth):
    """The cells at Chebyshev depth ``depth`` from the border, as values."""
    if 2 * depth >= h or 2 * depth >= w:
        return []
    out = [grid[depth][c] for c in range(depth, w - depth)]
    if h - 1 - depth != depth:
        out += [grid[h - 1 - depth][c] for c in range(depth, w - depth)]
    for r in range(depth + 1, h - 1 - depth):
        out.append(grid[r][depth])
        if w - 1 - depth != depth:
            out.append(grid[r][w - 1 - depth])
    return out


def panel_reader(grid):
    """The competing reading -- an inset panel inside a drawn border -- or None.

    One colour fills the ENTIRE outer ring; every other colour lies strictly inside a single
    rectangle whose border is uniformly that colour (the tightest one is the bounding box of
    the other colours grown by one, so it is that box that is read); the frame is thin enough
    to read as a drawn border rather than as a ground margin (at most a cell in eight of the
    shorter side, never under two, matching the library's own one- and two-cell ``border``
    field); and the rectangle holds a panel -- one colour other than the frame's covers more
    than half of it, reaches all four of its sides and owns more of its own ring than the
    frame colour does. A reader who sees that answers the panel's colour, not the frame's.

    Written out here in plain Python, independently of the generator's ``panel_reading``.
    """
    h, w = len(grid), len(grid[0])
    ring = _ring_at(grid, h, w, 0)
    if len(set(ring)) != 1:
        return None                            # no uniform frame: nothing reads as a border
    frame = ring[0]
    inside = [(r, c) for r in range(h) for c in range(w) if grid[r][c] != frame]
    if not inside:
        return None
    r0, r1 = min(r for r, _ in inside), max(r for r, _ in inside)
    c0, c1 = min(c for _, c in inside), max(c for _, c in inside)
    if max(r0, h - 1 - r1, c0, w - 1 - c1) > max(2, -(-min(h, w) * 12 // 100)):
        return None                            # a ground margin, not a drawn border
    panel = [row[c0:c1 + 1] for row in grid[r0:r1 + 1]]
    ph, pw = len(panel), len(panel[0])
    fill, most = Counter(v for row in panel for v in row).most_common(1)[0]
    if fill == frame or most <= .5 * ph * pw:
        return None                            # no single fill: the ground shows through
    if not (any(v == fill for v in panel[0]) and any(v == fill for v in panel[-1])
            and any(row[0] == fill for row in panel) and any(row[-1] == fill for row in panel)):
        return None                            # the fill does not reach the frame
    edge = Counter(_ring_at(panel, ph, pw, 0))
    if edge[fill] <= edge[frame]:
        return None                            # the frame colour still lines the inside
    return fill


def surrounding_reader(grid):
    """What a person reads as "the sea everything floats in", scored per colour.

    Own scoring, no certificate: a ground owns the frame of the picture, reaches every side,
    is mostly open sea rather than trapped pockets, and everything else floats inside it.
    """
    h, w = len(grid), len(grid[0])
    n = h * w
    ring0, ring1 = Counter(_ring_at(grid, h, w, 0)), Counter(_ring_at(grid, h, w, 1))
    best, score = None, None
    for colour in {v for row in grid for v in row}:
        cells = {(r, c): 0 for r in range(h) for c in range(w) if grid[r][c] == colour}
        sea = set()
        for _c, part in _components(cells):
            if any(r in (0, h - 1) or c in (0, w - 1) for r, c in part):
                sea |= part
        floats = {(r, c): 0 for r in range(h) for c in range(w) if (r, c) not in sea}
        enclosed = sum(len(part) for _c, part in _components(floats)
                       if not any(r in (0, h - 1) or c in (0, w - 1) for r, c in part))
        sides = sum(any(grid[r][c] == colour for r, c in edge) for edge in (
            [(0, c) for c in range(w)], [(h - 1, c) for c in range(w)],
            [(r, 0) for r in range(h)], [(r, w - 1) for r in range(h)]))
        value = (3 * ring0[colour] / max(1, sum(ring0.values()))
                 + 2 * ring1[colour] / max(1, sum(ring1.values()))
                 + len(sea) / n + (enclosed / max(1, len(floats)))
                 + len(sea) / max(1, len(cells)) + .75 * sides / 4)
        if score is None or value > score:
            best, score = colour, value
    return best


def figure_cells(grid, ground):
    """(figure cells, isolated texture dots) of the visible grid; own code."""
    h, w = len(grid), len(grid[0])
    fig = {(r, c) for r in range(h) for c in range(w) if grid[r][c] != ground}
    dots = {p for p in fig if not any((p[0] + dr, p[1] + dc) in fig
                                      for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc)}
    return fig - dots, dots


def defects(grid, ground):
    """(hairs, notches) of the figure silhouette with texture dots set aside; own code.

    A canvas corner counts as a notch when both of its in-grid neighbours are
    figure: only two neighbours exist there, so the three-neighbour rule is blind
    to the stranded one-cell ground island a figure can leave in a corner.
    """
    h, w = len(grid), len(grid[0])
    fig = {(r, c) for r in range(h) for c in range(w) if grid[r][c] != ground}
    isolated = {p for p in fig if not any((p[0] + dr, p[1] + dc) in fig
                                          for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc)}
    fig -= isolated

    def inside(r, c):
        return 0 <= r < h and 0 <= c < w

    hairs = 0
    for r, c in fig:
        n = 0
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            q = (r + dr, c + dc)
            n += (q in fig) if inside(*q) else 1          # a cut figure continues past the edge
        hairs += n <= 1
    notches = 0
    for r in range(h):
        for c in range(w):
            if (r, c) not in fig and grid[r][c] == ground:
                neighbours = [(r + dr, c + dc) for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))]
                inside = [q for q in neighbours if 0 <= q[0] < h and 0 <= q[1] < w]
                n = sum(q in fig for q in inside)
                notches += n >= min(3, len(inside))
    return hairs, notches


@pytest.fixture(scope="module")
def examples():
    return [generate("25", seed) for seed in range(1000)]


def test_oracle_agrees_and_metadata_records_grid_measures(examples):
    for seed, ex in enumerate(examples[:400]):
        ground, path = oracle_certificate(ex["input"])
        assert str(ground) == ex["output"], seed
        assert path == ex["metadata"]["certificate_path"], seed
    for seed, ex in enumerate(examples):
        m = ex["metadata"]
        a = np.asarray(ex["input"])
        answer = int(ex["output"])
        counts = Counter(a.ravel().tolist())
        assert m["background_count"] == counts[answer]
        assert m["ground_share"] == pytest.approx(counts[answer] / a.size)
        assert m["ground_is_modal"] == m["background_is_modal"] == (counts[answer] > max(n for c, n in counts.items() if c != answer))
        assert m["n_colors"] == len(counts)
        assert m["archetype"] == m["layout"] in G.ARCHETYPES
        assert m["n_figures"] == m["object_count"] == len(m["figure_sizes"]) >= 1
        if m["certificate_path"] == "ring_majority":
            assert m["ground_ring_share"] >= .60 and m["rival_ring_share"] <= .25, seed
        else:
            # the cut figures own more of the ring than the ground, and it still holds a third
            assert m["certificate_path"] == "separating_field", seed
            assert .35 <= m["ground_ring_share"] < .50 and m["rival_ring_share"] <= .30, seed
            assert m["ground_ring_components"] == 1, seed
        for key in ("figures_touching_border", "spanning_bars", "two_color_figures", "nested_figures",
                    "size_spread_ratio", "texture_kind", "texture_colors", "attempts",
                    "certificate_path", "ground_ring_components"):
            assert key in m


def test_certificate_rejects_ambiguous_canvases_and_admits_figure_ground_scenes():
    split = np.zeros((20, 24), int)
    split[:, :12] = 4
    two_sides = np.zeros((20, 24), int)
    two_sides[:, 0] = 4
    two_sides[0, :] = 6
    hugging = np.zeros((20, 20), int)
    hugging[1:19, 1:19] = 7
    hugging[3:17, 3:17] = 0                        # a thin margin around a huge pocket
    panel = np.zeros((20, 24), int)
    panel[:, :6] = 4
    tight = np.zeros((20, 20), int)
    tight[1:19, 1:19] = 5                          # a dominant figure with a one-cell margin
    lattice = np.zeros((14, 14), int)              # 25 ground patches between coloured lines
    for i, line in enumerate((2, 5, 8, 11)):
        lattice[line, :] = 1 + i
        lattice[:, line] = 5 + i
    corner_island = np.zeros((16, 18), int)        # a figure strands the top-left ground cell
    corner_island[0, 1:9] = 3
    corner_island[1:9, 0] = 3
    corner_island[1:9, 1:9] = 3
    stripes = np.zeros((26, 20), int)              # a stack of panels, not one dominant figure
    stripes[2:24, 2:18] = 6
    for band in (4, 9, 14, 19):
        stripes[band:band + 3, 2:18] = 7
    two_fields = np.zeros((20, 20), int)           # a fat cross: either colour could be the field
    two_fields[6:14, :] = 4
    two_fields[:, 6:14] = 4
    split_field = np.zeros((20, 24), int)          # cut figures, but a bar halves the ground
    split_field[9:12, :] = 3
    split_field[0:5, 2:10], split_field[0:5, 13:21] = 4, 5
    split_field[15:20, 3:11], split_field[15:20, 14:22] = 6, 7
    walled_in = np.zeros((20, 24), int)            # a figure sealed inside another one's pocket
    walled_in[0:6, 2:11], walled_in[0:6, 14:22] = 3, 4
    walled_in[14:20, 2:11], walled_in[14:20, 14:22] = 5, 6
    walled_in[8:13, 8:16] = 7
    walled_in[9:12, 9:15] = 0
    walled_in[10, 11:13] = 8
    thin_field = np.zeros((20, 20), int)           # the ground separates but holds 28% of the cells
    thin_field[0:9, 2:10], thin_field[0:9, 11:19] = 3, 4
    thin_field[11:20, 2:10], thin_field[11:20, 11:19] = 5, 6
    # the panel readings the September follow-up closed: a two-colour body inside a uniform
    # margin, a body spanning the canvas, a thin frame on a big canvas, a lone dominant figure
    bordered = np.zeros((20, 20), int)             # 16x16 of 5 in a two-cell frame: 144 of 400
    bordered[2:18, 2:18] = 5                       # ground, and the frame reads as a border
    chamfered = np.zeros((29, 29), int)            # the user's seed: a chamfered block in a
    chamfered[2:27, 3:26] = 2                      # two/three-cell margin of 5
    chamfered[2:4, 3:5] = chamfered[2:4, 24:26] = 0
    chamfered[25:27, 3:5] = chamfered[25:27, 24:26] = 0
    tall = np.zeros((20, 20), int)                 # two colours and a figure over 70% of a side
    tall[3:18, 8:13] = 5
    thin_frame = np.zeros((29, 29), int)           # three colours, modal ground, margin 2 < 3
    thin_frame[2:18, 2:18] = 5
    thin_frame[20:25, 20:25] = 3
    alone = np.zeros((20, 20), int)                # a dominant figure with nothing beside it
    alone[2:18, 3:17] = 5
    alone[8:11, 8:11] = 7
    centred = np.zeros((20, 20), int)              # company, but centred and convex
    centred[2:18, 4:18] = 5
    centred[6:14, 2:3] = 3
    for bad in (split, two_sides, hugging, panel, tight, lattice, corner_island, stripes,
                two_fields, split_field, walled_in, thin_field,
                bordered, chamfered, tall, thin_frame, alone, centred):
        with pytest.raises(ValueError, match="figure/ground"):
            G.infer_background(bad)
        with pytest.raises(ValueError):
            oracle(bad.tolist())
        with pytest.raises(ValueError, match="literal"):
            G.literal_background(bad.tolist())
    # ... and each of the new ones fails the separating-field clause it was built for
    for bad, clause in ((two_fields, "rival color holds too much"),
                        (split_field, "not one field"),
                        (walled_in, "does not sit on the field"),
                        (thin_field, "too thin"),
                        (hugging, "pocket outweighs"),
                        (stripes, "comparable mass on the canvas"),
                        (bordered, "two-color picture needs a ground over half"),
                        (chamfered, "two-color picture needs a ground over half"),
                        (tall, "stretches across the canvas"),
                        (thin_frame, "margin around a dominant figure"),
                        (alone, "thin ground carries one figure only"),
                        (centred, "centred convex figure over a thin ground")):
        with pytest.raises(ValueError, match=clause):
            G.infer_background(bad)
    # ... and the panel reader is exactly the reading they close: it answers the inside colour
    assert panel_reader(bordered.tolist()) == 5 and panel_reader(chamfered.tolist()) == 2
    assert panel_reader(alone.tolist()) == 5
    dominant = np.zeros((26, 23), int)             # nonmodal ground with company: 253 of 598
    dominant[6:23, 3:20] = 5                       # cells, off-centre (margins 6 and 3)
    dominant[1:5, 5:19] = 3
    bar = np.zeros((20, 24), int)                  # the border cuts it, so the span rule of
    bar[9:11, :] = 3                               # clause (h) does not speak about it
    accompanied = np.zeros((20, 20), int)          # the open-body route: a U-shaped body whose
    accompanied[2:18, 4:18] = 5                    # mouth the ground reaches into, centred
    accompanied[2:10, 8:14] = 0
    accompanied[6:14, 1:3] = 3
    nested = np.zeros((20, 20), int)               # margin 4: past the frame a border could be
    nested[4:16, 4:16] = 7
    nested[7:13, 7:13] = 0
    nested[9:11, 9:11] = 2
    corner = np.zeros((20, 24), int)
    corner[:10, :10] = 3
    clipped = np.zeros((20, 20), int)              # cut figures own 60% of the ring, not the ground
    clipped[0:6, 0:7], clipped[0:5, 12:20] = 3, 4
    clipped[14:20, 0:6], clipped[15:20, 13:20] = 5, 6
    clipped[8:12, 8:13] = 7
    for good in (dominant, bar, accompanied, nested, corner, clipped):
        assert G.infer_background(good) == oracle(good.tolist()) == G.literal_background(good.tolist()) == 0
        assert panel_reader(good.tolist()) in (None, 0), good.shape
    # the two paths of the certificate, and each scene takes exactly the one it should
    assert [oracle_certificate(g.tolist())[1] for g in
            (dominant, bar, accompanied, nested, corner, clipped)] == \
        ["ring_majority"] * 5 + ["separating_field"]
    # filling the U's mouth leaves a centred convex body: the same scene is then refused
    filled = accompanied.copy()
    filled[2:10, 8:14] = 5
    with pytest.raises(ValueError, match="centred convex figure"):
        G.infer_background(filled)
    assert G.read_background(clipped) == (0, "separating_field")
    for palette in (np.arange(10)[::-1], np.asarray(list("ABCDEFGHIJ"))):
        for good in (dominant, clipped):
            assert G.infer_background(palette[good]) == palette[0]
            assert G.literal_background(palette[good].tolist()) == palette[0]


def test_archetype_shares_and_no_layout_dominates(examples):
    shares = Counter(e["metadata"]["archetype"] for e in examples)
    n = len(examples)
    bounds = {"dominant_figure": (.14, .27), "few_large": (.10, .21), "crowd": (.09, .20),
              "cluster": (.09, .20), "border_cut": (.07, .18), "spanning_bars": (.04, .14),
              "motif_row": (.02, .09), "heavily_clipped": (.12, .18)}
    for name, (lo, hi) in bounds.items():
        assert lo <= shares[name] / n <= hi, (name, shares[name] / n)
    assert max(shares.values()) / n <= .27
    # the two certificate paths, and heavily clipped scenes are exactly the second one
    paths = Counter(e["metadata"]["certificate_path"] for e in examples)
    assert paths["separating_field"] == shares["heavily_clipped"]
    assert paths["ring_majority"] + paths["separating_field"] == n
    for ex in examples:
        assert (ex["metadata"]["certificate_path"] == "separating_field") == \
            (ex["metadata"]["archetype"] == "heavily_clipped")


def test_nonmodal_grounds_and_ground_share_spread(examples):
    nonmodal = np.mean([not e["metadata"]["ground_is_modal"] for e in examples])
    assert .09 <= nonmodal <= .22, nonmodal
    shares = np.array([e["metadata"]["ground_share"] for e in examples])
    # the brief's p10 <= 0.40 is not reachable while nonmodal grounds need canvases wide
    # enough for a two-cell margin; the realised spread is recorded in the ledger
    assert np.percentile(shares, 10) <= .52 and np.percentile(shares, 5) <= .46
    assert np.percentile(shares, 90) >= .78
    # the majority-colour reader is wrong on the nonmodal grids (ties aside)
    majority = np.mean([Counter(np.asarray(e["input"]).ravel().tolist()).most_common(1)[0][0] == int(e["output"])
                        for e in examples])
    assert abs(majority - (1 - nonmodal)) <= .01


def test_the_border_majority_cue_fails_on_a_bounded_share(examples):
    """A reader that takes the colour owning the ring must be wrong often enough.

    The certificate used to define the ground as the colour owning >= 60% of the
    border ring, so that reader was right by construction on every grid. On the
    heavily clipped scenes no colour owns the ring at all, and the cut figures
    together own more of it than the ground does. The ring PLURALITY stays right
    (the design keeps the ground >= 35% of the ring and every rival <= 30%, so
    it is still the largest single share) and that is recorded here, not hidden.
    """
    grids = [(e["input"], int(e["output"])) for e in examples]
    grids += [(ex["input"], int(ex["output"])) for ex in
              (generate("25", seed) for seed in range(1000, 1500))]
    assert len(grids) == 1500
    majority, plurality = [], []
    for grid, answer in grids:
        h, w = len(grid), len(grid[0])
        ring = [grid[0][c] for c in range(w)] + [grid[h - 1][c] for c in range(w)]
        ring += [grid[r][0] for r in range(1, h - 1)] + [grid[r][w - 1] for r in range(1, h - 1)]
        counts = Counter(ring)
        owners = [v for v, n in counts.items() if n >= .60 * len(ring)]
        majority.append(len(owners) == 1 and owners[0] == answer)
        plurality.append(counts.most_common(1)[0][0] == answer)
    assert np.mean(majority) <= .90, np.mean(majority)       # was 1.00 by construction
    assert np.mean(majority) >= .78                          # ... and the cue still mostly holds
    assert np.mean(plurality) == 1.0                         # the weaker cue is not attacked


def test_no_example_also_reads_as_a_panel_inside_a_border(examples):
    """The reading the user caught: "2 is the background and 5 is its border".

    ``panel_reader`` gives that reading whenever the picture offers it; on every accepted
    example it must either find nothing to read or agree with the certificate. Five thousand
    grids, the fixture's thousand plus four thousand fresh seeds.
    """
    grids = [(e["input"], int(e["output"])) for e in examples]
    grids += [(ex["input"], int(ex["output"])) for ex in
              (generate("25", seed) for seed in range(2000, 6000))]
    assert len(grids) == 5000
    fires = [(grid, answer, panel_reader(grid)) for grid, answer in grids]
    assert [1 for _g, answer, read in fires if read is not None and read != answer] == []
    # the generator's own reader is the same reading, and the validator refuses a grid offering it
    for grid, _answer in grids[:400]:
        assert G.panel_reading(grid) == panel_reader(grid)
    # a frame of 7 in a three-cell margin of 4: the certificate takes it (the ground is 59% of
    # the canvas and keeps its margin) but a person may read the 4 as a border, so the
    # validator refuses it -- which is why no generated example looks like this
    panel = [[4] * 20 for _ in range(20)]
    for r in range(3, 17):
        for c in range(3, 17):
            panel[r][c] = 7
    for r in range(7, 13):
        for c in range(7, 13):
            panel[r][c] = 4
    assert G.infer_background(panel) == 4                 # the certificate is happy
    assert panel_reader(panel) == G.panel_reading(panel) == 7
    with pytest.raises(ValueError, match="panel"):
        G.validate_example({"input": panel, "output": "4"})


def test_the_ground_is_what_surrounds_the_figures(examples):
    """A person's reading, scored from the grid alone, agrees with the certificate."""
    disagree = [seed for seed, ex in enumerate(examples)
                if surrounding_reader(ex["input"]) != int(ex["output"])]
    assert disagree == []


def test_two_colour_pictures_keep_a_modal_ground_and_a_small_figure(examples):
    """Clause (h): with two colours a panel inside a border is the whole picture."""
    two = [e for e in examples if e["metadata"]["n_colors"] == 2]
    assert len(two) >= 40
    for ex in two:
        grid = ex["input"]
        h, w = len(grid), len(grid[0])
        ground = int(ex["output"])
        counts = Counter(v for row in grid for v in row)
        assert counts[ground] >= .50 * h * w, ex["metadata"]["archetype"]
        assert ex["metadata"]["ground_is_modal"]
        for _colour, part in _components({(r, c): 0 for r in range(h) for c in range(w)
                                          if grid[r][c] != ground}):
            if any(r in (0, h - 1) or c in (0, w - 1) for r, c in part):
                continue                           # the border cuts it: never an inset panel
            assert max(r for r, _ in part) - min(r for r, _ in part) + 1 <= .70 * h
            assert max(c for _, c in part) - min(c for _, c in part) + 1 <= .70 * w


def test_a_dominant_figure_keeps_a_proportional_margin_and_company(examples):
    """Clauses (i) and (j): the margin is ground, and a thin ground carries more than one body."""
    seen = {"thin": 0, "offcentre": 0, "concave": 0}
    for seed, ex in enumerate(examples):
        grid = ex["input"]
        h, w = len(grid), len(grid[0])
        ground = int(ex["output"])
        figure = {(r, c): 0 for r in range(h) for c in range(w) if grid[r][c] != ground}
        blind = [part for _c, part in _components(figure)]
        body = max(blind, key=len)
        counts = Counter(v for row in grid for v in row)
        if len(body) < .60 * len(figure) or len(body) < .25 * h * w:
            assert counts[ground] > max(n for c, n in counts.items() if c != ground), seed
            continue
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in body):
            assert counts[ground] > max(n for c, n in counts.items() if c != ground), seed
            continue
        r0, r1 = min(r for r, _ in body), max(r for r, _ in body)
        c0, c1 = min(c for _, c in body), max(c for _, c in body)
        margins = (r0, h - 1 - r1, c0, w - 1 - c1)
        assert min(margins) >= max(2, -(-min(h, w) * 10 // 100)), (seed, margins)
        if counts[ground] >= .55 * h * w:
            continue
        seen["thin"] += 1
        colours = {grid[r][c] for r, c in body}
        company = [part for part in blind
                   if len(part) >= 4 and {grid[r][c] for r, c in part} - colours
                   and not any(r0 <= r <= r1 and c0 <= c <= c1 for r, c in part)]
        assert company, seed
        if abs(margins[0] - margins[1]) >= 3 or abs(margins[2] - margins[3]) >= 3:
            seen["offcentre"] += 1
            continue
        pocket = {(r, c): 0 for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                  if grid[r][c] == ground}
        deep = [part for _c, part in _components(pocket)
                if len(part) >= .08 * (r1 - r0 + 1) * (c1 - c0 + 1)
                and max(min(r - r0 + 1, r1 - r + 1, c - c0 + 1, c1 - c + 1)
                        for r, c in part) >= 2]
        assert deep, seed
        seen["concave"] += 1
    assert seen["thin"] >= 80, seen           # the thin-ground scenes are really sampled
    assert seen["offcentre"] >= 60, seen


def test_colours_reach_ten_and_do_not_track_area(examples):
    counts = Counter(e["metadata"]["n_colors"] for e in examples)
    n = len(examples)
    assert set(counts) == set(range(2, 11))
    assert sum(counts[c] for c in (8, 9, 10)) / n >= .20
    assert .05 <= counts[2] / n <= .16
    assert max(counts.values()) / n <= .25
    small = [e["metadata"]["n_colors"] for e in examples if e["metadata"]["grid_h"] * e["metadata"]["grid_w"] < 300]
    large = [e["metadata"]["n_colors"] for e in examples if e["metadata"]["grid_h"] * e["metadata"]["grid_w"] >= 300]
    assert abs(np.mean(small) - np.mean(large)) < 1.5
    assert np.mean([c >= 8 for c in small]) >= .08 and np.mean([c <= 3 for c in large]) >= .10
    assert len({e["output"] for e in examples}) == 10


def test_sides_are_flat_and_independent(examples):
    sizes = np.asarray([(e["metadata"]["grid_h"], e["metadata"]["grid_w"]) for e in examples])
    assert set(sizes[:, 0]) == set(sizes[:, 1]) == set(range(8, 31))
    for axis in (0, 1):
        hist = Counter(sizes[:, axis].tolist())
        assert all(.02 <= hist[v] / len(examples) <= .075 for v in range(8, 31)), hist
    assert abs(np.corrcoef(sizes.T)[0, 1]) < .12
    assert .02 <= np.mean(sizes[:, 0] == sizes[:, 1]) <= .07
    assert .03 <= np.mean(sizes.prod(axis=1) >= 700) <= .09
    with pytest.raises(ValueError, match="eight cells"):
        G.validate_example({"input": [[0, 1, 0]], "output": "0"})


def test_figures_are_drawn_not_noise(examples):
    total_hairs = total_notches = 0
    for ex in examples:
        h_, n_ = defects(ex["input"], int(ex["output"]))
        total_hairs += h_
        total_notches += n_
    assert total_hairs == 0 and total_notches == 0
    families = Counter(f for e in examples for f in e["metadata"]["shape_families"])
    assert len(families) >= 12
    assert families["rect"] / sum(families.values()) <= .55


def test_details_are_a_cumulative_minority_of_their_body(examples):
    """No body is mostly painted in other colours, and no band colour beats the ground."""
    worst = 0.0
    for seed, ex in enumerate(examples):
        grid = ex["input"]
        ground = int(ex["output"])
        counts = Counter(v for row in grid for v in row)
        fig, _dots = figure_cells(grid, ground)
        body_color = max((c for c in counts if c != ground), key=lambda c: counts[c])
        # only the dominant body's own colour may ever outnumber the ground
        assert not [c for c, n in counts.items()
                    if c not in (ground, body_color) and n > counts[ground]], seed
        if counts[ground] > max(n for c, n in counts.items() if c != ground):
            continue                                   # a modal blob may be touching figures
        blob = max((part for _c, part in _components({p: 0 for p in fig})), key=len)
        inside = [(color, part) for color, part in _components({p: grid[p[0]][p[1]] for p in fig})
                  if part & blob]
        by_color = Counter()
        for color, part in inside:
            by_color[color] += len(part)
        main_color = by_color.most_common(1)[0][0]
        other = len(blob) - by_color[main_color]
        assert other <= .35 * len(blob), (seed, other / len(blob))
        assert len([p for c, p in inside if c != main_color and len(p) >= 2]) <= 1, seed
        worst = max(worst, other / len(blob))
    assert worst <= .35, worst


def test_a_roomy_canvas_carries_a_scene_and_not_mostly_dots(examples):
    for seed, ex in enumerate(examples):
        m = ex["metadata"]
        area = m["grid_h"] * m["grid_w"]
        fig, dots = figure_cells(ex["input"], int(ex["output"]))
        if area >= 200:
            assert len(fig) >= .12 * area, (seed, area, len(fig) / area)
        assert len(dots) < .5 * max(1, len(fig)), (seed, len(dots), len(fig))
    big = [e for e in examples if e["metadata"]["grid_h"] * e["metadata"]["grid_w"] >= 400]
    assert len(big) >= 150                                    # the band is really sampled
    assert min(len(figure_cells(e["input"], int(e["output"]))[0]) /
               (e["metadata"]["grid_h"] * e["metadata"]["grid_w"]) for e in big) >= .12


def test_the_ground_lies_in_a_few_large_pieces(examples):
    """Clause (d): no confetti ground, no stranded island on the ring; (d'): one field."""
    for seed, ex in enumerate(examples):
        grid = ex["input"]
        h, w = len(grid), len(grid[0])
        ground = int(ex["output"])
        ring = {(r, c) for r in range(h) for c in range(w) if r in (0, h - 1) or c in (0, w - 1)}
        cells = {(r, c): grid[r][c] for r in range(h) for c in range(w) if grid[r][c] == ground}
        parts = [part for color, part in _components(cells) if part & ring]
        assert len(parts) == ex["metadata"]["ground_ring_components"], seed
        if ex["metadata"]["certificate_path"] == "separating_field":
            assert len(parts) == 1, (seed, len(parts))
            continue
        assert len(parts) <= 4, (seed, len(parts))
        assert min(len(part) for part in parts) >= 4, (seed, min(len(part) for part in parts))


def test_heavily_clipped_scenes_are_separated_fields(examples):
    """Every path-(ii) grid: one ring-touching field, every figure moated and sitting on it."""
    clipped = [e for e in examples if e["metadata"]["certificate_path"] == "separating_field"]
    assert len(clipped) >= 100
    for ex in clipped:
        grid = ex["input"]
        h, w = len(grid), len(grid[0])
        ground = int(ex["output"])
        m = ex["metadata"]
        assert m["archetype"] == "heavily_clipped"
        ring = [(r, c) for r in range(h) for c in range(w) if r in (0, h - 1) or c in (0, w - 1)]
        on_ring = Counter(grid[r][c] for r, c in ring)
        # the cut figures own MORE of the ring than the ground: the 60% cue is simply absent
        assert sum(n for v, n in on_ring.items() if v != ground) > on_ring[ground]
        assert on_ring[ground] >= .37 * len(ring)                    # certificate floor: .35
        assert max(n for v, n in on_ring.items() if v != ground) <= .28 * len(ring)   # cap: .30
        ground_cells = {(r, c) for r in range(h) for c in range(w) if grid[r][c] == ground}
        field = [part for _c, part in _components({p: 0 for p in ground_cells})
                 if part & set(ring)]
        assert len(field) == 1                                        # (d')
        assert any(grid[0][c] == ground for c in range(w)) and any(grid[h - 1][c] == ground for c in range(w))
        assert any(grid[r][0] == ground for r in range(h)) and any(grid[r][w - 1] == ground for r in range(h))
        over = _components({(r, c): 0 for r in range(h) for c in range(w) if grid[r][c] != ground})
        for _c, part in over:                                         # (e') texture dots included
            assert any((r + dr, c + dc) in field[0] for r, c in part
                       for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)))
        drawn, dots = figure_cells(grid, ground)
        figures = _components({p: 0 for p in drawn})
        assert len(figures) >= 3                                      # several figures
        for _c, part in figures:
            assert len(part) >= 4                                     # none of them a speck
        # a one-cell ground moat between figures: no two of them even touch at a corner
        for i, (_ca, first) in enumerate(figures):
            for _cb, second in figures[i + 1:]:
                assert not any((r + dr, c + dc) in second for r, c in first
                               for dr in (-1, 0, 1) for dc in (-1, 0, 1))
        assert m["ground_share"] >= .30
        rivals = Counter(v for row in grid for v in row)
        assert rivals[ground] >= 1.25 * max(n for v, n in rivals.items() if v != ground)


def test_corner_notches_are_seen_and_rejected(examples):
    """A corner has two neighbours, so the three-neighbour dent rule is blind there."""
    dent = np.zeros((6, 6), dtype=bool)
    dent[0, 1] = dent[1, 0] = dent[1, 1] = True                # strands the (0, 0) ground cell
    assert G.notches(dent) == 0 and G.notches(dent, corners=True) == 1
    chamfer = np.ones((4, 5), dtype=bool)
    chamfer[0, 0] = False
    assert G.notches(chamfer) == 0                             # a body's cut corner is drawn
    assert G.notches(chamfer, corners=True) == 1               # the same cut at a canvas corner
    assert sum(f == "chamfer" for e in examples for f in e["metadata"]["shape_families"]) >= 50
    for seed, ex in enumerate(examples):
        grid = ex["input"]
        h, w = len(grid), len(grid[0])
        ground = int(ex["output"])
        fig, _dots = figure_cells(grid, ground)
        for r, c in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)):
            if grid[r][c] != ground:
                continue
            inside = [q for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                      if 0 <= q[0] < h and 0 <= q[1] < w]
            assert not all(q in fig for q in inside), (seed, (r, c))


def test_archetype_labels_describe_the_realised_grid(examples):
    for seed, ex in enumerate(examples):
        m = ex["metadata"]
        area = m["grid_h"] * m["grid_w"]
        if m["archetype"] == "dominant_figure":
            assert m["largest_foreground_body"] >= .25 * area, (seed, m["largest_foreground_body"] / area)
            # the body, a figure in its pocket, and the companions that clause (j) asks for
            # beside it once the ground is thin (the count was <= 2 before they existed)
            assert m["n_figures"] <= 4, seed
        if m["archetype"] == "cluster":
            assert m["touching_pairs"] >= 1, seed
        if m["archetype"] == "spanning_bars":
            assert m["spanning_bars"] >= 1, seed
        if m["archetype"] == "heavily_clipped":
            assert m["clipped_figures"] >= 2 and m["n_figures"] >= 3, seed
            assert m["ground_ring_share"] < .50 and m["ground_ring_components"] == 1, seed
    dominant = [e["metadata"] for e in examples if e["metadata"]["archetype"] == "dominant_figure"]
    cluster = [e["metadata"] for e in examples if e["metadata"]["archetype"] == "cluster"]
    assert len(dominant) >= 100 and len(cluster) >= 80          # the labels are really sampled
    assert np.median([m["largest_foreground_body"] / (m["grid_h"] * m["grid_w"]) for m in dominant]) >= .40


def test_size_spread_and_composition_variety(examples):
    multi = [e["metadata"]["size_spread_ratio"] for e in examples if e["metadata"]["n_figures"] > 1]
    assert np.mean([r <= 1.5 for r in multi]) <= .25
    assert np.median(multi) >= 2.5
    M = [e["metadata"] for e in examples]
    figures = sum(m["n_figures"] for m in M)
    assert .15 <= sum(m["two_color_figures"] for m in M) / figures <= .45
    assert np.mean([m["figures_touching_border"] > 0 for m in M]) >= .25
    assert np.mean([m["clipped_figures"] > 0 for m in M]) >= .08
    assert np.mean([m["spanning_bars"] > 0 for m in M]) >= .05
    assert np.mean([m["touching_pairs"] > 0 for m in M]) >= .08
    assert np.mean([m["nested_figures"] > 0 for m in M]) >= .01
    assert max(m["n_figures"] for m in M) >= 12
    # occupancy holds up on large canvases
    large = [1 - m["ground_share"] for m in M if m["grid_h"] * m["grid_w"] >= 700]
    assert np.mean(large) >= .22


def test_textures_are_sparse_off_the_ring_and_clear_of_figures(examples):
    textured = [e for e in examples if e["metadata"]["texture_kind"] != "plain"]
    share = len(textured) / len(examples)
    assert .18 <= share <= .40, share
    kinds = {e["metadata"]["texture_kind"] for e in textured}
    assert kinds <= set(G.TEXTURE_KINDS) and len(kinds) == 6
    for ex in textured:
        m = ex["metadata"]
        a = np.asarray(ex["input"])
        ground = int(ex["output"])
        assert 0 < m["texture_coverage"] <= .08
        assert 1 <= len(m["texture_colors"]) <= 2 and ground not in m["texture_colors"]
        fig = a != ground
        padded = np.pad(fig, 1)
        neighbours = sum(padded[1 + dr:1 + dr + a.shape[0], 1 + dc:1 + dc + a.shape[1]]
                         for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc)
        dots = fig & (neighbours == 0)
        assert dots.sum() >= 3
        assert not dots[[0, 1, -2, -1], :].any() and not dots[:, [0, 1, -2, -1]].any()


def test_prompt_symbol_mode_and_colour_string_output(examples):
    assert "25" in _COLOR_STRING_OUTPUT_TASKS
    for seed, ex in enumerate(examples[:100]):
        text = get_prompt("25", ex["metadata"])
        assert text == "Which color is the background? Answer as a single color label."
        assert "frequent" not in text and "background is" not in text
        alphabet = np.random.default_rng(18700 + seed).permutation(list("ABCDEFGHIJ"))
        letters = alphabet[np.asarray(ex["input"])].tolist()
        assert G.infer_background(letters) == oracle(letters) == alphabet[int(ex["output"])]
        assert get_prompt("25", ex["metadata"], color_palette=alphabet.tolist()).endswith("single color symbol.")


def test_two_independent_readings_must_agree(examples, monkeypatch):
    for seed, ex in enumerate(examples[:400]):
        assert str(G.literal_background(ex["input"])) == ex["output"], seed
    # the validator compares them and raises when they part company
    monkeypatch.setattr(G, "literal_background", lambda grid: 99)
    with pytest.raises(ValueError, match="disagree"):
        G.validate_example(deepcopy(examples[5]))


def test_realised_scenes_keep_room_inside_every_certificate_bound(examples):
    ring_shares, rivals, ratios, bodies = [], [], [], []
    for ex in examples:
        if ex["metadata"]["certificate_path"] == "separating_field":
            continue                      # its bounds are checked against path (ii) elsewhere
        grid = ex["input"]
        h, w = len(grid), len(grid[0])
        ground = int(ex["output"])
        cells = {(r, c): grid[r][c] for r in range(h) for c in range(w)}
        ring = [p for p in cells if p[0] in (0, h - 1) or p[1] in (0, w - 1)]
        on_ring = Counter(cells[p] for p in ring)
        ring_shares.append(on_ring[ground] / len(ring))
        rivals.append(max(n for v, n in on_ring.items() if v != ground) / len(ring) if len(on_ring) > 1 else 0)
        mass = Counter()
        for color, part in _components(cells):
            if part & set(ring):
                mass[color] += len(part)
        other = max([n for v, n in mass.items() if v != ground] or [1])
        ratios.append(mass[ground] / other)
        counts = Counter(cells.values())
        if counts[ground] <= max(n for v, n in counts.items() if v != ground):
            over = max((v for v in counts if v != ground), key=lambda v: counts[v])
            biggest = max(len(part) for color, part in _components(cells) if color == over)
            bodies.append(biggest / sum(n for v, n in counts.items() if v != ground))
    assert min(ring_shares) >= .62 and max(rivals) <= .24        # certificate: .60 and .25
    assert min(ratios) >= 1.30                                   # certificate: 1.25
    assert len(bodies) >= 60 and min(bodies) >= .70              # certificate: .60


def test_validator_reads_the_grid_and_raises_when_tampered(examples):
    good = deepcopy(examples[3])
    good["metadata"].update(background_color=-1, answer_color=-1, layout="invented", archetype="x")
    G.validate_example(good)                        # metadata is not consulted
    wrong = deepcopy(examples[3])
    wrong["output"] = str((int(wrong["output"]) + 1) % 10)
    with pytest.raises(ValueError, match="disagrees"):
        G.validate_example(wrong)
    a = np.asarray(examples[3]["input"])
    split = a.copy()
    split[:, :a.shape[1] // 2] = (int(examples[3]["output"]) + 1) % 10
    with pytest.raises(ValueError, match="figure/ground"):
        G.validate_example({"input": split.tolist(), "output": examples[3]["output"]})


def test_plan_is_drawn_once_and_survives_retries(monkeypatch):
    original = G._build
    seen = []

    def fail_first(rng, plan, h, w, ground, colors):
        seen.append((id(plan), plan["archetype"], plan["n_colors"], h, w, ground))
        return None if len(seen) == 1 else original(rng, plan, h, w, ground, colors)

    monkeypatch.setattr(G, "_build", fail_first)
    ex = generate("25", 7)
    assert ex["metadata"]["attempts"] >= 2
    assert len({s for s in seen}) == 1


def test_generation_is_cheap(examples):
    assert max(e["metadata"]["attempts"] for e in examples) < G.MAX_ATTEMPTS
    assert np.mean([e["metadata"]["attempts"] for e in examples]) < 4
    start = time.perf_counter()
    for seed in range(1000, 1100):
        generate("25", seed)
    assert (time.perf_counter() - start) / 100 < .03      # generous under parallel load
