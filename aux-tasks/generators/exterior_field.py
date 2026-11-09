"""Conservative visible-field certificates for selected scene families.

This is an opt-in spatial restriction, not a universal definition of background.
Enclosed pockets may share the field color, but disconnected boundary bands do
not constitute one exterior field. No construction metadata is consulted.
"""
from __future__ import annotations

import numpy as np


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    while remaining:
        point = remaining.pop()
        component, pending = {point}, [point]
        for r, c in pending:
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in remaining:
                    remaining.remove(q)
                    component.add(q)
                    pending.append(q)
        yield component


def infer_exterior_field(grid, *, margin=1.25, majority=False,
                         allow_strips=False, exterior_mass=True):
    a = np.asarray(grid)
    if a.ndim != 2 or a.size < 3 or len(np.unique(a)) < 2:
        raise ValueError("Insufficient context for an exterior field")
    h, w = a.shape
    if min(h, w) == 1:
        line = a.ravel()
        if not allow_strips or line[0] != line[-1]:
            raise ValueError("No clear strip field")
        color = line[0]
        values, counts = np.unique(a, return_counts=True)
        amount = int(np.count_nonzero(a == color))
        if amount <= margin * max(counts[values != color]) or (majority and amount <= .55 * a.size):
            raise ValueError("Contested strip field")
        return color.item()
    border = np.zeros(a.shape, dtype=bool)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    values, counts = np.unique(a[border], return_counts=True)
    # At most one color can cover sixty percent of the perimeter.
    candidates = values[counts >= .60 * int(border.sum())]
    if len(candidates) != 1:
        raise ValueError("No predominant exterior field on the perimeter")
    color = candidates[0]
    field = a == color
    if not all((field[0].any(), field[-1].any(), field[:, 0].any(), field[:, -1].any())):
        raise ValueError("The field does not surround the drawing")
    external = [part for part in _components(field)
                if any(r in (0, h - 1) or c in (0, w - 1) for r, c in part)]
    if len(external) != 1:
        raise ValueError("Disconnected boundary regions compete as the field")
    outside = len(external[0])
    for body in _components(~field):
        rows, cols = zip(*body)
        if (min(rows) == 0 and max(rows) == h - 1) or (min(cols) == 0 and max(cols) == w - 1):
            raise ValueError("Foreground divides opposite canvas edges")
        if exterior_mass and len(body) >= outside:
            raise ValueError("An enclosed region overwhelms the exterior context")
    values, counts = np.unique(a, return_counts=True)
    amount = int(field.sum())
    if amount <= margin * max(counts[values != color]) or (majority and amount <= .50 * a.size):
        raise ValueError("Contested field color")
    return color.item()


# ---------------------------------------------------------------------------
# Figure/ground certificate (task 25). Majority is not required.
# ---------------------------------------------------------------------------

def label_same_color(a):
    """4-connected same-color components over the whole grid: (labels from 1, count)."""
    h, w = a.shape
    labels = np.zeros(h * w, dtype=np.int32)
    flat = a.ravel().tolist()
    n = 0
    for start in range(h * w):
        if labels[start]:
            continue
        n += 1
        color = flat[start]
        labels[start] = n
        pending = [start]
        while pending:
            p = pending.pop()
            r, c = divmod(p, w)
            if r > 0 and not labels[p - w] and flat[p - w] == color:
                labels[p - w] = n
                pending.append(p - w)
            if r < h - 1 and not labels[p + w] and flat[p + w] == color:
                labels[p + w] = n
                pending.append(p + w)
            if c > 0 and not labels[p - 1] and flat[p - 1] == color:
                labels[p - 1] = n
                pending.append(p - 1)
            if c < w - 1 and not labels[p + 1] and flat[p + 1] == color:
                labels[p + 1] = n
                pending.append(p + 1)
    return labels.reshape(h, w), n


RING_FLOOR = .35              # no color under this share of the border ring can be the ground
SEPARATING_RIVAL_RING = .30   # on the separating-field path, the cap on every other color
SEPARATING_MASS_SHARE = .30   # ... and the share of the canvas the ground must hold

# ---------------------------------------------------------------------------
# Scene rules against the competing "inset panel with a drawn border" reading
# (``panel_rules``; task 25 asks for them, task 83 reads a ground without them).
# ---------------------------------------------------------------------------
TWO_COLOR_SPAN = .70          # on two colors a figure may stretch this far across a side
LOW_GROUND_SHARE = .55        # under this share a dominant figure needs company and a broken margin
DOMINANT_FIGURE_SHARE = .60   # a figure this much of the non-ground cells dominates the picture
DOMINANT_CANVAS_SHARE = .25   # ... when it also holds this much of the canvas
GROUND_MARGIN_PERCENT = 10    # ground on every side of a dominant figure: 10% of the shorter side
CONCAVITY_DEPTH = 2           # the ground must reach this far into a figure's bounding box
CONCAVITY_SHARE = .08         # in one pocket of this share of the box (a chamfer is far smaller)
OFFCENTRE_GAP = 3             # ... or opposite margins must differ by this many cells
COMPANION_CELLS = 4           # a figure keeping a dominant one company is a body, not a dot


def ground_margin(h, w):
    """Ground the canvas must show on every side of a dominant figure inside it.

    Two cells, and a tenth of the shorter side once the canvas is big enough for that to be
    more (three from 21 cells on): a two-cell frame around a 29x29 body reads as a border.
    Integer arithmetic, so the plain-Python reading of the certificate matches it exactly.
    """
    return max(2, -(-min(h, w) * GROUND_MARGIN_PERCENT // 100))


def _dilate4(mask):
    grown = np.zeros(mask.shape, dtype=bool)
    grown[1:, :] |= mask[:-1, :]
    grown[:-1, :] |= mask[1:, :]
    grown[:, 1:] |= mask[:, :-1]
    grown[:, :-1] |= mask[:, 1:]
    return grown


def figure_ground_certificate(grid, *, ring_share=.60, rival_ring_share=.25, mass_ratio=1.25,
                              max_ground_parts=4, min_ground_part=4, dominant_body_share=.60,
                              ring_floor=RING_FLOOR,
                              separating_rival_ring=SEPARATING_RIVAL_RING,
                              separating_mass_share=SEPARATING_MASS_SHARE,
                              panel_rules=False):
    """The unique color that reads as the ground under every figure, and how it was read.

    Returns ``(color, path)`` with ``path`` either ``"ring_majority"`` or
    ``"separating_field"``; raises ValueError when no single color qualifies.

    Two clauses hold on both paths for the candidate color C:
      (A) C covers at least ``ring_floor`` (35%) of the border ring;
      (C) C reaches all four sides of the canvas;
      (F) every enclosed pocket of C (a C component that does not touch the
          border) is smaller than C's border-touching mass.

    Path (i), ``ring_majority`` -- C simply owns the frame of the picture:
      (a) C covers at least ``ring_share`` (60%) of the border ring;
      (b) no other color covers more than ``rival_ring_share`` (25%) of the ring;
      (d) C's border-touching components are few and substantial: at most
          ``max_ground_parts`` (4) of them, each of at least ``min_ground_part``
          (4) cells, so a lattice of ground patches and a stranded one-cell
          island in a corner are not a ground;
      (e) the mass of C's border-touching components is at least ``mass_ratio``
          (1.25x) the border-touching mass of any other color;
      (g) when C is not the most frequent color, the two outermost rings are
          entirely C, exactly one other color holds at least as many cells as
          C, and that color's largest component covers at least
          ``dominant_body_share`` (60%) of the non-C cells: one dominant figure
          inside a clean margin, never a stack of panels. Under ``panel_rules``
          the clean two-ring margin gives way to clauses (i)-(j), which ask the
          DOMINANT figure for a proportional margin and company instead, and let
          a smaller figure be cut by the border.

    Path (ii), ``separating_field`` -- figures cut by the border own more of the
    ring than C does, and C is still the one field they all sit on:
      (b') no other color covers more than ``separating_rival_ring`` (30%) of
           the ring, so C is the only color the ring can be read as;
      (d') C's border-touching cells lie in ONE 4-connected component: the
           ground is a single field, not two bands that face each other;
      (e') that component is 4-adjacent to EVERY connected component of non-C
           cells (components taken color-blind, so figures that touch count as
           one): every figure sits on the field, nothing is walled off inside
           another figure or inside a pocket;
      (f') C holds at least ``separating_mass_share`` (30%) of the canvas;
      (g') C's total mass is at least ``mass_ratio`` (1.25x) the total mass of
           any single other color, which is what keeps a body striped into
           full-width bands from reading as a field around panels.

    ``panel_rules`` adds the scene clauses that keep a picture from ALSO reading
    as an inset panel with a drawn border -- "2 is the background and 5 is its
    border" instead of "5 is the ground and the 2 block sits on it". They speak
    about the largest 4-connected component of non-C cells taken color-blind
    (the picture's one big figure), and only when it holds at least
    ``DOMINANT_FIGURE_SHARE`` (60%) of the non-C cells and ``DOMINANT_CANVAS_SHARE``
    (25%) of the canvas, and does not touch the border (a figure the border cuts
    can never be an inset panel):
      (h) a two-color picture needs C over half the canvas, and no figure the
          border leaves whole may stretch across more than ``TWO_COLOR_SPAN``
          (70%) of either side, so a panel inside a border is never the whole of
          a two-color example;
      (i) a dominant figure keeps a ground margin of at least ``ground_margin``
          (2 cells, ``GROUND_MARGIN_PERCENT`` of the shorter side once that is
          more) on EVERY side, so a thin frame look is gone on big canvases;
      (j) when C holds less than ``LOW_GROUND_SHARE`` (55%) of the canvas, the
          picture must show the ground carrying more than that one figure: some
          other figure of a color the dominant figure does not use sits wholly
          outside its bounding box, AND the dominant figure is either non-convex
          with the exterior ground reaching ``CONCAVITY_DEPTH`` (2) cells into
          one pocket of at least ``CONCAVITY_SHARE`` (8%) of its bounding box
          (U, H, T, L, a frame with an open mouth, a union), or sits off-centre
          with opposite margins differing by ``OFFCENTRE_GAP`` (3) cells. A
          chamfered or plain rectangle centred in a uniform margin is therefore
          never a dominant figure over a ground under 55%.

    Components are 4-connected and same-colored on path (i), so figures of
    different colors that touch are different components; figures may touch one
    another and may be cut by the canvas border. Path (i) therefore does not
    claim that C encloses every figure: what (a)-(e) certify is that C, and only
    C, holds the outside of the picture in a few large pieces. Path (ii) makes
    the stronger separation claim instead of the ring majority, and takes the
    figure components color-blind so that a two-color figure is one figure.

    Majority is never required, and neither is a ring majority. Two colors can
    never both qualify: a second color would need 35% of the ring, which makes
    it a rival over 30% for the first. Split canvases, panels, frames hugging
    the border, two colors each owning a side, ground lattices and panel stacks
    fail (A), (C), (F), (a)-(g) or (b')-(g').
    Conservative by design: when in doubt it rejects. No metadata is read.
    """
    raw = np.asarray(grid)
    if raw.ndim != 2 or min(raw.shape) < 2:
        raise ValueError("A figure/ground reading needs a two-dimensional canvas")
    h, w = raw.shape
    symbols, inverse, counts = np.unique(raw, return_inverse=True, return_counts=True)
    if len(symbols) < 2:
        raise ValueError("A single-color canvas has no figure")
    a = inverse.reshape(h, w).astype(np.int64)      # color codes 0..k-1
    values = np.arange(len(symbols))
    ring = np.zeros((h, w), dtype=bool)
    ring[[0, -1], :] = True
    ring[:, [0, -1]] = True
    ring_n = int(ring.sum())
    ring_values, ring_counts = np.unique(a[ring], return_counts=True)
    ring_count = dict(zip(ring_values.tolist(), ring_counts.tolist()))
    labels, n = label_same_color(a)
    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    touching = np.zeros(n + 1, dtype=bool)
    touching[np.unique(labels[ring])] = True
    label_color = np.zeros(n + 1, dtype=np.int64)
    label_color[labels.ravel()] = a.ravel()
    total = dict(zip(values.tolist(), counts.tolist()))

    def ring_majority(color, field, ground_parts, ground_mass):
        if ring_count.get(color, 0) < ring_share * ring_n:                       # (a)
            raise ValueError("No single color owns the border ring")
        if any(k > rival_ring_share * ring_n for v, k in ring_count.items() if v != color):
            raise ValueError("A rival color holds too much of the border ring")  # (b)
        if len(ground_parts) > max_ground_parts or int(sizes[ground_parts].min()) < min_ground_part:
            raise ValueError("The ground breaks into small pieces along the border")  # (d)
        rival_mass = max([int(sizes[1:][touching[1:] & (label_color[1:] == v)].sum())
                          for v in values.tolist() if v != color])
        if ground_mass < mass_ratio * rival_mass:                                # (e)
            raise ValueError("Another color has comparable mass on the border")
        if total[color] <= max(k for v, k in total.items() if v != color):       # (g)
            if not panel_rules:
                # without the scene clauses the clean two-cell margin is the whole guarantee;
                # with them, (i)-(j) ask for a proportional margin around the dominant figure
                # instead, and let smaller figures reach the border
                margin = np.zeros((h, w), dtype=bool)
                margin[[0, 1, -2, -1], :] = True
                margin[:, [0, 1, -2, -1]] = True
                if not field[margin].all():
                    raise ValueError("A dominant figure needs a clean two-cell ground margin")
            rivals = [v for v, k in total.items() if v != color and k >= total[color]]
            if len(rivals) != 1:
                raise ValueError("Several colors outweigh the ground")
            body = max(int(sizes[i]) for i in range(1, n + 1) if label_color[i] == rivals[0])
            if body < dominant_body_share * int((~field).sum()):
                raise ValueError("The colors over the ground do not read as one dominant figure")

    marks_cache = {}

    def figure_marks(color, figures):
        """4-connected components of the non-C cells, color-blind; computed once per color."""
        if color not in marks_cache:
            marks_cache[color] = label_same_color(figures.astype(np.int64))[0]
        return marks_cache[color]

    def separating_field(color, field, ground_parts, ground_mass):
        if any(k > separating_rival_ring * ring_n for v, k in ring_count.items() if v != color):
            raise ValueError("A rival color holds too much of the border ring")  # (b')
        if len(ground_parts) != 1:                                               # (d')
            raise ValueError("The ground on the border is not one field")
        figures = ~field
        marks = figure_marks(color, figures)                                     # (e')
        near = _dilate4(labels == ground_parts[0])
        if set(np.unique(marks[figures]).tolist()) != set(np.unique(marks[figures & near]).tolist()):
            raise ValueError("A figure does not sit on the field")
        if total[color] < separating_mass_share * a.size:                        # (f')
            raise ValueError("The field is too thin to hold the picture")
        if total[color] < mass_ratio * max(k for v, k in total.items() if v != color):
            raise ValueError("Another color has comparable mass on the canvas")  # (g')

    def scene_rules(color, field):
        """Clauses (h)-(j): the picture must not also read as a panel with a border."""
        figures = ~field
        n_figure = int(figures.sum())
        minority = total[color] <= max(k for v, k in total.items() if v != color)
        if len(symbols) > 2 and not minority and n_figure < DOMINANT_CANVAS_SHARE * a.size:
            return                       # too little drawn for any figure to read as a panel
        marks = figure_marks(color, figures)
        ids, blind = np.unique(marks[figures], return_counts=True)
        if len(symbols) == 2:                                                    # (h)
            if total[color] < .50 * a.size:
                raise ValueError("A two-color picture needs a ground over half the canvas")
            for i in ids.tolist():
                piece = marks == i
                if (piece & ring).any():
                    continue             # a figure the border cuts is not an inset panel
                rows, cols = np.nonzero(piece)
                if (int(rows.max() - rows.min()) + 1 > TWO_COLOR_SPAN * h
                        or int(cols.max() - cols.min()) + 1 > TWO_COLOR_SPAN * w):
                    raise ValueError("A two-color figure stretches across the canvas")
        # a ground the figures outnumber has to be the ground of ONE dominant figure that sits
        # inside the canvas; that is what the two-cell margin used to say, and (i)-(j) now say
        mass = int(blind.max())
        if mass < DOMINANT_FIGURE_SHARE * n_figure or mass < DOMINANT_CANVAS_SHARE * a.size:
            if minority:
                raise ValueError("A minority ground carries no one dominant figure")
            return                       # no single figure dominates: nothing reads as a panel
        body = marks == int(ids[int(np.argmax(blind))])
        if (body & ring).any():
            if minority:
                raise ValueError("A dominant figure over a minority ground must sit inside the canvas")
            return                       # a figure the border cuts is not an inset panel
        rows, cols = np.nonzero(body)
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        margins = (r0, h - 1 - r1, c0, w - 1 - c1)
        if min(margins) < ground_margin(h, w):                                   # (i)
            raise ValueError("The ground margin around a dominant figure reads as a border")
        if total[color] >= LOW_GROUND_SHARE * a.size:
            return                       # the ground is plainly the ground: majority and around
        box = np.zeros((h, w), dtype=bool)                                       # (j)
        box[r0:r1 + 1, c0:c1 + 1] = True
        body_colors = set(np.unique(a[body]).tolist())
        for i, k in zip(ids.tolist(), blind.tolist()):
            piece = marks == i
            if (k >= COMPANION_CELLS and not (piece & box).any()
                    and set(np.unique(a[piece]).tolist()) - body_colors):
                break                    # a body, not a texture dot, sits on the ground beside it
        else:
            raise ValueError("A thin ground carries one figure only")
        if (abs(margins[0] - margins[1]) >= OFFCENTRE_GAP
                or abs(margins[2] - margins[3]) >= OFFCENTRE_GAP):
            return                       # off-centre: the margin cannot read as a border
        exterior = np.isin(labels, np.nonzero(touching[1:] & (label_color[1:] == color))[0] + 1)
        sub = (exterior & box)[r0:r1 + 1, c0:c1 + 1]
        bh, bw = sub.shape
        pockets, _ = label_same_color(sub.astype(np.int64))
        for i in np.unique(pockets[sub]).tolist() if sub.any() else []:
            cells = np.argwhere(pockets == i)
            depth = max(min(int(r) + 1, bh - int(r), int(c) + 1, bw - int(c)) for r, c in cells)
            if depth >= CONCAVITY_DEPTH and len(cells) >= CONCAVITY_SHARE * bh * bw:
                return                   # the ground flows into the figure: not a panel either
        raise ValueError("A centred convex figure over a thin ground reads as a panel")

    winners, reasons = [], []
    for color in values.tolist():
        if ring_count.get(color, 0) < ring_floor * ring_n:                       # (A)
            continue
        field = a == color
        if not (field[0].any() and field[-1].any() and field[:, 0].any() and field[:, -1].any()):
            reasons.append("The ground does not reach every side")               # (C)
            continue
        ground_parts = np.nonzero(touching[1:] & (label_color[1:] == color))[0] + 1
        ground_mass = int(sizes[ground_parts].sum())
        pockets = sizes[1:][(~touching[1:]) & (label_color[1:] == color)]
        if len(pockets) and int(pockets.max()) >= ground_mass:                   # (F)
            reasons.append("An enclosed pocket outweighs the visible ground")
            continue
        for name, rule in (("ring_majority", ring_majority), ("separating_field", separating_field)):
            try:
                rule(color, field, ground_parts, ground_mass)
                if panel_rules:
                    scene_rules(color, field)
            except ValueError as error:
                reasons.append(str(error))
            else:
                winners.append((color, name))
                break
    if len(winners) != 1:
        if len(winners) > 1:
            raise ValueError("Two colors read as the ground")
        # every clause that turned a candidate away, so the reading can be read back
        raise ValueError("; ".join(dict.fromkeys(reasons)) if reasons
                         else "No color owns enough of the border ring")
    color, path = winners[0]
    winner = symbols[color]
    return (winner.item() if hasattr(winner, "item") else winner), path


def figure_ground_field(grid, **kwargs):
    """The unique color that reads as the ground; see ``figure_ground_certificate``."""
    return figure_ground_certificate(grid, **kwargs)[0]
