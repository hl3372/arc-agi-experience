"""Shared orthogonal route vocabulary for the path tasks (60 and 61-63).

Every route this module returns is an *induced* path: one cell wide,
4-connected, and with no two non-consecutive cells 4-adjacent.  That is the
property the drawn line needs to stay readable -- it gives exactly two cells
with a single path neighbour (the ends) and exactly one traversal between
them, which is what both tasks' re-solvers reconstruct from the picture.

Two vocabularies live here.  The six *templated* families draw the familiar
shapes (a straight bar, an elbow, a U, a staircase, a serpentine, a spiral).
The four *irregular* families draw routes no template predicts: a random walk
of short runs, a wandering line of long runs with detours and reversals, a
meander that crosses the canvas with irregular turns, and an unequal zigzag.
They are grown cell by cell with backtracking, so a dead end costs a retry
inside the walk rather than a rejected scene.
"""
from __future__ import annotations

import numpy as np

TEMPLATE_FAMILIES = ("straight", "elbow", "U", "staircase", "serpentine", "spiral")
IRREGULAR_FAMILIES = ("random_walk", "wander", "meander", "zigzag")
FAMILIES = TEMPLATE_FAMILIES + IRREGULAR_FAMILIES

# Smallest construction box each irregular family is grown in. Both tasks keep
# their own floors for the templated families; those differ by task.
IRREGULAR_FLOORS = {family: 4 for family in IRREGULAR_FAMILIES}
# An irregular route that barely bends is a template by another name.
MIN_TURNS = {"random_walk": 4, "wander": 3, "meander": 4, "zigzag": 4}
# Share of cells allowed to touch a far part of the line diagonally, so that a
# wandering route still reads as one line rather than as a patch.
MAX_DIAGONAL_CONTACT = .15

STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def line(start, end):
    """The cells strictly after ``start`` on the orthogonal segment to ``end``."""
    y, x = start
    yy, xx = end
    if y != yy and x != xx:
        raise ValueError("Route segment is not orthogonal")
    dy, dx = int(np.sign(yy-y)), int(np.sign(xx-x))
    return [(y+dy*k, x+dx*k) for k in range(1, abs(yy-y)+abs(xx-x)+1)]


# ---------------------------------------------------------------------------
# invariants (also used by the tests)
# ---------------------------------------------------------------------------

def turns(route):
    steps = [(b[0]-a[0], b[1]-a[1]) for a, b in zip(route, route[1:])]
    return sum(a != b for a, b in zip(steps, steps[1:]))


def is_simple_line(route):
    """One cell wide, 4-connected, two ends: the readability contract."""
    cells = set(route)
    if len(cells) != len(route) or len(route) < 2:
        return False
    index = {cell: i for i, cell in enumerate(route)}
    for i, (r, c) in enumerate(route):
        if i and abs(route[i-1][0]-r)+abs(route[i-1][1]-c) != 1:
            return False
        for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            j = index.get(q)
            if j is not None and abs(j-i) != 1:
                return False
    return True


def diagonal_contact_share(route):
    """Cells with a diagonal neighbour far away along the line.

    The two cells either side of a corner are always diagonal neighbours, so
    only index distances of three or more count as the line touching itself.
    """
    index = {cell: i for i, cell in enumerate(route)}
    touching = 0
    for i, (r, c) in enumerate(route):
        if any(abs(index.get(q, i)-i) >= 3
               for q in ((r-1, c-1), (r-1, c+1), (r+1, c-1), (r+1, c+1))):
            touching += 1
    return touching/len(route)


def acceptable(route, family):
    """Reject an irregular route that collapsed into a template or a tangle."""
    if family not in IRREGULAR_FAMILIES:
        return True
    return (turns(route) >= MIN_TURNS[family]
            and diagonal_contact_share(route) <= MAX_DIAGONAL_CONTACT)


# ---------------------------------------------------------------------------
# irregular routes
# ---------------------------------------------------------------------------

def _extendable(q, cells, h, w):
    """A cell the line may enter: inside, free, and touching the line once.

    The last clause is the one-cell-wide contract: a cell with two path
    neighbours would give the drawn line a junction or a second traversal.
    """
    r, c = q
    if not (0 <= r < h and 0 <= c < w) or q in cells:
        return False
    return sum(p in cells for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1))) == 1


def _last_sign(route, axis):
    """Direction of the most recent move along this axis; 0 if there was none."""
    for i in range(len(route)-1, 0, -1):
        delta = route[i][axis]-route[i-1][axis]
        if delta:
            return 1 if delta > 0 else -1
    return 0


def _spec(rng, family, h, w):
    """Run lengths and turn preferences: where a family's character lives."""
    if family == "random_walk":
        longest = int(np.clip(max(h, w)//4+2, 2, 5))
        return (lambda step: int(rng.integers(1, longest+1)),
                lambda route, step: rng.random())
    if family == "wander":
        axis = 0 if h >= w else 1
        if rng.random() < .35:
            axis = 1-axis
        longest = max(3, int(round((h if axis == 0 else w)*.55)))

        def run(step):
            # Long along the line's own direction, one or two cells aside.
            return int(rng.integers(3, longest+1)) if step[axis] else int(rng.integers(1, 3))

        def turn(route, step):
            if not step[axis]:
                return .5
            keep = (1 if step[axis] > 0 else -1) == _last_sign(route, axis)
            return 1.5*float(rng.random() < (.80 if keep else .20))
        return run, turn
    if family == "meander":
        # Cross the canvas: start in one region, lean towards the opposite one.
        goal = (h-1 if rng.integers(2) else 0, w-1 if rng.integers(2) else 0)

        def turn(route, step):
            head = route[-1]
            here = abs(head[0]-goal[0])+abs(head[1]-goal[1])
            there = abs(head[0]+step[0]-goal[0])+abs(head[1]+step[1]-goal[1])
            return 1.5*float(rng.random() < (.74 if there < here else .26))
        return (lambda step: int(rng.integers(1, 5))), turn
    longest = int(np.clip(max(h, w)//3+2, 3, 8))

    def run(step):
        # Deliberately unequal: a couple of cells or a long reach, rarely both.
        return int(rng.integers(1, 3)) if rng.random() < .45 else int(rng.integers(3, longest+1))

    def turn(route, step):
        axis = 0 if step[0] else 1
        keep = (1 if step[axis] > 0 else -1) == _last_sign(route, axis)
        return 1.5*float(rng.random() < (.70 if keep else .30))
    return run, turn


def _touches_far(q, index, position):
    """Would this cell sit diagonally against a distant part of the line?"""
    r, c = q
    return any(index.get(p, position) <= position-3
               for p in ((r-1, c-1), (r-1, c+1), (r+1, c-1), (r+1, c+1)))


def _options(rng, run, turn, route, direction, left, index, h, w):
    """Legal next steps, worst first: the walk pops the best one off the end."""
    head = route[-1]
    position = len(route)
    scored = []
    for step in STEPS:
        if direction is not None and step == (-direction[0], -direction[1]):
            continue
        q = (head[0]+step[0], head[1]+step[1])
        if not _extendable(q, index, h, w):
            continue
        # Folding back against itself is allowed but discouraged, so the line
        # keeps its distance from itself and stays one readable stroke.
        near = -2.5 if _touches_far(q, index, position) else .0
        if step == direction:
            # Straight past the end of a run stays available, but only as the
            # last resort that keeps the walk out of a dead end.
            scored.append((near+(4.0 if left > 0 else -1.0), step, left-1))
        else:
            scored.append((near+(.0 if left > 0 else 2.0)+turn(route, step)+.001*rng.random(),
                           step, run(step)-1))
    scored.sort(key=lambda option: option[0])
    return [(step, remaining) for _, step, remaining in scored]


def _walk(rng, h, w, start, target, run, turn):
    """Grow an induced path, backtracking out of dead ends, and keep the best."""
    route, index = [start], {start: 0}
    stack = [_options(rng, run, turn, route, None, 0, index, h, w)]
    best = list(route)
    budget = 24*target+200
    while stack and budget > 0:
        budget -= 1
        if not stack[-1]:
            stack.pop()
            if stack:
                index.pop(route.pop())
            continue
        step, left = stack[-1].pop()
        head = (route[-1][0]+step[0], route[-1][1]+step[1])
        if not _extendable(head, index, h, w):
            continue
        index[head] = len(route)
        route.append(head)
        if len(route) > len(best):
            best = list(route)
            if len(best) >= target:
                return best
        stack.append(_options(rng, run, turn, route, step, left, index, h, w))
    return best


def irregular_route(rng, h, w, family):
    """An irregular route filling a share of its box comparable to a template."""
    target = max(4, int(round(h*w*float(rng.uniform(.24, .40)))))
    best = []
    for _ in range(3):
        run, turn = _spec(rng, family, h, w)
        if family == "meander":
            # Start on a border so the crossing has somewhere to come from.
            start = ((int(rng.integers(h)), 0 if rng.integers(2) else w-1) if rng.integers(2)
                     else (0 if rng.integers(2) else h-1, int(rng.integers(w))))
        else:
            start = (int(rng.integers(h)), int(rng.integers(w)))
        route = _walk(rng, h, w, start, target, run, turn)
        if len(route) > len(best):
            best = route
        if len(best) >= target:
            break
    return best


# ---------------------------------------------------------------------------
# templated routes
# ---------------------------------------------------------------------------

def template_route(rng, h, w, family):
    route = [(0, 0)]
    if family == "straight":
        return [(0, x) for x in range(w)] if w >= h else [(y, 0) for y in range(h)]
    if family == "elbow":
        targets = [(0, w-1), (h-1, w-1)]
    elif family == "U":
        targets = [(h-1, 0), (h-1, w-1), (0, w-1)]
    elif family == "staircase":
        while route[-1] != (h-1, w-1):
            y, x = route[-1]
            horizontal = y == h-1 or (x < w-1 and rng.integers(2))
            end = (y, min(w-1, x+int(rng.integers(1, 6)))) if horizontal else (min(h-1, y+int(rng.integers(1, 6))), x)
            route.extend(line(route[-1], end))
        return route
    elif family == "serpentine":
        targets = []
        rows = list(range(0, h, int(rng.integers(2, min(4, h-1)+1))))
        for i, y in enumerate(rows):
            if i:
                targets.append((y, 0 if i % 2 == 0 else w-1))
            targets.append((y, w-1 if i % 2 == 0 else 0))
    else:
        targets = []
        top, bottom, left, right = 0, h-1, 0, w-1
        while top <= bottom and left <= right:
            targets.append((top, right))
            top += 2
            if top > bottom:
                break
            targets.append((bottom, right))
            right -= 2
            if left > right:
                break
            targets.append((bottom, left))
            bottom -= 2
            if top > bottom:
                break
            targets.append((top, left))
            left += 2
    for end in targets:
        route.extend(line(route[-1], end))
    return route


def route(rng, h, w, family):
    """The route vocabulary both path tasks build their scenes from."""
    if family in IRREGULAR_FAMILIES:
        return irregular_route(rng, h, w, family)
    return template_route(rng, h, w, family)


def shares(template_weights, irregular_share=.40):
    """Rescale six template weights to leave ``irregular_share`` for the rest.

    The templated families keep their proportions to one another; the four
    irregular families split the remainder.
    """
    template = np.asarray(template_weights, float)
    template = template/template.sum()*(1-irregular_share)
    irregular = np.asarray([.28, .24, .28, .20])*irregular_share
    return tuple(map(float, np.concatenate([template, irregular])))
