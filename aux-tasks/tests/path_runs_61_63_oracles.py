"""Independent prompt readers using breadth-first distances on the visible grid.

They do not call the production route builder, path tracer, or solver.
"""
from collections import deque
import re

import numpy as np


def solve(grid, prompt, task_id):
    a = np.asarray(grid)

    def color(pattern):
        value = re.search(pattern, prompt).group(1)
        return value if a.dtype.kind in "OUS" else int(value)

    path_color = color(r"line of color (\w+) cells")
    n = int(re.search(r"(?:first|last) (\d+) cells?", prompt).group(1))
    cells = set(map(tuple, np.argwhere(a == path_color)))
    marker = None
    if task_id != "61":
        marker_color = color(r"single start cell of color (\w+)")
        marks = np.argwhere(a == marker_color)
        assert len(marks) == 1
        marker = tuple(marks[0])
        cells.add(marker)
    links = {p: {q for q in cells if abs(p[0]-q[0])+abs(p[1]-q[1]) == 1} for p in cells}
    endpoints = [p for p in cells if len(links[p]) == 1]
    assert len(endpoints) == 2 and all(1 <= len(ns) <= 2 for ns in links.values())
    if marker is not None:
        assert marker in endpoints
    start = marker if marker is not None else min(endpoints)
    distance = {start: 0}
    frontier = deque([start])
    while frontier:
        p = frontier.popleft()
        for q in links[p]:
            if q not in distance:
                distance[q] = distance[p]+1
                frontier.append(q)
    assert len(distance) == len(cells)
    assert set(distance.values()) == set(range(len(cells)))
    length = len(cells)
    assert 1 <= n <= (length//2 if task_id == "61" else length-1)
    ordered = sorted(cells, key=distance.get)
    chosen = {p for p in cells if distance[p] >= length-n}
    if task_id == "61":
        chosen |= {p for p in cells if distance[p] < n}
    if task_id == "62":
        result = "[" + ",".join(f"({r},{c})" for r, c in sorted(chosen, key=distance.get)) + "]"
    else:
        highlight = color(r"with color (\w+)")
        result = a.copy()
        for p in chosen:
            result[p] = highlight
        result = result.tolist()
    directions = [(b[0]-p[0], b[1]-p[1]) for p, b in zip(ordered, ordered[1:])]
    stats = {
        "path_length": length, "n": n, "path_occupancy": length/a.size,
        "turns": sum(x != y for x, y in zip(directions, directions[1:])),
        "path": ordered, "chosen": sorted(chosen),
    }
    # Give the shortcut the true far endpoint, then let it choose the N closest
    # cells by geometric distance. This is a generous local-reader comparison;
    # merely identifying the far endpoint is not a path-traversal probe.
    far = ordered[-1]
    guessed = sorted(ordered[1:] if marker is not None else ordered,
                     key=lambda p: (abs(p[0]-far[0])+abs(p[1]-far[1]), p))[:n]
    if task_id == "61":
        near = ordered[0]
        guessed += sorted(ordered, key=lambda p: (abs(p[0]-near[0])+abs(p[1]-near[1]), p))[:n]
        stats["manhattan_shortcut_correct"] = set(guessed) == chosen
    elif task_id == "63":
        stats["manhattan_shortcut_correct"] = set(guessed) == chosen
    else:
        stats["manhattan_shortcut_correct"] = guessed[::-1] == ordered[-n:]
    return result, stats
