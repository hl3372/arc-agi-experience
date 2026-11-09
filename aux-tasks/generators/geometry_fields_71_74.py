"""Calm fields used only by the diamond/parallel-line tasks.

Colors and field family are selected before rendering retries. These are broad
rectangular washes, bands and coarse checks, never isolated palette-padding dots.
"""
from __future__ import annotations

import numpy as np


def _panel_seats(free, n, rng=None):
    windows = np.argwhere(free[:-1, :-1] & free[1:, :-1] & free[:-1, 1:] & free[1:, 1:])
    if rng is not None:
        rng.shuffle(windows)

    def search(start, selected):
        if len(selected) == n:
            return selected
        for i in range(start, len(windows)):
            r, c = map(int, windows[i])
            if all(abs(r - pr) >= 2 or abs(c - pc) >= 2 for pr, pc in selected):
                found = search(i + 1, selected + [(r, c)])
                if found is not None:
                    return found
        return None

    return search(0, [])


def field_plan(rng, h: int, w: int, *, drawing_diamonds: bool = False, free=None):
    kinds = ["solid", "panels", "bands", "checks"] if drawing_diamonds else ["solid", "panels"]
    weights = [.55, .22, .13, .10] if drawing_diamonds else [.75, .25]
    free = np.ones((h, w), bool) if free is None else free
    cap = min(3, 1 + h * w // 100)
    panel_cap = next((n for n in range(cap, 0, -1) if _panel_seats(free, n) is not None), 0)
    if not panel_cap:
        index = kinds.index("panels")
        kinds.pop(index)
        weights.pop(index)
    kind = str(rng.choice(kinds, p=np.asarray(weights) / sum(weights)))
    if kind == "panels":
        cap = panel_cap
    if kind == "bands":
        cap = min(cap, max(h, w) // 2 - 1)
    if kind == "checks":
        cap = min(cap, 2)
    n = 0 if kind == "solid" else int(rng.integers(1, cap + 1))
    return kind, n


def render_field(rng, h: int, w: int, bg: int, accents: list[int], kind: str, *, avoid=None):
    a = np.full((h, w), bg, dtype=int)
    if kind == "solid":
        return a
    if kind == "panels":
        used = np.zeros((h, w), bool) if avoid is None else avoid.copy()
        seats = _panel_seats(~used, len(accents), rng)
        if seats is None:
            raise RuntimeError("The planned panel capacity was not available")
        for r, c in seats:
            used[r:r + 2, c:c + 2] = True
        for color, (r, c) in zip(accents, seats):
            top, left, bottom, right = r, c, r + 2, c + 2
            for step in range(h + w):
                expansions = []
                if bottom - top < max(2, h // 3):
                    if top > 0 and not used[top - 1, left:right].any():
                        expansions.append((top - 1, left, bottom, right))
                    if bottom < h and not used[bottom, left:right].any():
                        expansions.append((top, left, bottom + 1, right))
                if right - left < max(2, w // 3):
                    if left > 0 and not used[top:bottom, left - 1].any():
                        expansions.append((top, left - 1, bottom, right))
                    if right < w and not used[top:bottom, right].any():
                        expansions.append((top, left, bottom, right + 1))
                if not expansions or (step > 0 and rng.random() < .12):
                    break
                top, left, bottom, right = expansions[int(rng.integers(len(expansions)))]
                used[top:bottom, left:right] = True
            a[top:bottom, left:right] = color
    elif kind == "bands":
        axes = [axis for axis, side in enumerate((h, w)) if side >= 2 * (len(accents) + 1)]
        axis = int(rng.choice(axes))
        side = (h, w)[axis]
        colors = [bg, *accents]
        rng.shuffle(colors)
        widths = np.full(len(colors), 2)
        for _ in range(side - int(widths.sum())):
            widths[int(rng.integers(len(colors)))] += 1
        start = 0
        for color, width in zip(colors, widths):
            if axis == 0:
                a[start:start + width, :] = color
            else:
                a[:, start:start + width] = color
            start += int(width)
    else:
        block = int(rng.integers(2, max(2, min(h, w) // 4) + 1))
        phase_r, phase_c = map(int, rng.integers(0, block, size=2))
        rr, cc = np.indices((h, w))
        colors = np.asarray([bg, *accents])
        a = colors[((rr + phase_r) // block + (cc + phase_c) // block) % len(colors)]
    return a


def field_metadata(field, bg, kind, accents):
    return {
        "texture_kind": kind,
        "texture_group": "solid" if kind == "solid" else "structured",
        "texture_colors": [int(x) for x in accents],
        "texture_named": False,
        "texture_coverage": float(np.mean(np.isin(np.asarray(field), accents))),
    }
