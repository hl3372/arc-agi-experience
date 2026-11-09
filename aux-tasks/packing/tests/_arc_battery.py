"""A fixed battery of grids for property tests of the ARC library (tests only)."""
from __future__ import annotations

import functools

import numpy as np

import pv2.pictures  # noqa: F401
from pv2.arc import gen
from pv2.arc.grids import DT
from registry import generate


@functools.lru_cache(maxsize=1)
def battery() -> list[np.ndarray]:
    r = np.random.default_rng(20260929)
    out = []
    # borrowed pictures, small and large, recoloured so every colour occurs somewhere
    tasks = gen.SOURCES
    for i in range(36):
        tid = tasks[int(r.integers(len(tasks)))]
        g = np.asarray(generate(tid, int(r.integers(2 ** 31)))["input"], dtype=DT)
        if g.size > 400:
            h = int(r.integers(3, 16))
            w = int(r.integers(3, 16))
            y = int(r.integers(0, g.shape[0] - min(h, g.shape[0]) + 1))
            x = int(r.integers(0, g.shape[1] - min(w, g.shape[1]) + 1))
            g = g[y:y + h, x:x + w].copy()
        lut = np.array(r.permutation(10), dtype=DT)
        out.append(lut[g])
    # drawn scenes with every feature
    for must, marker, pairs in [(("hole",), False, False), (("rect",), False, False), (("sizes",), False, False),
                                (("diag",), False, False), ((), True, False), ((), True, True),
                                (("hole", "rect"), False, False)]:
        for _ in range(3):
            g = None
            while g is None:
                pal = [int(c) for c in r.permutation(10)[:int(r.integers(2, 7))]]
                g = gen.draw_scene(r, int(r.integers(6, 16)), int(r.integers(6, 16)), pal, must=must,
                                   marker=int(r.permutation([c for c in range(10) if c not in pal])[0]) if marker else None,
                                   pairs=pairs)
            out.append(g)
    # objects of every size 1..20 under 4-connectivity, and pairs meeting at a corner whose
    # 8-connected size differs (sizes up to 20 both ways)
    for colour_mode in range(2):
        g = np.zeros((30, 30), DT)
        row = 0
        for n in range(1, 21):
            c = 1 + (n % 9 if colour_mode else 0)
            if row + 1 >= 30:
                break
            g[row, 0:n] = c
            if n <= 8:
                # a second bar of 20 - n - 2 cells touching the first at a corner (8-connected: 20 - 2)
                m = min(12, 18 - n)
                g[row + 1, n + 1:n + 1 + m] = c
                row += 3
            else:
                row += 2
        out.append(g)
    g = np.zeros((30, 30), DT)
    for k, n in enumerate(range(11, 21)):
        y, x = divmod(k, 2)
        g[y * 3 + 1, x * 15:x * 15 + min(n, 14)] = 2 + k % 7
        if n > 14:
            g[y * 3 + 2, x * 15 + 13 - (n - 15):x * 15 + 14] = 2 + k % 7
    out.append(g)
    for g in list(out[-3:]):
        for _ in range(2):
            out.append(np.array(r.permutation(10), DT)[g])
    # corner-touching pairs whose 8-connected sizes are 11..20 (4-connected parts are smaller)
    g = np.full((30, 30), 5, DT)
    row = 0
    for total in range(11, 21):
        a_len = total // 2
        b_len = total - a_len
        g[row, 0:a_len] = 7
        g[row + 1, a_len:a_len + b_len] = 7
        row += 3
    out.append(g)
    out.append(np.array(r.permutation(10), DT)[g])
    for f in range(2, 11):
        small = np.array(r.integers(0, 10, size=(max(1, 30 // (f * 5)) + 1, 2)), DT)
        big = np.repeat(np.repeat(small, f, 0), f, 1)[:30, :30]
        if big.shape[0] % f == 0 and big.shape[1] % f == 0:
            out.append(big)
        out.append(np.repeat(np.repeat(np.array([[1, 2], [3, 4]], DT), f, 0), f, 1)[:30, :30])
    # edge cases
    out += [np.array([[3]], DT), np.array([[1, 2, 1, 2, 5]], DT), np.array([[4], [4], [7]], DT),
            np.full((3, 3), 6, DT), np.array([[1, 2], [2, 1]], DT), np.array([[0, 0, 1], [0, 1, 1]], DT),
            np.tile(np.array([[1, 2, 3], [4, 5, 6]], DT), (3, 2)),
            np.repeat(np.repeat(np.array([[1, 2], [3, 0]], DT), 3, 0), 3, 1),
            np.array(r.integers(0, 10, size=(12, 17)), DT), np.array(r.integers(0, 3, size=(30, 30)), DT)]
    return out
