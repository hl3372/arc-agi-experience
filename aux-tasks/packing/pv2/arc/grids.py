"""Grid analysis shared by the ARC transform library, the rival search and the scalar rules.

Grids are handled as ``numpy.int8`` arrays inside the library (``arr``), as lists of lists at the
Part boundary.  ``Info`` wraps one array and caches what several rules ask for: the colour
histogram, the background (strict plurality colour, or None), same-colour objects under 4- or
8-connectivity, and a hashable key.

Background.  Library rules that need a background use the most frequent colour when it is a strict
plurality; the rule is undefined (returns None) otherwise.  Generation only uses such rules on
plain-ground pictures (``pv2.pictures.is_solid``), where that colour covers at least half the grid.
The prompts never name or define it (C7): rule descriptions say "background" and leave the
inference to the reader.
"""
from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np

DT = np.int8
MAX_SIDE = 30

_OFF4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_OFF8 = _OFF4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def arr(grid) -> np.ndarray:
    """A C-contiguous int8 copy of a grid (list of lists or array)."""
    a = np.ascontiguousarray(np.asarray(grid, dtype=DT))
    if a.ndim != 2 or a.shape[0] == 0 or a.shape[1] == 0:
        raise ValueError("a grid is a non-empty 2-D array")
    return a


def to_list(a: np.ndarray) -> list[list[int]]:
    return [[int(v) for v in row] for row in a]


def key_of(a: np.ndarray) -> tuple:
    return (a.shape[0], a.shape[1], np.ascontiguousarray(a, dtype=DT).tobytes())


def fits(h: int, w: int) -> bool:
    return 1 <= h <= MAX_SIDE and 1 <= w <= MAX_SIDE


class Comp(NamedTuple):
    """One object: a same-colour connected group (or a multicolour blob when ``color`` is -1)."""
    color: int
    cells: tuple            # flat indices r * w + c
    size: int
    r0: int
    c0: int
    r1: int                 # inclusive
    c1: int

    @property
    def bbox_area(self) -> int:
        return (self.r1 - self.r0 + 1) * (self.c1 - self.c0 + 1)

    @property
    def bbox_dims(self) -> tuple[int, int]:
        return (self.r1 - self.r0 + 1, self.c1 - self.c0 + 1)


class Info:
    """Cached analysis of one grid.  The array must not be modified after wrapping."""

    __slots__ = ("a", "h", "w", "_hist", "_bg", "_key", "_flat", "_comps", "_colors", "_extra")

    def __init__(self, a: np.ndarray):
        self.a = a
        self.h, self.w = a.shape
        self._hist = None
        self._bg = -2
        self._key = None
        self._flat = None
        self._comps = {}
        self._colors = None
        self._extra = {}

    @classmethod
    def of(cls, grid) -> "Info":
        return cls(arr(grid))

    @property
    def shape(self) -> tuple[int, int]:
        return (self.h, self.w)

    @property
    def key(self) -> tuple:
        if self._key is None:
            self._key = (self.h, self.w, self.a.tobytes())
        return self._key

    @property
    def hist(self) -> np.ndarray:
        if self._hist is None:
            self._hist = np.bincount(self.a.ravel(), minlength=10)
        return self._hist

    @property
    def colors(self) -> tuple[int, ...]:
        if self._colors is None:
            self._colors = tuple(int(c) for c in np.nonzero(self.hist)[0])
        return self._colors

    @property
    def bg(self) -> Optional[int]:
        """The strict-plurality colour, or None on a tie."""
        if self._bg == -2:
            h = self.hist
            top = int(h.max())
            winners = np.nonzero(h == top)[0]
            self._bg = int(winners[0]) if len(winners) == 1 else None
        return self._bg

    @property
    def flat(self) -> list[int]:
        if self._flat is None:
            self._flat = self.a.ravel().tolist()
        return self._flat

    def least(self) -> Optional[int]:
        """The strictly least frequent colour present, or None on a tie."""
        h = self.hist
        present = [(int(h[c]), c) for c in self.colors]
        present.sort()
        if len(present) >= 2 and present[0][0] == present[1][0]:
            return None
        return present[0][1]

    def objects(self, conn: int) -> Optional[list[Comp]]:
        """Same-colour conn-connected groups of non-background cells (singletons included);
        None when the background is undefined."""
        k = ("obj", conn)
        if k not in self._comps:
            bg = self.bg
            self._comps[k] = None if bg is None else _label(self, conn, bg, by_color=True)
        return self._comps[k]

    def blobs(self, conn: int) -> Optional[list[Comp]]:
        """Connected groups of non-background cells ignoring colour (colour -1)."""
        k = ("blob", conn)
        if k not in self._comps:
            bg = self.bg
            self._comps[k] = None if bg is None else _label(self, conn, bg, by_color=False)
        return self._comps[k]

    def memo(self, key, fn):
        """Cache any derived value on this grid."""
        if key not in self._extra:
            self._extra[key] = fn()
        return self._extra[key]


def _label(info: Info, conn: int, bg: int, by_color: bool) -> list[Comp]:
    h, w = info.h, info.w
    flat = info.flat
    offs = _OFF4 if conn == 4 else _OFF8
    seen = [False] * (h * w)
    comps = []
    for start in range(h * w):
        v = flat[start]
        if seen[start] or v == bg:
            continue
        seen[start] = True
        stack = [start]
        cells = []
        r0 = c0 = 10 ** 6
        r1 = c1 = -1
        while stack:
            p = stack.pop()
            cells.append(p)
            r, c = divmod(p, w)
            if r < r0:
                r0 = r
            if r > r1:
                r1 = r
            if c < c0:
                c0 = c
            if c > c1:
                c1 = c
            for dr, dc in offs:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    q = rr * w + cc
                    if not seen[q]:
                        fq = flat[q]
                        if (fq == v) if by_color else (fq != bg):
                            seen[q] = True
                            stack.append(q)
        cells.sort()
        comps.append(Comp(v if by_color else -1, tuple(cells), len(cells), r0, c0, r1, c1))
    return comps


def neighbour_count_same(a: np.ndarray, conn: int) -> np.ndarray:
    """For every cell, how many of its conn-neighbours (inside the grid) share its colour."""
    h, w = a.shape
    out = np.zeros((h, w), dtype=np.int16)
    offs = _OFF4 if conn == 4 else _OFF8
    for dr, dc in offs:
        src_r = slice(max(0, dr), h + min(0, dr))
        dst_r = slice(max(0, -dr), h + min(0, -dr))
        src_c = slice(max(0, dc), w + min(0, dc))
        dst_c = slice(max(0, -dc), w + min(0, -dc))
        out[dst_r, dst_c] += (a[dst_r, dst_c] == a[src_r, src_c])
    return out


def inside_count(h: int, w: int, conn: int) -> np.ndarray:
    """How many conn-neighbours every cell has inside the grid."""
    ones = np.ones((h, w), dtype=DT)
    return neighbour_count_same(ones, conn)


def near_mask(mask: np.ndarray, conn: int, t: int = 1) -> np.ndarray:
    """Cells within t steps (4: city-block, 8: king moves) of a True cell, the cells themselves
    included."""
    out = mask.copy()
    h, w = mask.shape
    for _ in range(t):
        cur = out.copy()
        offs = _OFF4 if conn == 4 else _OFF8
        for dr, dc in offs:
            src_r = slice(max(0, dr), h + min(0, dr))
            dst_r = slice(max(0, -dr), h + min(0, -dr))
            src_c = slice(max(0, dc), w + min(0, dc))
            dst_c = slice(max(0, -dc), w + min(0, -dc))
            cur[dst_r, dst_c] |= out[src_r, src_c]
        out = cur
    return out


def border_reachable(open_mask: np.ndarray) -> np.ndarray:
    """Cells of ``open_mask`` 4-connected (through open cells) to the grid border."""
    h, w = open_mask.shape
    reach = np.zeros_like(open_mask)
    stack = []
    for r in range(h):
        for c in (0, w - 1):
            if open_mask[r, c] and not reach[r, c]:
                reach[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if open_mask[r, c] and not reach[r, c]:
                reach[r, c] = True
                stack.append((r, c))
    while stack:
        r, c = stack.pop()
        for dr, dc in _OFF4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and open_mask[rr, cc] and not reach[rr, cc]:
                reach[rr, cc] = True
                stack.append((rr, cc))
    return reach


def regions(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """4-connected groups of True cells."""
    h, w = mask.shape
    seen = np.zeros_like(mask)
    out = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack = [(r, c)]
                cells = []
                while stack:
                    p = stack.pop()
                    cells.append(p)
                    pr, pc = p
                    for dr, dc in _OFF4:
                        rr, cc = pr + dr, pc + dc
                        if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                            seen[rr, cc] = True
                            stack.append((rr, cc))
                out.append(cells)
    return out


def is_solid(a: np.ndarray) -> bool:
    """Same test as ``pv2.pictures.is_solid`` (a plain ground), on an array."""
    info = Info(a) if not isinstance(a, Info) else a
    h = info.hist
    ground = int(np.argmax(h))
    if h[ground] < 0.5 * info.h * info.w:
        return False
    same = neighbour_count_same(info.a, 4)
    return bool(np.all((info.a == ground) | (same > 0)))
