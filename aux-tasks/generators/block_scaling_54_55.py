"""Relative enlargement and requested block compression of coherent patterns."""
from __future__ import annotations

from functools import lru_cache
import numpy as np

from generators.tiling_44_45 import _content

# user floor 2026-09-15: small sides are often ambiguous
MIN_SIDE = 5


def compression_factors(grid):
    a = np.asarray(grid)
    h, w = a.shape
    result = []
    for n in range(2, min(h, w)+1):
        if h % n or w % n:
            continue
        blocks = a.reshape(h//n, n, w//n, n)
        if np.all(blocks == blocks[:, :1, :, :1]):
            result.append(n)
    return result


def _crop(a, bg, r=0, c=0):
    ys, xs = np.nonzero(a != bg)
    if not len(ys):
        return None
    r0, r1, c0, c1 = int(ys.min()), int(ys.max())+1, int(xs.min()), int(xs.max())+1
    return a[r0:r1, c0:c1], (r+r0, c+c0, r+r1, c+c1)


def pair_readings(grid):
    """Enumerate every field-color/separator interpretation of the full input.

    This intentionally accepts a reference that is already an enlargement.
    Relative dimensions and every displayed cell decide the scale instead.
    """
    a = np.asarray(grid)
    readings = {}
    for bg in np.unique(a):
        for axis in (0, 1):
            empty = np.all(a == bg, axis=1-axis)
            for cut in range(1, len(empty)-1):
                if not empty[cut]:
                    continue
                # One split per run of field rows/columns is sufficient.
                if cut > 1 and empty[cut-1]:
                    continue
                pieces = ((_crop(a[:cut], bg), _crop(a[cut+1:], bg, cut+1, 0)) if axis == 0
                          else (_crop(a[:, :cut], bg), _crop(a[:, cut+1:], bg, 0, cut+1)))
                if any(p is None for p in pieces):
                    continue
                (small, sb), (big, bb) = sorted(pieces, key=lambda p: p[0].size)
                sh, sw = small.shape
                bh, bw = big.shape
                if bh % sh or bw % sw or bh//sh < 2 or bh//sh != bw//sw:
                    continue
                n = bh//sh
                # A coordinate map is independent of the repetition constructor.
                if np.array_equal(big, small[np.arange(bh)[:, None]//n, np.arange(bw)[None, :]//n]):
                    key = (bg.item(), sb, bb)
                    readings[key] = (n, sb, bb)
    return readings


def validate_pair(ex):
    a, m = np.asarray(ex["input"]), ex["metadata"]
    if a.ndim != 2 or not all(2 <= d <= 30 for d in a.shape):
        raise ValueError("Invalid enlargement canvas")
    readings = pair_readings(a)
    if len(readings) != 1:
        raise ValueError("Enlargement pair has competing field or pair interpretations")
    n, sb, bb = next(iter(readings.values()))
    if str(n) != ex["output"] or n != m["scale"]:
        raise ValueError("Relative enlargement answer does not match the visible pair")
    if list(sb) != m["small_bbox"] or list(bb) != m["big_bbox"]:
        raise ValueError("Visible enlargement pair differs from planned patches")


def validate_compression(ex):
    a, b, m = np.asarray(ex["input"]), np.asarray(ex["output"]), ex["metadata"]
    n = m["scale"]
    if a.ndim != 2 or b.ndim != 2 or not all(2 <= d <= 30 for d in a.shape) or n < 2:
        raise ValueError("Invalid compression dimensions")
    if a.shape != (b.shape[0]*n, b.shape[1]*n) or b.size < 2:
        raise ValueError("Compression output has the wrong dimensions")
    for r in range(b.shape[0]):
        for c in range(b.shape[1]):
            if not np.all(a[r*n:(r+1)*n, c*n:(c+1)*n] == b[r, c]):
                raise ValueError("Compression output differs from a displayed block")


def _pattern(rng, h, w, colors, bg=None):
    if bg is not None and min(h, w) >= 2 and rng.random() < (.80 if len(colors) == 1 else .35):
        families = ["L", "stair"]
        if w >= 3:
            families += ["T", "U"]
        if min(h, w) >= 3:
            families += ["H", "plus", "frame"]
        shape = str(rng.choice(families))
        yy, xx = np.indices((h, w))
        t = max(1, min(h, w)//4)
        if shape == "L":
            mask = (xx < t) | (yy >= h-t)
        elif shape == "T":
            mask = (yy < t) | (abs(xx-(w-1)//2) < t)
        elif shape == "U":
            mask = (xx < t) | (xx >= w-t) | (yy >= h-t)
        elif shape == "H":
            mask = (xx < t) | (xx >= w-t) | (abs(yy-(h-1)//2) < t)
        elif shape == "plus":
            mask = (abs(xx-(w-1)//2) < t) | (abs(yy-(h-1)//2) < t)
        elif shape == "stair":
            mask = xx <= ((yy+1)*w-1)//h
        else:
            mask = (xx < t) | (xx >= w-t) | (yy < t) | (yy >= h-t)
        cells = set(map(tuple, np.argwhere(mask)))
        if len(cells) >= len(colors):
            a = np.full((h, w), bg, dtype=np.uint8)
            # Color contiguous slices of a familiar silhouette, not random dots.
            axis = int(rng.integers(2))
            ordered = sorted(cells, key=lambda p: (p[axis], p[1-axis]))
            for i, (r, c) in enumerate(ordered):
                a[r, c] = colors[min(len(colors)-1, i*len(colors)//len(ordered))]
            if np.any(a != bg, axis=0).all() and np.any(a != bg, axis=1).all():
                return a, "silhouette_"+shape, 1
    if len(colors) == 1:
        return np.full((h, w), colors[0], dtype=np.uint8), "solid", 1
    if min(h, w) >= 2*len(colors)-1 and rng.random() < .15:
        r, c = np.indices((h, w))
        layer = np.minimum.reduce((r, c, h-1-r, w-1-c))
        return np.asarray(colors, dtype=np.uint8)[layer % len(colors)], "concentric", 1
    return _content(rng, h, w, colors, primitive=False)


@lru_cache(maxsize=None)
def _pair_shapes(h, w):
    result = []
    for n in range(2, 16):
        for sh in range(1, h//n+1):
            for sw in range(1, w//n+1):
                if sh*sw < 2:
                    continue
                horizontal, vertical = (n+1)*sw+2 <= w, (n+1)*sh+2 <= h
                if horizontal or vertical:
                    result.append((n, sh, sw, horizontal, vertical))
    return tuple(result)


def generate_pair(rng):
    while True:
        h, w = map(int, rng.integers(MIN_SIDE, 31, 2))
        options = _pair_shapes(h, w)
        if options:
            break
    # Mix a broad-scale draw with a palette-first draw. Both happen after the
    # canvas is fixed; neither rich references nor high factors dominate.
    if rng.random() < .30:
        scales = sorted({p[0] for p in options})
        ordinary, tail = [n for n in scales if n <= 8], [n for n in scales if n > 8]
        n = int(rng.choice(tail if tail and rng.random() < .06 else ordinary))
        options = [p for p in options if p[0] == n]
    max_colors = min(10, max(sh*sw for _, sh, sw, _, _ in options)+1)
    n_colors = int(rng.integers(2, max_colors+1))
    palette = list(map(int, rng.permutation(10)[:n_colors]))
    bg, colors = palette[0], palette[1:]
    options = [p for p in options if p[1]*p[2] >= len(colors)]
    nested = [(p, k) for p in options for k in (2, 3, 4)
              if p[1] % k == p[2] % k == 0 and p[1]*p[2]//(k*k) >= max(2, len(colors))]
    internal = bool(nested and rng.random() < .72)
    if internal:
        n = int(rng.choice(sorted({p[0] for p, _ in nested})))
        nested = [(p, k) for p, k in nested if p[0] == n]
        two_dimensional = [(p, k) for p, k in nested if min(p[1:3]) >= 2*k]
        if two_dimensional and rng.random() < .82:
            nested = two_dimensional
        pair, factor = nested[int(rng.integers(len(nested)))]
    else:
        # Choose the scale among feasible answers, then dimensions within it.
        scales = sorted({p[0] for p in options})
        n = int(rng.choice(scales))
        available = [p for p in options if p[0] == n]
        two_dimensional = [p for p in available if min(p[1:3]) >= 2]
        if two_dimensional and rng.random() < .82:
            available = two_dimensional
        weights = np.asarray([(p[1]*p[2])**.8 for p in available])
        pair, factor = available[int(rng.choice(len(available), p=weights/weights.sum()))], 1
    n, sh, sw, horizontal, vertical = pair
    arrangements = (["horizontal"] if horizontal else []) + (["vertical"] if vertical else [])
    if horizontal and vertical:
        arrangements.append("diagonal")
    arrangement = str(rng.choice(arrangements))
    for attempt in range(1, 201):
        root, family, content_attempts = _pattern(rng, sh//factor, sw//factor, colors, bg)
        small = np.repeat(np.repeat(root, factor, axis=0), factor, axis=1)
        big = np.repeat(np.repeat(small, n, axis=0), n, axis=1)
        bh, bw = big.shape
        if arrangement == "horizontal":
            left = int(rng.integers(w-sw-bw-1))
            gap = int(rng.integers(2, w-left-sw-bw+1))
            sr, br = int(rng.integers(h-sh+1)), int(rng.integers(h-bh+1))
            sc, bc = left, left+sw+gap
        elif arrangement == "vertical":
            top = int(rng.integers(h-sh-bh-1))
            gap = int(rng.integers(2, h-top-sh-bh+1))
            sc, bc = int(rng.integers(w-sw+1)), int(rng.integers(w-bw+1))
            sr, br = top, top+sh+gap
        else:
            sr, sc = 0, 0
            br = int(rng.integers(sh+2, h-bh+1))
            bc = int(rng.integers(sw+2, w-bw+1))
            gap = max(br-sh, bc-sw)
        a = np.full((h, w), bg, dtype=np.uint8)
        a[sr:sr+sh, sc:sc+sw] = small
        a[br:br+bh, bc:bc+bw] = big
        if rng.integers(2):
            a = a[::-1].copy()
            sr, br = h-sr-sh, h-br-bh
        if rng.integers(2):
            a = a[:, ::-1].copy()
            sc, bc = w-sc-sw, w-bc-bw
        sb, bb = [sr, sc, sr+sh, sc+sw], [br, bc, br+bh, bc+bw]
        readings = pair_readings(a)
        if len(readings) != 1 or next(iter(readings.values())) != (n, tuple(sb), tuple(bb)):
            continue
        m = {"grid_h": h, "grid_w": w, "scale": n, "bg": bg, "n_colors": n_colors,
             "n_grid_colors": len(np.unique(a)), "base_h": sh, "base_w": sw,
             "base_family": family, "arrangement": arrangement, "gap": gap,
             "small_bbox": sb, "big_bbox": bb, "reference_factor": factor,
             "reference_compression_factors": compression_factors(small),
             "pair_interpretations": len(readings), "attempts": attempt,
             "content_attempts": content_attempts}
        ex = {"input": a.tolist(), "output": str(n), "metadata": m}
        validate_pair(ex)
        return ex
    raise RuntimeError(f"Cannot realize unique enlargement pair on {h}x{w}")


@lru_cache(maxsize=None)
def _scales(h, w):
    return tuple(n for n in range(2, 16) if h % n == w % n == 0 and h*w//(n*n) >= 2)


def generate_compression(rng):
    # Broad feasible size coverage, plus an explicit share where several
    # requested granularities can be valid on the very same input.
    multiple = bool(rng.random() < .65)
    while True:
        h, w = map(int, rng.integers(MIN_SIDE, 31, 2))
        choices = _scales(h, w)
        composite = [n for n in choices if any(n % k == 0 for k in range(2, n))]
        if choices and (not multiple or composite):
            break
    possible = composite if multiple else list(choices)
    max_colors = min(10, max(h*w//(n*n) for n in possible))
    n_colors = int(rng.integers(2, max_colors+1))
    colors = list(map(int, rng.permutation(10)[:n_colors]))
    possible = [n for n in possible if h*w//(n*n) >= n_colors]
    block = int(rng.choice(possible))
    root, family, attempts = _pattern(rng, h//block, w//block, colors)
    a = np.repeat(np.repeat(root, block, axis=0), block, axis=1)
    valid = [n for n in compression_factors(a) if a.size//(n*n) >= 2]
    # The prompt is chosen after the complete scene; a smaller valid factor
    # preserves block structure in the correct output.
    n = int(rng.choice(valid))
    output = a[::n, ::n].copy()
    m = {"scale": n, "out_h": h//n, "out_w": w//n, "in_h": h, "in_w": w,
         "n_colors": n_colors, "content_style": family, "construction_factor": block,
         "valid_factors": valid, "retains_blocks": max(valid) > n,
         "planned_multiple": multiple, "attempts": attempts}
    ex = {"input": a.tolist(), "output": output.tolist(), "metadata": m}
    validate_compression(ex)
    return ex
