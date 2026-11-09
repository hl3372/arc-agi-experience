"""One Boolean task with per-example color encodings and geometric masks."""
from __future__ import annotations

import numpy as np

from generators.crop_frame_26 import _geometric_shape


OPERATIONS = ("and", "or", "xor", "nand", "nor", "xnor")
FAMILIES = ("shape", "shapes", "frame", "bands", "checker_blocks", "diagonal_bands",
            "half_plane", "stair_bands", "panels", "noise")


def choose_operation(rng):
    return str(rng.choice(OPERATIONS))


def _mask(rng, h, w, family):
    rr, cc = np.indices((h, w))
    mask = np.zeros((h, w), dtype=bool)
    if family in ("shape", "shapes", "frame"):
        for _ in range(int(rng.integers(2, 5)) if family == "shapes" else 1):
            bh, bw = int(rng.integers(1, h + 1)), int(rng.integers(1, w + 1))
            if family == "frame":
                thickness = int(rng.integers(1, max(2, min(bh, bw) // 3 + 1)))
                cells = {(r, c) for r in range(bh) for c in range(bw)
                         if min(r, c, bh - 1 - r, bw - 1 - c) < thickness}
            else:
                cells, _ = _geometric_shape(rng, bh, bw)
            ch, cw = max(r for r, _ in cells) + 1, max(c for _, c in cells) + 1
            top, left = int(rng.integers(h - ch + 1)), int(rng.integers(w - cw + 1))
            for r, c in cells:
                mask[top + r, left + c] = True
    elif family == "noise":
        mask = rng.random((h, w)) < float(rng.uniform(.12, .88))
    elif family == "panels":
        for _ in range(int(rng.integers(2, 6))):
            r0, r1 = sorted(map(int, rng.integers(0, h + 1, 2)))
            c0, c1 = sorted(map(int, rng.integers(0, w + 1, 2)))
            mask[r0:r1, c0:c1] = bool(rng.integers(2))
    else:
        y, x = (rr, cc) if rng.integers(2) else (cc, rr)
        step = int(rng.integers(1, max(3, max(h, w) // 3 + 1)))
        phase = int(rng.integers(0, 2 * step))
        if family == "bands":
            levels = x
        elif family == "checker_blocks":
            levels = (x // step + y // int(rng.integers(1, step + 2)))
            step = 1
        elif family == "stair_bands":
            levels = x + y // int(rng.integers(2, 6))
        else:
            levels = x + int(rng.choice([-2, -1, 1, 2])) * y
        mask = levels >= float(rng.uniform(float(levels.min()), float(levels.max()) + 1)) if family == "half_plane" else ((levels + phase) // step) % 2 == 0
    if rng.random() < .5:
        mask = ~mask
    return mask


def _shift(mask, dr, dc):
    h, w = mask.shape
    out = np.zeros_like(mask)
    r0, r1, c0, c1 = max(0, dr), min(h, h + dr), max(0, dc), min(w, w + dc)
    out[r0:r1, c0:c1] = mask[r0-dr:r1-dr, c0-dc:c1-dc]
    return out


def solve_visible(grid, meta):
    """Decode the displayed halves, then apply scalar Boolean definitions."""
    a = np.asarray(grid)
    axis = 1 if meta["layout"] == "horizontal" else 0
    if a.ndim != 2 or a.shape[axis] % 2 != 1 or a.shape[axis] < 3:
        raise ValueError("Malformed binary layout")
    mid = a.shape[axis] // 2
    first, divider, second = np.split(a, [mid, mid + 1], axis=axis)
    if not np.all(divider == meta["divider_color"]):
        raise ValueError("Malformed binary divider")
    false, t1, t2, true = (meta[k] for k in ("bg_color", "fg_color_left", "fg_color_right", "output_true_color"))
    if not set(np.unique(first)) <= {false, t1} or not set(np.unique(second)) <= {false, t2}:
        raise ValueError("Unspecified binary color")
    op = meta["operation"]
    if op not in OPERATIONS:
        raise ValueError("Unknown Boolean operation")
    out = np.full(first.shape, false, dtype=a.dtype)
    for r in range(first.shape[0]):
        for c in range(first.shape[1]):
            x, y = first[r, c] == t1, second[r, c] == t2
            if op in ("and", "nand"):
                result = bool(x and y)
            elif op in ("or", "nor"):
                result = bool(x or y)
            else:
                result = bool(x != y)
            if op in ("nand", "nor", "xnor"):
                result = not result
            out[r, c] = true if result else false
    return out


def validate(grid, output, meta):
    if not np.array_equal(solve_visible(grid, meta), output):
        raise ValueError("Displayed Boolean operation does not match output")


def generate(rng):
    layout = str(rng.choice(["horizontal", "vertical"]))
    depth, cross = int(rng.integers(1, 15)), int(rng.integers(1, 31))
    h, w = (cross, depth) if layout == "horizontal" else (depth, cross)
    shared = bool(rng.integers(2))
    colors = list(map(int, rng.choice(10, 3 if shared else 5, replace=False)))
    false, t1 = colors[:2]
    t2, true, divider = (t1, t1, colors[2]) if shared else tuple(colors[2:])
    families = list(map(str, rng.choice(FAMILIES, 2)))
    if h * w < 8:
        families = ["shape" if family == "shapes" else family for family in families]
    relation = str(rng.choice(["independent", "shifted", "nested", "disjoint", "identical", "complementary"],
                              p=[.66, .14, .06, .06, .04, .04]))
    for attempt in range(1, 401):
        a, b = (_mask(rng, h, w, family) for family in families)
        if h * w > 1 and (a.all() or not a.any() or b.all() or not b.any()):
            continue
        # Demand every truth-table row in the ordinary 2-D independent mode.
        # Thin/related masks have genuine structural exceptions. This decision
        # precedes the operation, so it cannot encode the requested rule.
        if relation == "independent" and min(h, w) >= 3:
            if len(np.unique(2 * a.astype(int) + b)) < 4:
                continue
        break
    else:
        raise RuntimeError(f"Cannot draw informative {families} masks on {h}x{w}")
    if relation == "shifted":
        b = _shift(a, int(rng.integers(-min(4, h - 1), min(4, h - 1) + 1)),
                   int(rng.integers(-min(4, w - 1), min(4, w - 1) + 1)))
    elif relation == "nested":
        b &= a
    elif relation == "disjoint":
        b &= ~a
    elif relation == "identical":
        b = a.copy()
    elif relation == "complementary":
        b = ~a
    if relation in ("shifted", "identical", "complementary"):
        families[1] = families[0]
    left, right = np.where(a, t1, false), np.where(b, t2, false)
    axis = 1 if layout == "horizontal" else 0
    line = np.full((h, 1) if axis else (1, w), divider)
    grid = np.concatenate([left, line, right], axis=axis)
    # No scene rejection or RNG call after the rule is chosen. Counterfactual
    # operations must see exactly the same masks, layout and color encoding.
    operation = choose_operation(rng)
    if operation in ("and", "nand"):
        result = a & b
    elif operation in ("or", "nor"):
        result = a | b
    else:
        result = a ^ b
    if operation in ("nand", "nor", "xnor"):
        result = ~result
    output = np.where(result, true, false)
    meta = {"operation": operation, "layout": layout, "encoding": "shared" if shared else "different",
            "bg_color": false, "fg_color": t1, "fg_color_left": t1, "fg_color_right": t2,
            "output_true_color": true, "divider_color": divider,
            "mask_families": families, "relationship": relation,
            "true_cells_first": int(a.sum()), "true_cells_second": int(b.sum()),
            "truth_case_counts": np.bincount((2 * a.astype(int) + b).ravel(), minlength=4).tolist(),
            "constant_output": bool(result.all() or not result.any()), "attempts": attempt}
    validate(grid, output, meta)
    return {"input": grid.tolist(), "output": output.tolist(), "metadata": meta}
