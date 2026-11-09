"""Exact reflection with both simple and count/centroid-preserving counterexamples."""
from __future__ import annotations

import numpy as np
from generators.crop_frame_26 import _render_scene

PROMPT38 = (
    "The divider is the unique full row or column drawn in a color used nowhere else in the grid. "
    "Is every cell on one side the exact mirror of the corresponding cell on the other side, "
    "including its color? Include all cells and background details. Answer 'true' or 'false'."
)
MODES = ("balanced_cells", "balanced_blocks", "recolor", "move", "several_changes")
CONTENT = ("objects", "objects_on_bands", "bands", "patchwork", "diagonal_bands", "stair_bands", "striped_panels", "tiles")


def divider_lines(grid):
    a = np.asarray(grid)
    colors, counts = np.unique(a, return_counts=True)
    total = dict(zip(colors.tolist(), counts.tolist()))
    result = []
    for axis, rows in enumerate((a, a.T)):
        for i, line in enumerate(rows):
            if np.all(line == line[0]) and total[line[0].item()] == len(line):
                result.append((axis, i))
    return result


def solve_mirrored(grid):
    a = np.asarray(grid)
    lines = divider_lines(a)
    if len(lines) != 1:
        raise ValueError("The divider is not uniquely identifiable")
    axis, index = lines[0]
    if axis == 1:
        a = a.T
    if index * 2 + 1 != len(a):
        raise ValueError("Divider must separate equal-sized halves")
    # Literal cell comparison, separate from the constructor's flipped arrays.
    match = all(a[r, c] == a[2 * index - r, c] for r in range(index) for c in range(a.shape[1]))
    return "true" if match else "false"


def validate_example(ex):
    if solve_mirrored(ex["input"]) != ex["output"]:
        raise ValueError("Visible reflection answer differs from construction")
    if len(np.unique(ex["input"])) != ex["metadata"]["n_colors"]:
        raise ValueError("Reflection palette budget was lost")


def _foil(rng, half, colors, mode, content="patchwork"):
    a = half.copy()
    h, w = a.shape
    if mode.startswith("balanced"):
        # Four congruent blocks form a 2x2 color switch. Each row and column
        # retains each color's count, so both centroids and histograms agree.
        bh = 1 if mode == "balanced_cells" else int(rng.integers(1, max(2, h // 3 + 1)))
        bw = 1 if mode == "balanced_cells" else int(rng.integers(1, max(2, w // 3 + 1)))
        bh, bw = min(bh, h // 2), min(bw, w // 2)
        if 4 * bh * bw > a.size - (len(colors) - 2):
            sizes = [(y, x) for y in range(1, h // 2 + 1) for x in range(1, w // 2 + 1)
                     if 4 * y * x <= a.size - (len(colors) - 2)]
            bh, bw = sizes[int(rng.integers(len(sizes)))]
        y1 = int(rng.integers(h - 2 * bh + 1))
        y2 = int(rng.integers(y1 + bh, h - bh + 1))
        x1 = int(rng.integers(w - 2 * bw + 1))
        x2 = int(rng.integers(x1 + bw, w - bw + 1))
        c1, c2 = map(int, rng.choice(colors, 2, replace=False))
        switched = np.zeros_like(a, dtype=bool)
        for y, x, c in ((y1, x1, c1), (y1, x2, c2), (y2, x1, c2), (y2, x2, c1)):
            a[y:y + bh, x:x + bw] = c
            switched[y:y + bh, x:x + bw] = True
        if set(colors) - set(np.unique(a)):
            # Tiny, colorful halves can lose a singleton color under the four
            # blocks. Reserve room for every other color and fill that common
            # region coherently in BOTH candidates; never reroll the palette.
            extras = [c for c in colors if c not in (c1, c2)]
            common = _render_scene(rng, ~switched, extras, content)
            if common is None:
                return None
            a[~switched] = common[0][~switched]
        b = a.copy()
        for y, x, c in ((y1, x1, c2), (y1, x2, c1), (y2, x1, c1), (y2, x2, c2)):
            b[y:y + bh, x:x + bw] = c
        return a, b
    b = a.copy()
    if mode == "move":
        # A swap preserves the palette and total color counts. It generally
        # changes a centroid, unlike the deliberately harder block switches.
        p = int(rng.integers(a.size))
        other = np.flatnonzero(a.ravel() != a.ravel()[p])
        if not len(other):
            return None
        q = int(rng.choice(other))
        b.ravel()[p], b.ravel()[q] = a.ravel()[q], a.ravel()[p]
    else:
        count = 1 if mode == "recolor" else int(rng.integers(2, min(15, a.size) + 1))
        for p in rng.choice(a.size, count, replace=False):
            choices = [c for c in colors if c != b.ravel()[p]]
            b.ravel()[p] = int(rng.choice(choices))
    return a, b


def generate_is_mirrored(rng):
    axis = int(rng.integers(2))
    depth = int(rng.integers(2, 15))
    cross = int(rng.integers(3, 31))
    cap = min(10, depth * cross - 1)
    n_colors = int(rng.integers(3, cap + 1))
    palette = [int(c) for c in rng.permutation(10)[:n_colors]]
    divider, colors = palette[0], palette[1:]
    mode = str(rng.choice(MODES, p=(.30, .40, .10, .10, .10)))
    content = str(rng.choice(CONTENT))
    # The planned counterexample and palette are the same for true/false.
    # Sample the answer only after the common scene has been constructed.
    for attempt in range(1, 201):
        rendered = _render_scene(rng, np.ones((depth, cross), dtype=bool), colors, content)
        if rendered is None:
            continue
        prepared = _foil(rng, rendered[0], colors, mode, content)
        if prepared is None:
            continue
        half, foil = prepared
        if set(np.unique(half)) != set(colors) or set(np.unique(foil)) != set(colors):
            continue
        if np.array_equal(half, foil):
            continue
        true_grid = np.concatenate((half, np.full((1, cross), divider), half[::-1]), axis=0)
        false_grid = np.concatenate((half, np.full((1, cross), divider), foil[::-1]), axis=0)
        if rng.integers(2):
            true_grid, false_grid = true_grid[::-1], false_grid[::-1]
        if axis:
            true_grid, false_grid = true_grid.T, false_grid.T
        if len(divider_lines(true_grid)) != 1 or len(divider_lines(false_grid)) != 1:
            continue
        mirrored = bool(rng.integers(2))
        a = true_grid if mirrored else false_grid
        ex = {"input": a.tolist(), "output": "true" if mirrored else "false", "metadata": {
            "divider_color": divider, "divider_orientation": "vertical" if axis else "horizontal",
            "n_colors": n_colors, "content_family": content, "foil_family": mode,
            "changed_pairs": 0 if mirrored else int((half != foil).sum()),
            "balanced_foil": mode.startswith("balanced"), "attempts": attempt}}
        validate_example(ex)
        return ex
    raise RuntimeError("Task 38 could not realize its fixed palette and counterexample family")
