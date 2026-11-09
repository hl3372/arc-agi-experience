"""Count inside a unique unnamed frame, with coherent content and exact budgets."""
from __future__ import annotations

import numpy as np

from generators.crop_frame_26 import Frame, find_frames
from generators.cell_operations_29_30 import CONTENT_FAMILIES, _cell_order, _fill_other_colors


def solve_count_in_frame(grid, query_color):
    a = np.asarray(grid)
    frames = find_frames(a)
    if len(frames) != 1:
        raise ValueError(f"Task 35 requires one visible frame; found {len(frames)}")
    r, c, h, w, _ = frames[0]
    return str(int(np.count_nonzero(a[r + 1:r + h - 1, c + 1:c + w - 1] == query_color)))


def validate_example(ex):
    a = np.asarray(ex["input"])
    if a.ndim != 2 or not all(3 <= n <= 30 for n in a.shape):
        raise ValueError("Task 35 invalid dimensions")
    expected = solve_count_in_frame(a, ex["metadata"]["query_color"])
    if expected != ex["output"]:
        raise ValueError("Task 35 visible reconstruction differs from output")
    if len(np.unique(a)) != ex["metadata"]["n_colors"]:
        raise ValueError("Task 35 lost palette budget")


def generate_count_color_in_roi(rng):
    h, w = map(int, rng.integers(3, 31, size=2))
    frame_mode = str(rng.choice(("full", "strip", "large", "any"), p=(.18, .14, .28, .40)))
    if frame_mode == "full":
        fh, fw = h, w
    elif frame_mode == "strip":
        fh, fw = (3, int(rng.integers(3, w + 1))) if rng.integers(2) else (int(rng.integers(3, h + 1)), 3)
    elif frame_mode == "large":
        fh, fw = int(rng.integers(max(3, (h * 3) // 5), h + 1)), int(rng.integers(max(3, (w * 3) // 5), w + 1))
    else:
        fh, fw = int(rng.integers(3, h + 1)), int(rng.integers(3, w + 1))
    r, c = int(rng.integers(h - fh + 1)), int(rng.integers(w - fw + 1))
    border = np.zeros((h, w), dtype=bool)
    border[r:r + fh, c] = border[r:r + fh, c + fw - 1] = True
    border[r, c:c + fw] = border[r + fh - 1, c:c + fw] = True
    inside = np.zeros_like(border)
    inside[r + 1:r + fh - 1, c + 1:c + fw - 1] = True
    outside = ~(border | inside)
    capacity, outside_area = int(inside.sum()), int(outside.sum())
    zero = bool(rng.random() < .065)
    absent = zero and (outside_area == 0 or rng.random() < .35)
    n_colors = int(rng.integers(2, min(9 if absent else 10, int((~border).sum()) + 1) + 1))
    if zero and n_colors == 2:
        absent = True
    palette = [int(x) for x in rng.permutation(10)]
    frame_color = palette[0]
    query_color = palette[n_colors] if absent else palette[1]
    others = palette[1:n_colors] if absent else palette[2:n_colors]
    # All non-frame colours are visible. Reserve cells for them before choosing
    # a count; no arbitrary answer cap and no palette repair with random pixels.
    minimum = 1
    maximum = min(capacity, int((~border).sum()) - len(others))
    count_mode = str(rng.choice(("small", "any", "dense"), p=(.25, .50, .25)))
    if zero:
        count = 0
    elif not others:
        count = capacity
    elif count_mode == "small":
        count = int(rng.integers(minimum, max(minimum, min(12, maximum)) + 1))
    elif count_mode == "dense":
        count = int(rng.integers(max(minimum, (maximum * 2) // 3), maximum + 1))
    else:
        count = int(rng.integers(minimum, maximum + 1))
    if absent:
        outside_count = 0
    elif not others:
        outside_count = outside_area
    else:
        outside_max = min(outside_area, int((~border).sum()) - count - len(others))
        outside_min = 1 if zero else 0
        if outside_max and rng.random() < .8:
            outside_min = 1
        outside_count = int(rng.integers(outside_min, max(outside_min, outside_max) + 1))
    family = str(rng.choice(CONTENT_FAMILIES))
    expected_frame = Frame(r, c, fh, fw, frame_color)
    for attempt in range(1, 151):
        if attempt > 1:
            # Position may change without changing dimensions, palette, count
            # or content family. A uniform two-colour surround can itself form
            # a competing frame when the intended frame is inset by one cell.
            r, c = int(rng.integers(h - fh + 1)), int(rng.integers(w - fw + 1))
            border[:] = False
            border[r:r + fh, c] = border[r:r + fh, c + fw - 1] = True
            border[r, c:c + fw] = border[r + fh - 1, c:c + fw] = True
            inside[:] = False
            inside[r + 1:r + fh - 1, c + 1:c + fw - 1] = True
            outside = ~(border | inside)
            expected_frame = Frame(r, c, fh, fw, frame_color)
        a = np.full((h, w), frame_color, dtype=np.uint8)
        query_mask = np.zeros((h, w), dtype=bool)
        order = _cell_order(rng, fh - 2, fw - 2, family)
        yy, xx = np.unravel_index(order[:count], (fh - 2, fw - 2))
        query_mask[yy + r + 1, xx + c + 1] = True
        order = _cell_order(rng, h, w, family)
        positions = order[outside.ravel()[order]][:outside_count]
        query_mask.ravel()[positions] = True
        a[query_mask] = query_color
        detail_family, _ = _fill_other_colors(rng, a, query_mask | border, others)
        if find_frames(a) != [expected_frame]:
            continue
        # The first colour's area organizes the scene, but must not make the
        # queried colour predictably the largest region. For positive answers,
        # choose uniformly among colours actually visible inside the frame.
        inside_counts = {color: int(np.count_nonzero(a[inside] == color)) for color in palette[:n_colors] if color != frame_color}
        choices = [color for color, amount in inside_counts.items() if amount]
        chosen_query = query_color if zero else choices[int(rng.integers(len(choices)))]
        answer = 0 if zero else inside_counts[chosen_query]
        chosen_outside_count = int(np.count_nonzero(a[outside] == chosen_query))
        ex = {"input": a.tolist(), "output": str(answer), "metadata": {
            "roi_color": frame_color, "query_color": chosen_query,
            "frame_bbox": [r, c, fh, fw], "frame_mode": frame_mode,
            "n_colors": n_colors, "content_family": family,
            "detail_family": detail_family, "layout_count_mode": count_mode,
            "count_mode": "zero" if zero else "uniform_inside_color",
            "outside_query_count": chosen_outside_count, "interior_area": capacity,
            "attempts": attempt}}
        validate_example(ex)
        return ex
    raise RuntimeError("Task 35 could not realize its fixed scene without competing frames")
