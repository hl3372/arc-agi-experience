"""Correspondence and change-classification task generators."""
from __future__ import annotations

import numpy as np

from common.grid import grid_to_list, make_grid
from generators.core2_extra.utils import add_patterned_background, unused_colors
from registry import register


# Default geometry (kept for back-compat / metadata defaults). Per-scene geometry
# is now sampled independently per call: scene height, slot width, slot gap, the
# object band offset, and the number of slots all vary (see ``_sample_geom``),
# so H and W decorrelate and span small->medium grids instead of a fixed 11x27.
SCENE_H = 11
SLOT_W = 6
SLOT_GAP = 1
OBJECT_TOP_ROW = 3
# Bounds for the sampled geometry. The structural minimum keeps room for a
# 2x-scaled object (4 rows tall / wide) plus a +/-1 translation, while the upper
# bounds stay well under the ~2900-token bundle cap for two (or three) scenes.
SLOT_W_CHOICES = (6, 7)
OBJECT_TOP_ROW_CHOICES = (3, 4, 5)
SCENE_H_MAX_TEXT = 18
SCENE_H_MAX_GRID = 13
N_SLOTS_MIN = 3
N_SLOTS_MAX_TEXT = 6
N_SLOTS_MAX_GRID = 5
INPUT_LABELS = list("ABCDEF")
# Single-example char budget (~2900-token bundle * 1.2 chars/token, minus a
# margin for the prompt + header). Two spaced HxW scenes cost ~4*H*W chars;
# the grid-output task renders a third scene so it gets a smaller area budget.
_SCENE_AREA_BUDGET_TEXT = 520
_SCENE_AREA_BUDGET_GRID = 380
# Task 242 has the longest prompt and a padding variant that enlarges the
# after-scene by up to +2 rows/+2 cols, so it needs a tighter area budget.
_SCENE_AREA_BUDGET_242 = 380
_SCENE_BASE_CAP_242 = 15
RELATION_CANDIDATES = [
    "top_row_order",
    "cell_count_order",
    "color_equality_pattern",
]
PROPERTY_ORDER = [
    "color_preserved",
    "shape_preserved",
    "size_preserved",
    "position_preserved",
    "connectedness_preserved",
    "slot_preserved",
]
CHANGED_PROPERTY_ORDER = [
    "color_changed",
    "shape_changed",
    "size_changed",
    "position_changed",
]
GLOBAL_PROPERTY_ORDER = [
    "background_color_preserved",
    "grid_size_preserved",
    "label_layout_preserved",
    "object_count_preserved",
    "slot_order_preserved",
]

SHAPES: dict[str, tuple[tuple[int, int], ...]] = {
    "L3": ((0, 0), (1, 0), (1, 1)),
    "mirror_L3": ((0, 1), (1, 0), (1, 1)),
    "square2": ((0, 0), (0, 1), (1, 0), (1, 1)),
    "corner3": ((0, 0), (0, 1), (1, 0)),
    "single": ((0, 0),),
    "domino_h": ((0, 0), (0, 1)),
    "domino_v": ((0, 0), (1, 0)),
}
TRANSLATIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


class _Geom:
    """Per-scene sampled geometry: scene height, slot width/gap, band offset."""

    __slots__ = ("scene_h", "slot_w", "slot_gap", "object_top_row")

    def __init__(self, scene_h: int, slot_w: int, slot_gap: int, object_top_row: int):
        self.scene_h = int(scene_h)
        self.slot_w = int(slot_w)
        self.slot_gap = int(slot_gap)
        self.object_top_row = int(object_top_row)


def _sample_geom(
    rng: np.random.Generator,
    n_slots: int,
    *,
    grid_out: bool = False,
    area_budget: int | None = None,
    base_cap: int | None = None,
) -> _Geom:
    """Sample independent scene geometry inside the feasible / budget envelope.

    Width follows from the slot count/width; ``scene_h`` is then sampled
    independently within a height cap that keeps a *single* rendered example
    under the ~2900-token bundle budget (the grid-output task renders a third
    scene, so it gets a tighter area budget). Heights are allowed to shrink to
    the structural minimum so the distribution shifts toward the small grids ARC
    favours instead of clustering at a fixed mid-teens value.
    """
    slot_w = int(rng.choice(SLOT_W_CHOICES))
    slot_gap = 1
    object_top_row = int(rng.choice(OBJECT_TOP_ROW_CHOICES))
    width = n_slots * slot_w + (n_slots - 1) * slot_gap
    if base_cap is None:
        base_cap = SCENE_H_MAX_GRID if grid_out else SCENE_H_MAX_TEXT
    if area_budget is None:
        area_budget = _SCENE_AREA_BUDGET_GRID if grid_out else _SCENE_AREA_BUDGET_TEXT
    budget_cap = max(1, area_budget // max(1, width))
    h_max = min(base_cap, budget_cap)
    # Smallest height that always fits the object band: object_top_row + a
    # 2x-scaled object (4 rows) shifted at most +1 by a translation.
    h_min = object_top_row + 4
    if h_max < h_min:
        h_max = h_min
    scene_h = int(rng.integers(h_min, h_max + 1))
    return _Geom(scene_h, slot_w, slot_gap, object_top_row)


def _output_labels(n: int) -> list[str]:
    start = ord("Z") - n + 1
    return [chr(start + i) for i in range(n)]


def _slot_left(index: int, geom: _Geom) -> int:
    return index * (geom.slot_w + geom.slot_gap)


def _scene_width(n_slots: int, geom: _Geom) -> int:
    return n_slots * geom.slot_w + (n_slots - 1) * geom.slot_gap


def _empty_scene(n_slots: int, bg: int, label_color: int, divider_color: int, geom: _Geom) -> np.ndarray:
    grid = make_grid(geom.scene_h, _scene_width(n_slots, geom), fill=bg)
    for index in range(n_slots):
        grid[0, _slot_left(index, geom) + geom.slot_w // 2] = label_color
    for index in range(n_slots - 1):
        grid[:, _slot_left(index, geom) + geom.slot_w] = divider_color
    return grid


def _decorate_background(
    before: np.ndarray,
    after: np.ndarray,
    rng: np.random.Generator,
    *,
    bg: int,
    geom: _Geom,
    used_colors: list[int] | tuple[int, ...],
) -> dict:
    """Sparsely texture the decorative band strictly above the object slots.

    Every task in this file parses objects from ``object_top_row`` downward, so
    confining texture to rows ``[2, object_top_row - 1]`` (never rows 0/1, never
    a divider/label/object cell, and never the cell the global-property check
    reads at ``[1, 0]``) makes the texture provably invariant for the answer of
    all eight tasks. The same texture is mirrored onto both scenes so it reads as
    a shared, ignorable backdrop. Returns a metadata record declaring it.
    """
    h, w = before.shape
    band = [2, geom.object_top_row - 1]
    if geom.object_top_row <= 2:
        return {"style": "solid", "colors": [], "period": 0, "band": band}
    # Protect rows 0-1 and the whole object region; non-bg cells are auto-skipped
    # by add_patterned_background, so dividers/labels/objects are never touched.
    protected_boxes = [(0, 0, 1, w - 1), (geom.object_top_row, 0, h - 1, w - 1)]
    record = add_patterned_background(
        before,
        rng,
        bg=bg,
        exclude_colors=tuple(int(c) for c in used_colors),
        protected_boxes=protected_boxes,
        solid_probability=0.30,
    )
    # Mirror the exact band texture onto the after-scene where it is background.
    r0, r1 = 2, geom.object_top_row - 1
    for r in range(r0, r1 + 1):
        for c in range(w):
            if before[r, c] != bg and after[r, c] == bg:
                after[r, c] = before[r, c]
    record["band"] = band
    return record


def _scale_shape(cells: set[tuple[int, int]], factor: int = 2) -> set[tuple[int, int]]:
    return {
        (r * factor + dr, c * factor + dc)
        for r, c in cells
        for dr in range(factor)
        for dc in range(factor)
    }


def _stamp(grid: np.ndarray, shape: set[tuple[int, int]], anchor: tuple[int, int], color: int) -> set[tuple[int, int]]:
    ar, ac = anchor
    cells = {(ar + r, ac + c) for r, c in shape}
    for r, c in cells:
        grid[r, c] = color
    return cells


def _normalized(cells: set[tuple[int, int]]) -> list[list[int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return [[int(r - min_r), int(c - min_c)] for r, c in sorted(cells)]


def _properties_for(record: dict) -> tuple[list[str], list[str]]:
    preserved = []
    changed = []
    if record["input_color"] == record["output_color"]:
        preserved.append("color_preserved")
    else:
        changed.append("color_changed")
    if record["input_shape"] == record["output_shape"]:
        preserved.append("shape_preserved")
    else:
        changed.append("shape_changed")
    if len(record["input_shape"]) == len(record["output_shape"]):
        preserved.append("size_preserved")
    else:
        changed.append("size_changed")
    if record["input_anchor"] == record["output_anchor"]:
        preserved.append("position_preserved")
    else:
        changed.append("position_changed")
    preserved.extend(["connectedness_preserved", "slot_preserved"])
    return (
        [label for label in PROPERTY_ORDER if label in preserved],
        changed,
    )


def _change_descriptor(record: dict) -> str:
    change = record["change_type"]
    if change == "translated":
        dr, dc = record["translation"]
        return f"translated(dr={dr},dc={dc})"
    if change == "recolored":
        return "recolored(to_output_color)"
    if change == "scaled_2x":
        return "scaled_2x(factor=2)"
    return change


def _class_line(records: list[dict], *, detailed: bool) -> str:
    parts = []
    for record in records:
        label = record["input_label"]
        value = record["change_type"] if detailed else ("unchanged" if record["change_type"] == "unchanged" else "changed")
        parts.append(f"{label}:{value}")
    return ";".join(parts)


def _metadata_base(
    *,
    n_slots: int,
    bg: int,
    label_color: int,
    divider_color: int,
    input_labels: list[str],
    output_labels: list[str],
    geom: _Geom,
) -> dict:
    return {
        "background_color": int(bg),
        "label_marker_color": int(label_color),
        "divider_color": int(divider_color),
        "num_slots": int(n_slots),
        "slot_width": geom.slot_w,
        "slot_gap": geom.slot_gap,
        "object_top_row": geom.object_top_row,
        "scene_height": geom.scene_h,
        "input_labels": input_labels,
        "output_labels": output_labels,
        "correspondence_rule": "same_slot_order",
        "change_labels": ["unchanged", "translated", "recolored", "scaled_2x", "deleted", "new"],
        "property_labels": list(PROPERTY_ORDER),
        "changed_property_labels": list(CHANGED_PROPERTY_ORDER),
    }


def _shape_extent(shape: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in shape]
    cs = [c for _, c in shape]
    return min(rs), min(cs), max(rs), max(cs)


def _place_record(
    rng: np.random.Generator,
    *,
    slot_left: int,
    geom: _Geom,
    input_shape: set[tuple[int, int]],
    output_shape: set[tuple[int, int]],
    translation: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Sample input/output anchors so both placements fit the band and the slot.

    Anchors are sampled from the full feasible window (rows: object_top_row down
    to the scene floor; cols: a 1-cell margin in from each slot edge), accounting
    for the union footprint of the input shape, the (possibly scaled) output
    shape, and any +/-1 translation. This decorrelates object position from the
    fixed mid-grid placement of the original generator.
    """
    dr, dc = translation
    in_min_r, in_min_c, in_max_r, in_max_c = _shape_extent(input_shape)
    out_min_r, out_min_c, out_max_r, out_max_c = _shape_extent(output_shape)
    # Output cells sit at anchor + (cell - 0) shifted by the translation.
    u_min_r = min(in_min_r, out_min_r + dr)
    u_max_r = max(in_max_r, out_max_r + dr)
    u_min_c = min(in_min_c, out_min_c + dc)
    u_max_c = max(in_max_c, out_max_c + dc)
    r_lo = geom.object_top_row - u_min_r
    r_hi = geom.scene_h - 1 - u_max_r
    if r_hi < r_lo:
        r_hi = r_lo
    anchor_r = int(rng.integers(r_lo, r_hi + 1))
    # Column offset within the slot, keeping a >=1-cell margin on both sides.
    c_lo = 1 - u_min_c
    c_hi = geom.slot_w - 2 - u_max_c
    if c_hi < c_lo:
        c_hi = c_lo
    c_off = int(rng.integers(c_lo, c_hi + 1))
    anchor_c = slot_left + c_off
    return (anchor_r, anchor_c), (anchor_r + dr, anchor_c + dc)


def _one_to_one_scene(
    rng: np.random.Generator,
    change_types: list[str] | None = None,
    geom: _Geom | None = None,
    *,
    area_budget: int | None = None,
    base_cap: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict], dict]:
    if change_types is None:
        n_slots = int(rng.integers(N_SLOTS_MIN, N_SLOTS_MAX_TEXT + 1))
        change_types = ["unchanged", "translated", "recolored", "scaled_2x"]
        # Repeat/trim the canonical four changes to the sampled slot count, then
        # shuffle so the slot<->change assignment is decorrelated.
        change_types = [change_types[i % len(change_types)] for i in range(n_slots)]
        rng.shuffle(change_types)
    else:
        n_slots = len(change_types)
    if geom is None:
        geom = _sample_geom(rng, n_slots, area_budget=area_budget, base_cap=base_cap)
    bg, label_color, divider_color = unused_colors(rng, 3)
    object_colors = unused_colors(rng, n_slots, bg, label_color, divider_color)
    input_labels = INPUT_LABELS[:n_slots]
    output_labels = _output_labels(n_slots)
    before = _empty_scene(n_slots, bg, label_color, divider_color, geom)
    after = _empty_scene(n_slots, bg, label_color, divider_color, geom)
    shape_names = list(SHAPES)
    records: list[dict] = []
    for index, change in enumerate(change_types):
        slot_left = _slot_left(index, geom)
        shape_name = shape_names[int(rng.integers(0, len(shape_names)))]
        input_shape = set(SHAPES[shape_name])
        input_color = int(object_colors[index])
        output_shape = set(input_shape)
        output_color = input_color
        translation = (0, 0)
        if change == "translated":
            translation = TRANSLATIONS[int(rng.integers(0, len(TRANSLATIONS)))]
        elif change == "recolored":
            output_color = int(object_colors[(index + 1) % n_slots])
        elif change == "scaled_2x":
            output_shape = _scale_shape(input_shape, factor=2)

        input_anchor, output_anchor = _place_record(
            rng,
            slot_left=slot_left,
            geom=geom,
            input_shape=input_shape,
            output_shape=output_shape,
            translation=translation,
        )
        translation = [int(translation[0]), int(translation[1])]

        _stamp(before, input_shape, input_anchor, input_color)
        _stamp(after, output_shape, output_anchor, output_color)
        record = {
            "input_label": input_labels[index],
            "output_label": output_labels[index],
            "slot_index": int(index),
            "change_type": change,
            "input_color": input_color,
            "output_color": output_color,
            "input_anchor": [int(input_anchor[0]), int(input_anchor[1])],
            "output_anchor": [int(output_anchor[0]), int(output_anchor[1])],
            "translation": translation,
            "input_shape": _normalized(input_shape),
            "output_shape": _normalized(output_shape),
        }
        preserved, changed = _properties_for(record)
        record["preserved_properties"] = preserved
        record["changed_properties"] = changed
        records.append(record)

    used = sorted(set(int(c) for c in np.unique(before)) | set(int(c) for c in np.unique(after)))
    texture = _decorate_background(before, after, rng, bg=bg, geom=geom, used_colors=used)

    meta = _metadata_base(
        n_slots=n_slots,
        bg=bg,
        label_color=label_color,
        divider_color=divider_color,
        input_labels=input_labels,
        output_labels=output_labels,
        geom=geom,
    )
    meta["object_records"] = records
    meta["background_texture"] = texture
    return before, after, records, meta


def _meta_geom(meta: dict) -> tuple[int, int, int]:
    return (
        int(meta["slot_width"]),
        int(meta["slot_gap"]),
        int(meta["object_top_row"]),
    )


def _slot_occupancy(grid: np.ndarray, meta: dict, bg: int | None = None) -> list[bool]:
    if bg is None:
        bg = int(meta["background_color"])
    bg = int(bg)
    slot_w, slot_gap, object_top_row = _meta_geom(meta)
    occupied = []
    for index in range(int(meta["num_slots"])):
        left = index * (slot_w + slot_gap)
        right = left + slot_w
        slot = grid[object_top_row:, left:right]
        occupied.append(bool(np.any(slot != bg)))
    return occupied


def _global_properties(before: np.ndarray, after: np.ndarray, meta: dict) -> list[str]:
    preserved = []
    before_bg = int(meta["background_color"])
    after_bg = int(meta.get("after_background_color", before_bg))
    if int(before[1, 0]) == before_bg and int(after[1, 0]) == after_bg and before_bg == after_bg:
        preserved.append("background_color_preserved")
    if before.shape == after.shape:
        preserved.append("grid_size_preserved")
    before_layout = (before == int(meta["label_marker_color"])) | (before == int(meta["divider_color"]))
    after_layout = (after == int(meta["label_marker_color"])) | (after == int(meta["divider_color"]))
    if before_layout.shape == after_layout.shape and np.array_equal(before_layout, after_layout):
        preserved.append("label_layout_preserved")
    before_occupied = _slot_occupancy(before, meta, before_bg)
    after_occupied = _slot_occupancy(after, meta, after_bg)
    if sum(before_occupied) == sum(after_occupied):
        preserved.append("object_count_preserved")
    if before_occupied == after_occupied:
        preserved.append("slot_order_preserved")
    return [label for label in GLOBAL_PROPERTY_ORDER if label in preserved]


def _relation_preserved(records: list[dict], candidate: str) -> bool:
    def sign(value: int) -> int:
        return (value > 0) - (value < 0)

    if candidate == "left_to_right_order":
        return True
    if candidate == "top_row_order":
        before_tops = [record["input_anchor"][0] for record in records]
        after_tops = [record["output_anchor"][0] for record in records]
        return all(
            sign(before_tops[i] - before_tops[j]) == sign(after_tops[i] - after_tops[j])
            for i in range(len(records))
            for j in range(i + 1, len(records))
        )
    if candidate == "cell_count_order":
        before_sizes = [len(record["input_shape"]) for record in records]
        after_sizes = [len(record["output_shape"]) for record in records]
        return all(
            sign(before_sizes[i] - before_sizes[j]) == sign(after_sizes[i] - after_sizes[j])
            for i in range(len(records))
            for j in range(i + 1, len(records))
        )
    return all(
        (records[i]["input_color"] == records[j]["input_color"]) == (records[i]["output_color"] == records[j]["output_color"])
        for i in range(len(records))
        for j in range(i + 1, len(records))
    )


@register("236")
def generate_correspondence_changed_unchanged(rng: np.random.Generator) -> dict:
    n_slots = int(rng.integers(N_SLOTS_MIN, N_SLOTS_MAX_TEXT + 1))
    changes = ["unchanged", "translated", "recolored", "scaled_2x"]
    change_types = [changes[int(rng.integers(0, len(changes)))] for _ in range(n_slots)]
    before, after, records, meta = _one_to_one_scene(rng, change_types=change_types)
    return {
        "input": {"input_scene": grid_to_list(before), "output_scene": grid_to_list(after)},
        "output": _class_line(records, detailed=False),
        "metadata": meta,
    }


@register("237")
def generate_correspondence_change_type(rng: np.random.Generator) -> dict:
    before, after, records, meta = _one_to_one_scene(rng)
    return {
        "input": {"input_scene": grid_to_list(before), "output_scene": grid_to_list(after)},
        "output": _class_line(records, detailed=True),
        "metadata": meta,
    }


@register("238")
def generate_selected_corresponding_object_and_change(rng: np.random.Generator) -> dict:
    before, after, records, meta = _one_to_one_scene(rng)
    selected = int(rng.integers(0, len(records)))
    record = records[selected]
    meta["selected_input_label"] = record["input_label"]
    meta["selected_output_label"] = record["output_label"]
    meta["selected_slot_index"] = selected
    output = f"{record['input_label']}->{record['output_label']}:{_change_descriptor(record)}"
    return {
        "input": {"input_scene": grid_to_list(before), "output_scene": grid_to_list(after)},
        "output": output,
        "metadata": meta,
    }


@register("239")
def generate_lifecycle_category_recolor_grid(rng: np.random.Generator) -> dict:
    # grid->grid bundle: cap slots/height tighter so three rendered scenes stay
    # under the ~2900-token budget while still spanning small->medium widths.
    n_slots = int(rng.integers(N_SLOTS_MIN, N_SLOTS_MAX_GRID + 1))
    geom = _sample_geom(rng, n_slots, grid_out=True)
    bg, label_color, divider_color = unused_colors(rng, 3)
    unchanged_color, changed_color, deleted_color, new_color = unused_colors(rng, 4, bg, label_color, divider_color)
    object_colors = unused_colors(rng, 3, bg, label_color, divider_color)
    input_labels = INPUT_LABELS[:n_slots]
    output_labels = _output_labels(n_slots)
    before = _empty_scene(n_slots, bg, label_color, divider_color, geom)
    after = _empty_scene(n_slots, bg, label_color, divider_color, geom)
    mask = _empty_scene(n_slots, bg, label_color, divider_color, geom)

    change_choices = ["unchanged", "translated", "recolored", "scaled_2x", "deleted", "new"]
    lifecycle = [change_choices[int(rng.integers(0, len(change_choices)))] for _ in range(n_slots)]
    shape_names = list(SHAPES)
    records: list[dict] = []
    for index, change in enumerate(lifecycle):
        slot_left = _slot_left(index, geom)
        shape = set(SHAPES[shape_names[int(rng.integers(0, len(shape_names)))]])
        output_shape = set(shape)
        input_color = int(object_colors[index % len(object_colors)])
        output_color = input_color
        translation = (0, 0)
        if change == "translated":
            translation = TRANSLATIONS[int(rng.integers(0, len(TRANSLATIONS)))]
        elif change == "recolored":
            output_color = int(object_colors[(index + 1) % len(object_colors)])
        elif change == "scaled_2x":
            output_shape = _scale_shape(shape, factor=2)

        input_anchor, output_anchor = _place_record(
            rng,
            slot_left=slot_left,
            geom=geom,
            input_shape=shape,
            output_shape=output_shape,
            translation=translation,
        )
        translation = [int(translation[0]), int(translation[1])]

        input_cells: set[tuple[int, int]] = set()
        output_cells: set[tuple[int, int]] = set()
        if change != "new":
            input_cells = _stamp(before, shape, input_anchor, input_color)
        if change != "deleted":
            output_cells = _stamp(after, output_shape, output_anchor, output_color)

        if change == "unchanged":
            category = "unchanged"
            color = unchanged_color
            cells = output_cells
        elif change == "deleted":
            category = "deleted"
            color = deleted_color
            cells = input_cells
        elif change == "new":
            category = "new"
            color = new_color
            cells = output_cells
        else:
            category = "changed"
            color = changed_color
            cells = output_cells
        for r, c in cells:
            mask[r, c] = color

        records.append({
            "input_label": input_labels[index],
            "output_label": output_labels[index],
            "slot_index": int(index),
            "change_type": change,
            "lifecycle_category": category,
            "input_color": input_color,
            "output_color": output_color,
            "input_anchor": [int(input_anchor[0]), int(input_anchor[1])],
            "output_anchor": [int(output_anchor[0]), int(output_anchor[1])],
            "translation": translation,
            "input_shape": _normalized(shape),
            "output_shape": _normalized(output_shape),
        })

    # Texture the decorative band of the two input scenes only; the output mask
    # keeps just the layout + category-recolored object cells on background, so
    # the answer is unchanged by the texture (texture cells are neither layout
    # nor object and thus map to background in the mask).
    used = sorted(
        set(int(c) for c in np.unique(before))
        | set(int(c) for c in np.unique(after))
        | {int(unchanged_color), int(changed_color), int(deleted_color), int(new_color)}
    )
    texture = _decorate_background(before, after, rng, bg=bg, geom=geom, used_colors=used)

    meta = _metadata_base(
        n_slots=n_slots,
        bg=bg,
        label_color=label_color,
        divider_color=divider_color,
        input_labels=input_labels,
        output_labels=output_labels,
        geom=geom,
    )
    meta.update({
        "object_records": records,
        "background_texture": texture,
        "category_to_color": [
            {"category": "unchanged", "color": int(unchanged_color)},
            {"category": "changed", "color": int(changed_color)},
            {"category": "deleted", "color": int(deleted_color)},
            {"category": "new", "color": int(new_color)},
        ],
    })
    return {
        "input": {"input_scene": grid_to_list(before), "output_scene": grid_to_list(after)},
        "output": grid_to_list(mask),
        "metadata": meta,
    }


@register("240")
def generate_selected_correspondence_preserved_changed_properties(rng: np.random.Generator) -> dict:
    before, after, records, meta = _one_to_one_scene(rng)
    selected = int(rng.integers(0, len(records)))
    record = records[selected]
    preserved = ",".join(record["preserved_properties"]) if record["preserved_properties"] else "none"
    changed = ",".join(record["changed_properties"]) if record["changed_properties"] else "none"
    meta["selected_input_label"] = record["input_label"]
    meta["selected_output_label"] = record["output_label"]
    meta["selected_slot_index"] = selected
    return {
        "input": {"input_scene": grid_to_list(before), "output_scene": grid_to_list(after)},
        "output": f"{record['input_label']}->{record['output_label']}:preserved={preserved};changed={changed}",
        "metadata": meta,
    }


@register("241")
def generate_properties_preserved_for_every_correspondence(rng: np.random.Generator) -> dict:
    n_slots = int(rng.integers(N_SLOTS_MIN, N_SLOTS_MAX_TEXT + 1))
    changes = ["unchanged", "translated", "recolored", "scaled_2x"]
    change_types = [changes[int(rng.integers(0, len(changes)))] for _ in range(n_slots)]
    before, after, records, meta = _one_to_one_scene(rng, change_types=change_types)
    common = set(PROPERTY_ORDER)
    for record in records:
        common &= set(record["preserved_properties"])
    output = ";".join(label for label in PROPERTY_ORDER if label in common) or "none"
    return {
        "input": {"input_scene": grid_to_list(before), "output_scene": grid_to_list(after)},
        "output": output,
        "metadata": meta,
    }


@register("242")
def generate_global_scene_properties_preserved(rng: np.random.Generator) -> dict:
    before, after, _records, meta = _one_to_one_scene(
        rng, area_budget=_SCENE_AREA_BUDGET_242, base_cap=_SCENE_BASE_CAP_242
    )
    slot_w, slot_gap, object_top_row = _meta_geom(meta)
    slot_pitch = slot_w + slot_gap
    bg = int(meta["background_color"])
    after_bg = bg
    variant = int(rng.integers(0, 6))
    if variant == 1:
        slot = int(rng.integers(0, int(meta["num_slots"])))
        left = slot * slot_pitch
        after[object_top_row:, left:left + slot_w] = bg
    elif variant == 2:
        slot = int(rng.integers(0, int(meta["num_slots"])))
        center = slot * slot_pitch + slot_w // 2
        after[0, center] = bg
        after[0, min(center + 1, slot * slot_pitch + slot_w - 1)] = int(meta["label_marker_color"])
    elif variant == 3:
        slot = int(rng.integers(0, int(meta["num_slots"] - 1)))
        left = slot * slot_pitch
        next_left = (slot + 1) * slot_pitch
        after[object_top_row:, left:left + slot_w] = bg
        after[object_top_row:, next_left:next_left + slot_w] = bg
    elif variant == 4:
        used = [int(c) for c in np.unique(np.concatenate([before.ravel(), after.ravel()]))]
        spare = unused_colors(rng, 1, *used)
        if spare:
            new_bg = spare[0]
            after = np.where(after == bg, new_bg, after)
            after_bg = int(new_bg)
        else:
            # No free colour to recolour the background with; leave the scene
            # untouched (equivalent to the no-op variant -> all properties hold).
            variant = 0
    elif variant == 5:
        pad_rows = int(rng.integers(0, 3))
        pad_cols = int(rng.integers(0, 3))
        if pad_rows == 0 and pad_cols == 0:
            pad_rows = 1
        h, w = after.shape
        padded = np.full((h + pad_rows, w + pad_cols), bg, dtype=after.dtype)
        padded[:h, :w] = after
        after = padded
    meta["after_background_color"] = after_bg
    meta["global_variant"] = variant
    meta["global_property_labels"] = list(GLOBAL_PROPERTY_ORDER)
    return {
        "input": {"input_scene": grid_to_list(before), "output_scene": grid_to_list(after)},
        "output": ";".join(_global_properties(before, after, meta)) or "none",
        "metadata": meta,
    }


@register("243")
def generate_candidate_relation_preserved(rng: np.random.Generator) -> dict:
    candidate = RELATION_CANDIDATES[int(rng.integers(0, len(RELATION_CANDIDATES)))]
    if candidate == "color_equality_pattern":
        # The default scene always recolors one object to a neighbour's colour, which always
        # creates a new equal pair (pattern broken). Coin-flip whether to include a recolor so
        # both relation_preserved and relation_not_preserved occur for this candidate.
        n_slots = int(rng.integers(N_SLOTS_MIN, N_SLOTS_MAX_TEXT + 1))
        non_recolor = ["unchanged", "translated", "scaled_2x"]
        change_types = [non_recolor[int(rng.integers(0, len(non_recolor)))] for _ in range(n_slots)]
        if rng.random() < 0.5:
            change_types[int(rng.integers(0, n_slots))] = "recolored"
        before, after, records, meta = _one_to_one_scene(rng, change_types=change_types)
    else:
        before, after, records, meta = _one_to_one_scene(rng)
    meta["candidate_relation"] = candidate
    meta["relation_result_labels"] = ["relation_preserved", "relation_not_preserved"]
    result = "relation_preserved" if _relation_preserved(records, candidate) else "relation_not_preserved"
    return {
        "input": {"input_scene": grid_to_list(before), "output_scene": grid_to_list(after)},
        "output": result,
        "metadata": meta,
    }
