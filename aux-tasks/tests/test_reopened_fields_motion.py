"""Counterinterpretations and fixed-layer regressions from the September review."""
import numpy as np
import pytest

from registry import generate
from generators.exterior_field import infer_exterior_field
from generators import anchor_translate_39_40 as motion
from generators import infer_translation_41 as inference
from generators import rectangle_edits_49_50 as rectangles
from tests.test_rebuild_46_50 import components
from tests.test_rebuild_41_45 import clipped_translation_candidates


def spatial_backgrounds(grid, *, allow_strips=False, exterior_mass=True):
    """Separate full component labeling, including opposite-color readings."""
    a = np.asarray(grid)
    h, w = a.shape
    if min(h, w) == 1:
        line = a.ravel()
        color = line[0]
        other = [np.count_nonzero(a == co) for co in np.unique(a) if co != color]
        return ([color.item()] if allow_strips and line[-1] == color and other
                and np.count_nonzero(a == color) > 1.25 * max(other) else [])
    boundary = {(r, c) for r in range(h) for c in range(w)
                if r in (0, h - 1) or c in (0, w - 1)}
    grouped = components(a)
    accepted = []
    for color in np.unique(a):
        field_parts = [cells for co, cells, _ in grouped if co == color]
        outside = [cells for cells in field_parts if cells & boundary]
        if len(outside) != 1:
            continue
        exterior = outside[0]
        if len(exterior & boundary) < .60 * len(boundary):
            continue
        if not all(any(p[axis] == edge for p in exterior)
                   for axis, edge in ((0, 0), (0, h-1), (1, 0), (1, w-1))):
            continue
        bodies = [(cells, bounds) for co, cells, bounds in components(a != color) if co]
        if any((exterior_mass and len(cells) >= len(exterior)) or r1-r0+1 == h or c1-c0+1 == w
               for cells, (r0, c0, r1, c1) in bodies):
            continue
        amount = sum(map(len, field_parts))
        if amount <= 1.25 * max(np.count_nonzero(a == co) for co in np.unique(a) if co != color):
            continue
        accepted.append(color.item())
    return accepted


def test_partitioned_seed2_is_rejected_in_every_orientation():
    a = np.full((26, 10), 2)
    a[10:21] = 4
    for turns in range(4):
        b = np.rot90(a, turns)
        assert spatial_backgrounds(b) == []
        with pytest.raises(ValueError):
            rectangles.solve(b, fill=False, thickness=1)
        with pytest.raises(ValueError):
            motion.solve_translation(b, 1, 1)
    e = generate('49', 2)
    assert spatial_backgrounds(e['input']) == [e['metadata']['bg_color']]


def test_clear_frame_pockets_and_corner_objects_keep_the_exterior():
    a = np.zeros((12, 16), int)
    a[2:10, 3:13] = 1
    a[4:8, 5:11] = 0
    assert infer_exterior_field(a) == 0
    assert spatial_backgrounds(a) == [0]
    b = np.zeros((20, 24), int)
    b[:10, :11] = 1
    assert infer_exterior_field(b) == 0
    assert spatial_backgrounds(b) == [0]
    # Most of the nominal field is inside this thick competing frame.
    c = np.zeros((20, 20), int)
    c[1:19, 1:19] = 1
    c[3:17, 3:17] = 0
    assert (c == 0).sum() > (c == 1).sum()
    with pytest.raises(ValueError):
        infer_exterior_field(c)


@pytest.mark.parametrize('task', ['40', '49'])
def test_generated_exterior_has_one_visible_reading(task):
    for seed in range(400):
        e = generate(task, seed)
        a = np.asarray(e['input'])
        assert min(a.shape) >= 4
        values = spatial_backgrounds(a)
        assert values == [e['metadata']['bg_color'] if task == '49' else e['metadata']['background_color']]
        if seed < 40:
            labels = np.random.default_rng(seed + 21).permutation(list('ABCDEFGHIJ'))
            assert spatial_backgrounds(labels[a]) == [labels[values[0]]]


def test_rotation_keeps_fixed_dots_and_can_cover_a_different_color():
    a = np.zeros((12, 13), int)
    a[6, 6] = 3
    a[6, 8] = 1
    a[6:8, 9] = 2
    a[6, 4] = 2
    a[1, 1] = 1
    expected = a.copy()
    expected[6, 8] = expected[6, 9] = expected[7, 9] = 0
    expected[6, 4] = 1
    expected[6, 3] = expected[5, 3] = 2
    assert np.array_equal(motion.solve_rotation(a, 3, 180), expected)
    assert np.array_equal(motion.solve_rotation(a, 3, 0), a)


def test_rotation_strip_requires_visible_field_at_both_ends():
    with pytest.raises(ValueError):
        motion.solve_rotation([[1, 2, 2, 3, 3, 3]], 1, 180)
    a = [[3, 2, 2, 1, 3, 3]]
    assert spatial_backgrounds(a, allow_strips=True, exterior_mass=False) == [3]
    assert np.array_equal(motion.solve_rotation(a, 1, 180), [[3, 3, 3, 1, 2, 2]])
    for seed in range(400):
        e = generate('39', seed)
        assert spatial_backgrounds(e['input'], allow_strips=True, exterior_mass=False) == [e['metadata']['background_color']]


def test_hidden_surviving_cell_does_not_prove_a_translation():
    a = np.zeros((3, 5), int)
    a[1:3, 1] = 1
    a[0, 4] = 1
    b = np.zeros_like(a)
    b[0, 4] = 1
    # (-2,+3) lands one source cell on the identical dot; complete clipping
    # gives the exact same observed pair. Both readers must reject it.
    with pytest.raises(ValueError):
        inference.validate(a, b, (-2, 3))
    with pytest.raises(ValueError):
        clipped_translation_candidates({'before': a, 'after': b})


def test_clipped_moving_singleton_is_not_reclassified_as_fixed():
    a = np.zeros((8, 9), int)
    a[1:3, 3] = 1
    a[6, 7] = 2
    b = np.zeros_like(a)
    b[0, 3] = 1
    b[6, 7] = 2
    inference.validate(a, b, (-2, 0))
    assert set(clipped_translation_candidates({'before': a, 'after': b})) == {(-2, 0)}
