"""Coverage and visible-scene invariants for the task28 redesign."""
from collections import Counter

import numpy as np
import pytest

from generators.fit_holes_28 import (
    _build, _count_capacity, _draw_plan, _place_joint,
    generate_fit_holes, parse_scene, validate_fit_holes,
)
from tests.test_task28_uniqueness import local_covers


@pytest.fixture(scope='module')
def samples():
    return [generate_fit_holes(np.random.default_rng(seed)) for seed in range(500)]


def test_dimensions_and_palette_are_fixed_before_geometry_retries(samples):
    heights, widths, palettes = [], [], []
    permitted = [(h, w) for h in range(3, 31) for w in range(3, 31)
                 if min(h, w) >= 5 or min(h, w) == 4 and max(h, w) >= 6
                 or min(h, w) == 3 and max(h, w) >= 8]
    for seed, sample in enumerate(samples):
        rng = np.random.default_rng(seed)
        h, w = permitted[int(rng.integers(len(permitted)))]
        n_colors = sample['metadata']['requested_palette_size']
        assert np.shape(sample['input']) == (h, w)
        assert np.shape(sample['output']) == (h, w)
        assert len(np.unique(sample['input'])) == n_colors
        assert sample['metadata']['size_attempts'] == 1
        heights.append(h)
        widths.append(w)
        palettes.append(n_colors)
    assert set(heights) == set(widths) == set(range(3, 31))
    assert abs(np.corrcoef(heights, widths)[0, 1]) < .15
    assert min(palettes) == 3 and max(palettes) >= 8
    assert set(palettes) <= set(range(3, 11))


def test_varied_hosts_multiple_pieces_and_leftovers_remain_common(samples):
    metadata = [sample['metadata'] for sample in samples]
    families = Counter(family for m in metadata for family in m['host_families'])
    assert set(families) == {'rect', 'chamfer', 'diamond', 'L', 'T', 'plus', 'step', 'union'}
    assert families['rect']/sum(families.values()) < .45
    assert .50 < np.mean([m['n_multi_piece_holes'] > 0 for m in metadata]) < .72
    assert .25 < np.mean([m['n_multi_piece_holes'] >= 2 for m in metadata]) < .45
    assert .13 < np.mean([m['n_unfilled_holes'] > 0 for m in metadata]) < .30
    assert any(m['n_unfilled_holes'] == 2 for m in metadata)
    assert .25 < np.mean([m['n_strays'] > 0 for m in metadata]) < .48
    assert .15 < np.mean([m['background'] == 'sparse' for m in metadata]) < .36
    assert all(m['multicolor_objects'] == 0 for m in metadata)
    assert sum(max(m['pieces_per_hole']) >= 3 for m in metadata) >= 15
    assert any(max(m['pieces_per_hole']) >= 4 for m in metadata)


def test_objects_and_cavities_grow_with_the_canvas(samples):
    small = [m for x in samples if (m := x['metadata'])['height']*m['width'] < 150]
    large = [m for x in samples if (m := x['metadata'])['height']*m['width'] >= 600]
    assert len(small) > 50 and len(large) > 25
    assert np.mean([np.mean(m['hole_areas']) for m in large]) > 2*np.mean([np.mean(m['hole_areas']) for m in small])
    for group in (small, large):
        assert .28 < np.mean([m['foreground_fraction'] for m in group]) < .47
    assert max(max(m['hole_areas']) for m in large) >= 50


def test_exactly_fillable_holes_are_filled_and_texture_is_unchanged(samples):
    for sample in samples[::5]:
        grid, output = np.array(sample['input']), np.array(sample['output'])
        scene = parse_scene(grid)
        semantic = {int(grid[p]) for obj in scene.hosts+scene.pieces for p in obj.cells}
        assert semantic | {scene.background} == set(map(int, grid.flat))
        host_colors = {int(grid[p]) for obj in scene.hosts for p in obj.cells}
        piece_colors = {int(grid[p]) for obj in scene.pieces for p in obj.cells}
        assert host_colors.isdisjoint(piece_colors)
        assert {int(grid[p]) for p in scene.texture} <= piece_colors
        assert all(len({int(grid[p]) for p in obj.cells}) == 1 for obj in scene.hosts + scene.pieces)
        assert sample['metadata']['background_kind'] in {
            'uniform', 'lattice', 'dotted_stripes', 'sparse_checker', 'dotted_border', 'dotted_diagonal'}
        holes = set().union(*scene.holes)
        source_cells = set().union(*(piece.cells for piece in scene.pieces))
        filled = 0
        for hole in scene.holes:
            fits = bool(local_covers(hole, [piece.cells for piece in scene.pieces]))
            assert all((output[p] != scene.background) == fits for p in hole)
            filled += fits
        assert sample['metadata']['n_filled_holes'] == filled
        assert sample['metadata']['n_unfilled_holes'] == len(scene.holes)-filled
        assert all(output[p] == grid[p] for host in scene.hosts for p in host.cells)
        assert all(output[p] == grid[p] for p in scene.texture)
        for r, c in zip(*np.where(grid != output)):
            assert (r, c) in holes or (r, c) in source_cells
        for piece in scene.pieces:
            old = [int(grid[p]) for p in sorted(piece.cells)]
            new = [int(output[p]) for p in sorted(piece.cells)]
            assert new == old or set(new) == {scene.background}


def test_enclosed_singleton_cannot_be_overwritten_as_empty_space():
    grid = np.zeros((12, 16), dtype=int)
    grid[1:6, 1:6] = 2
    grid[2:5, 2:5] = 0
    grid[2:5, 10:13] = 3
    output = grid.copy()
    output[2:5, 2:5] = 3
    output[2:5, 10:13] = 0
    validate_fit_holes(grid, output)
    # Isolated texture must stay fixed, so this cavity is no longer empty.
    grid[3, 3] = 9
    with pytest.raises(ValueError, match='nonempty enclosed material'):
        validate_fit_holes(grid, output)


@pytest.mark.parametrize('height,width', [(3, 8), (4, 6), (5, 5)])
def test_compact_boundary_witnesses_keep_closed_holes_and_identifiable_ground(height, width):
    grid = np.zeros((height, width), dtype=np.uint8)
    if height == 4:
        grid[:, :3] = 2
        grid[1:3, 1] = 0
        grid[1:3, 5] = 3
        target, source = [(1, 1), (2, 1)], [(1, 5), (2, 5)]
    else:
        grid[:3, :4] = 2
        grid[1, 1:3] = 0
        pr, pc = (1, 6) if height == 3 else (4, 2)
        grid[pr, pc:pc+2] = 3
        target, source = [(1, 1), (1, 2)], [(pr, pc), (pr, pc+1)]
    grid[0, 0] = 0  # One chamfered corner; diagonal air does not open the hole.
    expected = grid.copy()
    for destination, origin in zip(target, source):
        expected[destination] = grid[origin]
        expected[origin] = 0
    scene = parse_scene(grid)
    assert len(scene.hosts[0].cells) == 9 and scene.background_margin >= 1.25
    assert validate_fit_holes(grid, expected)['distinct_outputs'] == 1


def test_parent_silhouettes_are_often_surrounded_by_visible_background(samples):
    roomy = [e for e in samples if min(np.shape(e['input'])) >= 9 and max(np.shape(e['input'])) >= 12]
    assert sum(e['metadata']['hosts_touching_border'] > 0 for e in roomy) / len(roomy) < .25
    assert any(e['metadata']['hosts_touching_border'] > 0 for e in samples)
    for e in samples:
        m = e['metadata']
        assert m['n_colors'] <= 1 + m['n_hosts'] + m['n_loose_pieces']


def test_count_ranges_expand_without_changing_selected_budgets(samples):
    metadata = [sample['metadata'] for sample in samples]
    assert {m['n_hosts'] for m in metadata} == set(range(1, 7))
    assert {m['n_holes'] for m in metadata} == set(range(1, 13))
    assert max(m['n_loose_pieces'] for m in metadata) >= 20
    assert np.mean([m['n_hosts'] >= 2 for m in metadata]) > .32
    assert np.mean([m['n_holes'] >= 5 for m in metadata]) > .17
    assert np.mean([m['n_holes'] == 1 for m in metadata]) < .43
    assert np.mean([m['attempts'] for m in metadata]) < 3
    for m in metadata:
        assert m['requested_hosts'] == m['n_hosts'] <= m['capacity_hosts']
        assert m['requested_holes'] == m['n_holes'] <= m['capacity_holes']
        assert m['requested_pieces'] == m['n_loose_pieces'] <= 24
        assert m['requested_palette_size'] == m['n_colors']


def test_footprints_allow_staggered_parent_and_piece_boxes():
    # Bounding boxes overlap, while the actual drawn L leaves a generous place
    # for a second object and its full moat. A bbox-only packer rejects this.
    parent = frozenset((r, c) for r in range(8) for c in range(8) if r < 2 or c < 2)
    insert = frozenset((r, c) for r in range(3) for c in range(3))
    placed = _place_joint(np.random.default_rng(28), [parent, insert], 8, 8)
    assert placed is not None
    assert not placed[0] & placed[1]
    assert any(r >= 3 and c >= 3 for r, c in placed[1])


@pytest.mark.parametrize('height,width,hosts,holes', [
    (30, 30, 1, 12), (30, 30, 4, 12), (30, 30, 6, 12),
    (8, 30, 2, 4), (12, 30, 3, 6), (20, 20, 3, 6),
])
def test_high_count_plans_have_real_cavity_and_source_footprint_capacity(height, width, hosts, holes):
    rng = np.random.default_rng(280000+hosts+height)
    assert (holes, hosts) in _count_capacity(height, width)
    plan = _draw_plan(rng, height, width)
    plan.update(hosts=hosts, holes=holes, holes_per_host=[holes//hosts]*hosts,
                pieces_per_hole=[2]*holes, strays=0, palette_size=3,
                inset=int(height >= 9), texture_colors=0, colored_fits=False, broad_cavities=False)
    for _ in range(40):
        built = _build(rng, height, width, plan)
        if built is None:
            continue
        grid, output, metadata = built
        try:
            stats = validate_fit_holes(grid, output)
        except ValueError:
            continue
        assert stats['fillable_holes'] == metadata['n_holes'] == holes
        assert metadata['n_hosts'] == hosts
        assert metadata['n_loose_pieces'] == 2*holes
        break
    else:
        pytest.fail('The selected high-count footprint plan could not be built')


def test_multicolor_fits_and_selected_backgrounds_are_preserved(samples):
    multicolor = 0
    for sample in samples:
        metadata = sample['metadata']
        scene = parse_scene(sample['input'])
        output = np.asarray(sample['output'])
        count = sum(len({int(output[p]) for p in hole}) > 1 for hole in scene.holes)
        assert count == metadata['n_multicolor_filled_holes']
        assert bool(count) == metadata['colored_fit_budget']
        assert metadata['requested_background'] == metadata['background']
        multicolor += count > 0
    assert .07 < multicolor/len(samples) < .25


def test_failed_selected_sparse_background_rejects_the_geometry(monkeypatch):
    import generators.fit_holes_28 as fitting
    calls = []

    def cannot_paint(*args, **kwargs):
        calls.append(True)
        return None

    monkeypatch.setattr(fitting, 'apply_background', cannot_paint)
    rng = np.random.default_rng(2800912)
    plan = _draw_plan(rng, 20, 24)
    plan.update(hosts=1, holes=1, holes_per_host=[1], pieces_per_hole=[1],
                strays=0, palette_size=3, texture_colors=1, colored_fits=False)
    for _ in range(10):
        assert _build(rng, 20, 24, plan) is None
    assert calls


def test_broad_cavity_tail_keeps_large_simple_scenes(samples):
    broad = [sample for sample in samples if sample['metadata']['broad_cavity_budget']]
    assert broad
    assert max(max(sample['metadata']['hole_areas']) for sample in broad) >= 100
    for sample in broad:
        metadata = sample['metadata']
        assert metadata['n_hosts'] == metadata['n_holes'] == 1
        assert metadata['n_strays'] == 0
        assert min(metadata['height'], metadata['width']) >= 12


@pytest.mark.parametrize('seed,dimensions,hosts,holes,parts,palette', [
    (2433, (27, 5), 2, 4, [2, 2, 1, 2], 3),
    (2897, (20, 14), 4, 6, [1, 1, 1, 4, 1, 3], 6),
    (4079, (20, 7), 3, 4, [2, 1, 2, 2], 5),
    (5381, (5, 16), 2, 2, [2, 1], 4),
    (5718, (15, 5), 1, 2, [1, 1], 5),
    (6058, (5, 25), 1, 4, [1, 1, 2, 2], 4),
    (7506, (5, 15), 1, 2, [1, 1], 5),
    (8127, (5, 15), 1, 2, [2, 2], 5),
    (8461, (11, 7), 2, 2, [0, 1], 4),
    (8537, (5, 15), 2, 2, [1, 1], 6),
    (8648, (20, 14), 4, 6, [3, 3, 2, 2, 1, 3], 3),
    (8673, (11, 7), 2, 2, [0, 2], 3),
    (9132, (24, 5), 3, 3, [1, 2, 2], 5),
    (9919, (7, 14), 1, 3, [2, 1, 1], 3),
    (100595, (15, 7), 2, 3, [2, 2, 2], 6),
    (101328, (5, 19), 1, 3, [1, 2, 2], 4),
    (101927, (5, 25), 1, 4, [1, 1, 1, 2], 5),
])
def test_two_strays_fit_the_actual_capacity_without_reducing_the_selected_plan(
        seed, dimensions, hosts, holes, parts, palette):
    from tests.test_task28_uniqueness import compressed_oracle_outputs
    sample = generate_fit_holes(np.random.default_rng(seed))
    grid, output = np.asarray(sample['input']), np.asarray(sample['output'])
    metadata = sample['metadata']
    assert grid.shape == dimensions
    assert metadata['n_hosts'] == metadata['requested_hosts'] == hosts
    assert metadata['n_holes'] == metadata['requested_holes'] == holes
    assert metadata['pieces_per_hole'] == parts
    assert metadata['n_strays'] == 2
    assert metadata['n_loose_pieces'] == metadata['requested_pieces'] == sum(parts)+2
    assert metadata['n_colors'] == metadata['requested_palette_size'] == palette
    scene = parse_scene(grid)
    strays = [piece for piece in scene.pieces if all(output[p] == grid[p] for p in piece.cells)]
    assert len(strays) == 2
    # Every compact stray stays inside the footprint reserved for one source.
    for piece in strays:
        rows, cols = zip(*piece.cells)
        assert max(rows)-min(rows)+1 <= 2
        assert max(cols)-min(cols)+1 <= 2
    legal = compressed_oracle_outputs(grid)
    assert len(legal) == 1 and np.array_equal(legal[0], output)
