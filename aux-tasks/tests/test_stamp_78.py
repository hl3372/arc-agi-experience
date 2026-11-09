"""Visible-rule readers, preservation, clipping and drawing coverage for 78."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import re

import numpy as np
import pytest

from generators import stamp_78 as module
from prompts import get_prompt
from registry import generate


def _read(inp, prompt):
    match = re.search(r"canvas cells of color (\S+) are markers", prompt)
    assert match and "center cell of the stamp grid" in prompt
    assert "whole stamp is opaque" in prompt
    assert "outside the stamped footprints unchanged" in prompt
    pattern, canvas = inp["stamp"], inp["canvas"]
    marker = match[1]
    if isinstance(canvas[0][0], int):
        marker = int(marker)
    rows, cols = len(pattern), len(pattern[0])
    assert rows % 2 == cols % 2 == 1
    centers = [(r, c) for r, row in enumerate(canvas) for c, value in enumerate(row) if value == marker]
    assert centers
    writes = {}
    for r, c in centers:
        for a, row in enumerate(pattern):
            for b, value in enumerate(row):
                position = (r + a - rows // 2, c + b - cols // 2)
                if 0 <= position[0] < len(canvas) and 0 <= position[1] < len(canvas[0]):
                    assert position not in writes
                    writes[position] = value
    return [[writes.get((r, c), value) for c, value in enumerate(row)] for r, row in enumerate(canvas)]


@pytest.fixture(scope="module")
def examples():
    return [generate("78", seed) for seed in range(1000)]


def test_independent_reader(examples):
    for ex in examples[:400]:
        assert _read(ex["input"], get_prompt("78", ex["metadata"])) == ex["output"]


def test_size_and_count_coverage(examples):
    sizes = np.asarray([np.shape(ex["input"]["canvas"]) for ex in examples])
    assert set(sizes[:, 0]) == set(sizes[:, 1]) == set(range(5, 31))
    assert abs(np.corrcoef(sizes.T)[0, 1]) < .12
    assert np.mean(sizes[:, 0] == sizes[:, 1]) < .065
    assert np.mean(np.prod(sizes, axis=1) >= 700) > .025
    stamps = np.asarray([np.shape(ex["input"]["stamp"]) for ex in examples])
    assert np.all(stamps % 2 == 1)
    assert min(stamps.flat) == 3 and max(stamps.flat) >= 25
    assert np.mean(stamps[:, 0] != stamps[:, 1]) > .6
    assert np.mean(np.max(stamps, axis=1) > 7) > .5
    counts = Counter(ex["metadata"]["n_stamps"] for ex in examples)
    assert set(counts) == set(range(1, 15))
    assert .15 < counts[1] / len(examples) < .40
    assert .65 < sum(counts[n] for n in (1, 2, 3, 4)) / len(examples) < .88
    assert sum(n for k, n in counts.items() if k > 6) > 65


def test_family_palette_and_composition_plans(examples):
    families, fields, compositions, palettes = Counter(), Counter(), Counter(), Counter()
    for ex in examples:
        m = ex["metadata"]
        families[m["motif_family"]] += 1
        fields[m["background_style"]] += 1
        compositions[m["composition"]] += 1
        palettes[m["n_colors"]] += 1
        assert m["planned_motif_family"] == m["motif_family"]
        assert m["planned_background_style"] == m["background_style"]
        assert m["planned_n_stamps"] == m["n_stamps"] <= m["witnessed_capacity"]
        assert m["attempts"] == 1
        assert len(m["stamp_palette"]) in (2, 3, 4)
        assert set(np.unique(ex["input"]["stamp"])) == set(m["stamp_palette"])
        assert set(np.unique(ex["input"]["canvas"])) == set(m["canvas_palette"]) | {m["marker_color"]}
        assert "noisy" not in m["background_style"]
    assert set(families) == {"rectangle", "bar", "L", "T", "U", "H", "plus", "frame", "staircase", "diamond", "chamfer"}
    assert min(families.values()) >= 15
    assert set(fields) == set(module.CANVAS_STYLES)
    assert .45 < fields["solid"] / len(examples) < .65
    assert set(compositions) == set(module.COMPOSITIONS)
    assert min(compositions.values()) > 150
    assert set(palettes) == set(range(3, 11))  # Two stamp colors plus the reserved marker.
    assert sum(palettes[c] for c in (3, 4, 5)) > 500
    assert sum(palettes[c] for c in (8, 9, 10)) > 65
    marker_colors = Counter(ex["metadata"]["marker_color"] for ex in examples)
    assert set(marker_colors) == set(range(10)) and min(marker_colors.values()) > 60


def test_scaled_motifs_and_not_just_opaque_padding(examples):
    ms = [ex["metadata"] for ex in examples]
    small = [m for m in ms if m["grid_h"] * m["grid_w"] < 150]
    large = [m for m in ms if m["grid_h"] * m["grid_w"] >= 700]
    occupancy = lambda group: np.mean([m["visible_motif_cells"] / (m["grid_h"] * m["grid_w"]) for m in group])
    assert len(large) > 20
    assert occupancy(large) > .12
    assert occupancy(large) > .65 * occupancy(small)
    ordinary = [m for m in ms if max(m["stamp_h"], m["stamp_w"]) <= 7]
    expanded = [m for m in ms if max(m["stamp_h"], m["stamp_w"]) >= 15]
    ink_share = lambda group: np.mean([m["motif_cells"] / (m["stamp_h"] * m["stamp_w"]) for m in group])
    assert ink_share(expanded) > .30
    assert ink_share(expanded) > .6 * ink_share(ordinary)
    assert max(m["motif_cells"] for m in ms) > 200


def test_complete_and_clipped_scenes_and_opacity(examples):
    modes = Counter()
    covered_detail = opaque = 0
    for ex in examples:
        m = ex["metadata"]
        modes[m["layout_mode"]] += 1
        assert m["planned_layout"] == m["layout_mode"]
        assert (m["n_clipped"] == 0) == (m["layout_mode"] == "complete")
        assert m["n_complete"] + m["n_clipped"] == m["n_stamps"]
        assert min(m["visible_motif_cells_per_copy"]) >= m["minimum_visible_motif_cells"] >= 1
        assert m["outside_preserved"] and m["output_changed_cells"] > 0
        assert m["marker_color"] not in np.asarray(ex["output"])
        covered_detail += m["covered_detail_cells"] > 0
        opaque += m["opaque_vs_transparent_changes"]
    assert .45 < modes["complete"] / len(examples) < .65
    assert covered_detail > 300
    assert opaque > 550


def test_arbitrary_symbols_and_repeatability(examples):
    for seed, ex in enumerate(examples[:100]):
        palette = np.random.default_rng(seed + 7812).permutation(list("QWERTYUIOP")).tolist()
        inp = {name: [[palette[c] for c in row] for row in grid] for name, grid in ex["input"].items()}
        output = [[palette[c] for c in row] for row in ex["output"]]
        prompt = get_prompt("78", ex["metadata"], color_palette=palette)
        assert _read(inp, prompt) == output
        assert re.search(r"color \d\b", prompt) is None
        assert ex == generate("78", seed)


@pytest.mark.parametrize("fault", ("output", "outside", "marker", "center_metadata", "even_stamp", "count", "palette", "visible_motif"))
def test_validator_rejects_tampering(fault):
    ex = deepcopy(generate("78", 34))
    if fault == "output":
        r, c = ex["metadata"]["stamp_centers"][0]
        ex["output"][r][c] = (ex["output"][r][c] + 1) % 10
    elif fault == "outside":
        h, w = ex["metadata"]["stamp_h"], ex["metadata"]["stamp_w"]
        r, c = next((r, c) for r in range(len(ex["output"])) for c in range(len(ex["output"][0]))
                    if all(abs(r - mr) > h // 2 or abs(c - mc) > w // 2 for mr, mc in ex["metadata"]["stamp_centers"]))
        ex["output"][r][c] = (ex["output"][r][c] + 1) % 10
    elif fault == "marker":
        ex["metadata"]["marker_color"] = ex["metadata"]["stamp_palette"][0]
    elif fault == "center_metadata":
        ex["metadata"]["stamp_centers"][0][0] += 1
    elif fault == "even_stamp":
        ex["input"]["stamp"].pop()
    elif fault == "count":
        ex["metadata"]["planned_n_stamps"] += 1
    elif fault == "palette":
        ex["metadata"]["n_colors"] += 1
    else:
        ex["metadata"]["minimum_visible_motif_cells"] = 900
    with pytest.raises(ValueError):
        module.validate_stamp(ex)


def test_final_grid_reader_discovers_centers_and_rejects_overlap():
    stamp = [[0, 2, 0], [2, 2, 2], [0, 2, 0]]
    canvas = np.full((5, 7), 0)
    canvas[2, 1] = canvas[2, 5] = 3
    canvas[0, 3] = 4
    a = module.solve_stamp(stamp, canvas, 3)
    b = module.solve_stamp(stamp, canvas, 4)
    assert not np.array_equal(a, b)  # A second isolated color is harmless when the marker is named.
    assert a[0, 3] == 4
    canvas[2, 2] = 3
    with pytest.raises(ValueError, match="overlap"):
        module.solve_stamp(stamp, canvas, 3)


def test_no_retry_into_an_easier_scene(monkeypatch):
    def no_witness(rng, plan):
        raise RuntimeError("No witness for the fixed plan")
    monkeypatch.setattr(module, "_placement_witness", no_witness)
    with pytest.raises(RuntimeError, match="fixed plan"):
        generate("78", 42)


@pytest.mark.parametrize("seed", (5, 34, 64, 3499, 9085))
def test_old_ambiguity_clipping_and_size_regressions(seed):
    ex = generate("78", seed)
    assert _read(ex["input"], get_prompt("78", ex["metadata"])) == ex["output"]
    assert min(ex["metadata"]["visible_motif_cells_per_copy"]) > 0
