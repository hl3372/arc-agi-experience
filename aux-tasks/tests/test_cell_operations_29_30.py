"""Visible-input semantics and distribution contracts for the rebuilt cell tasks."""
from collections import Counter

import numpy as np
import pytest

from generators.cell_operations_29_30 import (
    generate_recolor, generate_count, validate_recolor, validate_count,
)


@pytest.fixture(scope="module")
def scenes():
    return {
        task: [fn(np.random.default_rng(seed)) for seed in range(2000)]
        for task, fn in (("29", generate_recolor), ("30", generate_count))
    }


def test_visible_input_answers_and_palette_permutations(scenes):
    for task, rows in scenes.items():
        for seed, ex in enumerate(rows):
            g, m = ex["input"], ex["metadata"]
            if task == "29":
                source, target = m["source_color"], m["target_color"]
                expected = [[target if cell == source else cell for cell in row] for row in g]
                assert ex["output"] == expected, seed
            else:
                query = m["query_color"]
                expected = sum(row.count(query) for row in g)
                assert ex["output"] == str(expected), seed
            if seed >= 300:
                continue
            p = np.random.default_rng(seed + 80000).permutation(10).tolist()
            permuted = [[p[x] for x in row] for row in g]
            if task == "29":
                out = [[p[x] for x in row] for row in ex["output"]]
                validate_recolor(permuted, out, p[source], p[target])
            else:
                validate_count(permuted, ex["output"], p[query])


def test_size_palette_and_content_coverage(scenes):
    for task, rows in scenes.items():
        n = len(rows)
        hs = [len(ex["input"]) for ex in rows]
        ws = [len(ex["input"][0]) for ex in rows]
        assert set(hs) == set(ws) == set(range(1, 31))
        assert abs(np.corrcoef(hs, ws)[0, 1]) < .1
        assert .04 < sum(min(h, w) == 1 for h, w in zip(hs, ws)) / n < .09
        assert .02 < sum(h * w >= 700 for h, w in zip(hs, ws)) / n < .065
        palette = Counter(len({cell for row in ex["input"] for cell in row}) for ex in rows)
        assert set(range(1, 11)) <= palette.keys()
        for k in range(2, 11):
            assert .055 < palette[k] / n < .16, (task, k, palette)
        families = Counter(ex["metadata"]["content_family"] for ex in rows)
        assert len(families) >= 12
        assert max(families.values()) / n < .2
        # The high palette tail is present on both modest and large grids.
        for lo, hi in ((80, 200), (400, 901)):
            eligible = [ex for ex in rows if lo <= len(ex["input"]) * len(ex["input"][0]) < hi]
            assert sum(ex["metadata"]["palette_size"] >= 8 for ex in eligible) / len(eligible) > .18


def test_recolor_includes_erasure_and_absent_source(scenes):
    rows = scenes["29"]
    identity = sum(ex["input"] == ex["output"] for ex in rows) / len(rows)
    erase = sum(ex["metadata"]["target_color"] == ex["metadata"]["background_color"] for ex in rows) / len(rows)
    assert .05 <= identity <= .11
    assert erase > .15
    assert all(ex["metadata"]["source_color"] != ex["metadata"]["target_color"] for ex in rows)


def test_count_has_small_zero_share_and_a_real_tail(scenes):
    rows = scenes["30"]
    answers = [int(ex["output"]) for ex in rows]
    assert .035 <= answers.count(0) / len(rows) <= .085
    assert max(answers) >= 700
    # Uniform choice among visible colours removes the former dominant-colour
    # shortcut.  Retain a substantial tail without preferentially querying
    # that largest region just to inflate the large-answer percentage.
    assert sum(x >= 100 for x in answers) / len(rows) > .12
    assert len(set(answers)) > 250
    large = [int(ex["output"]) for ex in rows if len(ex["input"]) * len(ex["input"][0]) >= 400]
    assert np.percentile(large, 90) > 350


def test_query_is_not_biased_toward_the_largest_region(scenes):
    for task, rows in scenes.items():
        observed, expected, n = 0, 0., 0
        for ex in rows:
            counts = Counter(cell for row in ex["input"] for cell in row)
            m = ex["metadata"]
            query = m["query_color"] if task == "30" else m["source_color"]
            if query not in counts:
                continue
            largest = max(counts.values())
            observed += counts[query] == largest
            expected += sum(v == largest for v in counts.values()) / len(counts)
            n += 1
        assert abs(observed - expected) / n < .025, (task, observed, expected)


def test_large_scenes_have_varied_counts_across_colors(scenes):
    for task, rows in scenes.items():
        repeated_count_shares = []
        for ex in rows:
            counts = Counter(cell for row in ex['input'] for cell in row)
            if sum(counts.values()) < 300 or len(counts) < 4:
                continue
            # If most colors occupy the same number of cells, that number
            # answers the query too often without identifying its color.
            multiplicities = Counter(counts.values())
            repeated_count_shares.append(max(multiplicities.values()) / len(counts))
        assert len(repeated_count_shares) > 400
        assert np.mean(repeated_count_shares) < .25, task


def test_validators_reject_tampering():
    ex = generate_recolor(np.random.default_rng(17))
    out = [row[:] for row in ex["output"]]
    out[0][0] = (out[0][0] + 1) % 10
    with pytest.raises(ValueError, match="visible-input rule"):
        validate_recolor(ex["input"], out, ex["metadata"]["source_color"], ex["metadata"]["target_color"])
    ex = generate_count(np.random.default_rng(17))
    with pytest.raises(ValueError, match="visible input"):
        validate_count(ex["input"], str(int(ex["output"]) + 1), ex["metadata"]["query_color"])


def test_count_extremes_and_every_cell_semantics():
    assert generate_count(np.random.default_rng(3401))["output"] == "900"
    validate_count([[7] * 30 for _ in range(30)], "900", 7)
    validate_count([[7] * 30 for _ in range(30)], "0", 2)
    # A colour in a pattern or a single isolated pixel still counts.
    validate_count([[1, 2, 1], [2, 1, 2], [1, 2, 3]], "4", 1)
    validate_recolor([[1, 2, 1], [2, 1, 2]], [[0, 2, 0], [2, 0, 2]], 1, 0)
