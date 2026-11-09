"""Tasks 77/79/86: prompt-only readers, misreading guards and coverage contracts."""
from collections import Counter, defaultdict
from copy import deepcopy
import re
import time

import numpy as np
import pytest

from generators import grid_slices_77_79_86 as impl
from prompts import PROMPT_TEMPLATES, get_prompt
from registry import RETIRED, generate, list_tasks

TASKS = ("77", "79", "86")
INDEX_NOTE = "Rows and columns are 0-indexed from the top-left corner."


# ---------------------------------------------------------------------------
# An independent reader: the prompt text and the visible grid, nothing else
# ---------------------------------------------------------------------------

def _span(text, noun):
    """'row 3' -> (3, 3); 'rows 3 through 8' -> (3, 8)."""
    m = re.fullmatch(rf"{noun}s (\d+) through (\d+)|{noun} (\d+)", text)
    if m.group(3) is not None:
        return int(m.group(3)), int(m.group(3))
    return int(m.group(1)), int(m.group(2))


def request(prompt):
    """(row_lo, row_hi, col_lo, col_hi) or None for 'every', read from the prompt alone."""
    assert INDEX_NOTE in prompt
    m = re.match(r"Output (row|column) (\d+) of the grid as a grid with one \1,", prompt)
    if m:
        i = int(m.group(2))
        return (i, i, None, None) if m.group(1) == "row" else (None, None, i, i)
    m = re.match(r"Output (rows|columns) (\d+) through (\d+) of the grid, .* The range includes both ends\.", prompt)
    if m:
        a, b = int(m.group(2)), int(m.group(3))
        return (a, b, None, None) if m.group(1) == "rows" else (None, None, a, b)
    m = re.match(r"Output the rectangle where (.+?) intersects? (.+?), keeping .* "
                 r"(Ranges include|The range includes) both ends\.", prompt)
    assert m, prompt
    return (*_span(m.group(1), "row"), *_span(m.group(2), "column"))


def cut(grid, req):
    r0, r1, c0, c1 = req
    h, w = len(grid), len(grid[0])
    r0, r1 = (0, h - 1) if r0 is None else (r0, r1)
    c0, c1 = (0, w - 1) if c0 is None else (c0, c1)
    return [[grid[r][c] for c in range(c0, c1 + 1)] for r in range(r0, r1 + 1)]


def misreadings(grid, req):
    """Every rival reading of the request that fits on the grid, written out per axis."""
    h, w = len(grid), len(grid[0])

    def axis(lo, hi, length):
        if lo is None:
            return [(None, None)]
        opts = [(lo, hi), (lo - 1, hi - 1), (length - 1 - hi, length - 1 - lo), (length - hi, length - lo),
                (lo, hi - 1), (lo + 1, hi + 1)]          # exclusive end, and a plain slip forward
        return [(a, b) for a, b in opts if 0 <= a <= b <= length - 1]
    r0, r1, c0, c1 = req
    out = {(a, b, c, d) for (a, b) in axis(r0, r1, h) for (c, d) in axis(c0, c1, w)}
    if None not in req and c1 <= h - 1 and r1 <= w - 1:
        out.add((c0, c1, r0, r1))                        # transposed: row numbers read as columns
    out.discard((r0, r1, c0, c1))                        # a reading that lands on the request is no rival
    return out


def is_forward_slip(window, req):
    """Lines one further on along some axis: a counting slip, not a reading the prompt allows."""
    r0, r1, c0, c1 = req
    return ((r0 is not None and window[:2] == (r0 + 1, r1 + 1))
            or (c0 is not None and window[2:] == (c0 + 1, c1 + 1)))


# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task", TASKS)
def test_prompt_only_reader_agrees(task):
    for seed in range(400):
        ex = generate(task, seed)
        prompt = get_prompt(task, ex["metadata"])
        assert cut(ex["input"], request(prompt)) == ex["output"], (task, seed)


@pytest.mark.parametrize("task", TASKS)
def test_symbol_palette_changes_nothing_but_the_cells(task):
    for seed in range(40):
        palette = list("αβγδεζηθικ")
        np.random.default_rng(seed).shuffle(palette)
        ex = generate(task, seed)
        prompt = get_prompt(task, ex["metadata"], color_palette=palette)
        assert prompt == get_prompt(task, ex["metadata"])
        inp = [[palette[v] for v in row] for row in ex["input"]]
        assert cut(inp, request(prompt)) == [[palette[v] for v in row] for row in ex["output"]]


def test_the_rival_readings_that_are_guarded_always_give_another_answer():
    """1-indexed, counted from the far edge (0- and 1-indexed) and transposed: never the answer."""
    slips = defaultdict(list)
    for task in TASKS:
        for seed in range(700):
            ex = generate(task, seed)
            grid, out = ex["input"], ex["output"]
            req = request(get_prompt(task, ex["metadata"]))
            for window in misreadings(grid, req):
                other = cut(grid, window)
                if is_forward_slip(window, req):
                    slips[task].append(other != out)     # not guarded: measured below
                else:
                    assert other != out, (task, seed, window)
    # the unguarded forward slip still lands on a different answer most of the time
    assert np.mean(slips["77"]) > .6
    assert np.mean(slips["79"]) > .8
    assert np.mean(slips["86"]) > .8


@pytest.mark.parametrize("task", TASKS)
def test_coverage(task):
    metas = [generate(task, seed)["metadata"] for seed in range(1500)]
    H = np.array([m["grid_h"] for m in metas])
    W = np.array([m["grid_w"] for m in metas])
    assert set(H) == set(W) == set(range(5, 31))
    assert abs(np.corrcoef(H, W)[0, 1]) < .08
    assert np.mean(H == W) < .08
    assert np.mean(H * W >= 700) > .025
    colours = Counter(m["n_colors"] for m in metas)
    assert set(colours) == set(range(2, 11))
    assert .2 < sum(colours[k] for k in (8, 9, 10)) / len(metas) < .45
    assert max(colours.values()) / len(metas) < .18      # a mild lean to few colours, no spike
    assert colours[10] / len(metas) < .13                 # no pile-up at ten (a texture on nine)
    families = Counter(m["content_family"] for m in metas)
    textured = np.mean([m["texture_kind"] is not None for m in metas])
    assert all(m["texture_named"] is False for m in metas)
    assert len({m["texture_kind"] for m in metas} - {None}) == 7
    if task == "77":
        assert set(families) == {"shapes", "multicolor_shapes"}
        assert .45 < textured < .65
    else:
        assert set(families) == set(impl.FAMILIES)
        assert .10 < textured < .22
    assert np.mean([m["draws"] for m in metas]) < 1.2


@pytest.mark.parametrize("task", ["77", "79"])
def test_axis_is_a_fair_coin_the_input_cannot_predict(task):
    metas = [generate(task, seed)["metadata"] for seed in range(3000)]
    axes = np.array([m["axis"] == "row" for m in metas])
    assert .46 < axes.mean() < .54
    # a per-bin majority vote over input-only features, trained on half and scored on the rest
    features = {
        "shape": lambda m: (m["grid_h"] > m["grid_w"], m["grid_h"] // 6, m["grid_w"] // 6),
        "scene": lambda m: (m["content_family"], m["density"], m["texture_kind"], m["n_colors"]),
    }
    for name, f in features.items():
        votes = defaultdict(Counter)
        for m, a in zip(metas[:1500], axes[:1500]):
            votes[f(m)][a] += 1
        hits = [votes[f(m)].most_common(1)[0][0] == a if votes[f(m)] else a
                for m, a in zip(metas[1500:], axes[1500:])]
        assert np.mean(hits) < .56, (task, name, np.mean(hits))


def _rel(i, length):
    return i / (length - 1)


def test_request_positions_reach_the_far_edges():
    """Deep lines are weighted up (DEPTH_GAMMA) without piling every request on the far edge."""
    m77 = [generate("77", s)["metadata"] for s in range(1500)]
    idx = Counter(m["index"] for m in m77)
    assert max(idx) >= 28
    assert sum(v for k, v in idx.items() if k >= 20) / len(m77) > .11   # ~14% (uniform: ~8%)
    assert sum(v for k, v in idx.items() if k >= 25) / len(m77) > .03   # ~5% (uniform: ~2%)
    rel = [_rel(m["index"], m["grid_h"] if m["axis"] == "row" else m["grid_w"]) for m in m77]
    assert .53 < np.mean(rel) < .62                                         # a lean, not a pile-up
    short = [m for m in m77 if (m["grid_h"] if m["axis"] == "row" else m["grid_w"]) <= 8]
    assert np.mean([m["index"] == 0 for m in short]) > .1                  # short axes stay near uniform
    assert .03 < np.mean([m["line_monochrome"] for m in m77]) < .12
    assert np.mean([m["informative_share"] for m in m77]) > .72      # the guard selects from most lines
    m79 = [generate("79", s)["metadata"] for s in range(1500)]
    assert max(m["start"] for m in m79) >= 22 and max(m["end"] for m in m79) == 29
    assert min(m["span"] for m in m79) == 2 and max(m["span"] for m in m79) >= 28
    assert 0 < np.mean([m["full_range"] for m in m79]) < .05
    assert np.mean([m["informative_share"] for m in m79]) > .8
    assert np.mean([m["start"] >= 15 for m in m79]) > .09                 # ~12% (uniform pairs: ~5.5%)
    assert np.mean([m["start"] >= 20 for m in m79]) > .03                 # ~5% (uniform pairs: ~1.3%)
    modes = Counter(m["range_mode"] for m in m79)
    assert .06 < modes["wide"] / len(m79) < .14 and modes["full"] / len(m79) < .05
    # the lean reaches the far edge more often than the near one, without deserting row 0
    near = np.mean([m["start"] <= 2 for m in m79])
    far = np.mean([m["end"] >= m["axis_length"] - 3 for m in m79])
    assert 1.0 < far / near < 2.5, (near, far)
    m86 = [generate("86", s)["metadata"] for s in range(1500)]
    assert max(m["row_end"] for m in m86) == 29 and max(m["col_end"] for m in m86) == 29
    assert min(m["row_start"] for m in m86) == 0
    assert .005 < np.mean([m["single_row"] for m in m86]) < .05
    assert .005 < np.mean([m["single_column"] for m in m86]) < .05
    assert not any(m["single_row"] and m["single_column"] for m in m86)
    for seed in range(300):
        m = generate("86", seed)["metadata"]
        prompt = get_prompt("86", m)
        single = m["single_row"] or m["single_column"]
        assert ("The range includes both ends." in prompt) == single
        assert ("Ranges include both ends." in prompt) == (not single)
    assert not any(m["full_height"] and m["full_width"] for m in m86)
    assert np.mean([m["crop_monochrome"] for m in m86]) < .04
    assert max(m["output_h"] * m["output_w"] for m in m86) > 500
    # big crops (the wide mode) and deep crops (the depth lean) both keep a real share
    assert np.mean([max(m["output_h"], m["output_w"]) >= 20 for m in m86]) > .11      # ~15% (was ~8%)
    assert np.mean([m["output_h"] >= 10 and m["output_w"] >= 10 for m in m86]) > .10  # ~14% (was ~9%)
    assert np.mean([m["row_start"] >= 20 or m["col_start"] >= 20 for m in m86]) > .04
    assert .2 < np.mean([m["row_mode"] == "wide" for m in m86]) < .4


def test_one_seed_shows_different_pictures_under_the_three_tasks():
    same = [generate("79", s)["input"] == generate("86", s)["input"]
            or generate("77", s)["input"] == generate("79", s)["input"] for s in range(200)]
    assert sum(same) <= 1


def test_prompts_are_byte_identical_per_bucket_and_name_nothing():
    for task, n_wordings in (("77", 2), ("79", 2), ("86", 3)):
        wordings = set()
        for seed in range(600):
            prompt = get_prompt(task, generate(task, seed)["metadata"])
            assert "background" not in prompt.lower() and "color" not in prompt.lower()
            wordings.add(re.sub(r"\d+", "#", prompt))
        assert len(wordings) == n_wordings, (task, wordings)


def test_prompt_builders_raise_on_a_missing_or_unknown_variant():
    meta = dict(generate("77", 0)["metadata"])
    meta["axis"] = "diagonal"
    with pytest.raises(ValueError):
        get_prompt("77", meta)
    del meta["axis"]
    with pytest.raises(KeyError):
        get_prompt("77", meta)


def test_validators_raise_on_tampering():
    for task, validator in (("77", impl.validate_77), ("79", impl.validate_79), ("86", impl.validate_86)):
        ex = generate(task, 3)
        bad = deepcopy(ex)
        bad["output"][0][0] = (bad["output"][0][0] + 1) % 10
        with pytest.raises(ValueError):
            validator(bad)
        bad = deepcopy(ex)
        bad["metadata"]["color_counts"][0] += 1
        with pytest.raises(ValueError):
            validator(bad)
        bad = deepcopy(ex)
        bad["input"][0][0] = 10
        with pytest.raises(ValueError):
            validator(bad)


def test_validator_rejects_a_request_whose_misreading_gives_the_same_answer():
    # make the 1-indexed rival line a copy of the requested line
    for seed in range(200):
        ex = generate("77", seed)
        m = ex["metadata"]
        if m["axis"] == "row" and m["index"] >= 1:
            break
    bad = deepcopy(ex)
    bad["input"][m["index"] - 1] = list(bad["input"][m["index"]])
    counts = [0] * 10
    for row in bad["input"]:
        for v in row:
            counts[v] += 1
    bad["metadata"]["color_counts"] = counts
    bad["metadata"]["n_colors"] = sum(1 for n in counts if n)
    with pytest.raises(ValueError, match="same answer"):
        impl.validate_77(bad)


def test_rival_helpers_match_their_definitions():
    assert impl.range_rivals(10, 3, 1) == [2, 6, 7]          # 1-indexed, far edge 0/1-indexed
    assert impl.range_rivals(10, 0, 1) == [9]                # near edge: 9 = 10-1-0; 10-0 is off-grid
    assert impl.range_rivals(10, 2, 3) == [1, 5, 6]          # rows 2..4 -> 1..3, 5..7, 6..8
    assert impl.range_rivals(10, 0, 10) == []                # the whole axis has no rival
    assert (2, 5, 4, 2) in impl.crop_rivals(10, 10, 5, 2, 2, 4)   # transposed reading of a 2x4 crop


def test_generation_is_cheap():
    for task in TASKS:
        t = time.perf_counter()
        for seed in range(300):
            generate(task, seed)
        assert (time.perf_counter() - t) / 300 < .010, task


def test_ids_77_and_86_are_active_again():
    assert "77" not in RETIRED and "86" not in RETIRED
    for task in TASKS:
        assert task in list_tasks() and task in PROMPT_TEMPLATES
