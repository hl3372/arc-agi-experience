"""Task 102 -- report which of five symmetries a grid has.

The oracle below is written from the prompt text alone, with list slicing
rather than the generator's coordinate arithmetic, and the rule sentences it
relies on are asserted to be in the prompt.
"""
from collections import Counter
import json
import time

import numpy as np
import pytest

from generators import symmetry_report_102 as module
from prompts import get_prompt
from registry import generate
from symbol_remap import palette_for_example

ORACLE_SEEDS = 400
SWEEP_SEEDS = 1400

LABELS = ["horizontal_reflection", "vertical_reflection", "main_diagonal_reflection",
          "anti_diagonal_reflection", "rotational_180"]
CLASSES = {frozenset(), frozenset({LABELS[0]}), frozenset({LABELS[1]}),
           frozenset({LABELS[4]}), frozenset({LABELS[0], LABELS[1], LABELS[4]}),
           frozenset({LABELS[2]}), frozenset({LABELS[3]}),
           frozenset({LABELS[2], LABELS[3], LABELS[4]}), frozenset(LABELS)}
LIBRARY_FAMILIES = {"polyomino", "bar", "rect", "square", "frame", "L", "T", "U",
                    "plus", "H", "staircase", "blob", "big_blob", "scaled"}

RULE_SENTENCES = [
    "horizontal_reflection means row r, column c matches row height - 1 - r, column c",
    "vertical_reflection means row r, column c matches row r, column width - 1 - c",
    "main_diagonal_reflection means a square grid where row r, column c matches row c, column r",
    "anti_diagonal_reflection means a square grid where row r, column c matches "
    "row width - 1 - c, column height - 1 - r",
    "rotational_180 means row r, column c matches row height - 1 - r, column width - 1 - c",
]


@pytest.fixture(scope="module")
def sweep():
    return [generate("102", seed) for seed in range(SWEEP_SEEDS)]


# ---------------------------------------------------------------------------
# an oracle that reads the prompt, not the metadata
# ---------------------------------------------------------------------------

def _oracle(rows):
    """The five predicates by list slicing, the way the prompt reads."""
    h, w = len(rows), len(rows[0])
    square = h == w
    columns = [list(col) for col in zip(*rows)]
    answers = {
        LABELS[0]: rows == rows[::-1],
        LABELS[1]: all(row == row[::-1] for row in rows),
        LABELS[2]: square and columns == rows,
        LABELS[3]: square and [col[::-1] for col in columns][::-1] == rows,
        LABELS[4]: rows == [row[::-1] for row in rows[::-1]],
    }
    return "\n".join(f"{label}: {'yes' if answers[label] else 'no'}" for label in LABELS)


def test_the_prompt_carries_the_rules_the_oracle_uses():
    prompt = get_prompt("102", generate("102", 0)["metadata"])
    for sentence in RULE_SENTENCES:
        assert sentence in prompt, sentence


def test_oracle_agrees_with_the_answer():
    for seed in range(ORACLE_SEEDS):
        ex = generate("102", seed)
        assert _oracle(ex["input"]) == ex["output"], seed


def test_examples_are_deterministic_and_serializable():
    for seed in range(40):
        ex = generate("102", seed)
        json.dumps(ex)
        assert generate("102", seed) == ex, seed


def test_prompt_renders_under_the_symbol_palette():
    from sft_wrapper import format_single
    for seed in range(12):
        prompt, answer = format_single("102", seed, color_palette=palette_for_example("102", seed))
        assert prompt and answer
        # the labels are words, not colours: a symbol palette must not touch them
        assert answer == generate("102", seed)["output"]


def test_answer_format(sweep):
    for ex in sweep:
        lines = ex["output"].split("\n")
        assert len(lines) == 5
        for label, line in zip(LABELS, lines):
            assert line in (f"{label}: yes", f"{label}: no")
        assert ex["metadata"]["allowed_symmetries"] == LABELS


# ---------------------------------------------------------------------------
# every answer class, and the canvas each one needs
# ---------------------------------------------------------------------------

def test_all_nine_classes_occur_and_are_balanced(sweep):
    hist = Counter(frozenset(ex["metadata"]["present_labels"]) for ex in sweep)
    assert set(hist) == CLASSES, sorted(map(sorted, set(hist) ^ CLASSES))
    for labels, count in hist.items():
        assert 0.06 <= count / len(sweep) <= 0.17, (sorted(labels), count)
    all_no = hist[frozenset()] / len(sweep)
    assert 0.07 <= all_no <= 0.17, all_no
    # the two classes the old generator could never reach
    assert hist[frozenset({LABELS[2], LABELS[3], LABELS[4]})] >= 30
    assert hist[frozenset(LABELS)] >= 30


def test_only_the_diagonal_classes_force_a_square(sweep):
    coincidental = free = 0
    hv_rect = 0
    for ex in sweep:
        meta = ex["metadata"]
        square = meta["grid_h"] == meta["grid_w"]
        labels = set(meta["present_labels"])
        diagonal = bool(labels & {LABELS[2], LABELS[3]})
        assert diagonal == meta["requires_square"]
        if diagonal:
            assert square
        else:
            free += 1
            coincidental += square
            if labels == {LABELS[0], LABELS[1], LABELS[4]} and not square:
                hv_rect += 1
    assert coincidental / free <= 0.09, coincidental / free
    # the fixed bug: the horizontal-plus-vertical answer used to be square-only
    assert hv_rect >= 40, hv_rect


def test_size_coverage(sweep):
    sides = Counter()
    big = 0
    for ex in sweep:
        meta = ex["metadata"]
        h, w = meta["grid_h"], meta["grid_w"]
        assert 5 <= h <= 30 and 5 <= w <= 30, meta
        assert h * w <= 900
        sides[h] += 1
        sides[w] += 1
        big += h * w >= 700
    assert min(sides) == 5 and max(sides) == 30
    assert len(sides) == 26, sorted(sides)
    counts = [sides[s] for s in range(5, 31)]
    assert max(counts) <= 3.0 * min(counts), counts
    assert 0.04 <= big / len(sweep) <= 0.16, big / len(sweep)


def test_color_coverage(sweep):
    hist = Counter(ex["metadata"]["n_colors"] for ex in sweep)
    assert set(hist) <= set(range(2, 11)), hist
    assert min(hist) == 2 and max(hist) == 10, hist
    rich = sum(v for k, v in hist.items() if k >= 8) / len(sweep)
    assert rich >= 0.06, rich          # folding multiplies every body: capacity-bound
    assert hist[2] / len(sweep) >= 0.04, hist


def test_texture_and_occupancy(sweep):
    textured = [ex for ex in sweep if ex["metadata"]["texture_kind"] != "none"]
    share = len(textured) / len(sweep)
    assert 0.10 <= share <= 0.32, share
    for ex in textured:
        meta = ex["metadata"]
        assert min(meta["grid_h"], meta["grid_w"]) >= 8, meta
        assert meta["texture_named"] is False
    occupancy = [ex["metadata"]["occupancy"] for ex in sweep
                 if ex["metadata"]["grid_h"] * ex["metadata"]["grid_w"] >= 500]
    assert len(occupancy) >= 40
    assert sum(occupancy) / len(occupancy) >= 0.12, sum(occupancy) / len(occupancy)
    for ex in sweep:
        assert set(ex["metadata"]["body_families"]) <= LIBRARY_FAMILIES
        assert ex["metadata"]["n_bodies"] >= 1


# ---------------------------------------------------------------------------
# a "no" that turns on a handful of cells
# ---------------------------------------------------------------------------

def _violating(rows, label):
    h, w = len(rows), len(rows[0])
    if label in (LABELS[2], LABELS[3]) and h != w:
        return None
    bad = 0
    for r in range(h):
        for c in range(w):
            if label == LABELS[0]:
                rr, cc = h - 1 - r, c
            elif label == LABELS[1]:
                rr, cc = r, w - 1 - c
            elif label == LABELS[2]:
                rr, cc = c, r
            elif label == LABELS[3]:
                rr, cc = w - 1 - c, h - 1 - r
            else:
                rr, cc = h - 1 - r, w - 1 - c
            bad += rows[rr][cc] != rows[r][c]
    return bad


def test_near_misses_are_real_and_common(sweep):
    near_labels = comparable = 0
    tight = 0
    near_grids = 0
    for ex in sweep:
        meta = ex["metadata"]
        rows = ex["input"]
        area = meta["grid_h"] * meta["grid_w"]
        absent = [(label, _violating(rows, label)) for label in LABELS
                  if label not in meta["present_labels"]]
        counts = [v for _, v in absent if v is not None]
        assert all(v > 0 for v in counts), meta      # a "no" must be violated somewhere
        comparable += len(counts)
        near_labels += sum(1 for v in counts if v / area <= 0.08)
        tight += sum(1 for v in counts if v <= 6)
        if counts:
            assert meta["min_violating_cells"] == min(counts), meta
            assert meta["near_miss"] == (min(counts) / area <= 0.08), meta
            near_grids += meta["near_miss"]
        else:
            assert meta["min_violating_fraction"] == 1.0
    assert comparable > 800
    assert 0.25 <= tight / comparable <= 0.55, tight / comparable
    assert 0.50 <= near_labels / comparable <= 0.80, near_labels / comparable
    assert near_grids / len(sweep) >= 0.50, near_grids / len(sweep)


def test_a_spot_check_reader_fails(sweep):
    """Five sampled cells per label used to settle 97.5% of the grids."""
    rng = np.random.default_rng(7)
    exact = 0
    for ex in sweep:
        rows = ex["input"]
        h, w = len(rows), len(rows[0])
        guess = []
        for label in LABELS:
            if label in (LABELS[2], LABELS[3]) and h != w:
                guess.append("no")
                continue
            ok = True
            for _ in range(5):
                r, c = int(rng.integers(0, h)), int(rng.integers(0, w))
                if label == LABELS[0]:
                    rr, cc = h - 1 - r, c
                elif label == LABELS[1]:
                    rr, cc = r, w - 1 - c
                elif label == LABELS[2]:
                    rr, cc = c, r
                elif label == LABELS[3]:
                    rr, cc = w - 1 - c, h - 1 - r
                else:
                    rr, cc = h - 1 - r, w - 1 - c
                if rows[rr][cc] != rows[r][c]:
                    ok = False
                    break
            guess.append("yes" if ok else "no")
        reported = [line.split(": ")[1] for line in ex["output"].split("\n")]
        exact += guess == reported
    assert exact / len(sweep) <= 0.45, exact / len(sweep)


def test_the_class_is_not_readable_from_the_canvas(sweep):
    """Beyond the square the diagonal classes need, the shape says nothing."""
    by_square = {True: Counter(), False: Counter()}
    by_shape = {}
    for ex in sweep:
        meta = ex["metadata"]
        h, w = meta["grid_h"], meta["grid_w"]
        key = frozenset(meta["present_labels"])
        by_square[h == w][key] += 1
        bucket = (h == w, min(4, h // 8), min(4, w // 8))
        by_shape.setdefault(bucket, Counter())[key] += 1
    blind = sum(max(c.values()) for c in by_square.values()) / len(sweep)
    detailed = sum(max(c.values()) for c in by_shape.values()) / len(sweep)
    assert blind <= 0.30, blind
    assert detailed - blind <= 0.15, (detailed, blind)


# ---------------------------------------------------------------------------
# prompt policy
# ---------------------------------------------------------------------------

def test_prompt_policy(sweep):
    wordings = set()
    for ex in sweep[:400]:
        prompt = get_prompt("102", ex["metadata"])
        assert "Rows and columns are 0-indexed from the top-left corner." in prompt
        assert "A diagonal reflection is answered no whenever the grid is not square" in prompt
        assert "a single cell that differs makes the answer no" in prompt
        assert "Use this exact order: " + ", ".join(LABELS) in prompt
        assert "Return exactly one line for each label in that order" in prompt
        assert "label: yes or label: no" in prompt
        assert "Do not add extra lines or extra text." in prompt
        assert "zero-indexed" not in prompt.lower()
        assert "background" not in prompt.lower()
        assert "color" not in prompt.lower()
        wordings.add(prompt)
    assert len(wordings) == 1, len(wordings)


def test_no_color_is_named():
    import re
    for seed in range(40):
        ex = generate("102", seed)
        for palette in (list("ABCDEFGHIJ"), palette_for_example("102", seed)):
            prompt = get_prompt("102", ex["metadata"], color_palette=palette)
            assert not re.search(r"\bcolor [0-9]\b", prompt), prompt
            assert prompt == get_prompt("102", ex["metadata"]), "the prompt must not depend on the palette"


# ---------------------------------------------------------------------------
# the validator earns its place
# ---------------------------------------------------------------------------

def test_validator_raises_on_a_tampered_answer():
    ex = next(generate("102", seed) for seed in range(20)
              if generate("102", seed)["metadata"]["n_present"] > 0)

    bad = json.loads(json.dumps(ex))
    lines = bad["output"].split("\n")
    lines[0] = lines[0].replace(": yes", ": no") if lines[0].endswith("yes") \
        else lines[0].replace(": no", ": yes")
    bad["output"] = "\n".join(lines)
    with pytest.raises(ValueError):
        module.validate_example(bad)

    bad = json.loads(json.dumps(ex))
    bad["metadata"]["n_present"] += 1
    with pytest.raises(ValueError):
        module.validate_example(bad)

    bad = json.loads(json.dumps(ex))
    bad["metadata"]["min_violating_fraction"] = 0.5
    with pytest.raises(ValueError):
        module.validate_example(bad)

    bad = json.loads(json.dumps(ex))
    bad["input"][0][0] = (bad["input"][0][0] + 1) % 10
    with pytest.raises(ValueError):
        module.validate_example(bad)

    module.validate_example(json.loads(json.dumps(ex)))


def test_the_two_readings_of_every_predicate_agree(sweep):
    for ex in sweep[:200]:
        rows = ex["input"]
        for label in LABELS:
            assert module.present(rows, label) == module.literal_present(rows, label), label
    # a half turn without either diagonal, told apart by both readings
    rows = [[1, 2, 3], [4, 5, 4], [3, 2, 1]]
    for label, expected in ((LABELS[4], True), (LABELS[2], False), (LABELS[3], False),
                            (LABELS[0], False), (LABELS[1], False)):
        assert module.present(rows, label) is expected, label
        assert module.literal_present(rows, label) is expected, label


def test_a_reported_class_is_always_a_group(sweep):
    for ex in sweep:
        assert frozenset(ex["metadata"]["present_labels"]) in CLASSES, ex["metadata"]


def test_generation_is_cheap():
    start = time.perf_counter()
    for seed in range(200):
        generate("102", seed)
    per_seed = (time.perf_counter() - start) / 200
    assert per_seed < 0.09, per_seed


# ---------------------------------------------------------------------------
# the picture is budgeted before the class is allowed to matter
# ---------------------------------------------------------------------------

PLAN_DRAWS = 20000


@pytest.fixture(scope="module")
def plans():
    """Plans only: no drawing, so the sample is big enough to be quiet."""
    rng = np.random.default_rng(20260916)
    return [module._sample_plan(rng) for _ in range(PLAN_DRAWS)]


def test_the_budget_is_drawn_before_the_class(plans):
    """Occupancy, region size and palette come from the canvas, not the group."""
    occ, pal, reg = {}, {}, {}
    for plan in plans:
        occ.setdefault(plan["answer"], []).append(plan["occupancy_target"])
        pal.setdefault(plan["answer"], []).append(plan["palette_target"])
        reg.setdefault(plan["answer"], []).append(plan["region_cells"])
    assert len(occ) == 9, sorted(occ)
    # measured over 20,000 plans: 0.018, 0.23, 0.66 -- what is left is the
    # canvas, whose area distribution differs between the classes that need a
    # square one and the classes that draw their two sides independently
    for name, table, slack in (("occupancy", occ, 0.03), ("palette", pal, 0.40),
                               ("region", reg, 1.00)):
        means = {k: sum(v) / len(v) for k, v in table.items()}
        spread = max(means.values()) - min(means.values())
        assert spread <= slack, (name, sorted(means.items(), key=lambda kv: kv[1]))


def test_the_canvas_and_the_class_survive_every_retry():
    """A rejected drawing is redrawn on the SAME canvas, so nothing it dislikes
    can bend the side marginals or the class balance."""
    for seed in range(300):
        meta = generate("102", seed)["metadata"]
        plan = module._sample_plan(np.random.default_rng(seed))
        assert (plan["h"], plan["w"], plan["answer"]) == \
            (meta["grid_h"], meta["grid_w"], meta["answer_class"]), seed


def test_the_side_marginals_are_flat(plans):
    """Both sides are drawn flat over 5..30 for the classes that are free to."""
    hist = {"h": Counter(), "w": Counter()}
    for plan in plans:
        if plan["h"] == plan["w"]:
            continue
        hist["h"][plan["h"]] += 1
        hist["w"][plan["w"]] += 1
    for side, counts in hist.items():
        values = [counts[s] for s in range(5, 31)]
        mean = sum(values) / len(values)
        # measured 0.92-1.08 on both sides over 20,000 plans
        assert min(values) / mean >= 0.80, (side, values)
        assert max(values) / mean <= 1.20, (side, values)


def test_the_sweep_keeps_the_side_marginals_flat(sweep):
    for i, side in ((0, "h"), (1, "w")):
        counts = Counter()
        for ex in sweep:
            meta = ex["metadata"]
            if meta["grid_h"] != meta["grid_w"]:
                counts[(meta["grid_h"], meta["grid_w"])[i]] += 1
        values = [counts[s] for s in range(5, 31)]
        mean = sum(values) / len(values)
        chi = sum((v - mean) ** 2 / mean for v in values)
        assert chi <= 75, (side, chi, values)     # 25 df; measured 12-51


# ---------------------------------------------------------------------------
# the answer class is not written in the ink
# ---------------------------------------------------------------------------

FEATURES = ["h", "w", "square", "area", "n_colors", "modal_share",
            "n_components", "largest_component", "border_share", "aspect"]


def _features(rows):
    """The cheap things a reader can measure without doing the task."""
    grid = np.asarray(rows)
    h, w = grid.shape
    area = h * w
    values, counts = np.unique(grid, return_counts=True)
    modal = int(values[int(np.argmax(counts))])
    seen = np.zeros((h, w), dtype=bool)
    pieces = 0
    largest = 0
    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] == modal:
                continue
            color = grid[r, c]
            stack = [(r, c)]
            seen[r, c] = True
            size = 0
            while stack:
                rr, cc = stack.pop()
                size += 1
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] \
                            and grid[nr, nc] == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            pieces += 1
            largest = max(largest, size)
    ring = np.concatenate([grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]])
    return [h, w, float(h == w), area, len(values), float(counts.max()) / area,
            pieces, largest / area, float((ring == modal).mean()),
            max(h, w) / min(h, w)]


def _best_split(X, y, n_classes):
    best = None
    n = len(y)
    for j in range(X.shape[1]):
        column = X[:, j]
        for t in np.unique(np.quantile(column, np.linspace(0.05, 0.95, 19))):
            left = column <= t
            n_left = int(left.sum())
            if n_left == 0 or n_left == n:
                continue
            gl = 1.0 - ((np.bincount(y[left], minlength=n_classes) / n_left) ** 2).sum()
            gr = 1.0 - ((np.bincount(y[~left], minlength=n_classes)
                         / (n - n_left)) ** 2).sum()
            score = (n_left * gl + (n - n_left) * gr) / n
            if best is None or score < best[0] - 1e-12:
                best = (score, j, float(t))
    return best


def _cart_accuracy(X, y, depth=4, n_classes=9):
    """A depth-4 classification tree, fitted and scored on the same rows."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    guess = np.empty(len(y), dtype=int)

    def grow(idx, left_depth):
        here = y[idx]
        major = int(np.bincount(here, minlength=n_classes).argmax())
        split = None
        if left_depth and len(idx) >= 16 and len(set(here.tolist())) > 1:
            split = _best_split(X[idx], here, n_classes)
        if split is None:
            guess[idx] = major
            return
        _, j, t = split
        left = X[idx][:, j] <= t
        if left.all() or (~left).all():
            guess[idx] = major
            return
        grow(idx[left], left_depth - 1)
        grow(idx[~left], left_depth - 1)

    grow(np.arange(len(y)), depth)
    return float((guess == y).mean())


@pytest.fixture(scope="module")
def cheap_table(sweep):
    names = sorted({frozenset(ex["metadata"]["present_labels"]) for ex in sweep},
                   key=lambda s: sorted(s))
    index = {name: i for i, name in enumerate(names)}
    X = np.array([_features(ex["input"]) for ex in sweep], dtype=float)
    y = np.array([index[frozenset(ex["metadata"]["present_labels"])] for ex in sweep])
    return X, y


def test_the_ink_and_the_palette_do_not_report_the_class(sweep):
    """The old generator folded its bodies into a denser, poorer picture: mean
    occupancy 0.365 on the all-yes class against 0.287 on the all-no one, and
    4.14 colours against 5.36.  Measured over 10,000 seeds the spreads are now
    0.028 and 0.21; the bounds below carry the noise of a 1400-seed sweep."""
    occ, col = {}, {}
    for ex in sweep:
        meta = ex["metadata"]
        occ.setdefault(meta["answer_class"], []).append(meta["occupancy"])
        col.setdefault(meta["answer_class"], []).append(meta["n_colors"])
    assert len(occ) == 9, sorted(occ)
    mo = {k: sum(v) / len(v) for k, v in occ.items()}
    mc = {k: sum(v) / len(v) for k, v in col.items()}
    assert max(mo.values()) - min(mo.values()) <= 0.06, sorted(mo.items())
    assert max(mc.values()) - min(mc.values()) <= 0.80, sorted(mc.items())
    # and the two halves the square rule cuts the task into, where the sample is
    # big enough for the systematic part to show
    square = [k for k in mo if k in ("main_diagonal", "anti_diagonal", "diagonals", "full")]
    free = [k for k in mo if k not in square]
    gap_o = abs(sum(mo[k] for k in square) / 4 - sum(mo[k] for k in free) / 5)
    gap_c = abs(sum(mc[k] for k in square) / 4 - sum(mc[k] for k in free) / 5)
    assert gap_o <= 0.04, (gap_o, sorted(mo.items()))
    assert gap_c <= 0.50, (gap_c, sorted(mc.items()))


def test_a_cheap_classifier_learns_nothing_the_canvas_did_not_say(cheap_table):
    """A depth-4 tree on ten cheap input features used to score 25.9% exact
    against a 12.2% constant.  What is left is the canvas itself: the four
    classes with a diagonal need a square one.  The control keeps that channel
    and destroys every other by reshuffling the class inside each h x w cell,
    and the real tree may not beat it."""
    X, y = cheap_table
    rng = np.random.default_rng(11)
    shuffled = y.copy()
    cells = {}
    for i, row in enumerate(X):
        cells.setdefault((row[0], row[1]), []).append(i)
    for idx in cells.values():
        block = shuffled[idx]
        shuffled[idx] = block[rng.permutation(len(block))]

    scene = [i for i, f in enumerate(FEATURES)
             if f not in ("square", "aspect", "h", "w", "area")]
    real_all = _cart_accuracy(X, y)
    control_all = _cart_accuracy(X, shuffled)
    real_scene = _cart_accuracy(X[:, scene], y)
    control_scene = _cart_accuracy(X[:, scene], shuffled)
    # over 3000 seeds: 0.284 against 0.276 for the whole table and 0.167
    # against 0.172 for the scene alone; the slack is the noise of fitting a
    # 16-leaf tree to a 1400-row sweep
    assert real_all <= control_all + 0.04, (real_all, control_all)
    assert real_scene <= control_scene + 0.06, (real_scene, control_scene)
    # the canvas channel itself is the square rule and nothing more
    shape = [i for i, f in enumerate(FEATURES) if f in ("h", "w", "square", "area", "aspect")]
    assert real_all <= _cart_accuracy(X[:, shape], y) + 0.04, real_all


def test_a_drawn_scene_has_no_fragments(sweep):
    """Only the near miss leaves a mark of a cell or two; a scene that was drawn
    straight is bodies and bands all the way down."""
    marked = near = 0
    for ex in sweep:
        meta = ex["metadata"]
        sizes = module.colour_regions(ex["input"], meta["body_colors"])
        assert sizes and sum(sizes) == meta["body_cells"], meta
        if meta["generation_mode"] in ("folded", "free"):
            assert min(sizes) >= module.REGION_MIN, (meta["generation_mode"], sizes)
        else:
            near += 1
            marked += min(sizes) < module.REGION_MIN
    assert near > 300
    assert marked / near <= 0.75, marked / near
