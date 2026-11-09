"""Every encoding round-trips exactly and rejects malformed text."""
import numpy as np
import pytest

from pv2 import encodings as enc
from registry import generate

NAMES = sorted(enc.ENCODINGS)
SHAPES = [(1, 1), (1, 30), (30, 1), (30, 30), (5, 7), (13, 2)]


def _random_grids(n=60, seed=0):
    rng = np.random.default_rng(seed)
    for h, w in SHAPES:
        yield rng.integers(0, 10, size=(h, w)).tolist()
    for _ in range(n):
        h, w = rng.integers(1, 31, size=2)
        k = int(rng.integers(1, 11))
        colours = rng.choice(10, size=k, replace=False)
        yield rng.choice(colours, size=(h, w)).tolist()


@pytest.mark.parametrize("name", NAMES)
def test_round_trip_random(name):
    for g in _random_grids():
        text = enc.render(g, name)
        assert enc.parse(text, name) == g


@pytest.mark.parametrize("name", NAMES)
def test_round_trip_generator_grids(name):
    for tid in ("4", "25", "77", "86", "95"):
        for seed in range(5):
            g = generate(tid, seed)["input"]
            assert enc.parse(enc.render(g, name), name) == g


@pytest.mark.parametrize("name", NAMES)
def test_malformed_text_raises(name):
    g = [[1, 2, 3], [4, 5, 6]]
    text = enc.render(g, name)
    broken = [text + "\n7", text.replace("5", "X", 1), "", text.replace("\n", "\n\n", 1)]
    for bad in broken:
        if bad == text:
            continue
        with pytest.raises((ValueError, TypeError)):
            enc.parse(bad, name)


def test_sparse_is_offered_only_for_sparse_grids():
    dense = [[1, 2], [3, 4]]
    sparse = [[0] * 10 for _ in range(10)]
    sparse[3][4] = 7
    assert not enc.suits(dense, "sparse") and enc.suits(sparse, "sparse")
    assert all(enc.suits(dense, n) for n in enc.ENCODINGS if n != "sparse")


def test_invalid_grids_are_refused():
    for bad in ([], [[]], [[1, 2], [3]], [[10]], [[-1]], [[True]]):
        with pytest.raises(ValueError):
            enc.render(bad, "compact")


def test_flags_are_truthful():
    g = [[0, 0, 0], [0, 7, 0]]
    for name, e in enc.ENCODINGS.items():
        text = e.render(g)
        if not e.reveals_coords:
            assert "row 1" not in text and "r1" not in text and "(1,1)" not in text, name
        if not e.reveals_dims:
            assert "rows x" not in text, name
    assert "every cell is 0" in enc.render(g, "sparse")      # the default fill is printed
    assert enc.ENCODINGS["sparse"].reveals_background
    assert enc.colour_word(3, "names") == "green" and enc.colour_word(3, "compact") == "color 3"


def test_stage2_encoding_matches_the_stage2_renderer_byte_for_byte():
    import importlib.util
    import os
    from pv2.paths import SOLVER_PROMPT_DIR
    path = os.path.join(SOLVER_PROMPT_DIR, "arc_render.py")
    if not os.path.exists(path):
        pytest.skip("the solver renderer (../training/arcsolver/prompt/arc_render.py) is not present")
    spec = importlib.util.spec_from_file_location("arc_render_ref", path)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    for g in _random_grids(40, seed=3):
        block = ref._indexed_block("X", g)
        body = block.split("\n", 1)[1]            # drop the "X (H rows x W cols):" header line
        assert enc.render(g, "stage2") == body


def test_choose_respects_restrictions_and_weights():
    import numpy as np
    rng = np.random.default_rng(0)
    g = [[0] * 12 for _ in range(12)]
    g[2][3] = 5
    picks = [enc.choose(rng, g, coords_ok=False, dims_ok=False) for _ in range(2000)]
    for name in set(picks):
        e = enc.ENCODINGS[name]
        assert not (e.reveals_coords or e.reveals_dims or e.uses_names or e.reveals_background)
    assert picks.count("compact") / len(picks) > .45
    picks = [enc.choose(rng, g, names_ok=True, background_ok=True) for _ in range(3000)]
    assert {"names", "stage2", "sparse"} <= set(picks)


def test_runs_encoding_uses_the_stage2_notation():
    g = [[0, 0, 5, 5, 5, 7]]
    assert enc.render(g, "runs") == "r0: 0@c0-c1 5@c2-c4 7@c5"
    assert "x" not in enc.render([[3, 3, 3]], "runs")


def test_labelled_encodings_reveal_the_size():
    for name in ("stage2", "runs", "row_labelled", "indexed_table", "sparse"):
        assert enc.ENCODINGS[name].reveals_dims, name
    for name in ("compact", "spaced", "comma", "json_flat", "json_rows", "brackets", "pipes", "names"):
        assert not enc.ENCODINGS[name].reveals_dims, name


def test_sparse_states_its_index_base_and_descriptions_pin_the_shape():
    g = [[0] * 10 for _ in range(10)]
    g[0][3] = 2
    text = enc.render(g, "sparse")
    assert "0-indexed" in text and "(0,3)=2" in text
    assert enc.parse(text, "sparse") == g
    for name in ("pipes", "brackets"):
        for d in enc.ENCODINGS[name].describe:
            assert ("both ends" in d or "between vertical bars" in d) if name == "pipes" else ("space" in d)
