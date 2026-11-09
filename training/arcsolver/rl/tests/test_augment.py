"""Task augmentation (augment.py): the 8 dihedral transforms x colour permutations, applied coherently to every grid
of a task (ground truth and alt_outputs included), their inverses, rendering and scoring of augmented tasks."""
import json
import random

from arcsolver.prompt import arc_render as ar
from arcsolver.rl import augment as ra
from arcsolver.rl import rewards as rr

FORMULAS = {   # new[i][j] = g[src(i, j)] for an H x W grid; the second entry is the output shape
    "id": (lambda g, i, j, H, W: g[i][j], lambda H, W: (H, W)),
    "rot90": (lambda g, i, j, H, W: g[H - 1 - j][i], lambda H, W: (W, H)),
    "rot180": (lambda g, i, j, H, W: g[H - 1 - i][W - 1 - j], lambda H, W: (H, W)),
    "rot270": (lambda g, i, j, H, W: g[j][W - 1 - i], lambda H, W: (W, H)),
    "flip_h": (lambda g, i, j, H, W: g[i][W - 1 - j], lambda H, W: (H, W)),
    "flip_v": (lambda g, i, j, H, W: g[H - 1 - i][j], lambda H, W: (H, W)),
    "transpose": (lambda g, i, j, H, W: g[j][i], lambda H, W: (W, H)),
    "anti_transpose": (lambda g, i, j, H, W: g[H - 1 - j][W - 1 - i], lambda H, W: (W, H)),
}


def _rect(H=3, W=5):
    return [[r * W + c for c in range(W)] for r in range(H)]          # every cell distinct


def test_dihedral_transforms_match_their_index_formulas():
    g = _rect()
    H, W = 3, 5
    outs = {}
    for name in ra.D8:
        src, shape = FORMULAS[name]
        t = ra.dihedral(g, name)
        assert (len(t), len(t[0])) == shape(H, W), name
        assert all(t[i][j] == src(g, i, j, H, W) for i in range(len(t)) for j in range(len(t[0]))), name
        outs[name] = str(t)
    assert len(set(outs.values())) == 8                                 # the 8 symmetries are distinct
    assert g == _rect()                                                # input untouched


def test_each_transform_and_its_inverse_compose_to_the_identity():
    rng = random.Random(0)
    g = [[rng.randrange(10) for _ in range(4)] for _ in range(7)]
    for name in ra.D8:
        assert ra.dihedral(ra.dihedral(g, name), ra.INVERSE[name]) == g
        assert ra.dihedral(ra.dihedral(g, ra.INVERSE[name]), name) == g
    for k in range(50):
        aug = ra.sample_aug(random.Random(k), colors=("nonzero", "all", "none")[k % 3])
        inv = aug.inverse()
        assert inv.grid(aug.grid(g)) == g and aug.grid(inv.grid(g)) == g
        assert inv.inverse() == aug


def _arc_like_task(seed=3, n_train=3, H=4, W=6):
    """A task with a known rule: output = input mirrored left-right with colour 1 -> 2 (non-square grids)."""
    rng = random.Random(seed)
    rule = lambda g: [[2 if v == 1 else v for v in row[::-1]] for row in g]  # noqa: E731
    mk = lambda: [[rng.choice([0, 0, 1, 3, 5]) for _ in range(W)] for _ in range(H)]  # noqa: E731
    pairs = [mk() for _ in range(n_train + 2)]
    return ({"train": [{"input": x, "output": rule(x)} for x in pairs[:n_train]],
             "test": [{"input": x, "output": rule(x)} for x in pairs[n_train:]], "name": "keep-me"}, rule)


def test_augmented_task_is_consistent_across_all_grids_and_the_ground_truth():
    task, _ = _arc_like_task()
    orig = json.loads(json.dumps(task))
    for k in range(40):
        aug = ra.sample_aug(random.Random(k), colors="nonzero" if k % 2 else "all")
        at = aug.task(task)
        assert task == orig and at["name"] == "keep-me"                     # deep copy, other keys kept
        for split in ("train", "test"):
            for p0, p1 in zip(task[split], at[split]):
                assert p1["input"] == aug.grid(p0["input"])
                assert p1["output"] == aug.grid(p0["output"])             # the ground truth is transformed too
        assert aug.inverse().task(at) == task
    nz = ra.sample_aug(random.Random(1), colors="nonzero")
    assert nz.perm[0] == 0 and sorted(nz.perm) == list(range(10))
    assert ra.sample_aug(random.Random(1), colors="none").perm == tuple(range(10))
    assert any(ra.sample_aug(random.Random(k), colors="all").perm[0] != 0 for k in range(20))


def test_the_rule_carries_over_to_the_augmented_task():
    """The augmented task is solvable by the conjugated rule d . perm . rule . perm^-1 . d^-1 on every pair."""
    task, rule = _arc_like_task(seed=5)
    for k in range(30):
        aug = ra.sample_aug(random.Random(100 + k))
        inv = aug.inverse()
        at = aug.task(task)
        conj = lambda x: aug.grid(rule(inv.grid(x)))  # noqa: E731
        for pair in at["train"] + at["test"]:
            assert conj(pair["input"]) == pair["output"]


def test_a_known_solution_transformed_is_exactly_correct_for_the_augmented_task():
    task, _ = _arc_like_task(seed=9)
    rows = lambda g: "\n".join("".join(map(str, r)) for r in g)  # noqa: E731
    solution = "\n\n".join(rows(t["output"]) for t in task["test"])
    assert rr.answer_eval(task, "rule found</think>\n\n" + solution, "stop")["em_all"]
    n_diff = 0
    for k in range(30):
        aug = ra.sample_aug(random.Random(200 + k))
        at = aug.task(task)
        moved = "\n\n".join(rows(aug.grid(t["output"])) for t in task["test"])
        a = rr.answer_eval(at, "rule found</think>\n\n" + moved, "stop")
        assert a["em_all"] and a["shape_ok"]
        if not aug.is_identity():
            b = rr.answer_eval(at, "rule found</think>\n\n" + solution, "stop")
            n_diff += not b["em_all"]                                       # the untransformed answer no longer fits
    assert n_diff >= 25


def test_augmented_synthetic_tasks_render_and_score():
    rows = lambda g: "\n".join("".join(map(str, r)) for r in g)  # noqa: E731
    for k in range(16):
        task, _ = _arc_like_task(seed=40 + k, H=3 + k % 3, W=4 + k % 4)
        aug = ra.Aug(ra.D8[k % 8], tuple([0] + random.Random(k).sample(range(1, 10), 9)))
        at = aug.task(task)
        msgs = ar.build_messages(at)
        block = msgs[1]["content"]
        for i, t in enumerate(at["test"], 1):
            g = t["input"]
            assert f"Test input {i} ({len(g)} rows x {len(g[0])} cols):" in block
        ans = "\n\n".join(rows(t["output"]) for t in at["test"])
        assert rr.answer_eval(at, "x</think>\n\n" + ans, "stop")["em_all"]
        assert msgs[0] == ar.build_messages(task)[0]                     # same system prompt (arm)


def test_non_square_rendering_swaps_dimensions():
    task = {"train": [{"input": [[1, 2, 3], [4, 5, 6]], "output": [[1, 2, 3], [4, 5, 6]]}],
            "test": [{"input": [[7, 8, 9], [0, 1, 2]], "output": [[7, 8, 9], [0, 1, 2]]}]}
    for name in ("rot90", "rot270", "transpose", "anti_transpose"):
        at = ra.Aug(name).task(task)
        block = ar.task_block(at)
        assert "Training pair 1 input (3 rows x 2 cols):" in block and "Test input 1 (3 rows x 2 cols):" in block
        assert "r2 |" in block and "r3 |" not in block.split("Runs of the same grid")[0]
    t = ar.task_block(ra.Aug("transpose").task(task))
    assert "r0 | 14\nr1 | 25\nr2 | 36" in t                              # transpose of [[1,2,3],[4,5,6]]
    for name in ("id", "rot180", "flip_h", "flip_v"):
        assert "(2 rows x 3 cols)" in ar.task_block(ra.Aug(name).task(task))


def test_augmentation_sampling_identity_probability_and_determinism():
    n = 8000
    d = [ra.sample_aug(random.Random(k)).d8 for k in range(n)]
    assert abs(d.count("id") / n - 1 / 8) < 0.015 and set(d) == set(ra.D8)
    d0 = [ra.sample_aug(random.Random(k), identity_prob=0.0).d8 for k in range(2000)]
    assert "id" not in d0
    d5 = [ra.sample_aug(random.Random(k), identity_prob=0.5).d8 for k in range(4000)]
    assert abs(d5.count("id") / 4000 - 0.5) < 0.04
    a = ra.sample_aug(ra.group_rng(3, 17, 2))
    assert a == ra.sample_aug(ra.group_rng(3, 17, 2))                    # resumable: same draw for the same step/group
    assert len({ra.sample_aug(ra.group_rng(3, 17, g)) for g in range(16)}) == 16
    assert ra.Aug().is_identity() and ra.Aug("id", (0, 2, 1, 3, 4, 5, 6, 7, 8, 9)).meta() == {"d8": "id", "perm": "0213456789"}
