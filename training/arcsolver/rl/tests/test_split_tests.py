"""--train-split-tests / --eval-tests: config.split_test_view gives a multi-test task's rollout group ONE test (seeded,
recorded); train.py splits the eval into one prompt per test input and merges per task exactly as score_task averages
outputs (merge_split_scores); make_eval writes joint / split / both with the right files, metrics and COMPLETE marker."""
import gzip
import json
import os
import types

import pytest

from arcsolver.prompt import arc_render as ar
from arcsolver.rl import config as RT
from arcsolver.rl import train as TD

G1, G2, G3 = [[1, 2], [3, 4]], [[5, 5, 5]], [[7], [8]]
TASK2 = {"train": [{"input": [[0]], "output": [[1]]}], "test": [{"input": [[0, 0]], "output": G1},
                                                                {"input": [[0]], "output": G2}]}
TASK3 = {"train": [{"input": [[0]], "output": [[1]]}],
         "test": [{"input": [[0]], "output": G1}, {"input": [[0]], "output": G2}, {"input": [[0]], "output": G3}]}
TASK1 = {"train": [{"input": [[0]], "output": [[1]]}], "test": [{"input": [[0]], "output": G3}]}


def ans(*grids):
    body = "\n\n".join("\n".join("".join(map(str, row)) for row in g) for g in grids)
    return f"reasoning{ar.THINK_END}\n\n{body}"


def test_split_test_view_one_seeded_test():
    on = types.SimpleNamespace(train_split_tests=True, seed=3)
    off = types.SimpleNamespace(train_split_tests=False, seed=3)
    assert RT.split_test_view(off, TASK2, 5, 1) == (TASK2, None)
    assert RT.split_test_view(on, TASK1, 5, 1) == (TASK1, None)          # single-test tasks unchanged
    sub, j = RT.split_test_view(on, TASK3, 5, 1)
    assert len(sub["test"]) == 1 and sub["test"][0] is TASK3["test"][j] and sub["_test_index"] == j
    assert sub["_n_test"] == 3 and sub["train"] == TASK3["train"] and len(TASK3["test"]) == 3   # original untouched
    assert RT.split_test_view(on, TASK3, 5, 1)[1] == j                    # deterministic per (seed, step, group)
    seen = {RT.split_test_view(on, TASK3, s, g)[1] for s in range(1, 30) for g in range(8)}
    assert seen == {0, 1, 2}


def test_augment_group_renders_the_single_test_view(monkeypatch):
    monkeypatch.setattr(RT.oc, "student_messages", lambda task, arm: [len(task["test"])])
    monkeypatch.setattr(RT.oc, "render_ids", lambda tok, msgs: [100 + msgs[0]])
    it = types.SimpleNamespace(student_ids=[999])
    base = dict(train_split_tests=True, seed=3, arm="x", aug_original_prob=0.0, aug_colors="nonzero",
                aug_identity_prob=0.0, aug_max_tries=4, max_prompt_tokens=10 ** 6)
    a = types.SimpleNamespace(augment=False, **base)
    task, ids, aug, _ = RT.augment_group(a, TASK2, it, None, 7, 2)
    assert len(task["test"]) == 1 and ids == [101] and "_test_index" in task
    t1, ids1, _, _ = RT.augment_group(a, TASK1, it, None, 7, 2)
    assert t1 is TASK1 and ids1 == [999]                                  # single test: the item's usual prompt
    b = types.SimpleNamespace(augment=True, **base)
    task, ids, aug, _ = RT.augment_group(b, TASK2, it, None, 7, 2)
    assert len(task["test"]) == 1 and ids == [101] and task["_test_index"] in (0, 1)   # kept through Aug.task


def test_merge_split_scores_equals_joint_scoring():
    # joint: 4 samples answering both tests; split: the same per-test answers as separate prompts
    joint = [ans(G1, G2), ans(G1, [[0]]), ans([[9]], G2), ans([[9]], [[9]])]
    s0 = ar.score_task({**TASK2, "test": [TASK2["test"][0]]}, [ans(G1), ans(G1), ans([[9]]), ans([[9]])], ks=(1, 2, 4))
    s1 = ar.score_task({**TASK2, "test": [TASK2["test"][1]]}, [ans(G2), ans([[0]]), ans(G2), ans([[9]])], ks=(1, 2, 4))
    m = TD.merge_split_scores([s0, s1])
    j = ar.score_task(TASK2, joint, ks=(1, 2, 4))
    for k in ("pass@1", "pass@2", "pass@4", "vote@2"):
        assert m[k] == pytest.approx(j[k])
    assert [o["correct"] for o in m["per_output"]] == [o["correct"] for o in j["per_output"]] == [2, 2]
    assert m["n_samples"] == 8 and m["n_test"] == 2 and m["closed"] == 8


def test_split_eval_prompts(monkeypatch):
    items = [types.SimpleNamespace(path="a", student_ids=[1]), types.SimpleNamespace(path="b", student_ids=[2])]
    tasks = {"a": TASK1, "b": TASK3}.get
    monkeypatch.setattr(TD.oc, "student_messages", lambda task, arm: [task["test"][0]["output"]])
    monkeypatch.setattr(TD.oc, "render_ids", lambda tok, m: [str(m)])
    args = types.SimpleNamespace(arm="x")
    subs = TD.split_eval_prompts(args, None, tasks, items)
    assert [(ii, j, nt) for ii, j, nt, _, _ in subs] == [(0, 0, 1), (1, 0, 3), (1, 1, 3), (1, 2, 3)]
    assert subs[0][4] == [1] and [s[3]["test"][0]["output"] for s in subs[1:]] == [G1, G2, G3]


class FakeSampler:
    """Answers every prompt with the expected grids of the task the prompt was built from (prompt ids carry it)."""
    def __init__(self, lookup):
        self.lookup, self.calls = lookup, []

    def call(self, what, prompts, params, max_tokens, seeds):
        self.calls.append(len(prompts))
        outs = []
        for p in prompts:
            grids = self.lookup[tuple(p)]
            outs.append([{"text": ans(*grids), "token_ids": [0] * 5, "finish": "stop"}] * params["n"])
        return {"outputs": outs}


def test_make_eval_modes(tmp_path, monkeypatch):
    monkeypatch.setattr(TD.oc, "student_messages", lambda task, arm: [json.dumps(task["test"])])
    monkeypatch.setattr(TD.oc, "render_ids", lambda tok, msgs: [msgs[0]])
    lookup = {(json.dumps(TASK2["test"][j:j + 1]),): [TASK2["test"][j]["output"]] for j in range(2)}
    lookup[("joint2",)] = [G1, G2]
    lookup[("joint1",)] = [G3]
    items = [types.SimpleNamespace(task_id="t2", source="arc_eval", seen=False, path="t2", student_ids=["joint2"]),
             types.SimpleNamespace(task_id="t1", source="arc_eval", seen=False, path="t1", student_ids=["joint1"])]
    tasks = {"t2": TASK2, "t1": TASK1}.get
    os.makedirs(tmp_path / "evals")
    for mode, steps, expect in (("joint", "", [""]), ("split", "", [".split"]), ("both", "", ["", ".split"]),
                                ("split", "40", [".split"])):
        out = tmp_path / mode / steps
        os.makedirs(out / "evals")
        args = types.SimpleNamespace(out=str(out), eval_n=2, eval_temperature=1.0, eval_top_p=1.0, eval_top_k=0,
                                     eval_max_tokens=100, sampler_max_model_len=1000, seed=1, eval_budget_force=0,
                                     arm="x", eval_tests=mode, eval_joint_steps=steps)
        emitted, logs = [], []
        s = FakeSampler(lookup)
        tok = types.SimpleNamespace(all_special_ids=[], added_tokens_decoder={})      # texts come from the sampler
        run = TD.make_eval(args, s, tok, tasks, items, lambda m, st: emitted.append(m), logs.append)
        run(40)
        files = sorted(os.listdir(out / "evals"))
        want = sorted([f"step_000040{x}.{e}" for x in (expect if steps != "40" else ["", ".split"])
                       for e in ("json", "jsonl.gz")] + ["step_000040.COMPLETE"])
        assert files == want, (mode, files)
        m = emitted[0]
        if "" in expect or steps == "40":
            assert m["eval/arc_eval/pass@1"] == 1.0
        if ".split" in expect:
            assert m["eval_split/arc_eval/pass@1"] == 1.0 and m["eval_split/arc_eval/n_tasks"] == 2
            rows = [json.loads(x) for x in gzip.open(out / "evals" / "step_000040.split.jsonl.gz", "rt")]
            assert [(r["task_id"], r["test_index"], r["n_test"]) for r in rows] == [("t2", 0, 2), ("t2", 1, 2), ("t1", 0, 1)]
            assert any(line.startswith("eval-split step 40") for line in logs)
