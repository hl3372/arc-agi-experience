"""CPU tests of the P4 eval scoring path (no GPU, no vLLM / torch / transformers).

    python3 tests/test_eval_scoring.py        (or: python3 -m pytest tests/test_eval_scoring.py)

A fake task and hand-written completions cover the cases that decide a score: correct, wrong, unclosed reasoning
(natural wrong), unclosed + forced answer (right and wrong), extra blocks after the grid, no-think replies, and the
task-level averaging of pass@k over split test inputs; then the re-scoring CLI end to end on temporary files.
"""
import json
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arcsolver.prompt import arc_render as ar  # noqa: E402
from arcsolver.eval import scoring as S  # noqa: E402
from arcsolver.eval import score as score_cli  # noqa: E402

SUF = S.DEFAULT_FORCE_SUFFIX
TASK = {
    "train": [{"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]},
              {"input": [[2]], "output": [[3]]}],
    "test": [{"input": [[7, 7], [7, 7]], "output": [[1, 2], [3, 4]]},
             {"input": [[9]], "output": [[5]]}],
}
TRUTH0 = ar.grid_key(TASK["test"][0]["output"])        # "12\n34"
TRUTH1 = ar.grid_key(TASK["test"][1]["output"])        # "5"

RIGHT = "Rows map to 12 / 34.\n</think>\n\n12\n34"
WRONG = "Some reasoning.\n</think>\n\n12\n33"
OPEN = "Still thinking about 12 and 34 when the cap hits"
FORCED_RIGHT = SUF + "12\n34"
FORCED_WRONG = SUF + "The answer is 12"


def close(a, b):
    return math.isclose(a, b, rel_tol=0, abs_tol=1e-12)


def test_prompt_renders_one_test_input():
    msgs = ar.build_messages(S.single_test_task(TASK, 1), S.ARM)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"].startswith("Solve this ARC grid-transformation task.")
    assert "Test input 1 (1 rows x 1 cols):" in msgs[1]["content"]
    assert "Test input 2" not in msgs[1]["content"] and "77" not in msgs[1]["content"]
    assert "Training pair 2 output" in msgs[1]["content"]


def test_is_correct_cases():
    assert S.is_correct(RIGHT, TRUTH0) == (True, True)
    assert S.is_correct(WRONG, TRUTH0) == (True, False)
    assert S.is_correct(OPEN, TRUTH0) == (False, False)                       # never closed: nothing parsed
    assert S.is_correct("x</think>\n\n  12\n34  \n", TRUTH0) == (True, True)  # surrounding whitespace is fine
    assert S.is_correct("x</think>\n\n12\n34\n\n12\n34", TRUTH0) == (True, False)   # two grids: not one answer
    assert S.is_correct("x</think>\n\n12\n34\n\nDone.", TRUTH0) == (True, False)    # prose block after the grid
    assert S.is_correct("x</think>\n\n```\n12\n34\n```", TRUTH0) == (True, False)   # code fence
    assert S.is_correct("a</think>b</think>\n\n12\n34", TRUTH0) == (True, True)     # the LAST </think> closes
    assert S.is_correct("", TRUTH0) == (False, False)
    # --no-think: the template closed the reasoning in the prompt; the reply is the answer
    assert S.is_correct("12\n34", TRUTH0, think=False) == (True, True)
    assert S.is_correct("12\n34\n\n56", TRUTH0, think=False) == (True, False)
    assert S.is_correct(RIGHT, TRUTH0, think=False) == (True, False)


def test_needs_forced_answer():
    assert S.needs_forced_answer("length", OPEN)
    assert not S.needs_forced_answer("length", RIGHT)          # capped inside its answer: keeps its natural text
    assert not S.needs_forced_answer("stop", OPEN)             # ended by itself: not capped
    assert not S.needs_forced_answer("length", OPEN, think=False)


def test_score_samples_natural_and_forced():
    texts = [RIGHT, WRONG, OPEN, OPEN]
    forced = [None, None, FORCED_RIGHT, FORCED_WRONG]
    sc = S.score_samples(texts, forced, TRUTH0)
    assert sc["closed"] == [True, True, False, False]
    assert sc["correct"] == [True, False, False, False]
    assert sc["correct_forced"] == [True, False, True, False]
    assert S.score_samples(texts, None, TRUTH0)["correct_forced"] == sc["correct"]


def test_pass_metrics():
    m = S.pass_metrics([True, False, False, False])
    assert close(m["pass@1"], 0.25) and close(m["pass@2"], 0.5) and close(m["pass@4"], 1.0) and "pass@8" not in m
    m = S.pass_metrics([True, False, True, False])
    assert close(m["pass@2"], 1 - 1 / 6)
    assert S.pass_metrics([False] * 4) == {"pass@1": 0.0, "pass@2": 0.0, "pass@4": 0.0}


def scored_row(tid, j, texts, forced=None, truth=TRUTH0, finish=None, tokens=None):
    sc = S.score_samples(texts, forced, truth)
    return {"task_id": tid, "test_index": j, **sc, "forced": forced,
            "finish": finish or ["stop"] * len(texts), "gen_tokens": tokens or [100] * len(texts)}


def test_summary_averages_within_then_across_tasks():
    rows = [
        # task A, test 0: natural 1/4 right, forced 2/4 (one capped sample rescued, one not)
        scored_row("A", 0, [RIGHT, WRONG, OPEN, OPEN], [None, None, FORCED_RIGHT, FORCED_WRONG],
                   finish=["stop", "stop", "length", "length"], tokens=[10, 20, 30, 40]),
        # task A, test 1: nothing right
        scored_row("A", 1, ["x</think>\n\n4"] * 4, truth=TRUTH1),
        # task B (one test input): every sample right
        scored_row("B", 0, [RIGHT] * 4),
        # task C: only one of its two test inputs has a row -> left out, reported incomplete
        scored_row("C", 0, [RIGHT] * 4),
    ]
    s = S.summarize(rows, {"A": 2, "B": 1, "C": 2})
    assert s["tasks"] == 2 and s["test_inputs"] == 3 and s["samples"] == 12 and s["incomplete_tasks"] == ["C"]
    nat_a = {"pass@1": (0.25 + 0) / 2, "pass@2": (0.5 + 0) / 2, "pass@4": (1 + 0) / 2}
    frc_a = {"pass@1": (0.5 + 0) / 2, "pass@2": (1 - 1 / 6) / 2, "pass@4": (1 + 0) / 2}
    for k in (1, 2, 4):
        assert close(s["natural"][f"pass@{k}"], (nat_a[f"pass@{k}"] + 1.0) / 2)
        assert close(s["forced"][f"pass@{k}"], (frc_a[f"pass@{k}"] + 1.0) / 2)
        assert close(s["per_task"]["A"]["natural"][f"pass@{k}"], nat_a[f"pass@{k}"])
    assert close(s["closed_frac"], 10 / 12)
    assert close(s["capped_frac"], 2 / 12) and close(s["forced_frac"], 2 / 12) and s["forced_rescued"] == 1
    assert close(s["mean_gen_tokens"], (10 + 20 + 30 + 40 + 100 * 8) / 12)
    assert s["tasks_solved_natural"] == ["A", "B"]
    assert "pass@1" in S.format_summary(s)


def test_read_items_formats_and_truncated_jsonl():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "items")
        for content in ('["A", "B.json", "A"]', '{"items": [{"task_id": "A"}, {"task_id": "B"}]}',
                        "A\n# comment\n\nB.json  # trailing\n"):
            with open(p, "w") as f:
                f.write(content)
            assert S.read_items(p) == ["A", "B"], content
        j = os.path.join(d, "rows.jsonl")
        with open(j, "w") as f:
            f.write(json.dumps({"task_id": "A", "test_index": 0}) + "\n" + '{"task_id": "B", "te')
        rows, bad = S.read_jsonl(j)
        assert [S.row_key(r) for r in rows] == [("A", 0)] and bad == 1


def test_rescore_cli_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        tasks_dir = os.path.join(d, "tasks")
        os.makedirs(tasks_dir)
        with open(os.path.join(tasks_dir, "A.json"), "w") as f:
            json.dump(TASK, f)
        single = {"train": TASK["train"], "test": [TASK["test"][0]]}
        with open(os.path.join(tasks_dir, "B.json"), "w") as f:
            json.dump(single, f)
        rows = []
        for tid, j, texts, forced, truth in (
                ("A", 0, [RIGHT, WRONG, OPEN, OPEN], [None, None, FORCED_RIGHT, FORCED_WRONG], TRUTH0),
                ("A", 1, ["x</think>\n\n5", WRONG, OPEN, "x</think>\n\n4"], [None, None, SUF + "5", None], TRUTH1),
                ("B", 0, [WRONG] * 4, None, TRUTH0)):
            sc = S.score_samples(texts, forced, truth)
            rows.append({"task_id": tid, "test_index": j, "think": True, "texts": texts, "forced": forced,
                         "finish": ["stop"] * 4, "gen_tokens": [50] * 4, **sc})
        rows[2]["correct"] = [True, False, False, False]          # a stored flag that the texts do not support
        samples = os.path.join(d, "samples.jsonl")
        with open(samples, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        out = os.path.join(d, "summary.json")
        assert score_cli.main(["--samples", samples, "--tasks-dir", tasks_dir, "--out", out, "--per-task"]) == 0
        with open(out) as f:
            s = json.load(f)
        assert s["tasks"] == 2 and s["rescore"]["count_mismatches_vs_stored"] == {"correct": 1, "correct_forced": 0}
        # A: test 0 natural 1/4, forced 2/4; test 1 natural 1/4, forced 2/4.  B: 0.
        assert close(s["per_task"]["A"]["natural"]["pass@1"], 0.25)
        assert close(s["per_task"]["A"]["forced"]["pass@1"], 0.5)
        assert close(s["natural"]["pass@1"], 0.125) and close(s["forced"]["pass@1"], 0.25)
        assert close(s["forced"]["pass@2"], (1 - 1 / 6) / 2)
        # restricting to B through an item list
        items = os.path.join(d, "items.txt")
        with open(items, "w") as f:
            f.write("B\n")
        assert score_cli.main(["--samples", samples, "--tasks-dir", tasks_dir, "--items", items, "--out", out]) == 0
        with open(out) as f:
            s = json.load(f)
        assert s["tasks"] == 1 and s["natural"]["pass@4"] == 0.0


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} tests passed")
