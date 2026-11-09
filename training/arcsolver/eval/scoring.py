"""Scoring for the P4 ARC-AGI-2 eval protocol (pure python: no vLLM, torch or transformers).

A sample is correct when its reasoning closed (`arc_render.final_answer_text` finds "</think>") and the text
after the reasoning parses (`arc_render.answer_grids(answer, 1)`) to exactly one grid equal to the test output.
With --no-think prompting the chat template closes the (empty) reasoning block inside the prompt, so the whole
reply is the answer and every sample counts as closed.

Two views per sample:
  natural  the sample as generated (the headline numbers);
  forced   a sample cut at the token cap with its reasoning still open is continued after the force suffix
           (reasoning closed by the harness) for a short answer budget, and scored on text + suffix + answer;
           every other sample keeps its natural score.

Per test input: fraction correct of the k samples (pass@1) and unbiased pass@k for k in KS (k <= samples).
Per task: the mean over its test inputs. Reported: the mean over tasks (tasks with every test input scored).
"""
from __future__ import annotations

import glob
import gzip
import json
import os

from arcsolver.prompt import arc_render as ar

ARM = "r2k_zb_max"
# The force suffix of the reference runs: the traces' own last reasoning line, then the closing tag.
DEFAULT_FORCE_SUFFIX = "\n\nNow final.\n</think>\n\n"
KS = (1, 2, 4, 8)


# --- tasks and item lists -------------------------------------------------------------------------

def task_ids_in(tasks_dir: str) -> list[str]:
    """Task ids (file stems) of the *.json files in a task directory, sorted."""
    return sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(tasks_dir, "*.json")))


def read_items(path: str) -> list[str]:
    """Task ids from an item list file, in file order, duplicates dropped. Accepted formats:
    a JSON list of ids or of {"task_id": ...} objects, a JSON object with such a list under "items",
    or plain text with one id per line (blank lines and '#' comments ignored). A trailing ".json" is dropped."""
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = None
    if data is None:
        ids = [line.split("#", 1)[0].strip() for line in raw.splitlines()]
    else:
        if isinstance(data, dict):
            data = data.get("items")
        if not isinstance(data, list):
            raise ValueError(f"{path}: expected a JSON list of task ids or an object with an 'items' list")
        ids = [x["task_id"] if isinstance(x, dict) else str(x) for x in data]
    out, seen = [], set()
    for t in ids:
        t = t[:-5] if t.endswith(".json") else t
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def load_tasks(tasks_dir: str, task_ids: list[str], need_outputs: bool = True) -> dict[str, dict]:
    """{task_id: task JSON}. Raises if a task file is missing or (need_outputs) a test output is absent."""
    missing = [t for t in task_ids if not os.path.isfile(os.path.join(tasks_dir, t + ".json"))]
    if missing:
        raise FileNotFoundError(f"{len(missing)} task files not in {tasks_dir}, e.g. {missing[:5]}")
    tasks = {t: ar.load_task(os.path.join(tasks_dir, t + ".json")) for t in task_ids}
    if need_outputs:
        bad = [t for t, task in tasks.items() if not all("output" in x for x in task["test"])]
        if bad:
            raise ValueError(f"{len(bad)} tasks have test inputs without an output (needed for scoring), "
                             f"e.g. {bad[:5]}")
    return tasks


def single_test_task(task: dict, test_index: int) -> dict:
    """The sub-task of one test input: every training pair, only that test input (split tests)."""
    return {"train": task["train"], "test": [task["test"][test_index]]}


# --- rows on disk ---------------------------------------------------------------------------------

def read_jsonl(path: str) -> tuple[list[dict], int]:
    """(rows, number of unreadable lines). Gzip if the name ends in .gz. An interrupted write leaves at most a
    truncated last line, which is counted and skipped."""
    rows, bad = [], 0
    if not os.path.exists(path):
        return rows, bad
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    return rows, bad


def row_key(row: dict) -> tuple[str, int]:
    return str(row["task_id"]), int(row.get("test_index", 0))


# --- per-sample scoring ---------------------------------------------------------------------------

def answer_of(text: str, think: bool = True) -> tuple[str, bool]:
    """(answer text, reasoning closed)."""
    if not think:
        return text or "", True
    return ar.final_answer_text(text or "", "think")


def is_correct(text: str, truth_key: str, think: bool = True) -> tuple[bool, bool]:
    """(closed, correct) of one completion against one test output's grid_key."""
    if truth_key is None:
        raise ValueError("no test output to score against")
    answer, closed = answer_of(text, think)
    if not closed:
        return False, False
    grids = ar.answer_grids(answer, 1, mode="strict")[0]
    ok = bool(grids) and len(grids) == 1 and grids[0] is not None and ar.grid_key(grids[0]) == truth_key
    return True, ok


def needs_forced_answer(finish: str, text: str, think: bool = True) -> bool:
    """A sample gets a forced answer when it stopped at the token cap with its reasoning still open."""
    return think and finish == "length" and not answer_of(text, True)[1]


def score_samples(texts: list[str], forced: list | None, truth_key: str, think: bool = True) -> dict:
    """Per-sample closed / correct (natural) / correct_forced flags. forced[k] is the continuation appended to
    texts[k] (force suffix + forced answer) or None."""
    closed, correct, correct_forced = [], [], []
    for k, text in enumerate(texts):
        c, ok = is_correct(text, truth_key, think)
        f = forced[k] if forced and k < len(forced) else None
        closed.append(c)
        correct.append(ok)
        correct_forced.append(is_correct(text + f, truth_key, think)[1] if f is not None else ok)
    return {"closed": closed, "correct": correct, "correct_forced": correct_forced}


def pass_metrics(correct: list[bool], ks=KS) -> dict:
    """pass@k (unbiased) for every k in ks with k <= n; pass@1 = fraction correct."""
    n, c = len(correct), sum(bool(x) for x in correct)
    return {f"pass@{k}": ar.pass_at_k(n, c, k) for k in ks if k <= n}


def rescore_row(row: dict, truth_key: str) -> dict:
    """A stored row's flags recomputed from its texts (and forced continuations, if any)."""
    think = bool(row.get("think", True))
    sc = score_samples(row["texts"], row.get("forced"), truth_key, think)
    return {"task_id": str(row["task_id"]), "test_index": int(row.get("test_index", 0)), **sc,
            "gen_tokens": list(row.get("gen_tokens") or []), "finish": list(row.get("finish") or []),
            "forced": row.get("forced")}


# --- aggregation ----------------------------------------------------------------------------------

def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def summarize(rows: list[dict], task_tests: dict[str, int], ks=KS) -> dict:
    """The P4 summary of scored rows (each with task_id, test_index, closed, correct, correct_forced and, if
    available, gen_tokens / finish / forced). task_tests = {task_id: number of test inputs} of the selected
    tasks; a task counts only when every one of its test inputs has a row (others are listed as incomplete).
    Rates are fractions in [0, 1]."""
    by_task: dict[str, dict[int, dict]] = {}
    for r in rows:
        if r["task_id"] in task_tests:
            by_task.setdefault(r["task_id"], {}).setdefault(int(r["test_index"]), r)
    complete = [t for t in task_tests if len(by_task.get(t, {})) == task_tests[t]
                and set(by_task[t]) == set(range(task_tests[t]))]
    done = set(complete)
    incomplete = sorted(t for t in task_tests if t not in done)
    used = [by_task[t][j] for t in complete for j in range(task_tests[t])]
    n_min = min((len(r["correct"]) for r in used), default=0)
    n_max = max((len(r["correct"]) for r in used), default=0)
    ks = [k for k in ks if k <= n_min]
    per_task = {}
    for t in complete:
        parts = [by_task[t][j] for j in range(task_tests[t])]
        per_task[t] = {view: {f"pass@{k}": _mean(pass_metrics(p[flag], ks)[f"pass@{k}"] for p in parts) for k in ks}
                       for view, flag in (("natural", "correct"), ("forced", "correct_forced"))}
    samples = [(r, k) for r in used for k in range(len(r["correct"]))]
    tokens = [n for r in used for n in (r.get("gen_tokens") or [])]
    finish = [x for r in used for x in (r.get("finish") or [])]
    n_forced = sum(1 for r in used for f in (r.get("forced") or []) if f is not None)
    out = {
        "tasks": len(complete), "test_inputs": len(used), "samples": len(samples),
        "samples_per_test_input": n_min if n_min == n_max else [n_min, n_max],
        "natural": {f"pass@{k}": _mean(v["natural"][f"pass@{k}"] for v in per_task.values()) for k in ks},
        "forced": {f"pass@{k}": _mean(v["forced"][f"pass@{k}"] for v in per_task.values()) for k in ks},
        "closed_frac": _mean(float(r["closed"][k]) for r, k in samples),
        "capped_frac": _mean(float(x == "length") for x in finish) if finish else None,
        "forced_frac": n_forced / len(samples) if samples else 0.0,
        "forced_rescued": sum(1 for r, k in samples if r["correct_forced"][k] and not r["correct"][k]),
        "mean_gen_tokens": _mean(tokens) if tokens else None,
        "tasks_solved_natural": sorted(t for t in complete if any(any(by_task[t][j]["correct"])
                                                                  for j in by_task[t])),
        "incomplete_tasks": incomplete,
        "per_task": per_task,
    }
    return out


def format_summary(s: dict) -> str:
    """A short human-readable report of a summary (percentages)."""
    ks = [k for k in KS if f"pass@{k}" in s["natural"]]

    def pct(x):
        return "-" if x is None else f"{100 * x:.1f}"
    lines = [f"tasks {s['tasks']}  test inputs {s['test_inputs']}  samples {s['samples']}"
             + (f"  (incomplete tasks: {len(s['incomplete_tasks'])})" if s["incomplete_tasks"] else "")]
    for view in ("natural", "forced"):
        lines.append(f"{view:8s} " + "  ".join(f"pass@{k} {pct(s[view][f'pass@{k}'])}" for k in ks))
    tok = s["mean_gen_tokens"]
    lines.append(f"closed {pct(s['closed_frac'])}%  capped {pct(s['capped_frac'])}%  forced {pct(s['forced_frac'])}% "
                 f"(rescued {s['forced_rescued']})  mean generated tokens "
                 + ("-" if tok is None else f"{tok:,.0f}"))
    return "\n".join(lines)
