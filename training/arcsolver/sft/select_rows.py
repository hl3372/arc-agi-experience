#!/usr/bin/env python
"""Row selection for ablation arms: from cleaned CoT rows (clean_oov.py output) keep one row per base task (a random
variant / test unit, seeded per task), excluding held-out tasks and rows whose estimated length exceeds margin x cap
(estimate = (prompt_tokens + completion_tokens) x ratio; the exact cap is applied after tokenizing, subset_data.py).

Task ids: a row's base task is its task_id with a trailing "__v<k>" (variant) and then "__t<k>" (test-input split)
removed; its unit is the row's "unit" (else "trace_id"). Task subsets are nested: --frac / --n-tasks take a prefix of
one seeded shuffle of the eligible tasks.

    python -m arcsolver.sft.select_rows --cot CLEAN.jsonl [--holdout holdout.json] --out SEL.jsonl --manifest SEL.json
holdout.json: {"tasks": [base task id, ...]}.
"""
import argparse
import collections
import json
import random
import re


def base_task(task_id: str) -> str:
    return re.sub(r"__t\d+$", "", re.sub(r"__v\d+$", "", task_id))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cot", required=True)
    ap.add_argument("--holdout", default=None, help='JSON {"tasks": [...]} of base task ids to exclude')
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cap", type=int, default=131072)
    ap.add_argument("--margin", type=float, default=0.95)
    ap.add_argument("--ratio", type=float, default=1.13, help="templated tokens / (prompt + completion tokens) estimate")
    ap.add_argument("--seed", default="abl2b-sel-20261106", help="selection seed (a string; the ablation recipe's default)")
    ap.add_argument("--variants", type=int, default=1, help="rows per task (distinct variants / units)")
    ap.add_argument("--frac", type=float, default=1.0, help="fraction of eligible tasks (nested by seed)")
    ap.add_argument("--n-tasks", type=int, default=None,
                    help="exact number of tasks (a prefix of the same shuffle; overrides --frac)")
    ap.add_argument("--pool-min-variants", type=int, default=1, help="only tasks with >= K eligible distinct units")
    ap.add_argument("--all-rows", action="store_true",
                    help="keep every eligible row of each selected task (overrides --variants)")
    ap.add_argument("--orig-first", action="store_true",
                    help="with --variants k: take the task's original-orientation units (no __v suffix) first, then "
                         "random variants")
    ap.add_argument("--answer-only", action="store_true", help="empty the reasoning of the selected rows")
    a = ap.parse_args(argv)

    hold = set(json.load(open(a.holdout))["tasks"]) if a.holdout else set()
    by = collections.defaultdict(list)
    n_rows = n_hold = n_long = 0
    for i, line in enumerate(open(a.cot)):
        r = json.loads(line)
        n_rows += 1
        t = base_task(r["task_id"])
        if t in hold:
            n_hold += 1
            continue
        est = ((r.get("prompt_tokens") or 0) + (r.get("completion_tokens") or 0)) * a.ratio
        if est > a.margin * a.cap:
            n_long += 1
            continue
        by[t].append((r.get("unit") or r.get("trace_id"), i))
    tasks = sorted(t for t in by if len({u for u, _ in by[t]}) >= a.pool_min_variants)
    rng = random.Random(a.seed)
    order = tasks[:]
    rng.shuffle(order)          # nested subsets: a prefix of one shuffle
    tasks_kept = set(order[:a.n_tasks] if a.n_tasks is not None else order[:round(a.frac * len(order))])
    assert a.n_tasks is None or len(tasks_kept) == a.n_tasks, (len(tasks_kept), a.n_tasks)
    pick = {}
    for t in tasks:
        if t not in tasks_kept:
            continue
        cand = sorted(set(by[t]), key=lambda x: x[1])
        units = {}
        for u, i in cand:
            units.setdefault(u, []).append(i)
        if a.all_rows:
            pick[t] = [i for u, i in cand]
            continue
        if len(units) < a.variants:
            continue
        r = random.Random(f"{a.seed}|{t}")
        if a.orig_first:
            origs = sorted(u for u in units if "__v" not in u)
            rest = sorted(u for u in units if "__v" in u)
            us = r.sample(origs, min(len(origs), a.variants)) + r.sample(rest, max(0, a.variants - len(origs)))
        else:
            us = r.sample(sorted(units), a.variants)
        pick[t] = [r.choice(units[u]) for u in us]
    want = {i for v in pick.values() for i in v}
    with open(a.out, "w") as f:
        for i, line in enumerate(open(a.cot)):
            if i in want:
                if a.answer_only:
                    r = json.loads(line)
                    pm = r.pop("prompt_messages")
                    ans = str(r.pop("answer")).strip("\n")
                    r.pop("reasoning", None)
                    r.pop("body", None)
                    # what the chat template writes for a non-thinking turn
                    r["messages"] = pm + [{"role": "assistant", "content": "<think>\n\n</think>\n\n" + ans}]
                    line = json.dumps(r, ensure_ascii=False) + "\n"
                f.write(line)
    json.dump({"args": vars(a), "rows_in": n_rows, "rows_heldout": n_hold, "rows_too_long_est": n_long,
               "tasks_eligible": len(tasks), "tasks_selected": len(pick), "rows_selected": len(want), "pick": pick},
              open(a.manifest, "w"))
    print(f"SELECT rows_in {n_rows} heldout_rows {n_hold} too_long_est {n_long} eligible_tasks {len(tasks)} "
          f"selected_tasks {len(pick)} rows {len(want)}")


if __name__ == "__main__":
    main()
