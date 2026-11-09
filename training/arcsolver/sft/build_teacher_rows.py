#!/usr/bin/env python
"""Teacher attempts -> CoT rows for build_cache.py --cot.

Keeps the attempts that pass two checks:
  1. the request's messages equal arc_render.build_messages(task, --arm) byte for byte (role and content of every
     message), so the student is trained on exactly the prompt format it is evaluated / RL-trained with;
  2. reasoning + "\\n</think>\\n\\n" + answer scores correct on the task with arc_render.score_task (the RL reward's
     parser): the first test output by default (attempts on single-test-input units), every test output with
     --require-all-tests.
Attempts with an explicit "scored_correct": false (the generation harness's own score) are skipped before the checks.

Input: one JSON object per teacher attempt, either one per line in .jsonl files or one per .json file; a directory
argument is searched recursively for *.jsonl and *.json. Fields:
    messages          list of {"role", "content"}: the chat request sent to the teacher (system + user)   required
    reasoning         str: the teacher's reasoning / thinking text (no <think> tags)                       required
    answer            str: the teacher's final answer (the visible reply)                                  required
    task_file         str: the ARC task JSON the prompt was rendered from (relative: to --task-root)       required
    task_id           str: default = the task file's stem (variants "__v<k>", test splits "__t<k>")
    unit              str: default = task_id (one attempt target; --exclude matches it)
    trace_id          str: default = "<bench>:<unit>" when bench is set, else "<input file>:<line or name>"
    scored_correct    bool, optional
    model, source, cohort, bench   optional metadata, copied (source defaults to "teacher")
    prompt_tokens, completion_tokens   optional ints (select_rows.py's length estimate, stats)
Output jsonl: {task_id, trace_id, model, source, arm, cohort, bench, unit, prompt_messages, reasoning, answer,
completion_tokens, prompt_tokens}; prints counts per status, tokens, the length distribution and rows over common caps.

    python -m arcsolver.sft.build_teacher_rows OUT.jsonl --attempts ATTEMPTS [...] [--task-root DIR] [--workers N]
        [--exclude FILE ...] [--arm r2k_zb_max]
"""
import argparse
import collections
import glob
import json
import os
import sys
from multiprocessing import Pool

from arcsolver.prompt import arc_render as ar

CFG = {"arm": "r2k_zb_max", "task_root": "", "exclude": [], "all_tests": False}


def _init(cfg):
    CFG.update(cfg)


def iter_jobs(paths):
    """(label, raw JSON text) for every attempt in the given files / directories."""
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "**", "*.jsonl"), recursive=True))
            files += sorted(glob.glob(os.path.join(p, "**", "*.json"), recursive=True))
        else:
            files.append(p)
    for f in files:
        if f.endswith(".jsonl"):
            with open(f, encoding="utf-8") as fh:
                for line_no, line in enumerate(fh, 1):
                    if line.strip():
                        yield f"{os.path.basename(f)}:{line_no}", line
        else:
            with open(f, encoding="utf-8") as fh:
                yield os.path.splitext(os.path.basename(f))[0], fh.read()


def one(job):
    label, text = job
    try:
        r = json.loads(text)
    except ValueError:
        return ("bad_record", None)
    if not isinstance(r, dict) or not r.get("messages") or not r.get("task_file"):
        return ("bad_record", None)
    tf = r["task_file"]
    if not os.path.isabs(tf) and CFG["task_root"]:
        tf = os.path.join(CFG["task_root"], tf)
    task_id = r.get("task_id") or os.path.splitext(os.path.basename(tf))[0]
    unit = r.get("unit") or task_id
    if any(x in unit for x in CFG["exclude"]):
        return ("excluded", None)
    if r.get("scored_correct") is False:
        return ("not_correct", None)
    try:
        task = ar.load_task(tf)
    except (OSError, ValueError):
        return ("no_task", None)
    req = r["messages"]
    want = ar.build_messages(task, CFG["arm"])
    if [(m.get("role"), m.get("content")) for m in req] != [(m["role"], m["content"]) for m in want]:
        return ("prompt_mismatch", None)
    reasoning = str(r.get("reasoning") or "").strip("\n")
    answer = str(r.get("answer") or "").strip("\n")
    if not reasoning or not answer or "</think>" in reasoning or "<think>" in reasoning:
        return ("bad_text", None)
    per = ar.score_task(task, [reasoning + "\n</think>\n\n" + answer], ks=(1,))["per_output"]
    ok = all(p.get("correct", 0) == 1 for p in per) if CFG["all_tests"] else per[0].get("correct", 0) == 1
    if not ok:
        return ("rescore_wrong", None)
    bench = r.get("bench")
    trace_id = r.get("trace_id") or (f"{bench}:{unit}" if bench else label)
    return ("kept", {"task_id": task_id, "trace_id": trace_id, "model": r.get("model"),
                     "source": r.get("source") or "teacher", "arm": CFG["arm"], "cohort": r.get("cohort"),
                     "bench": bench, "unit": unit,
                     "prompt_messages": [{"role": m["role"], "content": m["content"]} for m in req],
                     "reasoning": reasoning, "answer": answer,
                     "completion_tokens": int(r.get("completion_tokens") or 0),
                     "prompt_tokens": int(r.get("prompt_tokens") or 0)})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--attempts", nargs="+", required=True, help="attempt .jsonl / .json files or directories")
    ap.add_argument("--task-root", default="", help="base dir for relative task_file paths (default: the CWD)")
    ap.add_argument("--arm", default="r2k_zb_max", help="arc_render arm the prompts must match")
    ap.add_argument("--require-all-tests", action="store_true",
                    help="re-score: every test output must be correct (default: the first one)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--exclude", nargs="+", default=[], help="file(s) of unit-name substrings to drop (# comments)")
    a = ap.parse_args(argv)
    if a.arm not in ar.ARMS:
        sys.exit(f"unknown arm {a.arm!r} (arc_render.ARMS: {sorted(ar.ARMS)})")
    exclude = []
    for path in a.exclude:
        exclude.extend(ln.strip() for ln in open(path) if ln.strip() and not ln.lstrip().startswith("#"))
    if exclude:
        print(f"excluding units containing any of {len(exclude)} substrings:", exclude)
    cfg = {"arm": a.arm, "task_root": a.task_root, "exclude": exclude, "all_tests": a.require_all_tests}
    st, kept = collections.Counter(), []
    with Pool(a.workers, initializer=_init, initargs=(cfg,)) as pool, open(a.out, "w", encoding="utf-8") as f:
        for status, rec in pool.imap_unordered(one, iter_jobs(a.attempts), chunksize=16):
            st[status] += 1
            if rec:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept.append((rec["bench"], rec["cohort"], rec["unit"], rec["completion_tokens"], rec["prompt_tokens"]))
    print("attempts:", sum(st.values()), dict(st))
    if not kept:
        return
    tot = sum(c for *_, c, p in kept)
    ptot = sum(p for *_, c, p in kept)
    print(f"kept {len(kept)} traces: completion {tot/1e6:.1f}M tokens, prompt {ptot/1e6:.1f}M, total {(tot+ptot)/1e6:.1f}M")
    rows = sorted(c + p for *_, c, p in kept)

    def q(x):
        return rows[min(len(rows) - 1, int(x * len(rows)))]
    print(f"row length (prompt+completion) p50 {q(.5)/1000:.0f}k p90 {q(.9)/1000:.0f}k p99 {q(.99)/1000:.0f}k "
          f"max {rows[-1]/1000:.0f}k")
    for cap in (131072, 147456, 163840, 196608):
        over = [x for x in rows if x > cap]
        print(f"  rows > {cap}: {len(over)} ({100*len(over)/len(rows):.1f}%), {sum(over)/1e6:.1f}M tokens")
    print("by bench:", dict(collections.Counter(k[0] for k in kept)))
    print("by cohort:", dict(collections.Counter(k[1] for k in kept).most_common()))
    print("distinct units:", len({k[2].rsplit('__v', 1)[0] for k in kept}))


if __name__ == "__main__":
    main()
