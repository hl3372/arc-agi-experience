"""Re-score an eval JSONL against a task directory, without generation.

    python -m arcsolver.eval.score --samples OUT/samples.jsonl --tasks-dir DATA/evaluation [--items ITEMS] [--out S.json]

Every row's natural and forced flags are recomputed from its stored texts (and forced continuations) with the
P4 rule (arcsolver.eval.scoring), the counts are compared with the flags stored in the row, and the P4 summary
is printed and written. Rows of other tasks than the selected ones are ignored; selected tasks without a row for
every test input are reported as incomplete and left out of the averages. A .jsonl.gz file is read as gzip.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from arcsolver.eval import scoring as S
from arcsolver.prompt import arc_render as ar


def stored_count(row: dict, key: str) -> int | None:
    """The natural / forced correct count stored in a row (this tool's format), if any."""
    flags = row.get(key)
    if isinstance(flags, list) and flags and all(isinstance(x, bool) for x in flags):
        return sum(flags)
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", required=True, help="the eval JSONL (one row per test input)")
    ap.add_argument("--tasks-dir", required=True, help="directory of ARC task JSONs with test outputs")
    ap.add_argument("--items", default="", help="restrict to these task ids (see scoring.read_items for formats); "
                                                "default: every task that has a row")
    ap.add_argument("--out", default="", help="summary JSON (default: <samples dir>/summary_rescored.json)")
    ap.add_argument("--per-task", action="store_true", help="keep per-task scores in the written summary")
    args = ap.parse_args(argv)

    rows, bad = S.read_jsonl(args.samples)
    if bad:
        print(f"warning: {bad} unreadable lines in {args.samples} skipped", file=sys.stderr)
    if not rows:
        print(f"no rows in {args.samples}", file=sys.stderr)
        return 1
    task_ids = S.read_items(args.items) if args.items else sorted({str(r["task_id"]) for r in rows})
    tasks = S.load_tasks(args.tasks_dir, task_ids)
    wanted = set(task_ids)

    scored, seen, dupes, out_of_range = [], set(), 0, 0
    mism = {"correct": 0, "correct_forced": 0}
    for row in rows:
        tid, j = S.row_key(row)
        if tid not in wanted:
            continue
        if (tid, j) in seen:
            dupes += 1
            continue
        seen.add((tid, j))
        if not 0 <= j < len(tasks[tid]["test"]):
            out_of_range += 1
            continue
        r = S.rescore_row(row, ar.grid_key(tasks[tid]["test"][j]["output"]))
        for key in mism:
            stored = stored_count(row, key)
            mism[key] += stored is not None and stored != sum(r[key])
        scored.append(r)
    if dupes or out_of_range:
        print(f"warning: {dupes} duplicate rows ignored (first kept), {out_of_range} rows with a test index "
              f"outside their task", file=sys.stderr)

    summary = S.summarize(scored, {t: len(tasks[t]["test"]) for t in task_ids})
    if not args.per_task:
        summary.pop("per_task")
    summary["rescore"] = {"samples_file": os.path.abspath(args.samples), "tasks_dir": os.path.abspath(args.tasks_dir),
                          "rows_scored": len(scored), "count_mismatches_vs_stored": mism}
    print(S.format_summary(summary))
    print(f"stored-flag mismatches (test inputs): natural {mism['correct']}, forced {mism['correct_forced']}")
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.samples)), "summary_rescored.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"summary written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
