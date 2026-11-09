"""Evaluation split for packing_v2 rows (DESIGN.md section 11 and 13.12).

    python -m pv2.eval_v2 make  --n 600 --seed 424242 --out out/eval_v2.jsonl
    python -m pv2.eval_v2 score --eval out/eval_v2.jsonl --pred preds.jsonl      (one {"id", "text"} per line)

Scores (per row and per question):
* strict   -- the prediction, after ``</think>``, equals the target exactly (whitespace-trimmed);
* lenient  -- per question: the section is parsed with the row's own style parser and each answer
              compared (grids as grids, text with whitespace normalised);
* found    -- per question, a last resort for predictions that break the style: the expected answer
              occurs in the prediction section (single-line answers only).
Breakdowns per family, answer style and grid encoding.  The old eval split (single items,
instruction first) is untouched and stays the comparison baseline.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from types import SimpleNamespace

from pv2 import encodings as enc
from pv2.assemble import styles as S
from pv2.validate import ValidationError, _target_sections

THINK_END = "</think>"
_STOPS = ("<|im_end|>", "<|im_start|>", "\nuser\n", "\nassistant\n")


def answer_text(raw: str) -> str:
    """The model's answer: after the last ``</think>``, cut at chat control tokens, trimmed."""
    text = raw.split(THINK_END)[-1]
    for stop in _STOPS:
        i = text.find(stop)
        if i >= 0:
            text = text[:i]
    return text.strip()


def _norm(s: str) -> str:
    return re.sub(r"[ \t]+", " ", s.strip())


def _grid_value(text: str, encoding: str):
    """A grid answer read as VALUES: JSON by json.loads (any spacing), others line by line with
    trailing spaces dropped, then the encoding's canonical parser."""
    text = text.strip("\n")
    if encoding in ("json_flat", "json_rows"):
        return enc.check_grid(json.loads(text))
    return enc.parse("\n".join(line.rstrip() for line in text.split("\n")), encoding)


def _same(expected: str, got: str, grid_encoding: str | None, is_grid: bool) -> bool:
    if got is None:
        return False
    if is_grid and grid_encoding:
        try:
            return _grid_value(got, grid_encoding) == _grid_value(expected, grid_encoding)
        except (ValueError, KeyError, TypeError):
            return False
    return _norm(expected) == _norm(got)


def _found(expected: str, section: str) -> bool:
    """The expected one-line value occurs in the section as a whole word (so "no" is not found
    inside "not", 7 not inside 17)."""
    if "\n" in expected:
        return False
    return re.search(rf"(?<![\w]){re.escape(_norm(expected))}(?![\w])", _norm(section)) is not None


def make_eval_set(n: int, seed: int, *, families=None) -> list[dict]:
    """``n`` v2 rows (no classic rows: the old eval covers those) with everything scoring needs."""
    from pv2.build_dataset import BuildConfig, build_row
    from pv2.tokens import load_tokenizer
    cfg = BuildConfig(classic_share=0.0)
    if families:
        cfg.family_weights = {f: w for f, w in cfg.family_weights.items() if f in families}
    tok = load_tokenizer()
    out, s = [], 0
    while len(out) < n:
        row_seed = seed * 10_000_000 + s
        s += 1
        try:
            built = build_row(row_seed, cfg, tok)
        except Exception:  # noqa: BLE001
            continue
        if built is None:
            continue
        prompt, target, meta = built
        out.append({"id": f"pv2eval|{row_seed}", "prompt": prompt, "target": target,
                    "families": meta["families"], "checks": meta["assembled"]["checks"],
                    "parts": [{"style": p["style"], "encodings": p["encodings"],
                               "grid_answer_encoding": p["grid_answer_encoding"]}
                              for p in meta["assembled"]["parts"]]})
    return out


def score(record: dict, raw_prediction: str) -> dict:
    pred = answer_text(raw_prediction)
    target = record["target"].strip()
    checks = record["checks"]
    try:
        sections = _target_sections(SimpleNamespace(target=pred, meta={"checks": checks}))
    except ValidationError:
        sections = [pred] if not checks["multi"] else [pred] * len(checks["parts"])
    questions = []
    for pc, section in zip(checks["parts"], sections):
        spec = S.StyleSpec.from_dict(pc["style"])
        grid_enc = pc.get("grid_answer_encoding")
        try:
            got = S.parse(spec, section)
        except (ValueError, KeyError, TypeError):
            got = None
        for i, (shape, d) in enumerate(zip(spec.shapes, pc["questions"])):
            expected = d["answer"]
            g = got[i] if got is not None and i < len(got) else None
            ok = _same(expected, g, grid_enc, shape.grid)
            # "found" is the fallback for a reply that BREAKS the style (nothing parsed); a reply that
            # parses but gives a wrong value is wrong, whatever else the section contains
            found = ok or (got is None and _found(expected, section))
            questions.append({"tag": d["tag"], "kind": shape.kind, "grid": shape.grid, "style": spec.style,
                              "ok": ok, "found": found})
    n = max(1, len(questions))
    return {"id": record["id"], "strict": pred == target,
            "lenient": sum(q["ok"] for q in questions) / n,
            "found": sum(q["found"] for q in questions) / n,
            "all_ok": all(q["ok"] for q in questions), "questions": questions}


def summarize(records: list[dict], scores: list[dict]) -> dict:
    by = defaultdict(lambda: {"rows": 0, "strict": 0, "q": 0, "ok": 0, "found": 0})
    for rec, sc in zip(records, scores):
        keys = ["all"] + [f"family:{f}" for f in set(rec["families"])]
        keys += [f"encoding:{e}" for p in rec["parts"] for e in set(p["encodings"])]
        for k in set(keys):
            by[k]["rows"] += 1
            by[k]["strict"] += sc["strict"]
        for q in sc["questions"]:
            for k in ("all", f"style:{q['style']}", f"kind:{q['kind']}"):
                by[k]["q"] += 1
                by[k]["ok"] += q["ok"]
                by[k]["found"] += q["found"]
    out = {}
    for k, v in sorted(by.items()):
        out[k] = {"rows": v["rows"], "strict": round(v["strict"] / v["rows"], 4) if v["rows"] else None,
                  "questions": v["q"], "lenient": round(v["ok"] / v["q"], 4) if v["q"] else None,
                  "found": round(v["found"] / v["q"], 4) if v["q"] else None}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make")
    mk.add_argument("--n", type=int, default=600)
    mk.add_argument("--seed", type=int, default=424242)
    mk.add_argument("--out", required=True)
    sc = sub.add_parser("score")
    sc.add_argument("--eval", required=True)
    sc.add_argument("--pred", required=True, help='jsonl of {"id", "text"} (raw generations)')
    args = ap.parse_args(argv)
    if args.cmd == "make":
        records = make_eval_set(args.n, args.seed)
        with open(args.out, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(records)} eval rows to {args.out}")
        return
    records = [json.loads(line) for line in open(args.eval, encoding="utf-8")]
    preds = {}
    for line in open(args.pred, encoding="utf-8"):
        d = json.loads(line)
        preds[d["id"]] = d["text"]
    kept = [r for r in records if r["id"] in preds]
    scores = [score(r, preds[r["id"]]) for r in kept]
    json.dump(summarize(kept, scores), sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
