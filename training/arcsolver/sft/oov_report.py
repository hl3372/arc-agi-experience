#!/usr/bin/env python
"""Out-of-vocab report for CoT rows against a pruned vocabulary: every occurrence of an original-tokenizer token
outside the keep set, per field (system / user / reasoning / answer), grouped by the Unicode category of its
non-ASCII characters (latin with diacritics, CJK/kana/hangul, greek, cyrillic, symbols/emoji, box drawing/geometric
shapes, other), with counts, rows affected and context snippets. Also the helpers clean_oov.py uses.
    python -m arcsolver.sft.oov_report --cot COT.jsonl --orig ORIGINAL_TOKENIZER_DIR --keep keep.json --out report.json
"""
import argparse
import collections
import json
import unicodedata
from multiprocessing import Pool

TOK = KEEP = None


def init(orig, keep):
    global TOK, KEEP
    from transformers import AutoTokenizer
    TOK = AutoTokenizer.from_pretrained(orig)
    KEEP = set(json.load(open(keep))["keep"])


def category(s):
    cats = set()
    for ch in s:
        if ord(ch) < 128:
            continue
        name = unicodedata.name(ch, "")
        cp = ord(ch)
        if 0x4E00 <= cp <= 0x9FFF or 0x3040 <= cp <= 0x30FF or 0xAC00 <= cp <= 0xD7AF or 0x3000 <= cp <= 0x303F \
                or 0xFF00 <= cp <= 0xFFEF:
            cats.add("cjk")
        elif "LATIN" in name:
            cats.add("latin_diacritic")
        elif "GREEK" in name:
            cats.add("greek")
        elif "CYRILLIC" in name:
            cats.add("cyrillic")
        elif 0x2500 <= cp <= 0x25FF or 0x2190 <= cp <= 0x21FF or 0x2460 <= cp <= 0x24FF:
            cats.add("box_shape_arrow")
        elif unicodedata.category(ch) == "So" or 0x1F000 <= cp or 0x2600 <= cp <= 0x27BF:
            cats.add("symbol_emoji")
        else:
            cats.add("other")
    return "+".join(sorted(cats)) or "ascii_or_bytes"


def one(line):
    t = json.loads(line)
    fields = [("system", t["prompt_messages"][0]["content"]), ("user", t["prompt_messages"][1]["content"]),
              ("reasoning", t["reasoning"]), ("answer", t["answer"])]
    hits = []
    for fname, text in fields:
        enc = TOK(text, add_special_tokens=False, return_offsets_mapping=True)
        for tid, (a, b) in zip(enc["input_ids"], enc["offset_mapping"]):
            if tid not in KEEP:
                hits.append((fname, text[a:b], text[max(0, a - 60):b + 40]))
    return t["trace_id"], hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cot", required=True)
    ap.add_argument("--orig", required=True)
    ap.add_argument("--keep", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    by_tok = collections.defaultdict(lambda: {"count": 0, "rows": set(), "fields": collections.Counter(), "snips": []})
    by_cat = collections.defaultdict(lambda: {"count": 0, "rows": set(), "tokens": collections.Counter()})
    by_field = collections.Counter()
    rows_hit = set()
    with Pool(a.workers, initializer=init, initargs=(a.orig, a.keep)) as pool:
        for tid, hits in pool.imap_unordered(one, open(a.cot, encoding="utf-8"), chunksize=8):
            for fname, s, snip in hits:
                rows_hit.add(tid)
                by_field[fname] += 1
                e = by_tok[s]
                e["count"] += 1; e["rows"].add(tid); e["fields"][fname] += 1
                if len(e["snips"]) < 3:
                    e["snips"].append(snip)
                c = by_cat[category(s)]
                c["count"] += 1; c["rows"].add(tid); c["tokens"][s] += 1
    out = {"rows_hit": len(rows_hit), "occurrences": sum(by_field.values()), "by_field": dict(by_field),
           "by_category": {k: {"count": v["count"], "rows": len(v["rows"]), "top": v["tokens"].most_common(25)}
                           for k, v in sorted(by_cat.items(), key=lambda kv: -kv[1]["count"])},
           "tokens": [{"token": s, "count": e["count"], "rows": len(e["rows"]), "fields": dict(e["fields"]),
                       "category": category(s), "snips": e["snips"]}
                      for s, e in sorted(by_tok.items(), key=lambda kv: -kv[1]["count"])]}
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: out[k] for k in ("rows_hit", "occurrences", "by_field")}))
    for k, v in out["by_category"].items():
        print(f"{k:28s} {v['count']:5d} occ {v['rows']:4d} rows  top: " + ", ".join(f"{s!r}x{n}" for s, n in v["top"][:12]))


if __name__ == "__main__":
    main()
