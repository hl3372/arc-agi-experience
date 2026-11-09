#!/usr/bin/env python
"""Out-of-vocab cleanup of CoT rows against a pruned vocabulary (vocab_prune.py keep set).

In the REASONING only (prompts and answers are never touched), every original-tokenizer token outside the keep set is
  REPLACED when it is one of four replaceable kinds:
    1. Latin letters with diacritics -> plain letters (NFKD, combining marks dropped; standalone Ø/ø -> "-", in words O/o;
       ß ss, æ ae, œ oe, ł l, đ d, ı i)
    2. check marks / stars: ✅ ✔ ☑ -> "(yes)", ❌ ✗ ✘ -> "(no)", ★ ☆ -> "*", ❮ ❯ -> < >
    3. full-width / CJK punctuation -> ASCII (NFKC; 、-> , 。-> . 「」-> "), U+FFFD removed, Cyrillic look-alike letters
       next to Latin letters -> Latin (ARС -> ARC); a few math forms (ℕ -> N, 𝑓 -> f, ⟨ ⟩ -> < >, € -> EUR, ، -> ,)
    4. Russian words -> English (RUSSIAN glossary, whole-word lookup; WORDS: exact mixed-script words)
  otherwise the row is DROPPED and listed for regeneration (Chinese/Japanese/Korean/other-script words, Greek letters,
  box / shape / arrow symbols, glyphs used as shapes, anything unmapped).
After the edits the reasoning is re-tokenized: a kept row must have no OOV token left (else dropped as well).
Input rows: {"trace_id", "reasoning", ...} (build_teacher_rows.py output). Outputs: --out (cleaned jsonl, same
records; edited ones get "oov_edits": n), --regen (dropped rows: trace_id, bench, unit, task_id, cohort, pieces),
--edits (tsv: trace_id, before context, after context), summary on stdout.
    python -m arcsolver.sft.clean_oov --cot IN.jsonl --orig ORIGINAL_TOKENIZER_DIR --keep keep.json --out OUT.jsonl \\
        --regen REGEN.jsonl --edits EDITS.tsv [--workers 64]
"""
import argparse
import collections
import json
import os
import re
import unicodedata
from multiprocessing import Pool

from arcsolver.sft import oov_report as orep

CHECKS = {"✅": "(yes)", "✔": "(yes)", "☑": "(yes)", "✓": "(yes)", "❌": "(no)", "✗": "(no)", "✘": "(no)", "❎": "(no)",
          "★": "*", "☆": "*", "❮": "<", "❯": ">"}
LATIN_EXTRA = {"ß": "ss", "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE", "ł": "l", "Ł": "L", "đ": "d", "Đ": "D", "ı": "i",
               "ø": "o", "Ø": "O", "þ": "th", "ð": "d"}
PUNCT = {"、": ",", "。": ".", "「": '"', "」": '"', "『": '"', "』": '"', "【": "[", "】": "]", "〈": "<", "〉": ">",
         "《": "<<", "》": ">>", "～": "~", "⟨": "<", "⟩": ">", "€": "EUR", "،": ",", "�": ""}
HOMOGLYPH = {"А": "A", "В": "B", "С": "C", "Е": "E", "Н": "H", "К": "K", "М": "M", "О": "O", "Р": "P", "Т": "T",
             "Х": "X", "а": "a", "с": "c", "е": "e", "о": "o", "р": "p", "х": "x", "у": "y", "і": "i"}
RUSSIAN = {"правильно": "properly", "результат": "result", "большой": "big", "квадрат": "square", "отличается": "differs"}
WORDS = {"tetромino": "tetromino", "Hмм": "Hmm", "hмм": "hmm"}     # mixed-script words seen in the data (exact)
CYR = re.compile(r"[Ѐ-ӿ]+")


def latin_plain(s):
    out = []
    for ch in s:
        if ord(ch) < 128:
            out.append(ch)
        elif ch in LATIN_EXTRA:
            out.append(LATIN_EXTRA[ch])
        else:
            base = "".join(c for c in unicodedata.normalize("NFKD", ch) if not unicodedata.combining(c))
            out.append(base)
    return "".join(out)


def replace(text, a, b):
    """(new_a, new_b, replacement) for the OOV span text[a:b], or None (= drop the trace)."""
    s = text[a:b]
    if all(ord(c) < 128 or c in PUNCT or c in CHECKS for c in s):
        return a, b, "".join(PUNCT.get(c, CHECKS.get(c, c)) for c in s)
    cat = orep.category(s)
    if cat == "latin_diacritic":
        core = s.strip()
        if core in ("Ø", "ø"):                         # standalone empty-cell marker
            return a, b, s.replace(core, "-")
        new = latin_plain(s)
        return (a, b, new) if all(ord(c) < 128 for c in new) else None
    if cat == "symbol_emoji":
        new = "".join(CHECKS.get(c, PUNCT.get(c, c)) for c in s)
        return (a, b, new) if all(ord(c) < 128 for c in new) else None
    if cat == "cyrillic":
        while a < b and text[a].isspace():         # the token's leading space is not part of the word
            a += 1
        # whole word around the span
        wa, wb = a, b
        while wa > 0 and (text[wa - 1].isalpha()):
            wa -= 1
        while wb < len(text) and text[wb].isalpha():
            wb += 1
        word = text[wa:wb]
        if word in WORDS:
            return wa, wb, WORDS[word]
        if CYR.fullmatch(word) and word.lower() in RUSSIAN:
            return wa, wb, RUSSIAN[word.lower()]
        if not CYR.fullmatch(word) and all(ord(c) < 128 or c in HOMOGLYPH for c in word):   # look-alikes in a Latin word
            return wa, wb, "".join(HOMOGLYPH.get(c, c) for c in word)
        return None
    if cat in ("cjk", "other"):
        new = "".join(PUNCT.get(c, c) for c in unicodedata.normalize("NFKC", s))
        return (a, b, new) if all(ord(c) < 128 for c in new) else None
    return None


def oov_spans(text):
    enc = orep.TOK(text, add_special_tokens=False, return_offsets_mapping=True)
    return [(a, b) for tid, (a, b) in zip(enc["input_ids"], enc["offset_mapping"]) if tid not in orep.KEEP]


def one(line):
    t = json.loads(line)
    text = t["reasoning"]
    spans = oov_spans(text)
    if not spans:
        return "clean", line, None, []
    reps, bad = [], []
    for a, b in spans:
        r = replace(text, a, b)
        (bad.append(text[a:b]) if r is None else reps.append(r))
    if bad:
        return "regen", None, {"trace_id": t["trace_id"], "bench": t.get("bench"), "unit": t.get("unit"),
                               "task_id": t.get("task_id"), "cohort": t.get("cohort"),
                               "pieces": sorted(set(text[a:b] for a, b in spans)), "blocking": sorted(set(bad)),
                               "cyrillic_words": sorted(set(CYR.findall(text)))}, []
    reps = sorted(set(reps), reverse=True)
    new, edits, last = text, [], None
    for a, b, rep in reps:                      # back to front, overlapping spans (word expansion) merged
        if last is not None and b > last:
            continue
        edits.append((t["trace_id"], text[max(0, a - 40):b + 30], text[max(0, a - 40):a] + rep + text[b:b + 30]))
        new = new[:a] + rep + new[b:]
        last = a
    if oov_spans(new) or "<think>" in new or "</think>" in new:
        return "regen", None, {"trace_id": t["trace_id"], "bench": t.get("bench"), "unit": t.get("unit"),
                               "task_id": t.get("task_id"), "cohort": t.get("cohort"),
                               "pieces": ["(OOV left after replacement)"]}, []
    t["reasoning"] = new
    t["oov_edits"] = len(edits)
    return "edited", json.dumps(t, ensure_ascii=False) + "\n", None, edits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cot", required=True)
    ap.add_argument("--orig", required=True)
    ap.add_argument("--keep", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--regen", required=True)
    ap.add_argument("--edits", required=True)
    ap.add_argument("--workers", type=int, default=64)
    a = ap.parse_args()
    st = collections.Counter()
    pieces = collections.Counter()
    with Pool(a.workers, initializer=orep.init, initargs=(a.orig, a.keep)) as pool, \
            open(a.out + ".part", "w", encoding="utf-8") as fo, open(a.regen, "w", encoding="utf-8") as fr, \
            open(a.edits, "w", encoding="utf-8") as fe:
        fe.write("trace_id\tbefore\tafter\n")
        for status, line, regen, edits in pool.imap(one, open(a.cot, encoding="utf-8"), chunksize=8):
            st[status] += 1
            if line:
                fo.write(line)
            if regen:
                fr.write(json.dumps(regen, ensure_ascii=False) + "\n")
                pieces.update(regen.get("blocking", regen["pieces"]))
            for tid, before, after in edits:
                st["edits"] += 1
                fe.write(f"{tid}\t{json.dumps(before, ensure_ascii=False)}\t{json.dumps(after, ensure_ascii=False)}\n")
    os.replace(a.out + ".part", a.out)
    print(json.dumps(dict(st)))
    print("blocking pieces:", ", ".join(f"{p!r}x{n}" for p, n in pieces.most_common(30)))


if __name__ == "__main__":
    main()
