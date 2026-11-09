"""Build a packing_v2 SFT cache in the stage-1 format (DESIGN.md C5).

    python -m pv2.build_dataset --size 20000 --seed 1 --output out/pv2_20k.json

The payload is ``{"metadata": {...}, "examples": [{"messages": [user, assistant], "length", "prov"}]}``,
the format ``stage2/build_stage2_cache.py --aux-cache`` reads.  Every row is deterministic in
(config, row seed): ``prov = "pv2|<version>|<kind>|<row seed>"`` re-renders it with ``build_row``.

Row mix: a ``classic_share`` of rows are the old formats, byte-identical to the old builder (single
items, same-task bundles and mixed bundles with the long_tail_32k size distribution); the rest are v2
rows of 1-12 parts whose families are drawn independently, so long rows mix families (USER).
``length`` is the real chat-templated token count; rows are trimmed part by part to fit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from pv2.paths import ensure_generator_package  # noqa: E402

GENERATOR_ROOT = ensure_generator_package()

from pv2 import classic as C  # noqa: E402
from pv2 import families as F  # noqa: E402
from pv2 import items as I  # noqa: E402
from pv2 import library as L  # noqa: E402
from pv2.assemble import AssembleConfig, AssemblyError, assemble_row  # noqa: E402
from pv2.library.parts import _conflicts  # noqa: E402
from pv2.model import FLAG_NOTHING_ELSE  # noqa: E402
from pv2.tokens import load_tokenizer, rough_len, templated_len  # noqa: E402

VERSION = "1"
FAMILIES = ("reframed", "gallery", "multi_question", "arc", "comparison", "chain")

# the old builder's long_tail_32k bundle-size distribution (sizes 1..48)
LONG_TAIL_32K = [0.55 / 5] * 5 + [0.25 / 7] * 7 + [0.13 / 12] * 12 + [0.07 / 24] * 24


@dataclass
class BuildConfig:
    classic_share: float = 0.30
    classic_mixed_rate: float = 0.8
    family_weights: dict = field(default_factory=lambda: {
        "reframed": 0.12, "gallery": 0.15, "multi_question": 0.10, "arc": 0.22, "comparison": 0.10,
        "chain": 0.05})
    parts_per_row: dict = field(default_factory=lambda: {
        1: 0.50, 2: 0.12, 3: 0.10, 4: 0.08, 5: 0.06, 6: 0.05, 8: 0.05, 10: 0.02, 12: 0.02})
    mix_encodings: bool = True            # USER: toggle; False -> every grid of a v2 row is compact
    companion: bool = True                # a yes/no-only row gets one reading question (add_companion)
    max_seq_length: int = 32768
    assemble: dict = field(default_factory=dict)   # AssembleConfig overrides
    # ARC options (pv2.families.arc_part): "modes" (subset of pv2.arc.bundle.MODES), "questions"
    # ("all" | "reading": no question that uses the rule), "aux_tasks" (task ids whose rule may build a
    # bundle), "families" (transform / scalar families).  Empty = today's behaviour.
    arc: dict = field(default_factory=dict)

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, default=str).encode()).hexdigest()[:12]


def _choice(rng, weights: dict):
    keys = list(weights)
    p = np.asarray([weights[k] for k in keys], dtype=float)
    return keys[int(rng.choice(len(keys), p=p / p.sum()))]


# --- classic rows (the old formats) ---------------------------------------------------------------------

def _variant(rng, tid: str):
    n = len(C._pv.variants_for(tid, C._store()))
    return C.variant_template(tid, int(rng.integers(0, n + 1)))


def classic_tasks() -> list[str]:
    """Every registered task in 1-102, 14 / 41 / 78 included (the old formats handle them)."""
    import re
    from registry import list_tasks
    return [t for t in list_tasks() if int(re.match(r"\d+", t).group(0)) <= 102]


def classic_row(rng, cfg: BuildConfig, fits) -> Optional[tuple[str, str, dict]]:
    tasks = classic_tasks()
    size = int(rng.choice(np.arange(1, len(LONG_TAIL_32K) + 1), p=np.asarray(LONG_TAIL_32K) / sum(LONG_TAIL_32K)))
    if size >= 2 and rng.random() < cfg.classic_mixed_rate:
        chosen = [str(t) for t in rng.choice(tasks, size=min(size, len(tasks)), replace=False)]
        items = [{"task_id": t, "seed": int(rng.integers(0, 2 ** 31))} for t in chosen]
        variants = [_variant(rng, t) for t in chosen]
        while len(items) >= 1:
            prompt, answer = C.classic_mixed(items, item_prompt_variants=variants)
            if fits(prompt, answer):
                return prompt, answer, {"kind": "classic_mixed", "n": len(items)}
            items, variants = items[:-1], variants[:-1]
        return None
    tid = str(tasks[int(rng.integers(len(tasks)))])
    seeds = [int(rng.integers(0, 2 ** 31)) for _ in range(min(size, 5))]
    variant = _variant(rng, tid)
    while seeds:
        prompt, answer = C.classic_bundle(tid, seeds, prompt_variant=variant)
        if fits(prompt, answer):
            return prompt, answer, {"kind": "classic_bundle", "n": len(seeds)}
        seeds = seeds[:-1]
    return None


# --- v2 rows ---------------------------------------------------------------------------------------------

def build_part(rng, family: str, *, multi: bool):
    # natural generator sizes (28% of grids have 400+ cells, like the classic rows); the training
    # side's audit found the earlier per-grid caps left large grids at 4% of v2 grids
    if family == "reframed":
        return F.reframed_part(rng, multi_answer=multi, max_cells=900)
    if family == "gallery":
        n = int(rng.choice(np.arange(2, 13), p=_gallery_p()))
        return F.gallery_part(rng, n_grids=n, n_asked=int(rng.integers(1, n + 1)))
    if family == "multi_question":
        return F.one_grid_part(rng, n_questions=int(rng.integers(2, 7)))
    if family == "comparison":
        return F.comparison_part(rng)
    if family == "chain":
        return F.chain_part(rng)
    if family == "arc":
        return F.arc_part(rng, **ARC_OPTIONS)
    raise ValueError(family)


def _gallery_p():
    w = np.asarray([0.20, 0.18, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04, 0.03, 0.02, 0.02])   # 2..12 grids
    return w / w.sum()


def _compact_only(parts):
    for p in parts:
        for s in p.slots:
            s.encoding = "compact"
    return parts


_VERIFY_COUNTS = frozenset({"lib:verify_diff", "arc:verify_diff"})
COMPANION_FAMILIES = ("comparison", "reframed", "gallery", "multi_question")


def _yes_no(q) -> bool:
    """A yes/no question: a bool, or an item whose task answers yes / no or true / false (a token; both answers
    qualify, so the test never depends on which one it is), or a verification's cell count."""
    return q.answer.kind == "bool" or q.qid in _VERIFY_COUNTS or (
        q.answer.kind == "token" and str(q.answer.text).strip().lower() in ("yes", "no", "true", "false"))


def _yes_no_only(parts) -> bool:
    """Every question of the row is a yes/no question (question kinds, not answer values: a count
    that happens to be 0 or 1 does not qualify, or a companion would tell its value)."""
    return all(_yes_no(q) for p in parts for q in p.queries)


def _companion_slots(part) -> list[int]:
    """Grids of ``part`` a reading question may be asked about: plain grids (not a proposed or
    candidate grid) that no literal item reads (a reading question could hand its answer over)."""
    literal = {i for q in part.queries if q.source == "item" and q.qid.split(":")[1] in F.LITERAL_ITEM_TASKS
               for i in q.reads}
    return [i for i, s in enumerate(part.slots) if s.role == "grid" and i not in literal]


def add_companion(rng, parts) -> bool:
    """A row whose questions are all yes/no (parity review 2026-09-30: 3.2% of new rows answered only
    yes / no / 0 / 1, one or two tokens of supervision) gets one reading question -- a cell, a line or
    a run -- about a grid of its first part that can take one.  The trigger looks at question kinds
    only, so the companion's presence says nothing about the answers, and no yes/no answer implies
    what a cell or a line of a grid holds."""
    if not _yes_no_only(parts):
        return False
    exclude = F._lib_exclude_except(F.READING_QIDS)
    for part in parts:
        if part.family not in COMPANION_FAMILIES or part.single_answer or part.labels_required \
                or any(FLAG_NOTHING_ELSE in q.flags for q in part.queries):
            continue
        slots = _companion_slots(part)
        for _ in range(6):
            if not slots:
                break
            q = L.pose(rng, part.slots, (int(rng.choice(slots)),), exclude=exclude | {q.qid for q in part.queries})
            if q is not None and not _conflicts(q, part.queries):
                part.queries.append(q)
                part.prov["companion"] = q.qid
                return True
    return False


def v2_row(rng, cfg: BuildConfig, fits, acfg: AssembleConfig) -> Optional[tuple[str, str, dict]]:
    n_parts = int(_choice(rng, cfg.parts_per_row))
    weights = dict(cfg.family_weights)
    parts = []
    for _ in range(n_parts):
        for _attempt in range(6):
            family = _choice(rng, weights)
            part = build_part(rng, family, multi=n_parts > 1)
            if part is None:                       # e.g. no ARC producer in this tree
                weights.pop(family, None)
                continue
            if n_parts > 1 and (part.single_answer or any("nothing_else" in q.flags for q in part.queries)):
                continue
            parts.append(part)
            break
    if not parts:
        return None
    if cfg.companion:
        add_companion(rng, parts)
    if not cfg.mix_encodings:
        parts = _compact_only(parts)
    while parts:
        try:
            row = assemble_row(rng, parts, config=acfg)
        except AssemblyError:
            return None
        if fits(row.prompt, row.target):
            meta = {"kind": "v2", "n_parts": len(parts), "families": [p.family for p in parts],
                    "n_questions": sum(len(p.queries) for p in parts), "assembled": row.meta}
            return row.prompt, row.target, meta
        parts = parts[:-1]
    return None


ARC_OPTIONS: dict = {}     # set from BuildConfig.arc by build_row (build_part has no cfg argument)


def build_row(seed: int, cfg: BuildConfig, tokenizer, acfg: Optional[AssembleConfig] = None):
    """(prompt, target, meta) for one row seed, or None when the draw cannot be packed."""
    acfg = acfg or AssembleConfig(**cfg.assemble)
    ARC_OPTIONS.clear()
    ARC_OPTIONS.update(cfg.arc)
    rng = np.random.default_rng([int(seed), 0x70763])

    def fits(prompt, answer):
        if rough_len(prompt, answer) < cfg.max_seq_length * 0.6:
            return True
        return templated_len(tokenizer, prompt, answer) <= cfg.max_seq_length

    if rng.random() < cfg.classic_share:
        return classic_row(rng, cfg, fits)
    return v2_row(rng, cfg, fits, acfg)


def build(n_rows: int, seed: int, cfg: BuildConfig, tokenizer, log_every: int = 1000):
    examples, stats, failures = [], Counter(), Counter()
    t0 = time.time()
    s = 0
    while len(examples) < n_rows:
        row_seed = seed * 10_000_000 + s
        s += 1
        try:
            built = build_row(row_seed, cfg, tokenizer)
        except Exception as e:  # noqa: BLE001  (a producer or the validator refused this draw)
            failures[type(e).__name__] += 1
            if sum(failures.values()) > max(100, n_rows):
                raise
            continue
        if built is None:
            failures["unpackable"] += 1
            continue
        prompt, answer, meta = built
        length = templated_len(tokenizer, prompt, answer)
        if length > cfg.max_seq_length:
            failures["over_length"] += 1
            continue
        kind = meta["kind"]
        stats[kind] += 1
        for fam in meta.get("families", []):
            stats["family:" + fam] += 1
        stats["tokens_" + kind] += length
        stats["answer_tokens_" + kind] += len(tokenizer.encode(answer, add_special_tokens=False))
        examples.append({"messages": [{"role": "user", "content": prompt},
                                      {"role": "assistant", "content": answer}],
                         "length": length, "prov": f"pv2|{VERSION}|{kind}|{row_seed}"})
        if log_every and len(examples) % log_every == 0:
            el = time.time() - t0
            print(f"  {len(examples)}/{n_rows} rows ({el:.0f}s, {1000 * el / len(examples):.0f} ms/row)", flush=True)
    return examples, stats, failures


def _git_commit(root: str, paths=()) -> str:
    import subprocess
    try:
        commit = subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, timeout=10).stdout.strip()
        if commit:
            status = subprocess.run(["git", "-C", root, "status", "--porcelain", "--untracked-files=no", "--",
                                     *paths], capture_output=True, text=True, timeout=10).stdout
            if [ln for ln in status.splitlines() if ln.strip() and "__pycache__" not in ln and not ln.endswith(".pyc")]:
                commit += "-dirty"
        return commit
    except Exception:  # noqa: BLE001  (git absent or not a checkout)
        return ""


def generator_provenance() -> dict:
    """The old builder's ``sft_dataset.generator_provenance`` (same files, same digest), so the stage-2
    blender and the trainer can compare a v2 cache with the tree they run."""
    import registry as _registry
    root = os.path.dirname(os.path.realpath(_registry.__file__))
    commit = _git_commit(root, ("registry.py", "prompts.py", "prompt_variants.py", "prompt_variants.json",
                                "generators", "common", "prompt_modules"))
    files = [os.path.join(root, n) for n in ("registry.py", "prompts.py", "prompt_variants.py", "prompt_variants.json")
             if os.path.exists(os.path.join(root, n))]
    for sub in ("generators", "common", "prompt_modules"):
        d = os.path.join(root, sub)
        if os.path.isdir(d):
            for dirpath, _dirs, fnames in os.walk(d):
                files.extend(os.path.join(dirpath, f) for f in fnames if f.endswith(".py"))
    h = hashlib.sha256()
    for path in sorted(files):
        h.update(os.path.relpath(path, root).encode("utf-8"))
        with open(path, "rb") as f:
            h.update(f.read())
    return {"generator_commit": commit, "generator_digest": h.hexdigest()[:16], "generator_root": root,
            "packing_commit": _git_commit(HERE, ("pv2",))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-seq-length", type=int, default=32768)
    ap.add_argument("--classic-share", type=float, default=0.30)
    ap.add_argument("--no-mix-encodings", action="store_true", help="every v2 grid compact")
    ap.add_argument("--tokenizer", default=None, help="path or hub id (default: $PV2_TOKENIZER, else Qwen/Qwen3.5-4B)")
    ap.add_argument("--exclude-families", default="",
                    help=f"comma-separated v2 families to leave out ({', '.join(FAMILIES)})")
    ap.add_argument("--arc-modes", default="", help="comma-separated ARC modes to draw from "
                    "(transform, composition, scalar, stated, hidden); default all")
    ap.add_argument("--arc-questions", default="all", choices=("all", "reading"),
                    help="reading: ARC-shaped bundles with no question that uses the rule "
                         "(no predict, no identify-the-rule menu, no verification)")
    ap.add_argument("--arc-tasks", default="", help="comma-separated aux task ids allowed as ARC rules")
    ap.add_argument("--arc-families", default="", help="comma-separated transform / scalar families allowed")
    ap.add_argument("--grid-answer-encoding", default="", help="write every grid ANSWER in this one encoding "
                    "(e.g. compact); JSON is then not used where a part has a grid answer. Answer styles "
                    "(sentences, paragraphs, labels ...) stay on")
    args = ap.parse_args(argv)
    cfg = BuildConfig(classic_share=args.classic_share, max_seq_length=args.max_seq_length,
                      mix_encodings=not args.no_mix_encodings)
    excluded = [f.strip() for f in args.exclude_families.split(",") if f.strip()]
    unknown = sorted(set(excluded) - set(FAMILIES))
    if unknown:
        ap.error(f"unknown families {unknown}; choose from {FAMILIES}")
    cfg.family_weights = {f: w for f, w in cfg.family_weights.items() if f not in excluded}
    if not cfg.family_weights and cfg.classic_share < 1:
        ap.error("every v2 family is excluded; use --classic-share 1 for classic rows only")
    listed = lambda text: [x.strip() for x in text.split(",") if x.strip()]  # noqa: E731
    if listed(args.arc_modes):
        cfg.arc["modes"] = listed(args.arc_modes)
    if args.arc_questions != "all":
        cfg.arc["questions"] = args.arc_questions
    if listed(args.arc_tasks):
        cfg.arc["aux_tasks"] = listed(args.arc_tasks)
    if listed(args.arc_families):
        cfg.arc["families"] = listed(args.arc_families)
    if args.grid_answer_encoding:
        from pv2 import encodings as _enc
        if args.grid_answer_encoding not in _enc.ENCODINGS:
            ap.error(f"unknown encoding {args.grid_answer_encoding!r}")
        cfg.assemble["single_grid_encoding"] = args.grid_answer_encoding
    tok = load_tokenizer(args.tokenizer)
    t0 = time.time()
    examples, stats, failures = build(args.size, args.seed, cfg, tok)
    lengths = sorted(e["length"] for e in examples)
    meta = {"builder": "packing_v2", "version": VERSION, "config": asdict(cfg), "config_digest": cfg.digest(),
            "seed": args.seed, "max_seq_length": args.max_seq_length, "grid_format": "mixed",
            "symbol_remap": False, **generator_provenance(),
            "stats": dict(stats), "failures": dict(failures),
            "length_p50": lengths[len(lengths) // 2], "length_max": lengths[-1],
            "tokens_total": sum(lengths), "seconds": round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    tmp = args.output + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"metadata": meta, "examples": examples}, f, ensure_ascii=False)
    os.replace(tmp, args.output)
    print(json.dumps({k: meta[k] for k in ("stats", "failures", "length_p50", "length_max", "tokens_total", "seconds")}))


if __name__ == "__main__":
    main()
