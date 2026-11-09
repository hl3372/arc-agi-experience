"""Checks one ARC part from scratch (tests only; run in worker processes by test_arc_bundle).

Everything is re-derived from the part's slots and provenance with code that does not go through
the builder: the rule is rebuilt from ``prov['steps']`` / ``prov['scalar']`` and re-applied, aux
examples are regenerated from the frozen generator with the recorded seeds and colour maps, the
predict answers are re-checked against the whole rival set with ``_arc_naive`` (plain nested
loops), siblings are enumerated here, and every reading question is recomputed from the grids.
"""
from __future__ import annotations

import re
import time

import numpy as np

import _arc_naive as N
import pv2.pictures  # noqa: F401
from pv2.arc import aux_rules
from pv2.arc.bundle import build
from pv2.arc.grids import Info, arr, neighbour_count_same
from pv2.arc.scalars import SCALARS, ScalarRule, all_scalar_rules
from pv2.arc.transforms import FAMILIES, Rule
from pv2.encodings import ENCODINGS, check_grid, render
from registry import generate

DIMS_REVEALING = {"stage2", "runs", "row_labelled", "indexed_table", "sparse"}
D4 = [("rot", (1,)), ("rot", (2,)), ("rot", (3,)), ("flip", ("lr",)), ("flip", ("ud",)),
      ("transpose", ("main",)), ("transpose", ("anti",))]
GENERIC = [Rule((s,)) for s in D4] + [Rule.single("gravity", d) for d in ("down", "up", "left", "right")] + \
    [Rule.single("crop_content"), Rule.single("compress", "both")]
_PH = re.compile(r"⟪([gG])(\d+)(s?)⟫")


def same(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and bool(np.array_equal(a, b))


def my_siblings(rule: Rule) -> list[Rule]:
    out = []
    for i, (fam, p) in enumerate(rule.steps):
        alts = []
        if (fam, p) in D4:
            alts += [s for s in D4 if s != (fam, p)]
        for q in FAMILIES[fam].instances:
            if len(q) == len(p) and sum(1 for a, b in zip(p, q) if a != b) == 1:
                alts.append((fam, q))
        for s in alts:
            steps = list(rule.steps)
            steps[i] = s
            out.append(Rule(tuple(steps)))
    return out


def plain(a, lone_ok=None) -> bool:
    info = Info(arr(a))
    bg = info.bg
    if bg is None or info.hist[bg] < 0.5 * info.a.size:
        return False
    lone = (info.a != bg) & (neighbour_count_same(info.a, 4) == 0)
    if lone_ok is True:
        return True
    if lone_ok is None:
        return not lone.any()
    return bool(np.all(info.a[lone] == lone_ok))


def _build(seed, mode, family, opts=None):
    if family is None:
        return build(seed, mode, **(opts or {}))
    from pv2.arc import gen
    orig = gen.choose_spec
    gen.choose_spec = lambda r, pool=None: gen.GEN_BY_FAM[family] if pool is None else orig(r, pool)
    try:
        return build(seed, "transform")
    finally:
        gen.choose_spec = orig


def check_seed(seed: int, mode=None, determinism: bool = False, family=None, opts=None) -> dict:
    """``family`` forces a transform-mode part of that library family; ``opts`` are passed to
    ``build`` (questions / families / aux_tasks)."""
    t0 = time.perf_counter()
    part = _build(seed, mode, family, opts)
    dt = time.perf_counter() - t0
    errs: list[str] = []
    try:
        stats = _check(part, errs)
    except Exception as e:  # noqa: BLE001  (report, do not crash the pool)
        errs.append(f"exception {type(e).__name__}: {e}")
        stats = {}
    if determinism:
        again = _build(seed, mode, family, opts)
        if repr(again) != repr(part):
            errs.append("not deterministic")
    # stage-2 token estimate of the part, computed here from the rendering (0.9 tokens per character
    # bounds the measured ratio; 14 per header; grid answers in compact digits; 300 for the texts)
    s2 = 300 + sum(int(0.9 * len(render(s.grid, "stage2"))) + 14 for s in part.slots)
    s2 += sum(len(q.answer.value) * (len(q.answer.value[0]) + 1) for q in part.queries if q.answer.kind == "grid")
    stats.update({"seed": seed, "ms": dt * 1000, "mode": part.prov.get("mode"), "errors": errs,
                  "s2tokens": s2, "yn": [(q.qid, bool(q.answer.value)) for q in part.queries if q.answer.kind == "bool"],
                  "answer_cells": [len(q.answer.value) * len(q.answer.value[0]) for q in part.queries
                                   if q.answer.kind == "grid"],
                  "roles": [s.role for s in part.slots], "context": list(part.context),
                  "rule": part.prov.get("rule"), "steps": part.prov.get("steps"),
                  "scalar": part.prov.get("scalar"), "task": part.prov.get("task")})
    return stats


def _check(part, errs) -> dict:
    prov = part.prov
    mode = prov["mode"]
    part.validate_shape()
    if part.family != "arc" or not part.labels_required:
        errs.append("family/labels")
    encs = {s.encoding for s in part.slots}
    enc = next(iter(encs))
    if len(encs) != 1 or enc not in ENCODINGS or ENCODINGS[enc].uses_names or enc != prov["encoding"]:
        errs.append(f"encoding {encs}")
    slots = part.slots
    for s in slots:
        check_grid(s.grid)
        if len(s.grid) > 30 or len(s.grid[0]) > 30:
            errs.append("size > 30")
    tin = sorted((s.pair, i) for i, s in enumerate(slots) if s.role == "train_input")
    tout = sorted((s.pair, i) for i, s in enumerate(slots) if s.role == "train_output")
    tst = sorted((s.pair, i) for i, s in enumerate(slots) if s.role == "test_input")
    props = [(s.pair, i) for i, s in enumerate(slots) if s.role == "proposed"]
    n_train, n_test = len(tin), len(tst)
    if [p for p, _ in tin] != list(range(n_train)) or [p for p, _ in tst] != list(range(n_test)):
        errs.append("pair indices")
    if not 2 <= n_train <= 5 or not 1 <= n_test <= 2 or n_train != prov["n_train"] or n_test != prov["n_test"]:
        errs.append(f"counts {n_train} {n_test}")
    scalar = mode == "scalar"
    if scalar != (len(tout) == 0) or (not scalar and [p for p, _ in tout] != list(range(n_train))):
        errs.append("train outputs")
    if len({p for p, _ in props}) != len(props) or any(p >= n_test for p, _ in props):
        errs.append("proposed slots")
    X = [arr(slots[i].grid) for _, i in tin]
    Y = [arr(slots[i].grid) for _, i in tout]
    TS = [arr(slots[i].grid) for _, i in tst]
    stats = {"n_train": n_train, "n_test": n_test, "encoding": enc,
             "cells": [np.asarray(s.grid).size for s in slots], "sides": [np.asarray(s.grid).shape for s in slots]}

    truth = None           # per test: the correct output (grid) or value
    rule = None
    srule = None
    if mode in ("transform", "composition", "hidden"):
        rule = Rule(tuple((f, tuple(p)) for f, p in prov["steps"]))
        stats["family"] = "+".join(f for f, _ in rule.steps)
        for x, y in zip(X, Y):
            if not same(rule.apply(x), y):
                errs.append("rule does not give a train output")
        truth = [arr(rule.apply(t)) for t in TS]
        _binding(rule, X, Y, TS, truth, strict=(mode == "hidden"), errs=errs)
        if len(rule.steps) == 1 and rule.steps[0][0] in ("extract_row", "extract_col"):
            for t, a in zip(TS, truth):
                c = np.bincount(t.ravel(), minlength=10)
                top = np.nonzero(c == c.max())[0]
                blank = len(np.unique(a)) < 2 if len(top) > 1 else not bool((a != top[0]).any())
                if blank:
                    errs.append("extracted test line is blank")
        preds, amb, toks = N.predictions(list(zip(X, Y)), TS)
        for j, t in enumerate(truth):
            if amb[j] or preds[j] != {N.key(t)}:
                errs.append(f"naive rival check: test {j} has {len(preds[j])} predictions, ambiguous={amb[j]}")
        if rule.is_composition and any(isinstance(tk[0], str) for tk in toks):
            errs.append("a single rule explains the composition")
        stats["naive_rivals"] = len(toks)
    if mode in ("stated", "hidden"):
        tid = prov["task"]
        stats["family"] = f"aux:{tid}"
        exs = []
        for sd, sigma in zip(prov["seeds"], prov["sigmas"]):
            ex = generate(tid, sd)
            lut = np.array(sigma, dtype=np.int8)
            if sorted(sigma) != list(range(10)):
                errs.append("sigma is not a permutation")
            exs.append((lut[arr(ex["input"])], lut[arr(ex["output"])], ex["metadata"], sigma))
        if len(exs) != n_train + n_test:
            errs.append("aux example count")
        for k, (a, b, meta, sigma) in enumerate(exs[:n_train]):
            if not same(a, X[k]) or not same(b, Y[k]):
                errs.append("aux train pair differs from the regenerated example")
        for j, (a, b, meta, sigma) in enumerate(exs[n_train:]):
            if not same(a, TS[j]):
                errs.append("aux test input differs from the regenerated example")
        tex = [b for _, b, _, _ in exs[n_train:]]
        if truth is not None and not all(same(a, b) for a, b in zip(truth, tex)):
            errs.append("library rule and generator disagree on a test")
        truth = tex
        canon = aux_rules.render_under(tid, exs[0][2], exs[0][3])
        for a, b, meta, sigma in exs:
            if aux_rules.render_under(tid, meta, sigma) != canon:
                errs.append("aux examples do not share one rule")
        for x, y in zip(X, Y):
            if same(x, y):
                errs.append("identity pair")
        for t, a in zip(TS, truth):
            if same(t, a):
                errs.append("test output equals test input")
        if mode == "stated":
            text, template = prov["rule_text"], prov["wording"]
            if not aux_rules.wording_ok(text) or "?" in text or not any(text in c for c in part.context):
                errs.append("stated rule text missing, not allowed or a question")
            frame = " ".join(part.context)
            if ("the test input follows" in frame and n_test != 1) or \
                    ("the test inputs follow" in frame and n_test != 2):
                errs.append("stated frame number of test inputs")
            for a_, b_, meta, sigma in exs:
                if aux_rules.render_under(tid, meta, sigma, template) != text:
                    errs.append("stated wording does not hold for every example")
        if mode == "hidden":
            for f in aux_rules.SIBLING_FLAGS.get(tid, ()):
                if not any(bool(e[2].get(f)) for e in exs[:n_train]):
                    errs.append(f"flag {f} never set")
    if scalar:
        name, params = prov["scalar"]
        srule = ScalarRule(name, tuple(params))
        stats["family"] = f"scalar:{name}"
        vals = _context_values(part, [i for _, i in tin], errs)
        fmt = {"number": "is a number", "color": "written as its number", "size": "written rows x columns"}
        if not any(fmt[SCALARS[name].kind] in c for c in part.context):
            errs.append("scalar value format not stated")
        for x, v in zip(X, vals):
            if str(srule.value(x)) != v:
                errs.append("scalar train value wrong")
        if len(set(vals)) != len(vals) or n_train < 3:
            errs.append("scalar values repeat / too few pairs")
        truth = [srule.value(t) for t in TS]
        _scalar_rivals(X, [srule.value(x) for x in X], TS, truth, errs)
        if SCALARS[name].reveals_dims and (enc in DIMS_REVEALING or any(s.dims_header_ok for s in slots)):
            errs.append("size-valued scalar bundle shows sizes")
        if SCALARS[name].reveals_dims and any(q.qid in ("arc:dims",) + GRID_READS for q in part.queries):
            errs.append("size-valued scalar bundle asks a size or a window (it shows a width or height)")

    _queries(part, X, Y, TS, truth, rule, srule, errs, stats)
    return stats


def _binding(rule, X, Y, TS, truth, *, strict, errs):
    n = len(X)
    mp = 4 if rule.is_composition else (2 if rule.geometric else 3)
    if n < mp:
        errs.append(f"too few pairs for {rule.token}")
    for x, y in zip(X, Y):
        if same(x, y):
            errs.append("identity pair")
    if all(same(Y[0], y) for y in Y[1:]):
        errs.append("constant outputs")
    for t, a in zip(TS, truth):
        if same(t, a):
            errs.append("test output equals test input")
    mono = [len(np.unique(g)) < 2 for g in Y + truth]
    if rule.mono_ok is False and any(mono):
        errs.append("single-colour output")
    if rule.mono_ok == "some" and all(mono[:n]):
        errs.append("every train output single-colour")
    cin = set().union(*[set(np.unique(x).tolist()) for x in X])
    cout = set().union(*[set(np.unique(y).tolist()) for y in Y])
    for t, a in zip(TS, truth):
        if not set(np.unique(t).tolist()) <= cin or not set(np.unique(a).tolist()) <= cout:
            errs.append("test colour not witnessed")
    if rule.bg:
        first = rule.steps[0][0]
        lone = True if first == "remove_isolated" else (rule.steps[0][1][0] if first in ("rays", "connect") else None)
        if not all(plain(g, lone) for g in X + TS):
            errs.append("background rule on a grid without a plain ground")
    sibs = my_siblings(rule)
    pair_hit = [False] * n
    for s in sibs:
        hit = False
        for i, (x, y) in enumerate(zip(X, Y)):
            o = s.apply(x)
            if o is None or not same(o, y):
                hit = True
                pair_hit[i] = True
        if not hit:
            if strict:
                errs.append(f"sibling {s.token} fits every pair")
            for t, a in zip(TS, truth):
                o = s.apply(t)
                if o is not None and not same(o, a):
                    errs.append(f"sibling {s.token} fits the pairs and differs on a test")
    for i, (x, y) in enumerate(zip(X, Y)):
        if not pair_hit[i]:
            if not any((lambda o: o is None or not same(o, y))(g.apply(x)) for g in GENERIC if g.token != rule.token):
                errs.append(f"pair {i} contradicts no sibling or generic rule")


def _context_values(part, tin_slots, errs):
    vals = {}
    for line in part.context:
        m = re.fullmatch(r"⟪G(\d+)⟫ -> (\S+)", line)
        if m:
            vals[int(m.group(1))] = m.group(2)
    if sorted(vals) != sorted(tin_slots):
        errs.append("scalar context lines do not name every train input")
    return [vals.get(s) for s in tin_slots]


def _scalar_rivals(X, vals, TS, truth, errs):
    def norm(v):
        return v if isinstance(v, str) or v is None else int(v)
    for r in all_scalar_rules():
        if all(norm(r.value(x)) == norm(v) for x, v in zip(X, vals)):
            for t, want in zip(TS, truth):
                got = norm(r.value(t))
                if got is not None and got != norm(want):
                    errs.append(f"scalar rival {r.token} disagrees on a test")


def _queries(part, X, Y, TS, truth, rule, srule, errs, stats):
    slots = part.slots
    grid = lambda i: arr(slots[i].grid)  # noqa: E731
    test_of = {i: s.pair for i, s in enumerate(slots) if s.role == "test_input"}
    qids = [q.qid for q in part.queries]
    stats["qids"] = qids
    stats["predict_only"] = qids in (["arc:predict"], ["arc:predict_value"])
    predicted = set()
    verified = set()
    for q in part.queries:
        if q.answer.kind == "color" and q.statement is not None and "color ⟪answer⟫" not in q.statement:
            errs.append(f"{q.qid}: colour statement without 'color'")
        if q.qid == "arc:verify_value" and part.prov.get("scalar") and \
                SCALARS[part.prov["scalar"][0]].kind == "color" and "is color " not in q.statement \
                and "is not color " not in q.statement:
            errs.append("verify_value: colour statement without 'color'")
        for m in _PH.finditer(q.text):
            if int(m.group(2)) >= len(q.reads):
                errs.append(f"{q.qid}: placeholder beyond reads")
        if q.answer.text is not None:
            for m in _PH.finditer(q.answer.text):
                if int(m.group(2)) >= len(q.reads):
                    errs.append(f"{q.qid}: answer placeholder beyond reads")
        a = q.answer
        if q.qid == "arc:predict":
            j = test_of[q.reads[0]]
            predicted.add(j)
            if not same(a.value, truth[j]):
                errs.append("predict answer wrong")
        elif q.qid == "arc:predict_value":
            j = test_of[q.reads[0]]
            predicted.add(j)
            if a.text != str(truth[j]):
                errs.append("scalar predict answer wrong")
        elif q.qid == "arc:verify":
            j = test_of[q.reads[1]]
            verified.add(j)
            if slots[q.reads[0]].role != "proposed" or slots[q.reads[0]].pair != j:
                errs.append("verify reads")
            if a.value != same(grid(q.reads[0]), truth[j]) or a.text != ("yes" if a.value else "no"):
                errs.append("verify answer wrong")
        elif q.qid == "arc:verify_diff":
            j = test_of[q.reads[1]]
            p = grid(q.reads[0])
            if p.shape != truth[j].shape or a.value != int((p != truth[j]).sum()):
                errs.append("verify_diff answer wrong")
        elif q.qid == "arc:verify_value":
            j = test_of[q.reads[0]]
            verified.add(j)
            m = re.search(r"give (\S+) for|Is (\S+) the value", q.text)
            prop = m.group(1) or m.group(2)
            if a.value != (prop == str(truth[j])):
                errs.append("verify_value answer wrong")
        elif q.qid == "arc:identify":
            _menu(part, X, Y, rule, srule, a, errs)
            if "every" not in q.statement:
                errs.append("identify statement does not cover every training pair")
            stats["menu"] = len(part.prov.get("menu", []))
        elif q.qid == "arc:count_color":
            x = int(re.search(r"color (\d)", q.text).group(1))
            if a.value != int((grid(q.reads[0]) == x).sum()):
                errs.append("count_color wrong")
        elif q.qid == "arc:n_colors":
            if a.value != len(np.unique(grid(q.reads[0]))):
                errs.append("n_colors wrong")
        elif q.qid == "arc:mode_color":
            c = np.bincount(grid(q.reads[0]).ravel(), minlength=10)
            if (c == c.max()).sum() != 1 or a.value != int(c.argmax()):
                errs.append("mode_color wrong")
        elif q.qid == "arc:dims":
            g = grid(q.reads[0])
            if a.text != f"{g.shape[0]}x{g.shape[1]}" or slots[q.reads[0]].dims_header_ok or \
                    part.slots[q.reads[0]].encoding in DIMS_REVEALING:
                errs.append("dims wrong or revealed")
        elif q.qid == "arc:color_at":
            r_, c_ = (int(v) for v in re.search(r"row (\d+), column (\d+)", q.text).groups())
            if a.value != int(grid(q.reads[0])[r_, c_]):
                errs.append("color_at wrong")
        elif q.qid == "arc:changed_cells":
            if a.value != int((grid(q.reads[0]) != grid(q.reads[1])).sum()):
                errs.append("changed_cells wrong")
        elif q.qid == "arc:identical":
            if a.value != same(grid(q.reads[0]), grid(q.reads[1])):
                errs.append("identical wrong")
            if slots[q.reads[0]].role != "candidate":
                errs.append("identical does not compare a candidate grid")
        elif q.qid == "arc:which_contain":
            x = int(re.search(r"color (\d)", q.text).group(1))
            hits = [s for s in q.reads if bool((grid(s) == x).any())]
            if a.value != hits:
                errs.append("which_contain wrong")
            want = "none" if not hits else None
            if want and a.text != "none":
                errs.append("which_contain text")
        elif q.qid == "arc:each_count":
            x = int(re.search(r"color (\d)", q.text).group(1))
            if a.value != [int((grid(s) == x).sum()) for s in q.reads]:
                errs.append("each_count wrong")
        elif q.qid == "arc:each_n_colors":
            if a.value != [len(np.unique(grid(s))) for s in q.reads]:
                errs.append("each_n_colors wrong")
        elif q.qid in GRID_READS:
            _grid_read(q, slots, truth, errs)
        else:
            errs.append(f"unknown query {q.qid}")
    if predicted & verified:
        errs.append("a test input is both solved and verified (the proposal would hand over the answer)")
    windows = [q.reads[0] for q in part.queries if q.qid in GRID_READS]
    sized = {q.reads[0] for q in part.queries if q.qid == "arc:dims"}
    if len(windows) != len(set(windows)) or set(windows) & sized:
        errs.append("two windows of one grid, or a window next to a size question on its grid")


GRID_READS = ("arc:row", "arc:rows", "arc:col", "arc:crop")
_WIN_RE = {
    "arc:row": re.compile(r"\brow (\d+) of ⟪g0⟫"),
    "arc:rows": re.compile(r"\brows (\d+)(?: to | through |-)(\d+) of ⟪g0⟫"),
    "arc:col": re.compile(r"\bcolumn (\d+) of ⟪g0⟫"),
    "arc:crop": re.compile(r"\brows (\d+)(?:-|\.\.| to )(\d+) and columns (\d+)(?:-|\.\.| to )(\d+)\b"),
}


def _grid_read(q, slots, truth, errs):
    """A window of one shown grid, recomputed from the text's coordinates; never the whole grid, never
    a shown grid or a test output, and no misreading of the coordinates (1-indexed, from the far edge,
    rows and columns swapped) lands on the same cells."""
    if len(q.reads) != 1 or slots[q.reads[0]].role not in ("train_input", "train_output", "test_input"):
        errs.append(f"{q.qid}: reads a grid that is not a train input, train output or test input")
        return
    g = arr(slots[q.reads[0]].grid)
    h, w = g.shape
    m = _WIN_RE[q.qid].search(q.text)
    if m is None or len(_WIN_RE[q.qid].findall(q.text)) != 1:
        errs.append(f"{q.qid}: coordinates not found once in {q.text!r}")
        return
    v = [int(x) for x in m.groups()]
    if q.qid == "arc:row":
        a, b, c0, c1 = v[0], v[0], 0, w - 1
    elif q.qid == "arc:rows":
        a, b, c0, c1 = v[0], v[1], 0, w - 1
        if b <= a:
            errs.append("arc:rows: not a range")
    elif q.qid == "arc:col":
        a, b, c0, c1 = 0, h - 1, v[0], v[0]
    else:
        a, b, c0, c1 = v
    if not (0 <= a <= b < h and 0 <= c0 <= c1 < w):
        errs.append(f"{q.qid}: window outside the grid")
        return
    want = g[a:b + 1, c0:c1 + 1]
    if not same(q.answer.value, want):
        errs.append(f"{q.qid} wrong")
    if (a, b, c0, c1) == (0, h - 1, 0, w - 1):
        errs.append(f"{q.qid}: the whole grid")
    if q.qid == "arc:crop" and (a, c0) == (b, c1):
        errs.append("arc:crop: a single cell")
    if any(same(want, arr(s.grid)) for s in slots) or \
            any(t is not None and not isinstance(t, (int, str)) and same(want, t) for t in (truth or [])):
        errs.append(f"{q.qid}: the answer is a shown grid or a test output")
    kh, kw = b - a + 1, c1 - c0 + 1
    rows = {a - 1, h - 1 - b, h - b} if kh < h else set()
    cols = {c0 - 1, w - 1 - c1, w - c1} if kw < w else set()
    starts = {(y, c0) for y in rows} | {(a, x) for x in cols} | {(y, x) for y in rows for x in cols}
    for y, x in starts:
        if (y, x) != (a, c0) and 0 <= y and y + kh <= h and 0 <= x and x + kw <= w and same(g[y:y + kh, x:x + kw], want):
            errs.append(f"{q.qid}: a misread window ({y},{x}) gives the same answer")
    if q.qid == "arc:crop" and c0 + kw <= h and a + kh <= w and (c0, a, kw, kh) != (a, c0, kh, kw) \
            and same(g[c0:c0 + kw, a:a + kh], want):
        errs.append("arc:crop: the swapped reading gives the same answer")
    if "coords" not in q.flags:
        errs.append(f"{q.qid}: no coords flag")
    st = q.statement or ""
    if "⟪answer⟫" in st or not st.endswith(":") or not re.search(r"⟪[gG]0⟫", st):
        errs.append(f"{q.qid}: statement is not an answer-neutral lead-in naming the grid")


def _menu(part, X, Y, rule, srule, answer, errs):
    menu = part.prov.get("menu")
    if not menu:
        errs.append("identify without a menu")
        return
    labels = [lab for lab, _ in menu]
    if answer.text not in labels or len(set(labels)) != len(labels) or len(menu) < 3:
        errs.append("menu labels")
        return
    if labels != list("ABCDEF"[:len(labels)]) or \
            [c[:3] for c in part.context if re.match(r"[A-F]\. ", c)] != [f"{lab}. " for lab in labels]:
        errs.append("menu not labelled A., B., ...")
    sigs = set()
    for lab, entry in menu:
        if srule is not None:
            r = ScalarRule(entry[0], tuple(entry[1]))
            out = tuple(str(r.value(x)) for x in X)
            fits_all = out == tuple(str(srule.value(x)) for x in X)
        else:
            r = Rule(tuple((f, tuple(p)) for f, p in entry))
            res = [r.apply(x) for x in X]
            out = tuple(None if o is None else N.key(arr(o)) for o in res)
            fits_all = all(o is not None and same(o, y) for o, y in zip(res, Y))
        if out in sigs:
            errs.append("two menu entries behave the same on the train inputs")
        sigs.add(out)
        is_answer = lab == answer.text
        if is_answer and not fits_all:
            errs.append("menu answer does not fit")
        if not is_answer and fits_all:
            errs.append("menu distractor fits every pair")
    descs = [c for c in part.context if any(c.startswith(lab) for lab in labels)]
    if len(set(descs)) != len(descs):
        errs.append("menu descriptions repeat")
