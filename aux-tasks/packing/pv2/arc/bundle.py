"""ARC-style bundles: a few train pairs and one or two test inputs, with questions about them
(DESIGN.md sections 4, 8, 13.10, 13.11).

``arc_part(rng, mode=None)`` returns a ``Part`` of family "arc" (``build(seed, mode)`` re-creates
it from ``prov['seed']``).  Modes and default shares of ARC parts:

* ``transform``    40%  one library rule (``pv2.arc.transforms``), rule hidden
* ``composition``  12%  two library rules in a row, rule hidden, 4-5 pairs
* ``scalar``       15%  grid -> number / colour / size (``pv2.arc.scalars``); values shown as
                        "grid -> value" context lines, >= 3 pairs with distinct values
* ``stated``       20%  an aux task's own rule, its instruction stated as context (mode A)
* ``hidden``       13%  an aux task whose rule is a library rule, rule hidden (mode B)

Pictures: the reviewed generators' inputs (whole, or a window of a large one) for most rules, and
for object / painting rules also scenes drawn from ``common.shapes`` (``gen.draw_scene``).  One
encoding across the part: the stage-2 rendering in ~60% of parts, else a plain encoding
(``encodings.choose``, never colour names; never a size-printing one when sizes are asked).

Binding checks (DESIGN 13.10) on every hidden-rule bundle, all re-checked independently in
``tests/_arc_check.py``: pair counts (2+ for a global geometric rule, 3+ otherwise, 4-5 for
compositions); every train pair changes its grid and contradicts at least one sibling reading of
the rule (the same family with one parameter changed; the D4 group for rotations / reflections) or,
for rules with few siblings, a generic alternative (D4, gravity, crop, compress); test outputs
differ from test inputs; the rule is defined on every input; no single-colour output unless the
rule makes one by design; every test colour appears in the train inputs, every test-output colour
in the train outputs; background-dependent rules only on plain grounds (``plain_ground``: the
plurality colour covers half the grid and no other cell is isolated, except the marker cells of
marker rules and the noise that isolated-cell removal removes); the test outputs are certified by
the rival search (``pv2.arc.rivals``); a composition some single rule explains is refused.  Aux
rules in hidden mode are strict: every sibling reading (other angle, axis, colour, n,
connectivity, cells vs box ...) must be contradicted by a train pair, and the task's
``*_matters`` / ``differs_under_other_connectivity`` flag must be set on one.  For library rules a
sibling that fits every train pair must also give the same test outputs (a synonym on that bundle,
e.g. 4- vs 8-connectivity when no two objects meet at a corner).  Stated-rule bundles keep the
identity / constant / degeneracy checks; their rule is given, so the answer is fixed by statement.

Placeholders in ``Part.context`` lines name slots by their index in ``Part.slots`` (⟪G0⟫ is
``slots[0]``; ``prov['context_placeholders']`` says so).
"""
from __future__ import annotations

import re

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import pv2.pictures  # noqa: F401  (puts the generator package on the path)
from pv2 import encodings
from pv2.arc import aux_rules, gen, questions as Q
from pv2.arc.gen import Reject
from pv2.arc.grids import DT, Info, fits, is_solid, neighbour_count_same, to_list
from pv2.arc.rivals import certify
from pv2.arc.scalars import SCALARS, ScalarRule, certify_scalar
from pv2.arc.transforms import FAMILIES, Rule
from pv2.library.core import range_rivals
from pv2.library.single import crop_rivals
from pv2.model import Part, Slot

MODES = ("transform", "composition", "scalar", "stated", "hidden")
MODE_WEIGHTS = (0.40, 0.12, 0.15, 0.20, 0.13)
# The solver's own format for ARC bundles.  0.6 made the stage-2 runs title appear in 68% of v2 rows
# (red-team measurement, 2026-09-29): prominent, but a new fixed scaffold; 0.4 keeps it the largest
# single ARC encoding without dominating the rows.
STAGE2_SHARE = 0.4
DIMS_REVEALING = {"stage2", "runs", "row_labelled", "indexed_table", "sparse"}
COORD_PLAIN = {"runs", "row_labelled", "indexed_table", "sparse"}
MAX_ATTEMPTS = 80

# Large bundles (training-side audit, 2026-09-30: ARC-AGI-2 eval grids are >= 400 cells 43% of the
# time and 900 cells 8.7%, ARC bundles had 0.2% >= 400).  A share of each mode's bundles takes the
# fewest pairs its rule allows (2 for global geometric and stated rules, 3 otherwise), one test input,
# no verification and two questions, and spends a stage-2 token budget of 8-12k on big pictures.
# Compositions (4-5 pairs) stay small.
LARGE_P = {"transform": 0.75, "scalar": 0.75, "stated": 0.75, "hidden": 0.7}
LARGE_TOKENS = (9000, 12000)
TOKENS_HARD = 12500                 # estimated stage-2 tokens of a whole part, every bundle
PLAN_TOKENS_PER_CELL = 2.5          # planning figure for picture caps (measured ~3.0 median at >= 400 cells,
                                    # sparse pictures less); heavier bundles are refused by TOKENS_HARD


def stage2_tokens(grid) -> int:
    """Upper-side estimate of a grid's stage-2 tokens with its header: 0.9 tokens per rendered
    character (measured 0.84 median, 0.91 max over the reviewed pictures) plus 14."""
    return int(0.9 * len(encodings.render(grid if isinstance(grid, list) else to_list(grid), "stage2"))) + 14


def part_tokens(train, tests, answers, scalar: bool) -> int:
    """Estimated tokens of a part in the stage-2 encoding: every grid shown, the grid answers in
    compact digits, and ~300 for the questions and context."""
    n = 300 + sum(stage2_tokens(x) for x, _ in train) + sum(stage2_tokens(t) for t in tests)
    if not scalar:
        n += sum(stage2_tokens(y) for _, y in train) + sum(a.size + a.shape[0] for a in answers)
    return n


QUESTION_SETS = ("all", "reading")
# questions a "reading" part may ask: none of them needs the rule (they read the grids shown)
READING_QIDS = frozenset({"arc:count_color", "arc:n_colors", "arc:mode_color", "arc:dims", "arc:color_at",
                          "arc:changed_cells", "arc:identical", "arc:which_contain", "arc:each_count",
                          "arc:each_n_colors"})
# grid-writing reading questions (a row, rows a..b, a column or a crop of one shown grid; DESIGN 15)
READING_QIDS = READING_QIDS | frozenset(Q.GRID_READ_QIDS)
# reading questions per reading-only part (was 2 / 3 / 4 at 0.4 / 0.4 / 0.2): an ARC part is thousands of
# tokens of grids, and each question on them costs little (grid-writing rate, DESIGN 15)
READING_N = {2: 0.05, 3: 0.25, 4: 0.4, 5: 0.3}


@dataclass(frozen=True)
class Filters:
    """Rule sources a build may draw from (None: all).  ``families`` restricts the transform,
    composition and scalar modes (library / scalar family names); ``aux_tasks`` restricts the
    stated and hidden modes (aux task ids)."""
    families: Optional[frozenset] = None
    aux_tasks: Optional[frozenset] = None


_FILTERS = Filters()               # set by build() for the duration of one build

REJECTS: Counter = Counter()       # reason -> count, over the process (for reports)
ATTEMPTS: Counter = Counter()      # mode -> attempts


@dataclass
class Bundle:
    mode: str
    train: list                     # [(input array, output array | value)]
    tests: list                     # [input array]
    answers: list                   # [output array | value]
    rule: object = None             # Rule | ScalarRule | None (stated aux rules)
    rule_text: Optional[str] = None
    scalar_kind: Optional[str] = None
    no_dims: bool = False
    prov: dict = field(default_factory=dict)
    large: bool = False

    @property
    def scalar(self) -> bool:
        return self.scalar_kind is not None


# ================================================================================================
# checks
# ================================================================================================

def _same(a, b) -> bool:
    return a.shape == b.shape and bool(np.array_equal(a, b))


def _basic_checks(train, tests, answers, *, mono_ok) -> None:
    for x, y in train:
        if y is None or not fits(*y.shape):
            raise Reject("undefined")
        if _same(x, y):
            raise Reject("identity_pair")
    for t, a in zip(tests, answers):
        if a is None or not fits(*a.shape):
            raise Reject("undefined_test")
        if _same(t, a):
            raise Reject("test_identity")
    ys = [y for _, y in train]
    if all(_same(ys[0], y) for y in ys[1:]):
        raise Reject("constant_output")
    mono = [len(np.unique(g)) < 2 for g in ys + list(answers)]
    if mono_ok is False and any(mono):
        raise Reject("degenerate_output")
    if mono_ok == "some" and all(mono[:len(ys)]):
        raise Reject("degenerate_output")


def _witness(train, tests, answers) -> None:
    cin = set()
    cout = set()
    for x, y in train:
        cin |= set(np.unique(x).tolist())
        cout |= set(np.unique(y).tolist())
    for t, a in zip(tests, answers):
        if not set(np.unique(t).tolist()) <= cin or not set(np.unique(a).tolist()) <= cout:
            raise Reject("witness")


def plain_ground(a: np.ndarray, marker=None) -> bool:
    """A plain ground: the strict-plurality colour covers at least half the grid and every other
    cell has a 4-neighbour of its own colour (``is_solid``), except lone cells of ``marker``
    (marker rules) or any lone cells when ``marker`` is True (isolated-cell removal)."""
    info = Info(a)
    bg = info.bg
    if bg is None or info.hist[bg] < 0.5 * a.size:
        return False
    lone = (a != bg) & (neighbour_count_same(a, 4) == 0)
    if marker is True:
        return True
    if marker is None:
        return not lone.any()
    return bool(np.all(a[lone] == marker))


def line_has_content(grid: np.ndarray, line: np.ndarray) -> bool:
    """An extracted test line must show something: a cell other than the grid's background (or,
    with no plurality colour, two colours).  A blank line would also be 'the first line with
    content' in some other place, and 28% of test answers were blank before this rule."""
    bg = Info(grid).bg
    if bg is None:
        return len(np.unique(line)) >= 2
    return bool((line != bg).any())


def _cells(train, tests, answers) -> int:
    return sum(x.size + y.size for x, y in train) + sum(t.size + a.size for t, a in zip(tests, answers))


def sibling_report(rule: Rule, train) -> tuple[list, list[bool]]:
    """(siblings no train pair contradicts, per pair: contradicts some sibling or generic
    alternative)."""
    sibs = gen.siblings(rule)
    infos = [Info(x) for x, _ in train]
    pair_hit = [False] * len(train)
    alive = []
    for s in sibs:
        hit = False
        for i, xi in enumerate(infos):
            if hit and pair_hit[i]:
                continue
            out = s.apply_info(xi)
            if out is None or not _same(out, train[i][1]):
                hit = True
                pair_hit[i] = True
        if not hit:
            alive.append(s)
    if not all(pair_hit):
        for g in gen.GENERIC:
            if g.token == rule.token:
                continue
            for i, xi in enumerate(infos):
                if not pair_hit[i]:
                    out = g.apply_info(xi)
                    if out is None or not _same(out, train[i][1]):
                        pair_hit[i] = True
            if all(pair_hit):
                break
    return alive, pair_hit


def _hidden_checks(train, tests, answers, rule: Rule, min_pairs: int, *, strict: bool = False) -> dict:
    """``strict`` (aux rules, DESIGN 13.10 mode B): every sibling reading must be contradicted by a
    train pair.  Otherwise a sibling that fits every train pair must also give the same test outputs
    (a synonym on this bundle, e.g. 4- vs 8-connectivity when no objects meet at a corner); one that
    does not is caught by the certificate."""
    if len(train) < min_pairs:
        raise Reject("pair_count")
    _witness(train, tests, answers)
    alive, pair_hit = sibling_report(rule, train)
    if not all(pair_hit):
        raise Reject("pair_uninformative")
    if alive and strict:
        raise Reject("sibling_consistent")
    for s in alive:
        for t, a in zip(tests, answers):
            o = s.apply_info(Info(t))
            if o is not None and not _same(o, a):
                raise Reject("sibling_differs_on_test")
    cert = certify([(x, y) for x, y in train], tests, rule)
    if cert.outputs is None:
        raise Reject("cert_" + cert.reason)
    for want, got in zip(answers, cert.outputs):
        if not _same(want, got):
            raise Reject("cert_mismatch")
    if rule.is_composition and cert.single_alive:
        raise Reject("composition_collapses")
    return {"n_pivot": cert.n_pivot, "eliminated": cert.eliminated, "n_consistent": len(cert.consistent),
            "n_vacuous": cert.n_vacuous, "synonym_siblings": len(alive)}


def min_pairs(rule: Rule) -> int:
    if rule.is_composition:
        return 4
    return 2 if rule.geometric else 3


# ================================================================================================
# builders
# ================================================================================================

def _counts(r, lo, hi, p=None):
    return int(r.choice(np.arange(lo, hi + 1), p=p))


def _n_test(r) -> int:
    return 1 if r.random() < 0.72 else 2


def _budget(r) -> int:
    """Cells over every grid of a bundle, answers included (about 3.7 stage-2 tokens per cell)."""
    return int(r.integers(1000, 2301))


def _pictures_for(r, spec: gen.GenSpec, n: int, n_train: int, draw: dict, budget: int, growth: float = 1.0,
                  tokens: Optional[int] = None):
    """``tokens`` (large bundles): picture caps from a stage-2 token budget, and a floor of half
    the cap so the pictures come out big."""
    max_h, max_w, g = spec.limits(draw)
    min_cells = 0
    if tokens is None:
        cap = int(budget / (n * (1.0 + max(growth, g))))
        cap = max(9, min(400, cap))
    else:
        cap = max(9, min(900, max_h * max_w, int(tokens / (n * (1.0 + max(growth, g)) * PLAN_TOKENS_PER_CELL))))
        min_cells = cap * 3 // 5 if cap >= 100 else 0
    if spec.make is None and max_h >= 5 and max_w >= 5 and r.random() < spec.scene_share:
        arrays, prov = gen.scene_pictures(r, n, max_cells=min(cap, max_h * max_w), must=spec.must,
                                          marker=spec.marker, pairs=spec.pairs, solid=spec.needs_solid,
                                          ncolors=spec.ncolors, margin=spec.margin, max_h=max_h, max_w=max_w,
                                          min_cells=min_cells)
    else:
        pics = gen.sample_pictures(r, n, solid=spec.needs_solid, max_cells=cap, max_h=max_h, max_w=max_w,
                                   min_side=spec.min_side, sources=spec.sources, min_cells=min_cells)
        arrays = [p.grid for p in pics]
        prov = [(p.tid, p.seed, p.window) for p in pics]
        if r.random() < 0.9:
            arrays = gen.relabel(r, arrays)
    # the colour-richest picture goes to the train pairs, so the test colours are witnessed
    k = max(range(n), key=lambda i: len(np.unique(arrays[i])))
    if k >= n_train:
        j = int(r.integers(n_train))
        arrays[j], arrays[k] = arrays[k], arrays[j]
        prov[j], prov[k] = prov[k], prov[j]
    return arrays, prov


def _steps(rule: Rule) -> list:
    return [[f, list(p)] for f, p in rule.steps]


def _solid_ok(infos) -> bool:
    return all(i.bg is not None and is_solid(i.a) for i in infos)


def _gen_pool() -> list:
    return [g for g in gen.GEN if g.fam in _FILTERS.families]


def _comp_pools() -> tuple[list, list]:
    fam = _FILTERS.families
    seconds = [f for f in gen.COMP_SECOND if f in fam]
    firsts = [f for f in gen.COMP_FIRST if f in fam and any(s != f for s in seconds)]
    return firsts, seconds


def _scalar_pool() -> list:
    return [(n, w) for n, w in SCALAR_GEN if n in _FILTERS.families]


def _aux_pool(hidden: bool) -> list:
    pool = aux_rules.HIDDEN if hidden else aux_rules.STATED_OK
    return pool if _FILTERS.aux_tasks is None else [t for t in pool if t in _FILTERS.aux_tasks]


def buildable(mode: str, filters: Filters) -> bool:
    """Whether ``mode`` has any rule source left under ``filters``."""
    global _FILTERS
    saved, _FILTERS = _FILTERS, filters
    try:
        if mode in ("stated", "hidden"):
            return bool(_aux_pool(mode == "hidden"))
        if filters.families is None:
            return True
        if mode == "transform":
            return bool(_gen_pool())
        if mode == "composition":
            return bool(_comp_pools()[0])
        return bool(_scalar_pool())
    finally:
        _FILTERS = saved


def build_transform(r) -> Bundle:
    spec = gen.choose_spec(r) if _FILTERS.families is None else gen.choose_spec(r, _gen_pool())
    fam = FAMILIES[spec.fam]
    large = r.random() < LARGE_P["transform"]
    if large:
        n_train, n_test, tokens = (2 if fam.geometric else 3), 1, int(r.integers(*LARGE_TOKENS))
    else:
        n_train = _counts(r, 2, 4, [0.45, 0.35, 0.20]) if fam.geometric else _counts(r, 3, 5, [0.5, 0.35, 0.15])
        n_test, tokens = _n_test(r), None
    n = n_train + n_test
    draw = spec.draw(r)
    budget = _budget(r)
    arrays, pics = _pictures_for(r, spec, n, n_train, draw, budget, tokens=tokens)
    if spec.needs_solid and not spec.marker and not _solid_ok([Info(a) for a in arrays]):
        raise Reject("not_solid")
    if spec.make is not None:
        arrays = spec.make(r, arrays, draw)
    infos = [Info(a) for a in arrays]
    if spec.needs_solid and not spec.marker and \
            not all(plain_ground(a, True if spec.fam == "remove_isolated" else None) for a in arrays):
        raise Reject("not_solid")
    params, outs = gen.choose_params(r, spec, infos, n_train, draw)
    rule = Rule.single(spec.fam, *params)
    if fam.bg and spec.marker and not all(plain_ground(a, params[0]) for a in arrays):
        raise Reject("not_solid")
    if spec.fam in ("extract_row", "extract_col") and \
            not all(line_has_content(t, o) for t, o in zip(arrays[n_train:], outs[n_train:])):
        raise Reject("blank_test_line")
    train = list(zip(arrays[:n_train], outs[:n_train]))
    tests, answers = arrays[n_train:], outs[n_train:]
    _basic_checks(train, tests, answers, mono_ok=rule.mono_ok)
    if (not large and _cells(train, tests, answers) > budget * 1.25) or \
            part_tokens(train, tests, answers, False) > TOKENS_HARD:
        raise Reject("budget")
    cert = _hidden_checks(train, tests, answers, rule, min_pairs(rule))
    return Bundle("transform", train, tests, answers, rule=rule, large=large,
                  prov={"rule": rule.token, "steps": _steps(rule), "pictures": pics, "cert": cert})


def build_composition(r) -> Bundle:
    firsts, seconds = (gen.COMP_FIRST, gen.COMP_SECOND) if _FILTERS.families is None else _comp_pools()
    f1 = gen.GEN_BY_FAM[gen.pick(r, firsts)]
    f2 = gen.GEN_BY_FAM[gen.pick(r, [f for f in seconds if f != f1.fam])]
    n_train = _counts(r, 4, 5, [0.65, 0.35])
    n_test = _n_test(r)
    n = n_train + n_test
    d1, d2 = f1.draw(r), f2.draw(r)
    g1, g2 = f1.limits(d1), f2.limits(d2)
    budget = _budget(r)
    spec = gen.GenSpec(f1.fam, 1.0, f1.cands,
                       limits=lambda d: (min(g1[0], g2[0]), min(g1[1], g2[1]), g1[2] * g2[2]),
                       sources=f1.sources, min_side=max(f1.min_side, f2.min_side),
                       solid=f1.needs_solid or f2.needs_solid)
    arrays, pics = _pictures_for(r, spec, n, n_train, d1, budget)
    if f1.make is not None:
        arrays = f1.make(r, arrays, d1)
    infos = [Info(a) for a in arrays]
    if spec.needs_solid and not _solid_ok(infos):
        raise Reject("not_solid")
    p1, mids = gen.choose_params(r, f1, infos, n_train, d1)
    minfos = [Info(m) for m in mids]
    if f2.needs_solid and not _solid_ok(minfos):
        raise Reject("not_solid_mid")
    p2, outs = gen.choose_params(r, f2, minfos, n_train, d2)
    rule = Rule(((f1.fam, tuple(p1)), (f2.fam, tuple(p2))))
    train = list(zip(arrays[:n_train], outs[:n_train]))
    tests, answers = arrays[n_train:], outs[n_train:]
    _basic_checks(train, tests, answers, mono_ok=rule.mono_ok)
    if _cells(train, tests, answers) > budget * 1.25 or part_tokens(train, tests, answers, False) > TOKENS_HARD:
        raise Reject("budget")
    cert = _hidden_checks(train, tests, answers, rule, 4)
    return Bundle("composition", train, tests, answers, rule=rule,
                  prov={"rule": rule.token, "steps": _steps(rule), "pictures": pics, "cert": cert})


# --- scalar ---------------------------------------------------------------------------------------

SCALAR_GEN = [
    ("count_color", 2.0), ("n_colors", 1.0), ("n_fg_colors", 0.7), ("count_non_bg", 1.0), ("n_objects", 1.5),
    ("n_objects_color", 1.0), ("extreme_size", 1.0), ("rows_with", 0.7), ("cols_with", 0.7), ("n_holes", 0.7),
    ("height", 0.4), ("width", 0.4), ("area", 0.3), ("mode_color", 0.4), ("fg_rank_color", 0.8),
    ("obj_color", 0.8), ("corner_color", 0.6), ("dims", 0.5), ("content_dims", 0.8), ("color_dims", 0.8),
    ("obj_dims", 0.6), ("n_isolated", 0.4), ("uniform_rows", 0.3),
]


# which parameters of a scalar family are colours (chosen from the bundle's colours)
SCALAR_COLOR_PARAMS = {"count_color": (0,), "count_two": (0, 1), "n_objects_color": (0,), "rows_with": (0,),
                       "cols_with": (0,), "color_dims": (0,)}


# alternatives every scalar pair should contradict when the rule has few siblings of its own
SCALAR_GENERIC = [ScalarRule("n_colors", ()), ScalarRule("count_non_bg", ()), ScalarRule("height", ()),
                  ScalarRule("width", ()), ScalarRule("count_mode", ()), ScalarRule("n_objects", (4,))]


def _scalar_params(r, name: str, infos) -> tuple:
    idx = SCALAR_COLOR_PARAMS.get(name, ())
    cols = set(gen.union(infos))
    cands = [p for p in SCALARS[name].instances if all(p[i] in cols for i in idx)]
    return gen.pick(r, cands)


def build_scalar(r) -> Bundle:
    pool = SCALAR_GEN if _FILTERS.families is None else _scalar_pool()
    names = [n for n, _ in pool]
    w = np.array([x for _, x in pool])
    name = names[int(r.choice(len(names), p=w / w.sum()))]
    fam = SCALARS[name]
    large = r.random() < LARGE_P["scalar"]
    if large:
        n_train, n_test = 3, 1
    else:
        n_train = _counts(r, 3, 5, [0.55, 0.3, 0.15])
        n_test = _n_test(r)
    n = n_train + n_test
    budget = _budget(r)
    min_cells = 0
    if large:
        cap = max(9, min(900, int(int(r.integers(*LARGE_TOKENS)) / (n * PLAN_TOKENS_PER_CELL))))
        min_cells = cap * 3 // 5
    else:
        cap = max(9, min(400, budget // n))
    solid = fam.bg and name != "n_isolated"
    if large and name != "n_isolated" and r.random() < 0.4:
        # big drawn scenes: generator pictures of 500+ cells are rare and slow to find; half of them
        # near the full 30x30 when the budget allows (four grids, no outputs)
        if cap >= 810 and r.random() < 0.5:
            min_cells = 810
        arrays, scene_prov = gen.scene_pictures(r, n, max_cells=cap, solid=solid, min_cells=min_cells,
                                                must=("hole",) if name == "n_holes" else ())
        pics = [gen.Picture(a, "scene", 0, None) for a in arrays]
    else:
        pics = gen.sample_pictures(r, n, solid=solid, max_cells=cap, min_side=3 if solid else 2,
                                   sources=gen.HOLE_SOURCES if name == "n_holes" else None, min_cells=min_cells)
    arrays = [p.grid for p in pics]
    if fam.kind != "color" and r.random() < 0.8:
        arrays = gen.relabel(r, arrays)
    elif fam.kind == "color":
        arrays = [np.array(r.permutation(10), dtype=DT)[a] for a in arrays]
    infos = [Info(a) for a in arrays]
    params = _scalar_params(r, name, infos)
    rule = ScalarRule(name, params)
    vals = [rule.value_info(i) for i in infos]
    if any(v is None for v in vals):
        raise Reject("scalar_undefined")
    train = list(zip(arrays[:n_train], vals[:n_train]))
    tests, answers = arrays[n_train:], vals[n_train:]
    if len(set(vals[:n_train])) < n_train:
        raise Reject("scalar_repeated")
    if part_tokens(train, tests, answers, True) > TOKENS_HARD:
        raise Reject("budget")
    cert = certify_scalar(train, tests, rule)
    if cert.values is None:
        raise Reject("scalar_" + cert.reason)
    # sibling readings (one parameter changed): each pair contradicts one; a sibling that fits every
    # pair must give the same test values (certify_scalar already guarantees it)
    sibs = [ScalarRule(name, q) for q in fam.instances
            if q != params and sum(1 for a, b in zip(params, q) if a != b) == 1]
    pair_hit = [False] * n_train
    for s in sibs + [g for g in SCALAR_GENERIC if g.name != name]:
        for i, (g, v) in enumerate(train):
            sv = s.value_info(infos[i])
            if sv is None or sv != v:
                pair_hit[i] = True
        if all(pair_hit):
            break
    if not all(pair_hit):
        raise Reject("pair_uninformative")
    return Bundle("scalar", train, tests, answers, rule=rule, scalar_kind=fam.kind, no_dims=fam.reveals_dims, large=large,
                  prov={"rule": rule.token, "scalar": [name, list(params)],
                        "pictures": [(p.tid, p.seed, p.window) for p in pics],
                        "n_consistent": len(cert.consistent)})


# --- aux tasks ---------------------------------------------------------------------------------------

def _aux_pick(r, pool: list) -> str:
    w = np.array([aux_rules.STATED[t] for t in pool], dtype=float)
    return pool[int(r.choice(len(pool), p=w / w.sum()))]


def build_aux(r, hidden: bool) -> Bundle:
    tid = _aux_pick(r, _aux_pool(hidden))
    geometric = tid in ("10", "11", "66", "6", "55", "44", "80")
    large = r.random() < LARGE_P["hidden" if hidden else "stated"]
    if large:
        n_train, n_test = (2 if geometric or not hidden else 3), 1
    elif hidden:
        n_train = _counts(r, 2, 4, [0.4, 0.4, 0.2]) if geometric else _counts(r, 3, 5, [0.55, 0.3, 0.15])
    else:
        n_train = _counts(r, 2, 4, [0.35, 0.45, 0.2])
    if not large:
        n_test = _n_test(r)
    n = n_train + n_test
    min_cells = 0
    if large:
        cap = max(16, min(900, int(int(r.integers(*LARGE_TOKENS)) / (2 * n * PLAN_TOKENS_PER_CELL))))
        min_cells = cap * 3 // 5
    else:
        budget = int(r.integers(1300, 2601))
        cap = max(16, min(450, int(budget / (2 * n))))
    exs = aux_rules.collect(r, tid, n, max_cells=cap, budget=aux_rules.budget_for(tid), min_cells=min_cells)
    if exs is None:
        raise Reject("aux_collect")
    ref, rest = exs[0], exs[1:]
    rest = [rest[i] for i in r.permutation(len(rest))]
    ordered = [ref] + rest
    grids = [e.mapped() for e in ordered]
    train = grids[:n_train]
    tests = [g[0] for g in grids[n_train:]]
    answers = [g[1] for g in grids[n_train:]]
    if part_tokens(train, tests, answers, False) > TOKENS_HARD:
        raise Reject("budget")
    prov = {"task": tid, "seeds": [e.seed for e in ordered], "sigmas": [e.sigma for e in ordered]}
    if not hidden:
        _basic_checks(train, tests, answers, mono_ok=False)
        stated = aux_rules.stated_text(r, tid, ordered)
        if stated is None:
            raise Reject("aux_no_wording")
        template, text = stated
        prov.update({"rule": f"aux:{tid}", "wording": template, "rule_text": text})
        return Bundle("stated", train, tests, answers, rule_text=text, prov=prov, large=large)
    rule = aux_rules.library_rule(tid, ordered)
    if rule is None:
        raise Reject("aux_no_library_rule")
    _basic_checks(train, tests, answers, mono_ok=rule.mono_ok)
    if rule.bg:
        lone = rule.steps[0][0] == "remove_isolated"
        if not all(plain_ground(x, True if lone else None) for x, _ in train) or \
                not all(plain_ground(t, True if lone else None) for t in tests):
            raise Reject("not_solid")
        if lone and not all(plain_ground(y) for _, y in train):
            raise Reject("not_solid")
    flags = aux_rules.SIBLING_FLAGS.get(tid, ())
    for f in flags:
        if not any(bool(e.meta.get(f)) for e in ordered[:n_train]):
            raise Reject("aux_flag")
    cert = _hidden_checks(train, tests, answers, rule, min_pairs(rule), strict=True)
    prov.update({"rule": f"aux:{tid}:{rule.token}", "steps": _steps(rule), "cert": cert})
    return Bundle("hidden", train, tests, answers, rule=rule, prov=prov, large=large)


BUILDERS = {
    "transform": build_transform,
    "composition": build_composition,
    "scalar": build_scalar,
    "stated": lambda r: build_aux(r, hidden=False),
    "hidden": lambda r: build_aux(r, hidden=True),
}


# ================================================================================================
# questions and the Part
# ================================================================================================

def _encoding(r, grid, *, no_dims: bool) -> str:
    if not no_dims and r.random() < STAGE2_SHARE:
        return "stage2"
    weights = {k: v for k, v in encodings.DEFAULT_WEIGHTS.items()
               if k != "stage2" and not (no_dims and k in DIMS_REVEALING)}
    return encodings.choose(r, to_list(grid), names_ok=False, coords_ok=True, dims_ok=not no_dims,
                            background_ok=False, weights=weights)


def _perturb(r, a: np.ndarray, palette: list) -> np.ndarray:
    out = a.copy()
    k = int(r.integers(1, 4))
    cells = r.choice(out.size, size=min(k, out.size), replace=False)
    flat = out.reshape(-1)
    for c in cells:
        opts = [v for v in palette if v != flat[c]] or [v for v in range(10) if v != flat[c]]
        flat[c] = opts[int(r.integers(len(opts)))]
    return out


def _grid_distractors(r, b: Bundle, k: int) -> list[Rule]:
    """Rules that fail on at least one train pair, are defined on every train input, and differ
    from each other on the train inputs: siblings first, then other library rules."""
    rule: Rule = b.rule
    pool = gen.siblings(rule)
    pool = [pool[i] for i in r.permutation(len(pool))][:30]
    extra = []
    for _ in range(40):
        spec = gen.choose_spec(r)
        fam = FAMILIES[spec.fam]
        p = fam.instances[int(r.integers(len(fam.instances)))]
        extra.append(Rule.single(spec.fam, *p))
    if rule.is_composition:
        extra = [Rule((s,)) for s in rule.steps] + extra
    seen_out = set()
    seen_desc = {rule.describe()}
    infos = [Info(x) for x, _ in b.train]
    correct_sig = tuple(Info(y).key for _, y in b.train)
    seen_out.add(correct_sig)
    out = []
    for cand in pool[:12] + extra:
        if len(out) >= k:
            break
        if cand.token == rule.token:
            continue
        res = [cand.apply_info(i) for i in infos]
        if any(x is None for x in res):
            continue
        sig = tuple(Info(x).key for x in res)
        if sig in seen_out:
            continue
        if all(_same(x, y) for x, (_, y) in zip(res, b.train)):
            continue
        d = cand.describe()
        if d in seen_desc:
            continue
        seen_out.add(sig)
        seen_desc.add(d)
        out.append(cand)
    return out


def _scalar_distractors(r, b: Bundle, k: int) -> list[ScalarRule]:
    rule: ScalarRule = b.rule
    infos = [Info(x) for x, _ in b.train]
    fam_names = list(SCALARS)
    cands = [ScalarRule(rule.name, q) for q in SCALARS[rule.name].instances if q != rule.params]
    cands = [cands[i] for i in r.permutation(len(cands))][:8]
    for _ in range(40):
        name = fam_names[int(r.integers(len(fam_names)))]
        inst = SCALARS[name].instances
        cands.append(ScalarRule(name, inst[int(r.integers(len(inst)))]))
    seen = {tuple(v for _, v in b.train)}
    descs = {rule.describe()}
    out = []
    for c in cands:
        if len(out) >= k:
            break
        vals = tuple(c.value_info(i) for i in infos)
        if any(v is None for v in vals) or vals in seen:
            continue
        d = c.describe()
        if d in descs:
            continue
        seen.add(vals)
        descs.add(d)
        out.append(c)
    return out


# how a scalar bundle's values are written (stated with the value lines, so a part without a menu
# still says it; the answer to "predict the value" follows the same form)
SCALAR_FORMAT = {"number": "each value is a number",
                 "color": "each value is a color, written as its number",
                 "size": "each value is a size, written rows x columns, like 3x5"}

STATED_FRAMES = (
    "Every training pair follows one instruction, stated here for a single grid: {text} Each training "
    "output is its input after this instruction; {tests}.",
    "Rule, written for one grid and applied to every training input to give its output: {text}",
    "The training outputs come from applying this instruction to the training inputs: \"{text}\"",
)
# reading-only parts ask nothing about the rule: it is quoted, so its imperatives ("Your job is to ...
# produce a grid") read as the instruction that made the outputs, not as the reader's task (parity review)
STATED_FRAMES_QUOTED = (
    "The training outputs come from applying this instruction to the training inputs: \"{text}\"",
    "Each training output was made from its input by following this instruction: \"{text}\"",
)


def _reading(r, b: Bundle, slots: list[Slot], tr_in: list[int], tr_out: list[int], test_idx: list[int],
             enc: str, questions: str = "all") -> Optional[object]:
    allowed_coords = enc == "stage2" or enc not in COORD_PLAIN
    kinds = ["count", "which", "each", "n_colors", "mode", "changed", "each_ncol"]
    if not b.large:
        kinds.append("identical")      # adds a candidate grid: small bundles only
    if allowed_coords:
        kinds.append("color_at")
    # a bundle whose values are sizes (height, width, area, dims ...) never asks a size, or a window whose
    # answer shows a width or height: the value lines, or the value question, would hand it over
    if enc not in DIMS_REVEALING and not b.no_dims:
        kinds.append("dims")
    if allowed_coords and not b.no_dims and r.random() < GRID_READ_P[questions]:
        return _grid_read(r, b, slots, tr_in + tr_out + test_idx)
    kind = gen.pick(r, kinds)
    all_slots = tr_in + tr_out + test_idx
    palette = sorted({int(v) for s in slots for v in np.unique(np.asarray(s.grid))})
    if kind == "count":
        # mostly a colour of that grid: the bundle palette made 32% of the answers 0 (parity review)
        s = gen.pick(r, all_slots)
        own = sorted(int(v) for v in np.unique(np.asarray(slots[s].grid)))
        x = gen.pick(r, own) if r.random() < 0.85 else gen.pick(r, range(10))
        return Q.count_color(r, s, slots[s].grid, x, enc)
    if kind in ("which", "each", "each_ncol"):
        groups = [("train_input", tr_in)]
        if tr_out:
            groups.append(("train_output", tr_out))
        groups.append(("all", tr_in + tr_out + test_idx))
        group, members = groups[int(r.integers(len(groups)))]
        members = sorted(members)
        if len(members) > 7:
            return None
        grids = [slots[s].grid for s in members]
        if kind == "each_ncol":
            return Q.each_n_colors(r, members, grids, group)
        x = gen.pick(r, palette)
        if kind == "which":
            # mostly a colour some but not all of the grids contain: a palette colour named every
            # grid in 52% of the answers (parity review)
            hits = {c: sum(bool((np.asarray(g) == c).any()) for g in grids) for c in palette}
            split = [c for c in palette if 0 < hits[c] < len(grids)]
            if split and r.random() < 0.7:
                x = gen.pick(r, split)
            return Q.which_contain(r, members, grids, x, group, enc)
        return Q.each_count(r, members, grids, x, group, enc)
    if kind == "n_colors":
        s = gen.pick(r, all_slots)
        return Q.distinct_colors(r, s, slots[s].grid)
    if kind == "mode":
        s = gen.pick(r, all_slots)
        return Q.most_frequent(r, s, slots[s].grid, enc)
    if kind == "changed":
        if not tr_out:
            return None
        k = int(r.integers(len(tr_in)))
        return Q.changed_cells(r, tr_in[k], tr_out[k], slots[tr_in[k]].grid, slots[tr_out[k]].grid)
    if kind == "identical":
        # a candidate grid next to one of the bundle's grids: an exact copy half of the time, else the
        # same grid with one to three cells changed (never two different pictures: "no" every time)
        small = [s for s in all_slots if np.asarray(slots[s].grid).size <= 400]
        if not small:
            return None
        s = gen.pick(r, small)
        g = np.asarray(slots[s].grid, dtype=DT)
        cand = g.copy() if r.random() < 0.5 else _perturb(r, g, sorted(int(v) for v in np.unique(g)) + [
            gen.pick(r, range(10))])
        cs = len(slots)
        slots.append(Slot(to_list(cand), enc, "candidate", None, dims_header_ok=False))
        return Q.identical(r, cs, s, cand, g)
    if kind == "color_at":
        s = gen.pick(r, all_slots)
        return Q.color_at(r, s, slots[s].grid)
    if kind == "dims":
        s = gen.pick(r, all_slots)
        slots[s].dims_header_ok = False
        return Q.dims(r, s, slots[s].grid)
    return None


_COLOUR_IN_TEXT = re.compile(r"\bcolor (\d)\b")


def _measures(q) -> set:
    """What a reading question measures, per grid: ("count", colour, slot) for the colour counts
    (one grid or each of a group), ("ncol", slot) for the numbers of colours, ("has", colour, slot)
    for "which contain", else (qid, reads)."""
    if q.qid in Q.GRID_READ_QIDS or q.qid == "arc:dims":
        return _window_measures(q)
    if q.qid in ("arc:count_color", "arc:each_count"):
        m = _COLOUR_IN_TEXT.search(q.text)
        if m is not None:
            return {("count", int(m.group(1)), s) for s in q.reads}
    if q.qid in ("arc:n_colors", "arc:each_n_colors"):
        return {("ncol", s) for s in q.reads}
    if q.qid == "arc:which_contain":
        m = _COLOUR_IN_TEXT.search(q.text)
        if m is not None:
            return {("has", int(m.group(1)), s) for s in q.reads}
    return {(q.qid, tuple(q.reads))}


def _window_measures(q) -> set:
    """A grid-writing question writes a window of its grid ("window", slot) and its answer shows the
    grid's width or height ("size", slot): one window per grid, and never next to a size question on
    that grid (the library's bounds-size guard)."""
    s = q.reads[0]
    if q.qid == "arc:dims":
        return {(q.qid, tuple(q.reads)), ("size", s)}
    return {("window", s), ("size", s)}


def _answer_tokens(grid) -> int:
    return len(grid) * (len(grid[0]) + 1)


def _grid_budget_ok(slots, queries, q) -> bool:
    """A grid-writing reading question fits the part's stage-2 token budget (``TOKENS_HARD``): every
    grid shown, every grid answer asked so far (predictions included) and this one."""
    if q.qid not in Q.GRID_READ_QIDS:
        return True
    n = 300 + sum(stage2_tokens(s.grid) for s in slots)
    n += sum(_answer_tokens(o.answer.value) for o in queries if o.answer.kind == "grid")
    return n + _answer_tokens(q.answer.value) + 40 <= TOKENS_HARD


def _grid_read_shown(q, slots, b: Bundle) -> bool:
    """A grid-writing answer equal to a grid the part shows (a candidate or proposed grid added after
    it included) or to a test output: never asked."""
    if q.qid not in Q.GRID_READ_QIDS:
        return False
    a = np.asarray(q.answer.value)
    grids = [np.asarray(s.grid) for s in slots] + ([] if b.scalar else [np.asarray(x) for x in b.answers])
    return any(_same(a, g) for g in grids)


# Grid-writing reading questions: the share of reading draws that ask one (when the encoding allows
# coordinates), the kinds' weights, the largest answer (cells), how strongly bigger grids are preferred
# (weight cells ** power) and how often a single-colour window is kept.
GRID_READ_P = {"reading": 0.75, "all": 0.5}
GRID_READ_KINDS = {"rows": 0.40, "row": 0.15, "col": 0.20, "crop": 0.25}
GRID_READ_MAX_CELLS = 600
GRID_READ_SIZE_POWER = 0.5
GRID_READ_MONO_P = 0.25
GRID_READ_BIG_CROP_P = 0.5


def _grid_read(r, b: Bundle, slots: list[Slot], members: list[int]):
    """A row, rows a..b, a column or a crop of one shown grid (a train input, train output or test
    input), never the whole grid, never a window that a misreading (1-indexed, counted from the far
    edge, rows and columns swapped) also gives, never equal to a shown grid or a test output."""
    kinds = list(GRID_READ_KINDS)
    w = np.asarray([GRID_READ_KINDS[k] for k in kinds])
    kind = kinds[int(r.choice(len(kinds), p=w / w.sum()))]
    sizes = np.asarray([np.asarray(slots[s].grid).size for s in members], dtype=float)
    p = sizes ** GRID_READ_SIZE_POWER
    s = members[int(r.choice(len(members), p=p / p.sum()))]
    g = np.asarray(slots[s].grid)
    h, wd = g.shape
    for _ in range(12):
        if kind == "row" and h >= 2:
            a = int(r.integers(h))
            win = g[a:a + 1]
            if any(np.array_equal(win, g[t:t + 1]) for t in range_rivals(h, a, 1)):
                continue
            cand = Q.grid_row(r, s, g, a)
        elif kind == "rows" and h >= 3 and 2 * wd <= GRID_READ_MAX_CELLS:
            k = int(r.integers(2, min(h - 1, GRID_READ_MAX_CELLS // wd) + 1))
            a = int(r.integers(0, h - k + 1))
            win = g[a:a + k]
            if any(np.array_equal(win, g[t:t + k]) for t in range_rivals(h, a, k)):
                continue
            cand = Q.grid_rows(r, s, g, a, a + k - 1)
        elif kind == "col" and wd >= 2 and h >= 2:
            c = int(r.integers(wd))
            win = g[:, c:c + 1]
            if any(np.array_equal(win, g[:, t:t + 1]) for t in range_rivals(wd, c, 1)):
                continue
            cand = Q.grid_col(r, s, g, c)
        elif kind == "crop" and h >= 2 and wd >= 2:
            # half of the crops span at least a third of each side: big blocks, not only small ones
            lo_h, lo_w = (max(1, h // 3), max(1, wd // 3)) if r.random() < GRID_READ_BIG_CROP_P else (1, 1)
            kh, kw = int(r.integers(lo_h, h + 1)), int(r.integers(lo_w, wd + 1))
            if (kh, kw) in ((h, wd), (1, 1)) or kh * kw > GRID_READ_MAX_CELLS:
                continue
            a, c0 = int(r.integers(0, h - kh + 1)), int(r.integers(0, wd - kw + 1))
            win = g[a:a + kh, c0:c0 + kw]
            if any(np.array_equal(win, g[y:y + hh, x:x + ww]) if y + hh <= h and x + ww <= wd else False
                   for y, x, hh, ww in crop_rivals(h, wd, a, a + kh - 1, c0, c0 + kw - 1)):
                continue
            cand = Q.grid_crop(r, s, g, a, a + kh - 1, c0, c0 + kw - 1)
        else:
            return None
        if len(np.unique(win)) < 2 and r.random() >= GRID_READ_MONO_P:
            continue
        if _grid_read_shown(cand, slots, b):
            continue
        return cand
    return None


def _listed_already(q, listed: dict) -> bool:
    """A reading question whose answer the scalar value lines already print: a size question next to
    "Training pair 1 input -> 29x25" (5 of 1200 reading rows), a count of the rule's colour."""
    vals = [listed[s] for s in q.reads if s in listed]
    if not vals:
        return False
    if len(q.reads) == 1:
        return q.answer.text == str(vals[0])
    if isinstance(q.answer.value, list) and len(q.answer.value) == len(q.reads):
        return any(s in listed and str(listed[s]) == str(v) for s, v in zip(q.reads, q.answer.value))
    return False


def assemble(r, b: Bundle, questions: str = "all") -> Part:
    """``questions="reading"``: only questions that never use the rule (no solving, no rule menu,
    no verification); the train pairs, test inputs, a stated rule and scalar value lines stay."""
    first = b.train[0][0]
    enc = _encoding(r, first, no_dims=b.no_dims)
    slots: list[Slot] = []
    tr_in, tr_out, test_idx = [], [], []
    for k, (x, y) in enumerate(b.train):
        tr_in.append(len(slots))
        slots.append(Slot(to_list(x), enc, "train_input", k, dims_header_ok=not b.no_dims))
        if not b.scalar:
            tr_out.append(len(slots))
            slots.append(Slot(to_list(y), enc, "train_output", k))
    for j, t in enumerate(b.tests):
        test_idx.append(len(slots))
        slots.append(Slot(to_list(t), enc, "test_input", j, dims_header_ok=not b.no_dims))
    context: list[str] = []
    queries = []
    menu = None
    hidden = b.mode != "stated"
    if b.rule_text is not None:
        tests_clause = "the test input follows it too" if len(b.tests) == 1 else "the test inputs follow it too"
        frames = STATED_FRAMES if questions == "all" else STATED_FRAMES_QUOTED
        context.append(gen.pick(r, frames).format(text=b.rule_text, tests=tests_clause))
    listed = {}
    if b.scalar:
        # the header says where the values come from (unexplained "values" in 12% of reading rows)
        head = gen.pick(r, ("Each training input with the value one rule gives it",
                            "Training inputs and the values one rule gives them",
                            "Values that one rule gives the training inputs"))
        context.append(f"{head} ({SCALAR_FORMAT[b.scalar_kind]}):")
        for s, (_, v) in zip(tr_in, b.train):
            context.append(f"⟪G{s}⟫ -> {v}")
            listed[s] = v

    # solve / verify each test input (never both on one test input)
    palette = sorted({int(v) for x, y in b.train for v in np.unique(x)} |
                     ({int(v) for _, y in b.train for v in np.unique(y)} if not b.scalar else set()))
    for j, s in enumerate(test_idx if questions == "all" else ()):
        ans = b.answers[j]
        do_predict = True if b.large else r.random() < (0.85 if j == 0 else 0.45)
        if do_predict:
            if b.scalar:
                queries.append(Q.predict_scalar(r, s, b.scalar_kind, ans))
            else:
                queries.append(Q.predict(r, s, to_list(ans)))
            continue
        correct = r.random() < 0.5
        if b.scalar:
            if correct:
                prop = ans
            elif b.scalar_kind == "number":
                prop = max(0, int(ans) + int(gen.pick(r, (-2, -1, 1, 2, 3))))
                prop = prop if prop != ans else int(ans) + 1
            elif b.scalar_kind == "color":
                prop = gen.pick(r, [c for c in range(10) if c != ans])
            else:
                h, w = (int(v) for v in str(ans).split("x"))
                prop = f"{h + 1}x{w}" if r.random() < 0.5 else f"{h}x{max(1, w - 1) if w > 1 else w + 1}"
            queries.append(Q.verify_scalar(r, s, prop, prop == ans, b.scalar_kind))
            continue
        if correct:
            prop = ans.copy()
        else:
            prop = None
            if hidden and r.random() < 0.5:
                for d in _grid_distractors(r, b, 3):
                    o = d.apply_info(Info(b.tests[j]))
                    if o is not None and fits(*o.shape) and not _same(o, ans):
                        prop = o
                        break
            if prop is None:
                prop = _perturb(r, ans, palette)
        is_right = _same(prop, ans)
        ps = len(slots)
        slots.append(Slot(to_list(prop), enc, "proposed", j))
        queries.append(Q.verify(r, ps, s, is_right))
        if prop.shape == ans.shape and r.random() < 0.5:
            queries.append(Q.verify_diff(r, ps, s, int((prop != ans).sum())))

    # identify the rule from a menu (hidden rules)
    if questions == "all" and hidden and b.rule is not None and r.random() < 0.4:
        k = int(r.integers(2, 5))
        dis = _scalar_distractors(r, b, k) if b.scalar else _grid_distractors(r, b, k)
        if len(dis) >= 2:
            entries = [b.rule] + dis
            order = r.permutation(len(entries))
            entries = [entries[i] for i in order]
            lines, labels = Q.menu_lines(r, [e.describe() for e in entries])
            label = labels[[i for i, e in enumerate(entries) if e is b.rule][0]]
            context.extend(lines)
            menu = [[lab, [e.name, list(e.params)] if b.scalar else _steps(e)] for lab, e in zip(labels, entries)]
            reads = (tr_in[0],) if b.scalar else (tr_in[0], tr_out[0])
            queries.append(Q.identify(r, reads, label, scalar=b.scalar))

    # reading questions about one grid, the train inputs, the train outputs or every grid
    if questions == "reading":
        n_read = int(r.choice(list(READING_N), p=list(READING_N.values())))
    elif b.large:
        n_read = 0 if len(queries) >= 2 else 1     # a big bundle asks two questions
    else:
        n_read = int(r.choice([0, 1, 2, 3], p=[0.3, 0.4, 0.2, 0.1]))
        only_predict = len(queries) == 1 and queries[0].qid in ("arc:predict", "arc:predict_value")
        if n_read == 0 and only_predict and r.random() < 0.55:
            n_read = 1
    for _ in range(n_read * 3):
        if n_read == 0:
            break
        q = _reading(r, b, slots, tr_in, tr_out, test_idx, enc, questions)
        # one question per measure and grid: a second wording of the same question copied its answer
        # (15 of 1200 reading rows), and a single-grid count repeated one entry of a per-grid list
        # (82 of 1200; parity review).  Keys, not placeholder texts, which are equal for two grids.
        if q is not None and not _listed_already(q, listed) \
                and not _measures(q) & set().union(*(_measures(o) for o in queries)) \
                and _grid_budget_ok(slots, queries, q):
            queries.append(q)
            n_read -= 1
    queries = [q for q in queries if not _grid_read_shown(q, slots, b)]
    if questions == "reading" and len(queries) < 2:
        raise Reject("reading_questions")
    if not queries:
        s = test_idx[0]
        queries.append(Q.predict_scalar(r, s, b.scalar_kind, b.answers[0]) if b.scalar
                       else Q.predict(r, s, to_list(b.answers[0])))
    part = Part("arc", slots, queries, context=context, labels_required=True, prov=dict(b.prov))
    part.prov.update({"mode": b.mode, "encoding": enc, "n_train": len(b.train), "n_test": len(b.tests),
                      "context_placeholders": "slot index"})
    if menu is not None:
        part.prov["menu"] = menu
    if questions != "all":
        part.prov["questions"] = questions
    part.validate_shape()
    return part


# ================================================================================================
# public API
# ================================================================================================

def _filters(families, aux_tasks) -> Filters:
    fam = aux = None
    if families is not None:
        fam = frozenset(families)
        known = {g.fam for g in gen.GEN} | {n for n, _ in SCALAR_GEN}
        if not fam or fam - known:
            raise ValueError(f"unknown or empty families {sorted(fam - known) or '[]'}; generated families: "
                             f"{sorted(known)}")
    if aux_tasks is not None:
        aux = frozenset(str(t) for t in aux_tasks)
        known = set(aux_rules.STATED)
        if not aux or aux - known:
            raise ValueError(f"unknown or empty aux_tasks {sorted(aux - known) or '[]'}; aux tasks usable as "
                             f"rules: {sorted(known, key=lambda t: int(t))}")
    return Filters(fam, aux)


def build(seed: int, mode: Optional[str] = None, *, questions: str = "all", families=None,
          aux_tasks=None) -> Part:
    """One ARC part from a seed.

    ``questions``: "all" (default: solve, identify, verify and reading questions) or "reading"
    (only questions that never use the rule; at least two).  ``families``: generated library /
    scalar family names the transform, composition and scalar modes may use.  ``aux_tasks``: aux
    task ids the stated and hidden modes may use.  With a filter, ``mode=None`` draws among the modes
    that still have a rule source (default weights, renormalised); a requested mode with none left
    raises ValueError.  With every option at its default the part is exactly the default build."""
    global _FILTERS
    if questions not in QUESTION_SETS:
        raise ValueError(f"questions must be one of {QUESTION_SETS}, not {questions!r}")
    filters = _filters(families, aux_tasks)
    r = np.random.default_rng(seed)
    if mode is None:
        if filters == Filters():
            mode = MODES[int(r.choice(len(MODES), p=MODE_WEIGHTS))]
        else:
            ok = [i for i, m in enumerate(MODES) if buildable(m, filters)]
            if not ok:
                raise ValueError(f"no ARC mode has a rule source under families={families} aux_tasks={aux_tasks}")
            w = np.array([MODE_WEIGHTS[i] for i in ok])
            mode = MODES[ok[int(r.choice(len(ok), p=w / w.sum()))]]
    if mode not in BUILDERS:
        raise ValueError(f"unknown ARC mode {mode!r}")
    if not buildable(mode, filters):
        raise ValueError(f"ARC mode {mode!r} has no rule source under families={families} aux_tasks={aux_tasks}")
    saved, _FILTERS = _FILTERS, filters
    try:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            ATTEMPTS[mode] += 1
            try:
                b = BUILDERS[mode](r)
                part = assemble(r, b, questions)
            except Reject as e:
                REJECTS[(mode, e.reason)] += 1
                continue
            part.prov.update({"seed": seed, "attempts": attempt})
            if filters.families is not None:
                part.prov["families_filter"] = sorted(filters.families)
            if filters.aux_tasks is not None:
                part.prov["aux_tasks_filter"] = sorted(filters.aux_tasks, key=lambda t: int(t))
            return part
    finally:
        _FILTERS = saved
    raise RuntimeError(f"no ARC bundle in {MAX_ATTEMPTS} attempts (mode {mode})")


def arc_part(rng, *, mode: Optional[str] = None, questions: str = "all", families=None,
             aux_tasks=None) -> Part:
    """One ARC-style part; deterministic in ``rng``'s state (the part records its own seed in
    ``prov['seed']``, and ``build(seed, mode, ...)`` with the same options re-creates it).  See
    ``build`` for ``questions``, ``families`` and ``aux_tasks``."""
    seed = int(rng.integers(0, 2 ** 63 - 1))
    return build(seed, mode, questions=questions, families=families, aux_tasks=aux_tasks)
