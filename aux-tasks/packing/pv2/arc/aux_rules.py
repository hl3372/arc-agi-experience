"""The aux tasks themselves as ARC rules (DESIGN.md 13.10, "aux tasks as rules").

A grid -> grid task whose rule is a function of its metadata becomes an ARC rule by holding the
rule fixed: every example of a bundle must render the SAME instruction.  "Same" is decided on the
instruction text itself, rendered with a symbol palette so colours can be told apart from other
digits: two examples share a rule when their instructions agree after the colours are renamed in
order of first appearance (``rule_key``).  Their colours are then aligned with a permutation
``sigma`` that sends the example's rule colours to the reference example's rule colours (and, by
frequency rank, its other colours onto the reference palette).  The tasks are colour-relabelling
invariant by design (the symbol-remap audit re-renders every prompt under a palette and checks the
example still holds), and the aligned example's instruction is re-rendered and compared with the
reference text, so every output comes from the frozen generator under exactly the stated rule.

Mode A ("stated"): the instruction (a wording without "input"/"output" as plain nouns, placement
or picture words, rendered identically for every example) is the part's context; the train pairs
are worked examples.  Mode B ("hidden"): only for tasks whose rule is a library rule
(``LIBRARY``); the library rule must reproduce every generator output exactly, and the bundle then
passes the full rival search plus the sibling checks of ``pv2.arc.bundle`` (every other parameter
reading contradicted by a train pair).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from pv2.paths import ensure_generator_package

ensure_generator_package()
import prompt_variants  # noqa: E402
from prompts import get_prompt  # noqa: E402
from registry import generate  # noqa: E402
from pv2.pictures import try_generate  # noqa: E402

from pv2.arc.grids import Info, arr  # noqa: E402
from pv2.arc.transforms import Rule  # noqa: E402

SYM = "□▪▲△▼▽◇◎●◯"          # palette symbols; none occurs in any task prompt (checked in tests)

# wordings a role-bearing part may state as its rule: no "input"/"output" (they clash with the ARC
# roles), no placement phrase ("the grid below", "shown above", "below:"), no answer-format sentence.
# Direction words inside a rule ("arriving from above") and "mirror image" stay allowed; ARC labels
# never use picture/panel words, so the label-word clash of DESIGN 13.3 cannot arise.
_BANNED = re.compile(
    r"\binputs?\b|\boutputs?\b|\banswer\b"
    r"|\b(?:grids?|pictures?|images?|panels?|figures?|shown|given|displayed)\s+(?:below|above)\b"
    r"|\b(?:below|above)\s*:",
    re.I)


def wording_ok(text: str) -> bool:
    return not _BANNED.search(text)


# --------------------------------------------------------------------------------------------
# task table
# --------------------------------------------------------------------------------------------

def _conn(m, key="connectivity"):
    return int(m.get(key))


def _lib_85(m):
    kind, v = m.get("filter_kind"), m.get("filter_value")
    if kind == "color":
        return [Rule.single("delete", int(m["filter_color"]))]
    cmp = {"size=": "eq", "size<": "lt", "size>": "gt"}.get(kind)
    if cmp is None or not 1 <= int(v) <= 20:
        return []
    return [Rule.single("remove_size", cmp, int(v), 4)]


LIBRARY: dict[str, Callable[[dict], list]] = {
    "10": lambda m: [Rule.single("rot", k) for k in (1, 2, 3)],
    "11": lambda m: [Rule.single("flip", a) for a in ("lr", "ud")],
    "66": lambda m: [Rule.single("transpose", d) for d in ("main", "anti")],
    "6": lambda m: [Rule.single("scale_up", m["scale"], m["scale"])] if m["scale"] <= 10 else [],
    "55": lambda m: [Rule.single("scale_down", m["scale"], m["scale"])] if m["scale"] <= 10 else [],
    "44": lambda m: [Rule.single("tile", m["tile_rows"], m["tile_cols"])]
    if (m["tile_rows"], m["tile_cols"]) != (1, 1) else [],
    "80": lambda m: [Rule.single("half", m["which_half"])],
    "29": lambda m: [Rule.single("recolor", m["source_color"], m["target_color"])],
    "70": lambda m: [Rule.single("swap", min(m["color_a"], m["color_b"]), max(m["color_a"], m["color_b"]))],
    "47": lambda m: [Rule.single("remove_isolated", _conn(m))],
    "85": _lib_85,
    "20": lambda m: [Rule.single("recolor_size", m["n"], m["target_color"], _conn(m))] if 1 <= m["n"] <= 20 else [],
    "17": lambda m: [Rule.single("recolor_extreme", "largest", "bbox", m["target_color"], _conn(m))],
    "18": lambda m: [Rule.single("recolor_extreme", "largest", "cells", m["target_color"], _conn(m))],
    "82": lambda m: [Rule.single("outline", m["outline_color"], 8, m["outline_thickness"])],
    "83": lambda m: [Rule.single("fill_interior", m["fill_color"], _conn(m))],
    "93": lambda m: [Rule.single("crop_color", m["target_color"])],
    "49": lambda m: [Rule.single("hollow_rects", m["border_thickness"])] if m["border_thickness"] <= 3 else [],
    "65": lambda m: [Rule.single("smallest_tile")],
}

# metadata flags that say another reading of the task (the other connectivity ...) changes the
# answer; a hidden-rule bundle needs at least one train example where it is True
SIBLING_FLAGS = {
    "20": ("differs_under_other_connectivity",),
    "17": ("connectivity_matters",),
    "18": ("connectivity_matters",),
}

# grid -> grid tasks usable with a stated rule, with a sampling weight (lower for slow generators
# or fragmented rule buckets).  14 (text input) and 78 (two-part input) are excluded.
STATED: dict[str, float] = {
    "10": 1.0, "11": 1.0, "66": 1.0, "6": 1.0, "55": 1.0, "80": 1.0, "29": 1.0, "70": 1.0, "47": 1.0,
    "17": 0.4, "18": 0.4, "82": 0.3, "83": 0.5, "93": 0.8, "49": 0.6, "65": 0.8, "12": 0.7, "39": 0.7,
    "31": 0.8, "91": 0.3, "7": 0.3, "8": 0.3, "16": 0.4, "26": 0.8, "34": 0.8, "37": 0.6, "50": 0.8,
    "61": 0.4, "64": 0.6, "67": 0.7, "81": 0.4, "84": 0.2, "87": 0.3, "88": 0.6, "97": 0.1, "99": 0.1,
    "13": 0.2, "28": 0.3, "27": 0.4,
}
# Left out after measurement (2026-09-29): 44, 20, 85, 23, 63 rarely reach enough examples of one
# rule within budget (fragmented rule buckets, or 13-27 ms per example); the library covers 44 / 20 /
# 85 as tile / recolor_size / remove_size.  71, 95, 96, 101 and 5a have no allowed wording.

# measured generate() cost in ms per example (the collection budget scales with it)
COST_MS = {"17": 4.0, "18": 3.5, "82": 6.0, "83": 3.5, "49": 3.0, "91": 4.0, "7": 5.0, "8": 5.0, "16": 3.5,
           "61": 2.0, "81": 3.0, "84": 9.0, "97": 8.0, "99": 7.0, "13": 5.0, "28": 6.0, "12": 1.5, "39": 1.5,
           "37": 1.6, "27": 1.5}


def budget_for(tid: str) -> int:
    """How many examples a collection may generate: ~100 ms worth, between 12 and 40."""
    return max(12, min(40, int(100 / COST_MS.get(tid, 1.0))))


# every stored wording of these uses "input"/"output" or a placement phrase: hidden mode only
NO_WORDING = {"83", "93"}
STATED_OK = [t for t in STATED if t not in NO_WORDING]
HIDDEN = [t for t in STATED if t in LIBRARY]


# --------------------------------------------------------------------------------------------
# rendering and rule keys
# --------------------------------------------------------------------------------------------

def _render(tid: str, meta: dict, template: Optional[str], palette=None) -> str:
    if template is None:
        return get_prompt(tid, meta, color_palette=palette)
    return prompt_variants.render(tid, template, meta, color_palette=palette)


def rule_key(tid: str, meta: dict, template: Optional[str] = None) -> tuple[str, tuple]:
    """(instruction with colours renamed by first appearance, the colours in that order)."""
    text = _render(tid, meta, template, list(SYM))
    order = []
    for ch in text:
        if ch in SYM and SYM.index(ch) not in order:
            order.append(SYM.index(ch))
    norm = "".join(f"⟨{order.index(SYM.index(ch))}⟩" if ch in SYM else ch for ch in text)
    return norm, tuple(order)


def render_under(tid: str, meta: dict, sigma: list, template: Optional[str] = None) -> str:
    """The instruction of an example after its colours were renamed by ``sigma``."""
    text = _render(tid, meta, template, list(SYM))
    return "".join(str(sigma[SYM.index(ch)]) if ch in SYM else ch for ch in text)


def wordings(tid: str) -> list[Optional[str]]:
    """None (the canonical prompt) and every approved stored template."""
    return [None] + [v["template"] for v in prompt_variants.variants_for(tid)]


# --------------------------------------------------------------------------------------------
# examples and colour alignment
# --------------------------------------------------------------------------------------------

@dataclass
class Example:
    seed: int
    inp: np.ndarray
    out: np.ndarray
    meta: dict
    order: tuple              # rule colours in order of first appearance
    sigma: list = field(default_factory=lambda: list(range(10)))

    def mapped(self):
        lut = np.array(self.sigma, dtype=np.int8)
        return lut[self.inp], lut[self.out]


def _by_freq(a_list) -> list[int]:
    counts = np.zeros(10, dtype=np.int64)
    for a in a_list:
        counts += np.bincount(a.ravel(), minlength=10)
    return [int(c) for c in sorted(np.nonzero(counts)[0], key=lambda c: (-counts[c], c))]


def align(ex: Example, ref: Example) -> list:
    """A permutation sending ex's rule colours to ref's, other colours by frequency rank onto
    ref's other colours, and anything left to the unused colours in ascending order."""
    sigma = [-1] * 10
    used = set()
    for e, r in zip(ex.order, ref.order):
        sigma[e] = r
        used.add(r)
    ref_rest = [c for c in _by_freq([ref.inp, ref.out]) if c not in used]
    ex_rest = [c for c in _by_freq([ex.inp, ex.out]) if sigma[c] == -1]
    for e, r in zip(ex_rest, ref_rest):
        sigma[e] = r
        used.add(r)
    free = [c for c in range(10) if c not in used]
    for c in range(10):
        if sigma[c] == -1:
            sigma[c] = free.pop(0)
    return sigma


def _is_grid(x) -> bool:
    return isinstance(x, list) and bool(x) and isinstance(x[0], list)


def collect(rng, tid: str, n: int, *, max_cells: int, budget: int, min_cells: int = 0) -> Optional[list[Example]]:
    """n examples of one rule of task ``tid`` (colours aligned to the richest one, which comes
    first), or None when the budget runs out.  Every generated example is kept in its rule bucket;
    the first bucket to reach n wins."""
    buckets: dict = {}
    for _ in range(budget):
        seed = int(rng.integers(0, 2 ** 31))
        ex = try_generate(tid, seed)
        if ex is None:
            continue
        inp, out = ex["input"], ex["output"]
        if not (_is_grid(inp) and _is_grid(out)):
            return None
        a, b = arr(inp), arr(out)
        if a.size > max_cells or b.size > max_cells or a.size < min_cells:
            continue
        if a.shape == b.shape and np.array_equal(a, b):
            continue
        meta = ex.get("metadata", {})
        key, order = rule_key(tid, meta)
        bucket = buckets.setdefault(key, [])
        bucket.append(Example(seed, a, b, meta, order))
        if len(bucket) == n:
            break
    else:
        return None
    exs = sorted(bucket, key=lambda e: -len(np.unique(np.concatenate([e.inp.ravel(), e.out.ravel()]))))
    ref = exs[0]
    canon = get_prompt(tid, ref.meta)
    for e in exs[1:]:
        e.sigma = align(e, ref)
        if render_under(tid, e.meta, e.sigma) != canon:
            return None
    return exs


def stated_text(rng, tid: str, exs: list[Example]) -> Optional[tuple]:
    """(template or None for the canonical prompt, text): a wording of the rule that renders
    identically for every example and is allowed in a role-bearing part; None when none does.
    Wordings phrased as questions ("What does this grid look like after ...?") are skipped: the
    stated frame presents the rule as an instruction."""
    options = wordings(tid)
    order = [options[i] for i in rng.permutation(len(options))]
    for template in order:
        try:
            texts = {render_under(tid, e.meta, e.sigma, template) for e in exs}
        except Exception:
            continue
        if len(texts) != 1:
            continue
        text = texts.pop()
        if wording_ok(text) and "⟪" not in text and "{" not in text and "?" not in text:
            return template, text
    return None


def library_rule(tid: str, exs: list[Example]) -> Optional[Rule]:
    """The library rule equal to the task's rule on every example (after colour alignment)."""
    if tid not in LIBRARY:
        return None
    ref = exs[0]
    for rule in LIBRARY[tid](ref.meta):
        ok = True
        for e in exs:
            a, b = e.mapped()
            r = rule.apply_info(Info(a))
            if r is None or r.shape != b.shape or not np.array_equal(r, b):
                ok = False
                break
        if ok:
            return rule
    return None
