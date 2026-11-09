"""Verifier tests for task 58 (highlight every occurrence of a shape template).

Semantics under test (stated in the prompt): an occurrence is a separate
4-connected group of template-colour cells whose normalized shape is one of
the orientations of the template allowed by ``orientation_rule`` (exact /
rotations / rotations_and_reflections).  Every such group is recoloured with
the highlight colour; nothing else changes.

The oracle below re-solves each example from the visible input grid + the
metadata the prompt exposes (template rows, colours, rule) using numpy
rot90/fliplr and a union-find labelling — an implementation independent of
the generator's DFS matcher.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from collections import Counter

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prompts import get_prompt  # noqa: E402
from registry import generate  # noqa: E402

ORACLE_SEEDS = 400
SWEEP_SEEDS = 3000
FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "misc_sibling_hashes_58.json")


# --- independent oracle ------------------------------------------------------

def _allowed_masks(template_rows, rule):
    mask = np.array([[ch == "#" for ch in row] for row in template_rows], dtype=bool)
    bases = [mask] if rule == "exact" else [np.rot90(mask, k) for k in range(4)]
    outs = []
    for b in bases:
        outs.append(b)
        if rule in ("dihedral", "rotations_and_reflections"):
            outs.append(np.fliplr(b))
    uniq = []
    for o in outs:
        if not any(o.shape == u.shape and np.array_equal(o, u) for u in uniq):
            uniq.append(o)
    return uniq


def _label4(mask):
    h, w = mask.shape
    lab = np.zeros((h, w), dtype=int)
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    nxt = 1
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            up = lab[r - 1, c] if r > 0 and mask[r - 1, c] else 0
            left = lab[r, c - 1] if c > 0 and mask[r, c - 1] else 0
            if not up and not left:
                lab[r, c] = nxt
                parent[nxt] = nxt
                nxt += 1
            elif up and left:
                a, b = find(up), find(left)
                lab[r, c] = a
                if a != b:
                    parent[b] = a
            else:
                lab[r, c] = up or left
    comps = {}
    for r in range(h):
        for c in range(w):
            if lab[r, c]:
                comps.setdefault(find(lab[r, c]), []).append((r, c))
    return list(comps.values())


def _solve_58(ex):
    m = ex["metadata"]
    a = np.array(ex["input"])
    variants = _allowed_masks(m["template_rows"], m["orientation_rule"])
    out = a.copy()
    n_match = 0
    for cells in _label4(a == m["template_color"]):
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        sub = np.zeros((max(rs) - min(rs) + 1, max(cs) - min(cs) + 1), dtype=bool)
        for r, c in cells:
            sub[r - min(rs), c - min(cs)] = True
        if any(sub.shape == v.shape and np.array_equal(sub, v) for v in variants):
            n_match += 1
            for r, c in cells:
                out[r, c] = m["highlight_color"]
    return out.tolist(), n_match


# --- tests -------------------------------------------------------------------

def test_oracle_recomputes_output():
    for s in range(ORACLE_SEEDS):
        ex = generate("58", s)
        m = ex["metadata"]
        pred, n_match = _solve_58(ex)
        assert pred == ex["output"], f"seed={s}: oracle disagrees with output"
        assert n_match == m["n_occurrences"], f"seed={s}: {n_match} matches vs {m['n_occurrences']}"
        assert (ex["output"] != ex["input"]) == (m["n_occurrences"] > 0), f"seed={s}"
        flat = {v for row in ex["input"] for v in row}
        assert m["highlight_color"] not in flat, f"seed={s}: highlight colour appears in input"
        assert all(0 <= v <= 9 for v in flat)
        assert isinstance(ex["input"], list) and isinstance(ex["output"], list)


def test_orientation_text_matches_variant_structure():
    """The prompt clause is omitted exactly when the rule adds no new variants."""
    for s in range(ORACLE_SEEDS):
        m = generate("58", s)["metadata"]
        rows, rule, text = m["template_rows"], m["orientation_rule"], m["orientation_text"]
        exact = len(_allowed_masks(rows, "exact"))
        rots = len(_allowed_masks(rows, "rotations"))
        full = len(_allowed_masks(rows, "rotations_and_reflections"))
        assert m["n_allowed_variants"] == {"exact": exact, "rotations": rots, "dihedral": full}[rule]
        if rule == "exact":
            assert (text == "") == (full == exact), f"seed={s}"
        elif rule == "rotations":
            assert (text == "") == (rots == exact and full == rots), f"seed={s}"
        else:
            assert (text == "") == (full == exact), f"seed={s}"
        if "reflected copies do not" in text:
            assert full > rots, f"seed={s}: reflection exclusion stated but reflections add nothing"


def test_prompt_is_symbol_safe_and_self_contained():
    palette = tuple("αβγδεζηθικ")
    for s in range(200):
        ex = generate("58", s)
        m = ex["metadata"]
        prompt = get_prompt("58", m, color_palette=palette)
        assert re.search(r"color [0-9]", prompt) is None, prompt
        assert m["template_pattern"] in prompt
        assert "4-connected" in prompt
        assert set(m["template_pattern"]) <= set("#./"), m["template_pattern"]
        assert m["template_pattern"].count("#") == m["template_cells"]
        plain = get_prompt("58", m)
        assert f"color {m['template_color']}" in plain and f"color {m['highlight_color']}" in plain


def test_distribution_and_coverage():
    rows = [generate("58", seed) for seed in range(SWEEP_SEEDS)]
    metadata = [ex["metadata"] for ex in rows]
    shapes = np.array([np.asarray(ex["input"]).shape for ex in rows])
    # MIN_SIDE = 5 floor, 2026-09-15 (user: small sides are often ambiguous).
    assert shapes.min() == 5 and shapes.max() == 30
    assert np.max(np.prod(shapes, axis=1)) <= 900
    assert abs(np.corrcoef(shapes.T)[0, 1]) < .12
    assert .02 < np.mean(np.prod(shapes, axis=1) >= 700) < .08
    assert np.mean(shapes[:, 0] == shapes[:, 1]) < .06
    counts = Counter(m["n_occurrences"] for m in metadata)
    assert set(range(16)) <= set(counts) and counts[0]/SWEEP_SEEDS < .12
    assert max(m["template_cells"] for m in metadata) >= 150
    assert min(m["template_cells"] for m in metadata) == 2
    assert len(set(m["template_pattern"] for m in metadata)) >= 350
    assert set(m["n_input_colors"] for m in metadata) == set(range(2, 10))
    assert set(m["orientation_rule"] for m in metadata) == {"exact", "rotations", "dihedral"}
    assert {"same_area", "same_box"} <= set(kind for m in metadata for kind in m["scene_kinds"])
    assert np.mean([m["corner_contact"] for m in metadata]) > .08
    assert np.mean([m["attempts"] for m in metadata]) < 3
    for ex in rows:
        m = ex["metadata"]
        assert m["highlight_color"] not in np.asarray(ex["input"])


def test_performance():
    times = []
    for s in range(500):
        t0 = time.perf_counter()
        generate("58", s)
        times.append((time.perf_counter() - t0) * 1000)
    assert np.mean(times) <= 5.0, np.mean(times)
    assert np.percentile(times, 99) <= 30.0


def test_determinism():
    for s in range(50):
        assert generate("58", s) == generate("58", s)


def test_siblings_in_misc_are_byte_identical():
    with open(FIXTURE) as fh:
        expected = json.load(fh)
    for tid, spec in expected.items():
        if tid.startswith("_"):
            continue
        # Tasks 26–28 were deliberately rebuilt after this snapshot.
        # The 31/35 hashes were refreshed for their approved September rebuilds,
        # and the 91/92 hashes for their approved 2026-09-15 rebuilds; the 92b
        # entry is gone with the id, which is merged into 92.  The 83 hash was
        # refreshed for its approved 2026-09-16 background pass (sparse textures).
        if tid in {"26", "27", "28"}:
            continue
        digest = hashlib.sha1()
        for s in range(spec["seeds"]):
            ex = generate(tid, s)
            digest.update(hashlib.sha1(json.dumps(ex, sort_keys=True).encode()).hexdigest().encode())
        assert digest.hexdigest() == spec["sha1"], f"task {tid} output changed"
