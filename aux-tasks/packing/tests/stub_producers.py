"""Small stand-ins for the real producers (items, library, ARC), written against pv2.model only.

They exercise every contract feature the assembler acts on: placeholders in questions, statements
and answers; lead clauses; every FLAG_*; label words; role slots; context lines; grid answers;
answers naming grids; single-answer parts.  The real producers replace them at integration.
"""
from __future__ import annotations

import numpy as np

from pv2 import encodings as enc
from pv2.model import (FLAG_COORDS, FLAG_DIMS, FLAG_EDIT, FLAG_IO_NOUNS, FLAG_LONG, FLAG_NOTHING_ELSE,
                       FLAG_PLACEMENT, FLAG_STRUCTURED, LABEL_WORD_PREFIX, Answer, Part, Query, Slot)


def rand_grid(rng, hmax=9, wmax=9, big_p=0.04):
    if rng.random() < big_p:
        hmax, wmax = 30, 30
    h, w = int(rng.integers(1, hmax + 1)), int(rng.integers(1, wmax + 1))
    k = int(rng.integers(1, 6))
    palette = rng.choice(10, size=k, replace=False)
    return rng.choice(palette, size=(h, w)).tolist()


def _colours(g):
    return sorted({v for row in g for v in row})


def _cells(n, noun="cell"):
    return f"{n} {noun}" + ("" if n == 1 else "s")


# --- library-like questions ------------------------------------------------------------------------------

def lib_dims(rng, g, i):
    h, w = len(g), len(g[0])
    return Query("lib", "lib:dims", (i,), "What are the dimensions of ⟪g0⟫, as rows x columns?",
                 Answer("text", (h, w), f"{h}x{w}"), statement="⟪G0⟫ measures ⟪answer⟫ (rows x columns).",
                 flags=frozenset({FLAG_DIMS}))


def lib_row(rng, g, i):
    r = int(rng.integers(len(g)))
    return Query("lib", "lib:row", (i,), f"Output row {r} of ⟪g0⟫ as a grid with one row.",
                 Answer("grid", [list(g[r])]), statement=f"Row {r} of ⟪g0⟫ is ⟪answer⟫.",
                 flags=frozenset({FLAG_COORDS}))


def lib_cell(rng, g, i):
    r, c = int(rng.integers(len(g))), int(rng.integers(len(g[0])))
    v = g[r][c]
    return Query("lib", "lib:cell", (i,), f"Which color is the cell at row {r}, column {c} of ⟪g0⟫?",
                 Answer("color", v, str(v)), statement=f"The cell at row {r}, column {c} of ⟪g0⟫ has color ⟪answer⟫.",
                 flags=frozenset({FLAG_COORDS}))


def lib_count(rng, g, i):
    k = int(rng.choice(_colours(g))) if rng.random() < 0.8 else int(rng.integers(10))
    n = sum(v == k for row in g for v in row)
    noun = "cell" if n == 1 else "cells"
    return Query("lib", "lib:count", (i,), f"How many cells of ⟪g0⟫ have color {k}?",
                 Answer("number", n, str(n)), statement=f"⟪G0⟫ has ⟪answer⟫ {noun} of color {k}.")


def lib_present(rng, g, i):
    k = int(rng.integers(10))
    yes = any(v == k for row in g for v in row)
    return Query("lib", "lib:present", (i,), f"Does color {k} occur anywhere in ⟪g0⟫?",
                 Answer("bool", yes, "yes" if yes else "no"), statement=f"Color {k} occurs in ⟪g0⟫: ⟪answer⟫.")


def lib_colours(rng, g, i):
    cs = _colours(g)
    text = ", ".join(str(c) for c in cs)
    return Query("lib", "lib:colours", (i,), "Which colors appear in ⟪g0⟫? List them in increasing order.",
                 Answer("list", cs, text), statement="The colors in ⟪g0⟫ are ⟪answer⟫.")


def lib_recolour(rng, g, i):
    a = int(rng.choice(_colours(g)))
    b = int((a + 1 + rng.integers(9)) % 10)
    out = [[b if v == a else v for v in row] for row in g]
    return Query("lib", "lib:recolour", (i,), f"Recolor every cell of color {a} in ⟪g0⟫ to color {b}.",
                 Answer("grid", out), statement=f"⟪G0⟫ with color {a} recolored to {b} is ⟪answer⟫.",
                 flags=frozenset({FLAG_EDIT}))


def lib_long(rng, g, i):
    n = len(g) * len(g[0])
    text = ("Count every cell of ⟪g0⟫, whatever its color, going row by row from the top row to the bottom "
            "row and within each row from left to right, and report how many cells there are in total.")
    return Query("lib", "lib:area", (i,), text, Answer("number", n, str(n)),
                 statement="⟪G0⟫ has ⟪answer⟫ cells in total.", flags=frozenset({FLAG_LONG}))


LIB_SINGLE = (lib_row, lib_cell, lib_count, lib_present, lib_colours, lib_recolour, lib_long, lib_dims)


def lib_same(rng, grids, i, j):
    same = grids[i] == grids[j]
    return Query("lib", "lib:same", (i, j), "Are ⟪g0⟫ and ⟪g1⟫ identical?",
                 Answer("bool", same, "yes" if same else "no"),
                 statement="Whether ⟪g0⟫ and ⟪g1⟫ are identical: ⟪answer⟫.")


def lib_which(rng, grids, reads):
    k = int(rng.integers(10))
    hits = [n for n, s in enumerate(reads) if any(v == k for row in grids[s] for v in row)]
    names = [f"⟪g{n}⟫" for n in hits]
    text = "none" if not names else (names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1])
    listed = ", ".join(f"⟪g{n}⟫" for n in range(len(reads) - 1)) + f" and ⟪g{len(reads) - 1}⟫"
    return Query("lib", "lib:which", tuple(reads), f"Which of {listed} contain color {k}?",
                 Answer("list", [reads[n] for n in hits], text),
                 statement=f"The grids that contain color {k}: ⟪answer⟫.")


# --- item-like questions ---------------------------------------------------------------------------------

def item_flip(rng, g, i):
    out = [list(reversed(row)) for row in g]
    return Query("item", "item:11:v0", (i,), "Reflect this grid across the vertical axis (flip left-right).",
                 Answer("grid", out), flags=frozenset({FLAG_EDIT}), lead_clause=True)


def item_ncolours(rng, g, i):
    n = len(_colours(g))
    return Query("item", "item:9:v1", (i,), "How many different colors appear in the grid?",
                 Answer("number", n, str(n)), lead_clause=True)


def item_symmetry(rng, g, i, nothing_else=False):
    lr = all(row == row[::-1] for row in g)
    ud = g == g[::-1]
    text = (f"horizontal_reflection: {'yes' if ud else 'no'}\n"
            f"vertical_reflection: {'yes' if lr else 'no'}")
    q = ("Decide for horizontal_reflection and then vertical_reflection whether the grid has that symmetry. "
         "Return exactly one line for each label in that order, formatted as label: yes or label: no.")
    flags = {FLAG_STRUCTURED}
    if nothing_else:
        q += " Do not add extra lines or extra text."
        flags.add(FLAG_NOTHING_ELSE)
    return Query("item", "item:102:v0", (i,), q, Answer("text", (ud, lr), text), flags=frozenset(flags),
                 lead_clause=True)


def item_coords(rng, g, i):
    k = int(rng.choice(_colours(g)))
    cells = [(r, c) for r, row in enumerate(g) for c, v in enumerate(row) if v == k][:12]
    text = "[" + ",".join(f"({r},{c})" for r, c in cells) + "]"
    return Query("item", "item:2:v0", (i,),
                 f"List the coordinates of the first {len(cells)} color {k} cells in row-major order as "
                 "[(r,c),...]. Rows and columns are 0-indexed from the top-left corner.",
                 Answer("text", cells, text), flags=frozenset({FLAG_COORDS, FLAG_STRUCTURED}), lead_clause=True)


def item_image(rng, g, i):
    n = sum(v != 0 for row in g for v in row)
    return Query("item", "item:30:v2", (i,), "How many cells of this image are not color 0?",
                 Answer("number", n, str(n)), flags=frozenset({LABEL_WORD_PREFIX + "image"}), lead_clause=True)


def item_io(rng, g, i):
    out = [list(r) for r in zip(*g[::-1])]
    return Query("item", "item:42:v0", (i,), "Rotate the input grid 90 degrees clockwise and give the output.",
                 Answer("grid", out), flags=frozenset({FLAG_IO_NOUNS, FLAG_EDIT}), lead_clause=True)


def item_placement(rng, g, i):
    return Query("item", "item:45:v5", (i,), "How many rows does the grid below have?",
                 Answer("number", len(g), str(len(g))), flags=frozenset({FLAG_PLACEMENT, FLAG_DIMS}),
                 lead_clause=True)


def item_shape_letter(rng, g, i):
    n = sum(v != 0 for row in g for v in row)
    return Query("item", "item:4:v1", (i,), "Count the cells that are not color 0; an L shape counts all its cells.",
                 Answer("number", n, str(n)), flags=frozenset({LABEL_WORD_PREFIX + "L"}), lead_clause=True)


def item_tile(rng, g, i):
    k = int(rng.integers(2, 4))
    out = [list(r) * k for r in g]
    return Query("item", "item:44:v1", (i,), f"Tile this grid {k} times horizontally.", Answer("grid", out),
                 flags=frozenset({LABEL_WORD_PREFIX + "grid_number", FLAG_EDIT}), lead_clause=True)


def item_style_label(rng, g, i):
    n = len(g)
    return Query("item", "item:26:v3", (i,), "Question: how many rows does the grid have? Answer: a number.",
                 Answer("number", n, str(n)), flags=frozenset({FLAG_DIMS, "style_label:question",
                                                               "style_label:answer"}), lead_clause=True)


def item_inner(rng, g, i):
    n = len(g[0])
    return Query("item", "item:42:v0", (i,), "Split the grid into two subgrids, the first on the left and the second "
                 "on the right. How many columns does the grid have?", Answer("number", n, str(n)),
                 flags=frozenset({"inner_ordinals", FLAG_DIMS}), lead_clause=True)


ITEM_SINGLE = (item_flip, item_ncolours, item_symmetry, item_coords, item_image, item_io, item_shape_letter,
               item_tile, item_style_label, item_inner)


# --- slots -----------------------------------------------------------------------------------------------

def _encode(rng, g, queries, i, *, stage2_p=0.25):
    reads = [q for q in queries if i in q.reads]
    coords = any(FLAG_COORDS in q.flags for q in reads)
    dims = any(FLAG_DIMS in q.flags for q in reads)
    items = any(q.source == "item" for q in reads)
    if coords and rng.random() < stage2_p and not dims:
        return "stage2"
    return enc.choose(rng, g, names_ok=not items and not reads, coords_ok=not coords, dims_ok=not dims,
                      background_ok=False)


def _slots(rng, grids, queries, role=None):
    slots = []
    for i, g in enumerate(grids):
        dims = any(FLAG_DIMS in q.flags and i in q.reads for q in queries)
        slots.append(Slot(g, encoding=_encode(rng, g, queries, i), dims_header_ok=not dims))
    return slots


# --- parts -----------------------------------------------------------------------------------------------

def reframed(rng):
    g = rand_grid(rng)
    maker = LIB_SINGLE[int(rng.integers(len(LIB_SINGLE)))] if rng.random() < 0.55 else \
        ITEM_SINGLE[int(rng.integers(len(ITEM_SINGLE)))]
    q = maker(rng, g, 0)
    return Part("reframed", _slots(rng, [g], [q]), [q], prov={"stub": "reframed"})


def placement_single(rng):
    g = rand_grid(rng)
    q = item_placement(rng, g, 0)
    return Part("reframed", _slots(rng, [g], [q]), [q], prov={"stub": "placement"})


def lone_item(rng):
    g = rand_grid(rng)
    q = item_symmetry(rng, g, 0, nothing_else=True)
    return Part("classic", _slots(rng, [g], [q]), [q], single_answer=True, prov={"stub": "lone"})


def one_grid_many(rng):
    g = rand_grid(rng)
    n = int(rng.integers(2, 7))
    makers = list(LIB_SINGLE)
    qs = []
    if rng.random() < 0.4:
        qs.append(ITEM_SINGLE[int(rng.integers(4))](rng, g, 0))       # the item's own question first
    while len(qs) < n:
        qs.append(makers[int(rng.integers(len(makers)))](rng, g, 0))
    return Part("one_grid_many", _slots(rng, [g], qs), qs, prov={"stub": "many"})


def gallery(rng, lib_only=False):
    n = int(rng.integers(2, 13))
    grids = [rand_grid(rng, 7, 7) for _ in range(n)]
    nq = int(rng.integers(1, 6))
    qs = []
    for _ in range(nq):
        i = int(rng.integers(n))
        if lib_only or rng.random() < 0.6:
            qs.append(LIB_SINGLE[int(rng.integers(len(LIB_SINGLE)))](rng, grids[i], i))
        else:
            qs.append(ITEM_SINGLE[int(rng.integers(len(ITEM_SINGLE)))](rng, grids[i], i))
    return Part("gallery", _slots(rng, grids, qs), qs, prov={"stub": "gallery"})


def compare(rng):
    n = int(rng.integers(2, 5))
    grids = [rand_grid(rng, 5, 5) for _ in range(n)]
    if rng.random() < 0.3:
        grids[1] = [list(r) for r in grids[0]]
    qs = [lib_same(rng, grids, 0, 1), lib_which(rng, grids, list(range(n)))]
    if rng.random() < 0.5:
        i = int(rng.integers(n))
        qs.append(lib_count(rng, grids[i], i))
    return Part("compare", _slots(rng, grids, qs), qs, prov={"stub": "compare"})


def arc(rng, n_pairs=None, n_tests=None, verify=False):
    n_pairs = n_pairs or int(rng.integers(2, 5))
    n_tests = n_tests or int(rng.integers(1, 3))
    slots, grids = [], []
    for k in range(n_pairs):
        g = rand_grid(rng, 6, 6)
        out = [list(reversed(r)) for r in g]
        slots += [Slot(g, "stage2", "train_input", k), Slot(out, "stage2", "train_output", k)]
    tests = []
    for k in range(n_tests):
        g = rand_grid(rng, 6, 6)
        tests.append(len(slots))
        slots.append(Slot(g, "stage2", "test_input", k))
    if rng.random() < 0.3:
        for s in slots:
            s.encoding = "compact"
    qs = []
    t = tests[int(rng.integers(len(tests)))]
    qs.append(Query("arc", "arc:predict", (t,), "Give the output grid for ⟪g0⟫ under the rule of the training pairs.",
                    Answer("grid", [list(reversed(r)) for r in slots[t].grid])))
    j = int(rng.integers(len(slots)))
    qs.append(lib_count(rng, slots[j].grid, j))
    if rng.random() < 0.5:
        j = int(rng.integers(len(slots)))
        qs.append(lib_cell(rng, slots[j].grid, j))
    if rng.random() < 0.3:
        j = int(rng.integers(len(slots)))
        q = lib_dims(rng, slots[j].grid, j)
        slots[j].dims_header_ok = False
        qs.append(q)
    if verify:
        prop = [list(reversed(r)) for r in slots[0].grid]
        if rng.random() < 0.5:
            prop = [list(r) for r in slots[0].grid]
        slots.append(Slot(prop, "compact", "proposed"))
        ok = prop == slots[1].grid
        qs.append(Query("arc", "arc:verify", (len(slots) - 1, 0),
                        "Is ⟪g0⟫ the correct output for ⟪g1⟫ under the rule?", Answer("bool", ok, "yes" if ok else "no")))
    ctx = ["The same rule maps every training input to its output: each row is reversed (a left-right flip)."] \
        if rng.random() < 0.5 else []
    return Part("arc", slots, qs, context=ctx, labels_required=True, prov={"stub": "arc"})


FAMILIES = (reframed, one_grid_many, gallery, compare, arc)


def random_parts(rng, max_parts=4):
    """One to ``max_parts`` parts of mixed families (never the lone single-answer kind)."""
    n = 1 if rng.random() < 0.45 else int(rng.integers(2, max_parts + 1))
    out = []
    for _ in range(n):
        f = FAMILIES[int(rng.integers(len(FAMILIES)))]
        out.append(f(rng) if f is not arc else arc(rng, verify=rng.random() < 0.3))
    return out
