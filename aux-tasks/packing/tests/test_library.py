"""The question library: every posed answer is recomputed here by an independent, pure-Python
implementation (lists and loops, no numpy, no library helpers), from the grids and the drawn
parameters; the texts are checked for the parameters they must state, clean statement forms,
pluralisation, probe hygiene, truthful flags, encoding restrictions and determinism."""
from __future__ import annotations

import importlib.util
import os
import re
from collections import Counter

import numpy as np
import pytest

from pv2 import encodings as enc
from pv2.library import (AS_SHOWN, QUESTIONS, LibQuery, chain_part, comparison_part, library_part, pose)
from pv2.library.core import ZERO_PARENS, ZERO_SENTENCES
from pv2.model import FLAG_COORDS, FLAG_DIMS, FLAG_EDIT, KINDS, LABEL_WORD_PREFIX, Part, Slot, fill

NAMES = ("black", "blue", "red", "green", "yellow", "grey", "magenta", "orange", "cyan", "maroon")
ZERO = ("0-indexed", "from 0", "starting at 0", "Row 0 is the top row", "(0,0) the top-left")
REFS = ["the third grid", "Grid B", "Training pair 1 input", "Training pair 1 output", "Test input 1",
        "the last grid", "grid number 7", "Image 4"]
EDIT_QIDS = {"rotate", "reflect", "transpose", "recolour", "upscale"}
DIMS_QIDS = {"size", "rows", "cols", "n_cells", "same_size", "larger", "each_rows", "each_cols", "each_size",
             "each_n_cells", "which_taller"}


# --- the independent oracle ------------------------------------------------------------------------

def H(g):
    return len(g)


def W(g):
    return len(g[0])


def label(v, names):
    return NAMES[v] if names else str(v)


def word(v, names):
    return NAMES[v] if names else "color " + str(v)


def column(g, c):
    return [g[r][c] for r in range(H(g))]


def cells(g):
    return [(r, c) for r in range(H(g)) for c in range(W(g))]


def count(g, x):
    return sum(1 for r, c in cells(g) if g[r][c] == x)


def tally(g):
    out = {}
    for r, c in cells(g):
        out[g[r][c]] = out.get(g[r][c], 0) + 1
    return out


def first_seen(g):
    out = []
    for r, c in cells(g):
        if g[r][c] not in out:
            out.append(g[r][c])
    return out


def rot_cw(g):
    return [[g[H(g) - 1 - j][i] for j in range(H(g))] for i in range(W(g))]


def rot_ccw(g):
    return [[g[j][W(g) - 1 - i] for j in range(H(g))] for i in range(W(g))]


def rot180(g):
    return [list(reversed(row)) for row in reversed(g)]


def flip_lr(g):
    return [list(reversed(row)) for row in g]


def flip_tb(g):
    return [list(row) for row in reversed(g)]


def transpose(g):
    return [[g[r][c] for r in range(H(g))] for c in range(W(g))]


def window(g, r, c, kh, kw):
    if r < 0 or c < 0 or r + kh > H(g) or c + kw > W(g):
        return None
    return [g[i][c:c + kw] for i in range(r, r + kh)]


def rival_starts(n, start, span):
    end = start + span - 1
    return [s for s in {start - 1, n - 1 - end, n - end} if 0 <= s <= n - span and s != start]


def runs(row):
    out, i = [], 0
    while i < len(row):
        j = i
        while j + 1 < len(row) and row[j + 1] == row[i]:
            j += 1
        out.append(f"{row[i]}@c{i}" if i == j else f"{row[i]}@c{i}-c{j}")
        i = j + 1
    return " ".join(out)


def run_span(line, i):
    a = i
    while a > 0 and line[a - 1] == line[i]:
        a -= 1
    b = i
    while b + 1 < len(line) and line[b + 1] == line[i]:
        b += 1
    return a, b


def run_answer(g, r, c, orient, what):
    line, i = (g[r], c) if orient == "h" else (column(g, c), r)
    a, b = run_span(line, i)
    return b - a + 1 if what == "length" else b


def cell_rival_cells(g, r, c):
    out = []
    for a, b in ((r - 1, c - 1), (H(g) - 1 - r, W(g) - 1 - c), (c, r)):
        if 0 <= a < H(g) and 0 <= b < W(g) and (a, b) != (r, c) and (a, b) not in out:
            out.append((a, b))
    return out


def name_list(positions):
    names = [f"⟪g{i}⟫" for i in positions]
    if not names:
        return "none"
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def yn(flag):
    return "yes" if flag else "no"


def expected(q, grids, names):
    """(value, text) the question must have, or raise AssertionError if it should not have been posed."""
    qid = q.qid[4:]
    p = q.params
    g = grids[0]
    h, w = (H(g), W(g))
    if qid == "size":
        return [h, w], f"{h}x{w}"
    if qid == "rows":
        return h, str(h)
    if qid == "cols":
        return w, str(w)
    if qid == "n_cells":
        return h * w, str(h * w)
    if qid == "cell":
        r, c = p["r"], p["c"]
        for a, b in cell_rival_cells(g, r, c):
            assert g[a][b] != g[r][c], "cell misreading guard"
        return g[r][c], label(g[r][c], names)
    if qid == "first_cell":
        return g[0][0], label(g[0][0], names)
    if qid == "last_cell":
        return g[-1][-1], label(g[-1][-1], names)
    if qid == "corner":
        pos = {"top-left": (0, 0), "top-right": (0, w - 1), "bottom-left": (h - 1, 0), "bottom-right": (h - 1, w - 1)}
        r, c = pos[p["corner"]]
        assert any(pos[k] != (r, c) and g[pos[k][0]][pos[k][1]] != g[r][c] for k in pos)
        return g[r][c], label(g[r][c], names)
    if qid in ("row", "runs_row"):
        r = p["r"]
        assert h >= 2 and all(g[s] != g[r] for s in rival_starts(h, r, 1)), "row misreading guard"
        return ([list(g[r])], None) if qid == "row" else (runs(g[r]), runs(g[r]))
    if qid == "col":
        c = p["c"]
        assert w >= 2 and all(column(g, s) != column(g, c) for s in rival_starts(w, c, 1))
        return [[v] for v in column(g, c)], None
    if qid in ("rows_range", "cols_range"):
        a, b = p["a"], p["b"]
        lines = g if qid == "rows_range" else transpose(g)
        n = len(lines)
        assert a < b and (a, b) != (0, n - 1)
        for s in rival_starts(n, a, b - a + 1):
            assert lines[s:s + b - a + 1] != lines[a:b + 1], "range misreading guard"
        cut = [list(r) for r in lines[a:b + 1]]
        return (cut if qid == "rows_range" else transpose(cut)), None
    if qid == "crop":
        a, b, c0, c1 = p["a"], p["b"], p["c0"], p["c1"]
        kh, kw = b - a + 1, c1 - c0 + 1
        assert (a, b, c0, c1) != (0, h - 1, 0, w - 1) and not (kh == 1 and kw == 1)
        crop = window(g, a, c0, kh, kw)
        for rs in [a] + rival_starts(h, a, kh):
            for cs in [c0] + rival_starts(w, c0, kw):
                if (rs, cs) != (a, c0):
                    assert window(g, rs, cs, kh, kw) != crop, "crop misreading guard"
        assert window(g, c0, a, kw, kh) != crop or (c0, a) == (a, c0)
        return crop, None
    if qid == "half":
        half = p["half"]
        if half in ("top", "bottom"):
            assert h % 2 == 0 and g[:h // 2] != g[h // 2:]
            part = g[:h // 2] if half == "top" else g[h // 2:]
        else:
            assert w % 2 == 0 and [r[:w // 2] for r in g] != [r[w // 2:] for r in g]
            part = [r[:w // 2] for r in g] if half == "left" else [r[w // 2:] for r in g]
        return [list(r) for r in part], None
    if qid == "count":
        n = count(g, p["X"])
        return n, str(n)
    if qid == "count_not":
        assert count(g, p["X"]) > 0
        n = h * w - count(g, p["X"])
        return n, str(n)
    if qid in ("most_frequent", "least_frequent"):
        t = tally(g)
        best = max(t.values()) if qid == "most_frequent" else min(t.values())
        winners = [v for v in t if t[v] == best]
        assert len(winners) == 1 and len(t) >= (2 if qid == "least_frequent" else 1), "tie or single colour"
        return winners[0], label(winners[0], names)
    if qid == "colours_present":
        order = first_seen(g) if p["order"] == "first" else sorted(tally(g))
        assert not (names and p["order"] == "sorted"), "numeric order is invisible in colour names"
        return order, ", ".join(label(v, names) for v in order)
    if qid == "n_colours":
        return len(tally(g)), str(len(tally(g)))
    if qid == "has_colour":
        return count(g, p["X"]) > 0, yn(count(g, p["X"]) > 0)
    if qid == "positions":
        found = [(r, c) for r, c in cells(g) if g[r][c] == p["X"]]
        assert found
        if p["cap"] is not None:
            assert len(found) > 10 and p["cap"] < len(found)
            found = found[:p["cap"]]
        else:
            assert len(found) <= 10
        return [[r, c] for r, c in found], "[" + ",".join(f"({r},{c})" for r, c in found) + "]"
    if qid == "histogram":
        t = tally(g)
        order = first_seen(g) if p["order"] == "first" else sorted(t)
        assert not (names and p["order"] == "sorted")
        return [[v, t[v]] for v in order], ", ".join(f"{label(v, names)}:{t[v]}" for v in order)
    if qid in ("bbox", "positions", "rows_containing", "first_row_containing", "n_rows_containing"):
        assert count(g, p["X"]) <= 0.6 * h * w, "a colour covering most of the grid makes this trivial"
    if qid == "bbox":
        pts = [(r, c) for r, c in cells(g) if g[r][c] == p["X"]]
        assert pts
        box = [min(r for r, _ in pts), min(c for _, c in pts), max(r for r, _ in pts), max(c for _, c in pts)]
        return box, "(" + ",".join(map(str, box)) + ")"
    if qid == "rows_containing":
        rows = [r for r in range(h) if p["X"] in g[r]]
        assert rows
        return rows, ", ".join(map(str, rows))
    if qid == "n_rows_containing":
        n = sum(1 for r in range(h) if p["X"] in g[r])
        return n, str(n)
    if qid == "first_row_containing":
        rows = [r for r in range(h) if p["X"] in g[r]]
        return (rows[0], str(rows[0])) if rows else (None, "none")
    if qid == "border_count":
        n = sum(1 for r, c in cells(g) if (r in (0, h - 1) or c in (0, w - 1)) and g[r][c] == p["X"])
        return n, str(n)
    if qid in ("row_count", "col_count"):
        lines = g if qid == "row_count" else transpose(g)
        i = p["r"] if qid == "row_count" else p["c"]
        counts = [sum(1 for v in line if v == p["X"]) for line in lines]
        assert all(counts[s] != counts[i] for s in rival_starts(len(lines), i, 1))
        return counts[i], str(counts[i])
    if qid in ("run_length", "run_end"):
        what = "length" if qid == "run_length" else "end"
        r, c, orient = p["r"], p["c"], p["orient"]
        ans = run_answer(g, r, c, orient, what)
        for a, b in cell_rival_cells(g, r, c):
            assert run_answer(g, a, b, orient, what) != ans, "run misreading guard"
        return ans, str(ans)
    if qid == "symmetric_lr":
        return g == flip_lr(g), yn(g == flip_lr(g))
    if qid == "symmetric_tb":
        return g == flip_tb(g), yn(g == flip_tb(g))
    if qid == "row_uniform":
        assert w >= 2
        flag = len(set(g[p["r"]])) == 1
        return flag, yn(flag)
    if qid == "n_uniform_rows":
        n = sum(1 for row in g if len(set(row)) == 1)
        return n, str(n)
    if qid == "row_equal":
        flag = g[p["a"]] == g[p["b"]]
        return flag, yn(flag)
    if qid == "n_distinct_rows":
        n = len({tuple(r) for r in g})
        return n, str(n)
    if qid == "rotate":
        out = {"cw": rot_cw, "ccw": rot_ccw, "180": rot180}[p["dir"]](g)
        assert out != g
        return out, None
    if qid == "reflect":
        out = flip_lr(g) if p["axis"] == "lr" else flip_tb(g)
        assert out != g
        return out, None
    if qid == "transpose":
        assert transpose(g) != g
        return transpose(g), None
    if qid == "recolour":
        assert count(g, p["X"]) > 0 and p["Y"] != p["X"]
        return [[p["Y"] if v == p["X"] else v for v in row] for row in g], None
    if qid == "upscale":
        k = p["k"]
        assert k * h <= 30 and k * w <= 30
        return [[g[r // k][c // k] for c in range(w * k)] for r in range(h * k)], None
    # --- two grids
    if qid in ("identical", "n_diff", "diff_cells", "more_colour", "same_size", "larger", "diff_rows", "first_diff"):
        a, b = grids[0], grids[1]
        same = (H(a), W(a)) == (H(b), W(b))
        diffs = [(r, c) for r, c in cells(a) if same and a[r][c] != b[r][c]]
        if qid == "identical":
            return a == b, yn(a == b)
        if qid == "same_size":
            return same, yn(same)
        if qid == "larger":
            assert H(a) * W(a) != H(b) * W(b)
            win = 0 if H(a) * W(a) > H(b) * W(b) else 1
            return [q.reads[win]], name_list([win])
        if qid == "more_colour":
            ca, cb = count(a, p["X"]), count(b, p["X"])
            assert ca != cb
            win = 0 if ca > cb else 1
            return [q.reads[win]], name_list([win])
        assert same
        if qid == "n_diff":
            return len(diffs), str(len(diffs))
        if qid == "diff_cells":
            assert diffs
            if p["cap"] is not None:
                assert len(diffs) > 10 and p["cap"] < len(diffs)
                diffs = diffs[:p["cap"]]
            else:
                assert len(diffs) <= 10
            return ([[r, c, a[r][c], b[r][c]] for r, c in diffs],
                    ", ".join(f"({r},{c}):{label(a[r][c], names)}->{label(b[r][c], names)}" for r, c in diffs))
        if qid == "diff_rows":
            rows = [r for r in range(H(a)) if a[r] != b[r]]
            return rows, (", ".join(map(str, rows)) if rows else "none")
        if qid == "first_diff":
            return ([list(diffs[0]), f"({diffs[0][0]},{diffs[0][1]})"] if diffs else [None, "none"])
    # --- several grids
    if qid.startswith("each_"):
        base = qid[5:]
        x = p.get("X")
        vals = []
        for gg in grids:
            if base == "rows":
                vals.append(H(gg))
            elif base == "cols":
                vals.append(W(gg))
            elif base == "size":
                vals.append([H(gg), W(gg)])
            elif base == "n_cells":
                vals.append(H(gg) * W(gg))
            elif base == "n_colours":
                vals.append(len(tally(gg)))
            elif base == "count":
                vals.append(count(gg, x))
            elif base == "first_cell":
                vals.append(gg[0][0])
            elif base == "most":
                t = tally(gg)
                winners = [v for v in t if t[v] == max(t.values())]
                assert len(winners) == 1
                vals.append(winners[0])
            elif base == "has":
                vals.append(count(gg, x) > 0)
            elif base == "symmetric":
                vals.append(gg == flip_lr(gg))
            else:
                raise AssertionError(qid)
        if base in ("first_cell", "most"):
            text = ", ".join(label(v, names) for v in vals)
        elif base in ("has", "symmetric"):
            text = ", ".join(yn(v) for v in vals)
        elif base == "size":
            text = ", ".join(f"{v[0]}x{v[1]}" for v in vals)
        else:
            text = ", ".join(map(str, vals))
        return vals, text
    if qid.startswith("which_"):
        x = p.get("X")
        preds = {"contain": lambda gg: count(gg, x) > 0, "lack": lambda gg: count(gg, x) == 0,
                 "symmetric_lr": lambda gg: gg == flip_lr(gg), "symmetric_tb": lambda gg: gg == flip_tb(gg),
                 "taller": lambda gg: H(gg) > W(gg), "first_cell": lambda gg: gg[0][0] == x}
        hits = [i for i, gg in enumerate(grids) if preds[qid[6:]](gg)]
        return [q.reads[i] for i in hits], name_list(hits)
    raise AssertionError(f"no oracle for {q.qid}")


# --- operations (verification, multiple choice) -------------------------------------------------------

def apply_op(p, g):
    op = p["op"]
    if op == "rotate":
        return {"cw": rot_cw, "ccw": rot_ccw, "180": rot180}[p["dir"]](g)
    if op == "reflect":
        return flip_lr(g) if p["axis"] == "lr" else flip_tb(g)
    if op == "transpose":
        return transpose(g)
    if op == "row":
        return [list(g[p["r"]])]
    if op == "col":
        return [[v] for v in column(g, p["c"])]
    if op == "crop":
        return window(g, p["a"], p["c0"], p["b"] - p["a"] + 1, p["c1"] - p["c0"] + 1)
    if op == "recolour":
        return [[p["Y"] if v == p["X"] else v for v in row] for row in g]
    if op == "half":
        h, w = H(g), W(g)
        return {"top": lambda: g[:h // 2], "bottom": lambda: g[h // 2:], "left": lambda: [r[:w // 2] for r in g],
                "right": lambda: [r[w // 2:] for r in g]}[p["half"]]()
    raise AssertionError(op)


def n_cells_differ(a, b):
    return sum(1 for r, c in cells(a) if a[r][c] != b[r][c])


def check_mistake(p, g, name, m):
    """The distractor / wrong proposal ``m`` is the named realistic mistake."""
    op, res = p["op"], apply_op(p, g)
    assert m != res, "a mistake must differ from the correct result"
    if name == "one_cell_off":
        assert (H(m), W(m)) == (H(res), W(res)) and n_cells_differ(m, res) == 1
        return
    if op == "rotate":
        table = {"cw": {"wrong_direction": rot_ccw, "transposed": transpose, "half_turn": rot180},
                 "ccw": {"wrong_direction": rot_cw, "transposed": transpose, "half_turn": rot180},
                 "180": {"mirrored_left_right": flip_lr, "flipped_upside_down": flip_tb, "quarter_turn": rot_cw}}
        assert m == table[p["dir"]][name](g)
    elif op == "reflect":
        other = flip_tb if p["axis"] == "lr" else flip_lr
        assert m == {"wrong_axis": other, "half_turn": rot180}[name](g)
    elif op == "transpose":
        assert m == {"rotated_clockwise": rot_cw, "rotated_counterclockwise": rot_ccw,
                     "anti_transposed": lambda x: transpose(rot180(x))}[name](g)
    elif op in ("row", "col"):
        lines = g if op == "row" else transpose(g)
        i = p["r"] if op == "row" else p["c"]
        n = len(lines)
        j = {"one_indexed": i - 1, "far_edge": n - 1 - i, "far_edge_one_indexed": n - i, "next_line": i + 1}.get(name)
        if name == "other_axis":
            line = column(g, i) if op == "row" else g[i]
        else:
            assert 0 <= j < n and j != i
            line = lines[j]
        assert m == ([list(line)] if op == "row" else [[v] for v in line])
    elif op == "crop":
        a, c0, kh, kw = p["a"], p["c0"], p["b"] - p["a"] + 1, p["c1"] - p["c0"] + 1
        want = {"one_indexed": (a - 1, c0 - 1, kh, kw), "rows_one_indexed": (a - 1, c0, kh, kw),
                "columns_one_indexed": (a, c0 - 1, kh, kw), "shifted_down": (a + 1, c0, kh, kw),
                "transposed_ranges": (c0, a, kw, kh)}[name]
        assert m == window(g, *want)
    elif op == "recolour":
        x, y = p["X"], p["Y"]
        if name == "swapped":
            assert m == [[y if v == x else (x if v == y else v) for v in row] for row in g]
        elif name == "missed_cell":
            diff = [(r, c) for r, c in cells(res) if res[r][c] != m[r][c]]
            assert len(diff) == 1 and g[diff[0][0]][diff[0][1]] == x and m[diff[0][0]][diff[0][1]] == x
        elif name == "other_colour":
            assert any(m == [[y if v == z else v for v in row] for row in g] for z in range(10) if z not in (x, y))
        else:
            raise AssertionError(name)
    elif op == "half":
        h, w = H(g), W(g)
        halves = {}
        if h % 2 == 0:
            halves.update(top=g[:h // 2], bottom=g[h // 2:])
        if w % 2 == 0:
            halves.update(left=[r[:w // 2] for r in g], right=[r[w // 2:] for r in g])
        opposite = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}[p["half"]]
        if name == "opposite_half":
            assert m == halves[opposite]
        else:
            assert name == "other_axis_half" and any(m == v for k, v in halves.items() if k not in (p["half"], opposite))
    else:
        raise AssertionError(op)


# --- text checks ----------------------------------------------------------------------------------------

def _probe_module():
    from pv2.paths import WORKSPACE_ROOT
    path = os.path.join(WORKSPACE_ROOT, "stage2", "probe_perception.py")
    if not os.path.exists(path):
        return None
    import sys
    sys.path.insert(0, os.path.dirname(path))
    spec = importlib.util.spec_from_file_location("probe_perception_ref", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PROBE_PATTERNS = [re.compile(p, re.I) for p in (
    r"How many cells differ between .+ and .+\?",
    r"What is the colou?r of the cell at row \d+, column \d+ \(zero-based\) of .+\?",
    r"What is the size of .+\?",
    r"How many cells of colou?r \S+ does .+ have\?",
    r"Write row \d+ \(zero-based\) of .+ as consecutive digits",
)]


def probe_sentences():
    """The probe's exact question sentences, numbers and grid names filled in."""
    mod = _probe_module()
    out = []
    grids = [("input", 0), ("output", 1), ("test", 0)]
    for zero in (True, False):
        for role, i in grids:
            for r, c in ((0, 0), (3, 5), (12, 7)):
                out.append(mod.question_text({"kind": "cell", "grid": [role, i], "r": r, "c": c}, zero))
                out.append(mod.question_text({"kind": "row", "grid": [role, i], "r": r}, zero))
            out.append(mod.question_text({"kind": "size", "grid": [role, i]}, zero))
            for k in range(10):
                out.append(mod.question_text({"kind": "count", "grid": [role, i], "colour": k}, zero))
        for k in range(3):
            out.append(mod.question_text({"kind": "diff", "grid": ["pair", k]}, zero))
    # without the answer-format tail too (the library never writes one, but be strict)
    return out + [s.split("? ")[0] + "?" for s in out if "? " in s]


PLURAL = re.compile(r"(?i)(?<![\w@(,-])(?<!color )(?<!row )(?<!rows )(?<!column )(?<!columns )(\d+) "
                    r"(cell|row|column|color(?! \d)|position)(s?)\b")     # "7 color 2 cells": a colour word


def check_texts(q, slots):
    """Statement form, placeholders, flags and hygiene of one posed question."""
    n = len(q.reads)
    refs = REFS[:n]
    assert q.source == "lib" and q.qid.startswith("lib:") and q.answer.kind in KINDS
    for bad in ("{", "}", "  ", " ?", " .", "input", "output", "below", "above", "nothing else"):
        assert bad not in q.text.lower() if bad.isalpha() else bad not in q.text, (bad, q.text)
    text = fill(q.text, refs)
    assert text[0].isupper(), text
    for i in range(n):
        uses = any(f"⟪{t}⟫" in q.text for t in (f"g{i}", f"G{i}", f"g{i}s"))   # what Part.validate_shape accepts
        assert uses, f"{q.qid} never names read {i}"
        assert any(f"⟪{t}⟫" in q.statement for t in (f"g{i}", f"G{i}", f"g{i}s", f"G{i}s")), \
            f"{q.qid}: the statement drops the reference to read {i}"
    if q.answer.kind == "grid":
        assert q.answer.text is None and "⟪answer⟫" not in q.statement and q.statement.endswith(":")
        stmt = fill(q.statement, refs)
    else:
        assert q.statement.count("⟪answer⟫") == 1, q.statement
        answer = fill(q.answer.text, refs)
        stmt = fill(q.statement, refs, answer=answer)
        assert "\n" not in answer
    assert stmt[0].isupper() and "{" not in stmt and "}" not in stmt, stmt
    for m in PLURAL.finditer(stmt):
        assert (int(m.group(1)) == 1) == (m.group(3) == ""), f"pluralisation: {stmt!r}"
    for m in re.finditer(r"\bThere (is|are) (\d+)", stmt):
        assert (m.group(1) == "is") == (m.group(2) == "1"), stmt
    for m in PLURAL.finditer(text):
        assert (int(m.group(1)) == 1) == (m.group(3) == ""), f"pluralisation: {text!r}"
    for pat in PROBE_PATTERNS:
        assert not pat.search(text) and not pat.search(stmt), (pat.pattern, text)
    # coordinates: flag <=> the text or the answer uses row / column numbers; flag => 0-indexed clause
    uses_coords = bool(re.search(r"\b(rows?|columns?) \d|\(\d+,\d+\)|\(r,c\)|row-major|\brow k|column k", text))
    if uses_coords:
        assert FLAG_COORDS in q.flags, (q.qid, text)
    if FLAG_COORDS in q.flags:
        assert any(z in text for z in ZERO), (q.qid, text)
    for w_ in ("image", "panel", "map", "figure", "picture", "left", "right", "top", "bottom"):
        if re.search(rf"\b{w_}s?\b", q.text.lower()):
            assert LABEL_WORD_PREFIX + w_ in q.flags, (w_, q.text)
    # encodings the question may not be shown with
    for i in q.reads:
        e = enc.ENCODINGS[slots[i].encoding]
        if FLAG_COORDS in q.flags:
            assert not e.reveals_coords or e.name == "stage2", (q.qid, e.name)
        if FLAG_DIMS in q.flags:     # labels r0..r(H-1) / c0..c(W-1) give the size away, stage2 included
            assert e.name not in ("stage2", "runs", "row_labelled", "indexed_table", "sparse"), (q.qid, e.name)
            assert not e.reveals_dims and not e.reveals_coords and not slots[i].dims_header_ok, (q.qid, e.name)
    if q.qid == "lib:runs_row":
        assert slots[q.reads[0]].encoding not in ("stage2", "runs", "names")
    if q.qid in ("lib:positions", "lib:run_length", "lib:run_end"):      # the runs listing prints them
        assert slots[q.reads[0]].encoding not in ("stage2", "runs")
    if q.qid in ("lib:most_frequent", "lib:each_most"):
        assert all(slots[i].encoding != "sparse" for i in q.reads)
    assert len({enc.ENCODINGS[slots[i].encoding].uses_names for i in q.reads}) == 1
    names = enc.ENCODINGS[slots[q.reads[0]].encoding].uses_names
    for key in ("X", "Y"):
        if key in q.params and q.qid.startswith("lib:") and not q.qid.startswith("lib:chain"):
            assert re.search(rf"(?<![\w-]){re.escape(word(q.params[key], names))}(?!\d)", text, re.I), (key, text)
    return text, stmt


BOUNDING = re.compile(r"\b(rows?|columns?) \d|\(\d+,\d+\)|\bhalf\b(?! turn)|\d+x\d+ block")


def bounds_size_text(text):
    """Whether a question text names an index, a half or a scale factor (each bounds the size)."""
    for z in ZERO_SENTENCES + ZERO_PARENS:
        text = text.replace(z, "")
    return bool(BOUNDING.search(text))


def check_no_size_hints(queries):
    """No size question shares a grid with a question whose text bounds that grid's size."""
    for q in queries:
        if FLAG_DIMS not in q.flags:
            continue
        for o in queries:
            if o is not q and set(o.reads) & set(q.reads):
                assert not bounds_size_text(o.text), (q.text, o.text)


def check_params_in_text(q, text):
    p = q.params
    if "r" in p and "c" in p and q.qid in ("lib:cell", "lib:run_length", "lib:run_end"):
        assert (re.search(rf"\brow {p['r']}\b", text) and re.search(rf"\bcolumn {p['c']}\b", text)) or \
            f"({p['r']},{p['c']})" in text, text
    elif "r" in p and q.qid in ("lib:row", "lib:runs_row", "lib:row_count", "lib:row_uniform"):
        assert re.search(rf"\brow {p['r']}\b", text), text
    elif "c" in p and q.qid in ("lib:col", "lib:col_count"):
        assert re.search(rf"\bcolumn {p['c']}\b", text), text
    if q.qid in ("lib:rows_range", "lib:cols_range", "lib:row_equal"):
        assert re.search(rf"\b{p['a']}\b.*\b{p['b']}\b", text), text
    if q.qid == "lib:crop":
        assert re.search(rf"\b{p['a']}\b.*\b{p['b']}\b.*\b{p['c0']}\b.*\b{p['c1']}\b", text), text
    if q.qid == "lib:corner":
        assert p["corner"] in text
    if q.qid == "lib:half":
        assert f"{p['half']} half" in text
    if q.qid == "lib:rotate":
        words = {"cw": "clockwise", "ccw": "counterclockwise", "180": ("180 degrees", "half turn")}[p["dir"]]
        words = (words,) if isinstance(words, str) else words
        assert any(re.search(rf"(?<!counter){w_}", text) for w_ in words), text
    if q.qid == "lib:reflect":
        assert ("left to right" if p["axis"] == "lr" else "top to bottom") in text
    if q.qid == "lib:upscale":
        assert f"{p['k']}x{p['k']}" in text
    if p.get("cap"):
        assert f"first {p['cap']} " in text
    if q.qid in ("lib:run_length",):
        assert ("horizontal" if p["orient"] == "h" else "vertical") in text
    if q.qid == "lib:run_end":
        assert ("which column" in text or "last column" in text) == (p["orient"] == "h")


def check_query(q, slots):
    """Answer (independent oracle), texts and flags of a library question."""
    assert isinstance(q, LibQuery)
    grids = [slots[i].grid for i in q.reads]
    names = enc.ENCODINGS[slots[q.reads[0]].encoding].uses_names
    value, text = expected(q, grids, names)
    assert q.answer.value == value, (q.qid, q.answer.value, value)
    assert q.answer.text == text, (q.qid, q.answer.text, text)
    qid = q.qid[4:]
    spec = QUESTIONS[qid]
    case = q.params["case"]
    if q.answer.kind == "bool":
        assert case == yn(q.answer.value)
    if case in ("one", "many"):
        assert (case == "one") == (len(q.answer.value) == 1)
    if case in ("found", "none"):
        assert (case == "none") == (q.answer.value in (None, []))
    if case == "none":
        assert q.answer.text == "none"
    assert q.answer.kind == spec.kind
    assert (FLAG_DIMS in q.flags) == (qid in DIMS_QIDS)
    assert (FLAG_EDIT in q.flags) == (qid in EDIT_QIDS)
    assert (q.answer.kind == "grid") == spec.grid_answer
    if FLAG_EDIT in q.flags:
        assert q.answer.value != grids[0]
    filled, _ = check_texts(q, slots)
    check_params_in_text(q, filled)


# --- grids for direct posing --------------------------------------------------------------------------------

def random_grid(rng, h=None, w=None):
    h = h or int(rng.integers(1, 31))
    w = w or int(rng.integers(1, 31))
    while h * w > 900:
        w = int(rng.integers(1, 31))
    k = int(rng.integers(1, 11))
    palette = [int(v) for v in rng.choice(10, size=k, replace=False)]
    g = np.full((h, w), palette[0])
    for _ in range(int(rng.integers(0, 8))):
        a, b = sorted(rng.integers(0, h, size=2))
        c, d = sorted(rng.integers(0, w, size=2))
        g[a:b + 1, c:d + 1] = palette[int(rng.integers(k))]
    noise = rng.random((h, w)) < rng.choice([0, 0.05, 0.3])
    g[noise] = rng.choice(palette, size=int(noise.sum()))
    if rng.random() < 0.15:
        g[:, w - w // 2:] = g[:, :w // 2][:, ::-1]
    if rng.random() < 0.1:
        g[h - h // 2:] = g[:h // 2][::-1]
    return g.tolist()


def edited(rng, g, k):
    g = [list(r) for r in g]
    for _ in range(k):
        r, c = int(rng.integers(len(g))), int(rng.integers(len(g[0])))
        g[r][c] = (g[r][c] + 1 + int(rng.integers(9))) % 10
    return g


def slots_for(rng, grids, *, names=None):
    names = rng.random() < 0.15 if names is None else names
    pool = [n for n in enc.ENCODINGS if n != "names"]
    return [Slot(g, "names" if names else pool[int(rng.integers(len(pool)))]) for g in grids]


SHAPES = [(1, 1), (1, 7), (7, 1), (2, 2), (30, 30), (4, 6), (30, 1), (1, 30), (3, 3)]


# --- tests ----------------------------------------------------------------------------------------------------

def test_catalogue_shape():
    assert len(QUESTIONS) >= 40
    for qid, spec in QUESTIONS.items():
        assert 2 <= len(spec.phrasings) <= 6, qid
        registers = {p.register for p in spec.phrasings}
        assert len(registers) >= 2, qid
        for p in spec.phrasings:
            st = p.statement if isinstance(p.statement, dict) else {"main": p.statement}
            for form in st.values():
                if spec.kind == "grid":
                    assert form.endswith(":") and "⟪answer⟫" not in form
                else:
                    assert form.count("⟪answer⟫") == 1
            assert "⟪answer⟫" not in p.text


def test_direct_pose_every_question_and_phrasing():
    """Each question with each of its phrasings, over random grids of every shape and encoding."""
    rng = np.random.default_rng(7)
    seen = Counter()
    posed = 0
    for qid, spec in QUESTIONS.items():
        for pi in range(len(spec.phrasings)):
            ok = 0
            for attempt in range(400):
                if ok >= 6:
                    break
                shape = SHAPES[attempt % len(SHAPES)] if attempt < 18 else (None, None)
                g = random_grid(rng, *shape)
                if spec.arity == 1:
                    grids = [g]
                elif spec.arity == 2:
                    other = rng.random()
                    grids = [g, edited(rng, g, int(rng.integers(0, 15))) if other < 0.7 else random_grid(rng)]
                else:
                    grids = [g] + [random_grid(rng) if rng.random() < 0.6 else edited(rng, g, 2)
                                   for _ in range(int(rng.integers(1, 5)))]
                slots = slots_for(rng, grids)
                q = pose(rng, slots, tuple(range(len(grids))), qid, phrasing=pi)
                if q is None:
                    continue
                check_query(q, slots)
                Part("multi_question", slots, [q]).validate_shape()
                ok += 1
                posed += 1
                seen[(qid, pi)] += 1
            assert ok, f"{qid} phrasing {pi} never posed"
    assert posed >= 1000


def test_library_parts():
    rng = np.random.default_rng(11)
    total = 0
    qids = Counter()
    for i in range(420):
        part = library_part(rng, n_questions=int(rng.integers(2, 7)))
        part.validate_shape()
        assert part.family == "multi_question" and len(part.slots) == 1 and 1 <= len(part.queries) <= 6
        assert len({q.qid for q in part.queries}) == len(part.queries)
        assert sum(FLAG_DIMS in q.flags for q in part.queries) <= 1
        slot = part.slots[0]
        assert slot.dims_header_ok == (not any(FLAG_DIMS in q.flags for q in part.queries))
        edits = any(FLAG_EDIT in q.flags for q in part.queries)
        assert (AS_SHOWN in part.context) == (edits and len(part.queries) > 1)
        check_no_size_hints(part.queries)
        asked = set().union(*(q.params["asks_presence"] for q in part.queries))
        presumed = set().union(*(q.params["presupposes"] for q in part.queries))
        assert not asked & presumed, "a question's text hands over another's presence answer"
        for q in part.queries:
            check_query(q, part.slots)
            qids[q.qid] += 1
            total += 1
    assert total >= 1500
    assert len(qids) >= 40, sorted(qids)


def _check_verification(part):
    src, prop = part.slots[0], part.slots[1]
    assert prop.role == "proposed"
    q1, q2 = part.queries[0], part.queries[1]
    p = q1.params
    res = apply_op(p, src.grid)
    is_right = prop.grid == res
    assert q1.qid == "lib:verify" and q1.answer.value == is_right and q1.answer.text == yn(is_right)
    assert (p["mistake"] is None) == is_right
    if not is_right:
        check_mistake(p, src.grid, p["mistake"], prop.grid)
    assert (H(prop.grid), W(prop.grid)) == (H(res), W(res))
    n = n_cells_differ(prop.grid, res)
    assert q2.qid == "lib:verify_diff" and q2.answer.value == n and q2.answer.text == str(n)
    for q in part.queries[2:]:
        assert q.qid == "lib:verify_redo" and q.answer.value == res and q.reads == (0,)
        assert (FLAG_EDIT in q.flags) == (p["op"] in ("rotate", "reflect", "transpose", "recolour"))
    for q in part.queries:
        assert (FLAG_COORDS in q.flags) == (p["op"] in ("row", "col", "crop"))
        check_texts(q, part.slots)


def _check_choice(part):
    q = part.queries[0]
    p = q.params
    src = part.slots[0]
    res = apply_op(p, src.grid)
    cands = part.slots[1:]
    assert all(s.role == "candidate" for s in cands) and 2 <= len(cands) <= 4
    if len(cands) == 2:          # only when three candidates would not fit the part's token budget
        assert q.params["dropped_for_budget"] > 0
    grids = [s.grid for s in cands]
    assert len({repr(g) for g in grids}) == len(grids), "candidates must be distinct"
    right = [i for i, g in enumerate(grids) if g == res]
    assert len(right) == 1
    slot_index = right[0] + 1
    assert q.answer.value == [slot_index] and q.answer.text == f"⟪g{slot_index}⟫"
    assert q.reads == tuple(range(len(part.slots)))
    assert p["candidates"][right[0]] == "correct"
    for name, g in zip(p["candidates"], grids):
        if name != "correct":
            check_mistake(p, src.grid, name, g)
    assert sum(n == "one_cell_off" for n in p["candidates"]) < len(cands) - 1
    check_texts(q, part.slots)
    for other in part.queries[1:]:
        check_query(other, part.slots)


def test_comparison_parts():
    rng = np.random.default_rng(5)
    modes = Counter()
    total = 0
    verdicts = Counter()
    for i in range(420):
        part = comparison_part(rng)
        part.validate_shape()
        assert part.family == "comparison" and len(part.slots) >= 2
        mode = part.prov["mode"]
        modes[mode] += 1
        check_no_size_hints(part.queries)
        uses = {enc.ENCODINGS[s.encoding].uses_names for s in part.slots}
        assert len(uses) == 1
        for i_, s in enumerate(part.slots):
            assert s.dims_header_ok == (not any(FLAG_DIMS in q.flags and i_ in q.reads for q in part.queries))
        if mode == "verify":
            _check_verification(part)
            verdicts[part.queries[0].answer.value] += 1
        elif mode == "choice":
            _check_choice(part)
        else:
            for q in part.queries:
                check_query(q, part.slots)
            if mode == "each" and len(part.queries) > 1:
                assert len({q.qid for q in part.queries}) == 1 and len(part.queries) == len(part.slots)
        total += len(part.queries)
    assert set(modes) == {"each", "cross", "which", "verify", "choice"}
    assert verdicts[True] > 20 and verdicts[False] > 20
    assert total >= 700


def _hop1(p, g):
    hop, hp = p["hop1"], p["hop1_params"]
    if hop == "rows":
        return H(g)
    if hop == "cols":
        return W(g)
    if hop == "n_colours":
        return len(tally(g))
    if hop == "count":
        return count(g, hp["X"])
    if hop == "n_rows_containing":
        return sum(1 for row in g if hp["X"] in row)
    if hop == "n_distinct_rows":
        return len({tuple(r) for r in g})
    if hop == "first_row_containing":
        return [r for r in range(H(g)) if hp["X"] in g[r]][0]
    if hop == "first_cell":
        return g[0][0]
    if hop == "last_cell":
        return g[-1][-1]
    if hop == "most_frequent":
        t = tally(g)
        winners = [v for v in t if t[v] == max(t.values())]
        assert len(winners) == 1
        return winners[0]
    if hop == "cell":
        return g[hp["r"]][hp["c"]]
    raise AssertionError(hop)


def test_chain_parts():
    rng = np.random.default_rng(9)
    total = 0
    hops = Counter()
    for i in range(420):
        part = chain_part(rng)
        part.validate_shape()
        assert part.family == "chain" and len(part.queries) == 2
        q1, q2 = part.queries
        p = q2.params
        g0 = part.slots[0].grid
        g1 = part.slots[q2.reads[-1]].grid
        check_query(q1, part.slots)                       # the intermediate value is asked on its own
        assert q1.qid == "lib:" + p["hop1"] and q1.reads == (0,)
        v = _hop1(p, g0)
        assert v == q1.answer.value
        hops[(p["hop1"], p["hop2"])] += 1
        hop2 = p["hop2"]
        if hop2 in ("row", "col", "top_rows"):
            k = v
            assert p["k"] == k
            if hop2 == "top_rows":
                assert 2 <= k < H(g1)
                want = [list(r) for r in g1[:k]]
            else:
                idx = k - 1 if p["expr"] == "k-1" else k
                assert p["idx"] == idx
                lines = g1 if hop2 == "row" else transpose(g1)
                assert 0 <= idx < len(lines) >= 2
                for j in (idx - 1, idx + 1):
                    assert not (0 <= j < len(lines)) or lines[j] != lines[idx], "chain off-by-one guard"
                want = [list(lines[idx])] if hop2 == "row" else [[x] for x in lines[idx]]
                assert re.search(rf"\b(row|column) {re.escape(p['expr'])}\b", q2.text)
            assert q2.answer.value == want
        else:
            x = v
            assert p["X"] == x
            names = enc.ENCODINGS[part.slots[0].encoding].uses_names
            if hop2 == "count":
                assert q2.answer.value == count(g1, x) and q2.answer.text == str(count(g1, x))
            elif hop2 == "has":
                assert q2.answer.value == (count(g1, x) > 0) and q2.answer.text == yn(count(g1, x) > 0)
                assert q2.reads == (0, 1)
            elif hop2 == "recolour":
                assert count(g1, x) > 0 and p["Y"] != x
                assert q2.answer.value == [[p["Y"] if c == x else c for c in row] for row in g1]
                assert word(p["Y"], names) in q2.text and FLAG_EDIT in q2.flags
            else:
                rows = [r for r in range(H(g1)) if x in g1[r]]
                assert rows and q2.answer.value == rows and q2.answer.text == ", ".join(map(str, rows))
        _, stmt = check_texts(q2, part.slots)
        assert str(v) in stmt or word(v, enc.ENCODINGS[part.slots[0].encoding].uses_names) in stmt, \
            "the statement shows the intermediate value"
        assert (FLAG_DIMS in q2.flags) == (p["hop1"] in ("rows", "cols"))
        if FLAG_EDIT in q2.flags and len(part.slots) == 1:
            assert AS_SHOWN in part.context
        total += 2
    assert total >= 800
    assert len(hops) >= 20


def test_probe_sentences_never_appear():
    mod = _probe_module()
    if mod is None:
        pytest.skip("stage2/probe_perception.py not present in this tree")
    banned = probe_sentences()
    rng = np.random.default_rng(3)
    texts = []
    for _ in range(150):
        for part in (library_part(rng, n_questions=4), comparison_part(rng), chain_part(rng)):
            for q in part.queries:
                for refs in (REFS, ["Training pair 1 input", "Training pair 1 output", "Test input 1"] + REFS):
                    r = refs[:len(q.reads)]
                    texts.append(fill(q.text, r))
                    if q.answer.kind != "grid":
                        texts.append(fill(q.statement, r, answer=fill(q.answer.text, r)))
    for t in texts:
        for s in banned:
            assert s not in t, (s, t)
    # the templates themselves, before any filling
    for spec in QUESTIONS.values():
        for p in spec.phrasings:
            for s in ("How many cells differ between", "What is the colour of the cell", "What is the color of the cell",
                      "What is the size of", "How many cells of colour", "How many cells of color", "(zero-based)"):
                assert s not in p.text


def test_determinism():
    def run(seed):
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(8):
            for part in (library_part(rng, n_questions=4), comparison_part(rng), chain_part(rng)):
                out.append(repr([(s.grid, s.encoding, s.role, s.dims_header_ok) for s in part.slots]))
                out.append(repr([(q.qid, q.reads, q.text, q.statement, q.answer, sorted(q.flags), q.params)
                                 for q in part.queries]))
                out.append(repr((part.context, part.prov)))
        return out
    assert run(21) == run(21)
    assert run(21) != run(22)


def test_pose_respects_encodings_and_exclude():
    rng = np.random.default_rng(1)
    g = random_grid(rng, 6, 8)
    assert pose(rng, [Slot(g, "row_labelled")], (0,), "cell") is None      # reveals coordinates
    assert pose(rng, [Slot(g, "sparse")], (0,), "size") is None            # reveals the size
    assert pose(rng, [Slot(g, "stage2")], (0,), "runs_row") is None        # prints the runs
    assert pose(rng, [Slot(g, "names")], (0,), "runs_row") is None         # digit notation
    two = [Slot(g, "names"), Slot(g, "compact")]
    assert pose(rng, two, (0, 1)) is None                                  # colour words would differ
    for labelled in ("stage2", "runs", "row_labelled", "indexed_table"):
        assert pose(rng, [Slot(g, labelled)], (0,), "rows") is None          # labels give the size
        assert pose(rng, [Slot(g, labelled)], (0,), "positions") is None or labelled not in ("stage2", "runs")
    assert pose(rng, [Slot(g, "stage2")], (0,), "cell") is not None         # stage-2 coordinates are allowed
    s = Slot(g, "compact")
    q = pose(rng, [s], (0,), "rows")
    assert q is not None and not s.dims_header_ok
    for _ in range(50):
        q = pose(rng, [Slot(g, "compact")], (0,), exclude={"lib:" + k for k in QUESTIONS if k != "count"})
        assert q is None or q.qid == "lib:count"
    with pytest.raises(ValueError):
        pose(rng, [Slot(g)], (0, 0))
    with pytest.raises(KeyError):
        pose(rng, [Slot(g)], (0,), "no_such_question")


def test_colour_words_follow_the_encoding():
    g = [[3, 3, 0], [0, 5, 5]]
    rng = np.random.default_rng(0)
    for encoding, x_word, answer in (("compact", "color 3", "3"), ("names", "green", "green")):
        q = pose(rng, [Slot(g, encoding)], (0,), "first_cell")
        assert q.answer.text == answer and q.answer.value == 3
        assert fill(q.statement, ["grid A"], answer=answer).endswith(("is " + x_word + ".", x_word + "."))


def test_library_part_uses_a_given_grid():
    rng = np.random.default_rng(4)
    g = [[1, 1, 2], [0, 2, 2], [0, 0, 7], [3, 3, 3]]
    for _ in range(30):
        part = library_part(rng, n_questions=5, grid=g)
        assert part.slots[0].grid == g and 1 <= len(part.queries) <= 5
        for q in part.queries:
            check_query(q, part.slots)
    with pytest.raises(ValueError):
        library_part(rng, n_questions=0)


def test_pluralisation_examples():
    rng = np.random.default_rng(0)
    one = [[1, 0, 0]]
    q = pose(rng, [Slot(one)], (0,), "rows", phrasing=0)
    assert fill(q.statement, ["the grid"], answer=q.answer.text) == "The grid has 1 row."
    zero = [[0, 0], [0, 0]]
    for _ in range(40):
        q = pose(rng, [Slot(zero)], (0,), "count", phrasing=0)
        if q.answer.value == 0:
            assert fill(q.statement, ["the grid"], answer="0").startswith("There are 0 color ")
            assert fill(q.statement, ["the grid"], answer="0").endswith(" cells in the grid.")
            break
    else:
        pytest.fail("no absent-colour count drawn")


def test_every_count_question_says_it_wants_a_number():
    """Red-team finding (2026-09-29): terse count phrasings such as "rows with X in them?" read as
    "list the rows" (target 1, a careful reader answers 12).  Every phrasing of a question whose
    answer is a count (or a list of counts) must say so."""
    from pv2 import library as L
    # a count says so; a size, a length or an index says which ("height", "how long", "which column")
    markers = ("how many", "number of", "count", "numbers of", "height", "width", "how long", "length",
               "in cells", "which column", "which row", "last column", "last row")
    bad = []
    for spec in L.QUESTIONS.values():
        for ph in spec.phrasings:
            statement = (ph.statement or "") if isinstance(ph.statement, str) else ""
            counts = spec.kind == "number" or "numbers of" in statement.lower() or "number of" in statement.lower()
            if counts and not any(m in ph.text.lower() for m in markers):
                bad.append((spec.qid, ph.register, ph.text))
    assert not bad, bad


# --- review findings (2026-09-29, libfix) ------------------------------------------------------------------

def test_upside_down_flip_is_always_defined():
    """"Flip it upside down" alone can be read as a 180-degree turn (the answer differed on 7/79
    draws); every wording that says it also says the rows are taken in reverse order."""
    texts = [p.text for s in QUESTIONS.values() for p in s.phrasings]
    for t in texts:
        if "upside down" in t:
            assert re.search(r"rows in reverse order|order of the rows|rows reversed|reverse order", t), t
    rng = np.random.default_rng(60)                  # the reported repro
    for _ in range(300):
        for q in library_part(rng, n_questions=4).queries:
            if "upside down" in q.text:
                assert "reverse order" in q.text or "order of the rows" in q.text, q.text


SAME_SIZE_ONLY = {"lib:n_diff", "lib:diff_cells", "lib:diff_rows", "lib:first_diff"}
DIFFERING_ONLY = {"lib:diff_cells", "lib:more_colour", "lib:larger"}


def test_cross_parts_never_imply_same_size_or_identical():
    """same_size next to a question asked only of same-size pairs is always "yes" (155/3000 cross
    parts); identical next to one asked only of differing grids is always "no"."""
    rng = np.random.default_rng(16)                  # the reported repro
    seen = Counter()
    for _ in range(600):
        part = comparison_part(rng, mode="cross")
        ids = {q.qid for q in part.queries}
        assert not ("lib:same_size" in ids and ids & SAME_SIZE_ONLY), ids
        assert not ("lib:identical" in ids and ids & DIFFERING_ONLY), ids
        seen.update(ids)
        for q in part.queries:
            check_query(q, part.slots)
    # the questions themselves are still asked, just not in those pairs
    assert seen["lib:same_size"] > 10 and seen["lib:identical"] > 50 and seen["lib:diff_cells"] > 30


def test_reflect_never_next_to_symmetry_on_its_axis():
    """A reflection is never the identity, so it implies "not symmetric" on its axis."""
    rng = np.random.default_rng(60)
    pairs = Counter()
    for _ in range(800):
        part = library_part(rng, n_questions=6)
        for q in part.queries:
            if q.qid == "lib:reflect":
                others = {o.qid for o in part.queries}
                assert "lib:symmetric_" + q.params["axis"] not in others, (q.text, others)
                pairs["reflect"] += 1
                pairs["other axis"] += ("lib:symmetric_" + {"lr": "tb", "tb": "lr"}[q.params["axis"]]) in others
    for _ in range(400):
        part = comparison_part(rng, mode="choice")
        choice = part.queries[0]
        if choice.params["op"] == "reflect":
            assert all(o.qid != "lib:symmetric_" + choice.params["axis"] for o in part.queries[1:])
    assert pairs["reflect"] > 60 and pairs["other axis"] > 0      # the other axis stays allowed


def test_present_colour_counts_presuppose_presence():
    """count_not / row_count / col_count only name colours present in the grid, so they are kept away
    from questions asking whether (or how often) that colour occurs on the same grid."""
    rng = np.random.default_rng(2)
    for qid in ("count_not", "row_count", "col_count"):
        for _ in range(30):
            q = pose(rng, [Slot(random_grid(rng, 8, 9))], (0,), qid)
            if q is not None:
                assert q.params["presupposes"] == [q.params["X"]], qid
    asks = {"lib:has_colour", "lib:count", "lib:first_row_containing", "lib:n_rows_containing"}
    rng = np.random.default_rng(60)
    for _ in range(800):
        part = library_part(rng, n_questions=6)
        for q in part.queries:
            if q.qid in ("lib:count_not", "lib:row_count", "lib:col_count"):
                assert not any(o.qid in asks and o.params.get("X") == q.params["X"] for o in part.queries)


def test_review_wordings():
    corner = QUESTIONS["corner"].phrasings
    assert all("color" in p.text for p in corner)
    assert all("number of" in p.text or "Which is" in p.text or "Give the number" in p.text
               for p in QUESTIONS["first_row_containing"].phrasings)
    for p in QUESTIONS["which_lack"].phrasings:
        assert "None of {refs} has no" not in p.statement["none"]
        assert p.statement["none"].startswith("Each of {refs} has at least one")


# --- review findings (2026-09-30, libfix2): yes/no balance, sizes, duplicates, wording, dominant colours

def _share(values):
    values = list(values)
    return sum(values) / max(len(values), 1)


def test_yes_no_questions_are_balanced_on_library_pictures():
    """Borrowed pictures are ~4% symmetric per axis and a third have no single-colour row, so these
    questions were 10-34% yes; the samplers keep near misses and few clearly-off "no"s, and library
    pictures include symmetric pictures and near misses."""
    from pv2.library.parts import _picture, symmetry_variant
    rng = np.random.default_rng(3)
    answers, near = {q: [] for q in ("symmetric_lr", "symmetric_tb", "row_uniform", "row_equal")}, Counter()
    for _ in range(1500):
        g = symmetry_variant(rng, _picture(rng)[0])[0]
        for qid in answers:
            q = pose(rng, [Slot(g)], (0,), qid)
            if q is not None:
                answers[qid].append(bool(q.answer.value))
                if qid.startswith("symmetric") and not q.answer.value:
                    near[qid] += q.params["mismatched_cells"] <= 8
    for qid, vals in answers.items():
        assert len(vals) > 200 and 0.40 <= _share(vals) <= 0.60, (qid, len(vals), _share(vals))
    for qid in ("symmetric_lr", "symmetric_tb"):         # the "no"s are mostly near misses, not far-off grids
        assert near[qid] >= 0.4 * answers[qid].count(False), (qid, near[qid])


def test_near_miss_pictures():
    from pv2.library.parts import near_miss, symmetrise
    rng = np.random.default_rng(4)
    for _ in range(400):
        g = random_grid(rng)
        sym, axis = symmetrise(rng, g)
        if axis is None:
            continue
        m = near_miss(rng, sym, axis)
        mirror = flip_lr(m) if axis == "lr" else flip_tb(m)
        off = n_cells_differ(m, mirror)
        assert 2 <= off <= 8 and off % 2 == 0, off               # one cell or one run of 2-4 cells, mirrored
        assert (H(m), W(m)) == (H(g), W(g))


def test_identical_and_same_size_are_balanced_in_cross_parts():
    rng = np.random.default_rng(12)
    got = {"lib:identical": [], "lib:same_size": []}
    near_no = 0
    for _ in range(500):
        part = comparison_part(rng, mode="cross")
        for q in part.queries:
            if q.qid in got:
                got[q.qid].append(bool(q.answer.value))
                check_query(q, part.slots)
                if q.qid == "lib:identical" and not q.answer.value and part.prov["pair"] == "near_edit":
                    near_no += 1
    for qid, vals in got.items():
        assert len(vals) > 40 and 0.40 <= _share(vals) <= 0.60, (qid, len(vals), _share(vals))
    assert near_no >= 0.3 * got["lib:identical"].count(False)    # "no" is often a one-to-three-cell near miss


def test_library_pictures_follow_at_least_the_natural_size_mix():
    """Library parts used to cap pictures (4-14% of grids at 400+ cells in comparisons); now every
    picture follows the natural mix tilted up (~40% at 400+), never capped."""
    from pv2.pictures import sample_picture
    rng = np.random.default_rng(6)
    big = Counter()
    for _ in range(300):
        for name, part in (("library", library_part(rng, n_questions=3)), ("chain", chain_part(rng)),
                           ("each", comparison_part(rng, mode="each")), ("cross", comparison_part(rng, mode="cross"))):
            for s_ in part.slots:
                big[name] += H(s_.grid) * W(s_.grid) >= 400
                big[name + " n"] += 1
    natural = _share(len(g) * len(g[0]) >= 400 for g in (sample_picture(rng)[0] for _ in range(1500)))
    for name in ("library", "chain", "each", "cross"):
        assert big[name] / big[name + " n"] >= natural, (name, big[name] / big[name + " n"], natural)


def test_parts_fit_the_token_budget():
    """Many-grid parts show fewer grids / candidates (or compact grids) instead of capping sizes: the
    grids and grid answers of a part stay within ~12k real tokens."""
    try:
        from pv2.tokens import load_tokenizer
        tok = load_tokenizer()
    except Exception:
        pytest.skip("tokenizer not available")
    rng = np.random.default_rng(7)
    worst = 0
    for _ in range(120):
        for part in (comparison_part(rng, mode="choice"), comparison_part(rng, mode="each"),
                     comparison_part(rng, mode="which"), comparison_part(rng, mode="verify"), chain_part(rng),
                     library_part(rng, n_questions=4)):
            n = sum(len(tok.encode(enc.render(s_.grid, s_.encoding), add_special_tokens=False)) for s_ in part.slots)
            n += sum(len(tok.encode(enc.render(q.answer.value, "compact"), add_special_tokens=False))
                     for q in part.queries if q.answer.kind == "grid")
            worst = max(worst, n)
    assert worst <= 12000, worst


def _read_window(q, g):
    """What a reading question reads, computed here from its id and parameters."""
    p, h, w, k = q.params, H(g), W(g), q.qid[4:]
    corners = {"top-left": (0, 0), "top-right": (0, w - 1), "bottom-left": (h - 1, 0), "bottom-right": (h - 1, w - 1)}
    if k == "cell":
        return (p["r"], p["r"], p["c"], p["c"])
    if k in ("first_cell", "last_cell", "corner"):
        r, c = {"first_cell": (0, 0), "last_cell": (h - 1, w - 1)}.get(k) or corners[p["corner"]]
        return (r, r, c, c)
    if k in ("row", "runs_row", "row_uniform"):
        return (p["r"], p["r"], 0, w - 1)
    if k == "col":
        return (0, h - 1, p["c"], p["c"])
    if k in ("rows_range", "cols_range"):
        return (p["a"], p["b"], 0, w - 1) if k == "rows_range" else (0, h - 1, p["a"], p["b"])
    if k == "crop":
        return (p["a"], p["b"], p["c0"], p["c1"])
    if k == "half":
        return {"top": (0, h // 2 - 1, 0, w - 1), "bottom": (h // 2, h - 1, 0, w - 1),
                "left": (0, h - 1, 0, w // 2 - 1), "right": (0, h - 1, w // 2, w - 1)}[p["half"]]
    return None


def test_a_part_never_asks_the_same_window_twice():
    """A part asked the bottom-right colour twice (last_cell and corner bottom-right); likewise
    first_cell / corner top-left / cell (0,0), row r / runs_row r, a crop equal to a row range."""
    from pv2 import families as F
    rng = np.random.default_rng(10)
    pairs = Counter()
    for i in range(1500):
        part = library_part(rng, n_questions=6) if i % 3 else F.one_grid_part(rng, n_questions=6)
        seen = {}
        for q in part.queries:
            if not q.qid.startswith("lib:") or len(q.reads) != 1:
                continue
            wdw = _read_window(q, part.slots[q.reads[0]].grid)
            if wdw is None:
                continue
            key = (q.reads[0], wdw)
            assert key not in seen, (seen.get(key), q.qid, wdw)
            seen[key] = q.qid
            pairs[q.qid] += 1
    assert pairs["lib:corner"] > 50 and pairs["lib:first_cell"] > 50 and pairs["lib:runs_row"] > 30


def test_verification_and_choice_read_naturally_with_any_label():
    rng = np.random.default_rng(13)
    labels = ["Grid A", "Proposed grid", "Candidate 1", "Candidate 2", "Candidate 3", "Candidate 4"]
    for _ in range(300):
        part = comparison_part(rng, mode=pick_mode(rng))
        for q in part.queries:
            text = fill(q.text, labels[:len(q.reads)])
            assert "proposed Proposed" not in text and "Someone" not in text, text
            assert not re.search(r"\b(\w+) \1\b", text, re.I), text          # no "proposed Proposed grid"


def pick_mode(rng):
    return ("verify", "choice")[int(rng.integers(2))]


def test_location_questions_avoid_dominant_colours():
    rng = np.random.default_rng(14)
    location = {"lib:bbox", "lib:positions", "lib:rows_containing", "lib:first_row_containing", "lib:n_rows_containing"}
    n = 0
    for _ in range(600):
        part = library_part(rng, n_questions=6)
        g = part.slots[0].grid
        for q in part.queries:
            if q.qid in location:
                # 0.6 * (H * W), as the sampler computes it: (0.6 * 6) * 15 rounds below 54, the 60% edge
                assert count(g, q.params["X"]) <= 0.6 * (H(g) * W(g)), (q.qid, q.params["X"])
                n += 1
    assert n > 300
    dominant = [[5] * 10 for _ in range(10)]
    dominant[3][4] = dominant[7][1] = 2
    for qid in ("bbox", "positions", "rows_containing", "first_row_containing", "n_rows_containing"):
        for _ in range(20):
            q = pose(rng, [Slot(dominant)], (0,), qid)
            assert q is None or q.params["X"] != 5, qid
