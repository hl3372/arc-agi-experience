"""Questions over two grids (identical? how many cells differ? which has more X?) and over several
grids ("for each of ... " with a list answer, "which of ... " with an answer naming grids)."""
from __future__ import annotations

import numpy as np

from pv2.library.core import Draw, Phrasing as P, Spec, colour_counts, inflect, pick, ref_list
from pv2.library.single import YN, _yn


def _same_shape(ctx) -> bool:
    return ctx.grids[0].shape == ctx.grids[1].shape


def _named(ctx, positions) -> tuple[str, list]:
    """Answer text naming grids by placeholder (reads positions) and the slot indices."""
    if not positions:
        return "none", []
    return ref_list(positions), [ctx.reads[i] for i in positions]


# --- two grids -------------------------------------------------------------------------------------

def s_identical(rng, ctx):
    a, b = ctx.grids[:2]
    flag = a.shape == b.shape and bool(np.array_equal(a, b))
    text, case = _yn(flag)
    return Draw(flag, text, case=case)


def s_n_diff(rng, ctx):
    if not _same_shape(ctx):
        return None
    n = int((ctx.grids[0] != ctx.grids[1]).sum())
    return Draw(n, str(n), fmt=inflect(n))


def s_diff_cells(rng, ctx):
    if not _same_shape(ctx):
        return None
    a, b = ctx.grids[:2]
    cells = [(r, c) for r in range(a.shape[0]) for c in range(a.shape[1]) if a[r, c] != b[r, c]]
    if not cells:
        return None
    if len(cells) <= 10:
        tag, n, case = "all", len(cells), ("one" if len(cells) == 1 else "many")
    else:
        tag, n, case = "cap", pick(rng, (3, 4, 5, 6, 8)), "many"
        cells = cells[:n]
    value = [[r, c, int(a[r, c]), int(b[r, c])] for r, c in cells]
    text = ", ".join(f"({r},{c}):{ctx.label(o)}->{ctx.label(v)}" for r, c, o, v in value)
    return Draw(value, text, tag=tag, case=case, fmt={"n": n}, params={"cap": n if tag == "cap" else None})


def s_more(rng, ctx):
    a, b = ctx.grids[:2]
    ca, cb = colour_counts(a), colour_counts(b)
    options = [x for x in sorted(set(ca) | set(cb)) if ca.get(x, 0) != cb.get(x, 0)]
    if not options:
        return None
    x = pick(rng, options)
    winner = 0 if ca.get(x, 0) > cb.get(x, 0) else 1
    text, value = _named(ctx, [winner])
    return Draw(value, text, fmt={"X": ctx.cw(x), "Xc": ctx.cwc(x)}, params={"X": x})


def s_same_size(rng, ctx):
    flag = ctx.grids[0].shape == ctx.grids[1].shape
    text, case = _yn(flag)
    return Draw(flag, text, case=case)


def s_larger(rng, ctx):
    a, b = ctx.grids[:2]
    if a.size == b.size:
        return None
    text, value = _named(ctx, [0 if a.size > b.size else 1])
    return Draw(value, text)


def s_diff_rows(rng, ctx):
    if not _same_shape(ctx):
        return None
    a, b = ctx.grids[:2]
    rows = [r for r in range(a.shape[0]) if not np.array_equal(a[r], b[r])]
    if not rows:
        return Draw([], "none", case="none")
    return Draw(rows, ", ".join(map(str, rows)), case="one" if len(rows) == 1 else "many")


def s_first_diff(rng, ctx):
    if not _same_shape(ctx):
        return None
    a, b = ctx.grids[:2]
    cells = np.argwhere(a != b)
    if len(cells) == 0:
        return Draw(None, "none", case="none")
    r, c = (int(v) for v in cells[0])        # argwhere is row-major
    return Draw([r, c], f"({r},{c})", case="found")


PAIR_SPECS = [
    Spec("identical", 2, "bool", s_identical, (
        P("interrogative", "Are ⟪g0⟫ and ⟪g1⟫ identical, with the same size and the same color in every cell?",
          YN("⟪G0⟫ and ⟪g1⟫ are identical", "⟪G0⟫ and ⟪g1⟫ are not identical"),
          declared="Asked whether ⟪g0⟫ and ⟪g1⟫ are identical, the answer is ⟪answer⟫."),
        P("request", "Tell me whether ⟪g1⟫ is an exact copy of ⟪g0⟫.",
          YN("⟪G1⟫ is an exact copy of ⟪g0⟫", "⟪G1⟫ is not an exact copy of ⟪g0⟫"),
          declared="On whether ⟪g1⟫ is an exact copy of ⟪g0⟫, the answer is ⟪answer⟫."),
        P("terse", "⟪G0⟫ and ⟪g1⟫: exactly the same?",
          YN("⟪G0⟫ and ⟪g1⟫ are exactly the same", "⟪G0⟫ and ⟪g1⟫ are not exactly the same"),
          declared="The answer to whether ⟪g0⟫ and ⟪g1⟫ are exactly the same is ⟪answer⟫."),
    )),
    Spec("n_diff", 2, "number", s_n_diff, (
        P("interrogative", "In how many positions do ⟪g0⟫ and ⟪g1⟫ have different colors?",
          "⟪G0⟫ and ⟪g1⟫ have different colors in ⟪answer⟫ {positions}.",
          declared="The number of positions in which ⟪g0⟫ and ⟪g1⟫ have different colors is ⟪answer⟫."),
        P("imperative", "Compare ⟪g0⟫ with ⟪g1⟫ cell by cell and count the cells whose colors disagree.",
          "Compared cell by cell, ⟪g0⟫ and ⟪g1⟫ disagree in ⟪answer⟫ {cells}.",
          declared="Compared cell by cell, the number of cells where the colors of ⟪g0⟫ and ⟪g1⟫ disagree is "
                   "⟪answer⟫."),
        P("terse", "⟪G0⟫ vs ⟪g1⟫: number of mismatched cells?",
          "The number of mismatched cells between ⟪g0⟫ and ⟪g1⟫ is ⟪answer⟫."),
        P("request", "Please count the cells of ⟪g1⟫ whose color differs from the cell in the same place in ⟪g0⟫.",
          "The cells of ⟪g1⟫ whose color differs from the cell in the same place in ⟪g0⟫ number ⟪answer⟫."),
        P("embedded", "Lay ⟪g1⟫ over ⟪g0⟫ cell for cell and tell me in how many places the two colors disagree.",
          "Laid over each other cell for cell, ⟪g1⟫ and ⟪g0⟫ disagree in ⟪answer⟫ {positions}.",
          declared="With ⟪g1⟫ laid over ⟪g0⟫ cell for cell, the number of places where the two colors disagree is "
                   "⟪answer⟫."),
    ), weight=1.5),
    Spec("diff_cells", 2, "text", s_diff_cells, (
        P("imperative", "List every cell where ⟪g1⟫ differs from ⟪g0⟫, in row-major order (top to bottom, then left "
                        "to right), as (r,c):old->new with old the color in ⟪g0⟫ and new the color in ⟪g1⟫, "
                        "separated by commas. {ZS}",
          {"many": "The cells where ⟪g1⟫ differs from ⟪g0⟫ are ⟪answer⟫.",
           "one": "The only cell where ⟪g1⟫ differs from ⟪g0⟫ is ⟪answer⟫."}, tag="all",
          declared="The cells where ⟪g1⟫ differs from ⟪g0⟫ are: ⟪answer⟫."),
        P("interrogative", "Which cells changed from ⟪g0⟫ to ⟪g1⟫? Give each as (r,c):old->new, with old from ⟪g0⟫ "
                           "and new from ⟪g1⟫, in row-major order (top to bottom, then left to right), separated by "
                           "commas. {ZS}",
          {"many": "The cells that changed from ⟪g0⟫ to ⟪g1⟫ are ⟪answer⟫.",
           "one": "The only cell that changed from ⟪g0⟫ to ⟪g1⟫ is ⟪answer⟫."}, tag="all",
          declared="The cells that changed from ⟪g0⟫ to ⟪g1⟫ are: ⟪answer⟫."),
        P("imperative", "List the first {n} of the cells where ⟪g1⟫ differs from ⟪g0⟫, taking them in row-major order "
                        "(top to bottom, then left to right), as (r,c):old->new with old the color in ⟪g0⟫ and new "
                        "the color in ⟪g1⟫, separated by commas. {ZS}",
          "The first {n} of the cells where ⟪g1⟫ differs from ⟪g0⟫ are ⟪answer⟫.", tag="cap"),
        P("terse", "From ⟪g0⟫ to ⟪g1⟫: the first {n} changed cells in row-major order, as (r,c):old->new {ZP}?",
          "From ⟪g0⟫ to ⟪g1⟫, the first {n} changed cells are ⟪answer⟫.", tag="cap"),
    ), coords=True),
    Spec("more_colour", 2, "token", s_more, (
        P("interrogative", "Which has more {X} cells, ⟪g0⟫ or ⟪g1⟫?", "Of ⟪g0⟫ and ⟪g1⟫, the one with more {X} cells is ⟪answer⟫."),
        P("imperative", "Name whichever of ⟪g0⟫ and ⟪g1⟫ contains more {X} cells (they do not tie).",
          "The one of ⟪g0⟫ and ⟪g1⟫ that contains more {X} cells is ⟪answer⟫."),
        P("terse", "More {X}: ⟪g0⟫ or ⟪g1⟫?", "Between ⟪g0⟫ and ⟪g1⟫, more {X} is found in ⟪answer⟫."),
    )),
    Spec("same_size", 2, "bool", s_same_size, (
        P("interrogative", "Do ⟪g0⟫ and ⟪g1⟫ have the same number of rows and the same number of columns?",
          YN("⟪G0⟫ and ⟪g1⟫ have the same number of rows and the same number of columns",
             "⟪G0⟫ and ⟪g1⟫ differ in size"),
          declared="Asked whether ⟪g0⟫ and ⟪g1⟫ have the same number of rows and the same number of columns, the "
                   "answer is ⟪answer⟫."),
        P("request", "Tell me whether ⟪g0⟫ and ⟪g1⟫ are the same size.",
          YN("⟪G0⟫ and ⟪g1⟫ are the same size", "⟪G0⟫ and ⟪g1⟫ are not the same size"),
          declared="On whether ⟪g0⟫ and ⟪g1⟫ are the same size, the answer is ⟪answer⟫."),
        P("terse", "⟪G0⟫ and ⟪g1⟫: same size?", YN("⟪G0⟫ and ⟪g1⟫ have the same size", "⟪G0⟫ and ⟪g1⟫ do not have the same size"),
          declared="The answer to whether ⟪g0⟫ and ⟪g1⟫ have the same size is ⟪answer⟫."),
    ), dims=True, weight=0.6),
    Spec("larger", 2, "token", s_larger, (
        P("interrogative", "Which has more cells in total, ⟪g0⟫ or ⟪g1⟫?", "Of ⟪g0⟫ and ⟪g1⟫, the one with more cells in total is ⟪answer⟫."),
        P("imperative", "Name whichever of ⟪g0⟫ and ⟪g1⟫ covers more cells (the totals differ).",
          "Of ⟪g0⟫ and ⟪g1⟫, ⟪answer⟫ covers more cells."),
        P("terse", "More cells: ⟪g0⟫ or ⟪g1⟫?", "Between ⟪g0⟫ and ⟪g1⟫, more cells are in ⟪answer⟫."),
    ), dims=True, weight=0.6),
    Spec("diff_rows", 2, "list", s_diff_rows, (
        P("interrogative", "In which rows do ⟪g0⟫ and ⟪g1⟫ differ? List the row numbers in increasing order, "
                           "separated by commas; if they match everywhere, the answer is none. {ZS}",
          {"many": "⟪G0⟫ and ⟪g1⟫ differ in rows ⟪answer⟫.", "one": "⟪G0⟫ and ⟪g1⟫ differ only in row ⟪answer⟫.",
           "none": "⟪G0⟫ and ⟪g1⟫ differ in no row, so the answer is ⟪answer⟫."},
          declared="The rows in which ⟪g0⟫ and ⟪g1⟫ differ are: ⟪answer⟫."),
        P("imperative", "List the rows, in increasing order and separated by commas, where ⟪g1⟫ does not match "
                        "⟪g0⟫ cell for cell, or none if every row matches. {ZS}",
          {"many": "The rows where ⟪g1⟫ does not match ⟪g0⟫ are rows ⟪answer⟫.",
           "one": "The only row where ⟪g1⟫ does not match ⟪g0⟫ is row ⟪answer⟫.",
           "none": "Every row of ⟪g1⟫ matches ⟪g0⟫, so the answer is ⟪answer⟫."},
          declared="The rows where ⟪g1⟫ does not match ⟪g0⟫ cell for cell are: ⟪answer⟫."),
    ), coords=True, weight=0.8),
    Spec("first_diff", 2, "text", s_first_diff, (
        P("interrogative", "Scanning in row-major order (top to bottom, then left to right), where is the first cell "
                           "at which ⟪g0⟫ and ⟪g1⟫ differ? Give it as (r,c), or none if they match everywhere. {ZS}",
          {"found": "The first cell, in row-major order, at which ⟪g0⟫ and ⟪g1⟫ differ is ⟪answer⟫.",
           "none": "⟪G0⟫ and ⟪g1⟫ match everywhere, so the answer is ⟪answer⟫."},
          declared="The first cell, in row-major order, at which ⟪g0⟫ and ⟪g1⟫ differ is ⟪answer⟫."),
        P("terse", "⟪G0⟫ vs ⟪g1⟫: first differing cell in row-major order, as (r,c) {ZP}, or none if identical?",
          {"found": "The first cell where ⟪g0⟫ and ⟪g1⟫ differ is ⟪answer⟫.",
           "none": "⟪G0⟫ and ⟪g1⟫ are identical, so the answer is ⟪answer⟫."},
          declared="The first cell where ⟪g0⟫ and ⟪g1⟫ differ is ⟪answer⟫."),
    ), coords=True, weight=0.8),
]


# --- several grids: "for each of ..." -------------------------------------------------------------------

def _refs_fmt(n):
    refs = ref_list(range(n))
    return {"refs": refs, "Refs": "⟪G0⟫" + refs[len("⟪g0⟫"):], "nrefs": n}


def _each(value_fn, text_fn, *, colour=None, all_no_keep=1.0):
    """Sampler for "for each grid, <base question>"; ``colour`` draws a colour X shared by all grids;
    ``all_no_keep``: share of all-"no" lists kept (yes/no balance of rare properties)."""
    def sample(rng, ctx):
        fmt = _refs_fmt(len(ctx.grids))
        params = {}
        x = None
        if colour is not None:
            union = sorted(set().union(*(colour_counts(g) for g in ctx.grids)))
            x = pick(rng, union)
            fmt.update(X=ctx.cw(x), Xc=ctx.cwc(x))
            params["X"] = x
        values = []
        for g in ctx.grids:
            v = value_fn(g, x)
            if v is None:
                return None
            values.append(v)
        if all_no_keep < 1.0 and not any(values) and rng.random() >= all_no_keep:
            return None
        return Draw(values, ", ".join(text_fn(ctx, v) for v in values), fmt=fmt, params=params)
    return sample


def _unique_most(g, _x):
    counts = colour_counts(g)
    top = max(counts.values())
    winners = [v for v, n in counts.items() if n == top]
    return winners[0] if len(winners) == 1 else None


_num = lambda ctx, v: str(v)                         # noqa: E731
_col = lambda ctx, v: ctx.label(v)                   # noqa: E731
_yes = lambda ctx, v: "yes" if v else "no"           # noqa: E731
_size = lambda ctx, v: f"{v[0]}x{v[1]}"              # noqa: E731


def _each_phr(imp, inter, terse, statement):
    return (P("imperative", imp, statement), P("interrogative", inter, statement), P("terse", terse, statement))


EACH_SPECS = [
    Spec("each_rows", "n", "list", _each(lambda g, x: g.shape[0], _num), _each_phr(
        "Give the number of rows of each of {refs}, in that order, separated by commas.",
        "How many rows does each of {refs} have? Give the numbers in the order listed, separated by commas.",
        "Number of rows of each of {refs}, in order, comma-separated?",
        "The numbers of rows of {refs}, in that order, are ⟪answer⟫."), dims=True),
    Spec("each_cols", "n", "list", _each(lambda g, x: g.shape[1], _num), _each_phr(
        "Give the number of columns of each of {refs}, in that order, separated by commas.",
        "How many columns does each of {refs} have? Give the numbers in the order listed, separated by commas.",
        "Number of columns of each of {refs}, in order, comma-separated?",
        "The numbers of columns of {refs}, in that order, are ⟪answer⟫."), dims=True),
    Spec("each_size", "n", "list", _each(lambda g, x: [g.shape[0], g.shape[1]], _size), _each_phr(
        "Give the size of each of {refs} as rows x columns (like 4x6), in that order, separated by commas.",
        "What are the dimensions of {refs}, each written as rows x columns (like 4x6)? Keep the order listed and "
        "separate them with commas.",
        "Sizes of {refs} as rows x columns, in order, comma-separated?",
        "The sizes of {refs} in rows x columns, in that order, are ⟪answer⟫."), dims=True),
    Spec("each_n_cells", "n", "list", _each(lambda g, x: int(g.size), _num), _each_phr(
        "Give the total number of cells of each of {refs}, in that order, separated by commas.",
        "How many cells does each of {refs} have in total? Keep the order listed and separate the numbers with "
        "commas.",
        "Number of cells in each of {refs}, in order, comma-separated?",
        "The total numbers of cells of {refs}, in that order, are ⟪answer⟫."), dims=True, weight=0.5),
    Spec("each_n_colours", "n", "list", _each(lambda g, x: len(colour_counts(g)), _num), _each_phr(
        "Give the number of different colors in each of {refs}, in that order, separated by commas.",
        "How many different colors does each of {refs} use? Keep the order listed and separate the numbers with "
        "commas.",
        "Number of distinct colors in each of {refs}, in order, comma-separated?",
        "The numbers of different colors in {refs}, in that order, are ⟪answer⟫.")),
    Spec("each_count", "n", "list", _each(lambda g, x: int((g == x).sum()), _num, colour=True), _each_phr(
        "Give the number of {X} cells in each of {refs}, in that order, separated by commas.",
        "How many {X} cells does each of {refs} contain? Keep the order listed and separate the numbers with commas.",
        "Number of {X} cells in each of {refs}, in order, comma-separated?",
        "The numbers of {X} cells in {refs}, in that order, are ⟪answer⟫."), weight=1.5),
    Spec("each_first_cell", "n", "list", _each(lambda g, x: int(g[0, 0]), _col), _each_phr(
        "Give the color of the top-left cell of each of {refs}, in that order, separated by commas.",
        "What color is the top-left cell of each of {refs}? Keep the order listed and separate the colors with "
        "commas.",
        "Top-left colors of {refs}, in order, comma-separated?",
        "The top-left cells of {refs}, in that order, are {Cs}⟪answer⟫.")),
    Spec("each_most", "n", "list", _each(_unique_most, _col), _each_phr(
        "Give the most frequent color of each of {refs} (each has exactly one), in that order, separated by commas.",
        "Which color covers the most cells in each of {refs}? Each has a single such color; keep the order listed "
        "and separate the colors with commas.",
        "Most common color of each of {refs} (unique in each), in order, comma-separated?",
        "The most frequent colors of {refs}, in that order, are {Cs}⟪answer⟫."), forbid=frozenset({"sparse"})),
    Spec("each_has", "n", "list", _each(lambda g, x: bool((g == x).any()), _yes, colour=True), _each_phr(
        "For each of {refs}, in that order, say yes if it contains {X} and no if it does not, separated by commas.",
        "Does each of {refs} contain {X}? Give yes or no for each, in the order listed, separated by commas.",
        "{Xc} present in {refs}? yes or no for each, in order, comma-separated.",
        "Asked whether each of {refs} contains {X}, the answers in that order are ⟪answer⟫.")),
    Spec("each_symmetric", "n", "list", _each(lambda g, x: bool(np.array_equal(g, g[:, ::-1])), _yes,
                                              all_no_keep=0.15), _each_phr(
        "For each of {refs}, in that order, say yes if it is symmetric left to right (every row reads the same "
        "backwards) and no otherwise, separated by commas.",
        "Is each of {refs} symmetric left to right, with every row reading the same backwards? Give yes or no for "
        "each, in the order listed, separated by commas.",
        "Left-right symmetric? yes or no for each of {refs}, in order, comma-separated.",
        "Asked whether each of {refs} is symmetric left to right, the answers in that order are ⟪answer⟫."),
         weight=0.6),
]


# --- several grids: "which of ... " -------------------------------------------------------------------

def _which(pred, *, colour=False, none_keep=0.3):
    def sample(rng, ctx):
        fmt = _refs_fmt(len(ctx.grids))
        params = {}
        x = None
        if colour:
            union = sorted(set().union(*(colour_counts(g) for g in ctx.grids)))
            absent = [v for v in range(10) if v not in union]
            x = pick(rng, absent) if absent and rng.random() < 0.1 else pick(rng, union)
            fmt.update(X=ctx.cw(x), Xc=ctx.cwc(x))
            params["X"] = x
        hits = [i for i, g in enumerate(ctx.grids) if pred(g, x)]
        if not hits and rng.random() >= none_keep:
            return None        # keep "none" answers a minority
        text, value = _named(ctx, hits)
        case = "none" if not hits else ("one" if len(hits) == 1 else "many")
        return Draw(value, text, case=case, fmt=fmt, params=params)
    return sample


def _which_phr(pl, sg, none=None):
    """``pl``/``sg``: the predicate for a plural / singular subject ("contain X" / "contains X");
    ``none``: the "none" statement when "None of ... <sg>" would be a double negative."""
    st = {"many": "The ones among {refs} that " + pl + " are ⟪answer⟫.",
          "one": "The only one among {refs} that " + sg + " is ⟪answer⟫.",
          "none": none or "None of {refs} " + sg + ", so the answer is ⟪answer⟫."}
    # declared (answer-neutral): one form for every case, "none" included
    return (
        P("interrogative", "Which of {refs} " + pl + "? Name them in the order listed; if none does, the answer is "
                           "none.", st, declared="The grids among {refs} that " + pl + " are: ⟪answer⟫."),
        P("imperative", "Name every one of {refs} that " + sg + ", in the order listed, or none if there is no such "
                        "grid.", st, declared="The ones among {refs} that " + pl + " are: ⟪answer⟫."),
        P("terse", "{Refs}: which " + pl + "? If none, the answer is none.", st,
          declared="Of {refs}, the ones that " + pl + " are: ⟪answer⟫."),
    )


WHICH_SPECS = [
    Spec("which_contain", "n", "list", _which(lambda g, x: bool((g == x).any()), colour=True, none_keep=1.0),
         _which_phr("contain {X}", "contains {X}"), weight=1.5),
    Spec("which_lack", "n", "list", _which(lambda g, x: not bool((g == x).any()), colour=True),
         _which_phr("have no {X} cell", "has no {X} cell",
                    none="Each of {refs} has at least one {X} cell, so the answer is ⟪answer⟫.")),
    Spec("which_symmetric_lr", "n", "list", _which(lambda g, x: bool(np.array_equal(g, g[:, ::-1])), none_keep=0.15),
         _which_phr("are symmetric left to right (every row reads the same backwards)",
                    "is symmetric left to right (every row reads the same backwards)"), weight=0.6),
    Spec("which_symmetric_tb", "n", "list", _which(lambda g, x: bool(np.array_equal(g, g[::-1])), none_keep=0.15),
         _which_phr("are symmetric top to bottom (the rows read the same in reverse order)",
                    "is symmetric top to bottom (the rows read the same in reverse order)"), weight=0.6),
    Spec("which_taller", "n", "list", _which(lambda g, x: g.shape[0] > g.shape[1]),
         _which_phr("have more rows than columns", "has more rows than columns"), dims=True, weight=0.7),
    Spec("which_first_cell", "n", "list", _which(lambda g, x: int(g[0, 0]) == x, colour=True),
         _which_phr("have {X} in the top-left cell", "has {X} in the top-left cell"), weight=0.7),
]

SPECS = PAIR_SPECS + EACH_SPECS + WHICH_SPECS
