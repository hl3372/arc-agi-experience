"""ARC parts end to end: >= 2000 parts, each re-checked from scratch by ``_arc_check`` (independent
brute-force rival loop, regeneration of aux examples, sibling loops, menus, every reading question),
plus determinism, the query mix, speed and sizes.

ARC_TEST_PARTS (default 2000) sets the number of parts; the checks run in a process pool.
"""
import collections
import hashlib
import os
import re
import subprocess
import sys
import time
from multiprocessing import get_context

import numpy as np
import pytest

import _arc_check as C
from pv2.arc import bundle as B
from pv2.arc import questions as Q
from pv2.encodings import render

N_PARTS = int(os.environ.get("ARC_TEST_PARTS", "2000"))
PROCS = max(1, min(20, os.cpu_count() or 1))
_RESULTS = {}


def _run(seeds, mode=None, determinism_every=10):
    args = [(s, mode, i % determinism_every == 0) for i, s in enumerate(seeds)]
    with get_context("fork").Pool(PROCS) as pool:
        return pool.starmap(C.check_seed, args, chunksize=2)


def results():
    if "main" not in _RESULTS:
        _RESULTS["main"] = _run(list(range(10_000, 10_000 + N_PARTS)))
    return _RESULTS["main"]


def test_every_part_passes_the_independent_checks():
    res = results()
    bad = [(r["seed"], r["mode"], r.get("family"), r["errors"][:3]) for r in res if r["errors"]]
    assert not bad, bad[:10]
    modes = collections.Counter(r["mode"] for r in res)
    assert set(modes) == set(B.MODES), modes


FLAGGED = ("rays", "recolor_size", "extract_row", "extract_col", "crop_object", "recolor_extreme", "mono",
           "keep_fill", "recolor_least")


def test_flagged_families_forced():
    """Families an independent verifier found ambiguous before the added readings (rays past an
    obstacle, size comparisons and extremes, blank extracted lines, colour-blind crops): 40 parts
    each through the whole independent check."""
    args = [(90_000 + 1000 * i + k, None, k % 8 == 0, fam) for i, fam in enumerate(FLAGGED) for k in range(40)]
    with get_context("fork").Pool(PROCS) as pool:
        res = pool.starmap(C.check_seed, args, chunksize=2)
    bad = [(r["seed"], r.get("family"), r["errors"][:3]) for r in res if r["errors"]]
    assert not bad, bad[:10]


@pytest.mark.parametrize("mode", B.MODES)
def test_each_mode_forced(mode):
    res = _run(list(range(90_000, 90_060)), mode=mode, determinism_every=6)
    bad = [(r["seed"], r.get("family"), r["errors"][:3]) for r in res if r["errors"]]
    assert not bad, bad[:10]


def test_query_mix():
    res = results()
    only = sum(1 for r in res if r.get("predict_only")) / len(res)
    assert only <= 0.40, only
    q = collections.Counter(qid for r in res for qid in r.get("qids", []))
    for qid in ("arc:predict", "arc:predict_value", "arc:verify", "arc:identify", "arc:count_color",
                "arc:which_contain", "arc:each_count", "arc:verify_value", "arc:row", "arc:rows", "arc:col",
                "arc:crop"):
        assert q[qid] > 0, qid
    enc = collections.Counter(r["encoding"] for r in res)
    share = enc["stage2"] / len(res)
    assert 0.30 <= share <= 0.50, enc          # STAGE2_SHARE 0.4 (red-team: 0.6 made it a new fixed scaffold)
    assert not any(e == "names" for e in enc)


def test_sizes():
    """Grid sizes near the ARC-AGI-2 eval mix (43% of its grids have >= 400 cells), every part within
    ~12.5k estimated stage-2 tokens."""
    res = results()
    tokens = np.array([r["s2tokens"] for r in res])
    assert tokens.max() <= 12_500, tokens.max()
    sides = [s for r in res for s in r["sides"]]
    assert max(max(s) for s in sides) <= 30
    cells = np.array([c for r in res for c in r["cells"] + r["answer_cells"]])
    big = (cells >= 400).mean()
    print(f"\nARC grid sizes: <100 {np.mean(cells < 100):.1%}, 100-399 {np.mean((cells >= 100) & (cells < 400)):.1%}, "
          f"400-899 {np.mean((cells >= 400) & (cells < 900)):.1%}, 900 {np.mean(cells == 900):.1%}; "
          f">= 800 {np.mean(cells >= 800):.1%}; stage-2 tokens p50 {np.percentile(tokens, 50):.0f} "
          f"p95 {np.percentile(tokens, 95):.0f}")
    assert 0.28 <= big <= 0.45, big
    assert (cells >= 800).mean() >= 0.005


def test_yes_no_balance():
    """No yes/no ARC question may be a constant (identical was "no" 1587/1587 before)."""
    yn = collections.defaultdict(list)
    for r in results():
        for qid, v in r["yn"]:
            yn[qid].append(v)
    assert set(yn) == {"arc:identical", "arc:verify", "arc:verify_value"}, set(yn)
    for qid, vals in yn.items():
        lo, hi = (0.35, 0.65) if len(vals) >= 100 else (0.2, 0.8)
        assert lo <= np.mean(vals) <= hi, (qid, np.mean(vals), len(vals))


def test_rebuild_from_provenance_and_rng():
    rng = np.random.default_rng(5)
    p = B.arc_part(rng)
    q = B.build(p.prov["seed"], None)
    assert repr(p) == repr(q)
    r1, r2 = np.random.default_rng(9), np.random.default_rng(9)
    assert repr(B.arc_part(r1, mode="scalar")) == repr(B.arc_part(r2, mode="scalar"))


def test_deterministic_across_processes():
    """Same parts in a fresh interpreter with another hash seed (set / dict order never leaks)."""
    seeds = [3, 17, 29, 41, 55]
    here = [repr(B.build(s)) for s in seeds]
    code = ("import sys; sys.path.insert(0, %r); import pv2.pictures; from pv2.arc import bundle as B; "
            "print('\\x1e'.join(repr(B.build(s)) for s in %r))" % (C.__file__.rsplit("/tests/", 1)[0], seeds))
    env = dict(os.environ, PYTHONHASHSEED="12345", PYTHONDONTWRITEBYTECODE="1")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
    assert out.stdout.rstrip("\n").split("\x1e") == here


def test_speed():
    """CPU ms per part in one process (the default mode mix; CPU time, since the machine is shared)."""
    B.build(1)
    times = []
    for s in range(50_000, 50_300):
        t0 = time.process_time()
        B.build(s)
        times.append((time.process_time() - t0) * 1000)
    p50, p95 = np.percentile(times, 50), np.percentile(times, 95)
    print(f"\nARC part speed: p50 {p50:.0f} ms, p95 {p95:.0f} ms, max {max(times):.0f} ms")
    assert p95 < 800, (p50, p95)       # big grids (2026-09-30) made the rival search slower


def test_token_budget():
    """Stage-2 rendering of every grid of a part (plus its grid answers) stays within ~10k tokens."""
    tok = pytest.importorskip("tokenizers")
    root = os.environ.get("PV2_TOKENIZER", "")
    path = os.path.join(root, "tokenizer.json")
    if not (root and os.path.exists(path)):
        pytest.skip("no tokenizer.json (set PV2_TOKENIZER to a local Qwen3.5-4B folder)")
    t = tok.Tokenizer.from_file(path)
    counts = []
    for s in range(60_000, 60_200):
        p = B.build(s)
        n = 0
        for slot in p.slots:
            n += len(t.encode(render(slot.grid, "stage2")).ids) + 14
        for q in p.queries:
            if q.answer.kind == "grid":
                n += len(t.encode(render(q.answer.value, "compact")).ids)
            n += len(t.encode(q.text).ids)
        n += sum(len(t.encode(c).ids) for c in p.context)
        counts.append(n)
    counts = np.array(counts)
    print(f"\nARC part tokens (stage-2 grids): p50 {np.percentile(counts, 50):.0f}, "
          f"p95 {np.percentile(counts, 95):.0f}, max {counts.max()}")
    assert counts.max() <= 12_500 and np.percentile(counts, 95) <= 11_500


# --- reading-only parts and rule-source filters -----------------------------------------------------

# the questions that never use the rule, listed here independently of pv2.arc.bundle
READING = {"arc:count_color", "arc:n_colors", "arc:mode_color", "arc:dims", "arc:color_at", "arc:changed_cells",
           "arc:identical", "arc:which_contain", "arc:each_count", "arc:each_n_colors",
           "arc:row", "arc:rows", "arc:col", "arc:crop"}
# digest of 200 default parts (build(0..199)) and 20 from arc_part(default_rng(2026)); refresh it only
# when default generation changes on purpose.  History: 7f2bf4f9... recorded at a663751 (the options
# questions / families / aux_tasks left it unchanged); replaced 2026-09-30 by the large-grid and
# yes/no-balance change and the object-size rivals beyond 20 cells, which alter default parts
# deliberately; replaced again 2026-09-30 by the parity-review fixes (random first side in the window
# and scene samplers, one reading question per (qid, grids), informative colour draws, explained
# scalar value headers, one question per measure and grid, DESIGN.md section 15); replaced again
# 2026-09-30 by the grid-writing reading questions (a row, rows a..b, a column or a crop of one shown
# grid: half of the reading draws in default parts, one window per grid, never next to a size question
# on that grid; size-valued scalar bundles ask neither a size nor a window; DESIGN.md section 15,
# grid-writing rate).
DEFAULT_DIGEST = "3bf198031f193d9ef6d24fff1dd72c9427108cf95665a6a127a54412038c0828"


def test_reading_parts_never_ask_about_the_rule():
    """questions="reading" over 500 parts, 100 per mode, each through the independent checks."""
    args = [(70_000 + 1000 * i + k, mode, k % 10 == 0, None, {"questions": "reading"})
            for i, mode in enumerate(B.MODES) for k in range(100)]
    with get_context("fork").Pool(PROCS) as pool:
        res = pool.starmap(C.check_seed, args, chunksize=2)
    bad = [(r["seed"], r["mode"], r["errors"][:3]) for r in res if r["errors"]]
    assert not bad, bad[:10]
    assert collections.Counter(r["mode"] for r in res) == {m: 100 for m in B.MODES}
    for r in res:
        assert len(r["qids"]) >= 2 and set(r["qids"]) <= READING, r["qids"]
        assert "proposed" not in r["roles"]
        assert not any(re.match(r"[A-F]\. ", c) or c.startswith(tuple(Q.MENU_HEADS)) for c in r["context"])
    assert set(B.READING_QIDS) == READING


def test_filters_are_respected():
    fams = ["rot", "recolor", "gravity", "outline"]
    for s in range(30):
        p = B.build(s, "transform", families=fams)
        assert p.prov["steps"][0][0] in fams and p.prov["families_filter"] == sorted(fams)
        p = B.build(s, "composition", families=fams)
        assert all(step[0] in fams for step in p.prov["steps"])
        p = B.build(s, "scalar", families=["count_color", "n_colors"])
        assert p.prov["scalar"][0] in ("count_color", "n_colors")
        for mode in ("stated", "hidden"):
            p = B.build(s, mode, aux_tasks=["10", "29"])
            assert p.prov["task"] in ("10", "29") and p.prov["aux_tasks_filter"] == ["10", "29"]
    # mode=None draws only among modes with a source left: one family cannot make a composition,
    # and no scalar family was named
    modes = collections.Counter(B.build(s, families=["rot"], aux_tasks=["10"]).prov["mode"] for s in range(60))
    assert set(modes) <= {"transform", "stated", "hidden"} and len(modes) == 3, modes
    # filtered parts still pass the independent checks
    args = [(80_000 + s, None, False, None, {"families": fams, "aux_tasks": ["10", "29", "70"]}) for s in range(40)]
    with get_context("fork").Pool(PROCS) as pool:
        res = pool.starmap(C.check_seed, args, chunksize=2)
    assert not [r["errors"] for r in res if r["errors"]]


def test_filters_that_leave_nothing_raise():
    with pytest.raises(ValueError, match="no rule source"):
        B.build(1, "scalar", families=["rot"])
    with pytest.raises(ValueError, match="no rule source"):
        B.build(1, "composition", families=["rot"])
    with pytest.raises(ValueError, match="no rule source"):
        B.build(1, "hidden", aux_tasks=["7"])          # task 7 is usable stated only
    with pytest.raises(ValueError, match="unknown"):
        B.build(1, families=["no_such_family"])
    with pytest.raises(ValueError, match="unknown"):
        B.build(1, aux_tasks=["14"])
    with pytest.raises(ValueError, match="questions"):
        B.build(1, questions="none")


def test_defaults_are_unchanged():
    h = hashlib.sha256()
    for s in range(200):
        p = B.build(s)
        h.update(repr(p).encode())
        if s % 20 == 0:
            assert repr(B.build(s, None, questions="all", families=None, aux_tasks=None)) == repr(p)
    rng = np.random.default_rng(2026)
    for _ in range(20):
        h.update(repr(B.arc_part(rng)).encode())
    assert h.hexdigest() == DEFAULT_DIGEST
