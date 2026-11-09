"""Tests for prompt_variants.py: the store, the compound tokens, validation and sampling.

The store ships empty (no extra wordings yet); the tests use temporary files.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import prompt_variants as pv  # noqa: E402
from prompts import ISOLATED_CELL_RULE, get_prompt  # noqa: E402
from prompts import _BG_IGNORE as BG_IGNORE  # noqa: E402
from registry import generate  # noqa: E402


def test_shipped_store_loads_and_every_variant_validates():
    payload = pv.load()
    assert payload["version"] == 1
    for tid, rows in payload["tasks"].items():
        assert 3 <= len(rows) <= 10, tid
        ids = [r["id"] for r in rows]
        assert len(set(ids)) == len(ids)
        for r in rows:
            assert r["status"] in pv.STATUSES
            if r["status"] == "approved":
                assert pv.check_template(tid, r["template"], seeds=(0, 1)) == [], (tid, r["id"])


def test_compound_tokens_render_the_policy_sentences():
    meta4 = generate("4", 0)["metadata"]
    t = pv.render("4", "{shape_object_rule} How many objects are {shape_query_phrase}?"
                  "{shape_size_filter_note} Answer with the number.", meta4)
    assert t == get_prompt("4", meta4)
    meta7 = generate("7", 0)["metadata"]
    t = pv.render("7", "{pair_object_rule}", meta7, color_palette=pv.LETTER_PALETTE)
    assert t.startswith(f"{BG_IGNORE} Treat each remaining") and ISOLATED_CELL_RULE in t
    assert pv.render("3", "{index_note} {unchanged_note}", {}) == pv.INDEX_NOTE + " " + pv.UNCHANGED_NOTE
    for seed in range(40):  # every task 3 mode reproduces the canonical wording through the token
        m3 = generate("3", seed)["metadata"]
        assert pv.render("3", "Place {placements_phrase}. {index_note} {unchanged_note}", m3) == get_prompt("3", m3)
        # ... and colours inside the compound token go through the palette
        assert pv.render("3", "Place {placements_phrase}.", m3, color_palette=pv.LETTER_PALETTE) == get_prompt("3", m3, color_palette=pv.LETTER_PALETTE).split(" Rows and")[0]
    import pytest
    with pytest.raises(ValueError):
        pv.render("4", "How many {objetcs}?", generate("4", 0)["metadata"])
    for seed in range(6):
        m9 = generate("9", seed)["metadata"]
        assert pv.render("9", "{image_a}/{image_b}", m9) in ("top image/bottom image", "left image/right image")
        m10 = generate("10", seed)["metadata"]
        assert pv.render("10", "Rotate this grid {rotation_phrase}.", m10) == get_prompt("10", m10)


def test_check_template_accepts_a_faithful_rewording():
    good = ("{shape_object_rule} Count the objects that are {shape_query_phrase}."
            "{shape_size_filter_note} Answer with the number.")
    assert pv.check_template("4", good, seeds=(0, 1, 2)) == []
    good2 = "Give the coordinates of every color {ci:query_color_index} cell in row-major order as [(r,c),...]. Rows and columns are 0-indexed from the top-left corner."
    assert pv.check_template("2", good2, seeds=(0, 1)) == []


def test_check_template_rejects_policy_violations():
    problems = pv.check_template("4", "The background is color {ci:background}. Count the {shape_query_phrase}.", seeds=(0,))
    joined = " | ".join(problems)
    assert "forbidden wording" in joined and "isolated single cells" in joined
    problems = pv.check_template("3", "Place {placements_phrase}.", seeds=(0,))
    joined = " | ".join(problems)
    assert "0-indexed" in joined and "Leave every other cell unchanged" in joined
    problems = pv.check_template("9", "Which rotation maps the first image to the second?", seeds=(0,))
    assert any("answer-format sentence" in p for p in problems)
    problems = pv.check_template("4", "{shape_object_rule} How many {unknown:key}?", seeds=(0,))
    assert any("render error" in p or "unresolved" in p for p in problems)


def test_store_roundtrip_add_validate_select():
    path = os.path.join(tempfile.mkdtemp(), "variants.json")
    rc = pv.main(["--path", path, "add", "--task", "2", "--seeds", "0,1",
                  "--template", "List every color {ci:query_color_index} cell as [(r,c),...] in row-major order. Rows and columns are 0-indexed from the top-left corner.",
                  "--note", "test"])
    assert rc == 0
    payload = pv.load(path)
    assert [v["id"] for v in payload["tasks"]["2"]] == ["2_v001"] and payload["tasks"]["2"][0]["status"] == "approved"
    # a failing template is not stored as approved
    rc = pv.main(["--path", path, "add", "--task", "2", "--seeds", "0", "--template", "List the cells."])
    assert rc == 1 and len(pv.load(path)["tasks"]["2"]) == 1
    rc = pv.main(["--path", path, "add", "--task", "2", "--seeds", "0", "--template", "List the cells.", "--force"])
    assert rc == 0 and pv.load(path)["tasks"]["2"][1]["status"] == "draft"
    report = pv.validate(["2"], seeds=(0, 1), payload=pv.load(path))
    assert report == {"2": {"2_v001": []}}
    report = pv.validate(["2"], seeds=(0,), include_drafts=True, payload=pv.load(path))
    assert report["2"]["2_v002"]
    rng = np.random.default_rng(0)
    seen = {pv.select_variant("2", rng, pv.load(path)) for _ in range(60)}
    assert seen == {None, payload["tasks"]["2"][0]["template"]}
    assert json.load(open(path))["version"] == 1


def test_coverage_seeds_span_the_canonical_wordings():
    seeds4 = pv.coverage_seeds("4", sweep=300)
    assert len(seeds4) >= 25          # categories x filter x texture kinds
    seeds3 = pv.coverage_seeds("3", sweep=60)
    modes = {(len(generate("3", s)["metadata"].get("placement_rows", [1])) > 1, generate("3", s)["metadata"].get("colour_mode")) for s in seeds3}
    assert len(modes) >= 3            # single cell, shared colour, per-cell colours
    seeds10 = pv.coverage_seeds("10", sweep=30)
    assert any(generate("10", s)["metadata"]["angle"] == 180 for s in seeds10)
    good = ("{shape_object_rule} Count the objects that are {shape_query_phrase}."
            "{shape_size_filter_note} Answer with the number.")
    assert pv.check_template("4", good, seeds=seeds4[:20]) == []


_TOKEN_TEMPLATES_11_21 = {
    "11": "Reflect this grid across the {axis_phrase} (flip {flip_phrase}).",
    "12": "Rotate only the color {ci:target_color} object ({connectivity_phrase}-connected) by {angle} degrees clockwise around the center of its square bounding box; the cells it vacates become background. {unchanged_note}",
    "13": "{piece_rule} {anchor_rule} {fit_rule} {receive_rule} {leftover_rule} {piece_tail}",
    "15": "{line_rule} {crossing_shown_rule} {distractor_note} Count the cells that lie on two or more lines. Answer with the number.",
    "15a": "{line_rule} {pairwise_count_rule} {distractor_note} Answer with the number.",
    "16": "{line_rule} {crossing_shown_rule} Paint every cell that lies on two or more lines with color {ci:highlight_color}. {distractor_note} {unchanged_note}",
    "17": "{largest_object_rule} Find the object with the largest bounding-box area, i.e. the height times the width of the smallest axis-aligned rectangle that encloses it (exactly one object has the largest box), and recolor every cell of that object to color {ci:target_color}. {unchanged_note}",
    "18": "{largest_object_rule} Find the object with the most cells (exactly one object has the most) and recolor every cell of that object to color {ci:target_color}. {unchanged_note}",
    "20": "{bg_ignore} {exact_object_rule} Recolor every object with exactly {cells_phrase} to color {ci_any:target_color|target_color_index}. {unchanged_note}",
}


def test_tokens_11_21_reproduce_every_canonical_wording():
    """The compound tokens for 11-21 rebuild the canonical prompt on one seed per wording
    (13's anchor rule x host/mixed/one-to-one x side counts; 20/21's '1 cell' vs 'k cells';
    11's two axes; 17's two connectivities), under digits and under the letter palette."""
    for tid, template in _TOKEN_TEMPLATES_11_21.items():
        seeds = pv.coverage_seeds(tid, sweep=300)
        assert seeds
        for seed in seeds:
            meta = generate(tid, seed)["metadata"]
            for palette in (None, pv.LETTER_PALETTE):
                assert pv.render(tid, template, meta, color_palette=palette) == get_prompt(tid, meta, color_palette=palette), (tid, seed)
    # the policy phrases the checker now enforces are all present in the canonical prompts
    assert "at least two cells past" in get_prompt("15", generate("15", 0)["metadata"])
    assert "count every pair" in get_prompt("15a", generate("15a", 0)["metadata"])
    assert "exactly one object has" in get_prompt("18", generate("18", 0)["metadata"])
    assert "do not rotate or reflect" in get_prompt("13", generate("13", 0)["metadata"])
    assert "stays exactly where it is" in get_prompt("13", generate("13", 0)["metadata"])
    problems = pv.check_template("13", "{piece_rule} {anchor_rule} {fit_rule} {receive_rule} {piece_tail}", seeds=(0,))
    assert any("stays exactly where it is" in p for p in problems)   # the leftover sentence is required
    problems = pv.check_template("16", "{line_rule} Paint every cell on two or more lines with color {ci:highlight_color}. {unchanged_note}", seeds=(0,))
    assert not problems  # rewording without the shown-colour sentence is allowed
    problems = pv.check_template("16", "Every line is straight and one cell wide. Paint every cell on two or more lines with color {ci:highlight_color}. {unchanged_note}", seeds=(0,))
    assert any("at least two cells past" in p for p in problems)


_GENERIC_TOKEN_TEMPLATES = {
    # constants, helpers (auto-bound metadata / connectivity), lookup tables and explicit arguments
    "5a": "{const:_SHAPE_OBJECT_RULE}{const:_P5A_ACTION}",
    "90": "{fn:_largest_object_rule} {const:_P90_QUESTION}",
    "44": ("Tile this grid {fn:_tile_times_phrase:m:tile_rows|'vertically'} and {fn:_tile_times_phrase:m:tile_cols|'horizontally'}. "
           "The output is {fn:_copies:m:tile_rows} tall and {fn:_copies:m:tile_cols} wide, counting the input as one copy."),
    "23": ("Starting from the single cell of color {ci:start_color}, draw a diagonal line in color {ci:trace_color} "
           "in the {m:direction} ({dict:_P23_GLOSS:direction}) direction. {fn:_p23_reflection_rule:n_bounces} "
           "{fn:_p23_overlap_sentence} Leave every other cell unchanged."),
}


def test_generic_tokens_reproduce_canonical_wordings():
    """{const:} {fn:} and {dict:} reach the task's own prompt module, on one seed per wording."""
    for tid, template in _GENERIC_TOKEN_TEMPLATES.items():
        if tid == "23" and not hasattr(pv.prompts, "_p23_overlap_sentence"):
            template = template.replace("{fn:_p23_overlap_sentence} ", "")
        for seed in pv.coverage_seeds(tid, sweep=200):
            meta = generate(tid, seed)["metadata"]
            for palette in (None, pv.LETTER_PALETTE):
                out = pv.render(tid, template, meta, color_palette=palette)
                canonical = get_prompt(tid, meta, color_palette=palette)
                assert canonical.startswith(out.split(" Leave every")[0]) if tid == "23" else out == canonical, (tid, seed)
    # a prompt-module task resolves names from its own module first
    meta99 = generate("99", 0)["metadata"]
    goal = pv.render("99", "{dict:_TRANSFORMS:transform:0}", meta99)
    assert goal and goal in get_prompt("99", meta99)
    # hygiene: unknown names, public names, non-string constants and missing arguments raise
    for bad in ("{const:_NO_SUCH_NAME}", "{const:ISOLATED_CELL_RULE}", "{const:_P23_GLOSS}",
                "{fn:_p23_reflection_rule}", "{dict:_P23_GLOSS:no_such_key}"):
        try:
            pv.render("23", bad, generate("23", 0)["metadata"])
        except ValueError:
            continue
        raise AssertionError(bad)
