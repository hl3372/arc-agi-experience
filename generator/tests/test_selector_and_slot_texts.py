"""No evaluation references reach the selector; the quota line, the canvas draw,
the plain second-test overlay and the JOINED slot."""
import pathlib, sys
import pytest
from arcgen.batch import admission, common, config, contracts, idea_v1, supply
from arcgen.batch.allocation import design
from support.pilot import run_case

V1 = pathlib.Path(__file__).resolve().parents[1] / 'arcgen/batch/prompts/v1'


def test_no_eval_references_reach_the_selector():
    # the reference corpus is the bank only: no real-task neighbours, and no setting to ask for them
    assert 'real_neighbors_per_candidate' not in config.SCIENCE['selector']
    assert common.SETTINGS['selector']['real_neighbors_per_candidate'] == 0


def test_no_four_ideas_quota_in_the_requirements():
    text = (V1 / 'ideas/requirements.md').read_text()
    assert 'four final ideas' not in text and 'The final ideas must differ' in text


def test_second_test_overlay_is_a_plain_second_instance():
    assert 'SECOND' in config.OVERLAY_KINDS
    text = contracts.OVERLAY_INVITATIONS['SECOND']
    assert 'two test inputs' in text and 'no case the training pairs do not show' in text


def test_joined_slot_has_its_idea_and_planner_texts():
    assert 'JOINED' in config.MODIFIER_NAMES and 'JOINED' in contracts.SLOT_BRIDGES
    idea = (V1 / 'ideas/modifiers/JOINED.md').read_text()
    assert 'Additional requirement for this call' in idea and 'not bolting a second mechanism' in idea
    assert 'Two separate puzzles that merely share a grid do not count' in idea
    assert 'not a bundle of unrelated' in (V1 / 'slots/JOINED.planner.md').read_text()


def rows(n, **study):
    return [{'source_candidate_id': f'k{i:04d}', 'private_source': 'ordinary_unconditioned',
             'study': {'primary_arm': 'SEE', **study}} for i in range(n)]


def test_canvas_draw_shares_and_margin_exclusions(monkeypatch):
    monkeypatch.setitem(config.SCIENCE, 'canvas_draw', {'square': 0.35, 'margin': 0.30, 'both': 0.10,
                                                         'margin_excluded': ['R2', 'R3', 'PACKED']})
    monkeypatch.setitem(admission.SCIENCE, 'canvas_draw', config.SCIENCE['canvas_draw'])
    monkeypatch.setitem(config.SCIENCE['routes']['scale_overlay'], 'rate', 0.0)
    out = admission.configured_routes(None, rows(2000))
    kinds = [r['study'].get('canvas') for r in out]
    for kind, share in (('square', .35), ('margin', .30), ('both', .10), (None, .25)):
        assert abs(kinds.count(kind) / len(kinds) - share) < 0.04, kind
    # the same seed gives the same draw for the same idea
    assert [r['study'].get('canvas') for r in admission.configured_routes(None, rows(50))] == kinds[:50]
    # an excluded rung or slot never gets open margins: 'both' becomes square, 'margin' becomes the control
    excluded = admission.configured_routes(None, rows(2000, rung='R2')) + admission.configured_routes(
        None, rows(2000, idea_modifiers=['PACKED']))
    assert not any(r['study'].get('canvas') in ('margin', 'both') for r in excluded)


def test_canvas_draw_validation():
    import copy, json
    base = json.loads((pathlib.Path(__file__).resolve().parents[1] / 'arcgen/batch/profiles/v1.json').read_text())
    for good in ({'square': 0.35, 'margin': 0.30, 'both': 0.10, 'margin_excluded': ['R2']}, {'square': 1.0}):
        value = copy.deepcopy(base); value['science']['canvas_draw'] = good
        config.validate(value)
    for bad in ({'square': 0.7, 'margin': 0.4}, {'square': 0.3, 'diagonal': 0.1}, {'square': -0.1},
                {'square': 0.3, 'margin_excluded': 'R2'}):
        value = copy.deepcopy(base); value['science']['canvas_draw'] = bad
        with pytest.raises(config.ConfigError):
            config.validate(value)


@pytest.mark.parametrize('canvas', ['square', 'margin', 'both'])
def test_canvas_line_comes_after_the_rung_line(canvas):
    b = contracts.Bindings()
    rows_ = b.planning_templates('SEE', (), rung='R1', canvas=canvas)
    texts = [r[0] for r in rows_]
    assert 'Canvas for this idea (binding' in texts[-1]
    assert any('Scale for this idea' in t for t in texts[:-1])
    with pytest.raises(Exception):
        b.planning_templates('SEE', (), canvas='diagonal')


@pytest.mark.parametrize('rate,expect', [(1.0, True), (0.0, False)])
def test_canvas_line_reaches_the_planner_prompt(tmp_path, monkeypatch, rate, expect):
    draw = {'square': rate} if rate else {'square': 0.0}
    monkeypatch.setitem(config.SCIENCE, 'canvas_draw', draw)
    monkeypatch.setitem(admission.SCIENCE, 'canvas_draw', draw)
    monkeypatch.setitem(config.SCIENCE['routes']['scale_overlay'], 'rate', 0.0)
    first, *_ = run_case(tmp_path, monkeypatch, 'clear', 4, True, arm='SEE', planner='sol')
    record = next(iter(first['runtime'].state['semantic_records'].values()))
    assert (record['study'].get('canvas') == 'square') is expect
    prompt, = list(tmp_path.glob('accounts/**/model_calls/astra_concrete_plan_v3/*/prompt.md'))
    assert ('Canvas for this idea (binding' in prompt.read_text()) is expect


def test_margin_lines_ask_for_a_real_band_of_background():
    """v1-b17-screen-t: the first margin line ("clear of at least one edge") was met with a one-cell border in 83% of
    margin-drawn inputs; eval inputs have a band of two or more cells on some side in 53%."""
    for name in ('canvas.margin.planner.md', 'canvas.both.planner.md'):
        text = (V1 / name).read_text()
        assert 'two or more cells wide' in text and 'A one-cell border is not open background' in text
        assert 'clear of at least one edge' not in text


def test_instruct_slot_texts():
    """Round 2 (2026-10-16): an ordered symbolic instruction applied step by step, its effects reaching across the grid."""
    assert 'INSTRUCT' in config.MODIFIER_NAMES and 'INSTRUCT' in contracts.SLOT_BRIDGES
    idea = (V1 / 'ideas/modifiers/INSTRUCT.md').read_text()
    assert 'Additional requirement for this call' in idea and 'read in an order the picture makes plain' in idea
    assert 'never from a colour\'s number' in idea and 'a single entry to look up, or a count to reuse, is too thin' in idea
    planner = (V1 / 'slots/INSTRUCT.planner.md').read_text()
    assert 'at least two pairs' in planner and 'one to three entries longer' in planner


def test_controlled_comparisons_are_not_near_copies():
    """Round 2 (2026-10-17, task d471c157: two training inputs differing in 4 of 400 cells): CONTRAST and MINIMAL keep the
    comparison but place it inside one grid or across pairs whose scenes otherwise differ."""
    for name in ('bridges/CONTRAST.planner.md', 'slots/MINIMAL.planner.md'):
        assert 'no two training inputs may share nine in ten of their cells' in (V1 / name).read_text()
    assert 'are a copy, not a comparison' in (V1 / 'ideas/cards/CONTRAST.md').read_text()
    minimal = (V1 / 'ideas/modifiers/MINIMAL.md').read_text()
    assert 'inputs that stay alike except where it matters' not in minimal and 'inside one picture' in minimal
