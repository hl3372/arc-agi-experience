"""Size weight for general runs (2026-10-15): a ladder of rung
shares drawn per idea at admission, with pure-scale planner lines ANY.R2 and ANY.R3 beside ANY.R1."""
import copy, json, pathlib
import pytest
from arcgen.batch import admission, config, contracts

FLOORS = {'R1': {'output_cells': 150, 'computed_cells': 30, 'escalation': 0},
          'R2': {'output_cells': 300, 'computed_cells': 80, 'escalation': 1.3},
          'R3': {'output_cells': 450, 'computed_cells': 150, 'escalation': 1.3}}
LADDER = {'ladder': {'R1': 0.15, 'R2': 0.30, 'R3': 0.25}, 'excluded': ['DIGEST', 'SCENE', 'MANY']}


def rows(n, **study):
    return [{'source_candidate_id': f'k{i:04d}', 'private_source': 'ordinary_unconditioned',
             'study': {'primary_arm': 'SEE', **study}} for i in range(n)]


@pytest.fixture
def general(monkeypatch):
    monkeypatch.setitem(config.SCIENCE, 'rungs', FLOORS)
    monkeypatch.setitem(config.SCIENCE['routes']['scale_overlay'], 'rate', 0.0)
    return lambda draw: monkeypatch.setitem(config.SCIENCE, 'rung_draw', draw)


def test_ladder_draws_each_rung_at_its_share_and_skips_excluded_ideas(general):
    general(LADDER)
    out = admission.configured_routes(None, rows(4000))
    got = [r['study'].get('rung') for r in out]
    for rung, share in (('R1', .15), ('R2', .30), ('R3', .25), (None, .30)):
        assert abs(got.count(rung) / len(got) - share) < 0.03, rung
    assert admission.configured_routes(None, rows(4000)) == out  # seeded: the same on resume
    for excluded in (rows(500, idea_modifiers=['DIGEST']), rows(500, idea_modifiers=['SCENE'])):
        assert not any(r['study'].get('rung') for r in admission.configured_routes(None, excluded))
    assert all(r['study']['rung'] == 'R2' for r in admission.configured_routes(None, rows(50, rung='R2')))


def test_one_rung_at_a_rate_keeps_its_draws(general):
    general({'rung': 'R1', 'rate': 0.3, 'excluded': []})
    old = [r['study'].get('rung') for r in admission.configured_routes(None, rows(2000))]
    assert 0.27 < old.count('R1') / len(old) < 0.33
    general({'ladder': {'R1': 0.3}, 'excluded': []})
    assert [r['study'].get('rung') for r in admission.configured_routes(None, rows(2000))] == old
    general({'ladder': {'R3': 0.2, 'R1': 0.3}, 'excluded': []})  # rung order, not key order: R1 takes the first share
    ladder = [r['study'].get('rung') for r in admission.configured_routes(None, rows(2000))]
    assert [i for i, r in enumerate(ladder) if r == 'R1'] == [i for i, r in enumerate(old) if r == 'R1']


def test_config_validates_the_ladder():
    base = json.loads((config.PROFILES / 'v1.json').read_text())
    def check(draw, rungs=FLOORS):
        value = copy.deepcopy(base)
        value['science'].update({'rung_draw': draw, **({'rungs': rungs} if rungs else {})})
        return config.validate(value)
    check(LADDER)
    check({'ladder': {'R2': 1.0}, 'excluded': []})
    check({'rung': 'R1', 'rate': 0.3, 'excluded': ['DIGEST']})
    for bad in ({'ladder': {'R1': 0.5, 'R2': 0.4, 'R3': 0.2}, 'excluded': []},
                {'ladder': {'R4': 0.1}, 'excluded': []}, {'ladder': {'R1': -0.1}, 'excluded': []},
                {'ladder': {}, 'excluded': []}, {'ladder': [['R1', 0.3]], 'excluded': []},
                {'ladder': {'R1': '0.3'}, 'excluded': []}, {'ladder': {'R1': 0.3}, 'rung': 'R1', 'excluded': []},
                {'ladder': {'R1': 0.3}, 'excluded': ['NOPE']}, {'ladder': {'R1': 0.3}}):
        with pytest.raises(config.ConfigError):
            check(bad)
    with pytest.raises(config.ConfigError):
        check(LADDER, rungs=None)


def test_general_planner_gets_the_drawn_rungs_scale_line():
    b = contracts.Bindings()
    for rung in config.RUNGS:
        texts = [r[0] for r in b.planning_templates('SEE', (), rung=rung)]
        line, = [t for t in texts if 'Scale for this idea (binding' in t]
        assert f"at least {FLOORS[rung]['output_cells']} cells" in line
        assert f"at least {FLOORS[rung]['computed_cells']} of them computed" in line
    for rung in ('R2', 'R3'):  # pure scale: none of T's family judgements or dependency chains
        text = (contracts.RUNG_DIR / f'ANY.{rung}.planner.md').read_text()
        assert 'boundary' not in text and 'chain' not in text and 'without padding' in text
        assert 'at least 1.3 times the size of the average training pair' in text


def test_max_rung_caps_named_ideas_and_leaves_the_others(general):
    """Round 3 (2026-10-20): G's composition and reading ideas take R1/R2 again but never R3 (max_rung)."""
    general(LADDER)
    plain = [r['study'].get('rung') for r in admission.configured_routes(None, rows(3000))]
    joined = [r['study'].get('rung') for r in admission.configured_routes(None, rows(3000, idea_modifiers=['JOINED']))]
    general({**LADDER, 'max_rung': {'JOINED': 'R2', 'KEY': 'R1'}})
    assert [r['study'].get('rung') for r in admission.configured_routes(None, rows(3000))] == plain
    capped = [r['study'].get('rung') for r in admission.configured_routes(None, rows(3000, idea_modifiers=['JOINED']))]
    assert capped == [None if r == 'R3' else r for r in joined] and 'R3' in joined
    both = [r['study'].get('rung') for r in admission.configured_routes(None, rows(3000, idea_modifiers=['JOINED', 'KEY']))]
    assert set(both) == {'R1', None}  # the lowest cap wins
    assert not any(r['study'].get('rung') for r in admission.configured_routes(None, rows(300, idea_modifiers=['JOINED', 'SCENE'])))


def test_config_validates_max_rung():
    base = json.loads((config.PROFILES / 'v1.json').read_text())
    def check(draw):
        value = copy.deepcopy(base)
        value['science'].update({'rung_draw': draw, 'rungs': FLOORS})
        return config.validate(value)
    check({**LADDER, 'max_rung': {'JOINED': 'R2', 'KEY': 'R1'}})
    for bad in ({**LADDER, 'max_rung': {'NOPE': 'R2'}}, {**LADDER, 'max_rung': {'JOINED': 'R4'}},
                {**LADDER, 'max_rung': {'SCENE': 'R2'}}, {**LADDER, 'max_rung': ['JOINED']}):
        with pytest.raises(config.ConfigError):
            check(bad)
