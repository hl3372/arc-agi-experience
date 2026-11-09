"""Domain A novelty: disguised textbook mechanisms are caught; different mechanisms are not confused."""
import copy, json
from research_envs.discover import novelty
from research_envs.discover.world import World

ARCH = {a['id']: a for a in novelty.load_archive()}


def rename(spec, mapping):
    s = json.dumps(spec)
    for a, b in mapping.items():
        s = s.replace(f'"{a}"', f'"{b}"')
    return json.loads(s)


def test_renamed_and_irrelevant_control_disguises_match_by_behaviour():
    w = ARCH['vending']['world']
    disguised = rename(w, {'coin': 'Q', 'refund': 'Z', 'dispense': 'r7'})
    disguised['actions']['noop'] = {'rules': [{'set': {}}]}          # an irrelevant control
    disguised['readouts']['always'] = '4'                            # a constant readout
    assert 'vending' in novelty.matches(disguised)['behaviour']


def test_recoded_readout_values_match_by_behaviour():
    w = copy.deepcopy(ARCH['ring_counter']['world'])
    w['readouts'] = {'pos': '(p * 3 + 1) % 5'}                       # a bijective relabelling of the readings
    assert 'ring_counter' in novelty.matches(w)['behaviour']


def test_changed_constants_are_the_same_family_by_structure():
    w = copy.deepcopy(ARCH['vending']['world'])
    w['actions']['coin']['rules'][0]['if'] = 'c >= 3'
    w['state']['c']['range'] = [0, 4]
    m = novelty.matches(w)
    assert 'vending' not in m['behaviour'] and 'vending' in m['structure']


def test_distinct_mechanisms_do_not_match():
    for i, a in ARCH.items():
        others = [j for j, b in ARCH.items() if j != i and novelty.same(a['fingerprint'], b['fingerprint'])]
        assert not others, (i, others)
