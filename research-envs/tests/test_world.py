"""Domain A world interpreter: semantics, bounds, compilation and the experiment interface."""
import pytest
from research_envs.discover.world import World, WorldError

# A hidden press counter: A increments a hidden count c (wrapping at 4); B moves x up by 1 only when c is 3, else down.
COUNTER = {'state': {'x': {'range': [0, 5], 'settable': True}, 'c': {'range': [0, 3], 'boundary': 'wrap'}},
           'actions': {'A': {'rules': [{'set': {'c': 'c + 1'}}]},
                       'B': {'rules': [{'if': 'c == 3', 'set': {'x': 'x + 1'}}, {'set': {'x': 'x - 1'}}]}},
           'readouts': {'r1': 'x', 'r2': 'x > 2'}, 'max_len': 6}


def test_semantics_first_rule_wins_wrap_and_clamp():
    w = World(COUNTER)
    assert w.n_states == 24 and w.inputs == [('A', None), ('B', None)]
    t = w.experiment({'start': {'x': 2}, 'actions': [['B'], ['A'], ['A'], ['A'], ['B'], ['B']]})
    assert [r['r1'] for r in t] == [2, 1, 1, 1, 1, 2, 3]       # c: 0,0,1,2,3,3,3 -> B adds only when c == 3
    assert [r['r2'] for r in t] == [0, 0, 0, 0, 0, 0, 1]
    t = w.experiment({'start': {'x': 0}, 'actions': [['B'], ['B']]})
    assert [r['r1'] for r in t] == [0, 0, 0]                    # clamped at 0
    t = w.experiment({'start': {'x': 0}, 'actions': [['A']] * 5})
    assert w.decode(w.next[0, w.next[0, w.next[0, w.next[0, w.next[0, w.start_state({'x': 0})]]]]])['c'] == 1  # wraps


def test_simultaneous_assignment_param_and_tick():
    swap = {'state': {'x': {'range': [0, 9], 'settable': True}, 'y': {'range': [0, 9], 'settable': True}},
            'actions': {'S': {'rules': [{'set': {'x': 'y', 'y': 'x'}}]},
                        'P': {'param': [1, 3], 'rules': [{'set': {'x': 'x + k'}}]}},
            'tick': [{'if': 'x > y', 'set': {'y': 'y + 1'}}],
            'readouts': {'r1': 'x', 'r2': 'y'}}
    w = World(swap)
    assert len(w.inputs) == 4
    t = w.experiment({'start': {'x': 1, 'y': 5}, 'actions': [['S'], ['P', 3]]})
    assert [(r['r1'], r['r2']) for r in t] == [(1, 5), (5, 2), (8, 3)]   # swap, tick +1 on y; then x+3, tick


@pytest.mark.parametrize('bad,msg', [
    ({**COUNTER, 'state': {'x': {'range': [0, 99]}}}, 'at most 64'),
    ({**COUNTER, 'actions': {'A': {'rules': [{'set': {'z': '1'}}]}}}, 'unknown variable'),
    ({**COUNTER, 'readouts': {'r1': 'x.real'}}, ''),
    ({**COUNTER, 'max_len': 50}, 'max_len'),
])
def test_bad_worlds_are_rejected(bad, msg):
    with pytest.raises(Exception) as e:
        World(bad)
    assert msg in str(e.value)


def test_interface_limits():
    w = World(COUNTER)
    with pytest.raises(WorldError):
        w.experiment({'start': {'c': 1}, 'actions': []})          # hidden variable is not settable
    with pytest.raises(WorldError):
        w.experiment({'start': {'x': 9}, 'actions': []})          # out of range
    with pytest.raises(WorldError):
        w.experiment({'start': {'x': 1}, 'actions': [['A']] * 7})  # longer than max_len
    assert len(w.starts()) == 6
