"""Domain A public baselines: experimenters and learners behave as designed on hand-built worlds."""
import random
from research_envs.discover.world import World
from research_envs.discover import baselines as Bl, fairness as F
from tests.test_world import COUNTER

TOGGLE = {'state': {'x': {'range': [0, 3], 'settable': True}, 'm': {'range': [0, 1], 'boundary': 'wrap'}},
          'actions': {'A': {'rules': [{'set': {'m': 'm + 1'}}]},
                      'B': {'rules': [{'if': 'm == 1', 'set': {'x': 'x + 1'}}, {'set': {'x': 'x - 1'}}]}},
          'readouts': {'r1': 'x'}, 'max_len': 6}


def test_merge_learns_a_small_hidden_mode_exactly_from_a_sweep():
    """A hidden toggle: the passive learner recovers the 8-state machine from a sweep and predicts every query."""
    w = World(TOGGLE)
    data = Bl.collect(w, Bl.sweep(w, 40))
    name, m, cv = Bl.cv_select(data)
    qs = F.sample_queries(w, random.Random(3), n_interp=30, n_comp=10)
    acc, _ = Bl.accuracy(m, w, qs)
    assert acc == 1.0, (name, acc, cv)
    # the memoryless window model cannot represent the hidden mode
    acc_w, _ = Bl.accuracy(Bl.Window(data, 1, 1), w, qs)
    assert acc_w < 1.0


def test_persistence_is_a_floor_and_budgets_are_respected():
    w = World(COUNTER)
    for exps in (Bl.sweep(w, 25), Bl.random_experiments(w, 25, random.Random(0))):
        assert len(exps) <= 25
    data = Bl.collect(w, Bl.sweep(w, 25))
    qs = F.sample_queries(w, random.Random(4))
    accs = {k: Bl.accuracy(f(data), w, qs)[0] for k, f in Bl.LEARNERS.items()}
    assert max(accs['merge_0'], accs['merge_2']) >= accs['persistence']
