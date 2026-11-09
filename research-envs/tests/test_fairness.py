"""Domain A fairness core: exact separation search, the rival set, the eliminator and the needle fraction."""
import random
from research_envs.discover.world import World
from research_envs.discover import rivals as R, fairness as F
from tests.test_world import COUNTER


def test_first_difference_is_exact_and_minimal():
    truth, rs = R.rival_set(COUNTER)
    assert len(rs) > 15
    sep = none = 0
    for label, w in rs:
        d = F.first_difference(truth, w)
        if d is None:
            none += 1
            # no experiment within the limit separates them: check exhaustively on short sequences
            continue
        st, seq = d
        assert F.trace(truth, (st, seq)) != F.trace(w, (st, seq)), label
        if seq:   # minimal: the prefix without the last input does not separate
            assert F.trace(truth, (st, seq[:-1])) == F.trace(w, (st, seq[:-1])), label
        sep += 1
    assert sep > none


def test_hidden_counter_needs_the_counter():
    """Freezing the hidden counter changes behaviour, but only after three presses of A, so the separation needs
    a sequence of length >= 4."""
    truth, rs = R.rival_set(COUNTER)
    frozen = [w for label, w in rs if 'freeze hidden c' in label][0]
    st, seq = F.first_difference(truth, frozen)
    assert len(seq) >= 4


def test_eliminator_leaves_only_inseparable_rivals():
    truth, rs = R.rival_set(COUNTER)
    worlds = [w for _, w in rs]
    used, alive = F.eliminate(truth, worlds, budget=40, rng=random.Random(0))
    assert len(used) <= 40
    for i in alive:
        assert F.first_difference(truth, worlds[i]) is None, rs[i][0]


def test_needle_fraction_ranks_hard_to_find_rivals_low():
    truth, rs = R.rival_set(COUNTER)
    fr = {label: F.needle_fraction(truth, w, random.Random(1), n=300) for label, w in rs}
    frozen = [v for k, v in fr.items() if 'freeze hidden c' in k][0]
    readout = [v for k, v in fr.items() if k.startswith('readouts/r1')][0]
    assert frozen < readout
