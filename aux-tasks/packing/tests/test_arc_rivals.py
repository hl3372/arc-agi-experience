"""The rival search on hand-made bundles: what it certifies, what it refuses and why."""
from pv2.arc.rivals import certify, unique_test_output
from pv2.arc.scalars import ScalarRule, certify_scalar
from pv2.arc.transforms import Rule

A = [[0, 0, 0, 0, 0],
     [0, 3, 3, 0, 0],
     [0, 3, 0, 0, 5],
     [0, 0, 0, 0, 5]]
B = [[0, 0, 0, 0],
     [0, 5, 5, 0],
     [0, 3, 0, 0],
     [0, 3, 0, 0],
     [0, 0, 0, 3]]
C = [[5, 0, 0],
     [5, 0, 3],
     [0, 0, 3],
     [0, 0, 0],
     [3, 0, 0],
     [0, 0, 0]]
T = [[0, 0, 0, 0],
     [3, 3, 0, 5],
     [0, 0, 0, 5],
     [0, 3, 0, 0]]


def pairs(rule, grids):
    return [(g, rule.apply(g)) for g in grids]


def test_unique_rotation_is_certified():
    rot = Rule.single("rot", 1)
    out = unique_test_output(pairs(rot, [A, B, C]), T, rot)
    assert out == rot.apply(T)


def test_one_symmetric_pair_is_ambiguous():
    """A single pair whose input is symmetric cannot tell rot 180 from both flips."""
    sym = [[1, 0, 1], [0, 2, 0], [1, 0, 1]]
    rot = Rule.single("rot", 2)
    assert unique_test_output([(sym, rot.apply(sym)), ([[1, 2, 2]], rot.apply([[1, 2, 2]]))], T, rot) is None


def test_unwitnessed_test_colour_is_rejected():
    """Colour 7 never appears in a train pair, so a colour map after the rule is free on it."""
    rot = Rule.single("rot", 1)
    t = [row[:] for row in T]
    t[0][0] = 7
    cert = certify(pairs(rot, [A, B, C]), [t], rot)
    assert cert.outputs is None and cert.reason == "ambiguous_colour"


def test_recolor_vs_swap_needs_the_target_colour_in_a_train_input():
    rec = Rule.single("recolor", 3, 5)
    only3 = [[[0, 3, 0], [3, 3, 0]], [[3, 0, 0], [0, 0, 3], [0, 3, 3]], [[0, 0, 3, 3]]]
    test_with_5 = [[5, 3], [0, 0]]
    cert = certify(pairs(rec, only3), [test_with_5], rec)
    assert cert.outputs is None          # swap(3,5) fits the pairs and differs on the test
    with5 = only3 + [[[5, 3, 0], [0, 3, 5]]]
    cert = certify(pairs(rec, with5), [test_with_5], rec)
    assert cert.outputs is not None and cert.outputs[0].tolist() == [[5, 5], [0, 0]]


def test_constant_and_identity_bundles_are_refused():
    same = [[1, 2], [2, 1]]
    assert certify([(A, same), (B, same)], [T], None, expected=[same]).reason == "constant_output"
    assert certify([(A, A), (B, B)], [T], None, expected=[T]).reason == "identity"


def test_generator_rule_must_fit():
    rot = Rule.single("rot", 1)
    flip = Rule.single("flip", "lr")
    assert certify(pairs(rot, [A, B]), [T], flip).reason == "rule_inconsistent"


def test_two_step_rival_is_found():
    """rot 90 then recolor 3->4 is the rule; the search finds it as a composition and certifies."""
    rule = Rule.chain(("rot", (1,)), ("recolor", (3, 4)))
    grids = [A, B, C, [[3, 5, 0], [0, 0, 5]]]
    cert = certify(pairs(rule, grids), [T], rule)
    assert cert.outputs is not None
    assert not cert.single_alive
    assert any(tok.startswith("rot(1)+") for tok in cert.consistent)


def test_composition_that_is_a_single_rule_is_flagged():
    rule = Rule.chain(("rot", (1,)), ("rot", (1,)))
    cert = certify(pairs(rule, [A, B, C]), [T], rule)
    assert cert.outputs is not None and cert.single_alive


def test_unobservable_step_does_not_block():
    """No train input has an object of 4 cells; the test does.  'remove 4-cell objects, then
    rotate' fits the pairs, but its first step never shows in them, so it is dominated by the
    plain rotation and does not make the bundle ambiguous."""
    rot = Rule.single("rot", 1)
    t = [[0, 0, 0, 0, 0],
         [0, 3, 3, 3, 0],
         [0, 0, 0, 3, 0],
         [5, 0, 0, 0, 0]]
    train = pairs(rot, [A, B, C])
    dominated = Rule.chain(("remove_size", ("eq", 4, 4)), ("rot", (1,)))
    assert all(dominated.apply(x) == y for x, y in train)
    assert dominated.apply(t) != rot.apply(t)
    cert = certify(train, [t], rot)
    assert cert.outputs is not None and cert.outputs[0].tolist() == rot.apply(t)


def test_observable_alternative_blocks():
    """flip-lr vs transpose-then-rotate style: with only square symmetric-ish inputs where two
    different rules agree, a test that separates them is refused."""
    diag = [[1, 0, 0], [0, 2, 0], [0, 0, 1]]
    diag2 = [[3, 0, 0], [0, 3, 0], [0, 0, 4]]
    tr = Rule.single("transpose", "main")
    # both inputs are symmetric about the main diagonal: transpose is the identity on them, so
    # use rot 180 instead and check that transpose-then-rot180 (= anti-transpose) is a rival
    r180 = Rule.single("rot", 2)
    cert = certify(pairs(r180, [diag, diag2]), [[[1, 2, 0], [0, 0, 0], [0, 0, 3]]], r180)
    assert cert.outputs is None
    assert tr.apply(diag) == diag


def test_scalar_certificate():
    count3 = ScalarRule("count_color", (3,))
    train = [(A, 3), (B, 3)]
    assert certify_scalar(train, [T], count3).reason == "repeated_value"
    train = [(A, 3), ([[3, 3, 3, 3, 1]], 4), ([[3, 1], [1, 1]], 1)]
    cert = certify_scalar(train, [T], count3)
    assert cert.values == [3], cert.reason


def _fits_but_differs(rival, train, test, rule):
    assert all(rival.apply(x) == y for x, y in train), rival.token
    assert rival.apply(test) != rule.apply(test), rival.token


def test_ray_that_meets_an_obstacle_only_in_the_test_is_refused():
    """No train ray meets an obstacle, the test's does: 'stop at it', 'pass it' and 'paint over
    it' all fit the pairs (verifier finding, rays 12/200)."""
    rule = Rule.single("rays", 2, "down")
    xs = [[[0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 3], [0, 0, 0, 3]],
          [[2, 0, 0], [0, 0, 3], [0, 0, 3], [0, 0, 0]],
          [[0, 0, 2, 0, 0], [3, 3, 0, 0, 0], [0, 0, 0, 0, 0]]]
    train = pairs(rule, xs)
    t = [[0, 2, 0], [0, 3, 0], [0, 0, 0], [0, 0, 0]]
    for rival in (Rule.single("rays_skip", 2, "down"), Rule.single("rays_over", 2, "down")):
        _fits_but_differs(rival, train, t, rule)
    assert certify(train, [t], rule).outputs is None
    # a train pair whose ray meets an obstacle with background beyond settles it
    shown = train + pairs(rule, [[[0, 2, 0], [0, 3, 0], [0, 0, 0]]])
    assert certify(shown, [t], rule).outputs is not None


def test_exact_size_vs_fewer_than_is_refused():
    """Objects of 3 cells are recoloured, 6-cell ones kept; a 2-cell object in the test separates
    'exactly 3' from 'fewer than 4' (verifier finding, recolor_size 9/199)."""
    rule = Rule.single("recolor_size", 3, 4, 4)
    xs = [[[1, 1, 1, 0, 0], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 1]],
          [[0, 0, 0, 0, 0, 0], [1, 0, 1, 1, 0, 0], [1, 0, 1, 1, 0, 0], [1, 0, 1, 1, 0, 0]],
          [[1, 1, 0, 0, 0, 0], [1, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 0, 0]]]
    train = pairs(rule, xs)
    t = [[1, 1, 0, 0, 0], [0, 0, 0, 1, 0], [1, 0, 0, 1, 0], [1, 0, 0, 1, 0]]
    _fits_but_differs(Rule.single("recolor_size_cmp", "lt", 4, 4, 4), train, t, rule)
    assert certify(train, [t], rule).outputs is None


def test_crop_of_touching_colours_is_refused():
    """The largest same-colour object vs the largest group of touching cells of any colours
    (verifier finding, crop_object 3/195)."""
    rule = Rule.single("crop_object", "largest", 4)
    xs = [[[0, 0, 0, 0, 0], [0, 1, 1, 0, 0], [0, 1, 1, 0, 2], [0, 0, 0, 0, 2]],
          [[0, 0, 0, 0], [0, 2, 2, 2], [0, 2, 0, 2], [1, 0, 0, 0]],
          [[1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 2, 0], [0, 0, 0, 0, 2, 0]]]
    train = pairs(rule, xs)
    t = [[0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 0, 0, 2, 0], [0, 0, 0, 2, 0], [0, 0, 0, 0, 0]]
    _fits_but_differs(Rule.single("crop_blob", "largest", 4), train, t, rule)
    assert certify(train, [t], rule).outputs is None


def test_tile_by_the_grids_own_size_is_a_rival():
    """Every train input is 3x3 and tiled 3x3: 'as many copies as it has rows' fits too."""
    rule = Rule.single("tile", 3, 3)
    xs = [[[1, 0, 0], [0, 2, 0], [0, 0, 1]], [[3, 3, 0], [0, 0, 0], [0, 4, 4]], [[0, 5, 0], [5, 5, 5], [0, 0, 0]]]
    train = pairs(rule, xs)
    t = [[1, 2], [2, 0]]
    _fits_but_differs(Rule.single("tile_self", "h"), train, t, rule)
    assert certify(train, [t], rule).outputs is None


def test_every_wrap_shift_is_a_rival():
    """On width-6 grids one column left is also five columns right; a width-5 test separates them."""
    from pv2.arc.grids import Info
    from pv2.arc.rivals import general_rolls
    rule = Rule.single("roll", 0, -1)
    xs = [[[1, 0, 0, 2, 0, 0], [0, 3, 0, 0, 0, 0]],
          [[0, 0, 4, 0, 1, 0], [2, 0, 0, 0, 0, 3], [0, 1, 0, 0, 0, 0]],
          [[4, 4, 0, 0, 0, 3]]]
    train = pairs(rule, xs)
    shifts = general_rolls([(Info.of(x), Info.of(y)) for x, y in train])
    assert (0, -1) in shifts and (0, 5) in shifts and (0, 11) in shifts
    t = [[1, 2, 0, 0, 3], [0, 0, 4, 0, 0]]
    assert certify(train, [t], rule).outputs is None
    wide = train + pairs(rule, [[[1, 0, 2, 0, 0], [0, 3, 0, 0, 4]]])        # a width-5 pair settles it
    assert certify(wide, [t], rule).outputs is not None


def test_keeping_the_largest_group_is_a_rival_of_removing_lone_cells():
    """With one group plus lone cells in every train input, 'remove lone cells' and 'keep only the
    largest group' agree; a test with two groups separates them."""
    rule = Rule.single("remove_isolated", 8)
    xs = [[[0, 0, 0, 0, 2], [0, 1, 1, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 0, 0], [3, 0, 0, 0, 0]],
          [[4, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 2, 2, 2, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 3]],
          [[0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 4], [0, 1, 0, 0], [0, 0, 0, 0]]]
    train = pairs(rule, xs)
    t = [[0, 0, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0], [0, 0, 0, 0, 2, 0], [0, 3, 0, 0, 2, 0], [0, 0, 0, 0, 2, 0],
         [0, 0, 0, 0, 0, 0]]
    _fits_but_differs(Rule.single("keep_blob", "largest", 8), train, t, rule)
    assert certify(train, [t], rule).outputs is None


def test_hollowing_where_the_edge_counts_as_enclosing_is_a_rival():
    """No train rectangle touches the grid edge; the test's does (task 49 in hidden mode)."""
    rule = Rule.single("hollow_rects", 1)
    xs = [[[0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 0, 0], [0, 1, 1, 1, 0, 0], [0, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0]],
          [[0, 0, 0, 0, 0, 0, 0], [0, 2, 2, 2, 2, 0, 0], [0, 2, 2, 2, 2, 0, 0], [0, 2, 2, 2, 2, 0, 0],
           [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]],
          [[0, 0, 0, 0, 0, 0], [0, 3, 3, 3, 3, 0], [0, 3, 3, 3, 3, 0], [0, 3, 3, 3, 3, 0], [0, 3, 3, 3, 3, 0],
           [0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0]]]
    train = pairs(rule, xs)
    t = [[1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 2, 2], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0]]
    _fits_but_differs(Rule.single("hollow_edge", 4), train, t, rule)
    assert certify(train, [t], rule).outputs is None


def test_object_size_rules_beyond_twenty_cells_are_rivals():
    """Every train input has one object of 24 cells and smaller ones: 'keep the largest object' and
    'remove the objects of fewer than 22 cells' agree; a test with two big objects separates them."""
    from pv2.arc.grids import Info
    from pv2.arc.rivals import size_thresholds
    rule = Rule.single("keep_extreme", "largest", 4)

    def grid(big, small):
        g = [[0] * 14 for _ in range(12)]
        for k, (r0, c0, h, w) in enumerate(big + small):
            for r in range(r0, r0 + h):
                for c in range(c0, c0 + w):
                    g[r][c] = 1 + k % 3
        return g
    xs = [grid([(0, 0, 4, 6)], [(6, 0, 2, 2), (9, 9, 1, 3)]),
          grid([(5, 5, 3, 8)], [(0, 0, 1, 4), (0, 8, 2, 3)]),
          grid([(1, 1, 6, 4)], [(8, 8, 2, 5)])]
    train = pairs(rule, xs)
    t = grid([(0, 0, 4, 6), (6, 6, 4, 6)], [(10, 0, 1, 2)])
    _fits_but_differs(Rule.single("remove_size", "lt", 22, 4), train, t, rule)
    found = size_thresholds([(Info.of(x), Info.of(y)) for x, y in train], [Info.of(t)])
    # thresholds between two object sizes behave alike; the search keeps one per range (here 24)
    assert ("remove_size", ("lt", 24, 4)) in found
    assert certify(train, [t], rule).outputs is None
