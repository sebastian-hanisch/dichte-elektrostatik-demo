"""Quadratische Platzierung: Pads, lineares System, Kollaps ohne Anker, Umgewichtung, Rundung, Abstieg; Handwerte, Eigenschaften, Randfälle (DEMO-PLAYBOOK §1b, ORACLE-PLAYBOOK §5)."""

import numpy as np
import pytest

import dla_constants as C
import dla_cost as K
import dla_instance as I
import dla_place as Pl
from test_landscape import _inst


def _hand_pads():
    """Zwei Pads auf der Zeile der Plätze: links bei x = −1 an A (Gewicht 2), rechts bei x = 3 an C (Gewicht 2)."""
    w = np.zeros((3, 2))
    w[0, 0] = 2.0
    w[2, 1] = 2.0
    return Pl.Pads(np.array([[-1.0, 0.0], [3.0, 0.0]]), w, 2.0, 1.0)


def test_pads_geometry_weights_and_determinism():
    inst = I.make_instance("netlist", 8, 35)
    for p in C.PAD_CHOICES:
        pads = Pl.make_pads(inst, p, 35)
        assert pads.count == p and pads.pos.shape == (p, 2) and pads.w.shape == (8, p)
        if p:
            assert (np.count_nonzero(pads.w, axis=0) == C.PAD_LINKS).all() and np.allclose(pads.w.max(axis=0), pads.base)
            outside = (pads.pos[:, 0] < inst.xy[:, 0].min()) | (pads.pos[:, 0] > inst.xy[:, 0].max()) | (pads.pos[:, 1] < inst.xy[:, 1].min()) | (pads.pos[:, 1] > inst.xy[:, 1].max())
            assert outside.all()
    a, b = Pl.make_pads(inst, 4, 35), Pl.make_pads(inst, 4, 35)
    assert np.array_equal(a.pos, b.pos) and np.array_equal(a.w, b.w) and not np.array_equal(a.w, Pl.make_pads(inst, 4, 36).w)
    assert np.allclose(Pl.make_pads(inst, 4, 35, 4.0).w, 4 * a.w) and a.base >= 1 and a.base == float(int(a.base))


def test_linear_costs_and_cost_of_by_hand(hand):
    pads = _hand_pads()
    c = Pl.linear_costs(hand, pads)
    # Bauteil A (Zeile 0) hängt mit 2 an Pad (−1, 0): Platz 0, 1, 2 kosten 2·1, 2·2, 2·3; C hängt mit 2 an (3, 0): 2·3, 2·2, 2·1
    assert c.tolist() == [[2, 4, 6], [0, 0, 0], [6, 4, 2]]
    assert Pl.cost_of([0, 1, 2], hand.F, hand.D, c) == 10 and Pl.cost_of([2, 1, 0], hand.F, hand.D, c) == 5 + 1 + 6 + 6
    assert (Pl.linear_costs(hand, Pl.make_pads(hand, 0, 1)) == 0).all()


def test_hand_example_the_quadratic_solution_by_hand(hand):
    """A–B 5, B–C 1, Pads links (an A) und rechts (an C) mit Gewicht 2: x_A = −1/11, x_B = 3/11, x_C = 23/11 (von Hand: 7x_A = 5x_B − 2, 3x_C = x_B + 6, 6x_B = 5x_A + x_C → 44x_B = 12)."""
    X = Pl.positions(hand, _hand_pads())
    assert X[:, 0] == pytest.approx([-1 / 11, 3 / 11, 23 / 11], abs=1e-5) and np.allclose(X[:, 1], 0, atol=1e-5)
    assert Pl.round_to_slots(hand, X) == [0, 1, 2]


def test_the_solution_zeroes_the_gradient_and_is_the_minimum():
    rng = np.random.default_rng(1)
    for fam in C.FAMILIES:
        inst = I.make_instance(fam, 8, 100000)
        pads = Pl.make_pads(inst, 4, 100000)
        X = Pl.positions(inst, pads)
        A_ = Pl.system_matrix(inst, pads) + C.CENTER_ANCHOR * np.eye(8)
        center = inst.xy.astype(float).mean(axis=0)
        for ax in (0, 1):
            b = (pads.w * pads.pos[None, :, ax]).sum(axis=1) + C.CENTER_ANCHOR * center[ax]
            assert np.allclose(A_ @ X[:, ax], b, atol=1e-8)
        base = Pl.quadratic_value(inst, X, pads)
        for _ in range(5):
            assert Pl.quadratic_value(inst, X + 0.05 * rng.normal(size=X.shape), pads) >= base - 1e-7


def test_without_anchors_everything_collapses_to_one_point():
    for fam in C.FAMILIES:
        inst = I.make_instance(fam, 8, 100001)
        X = Pl.positions(inst, Pl.make_pads(inst, 0, 1))
        assert np.allclose(X, X[0]) and Pl.spread(inst, X) == pytest.approx(0.0, abs=1e-12) and Pl.overlap(inst, X) == 7
        assert np.allclose(X[0], inst.xy.astype(float).mean(axis=0))


def test_pads_move_the_solution_with_them_and_stronger_pads_spread_more():
    inst = I.make_instance("uniform", 8, 100002)
    pads = Pl.make_pads(inst, 4, 100002)
    shifted = Pl.Pads(pads.pos + np.array([3.0, -2.0]), pads.w, pads.base, pads.factor)
    X, Y = Pl.positions(inst, pads), Pl.positions(inst, shifted)
    assert np.allclose(Y - X, [3.0, -2.0], atol=1e-3)
    spreads = [Pl.spread(inst, Pl.positions(inst, Pl.make_pads(inst, 4, 100002, f))) for f in C.FACTOR_CHOICES]
    assert spreads == sorted(spreads) and spreads[0] < 0.3 < spreads[-1]


def test_components_without_a_path_to_a_pad_fall_into_the_middle():
    """Ein Flussgraph in zwei Teilen: nur ein Teil hängt an Pads, der andere fällt in die Mitte (schwacher Anker)."""
    F = np.zeros((6, 6), dtype=np.int16)
    for a, b in ((0, 1), (1, 2), (3, 4), (4, 5)):
        F[a, b] = F[b, a] = 3
    inst = _inst(F, I.manhattan(I.grid_xy(2, 3)))
    w = np.zeros((6, 1))
    w[0, 0] = 3.0
    pads = Pl.Pads(np.array([[-2.0, 0.0]]), w, 3.0, 1.0)
    X = Pl.positions(inst, pads)
    center = inst.xy.astype(float).mean(axis=0)
    assert np.allclose(X[3:], center, atol=1e-4) and X[0, 0] < -1.0


def test_reweighting_shortens_the_continuous_length_and_reaches_a_fixed_point():
    worse = 0
    for fam in C.FAMILIES:
        for seed in range(100000, 100005):
            inst = I.make_instance(fam, 8, seed)
            pads = Pl.make_pads(inst, 4, seed)
            l0, l5 = (Pl.continuous_length(inst, Pl.positions(inst, pads, r), pads) for r in (0, 5))
            worse += l5 > l0 + 1e-9
    assert worse == 0
    inst = I.make_instance("netlist", 8, 35)
    pads = Pl.make_pads(inst, 4, 35)
    steps = Pl.positions(inst, pads, 20, trace=True)
    assert len(steps) == 21 and np.abs(steps[20] - steps[19]).max() < 1e-4 and np.abs(steps[1] - steps[0]).max() > 1e-3
    assert len(Pl.positions(inst, Pl.make_pads(inst, 0, 1), 3, trace=True)) == 4


def test_spread_and_overlap_by_hand(hand):
    X = np.array([[0.0, 0.0], [0.0, 0.0], [2.0, 0.0]])
    assert Pl.overlap(hand, X) == 1 and Pl.spread(hand, hand.xy.astype(float)) == pytest.approx(1.0)
    assert Pl.spread(hand, np.zeros((3, 2))) == 0.0 and Pl.overlap(hand, hand.xy.astype(float)) == 0


def test_rounding_gives_a_permutation_and_does_not_depend_on_rounding_noise():
    rng = np.random.default_rng(2)
    for fam in C.FAMILIES:
        inst = I.make_instance(fam, 8, 100003)
        for p in (0, 4):
            pads = Pl.make_pads(inst, p, 100003)
            X = Pl.positions(inst, pads, 5)
            perm = Pl.round_to_slots(inst, X)
            assert sorted(perm) == list(range(8))
            for _ in range(3):
                assert Pl.round_to_slots(inst, X + 1e-13 * rng.normal(size=X.shape)) == perm
    assert Pl.tie_quantized(np.zeros((2, 2))).sum() == 0 and Pl.tie_quantized(np.array([[1.0, 1.0 + 1e-13]]))[0, 0] == Pl.tie_quantized(np.array([[1.0, 1.0 + 1e-13]]))[0, 1]


def test_all_costs_equals_the_single_cost_and_the_reference_is_the_minimum(land6):
    inst, ls = land6
    pads = Pl.make_pads(inst, 4, 35)
    c = Pl.linear_costs(inst, pads)
    P, costs = Pl.all_costs(inst, c)
    for k in (0, 17, 400, 719):
        assert costs[k] == pytest.approx(Pl.cost_of(P[k], inst.F, inst.D, c))
    ref, perm, exact, mean = Pl.reference(inst, c, 35)
    assert exact and ref == costs.min() and Pl.cost_of(perm, inst.F, inst.D, c) == ref and mean == pytest.approx(costs.mean())
    assert Pl.cost_of(list(range(6)), inst.F, inst.D, np.zeros((6, 6))) == K.cost(list(range(6)), inst.F, inst.D)


def test_swap_costs_descent_and_best_found(land6):
    inst, ls = land6
    pads = Pl.make_pads(inst, 4, 35)
    c = Pl.linear_costs(inst, pads)
    F, D = inst.F.astype(float), inst.D.astype(float)
    perm = [3, 1, 5, 0, 2, 4]
    ia, ib, cost = Pl.swap_costs(perm, F, D, c)
    for k in (0, 5, 14):
        q = perm[:]
        q[ia[k]], q[ib[k]] = q[ib[k]], q[ia[k]]
        assert cost[k] == pytest.approx(Pl.cost_of(q, F, D, c))
    p, v = Pl.descent(perm, F, D, c)
    assert v <= Pl.cost_of(perm, F, D, c) and v == pytest.approx(Pl.cost_of(p, F, D, c)) and Pl.swap_costs(p, F, D, c)[2].min() >= v - 1e-9
    best, bp = Pl.best_found(inst, c, 35)
    ref = Pl.reference(inst, c, 35)[0]
    assert best >= ref - 1e-9 and best == pytest.approx(ref) and sorted(bp) == list(range(6))
    assert Pl.best_found(inst, c, 35)[0] == best


def test_large_instances_have_no_enumerated_reference():
    inst = I.make_instance("sparse", 12, 100000)
    c = Pl.linear_costs(inst, Pl.make_pads(inst, 4, 100000))
    ref, perm, exact, mean = Pl.reference(inst, c, 100000)
    assert not exact and mean is None and sorted(perm) == list(range(12)) and ref == pytest.approx(Pl.cost_of(perm, inst.F, inst.D, c))


def test_edge_cases():
    two = _inst([[0, 3], [3, 0]])
    pads = Pl.Pads(np.array([[-1.0, 0.0]]), np.array([[1.0], [0.0]]), 1.0, 1.0)
    X = Pl.positions(two, pads)
    assert Pl.round_to_slots(two, X) == [0, 1] and X[0, 0] < X[1, 0]
    z = _inst(np.zeros((4, 4)))
    pz = Pl.make_pads(z, 2, 1)
    assert pz.base == 1.0 and Pl.positions(z, pz).shape == (4, 2)
    assert Pl.continuous_length(two, X, Pl.make_pads(two, 0, 1)) == pytest.approx(3 * abs(X[0, 0] - X[1, 0]) + 3 * abs(X[0, 1] - X[1, 1]))
