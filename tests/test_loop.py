"""Look-ahead-Schleife: Pseudo-Anker im System, die beiden Legalisierer, die Runden; Handwerte, Eigenschaften, Randfälle (DEMO-PLAYBOOK §1b, ORACLE-PLAYBOOK §5)."""

import numpy as np
import pytest

import dla_constants as C
import dla_instance as I
import dla_loop as Lp
import dla_place as Pl


def _hand_pads():
    w = np.zeros((3, 2))
    w[0, 0] = w[2, 1] = 2.0
    return Pl.Pads(np.array([[-1.0, 0.0], [3.0, 0.0]]), w, 2.0, 1.0)


def test_hand_example_the_anchored_system_by_hand(hand):
    """Wie Stück 9 (A–B 5, B–C 1, Pads bei −1 an A und 3 an C mit Gewicht 2), dazu ein Pseudo-Anker mit Gewicht 1 je Bauteil am Platz 0, 1, 2: 8x_A − 5x_B = −2, 7x_B − 5x_A − x_C = 1, 4x_C − x_B = 8 → x_B = 14/29, x_A = 3/58, x_C = 123/58."""
    X = Lp.positions_anchored(hand, _hand_pads(), hand.xy.astype(float), 1.0)
    assert X[:, 0] == pytest.approx([3 / 58, 14 / 29, 123 / 58], abs=1e-5) and np.allclose(X[:, 1], 0, atol=1e-5)


def test_anchor_weight_zero_is_stueck_nine_and_a_large_weight_pins_the_cells():
    inst = I.make_instance("uniform", 8, 100002)
    pads = Pl.make_pads(inst, 4, 100002)
    anchors = inst.xy[[3, 1, 5, 0, 2, 4, 7, 6]].astype(float)
    for r in (0, 3):
        assert np.allclose(Lp.positions_anchored(inst, pads, anchors, 0.0, r), Pl.positions(inst, pads, r), atol=1e-9)
    X = Lp.positions_anchored(inst, pads, anchors, 1e6)
    assert np.abs(X - anchors).max() < 1e-3


def test_the_anchored_solution_zeroes_the_gradient():
    inst = I.make_instance("cluster", 8, 100001)
    pads = Pl.make_pads(inst, 4, 100001)
    anchors = inst.xy.astype(float)
    lam = 1.7
    X = Lp.positions_anchored(inst, pads, anchors, lam)
    A = Pl.system_matrix(inst, pads) + (lam + C.CENTER_ANCHOR) * np.eye(8)
    center = inst.xy.astype(float).mean(axis=0)
    for ax in (0, 1):
        b = (pads.w * pads.pos[None, :, ax]).sum(axis=1) + lam * anchors[:, ax] + C.CENTER_ANCHOR * center[ax]
        assert np.allclose(A @ X[:, ax], b, atol=1e-8)


def test_bisect_by_hand_on_the_two_by_two_and_the_two_by_three_grid():
    hand = I.hand_instance(C.HAND_GRID_FLOWS, *C.HAND_GRID_GRID)
    perm = Lp.bisect_legalize(hand, np.array([[0.2, 0.1], [0.1, 0.9], [0.9, 0.2], [0.8, 0.8]]))
    assert [tuple(int(c) for c in hand.xy[q]) for q in perm] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    # 2×3: die längere Seite sind die Spalten (3 > 2): zwei Bauteile links, vier rechts; die vier rechten werden nach der Spalte (2 gegen 1) und dann nach y geteilt
    inst = I.make_instance("netlist", 6, 1)
    X = np.array([[0.1, 0.2], [2.0, 0.9], [1.0, 0.1], [0.2, 0.8], [1.9, 0.1], [1.1, 0.9]])
    slots = [tuple(int(c) for c in inst.xy[p]) for p in Lp.bisect_legalize(inst, X)]
    assert {slots[0], slots[3]} == {(0, 0), (0, 1)} and slots[0] == (0, 0) and slots[3] == (0, 1)
    assert slots[2][0] == 1 and slots[5][0] == 1 and slots[1][0] == 2 and slots[4][0] == 2 and set(slots) == {(c, r) for c in range(3) for r in range(2)}


def test_both_legalizers_return_permutations_for_every_size_and_random_positions():
    rng = np.random.default_rng(7)
    for n in C.SIZES:
        inst = I.make_instance("uniform", n, 100000)
        for _ in range(3):
            X = rng.uniform(-1, 4, size=(n, 2))
            for kind in C.LEGALIZERS:
                assert sorted(Lp.legalize(inst, X, kind)) == list(range(n))
    with pytest.raises(ValueError):
        Lp.legalize(I.make_instance("uniform", 6, 1), np.zeros((6, 2)), "x")


def test_bisect_keeps_the_order_along_the_first_cut_and_ties_are_decided_by_the_number():
    inst = I.make_instance("uniform", 8, 100000)
    perm = Lp.bisect_legalize(inst, np.tile([1.0, 1.0], (8, 1)))              # alle Orte gleich: die Nummern entscheiden
    assert sorted(perm) == list(range(8)) and perm == Lp.bisect_legalize(inst, np.tile([1.0 + 1e-14, 1.0], (8, 1)))
    X = np.random.default_rng(3).uniform(0, 3, size=(8, 2))
    perm = Lp.bisect_legalize(inst, X)
    left = [i for i in range(8) if inst.xy[perm[i]][0] <= 1]                  # 2×4: die erste Teilung geht nach x in zwei Hälften zu je vier Bauteilen
    right = [i for i in range(8) if inst.xy[perm[i]][0] >= 2]
    assert len(left) == len(right) == 4 and max(X[left, 0]) <= min(X[right, 0]) + 1e-12


def test_the_loop_has_the_rounds_the_weights_and_consistent_values(land6):
    inst, ls = land6
    pads = Pl.make_pads(inst, 4, 35)
    c = Pl.linear_costs(inst, pads)
    base = Lp.base_weight(inst, pads)
    ref = Pl.reference(inst, c, 35)[0]
    for kind in C.LEGALIZERS:
        lp = Lp.run_loop(inst, pads, c, kind, 8, 2, 0.2)
        assert [r.t for r in lp.rounds] == list(range(9)) and lp.rounds[0].lam == 0 and [r.lam for r in lp.rounds[1:]] == pytest.approx([0.2 * base * t for t in range(1, 9)])
        for r in lp.rounds:
            assert sorted(r.perm) == list(range(6)) and r.upper == pytest.approx(Pl.cost_of(r.perm, inst.F, inst.D, c)) and r.lower == pytest.approx(Pl.continuous_length(inst, r.X, pads))
        assert lp.rounds[0].perm == Lp.legalize(inst, Pl.positions(inst, pads, 2), kind) and lp.uppers == [r.upper for r in lp.rounds] and min(lp.uppers) >= ref - 1e-9


def test_changes_last_change_and_the_lower_length_rises():
    rises, total, changed = 0, 0, 0
    for fam in C.FAMILIES:
        for seed in range(100000, 100004):
            inst = I.make_instance(fam, 8, seed)
            pads = Pl.make_pads(inst, 2, seed)
            c = Pl.linear_costs(inst, pads)
            lp = Lp.run_loop(inst, pads, c, "assign", 15, 2, 0.2)
            total += 1
            rises += all(lp.rounds[i + 1].lower >= lp.rounds[i].lower - 1e-9 for i in range(15)) and lp.rounds[-1].lower > lp.rounds[0].lower
            changed += lp.changed
            assert lp.changes >= (1 if lp.changed else 0) and (lp.last_change == 0) == (lp.changes == 0) and lp.last_change <= 15
            if not lp.changed:
                assert lp.rounds[-1].upper == lp.rounds[0].upper
    assert rises >= 0.9 * total and changed >= 1


def test_the_loop_does_not_depend_on_rounding_noise(monkeypatch):
    inst = I.make_instance("cluster", 8, 38)
    pads = Pl.make_pads(inst, 4, 38)
    c = Pl.linear_costs(inst, pads)
    base = {leg: Lp.run_loop(inst, pads, c, leg, 15, 2, 0.2) for leg in C.LEGALIZERS}
    orig = Lp.solve_axis_anchored
    rng = np.random.default_rng(11)
    monkeypatch.setattr(Lp, "solve_axis_anchored", lambda *a, **k: orig(*a, **k) + 1e-13 * rng.standard_normal(len(a[0])))
    for leg in C.LEGALIZERS:
        noisy = Lp.run_loop(inst, pads, c, leg, 15, 2, 0.2)
        assert [r.perm for r in noisy.rounds] == [r.perm for r in base[leg].rounds] and noisy.uppers == base[leg].uppers


def test_base_weight_and_edge_cases():
    inst = I.make_instance("star", 6, 100000)
    assert Lp.base_weight(inst, Pl.make_pads(inst, 4, 1)) == Pl.make_pads(inst, 4, 1).base and Lp.base_weight(inst, Pl.make_pads(inst, 0, 1)) >= 1
    z = I.make_instance("sparse", 6, 100000)
    pads = Pl.make_pads(z, 2, 1)
    lp = Lp.run_loop(z, pads, Pl.linear_costs(z, pads), "assign", 0, 0, 0.2)
    assert len(lp.rounds) == 1 and lp.changes == 0 and not lp.changed and lp.last_change == 0
    lp0 = Lp.run_loop(z, Pl.make_pads(z, 0, 1), np.zeros((6, 6)), "bisect", 3, 2, 0.2)
    assert len(lp0.rounds) == 4 and all(sorted(r.perm) == list(range(6)) for r in lp0.rounds)
