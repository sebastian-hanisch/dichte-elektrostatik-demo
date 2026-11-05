"""Strafverfahren: glatte Länge, Stufen, Zählung, Rundung, Reproduzierbarkeit und Rauschtest; Orakel: einzeln gerechnete Kosten, Aufzählung, `scipy`."""

from itertools import permutations

import numpy as np
import pytest

import dla_constants as C
import dla_instance as I
import dla_penalty as Pn
import dla_place as Pl


def independent_cost(perm, inst, pads):
    """Kosten als einfache Doppelschleife: Σ_{i<k} F·|Δ|₁ + Σ_{i,Pad} w·|Δ|₁."""
    xy = inst.xy
    total = 0.0
    for i in range(inst.n):
        for k in range(i + 1, inst.n):
            total += float(inst.F[i, k]) * (abs(int(xy[perm[i]][0]) - int(xy[perm[k]][0])) + abs(int(xy[perm[i]][1]) - int(xy[perm[k]][1])))
        for j in range(pads.count):
            total += float(pads.w[i, j]) * (abs(xy[perm[i]][0] - pads.pos[j][0]) + abs(xy[perm[i]][1] - pads.pos[j][1]))
    return total


def loop_smooth(X, inst, pads):
    """Glatte Länge als Doppelschleife."""
    g = C.EPS
    total = 0.0
    for i in range(inst.n):
        for k in range(i + 1, inst.n):
            total += float(inst.F[i, k]) * sum(np.sqrt((X[i, d] - X[k, d]) ** 2 + g * g) for d in range(2))
        for j in range(pads.count):
            total += float(pads.w[i, j]) * sum(np.sqrt((X[i, d] - pads.pos[j][d]) ** 2 + g * g) for d in range(2))
    return total


def setup(family="cluster", n=8, seed=48, p=4):
    inst = I.make_instance(family, n, seed)
    pads = Pl.make_pads(inst, p, seed)
    return inst, pads, Pl.linear_costs(inst, pads)


def test_smooth_length_equals_the_double_loop_and_its_gradient_the_finite_differences():
    inst, pads, _ = setup()
    X = np.random.default_rng(3).uniform(-0.5, 3.5, size=(8, 2))
    F = inst.F.astype(float)
    val, g = Pn.smooth_length(X, F, pads)
    assert val == pytest.approx(loop_smooth(X, inst, pads))
    num = np.zeros_like(X)
    for i in range(8):
        for a in range(2):
            Y, Z = X.copy(), X.copy()
            Y[i, a] += 1e-6
            Z[i, a] -= 1e-6
            num[i, a] = (Pn.smooth_length(Y, F, pads)[0] - Pn.smooth_length(Z, F, pads)[0]) / 2e-6
    assert g == pytest.approx(num, abs=1e-5)
    assert val >= Pl.continuous_length(inst, X, pads) - 1e-9           # sqrt(d² + γ²) ≥ |d|


def test_the_stages_follow_the_schedule():
    inst, pads, c = setup()
    pen = Pn.run_penalty(inst, pads, c, "electro", 0.5, 3.0, 48)
    assert len(pen.stages) == C.STAGES and [s.t for s in pen.stages] == list(range(C.STAGES))
    assert [s.side for s in pen.stages] == pytest.approx([3.0 - 2.0 * k / (C.STAGES - 1) for k in range(C.STAGES)])
    lam = [s.lam for s in pen.stages]
    assert lam[0] == pytest.approx(pen.lam0) and all(lam[k + 1] == pytest.approx(C.GROWTH * lam[k]) for k in range(C.STAGES - 1))
    ev = [s.evals for s in pen.stages]
    assert all(b > a for a, b in zip(ev, ev[1:])) and ev[0] >= 2 and pen.evals == ev[-1]
    assert ev[0] <= C.INNER * 6 and ev[-1] <= C.STAGES * C.INNER * 6      # je Schritt höchstens eine Auswertung und wenige Rückverfolgungen (gemessen: rund 77 je Stufe)


def test_the_rounded_arrangements_are_permutations_with_the_independent_cost():
    for fam in ("cluster", "uniform", "star"):
        inst, pads, c = setup(fam, 8, 100003)
        for model in C.MODELS:
            pen = Pn.run_penalty(inst, pads, c, model, 0.5, 3.0, 100003)
            for s in pen.stages:
                assert sorted(s.perm) == list(range(8))
                assert s.cost == pytest.approx(independent_cost(s.perm, inst, pads))
                assert s.cont == pytest.approx(Pl.continuous_length(inst, s.X, pads)) and s.disp == pytest.approx(float(np.abs(s.X - inst.xy[s.perm].astype(float)).sum()))


def test_the_positions_stay_inside_the_region():
    inst, pads, c = setup()
    for margin in C.MARGIN_CHOICES:
        pen = Pn.run_penalty(inst, pads, c, "electro", margin, 3.0, 48)
        lo, hi = np.array([-margin, -margin]), np.array([3.0 + margin, 1.0 + margin])
        for s in pen.stages:
            assert (s.X >= lo - 1e-9).all() and (s.X <= hi + 1e-9).all()


def test_no_stage_beats_the_enumerated_optimum():
    for fam in C.FAMILIES:
        inst, pads, c = setup(fam, 6, 100007, 4)
        ref = min(independent_cost(q, inst, pads) for q in permutations(range(6)))
        assert Pl.reference(inst, c, 100007)[0] == pytest.approx(ref)
        for model in C.MODELS:
            assert all(x >= ref - 1e-9 for x in Pn.run_penalty(inst, pads, c, model, 0.5, 3.0, 100007).costs)


def test_the_run_is_deterministic():
    inst, pads, c = setup()
    a = Pn.run_penalty(inst, pads, c, "electro", 0.5, 3.0, 48)
    b = Pn.run_penalty(inst, pads, c, "electro", 0.5, 3.0, 48)
    assert a.costs == b.costs and all(np.array_equal(x.X, y.X) for x, y in zip(a.stages, b.stages)) and a.pick("best").perm == b.pick("best").perm


@pytest.mark.parametrize("family,seed", [("cluster", 48), ("cluster", 38), ("netlist", 100003), ("uniform", 100005), ("sparse", 100011)])
def test_rounding_noise_in_the_start_does_not_change_the_arrangement_at_n8(family, seed):
    """Lehre aus Stück 6, 9, 10 und 12: Gleitkommarauschen darf die Anordnung nicht ändern (hier: Rauschen 1e-13 im Startwert, n = 8)."""
    inst, pads, c = setup(family, 8, seed)
    x0 = Pn.start_positions(inst, pads, seed)
    a = Pn.run_penalty(inst, pads, c, "electro", 0.5, 3.0, seed, x0=x0)
    noise = np.random.default_rng(11).normal(size=x0.shape) * 1e-13
    b = Pn.run_penalty(inst, pads, c, "electro", 0.5, 3.0, seed, x0=x0 + noise)
    assert a.stages[-1].perm == b.stages[-1].perm and a.costs[-1] == b.costs[-1] and min(a.costs) == min(b.costs)


def test_the_electrostatic_overflow_falls_and_the_clump_without_a_penalty_stays():
    inst, pads, c = setup()
    e = Pn.run_penalty(inst, pads, c, "electro", 0.5, 3.0, 48)
    n0 = Pn.run_penalty(inst, pads, c, "none", 0.5, 3.0, 48)
    assert e.stages[-1].overflow < e.stages[0].overflow - 0.3 and n0.stages[-1].overflow > 0.7 and n0.lam0 == 0.0 and n0.stages[-1].spread < 0.3 < e.stages[-1].spread


def test_pick_and_best_index_follow_the_costs():
    inst, pads, c = setup()
    pen = Pn.run_penalty(inst, pads, c, "electro", 0.5, 3.0, 48)
    assert pen.pick("last") is pen.stages[-1] and pen.pick("best").cost == min(pen.costs) and pen.costs[pen.best_index] == min(pen.costs)
    assert all(pen.costs[k] > min(pen.costs) + 1e-9 for k in range(pen.best_index))     # die erste Stufe mit dem kleinsten Wert


def test_start_positions_are_the_placement_plus_a_small_fixed_jitter():
    inst, pads, _ = setup()
    X = Pn.start_positions(inst, pads, 48)
    base = Pl.positions(inst, pads, C.ROUND_REWEIGHTS)
    assert np.abs(X - base).max() <= C.JITTER / 2 + 1e-12 and np.abs(X - base).max() > 0 and np.array_equal(X, Pn.start_positions(inst, pads, 48))
    assert not np.array_equal(X, Pn.start_positions(inst, pads, 49)) and np.array_equal(Pn.start_positions(inst, pads, 48, jitter=0.0), base)


def test_unknown_model_is_rejected_and_the_hand_instance_runs():
    inst, pads, c = setup()
    with pytest.raises(ValueError):
        Pn.run_penalty(inst, pads, c, "gauss", 0.5, 3.0, 48)
    hand = I.hand_instance(C.HAND_GRID_FLOWS, *C.HAND_GRID_GRID)
    hp = Pl.make_pads(hand, 2, 1)
    pen = Pn.run_penalty(hand, hp, Pl.linear_costs(hand, hp), "electro", 0.5, 2.0, 1, stages=4, inner=10)
    assert len(pen.stages) == 4 and sorted(pen.stages[-1].perm) == [0, 1, 2, 3]


def test_the_rounding_is_an_optimal_assignment_for_scipy():
    opt = pytest.importorskip("scipy.optimize")
    for fam in C.FAMILIES:
        inst, pads, c = setup(fam, 8, 100008)
        s = Pn.run_penalty(inst, pads, c, "electro", 0.5, 3.0, 100008, stages=3, inner=10).stages[-1]
        d2 = ((s.X[:, None, :] - inst.xy[None, :, :].astype(float)) ** 2).sum(axis=2)
        r, cc = opt.linear_sum_assignment(d2)
        assert d2[np.arange(8), s.perm].sum() == pytest.approx(d2[r, cc].sum(), abs=1e-6)
