"""Die vollständige Landschaft: Handbeispiel, Gegenprobe durch Einzelrechnung, Eigenschaften (DEMO-PLAYBOOK §1b, ORACLE-PLAYBOOK §5)."""

from itertools import permutations

import numpy as np
import pytest

import dla_constants as C
import dla_cost as K
import dla_instance as I
import dla_landscape as L


def _inst(F, D=None):
    F = np.asarray(F, dtype=np.int16)
    n = len(F)
    D = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]).astype(np.int16) if D is None else np.asarray(D, dtype=np.int16)
    return I.Instance(n, 1, n, F, D, np.array([(i, 0) for i in range(n)]), tuple(f"B{i + 1}" for i in range(n)), (), 0, "test")


def _reference(inst):
    """Einzelrechnung ohne Vektorisierung: Kosten aller Anordnungen, schwache und strenge lokale Optima, Ende des steilsten Abstiegs von jedem Start."""
    n, F, D = inst.n, inst.F, inst.D
    pairs = K.pair_list(n)
    cost = {p: K.cost(p, F, D) for p in permutations(range(n))}

    def descend(p):
        cur = list(p)
        steps = 0
        while True:
            deltas = [K.swap_delta(cur, a, b, F, D) for a, b in pairs]
            j = min(range(len(pairs)), key=lambda i: (deltas[i], i))
            if deltas[j] >= 0:
                return tuple(cur), steps
            cur = K.swapped(cur, *pairs[j])
            steps += 1
    ends = {p: descend(p) for p in cost}
    local = {p for p in cost if all(K.swap_delta(p, a, b, F, D) >= 0 for a, b in pairs)}
    strict = {p for p in cost if all(K.swap_delta(p, a, b, F, D) > 0 for a, b in pairs)}
    return cost, ends, local, strict


def test_hand_landscape_by_hand(hand):
    """Sechs Anordnungen: Kosten 6, 6, 7, 7, 11, 11; beide Optima sind schwache lokale Optima (Gleichstand beim Tausch A↔C), keines ist streng; jeder Abstieg endet im Optimum."""
    ls = L.landscape(hand)
    assert ls.perms == 6 and ls.opt == 6 and ls.n_opt == 2 and ls.cost_values.tolist() == [6, 7, 11] and ls.cost_counts.tolist() == [2, 2, 2]
    assert ls.mean_cost == pytest.approx(8.0) and ls.mean_gap == pytest.approx(1 / 3)
    assert ls.n_local == 2 and ls.n_local_strict == 0 and ls.n_local_nonglobal == 0 and sorted(ls.local_basin.tolist()) == [3, 3]
    assert ls.success == 1.0 and ls.end_gap == 0.0 and ls.restarts == 1.0
    assert ls.mean_steps == pytest.approx(4 / 6) and ls.max_steps == 1
    assert ls.optima.tolist() == [[0, 1, 2], [2, 1, 0]]


@pytest.mark.parametrize("seed", range(4))
def test_vectorised_landscape_equals_the_single_computation(seed):
    rng = np.random.default_rng(seed)
    n = 5 + seed % 2
    F = np.triu(rng.integers(0, 6, size=(n, n)), 1)
    F = F + F.T
    xy = rng.integers(0, 4, size=(n, 2))
    inst = _inst(F, I.manhattan(xy))
    ls = L.landscape(inst)
    cost, ends, local, strict = _reference(inst)
    assert ls.opt == min(cost.values()) and ls.n_opt == sum(1 for c in cost.values() if c == ls.opt)
    assert {tuple(int(x) for x in r) for r in ls.local} == local and {tuple(int(x) for x in r) for r, s in zip(ls.local, ls.local_strict) if s} == strict
    endpoint = {}
    for p, (e, _) in ends.items():
        endpoint[e] = endpoint.get(e, 0) + 1
    assert {tuple(int(x) for x in r): int(b) for r, b in zip(ls.local, ls.local_basin)} == endpoint
    assert ls.mean_steps == pytest.approx(np.mean([s for _, s in ends.values()])) and ls.max_steps == max(s for _, s in ends.values())
    assert ls.success == pytest.approx(sum(1 for e, _ in ends.values() if cost[e] == ls.opt) / len(cost))


@pytest.mark.parametrize("family", C.FAMILIES)
def test_landscape_properties(family):
    ls = L.landscape(I.make_instance(family, 6, 7))
    assert ls.perms == 720 and int(ls.local_basin.sum()) == 720 and int(ls.cost_counts.sum()) == 720
    assert 0 < ls.success <= 1 and ls.n_local_strict <= ls.n_local and ls.n_local_nonglobal <= ls.n_local
    assert (ls.local_cost >= ls.opt).all() and ls.opt in ls.local_cost                              # jedes globale Optimum ist ein lokales; es gibt mindestens eines
    assert ls.end_gap >= 0 and ls.mean_gap >= ls.end_gap - 1e-12 if ls.opt else True
    assert ls.restarts == pytest.approx(1 / ls.success) and ls.mean_steps <= ls.max_steps


def test_every_global_optimum_is_a_local_optimum_and_the_basins_of_optima_give_the_success():
    inst = I.make_instance("cluster", 6, 9)
    ls = L.landscape(inst)
    optima = {tuple(int(x) for x in r) for r in ls.optima}
    local = {tuple(int(x) for x in r) for r in ls.local}
    assert optima <= local and ls.success == pytest.approx(int(ls.local_basin[ls.local_cost == ls.opt].sum()) / 720)


def test_the_end_gap_is_the_basin_weighted_excess():
    ls = L.landscape(I.make_instance("sparse", 6, 2))
    manual = sum(int(b) * (int(c) - ls.opt) for b, c in zip(ls.local_basin, ls.local_cost)) / (720 * ls.opt)
    assert ls.end_gap == pytest.approx(manual)


def test_all_perms_bounds_and_cache():
    assert L.all_perms(4).shape == (24, 4) and L.all_perms(4) is L.all_perms(4)
    with pytest.raises(ValueError):
        L.all_perms(1)
    with pytest.raises(ValueError):
        L.all_perms(11)


def test_costs_of_equals_single_costs():
    inst = I.make_instance("star", 6, 3)
    P = L.all_perms(6)[::53]
    assert L.costs_of(P, inst.F, inst.D).tolist() == [K.cost(p.tolist(), inst.F, inst.D) for p in P]
