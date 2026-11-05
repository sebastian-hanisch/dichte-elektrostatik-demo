"""Kosten und Tausch-Delta: Handwerte, Eigenschaften und Randfälle (DEMO-PLAYBOOK §1b, ORACLE-PLAYBOOK §5)."""

from itertools import permutations

import numpy as np
import pytest

import dla_constants as C
import dla_cost as K
import dla_instance as I


def test_hand_costs_by_hand(hand):
    """A–B 5, B–C 1 auf den Plätzen 0 1 2: p = Platz von A, B, C."""
    assert K.cost((0, 1, 2), hand.F, hand.D) == 5 * 1 + 1 * 1 == 6
    assert K.cost((1, 0, 2), hand.F, hand.D) == 5 * 1 + 1 * 2 == 7
    assert K.cost((0, 2, 1), hand.F, hand.D) == 5 * 2 + 1 * 1 == 11


def test_hand_swap_deltas_by_hand(hand):
    """Von (1, 0, 2) (Kosten 7): Tausch A↔B → (0, 1, 2) Kosten 6 (Δ = −1); A↔C → (2, 0, 1) Kosten 11 (Δ = +4); B↔C → (1, 2, 0) Kosten 7 (Δ = 0)."""
    p = (1, 0, 2)
    assert K.swap_delta(p, 0, 1, hand.F, hand.D) == -1
    assert K.swap_delta(p, 0, 2, hand.F, hand.D) == 4
    assert K.swap_delta(p, 1, 2, hand.F, hand.D) == 0
    assert K.swapped(p, 0, 1) == [0, 1, 2]


@pytest.mark.parametrize("family", C.FAMILIES)
def test_swap_delta_equals_the_difference_of_recomputed_costs(family):
    inst = I.make_instance(family, 6, 11)
    rng = np.random.default_rng(0)
    for _ in range(5):
        p = [int(x) for x in rng.permutation(6)]
        for a, b in K.pair_list(6):
            assert K.swap_delta(p, a, b, inst.F, inst.D) == K.cost(K.swapped(p, a, b), inst.F, inst.D) - K.cost(p, inst.F, inst.D)


def test_vectorised_deltas_equal_the_single_formula():
    inst = I.make_instance("skewed", 6, 3)
    Q = np.array(list(permutations(range(6)))[::37], dtype=np.intp)
    out = K.all_deltas(Q, inst.F, inst.D)
    pairs = K.pair_list(6)
    assert out.shape == (len(Q), 15)
    for i in range(0, len(Q), 5):
        for j, (a, b) in enumerate(pairs):
            assert out[i, j] == K.swap_delta(Q[i].tolist(), a, b, inst.F, inst.D)


def test_swap_is_symmetric_and_swapping_twice_restores():
    inst = I.make_instance("cluster", 6, 5)
    p = [3, 0, 5, 1, 4, 2]
    q = K.swapped(p, 1, 4)
    assert K.swap_delta(p, 1, 4, inst.F, inst.D) == K.swap_delta(p, 4, 1, inst.F, inst.D) == -K.swap_delta(q, 1, 4, inst.F, inst.D)
    assert K.swapped(q, 1, 4) == p


def test_rejections():
    inst = I.make_instance("uniform", 6, 1)
    with pytest.raises(ValueError):
        K.cost((0, 0, 1, 2, 3, 4), inst.F, inst.D)
    with pytest.raises(ValueError):
        K.cost((0, 1, 2), inst.F, inst.D)
    with pytest.raises(ValueError):
        K.swap_delta(list(range(6)), 2, 2, inst.F, inst.D)


def test_pair_list_counts():
    assert K.pair_list(3) == [(0, 1), (0, 2), (1, 2)] and [len(K.pair_list(n)) for n in (6, 8, 9)] == [15, 28, 36]
