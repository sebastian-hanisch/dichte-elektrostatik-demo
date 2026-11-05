"""Lineare Zuordnung (Ungarische Methode): Handmatrizen, Gegenprobe durch Aufzählung, Eigenschaften, Randfälle."""

from itertools import permutations

import numpy as np
import pytest

from dla_assignment import hungarian


def _brute(a):
    n = len(a)
    return min(sum(a[i][p[i]] for i in range(n)) for p in permutations(range(n)))


def test_hand_matrix_by_hand():
    """Kosten 4 1 3 / 2 0 5 / 3 2 2: die kleinste Zuordnung ist Zeile 1 → Spalte 2 (1), Zeile 2 → Spalte 1 (2), Zeile 3 → Spalte 3 (2): Summe 5."""
    a = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
    cols, total = hungarian(a)
    assert total == 5 and sorted(cols.tolist()) == [0, 1, 2] and sum(a[i, cols[i]] for i in range(3)) == 5 and cols.tolist() == [1, 0, 2]


def test_the_dla_matrices_of_the_readme_by_hand():
    """Drei Bauteile auf einer Zeile: L = 5 5 5 / 7 6 7 / 1 1 1, kleinste Zuordnung 5 + 6 + 1 = 12; Raster 2×2: 4, 5, 5, 6 in jeder Spalte gleich: 20; Zeile mit Lücke: 3 2 2 3 je Zeile: 10."""
    assert hungarian(np.array([[5, 5, 5], [7, 6, 7], [1, 1, 1]]))[1] == 12
    assert hungarian(np.array([[4, 4, 4, 4], [5, 5, 5, 5], [5, 5, 5, 5], [6, 6, 6, 6]]))[1] == 20
    assert hungarian(np.array([[3, 2, 2, 3]] * 4))[1] == 10


@pytest.mark.parametrize("seed", range(12))
def test_integer_matrices_equal_the_enumeration(seed):
    rng = np.random.default_rng(seed)
    n = 2 + seed % 5
    a = rng.integers(-5, 30, size=(n, n))
    cols, total = hungarian(a)
    assert total == _brute(a.tolist()) and sorted(cols.tolist()) == list(range(n)) and total == sum(int(a[i, cols[i]]) for i in range(n))


@pytest.mark.parametrize("seed", range(6))
def test_float_matrices_equal_the_enumeration(seed):
    rng = np.random.default_rng(100 + seed)
    a = rng.random((5, 5)) * 10
    cols, total = hungarian(a)
    assert total == pytest.approx(_brute(a.tolist()), abs=1e-9) and sorted(cols.tolist()) == list(range(5))


def test_adding_a_constant_to_a_row_or_column_shifts_the_optimum_by_that_constant():
    rng = np.random.default_rng(3)
    a = rng.integers(0, 20, size=(6, 6))
    base = hungarian(a)[1]
    b = a.copy()
    b[2, :] += 7
    c = a.copy()
    c[:, 4] += 3
    assert hungarian(b)[1] == base + 7 and hungarian(c)[1] == base + 3


def test_transposition_and_permutation_of_rows_do_not_change_the_value():
    rng = np.random.default_rng(4)
    a = rng.integers(0, 50, size=(7, 7))
    base = hungarian(a)[1]
    assert hungarian(a.T)[1] == base and hungarian(a[rng.permutation(7)])[1] == base and hungarian(a[:, rng.permutation(7)])[1] == base


def test_ties_still_give_a_valid_optimal_assignment():
    a = np.ones((5, 5), dtype=int)
    cols, total = hungarian(a)
    assert total == 5 and sorted(cols.tolist()) == list(range(5))


def test_edge_cases():
    assert hungarian(np.zeros((0, 0)))[1] == 0
    c1, t1 = hungarian(np.array([[7]]))
    assert t1 == 7 and c1.tolist() == [0]
    with pytest.raises(ValueError):
        hungarian(np.ones((2, 3)))
    with pytest.raises(ValueError):
        hungarian(np.ones(4))
    with pytest.raises(ValueError):
        hungarian(np.array([[1.0, float("nan")], [2.0, 3.0]]))
    with pytest.raises(ValueError):
        hungarian(np.array([[1.0, float("inf")], [2.0, 3.0]]))


def test_large_integer_costs_stay_exact():
    a = np.array([[10**12, 10**12 + 1], [10**12 + 2, 10**12 + 5]], dtype=np.int64)
    assert hungarian(a)[1] == 2 * 10**12 + 3
