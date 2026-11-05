"""Instanzen: sechs Familien, zwei Arten von Plätzen, SplitMix64 (deterministisch auf jedem Rechner), Merkmale jeder Familie."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

import dla_constants as C
import dla_instance as I
from dla_rng import SplitMix64


def test_splitmix64_reference_values():
    """Die ersten Ausgaben von SplitMix64 mit Saat 0 (Referenzwerte der Veröffentlichung von Vigna)."""
    r = SplitMix64(0)
    assert [r.next() for _ in range(3)] == [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F]


def test_instances_are_deterministic_and_seed_dependent():
    for fam in C.FAMILIES:
        a, b, c = I.make_instance(fam, 8, 35), I.make_instance(fam, 8, 35), I.make_instance(fam, 8, 36)
        assert np.array_equal(a.F, b.F) and not np.array_equal(a.F, c.F), fam


def test_families_use_different_random_streams():
    mats = {fam: I.make_instance(fam, 8, 35).F.tolist() for fam in C.FAMILIES}
    assert len({str(m) for m in mats.values()}) == len(C.FAMILIES)


@pytest.mark.parametrize("n", C.SIZES)
@pytest.mark.parametrize("family", C.FAMILIES)
def test_the_flow_is_symmetric_nonnegative_integer_with_zero_diagonal(n, family):
    for seed in (0, 1, 35, 999999):
        inst = I.make_instance(family, n, seed)
        assert (inst.F == inst.F.T).all() and (np.diag(inst.F) == 0).all() and (inst.F >= 0).all() and inst.F.dtype.kind == "i"
        assert inst.D.shape == (n, n) and (inst.D == inst.D.T).all() and (np.diag(inst.D) == 0).all() and inst.xy.shape == (n, 2)
        assert len(inst.names) == n and len(set(inst.names)) == n and inst.family == family


def test_uniform_flows_lie_in_range_and_are_dense():
    vals = np.concatenate([np.triu(I.make_instance("uniform", 8, s).F, 1)[np.triu_indices(8, 1)] for s in range(30)])
    assert vals.min() == 0 and vals.max() == C.UNIFORM_MAX and 0.05 < (vals == 0).mean() < 0.2    # 0 mit Wahrscheinlichkeit 1/10


def test_sparse_flows_are_zero_or_positive_and_about_thirty_percent_dense():
    vals = np.concatenate([I.make_instance("sparse", 8, s).F[np.triu_indices(8, 1)] for s in range(40)])
    assert set(np.unique(vals)) <= set(range(0, 10)) and vals.min() == 0 and 0.2 < (vals > 0).mean() < 0.4 and vals[vals > 0].min() == 1


def test_cluster_flows_are_strong_inside_and_weak_between_the_halves():
    for n in C.SIZES:
        half = n // 2
        for s in range(10):
            F = I.make_instance("cluster", n, s).F
            for a in range(n):
                for b in range(a + 1, n):
                    same = (a < half) == (b < half)
                    assert (C.CLUSTER_STRONG <= F[a, b] <= C.CLUSTER_STRONG + C.CLUSTER_WEAK_MAX) if same else (0 <= F[a, b] <= C.CLUSTER_WEAK_MAX)


def test_star_has_a_strong_hub_and_at_most_one_elsewhere():
    for n in C.SIZES:
        for s in range(10):
            F = I.make_instance("star", n, s).F
            assert (F[0, 1:] >= 1).all() and (F[0, 1:] <= C.UNIFORM_MAX).all() and (F[1:, 1:] <= 1).all()


def test_skewed_flows_come_from_the_value_list_and_have_a_heavy_tail():
    vals = np.concatenate([I.make_instance("skewed", 9, s).F[np.triu_indices(9, 1)] for s in range(30)])
    assert set(np.unique(vals)) <= set(C.SKEW_VALUES) and vals.max() == max(C.SKEW_VALUES) and (vals == 0).mean() > 0.3


def test_the_netlist_family_counts_the_shared_nets_by_hand():
    F = I.flow_from_nets(4, ((0, 1, 2), (1, 2), (0, 3)))
    assert F.tolist() == [[0, 1, 1, 1], [1, 0, 2, 0], [1, 2, 0, 0], [1, 0, 0, 0]]
    inst = I.make_instance("netlist", 8, 35)
    assert len(inst.nets) == 12 and np.array_equal(inst.F, I.flow_from_nets(8, inst.nets))


def test_the_netlist_family_equals_stueck_1_when_the_sister_repo_is_next_door():
    """Die Netzliste ist dieselbe wie in `quadratische-zuordnung-demo` (gleiche Netze für gleichen Seed und gleiche Größe)."""
    sibling = Path(__file__).resolve().parent.parent.parent / "quadratische-zuordnung-demo" / "qmh_instance.py"
    if not sibling.exists():
        pytest.skip("Schwester-Repo liegt nicht nebenan")
    import sys
    sys.path.insert(0, str(sibling.parent))
    try:
        spec = importlib.util.spec_from_file_location("qmh_instance_sibling", sibling)
        other = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(other)
    finally:
        sys.path.remove(str(sibling.parent))
    for n in (6, 8, 9):
        for seed in (0, 35, 100000):
            a, b = I.make_instance("netlist", n, seed), other.make_instance(n, seed)
            assert a.nets == b.nets and np.array_equal(a.F, b.F) and np.array_equal(a.D, b.D) and a.names == b.names


def test_component_names():
    assert I.component_names(8, "netlist") == ("IC1", "C1", "R1", "IC2", "C2", "R2", "IC3", "C3") and I.component_names(3, "uniform") == ("B1", "B2", "B3")


def test_grid_distance_by_hand():
    xy = I.grid_xy(2, 3)
    assert I.manhattan(xy).tolist() == [[0, 1, 2, 1, 2, 3], [1, 0, 1, 2, 1, 2], [2, 1, 0, 3, 2, 1], [1, 2, 3, 0, 1, 2], [2, 1, 2, 1, 0, 1], [3, 2, 1, 2, 1, 0]]


def test_rejections():
    with pytest.raises(ValueError):
        I.make_instance("uniform", 7, 1)
    with pytest.raises(ValueError):
        I.make_instance("gaussian", 6, 1)
    with pytest.raises(ValueError):
        I.grid_xy(0, 3)


def test_the_hand_instances_match_the_readme(hand, hand_grid, hand_gap):
    assert hand.n == 3 and hand.F[0, 1] == 5 and hand.F[1, 2] == 1 and hand.F[0, 2] == 0 and hand.D.tolist() == [[0, 1, 2], [1, 0, 1], [2, 1, 0]]
    assert hand_grid.n == 4 and hand_grid.F[2, 3] == 4 and hand_grid.F.sum() == 2 * (3 + 1 + 2 + 4) and hand_grid.D.max() == 2
    assert hand_gap.n == 4 and hand_gap.F.sum() == 8 and hand_gap.F[0, 1] == 0 and hand_gap.F[2, 3] == 0 and hand_gap.D.tolist()[0] == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        I.hand_instance(C.HAND_FLOWS)


def test_large_sizes_have_their_grids_and_every_family_works():
    for n in C.LARGE_SIZES:
        for fam in C.FAMILIES:
            inst = I.make_instance(fam, n, 7)
            assert inst.rows * inst.cols == n and inst.F.shape == (n, n) and (inst.F == inst.F.T).all()


def test_all_families_equal_stueck_2_when_the_sister_repo_is_next_door():
    """Die sechs Familien sind dieselben wie in `instanzstruktur-demo` (gleiche Flussmatrix für gleichen Seed und gleiche Größe; Raster)."""
    sibling = Path(__file__).resolve().parent.parent.parent / "instanzstruktur-demo" / "izs_instance.py"
    if not sibling.exists():
        pytest.skip("Schwester-Repo liegt nicht nebenan")
    import sys
    sys.path.insert(0, str(sibling.parent))
    try:
        spec = importlib.util.spec_from_file_location("izs_instance_sibling", sibling)
        other = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(other)
    finally:
        sys.path.remove(str(sibling.parent))
    for fam in C.FAMILIES:
        for n in (6, 8, 9):
            for seed in (0, 35, 100000):
                a, b = I.make_instance(fam, n, seed), other.make_instance(fam, n, seed)
                assert np.array_equal(a.F, b.F) and np.array_equal(a.D, b.D), (fam, n, seed)
