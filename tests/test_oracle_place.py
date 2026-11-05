"""Orakel (ORACLE-PLAYBOOK §5): `scipy` für das lineare System, die Zuordnung und das Minimum der quadratischen Zielfunktion; die Aufzählung aller Anordnungen für das Optimum und für die Rundung.
Ohne scipy (nur requirements-dev.txt; die App braucht scipy nicht) werden die Tests übersprungen."""

import numpy as np
import pytest

import dla_constants as C
import dla_instance as I
import dla_place as Pl

sp = pytest.importorskip("scipy.optimize")
linalg = pytest.importorskip("scipy.linalg")


@pytest.mark.parametrize("family", C.FAMILIES)
def test_the_system_solution_equals_scipy_least_squares_and_the_numerical_minimum(family):
    inst = I.make_instance(family, 6, 100004)
    pads = Pl.make_pads(inst, 4, 100004)
    X = Pl.positions(inst, pads)
    A = Pl.system_matrix(inst, pads) + C.CENTER_ANCHOR * np.eye(6)
    center = inst.xy.astype(float).mean(axis=0)
    for ax in (0, 1):
        b = (pads.w * pads.pos[None, :, ax]).sum(axis=1) + C.CENTER_ANCHOR * center[ax]
        assert X[:, ax] == pytest.approx(np.linalg.lstsq(A, b, rcond=None)[0], abs=1e-7)
        assert X[:, ax] == pytest.approx(linalg.solve(A, b, assume_a="pos"), abs=1e-7)
    # das Minimum der quadratischen Zielfunktion, unabhängig mit einem allgemeinen Löser (BFGS) gefunden
    f = lambda x: Pl.quadratic_value(inst, x.reshape(6, 2), pads) + C.CENTER_ANCHOR * ((x.reshape(6, 2) - center) ** 2).sum()
    res = sp.minimize(f, np.tile(center, (6, 1)).ravel(), method="BFGS", options={"gtol": 1e-10, "maxiter": 500})
    assert res.x.reshape(6, 2) == pytest.approx(X, abs=1e-4)


def test_the_rounding_assignment_equals_scipy_and_enumeration():
    rng = np.random.default_rng(4)
    for fam in C.FAMILIES:
        inst = I.make_instance(fam, 6, 100005)
        X = rng.uniform(-0.5, 3, size=(6, 2))
        perm = Pl.round_to_slots(inst, X)
        d2 = ((X[:, None, :] - inst.xy[None, :, :].astype(float)) ** 2).sum(axis=2)
        r, cc = sp.linear_sum_assignment(d2)
        assert d2[np.arange(6), perm].sum() == pytest.approx(d2[r, cc].sum(), abs=1e-6)
        from itertools import permutations

        assert d2[np.arange(6), perm].sum() == pytest.approx(min(d2[np.arange(6), list(q)].sum() for q in permutations(range(6))), abs=1e-6)


def test_the_best_found_arrangement_never_beats_the_enumerated_optimum_and_the_rounding_never_either():
    for fam in C.FAMILIES:
        for seed in (100006, 100007):
            inst = I.make_instance(fam, 6, seed)
            for p in (0, 4, 8):
                pads = Pl.make_pads(inst, p, seed)
                c = Pl.linear_costs(inst, pads)
                ref = Pl.reference(inst, c, seed)[0]
                assert Pl.best_found(inst, c, seed, 30)[0] >= ref - 1e-9
                for r in (0, 5):
                    perm = Pl.round_to_slots(inst, Pl.positions(inst, pads, r))
                    assert Pl.cost_of(perm, inst.F, inst.D, c) >= ref - 1e-9
