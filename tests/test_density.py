"""Dichte, Poisson-Gleichung und die beiden Dichtestrafen: Handwerte, Eigenschaften und unabhängige Gegenrechnungen (Doppelschleife, ausgeschriebene Laplace-Matrix, Differenzenquotient)."""

import numpy as np
import pytest

import dla_constants as C
import dla_density as Dn
import dla_instance as I


def laplace_neumann(m, h):
    """Die Differenzenmatrix −d²/dx² mit Neumann-Rand, einzeln ausgeschrieben (Randzeilen 1, −1)."""
    L = np.zeros((m, m))
    for i in range(m):
        L[i, i] = 2
        if i > 0:
            L[i, i - 1] = -1
        if i < m - 1:
            L[i, i + 1] = -1
    L[0, 0] = L[-1, -1] = 1
    return L / h ** 2


def loop_density(bins, X):
    """Dichte als Doppelschleife über Bauteile und Bins mit einzeln gerechneter Überdeckung zweier Intervalle (ohne die Matrixschreibweise der Demo)."""
    rho = np.zeros((bins.ny, bins.nx))
    for x, y in X:
        for j in range(bins.ny):
            for i in range(bins.nx):
                ox = max(0.0, min(x + bins.side / 2, bins.edges_x[i + 1]) - max(x - bins.side / 2, bins.edges_x[i]))
                oy = max(0.0, min(y + bins.side / 2, bins.edges_y[j + 1]) - max(y - bins.side / 2, bins.edges_y[j]))
                rho[j, i] += ox * oy / (bins.area * bins.side ** 2)
    return rho


def scattered(n, rows, cols, seed=1):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(0.15, cols - 1.15, n), rng.uniform(0.15, rows - 1.15, n)])


@pytest.fixture
def inst8():
    return I.make_instance("cluster", 8, 46)


def test_hand_example_density_overflow_and_force():
    bins = Dn.Bins(I.hand_instance(((0, 1, 1),), 1, 2), 0.0, 1.0)
    X = np.array([[0.4, 0.0], [0.6, 0.0]])
    rho, _ = bins.density(X)
    assert (bins.nx, bins.ny, bins.area, bins.target) == (4, 2, 0.25, 1.0)
    assert rho == pytest.approx(np.array([[0.2, 1.8, 1.8, 0.2]] * 2))
    assert bins.overflow(rho) == pytest.approx(0.4)
    value, g, _ = bins.energy(X, "electro")
    assert g[0, 0] == pytest.approx(-g[1, 0]) and g[0, 0] > 0 and abs(g[0, 1]) < 1e-12 and abs(g[1, 1]) < 1e-12 and value > 0
    assert value == pytest.approx(0.08)


def test_the_mass_is_n_when_every_footprint_lies_inside(inst8):
    for margin, side in ((0.0, 1.0), (0.5, 2.0), (1.0, 3.0)):
        bins = Dn.Bins(inst8, margin, side)
        X = scattered(8, 2, 4)
        assert bins.density(X)[0].sum() * bins.area == pytest.approx(8.0)


@pytest.mark.parametrize("margin,side", [(0.0, 1.0), (0.5, 3.0), (0.25, 2.0), (1.0, 1.0)])
def test_density_equals_the_double_loop(inst8, margin, side):
    bins = Dn.Bins(inst8, margin, side)
    X = scattered(8, 2, 4, seed=3)
    assert bins.density(X)[0] == pytest.approx(loop_density(bins, X), abs=1e-12)


def test_poisson_solves_the_written_out_neumann_laplacian(inst8):
    for margin in (0.0, 0.5):
        bins = Dn.Bins(inst8, margin, 1.0)
        rng = np.random.default_rng(5)
        rt = rng.normal(size=(bins.ny, bins.nx))
        rt -= rt.mean()
        psi = bins.poisson(rt)
        Ly, Lx = laplace_neumann(bins.ny, bins.h), laplace_neumann(bins.nx, bins.h)
        assert Ly @ psi + psi @ Lx.T == pytest.approx(rt, abs=1e-9) and psi.sum() == pytest.approx(0.0, abs=1e-9)


def test_cosine_matrix_is_orthogonal():
    for m in (2, 4, 8, 12):
        T = Dn.Bins._cosine_matrix(m)
        assert T @ T.T == pytest.approx(np.eye(m), abs=1e-12)


@pytest.mark.parametrize("kind", ["electro", "overflow"])
@pytest.mark.parametrize("margin,side", [(0.0, 1.0), (0.5, 3.0), (0.5, 1.0)])
def test_gradient_equals_finite_differences(inst8, kind, margin, side):
    bins = Dn.Bins(inst8, margin, side)
    X = scattered(8, 2, 4, seed=7) + np.array([0.013, 0.007])
    _, g, _ = bins.energy(X, kind)
    num = np.zeros_like(X)
    eps = 1e-6
    for i in range(8):
        for a in range(2):
            Y, Z = X.copy(), X.copy()
            Y[i, a] += eps
            Z[i, a] -= eps
            num[i, a] = (bins.energy(Y, kind)[0] - bins.energy(Z, kind)[0]) / (2 * eps)
    assert g == pytest.approx(num, abs=2e-6)


def test_a_perfect_tiling_has_no_charge_energy_and_no_overflow(inst8):
    bins = Dn.Bins(inst8, 0.0, 1.0)
    X = inst8.xy.astype(float)
    rho, _ = bins.density(X)
    assert rho == pytest.approx(np.ones_like(rho)) and bins.overflow(rho) == pytest.approx(0.0, abs=1e-12)
    for kind in ("electro", "overflow"):
        value, g, _ = bins.energy(X, kind)
        assert value == pytest.approx(0.0, abs=1e-12) and np.abs(g).max() < 1e-9


def test_the_electrostatic_energy_is_nonnegative_and_the_clump_costs_more_than_a_spread(inst8):
    bins = Dn.Bins(inst8, 0.5, 1.0)
    clump = np.tile(inst8.xy.astype(float).mean(axis=0), (8, 1)) + (np.random.default_rng(2).uniform(size=(8, 2)) - 0.5) * 0.05
    spread = inst8.xy.astype(float)
    assert bins.energy(clump, "electro")[0] > bins.energy(spread, "electro")[0] >= -1e-12
    assert bins.overflow(bins.density(clump)[0]) > bins.overflow(bins.density(spread)[0])


def test_the_force_pushes_a_symmetric_pair_apart_and_the_mirror_image_mirrors_the_force(inst8):
    bins = Dn.Bins(inst8, 0.5, 2.0)
    X = np.array([[1.4, 0.5], [1.6, 0.5]] + [[0.0, 0.0], [3.0, 0.0], [0.0, 1.0], [3.0, 1.0], [2.0, 0.0], [1.0, 1.0]])
    _, g, _ = bins.energy(X, "electro")
    assert g[0, 0] > 0 > g[1, 0]                                   # Gradient positiv = Verschieben nach rechts verteuert: die Kraft −g schiebt nach links
    mirror = X.copy()
    mirror[:, 0] = inst8.cols - 1 - X[:, 0]                        # Spiegelung an der Mitte des Blocks (Plätze 0 bis 3, Mitte 1,5)
    _, gm, _ = bins.energy(mirror, "electro")
    assert gm[:, 0] == pytest.approx(-g[:, 0], abs=1e-9) and gm[:, 1] == pytest.approx(g[:, 1], abs=1e-9)


def test_overflow_zero_means_no_bin_is_over_target_and_none_returns_zeros(inst8):
    bins = Dn.Bins(inst8, 0.5, 4.0)
    X = scattered(8, 2, 4)
    value, g, rho = bins.energy(X, "none")
    assert value == 0.0 and not g.any() and rho.shape == (bins.ny, bins.nx)
    assert Dn.unit_overflow(inst8, X, 0.5) == pytest.approx(Dn.Bins(inst8, 0.5, 1.0).overflow(Dn.Bins(inst8, 0.5, 1.0).density(X)[0]))


def test_unknown_model_and_bin_counts(inst8):
    with pytest.raises(ValueError):
        Dn.Bins(inst8, 0.0, 1.0).energy(scattered(8, 2, 4), "gauss")
    for margin, (nx, ny) in ((0.0, (8, 4)), (0.25, (9, 5)), (0.5, (10, 6)), (1.0, (12, 8))):
        b = Dn.Bins(inst8, margin, 1.0)
        assert (b.nx, b.ny) == (nx, ny) and b.region == (-0.5 - margin, 3.5 + margin, -0.5 - margin, 1.5 + margin)


def test_potential_matches_the_poisson_solution(inst8):
    bins = Dn.Bins(inst8, 0.5, 3.0)
    X = scattered(8, 2, 4, seed=9)
    rho, _ = bins.density(X)
    assert bins.potential(X) == pytest.approx(bins.poisson(rho - rho.mean()))
