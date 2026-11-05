"""Instanzen: sechs Familien der Flussmatrix auf einem Raster (Familien aus Stück 2). Alle Zufallsentscheidungen über SplitMix64.

F ist symmetrisch, ganzzahlig, Diagonale 0 (Fluss je Paar von Bauteilen); D ist der Manhattan-Abstand der Plätze eines Rasters.
Die Familien sind dieselben wie in Stück 2 (`instanzstruktur-demo`) und die Netzliste dieselbe wie in Stück 1: gleiche Seeds, gleiche Instanzen."""

from dataclasses import dataclass
from itertools import combinations

import numpy as np

import dla_constants as C
from dla_rng import SplitMix64


@dataclass(frozen=True, eq=False)
class Instance:
    n: int
    rows: int
    cols: int
    F: np.ndarray            # (n, n) ganzzahlig, symmetrisch, Diagonale 0
    D: np.ndarray            # (n, n) Abstand der Plätze
    xy: np.ndarray           # (n, 2) Lage der Plätze (Raster: Spalte, Zeile; zufällig: Koordinaten im Feld)
    names: tuple             # Bauteilnamen
    nets: tuple              # je Netz ein Tupel von Bauteilnummern (nur Familie Netzliste, sonst leer)
    seed: int
    family: str


def manhattan(xy: np.ndarray) -> np.ndarray:
    return np.abs(xy[:, None, :] - xy[None, :, :]).sum(axis=2).astype(np.int16)


def grid_xy(rows: int, cols: int) -> np.ndarray:
    """Plätze eines Rasters; Platz = Zeile · Spalten + Spalte, Lage (Spalte, Zeile)."""
    if rows < 1 or cols < 1:
        raise ValueError("Raster braucht mindestens eine Zeile und eine Spalte")
    return np.array([(c, r) for r in range(rows) for c in range(cols)])


def component_names(n: int, family: str) -> tuple:
    if family == "netlist":
        kinds = ("IC", "C", "R")
        return tuple(f"{kinds[i % 3]}{i // 3 + 1}" for i in range(n))
    return tuple(f"B{i + 1}" for i in range(n))


def flow_from_nets(n: int, nets) -> np.ndarray:
    F = np.zeros((n, n), dtype=np.int16)
    for net in nets:
        for a, b in combinations(net, 2):
            F[a, b] += 1
            F[b, a] += 1
    return F


def _symmetric(n: int, value) -> np.ndarray:
    """Symmetrische Matrix, Eintrag (a < b) = value(a, b), Reihenfolge der Aufrufe: a aufsteigend, dann b aufsteigend."""
    F = np.zeros((n, n), dtype=np.int16)
    for a in range(n):
        for b in range(a + 1, n):
            F[a, b] = F[b, a] = value(a, b)
    return F


def _netlist(n: int, rng: SplitMix64):
    groups = (n + C.GROUP - 1) // C.GROUP
    nets = []
    for _ in range(max(2, int(round(C.N_NETS_PER_COMPONENT * n)))):
        k = min(rng.weighted(C.NET_SIZES, C.NET_WEIGHTS), n)
        pool = list(range(n))
        if rng.below(100) < C.LOCAL_NET_PERCENT:
            g = rng.below(groups)
            local = [i for i in range(n) if i // C.GROUP == g]
            if len(local) >= k:
                pool = local
        rng.shuffle(pool)
        nets.append(tuple(sorted(pool[:k])))
    return flow_from_nets(n, nets), tuple(nets)


def make_flow(family: str, n: int, seed: int):
    """(F, nets). Der Zufallsstrom hängt von Seed, n und Familie ab; für die Netzliste ist er der von Stück 1."""
    if family not in C.FAMILIES:
        raise ValueError(f"Familie muss eine von {C.FAMILIES} sein")
    if family == "netlist":
        return _netlist(n, SplitMix64(seed * 1000003 + n))
    rng = SplitMix64(seed * 1000003 + n + 7919 * (C.FAMILIES.index(family) + 1))
    if family == "uniform":
        return _symmetric(n, lambda a, b: rng.randint(0, C.UNIFORM_MAX)), ()
    if family == "sparse":
        return _symmetric(n, lambda a, b: rng.randint(1, C.UNIFORM_MAX) if rng.below(100) < C.SPARSE_PERCENT else 0), ()
    if family == "cluster":
        half = n // 2

        def cluster(a, b):
            same = (a < half) == (b < half)
            return rng.below(C.CLUSTER_WEAK_MAX + 1) + (C.CLUSTER_STRONG if same else 0)
        return _symmetric(n, cluster), ()
    if family == "star":
        hub = [rng.randint(1, C.UNIFORM_MAX) for _ in range(n - 1)]
        return _symmetric(n, lambda a, b: hub[b - 1] if a == 0 else rng.below(2)), ()
    return _symmetric(n, lambda a, b: rng.weighted(C.SKEW_VALUES, C.SKEW_WEIGHTS)), ()


def make_instance(family: str, n: int, seed: int) -> Instance:
    if n not in C.GRID:
        raise ValueError(f"n muss eine der Größen {C.SIZES} sein")
    F, nets = make_flow(family, n, seed)
    rows, cols = C.GRID[n]
    xy = grid_xy(rows, cols)
    return Instance(n, rows, cols, F, manhattan(xy), xy, component_names(n, family), nets, seed, family)


def hand_instance(flows=None, rows: int | None = None, cols: int | None = None) -> Instance:
    """Ein Handbeispiel auf einem Raster. Ohne Angaben: drei Bauteile auf einer Zeile, Fluss A–B: 5, B–C: 1; der Rechenweg steht in der README."""
    if flows is None:
        flows, rows, cols = C.HAND_FLOWS, C.HAND_GRID[0], C.HAND_GRID[1]
    if rows is None or cols is None:
        raise ValueError("rows und cols zusammen mit flows angeben")
    n = rows * cols
    F = np.zeros((n, n), dtype=np.int16)
    for a, b, w in flows:
        F[a, b] = F[b, a] = w
    xy = grid_xy(rows, cols)
    return Instance(n, rows, cols, F, manhattan(xy), xy, tuple("ABCDEFGHIJKLMNOPQRST"[:n]), (), 0, "hand")
