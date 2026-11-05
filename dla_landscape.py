"""Die vollständige Landschaft einer Instanz: alle n! Anordnungen, Optima, lokale Optima der Tauschnachbarschaft und der steilste Abstieg von jedem Start.

Lokales Optimum (schwach): kein Tausch zweier Bauteile senkt die Kosten (Gleichstände erlaubt). Streng: jeder Tausch erhöht die Kosten.
Der steilste Abstieg tauscht den Tausch mit dem kleinsten Δ (bei Gleichstand den ersten in der Reihenfolge von `pair_list`), solange Δ < 0.
Gezählt wird roh (Spiegelbilder eines Rasters zählen mit); hier dient die Aufzählung nur dazu, das Optimum zu kennen."""

from dataclasses import dataclass
from itertools import permutations

import numpy as np

import dla_cost as K
from dla_instance import Instance

_PERMS: dict = {}
CHUNK = 250_000


def all_perms(n: int) -> np.ndarray:
    if n < 2 or n > 10:
        raise ValueError("Aufzählung nur für 2 bis 10 Bauteile")
    if n not in _PERMS:
        _PERMS[n] = np.array(list(permutations(range(n))), dtype=np.int8)
    return _PERMS[n]


def costs_of(P: np.ndarray, F, D, chunk: int = 200_000) -> np.ndarray:
    out = np.empty(len(P), dtype=np.int32)
    F32 = F.astype(np.int32)
    D32 = D.astype(np.int32)
    for s in range(0, len(P), chunk):
        Q = P[s:s + chunk].astype(np.intp)
        out[s:s + chunk] = (F32[None] * D32[Q[:, :, None], Q[:, None, :]]).sum(axis=(1, 2)) // 2
    return out


@dataclass(frozen=True, eq=False)
class Landscape:
    n: int
    perms: int                  # n!
    opt: int
    n_opt: int                  # Zahl der Anordnungen mit Optimalkosten
    optima: np.ndarray          # alle optimalen Anordnungen (Zeilen), lexikographisch
    cost_values: np.ndarray     # verschiedene Kostenwerte, aufsteigend
    cost_counts: np.ndarray     # Zahl der Anordnungen je Kostenwert
    mean_cost: float
    local: np.ndarray           # alle schwachen lokalen Optima (Zeilen)
    local_cost: np.ndarray
    local_basin: np.ndarray     # Zahl der Starts, deren Abstieg dort endet
    local_strict: np.ndarray    # bool: streng
    success: float              # Anteil der Starts, deren Abstieg ein globales Optimum erreicht
    mean_steps: float
    max_steps: int

    @property
    def n_local(self) -> int:
        return len(self.local)

    @property
    def n_local_strict(self) -> int:
        return int(self.local_strict.sum())

    @property
    def n_local_nonglobal(self) -> int:
        return int((self.local_cost != self.opt).sum())

    @property
    def mean_gap(self) -> float:
        """(mittlere Kosten − Optimum) / Optimum: wie viel schlechter eine zufällige Anordnung ist. NaN bei Optimum 0."""
        return (self.mean_cost - self.opt) / self.opt if self.opt else float("nan")

    @property
    def end_gap(self) -> float:
        """Mittlere relative Lücke der Endkosten eines Abstiegs zum Optimum, über alle Starts: Σ Starts · (Endkosten − Optimum) / (n! · Optimum). NaN bei Optimum 0."""
        if not self.opt:
            return float("nan")
        return float((self.local_basin * (self.local_cost - self.opt)).sum()) / (self.perms * self.opt)

    @property
    def restarts(self) -> float:
        """Erwartete Zahl der Zufallsstarts bis zum ersten, der ein globales Optimum erreicht (1 / Erfolg)."""
        return 1.0 / self.success


def landscape(inst: Instance) -> Landscape:
    """Rechnet alle n! Anordnungen (n = 8: rund 1 s, n = 9: rund 15 s auf einem Kern)."""
    n, F, D = inst.n, inst.F, inst.D
    P = all_perms(n)
    c = costs_of(P, F, D)
    opt = int(c.min())
    values, counts = np.unique(c, return_counts=True)
    pairs = K.pair_list(n)
    pa = np.array([a for a, _ in pairs])
    pb = np.array([b for _, b in pairs])
    cur = P.astype(np.intp).copy()
    cur_cost = c.astype(np.int64).copy()
    steps = np.zeros(len(P), dtype=np.int32)
    active = np.arange(len(P))
    while len(active):
        nxt = []
        for s in range(0, len(active), CHUNK):
            idx = active[s:s + CHUNK]
            dl = K.all_deltas(cur[idx], F, D)
            j = dl.argmin(axis=1)
            dm = dl[np.arange(len(idx)), j]
            go = dm < 0
            if go.any():
                gi = idx[go]
                Qg = cur[gi]
                r = np.arange(len(gi))
                ta = Qg[r, pa[j[go]]].copy()
                Qg[r, pa[j[go]]] = Qg[r, pb[j[go]]]
                Qg[r, pb[j[go]]] = ta
                cur[gi] = Qg
                cur_cost[gi] += dm[go]
                steps[gi] += 1
                nxt.append(gi)
        active = np.concatenate(nxt) if nxt else np.array([], dtype=np.intp)
    key = (cur * (n ** np.arange(n))[None, :]).sum(axis=1)
    ukey, first, basin = np.unique(key, return_index=True, return_counts=True)
    local = cur[first]
    local_cost = cur_cost[first]
    strict = (K.all_deltas(local, F, D).min(axis=1) > 0) if len(local) else np.zeros(0, dtype=bool)
    return Landscape(n, len(P), opt, int((c == opt).sum()), P[c == opt].astype(np.intp), values, counts, float(c.mean()),
                     local, local_cost.astype(np.int32), basin, strict, float((cur_cost == opt).mean()), float(steps.mean()), int(steps.max()))
