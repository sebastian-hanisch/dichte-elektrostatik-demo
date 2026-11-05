"""Quadratische Platzierung: die Orte der Bauteile reell relaxieren, mit festen Pads (Ankern), Umgewichtung und Rundung auf die Plätze.

Instanz: die Familien der Stücke 1 bis 8 (Fluss F, Raster der Plätze) plus `p` feste Pads auf einem Ring um das Raster; jedes Pad hängt mit dem Gewicht w an zwei zufälligen Bauteilen.
Kosten einer Anordnung (Bauteil i auf Platz p_i): cost(p) = Σ_{i<k} F[i,k] · |p_i − p_k|₁ + Σ_{i,Pad} w[i,Pad] · |p_i − Pad|₁ (das QAP mit linearem Term, wie in Stück 8).

**Relaxation der Orte:** minimiere je Achse Σ_{i<k} F[i,k] (x_i − x_k)² + Σ w (x_i − a)². Nullsetzen des Gradienten gibt (L + diag(W)) x = b mit L = diag(Σ_k F) − F, W_i = Σ_Pads w[i,Pad], b_i = Σ w a.
Ohne Pads ist L singulär (alle x_i gleich ist ein Minimum): alles fällt auf einen Punkt. Bauteile ohne Weg zu einem Pad fallen ebenso; ein schwacher Anker in der Mitte hält das System lösbar.
**Umgewichtung („Bound2Bound-artig“, eigene Fassung für 2-Pin-Verbindungen):** je Achse F / max(|Δ|, ε) statt F, mehrfach neu gerechnet: das minimiert die lineare Länge statt der quadratischen.
**Rundung:** lineare Zuordnung der stetigen Orte auf die Plätze, kleinste Summe der Abstandsquadrate; die Quadrate werden vorher auf acht Stellen gerundet (`tie_quantized`), damit Gleichstände auf jedem Rechner gleich entschieden werden."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import dla_assignment as A
import dla_constants as C
import dla_landscape as L
from dla_rng import SplitMix64


@dataclass(frozen=True, eq=False)
class Pads:
    pos: np.ndarray           # (p, 2) Lage der Pads
    w: np.ndarray             # (n, p) Gewicht der Verbindung Bauteil–Pad (0 = keine)
    base: float               # Grundgewicht (Median der positiven Flüsse, mindestens 1)
    factor: float             # Faktor auf das Grundgewicht

    @property
    def count(self) -> int:
        return len(self.pos)


def make_pads(inst, p: int, seed: int, factor: float = 1.0) -> Pads:
    """`p` Pads gleichmäßig auf dem Ring um das Raster (Abstand RING_MARGIN); je Pad zwei zufällige Bauteile mit dem Gewicht factor · Grundgewicht."""
    n = inst.n
    F = np.asarray(inst.F)
    positive = F[F > 0]
    base = max(1.0, float(round(float(np.median(positive))))) if len(positive) else 1.0
    if p == 0:
        return Pads(np.zeros((0, 2)), np.zeros((n, 0)), base, factor)
    xs, ys = inst.xy[:, 0].astype(float), inst.xy[:, 1].astype(float)
    x0, x1, y0, y1 = xs.min() - C.RING_MARGIN, xs.max() + C.RING_MARGIN, ys.min() - C.RING_MARGIN, ys.max() + C.RING_MARGIN
    wd, ht = x1 - x0, y1 - y0
    per = 2 * (wd + ht)
    pos = []
    for k in range(p):
        t = (k + 0.5) / p * per
        if t < wd:
            pos.append((x0 + t, y0))
        elif t < wd + ht:
            pos.append((x1, y0 + t - wd))
        elif t < 2 * wd + ht:
            pos.append((x1 - (t - wd - ht), y1))
        else:
            pos.append((x0, y1 - (t - 2 * wd - ht)))
    rng = SplitMix64(seed * 7919 + 13)
    w = np.zeros((n, p))
    for j in range(p):
        a = rng.below(n)
        b = rng.below(n - 1)
        b = b if b < a else b + 1
        w[a, j] = w[b, j] = base * factor
    return Pads(np.array(pos), w, base, factor)


def linear_costs(inst, pads: Pads) -> np.ndarray:
    """c[i,j]: Kosten des Bauteils i auf Platz j gegen die Pads (Manhattan)."""
    if not pads.count:
        return np.zeros((inst.n, inst.n))
    d = np.abs(inst.xy[:, None, :].astype(float) - pads.pos[None, :, :]).sum(axis=2)       # (Plätze, Pads)
    return pads.w @ d.T


def cost_of(perm, F, D, c) -> float:
    perm = np.asarray(perm, dtype=np.intp)
    return float((np.asarray(F, dtype=float) * np.asarray(D, dtype=float)[np.ix_(perm, perm)]).sum() / 2 + c[np.arange(len(perm)), perm].sum())


def all_costs(inst, c) -> tuple:
    """(alle Anordnungen, ihre Kosten) für n ≤ 9."""
    P = L.all_perms(inst.n)
    base = L.costs_of(P, inst.F, inst.D).astype(float)
    return P, base + c[np.arange(inst.n)[None, :], P.astype(np.intp)].sum(axis=1)


# ----------------------------------------------------------------------------- das lineare System

def solve_axis(F, w, a, Fw=None, ww=None, center: float = 0.0) -> np.ndarray:
    """Eine Achse: (diag(Σ Fw + Σ ww) − Fw + δ·I) x = Σ ww·a + δ·center. Für `Fw`, `ww` gleich F, w die quadratische Platzierung."""
    F = np.asarray(F, dtype=float)
    n = len(F)
    Fw = F if Fw is None else Fw
    ww = w if ww is None else ww
    A = np.diag(Fw.sum(axis=1) + ww.sum(axis=1)) - Fw + C.CENTER_ANCHOR * np.eye(n)
    b = (ww * np.asarray(a, dtype=float)[None, :]).sum(axis=1) + C.CENTER_ANCHOR * center
    return np.linalg.solve(A, b)


def system_matrix(inst, pads: Pads) -> np.ndarray:
    """Die Matrix des Systems (für beide Achsen gleich): Laplace-Matrix des Flussgraphen plus Diagonale der Pad-Gewichte (ohne den schwachen Anker)."""
    F = np.asarray(inst.F, dtype=float)
    return np.diag(F.sum(axis=1) + pads.w.sum(axis=1)) - F


def positions(inst, pads: Pads, reweights: int = 0, trace: bool = False):
    """Stetige Orte (n, 2): reine quadratische Platzierung (reweights = 0) oder nach so vielen Umgewichtungen F / max(|Δ|, ε) je Achse. Mit trace=True die Liste aller Zwischenstände (Länge reweights + 1)."""
    F = np.asarray(inst.F, dtype=float)
    n = inst.n
    center = inst.xy.astype(float).mean(axis=0)
    X = np.tile(center, (n, 1))
    if pads.count:
        for ax in (0, 1):
            X[:, ax] = solve_axis(F, pads.w, pads.pos[:, ax], center=center[ax])
    steps = [X.copy()]
    if pads.count:
        for _ in range(reweights):
            new = X.copy()
            for ax in (0, 1):
                d = np.abs(X[:, None, ax] - X[None, :, ax])
                dp = np.abs(X[:, None, ax] - pads.pos[None, :, ax])
                new[:, ax] = solve_axis(F, pads.w, pads.pos[:, ax], F / np.maximum(d, C.EPS), pads.w / np.maximum(dp, C.EPS), center[ax])
            X = new
            steps.append(X.copy())
    else:
        steps += [X.copy() for _ in range(reweights)]
    return steps if trace else X


def continuous_length(inst, X, pads: Pads) -> float:
    """Lineare (Manhattan-)Länge der stetigen Lösung: Σ_{i<k} F |Δ| + Σ w |Δ|, ohne Rundung."""
    F = np.asarray(inst.F, dtype=float)
    d = np.abs(X[:, None, :] - X[None, :, :]).sum(axis=2)
    out = float((F * d).sum() / 2)
    if pads.count:
        out += float((pads.w * np.abs(X[:, None, :] - pads.pos[None, :, :]).sum(axis=2)).sum())
    return out


def quadratic_value(inst, X, pads: Pads) -> float:
    """Wert der quadratischen Zielfunktion: Σ_{i<k} F |Δ|² + Σ w |Δ|² (beide Achsen)."""
    F = np.asarray(inst.F, dtype=float)
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
    out = float((F * d2).sum() / 2)
    if pads.count:
        out += float((pads.w * ((X[:, None, :] - pads.pos[None, :, :]) ** 2).sum(axis=2)).sum())
    return out


def spread(inst, X) -> float:
    """Streuung der stetigen Orte im Verhältnis zur Streuung der Plätze (1 = so verteilt wie die Plätze, 0 = alles in einem Punkt)."""
    s0 = inst.xy.astype(float).std(axis=0).sum()
    return float(np.asarray(X).std(axis=0).sum() / s0)


def nearest_slots(inst, X) -> np.ndarray:
    d2 = ((np.asarray(X)[:, None, :] - inst.xy[None, :, :].astype(float)) ** 2).sum(axis=2)
    return d2.argmin(axis=1)


def overlap(inst, X) -> int:
    """Zahl der Bauteile, die mit einem anderen denselben nächsten Platz haben (n − Zahl verschiedener nächster Plätze)."""
    return int(inst.n - len(set(nearest_slots(inst, X).tolist())))


# ----------------------------------------------------------------------------- Rundung

def tie_quantized(M, decimals: int = C.TIE_DECIMALS) -> np.ndarray:
    """M, auf `decimals` Stellen relativ zum größten Betrag gerundet: Gleichstände, die nur durch Rechenrauschen getrennt sind, werden exakte Gleichstände, die die Ungarische Methode überall gleich auflöst."""
    M = np.asarray(M, dtype=float)
    scale = float(np.abs(M).max())
    return M if scale == 0.0 else np.round(M / scale, decimals)


def round_to_slots(inst, X) -> list:
    """Zuordnung der stetigen Orte auf die Plätze: kleinste Summe der Abstandsquadrate (lineare Zuordnung, Gleichstände wie in `tie_quantized`). Rückgabe: Platz je Bauteil."""
    d2 = ((np.asarray(X, dtype=float)[:, None, :] - inst.xy[None, :, :].astype(float)) ** 2).sum(axis=2)
    cols, _ = A.hungarian(tie_quantized(d2))
    return [int(j) for j in cols]


# ----------------------------------------------------------------------------- Vergleichswert ab n = 12

def swap_costs(perm, F, D, c) -> tuple:
    """Kosten aller Vertauschungen zweier Bauteile (Paare (a, b), a < b) für die Anordnung `perm`."""
    perm = np.asarray(perm, dtype=np.intp)
    n = len(perm)
    ia, ib = np.triu_indices(n, 1)
    Q = np.tile(perm, (len(ia), 1))
    rows = np.arange(len(ia))
    Q[rows, ia], Q[rows, ib] = perm[ib], perm[ia]
    cost = (F[None] * D[Q[:, :, None], Q[:, None, :]]).sum(axis=(1, 2)) / 2 + c[np.arange(n)[None, :], Q].sum(axis=1)
    return ia, ib, cost


def descent(perm, F, D, c) -> tuple:
    """Steilster Abstieg über Vertauschungen: (Anordnung, Kosten)."""
    F, D = np.asarray(F, dtype=float), np.asarray(D, dtype=float)
    perm = np.array(perm, dtype=np.intp)
    cur = cost_of(perm, F, D, c)
    while True:
        ia, ib, cost = swap_costs(perm, F, D, c)
        k = int(cost.argmin())
        if cost[k] < cur - 1e-9:
            perm[ia[k]], perm[ib[k]] = perm[ib[k]], perm[ia[k]]
            cur = float(cost[k])
        else:
            return [int(x) for x in perm], cur


def best_found(inst, c, seed: int, starts: int = C.STARTS) -> tuple:
    """(Kosten, Anordnung) der besten von `starts` steilsten Abstiegen aus Zufallsstarts (SplitMix64): eine **obere** Schranke des Optimums."""
    rng = SplitMix64(seed * 104729 + 7)
    F, D = np.asarray(inst.F, dtype=float), np.asarray(inst.D, dtype=float)
    best = None
    for _ in range(starts):
        start = list(range(inst.n))
        rng.shuffle(start)
        p, v = descent(start, F, D, c)
        if best is None or v < best[0] - 1e-9:
            best = (v, p)
    return best


def reference(inst, c, seed: int) -> tuple:
    """(Vergleichswert, Anordnung, aufgezählt?, mittlere Kosten aller Anordnungen oder None): das aufgezählte Optimum bis n = 9, sonst die beste von 100 Abstiegen."""
    if inst.n in C.EXACT_SIZES:
        P, costs = all_costs(inst, c)
        k = int(costs.argmin())
        return float(costs[k]), [int(x) for x in P[k]], True, float(costs.mean())
    v, p = best_found(inst, c, seed)
    return v, p, False, None
