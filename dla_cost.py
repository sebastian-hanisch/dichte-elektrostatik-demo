"""Kosten und Tausch-Delta (Koopmans–Beckmann-Form, wie in Stück 1).

Konvention: p[i] = Platz des Bauteils i. Kosten = Σ_{i<k} F[i,k] · D[p[i], p[k]] (jedes Paar einmal; F und D symmetrisch, Diagonale 0).
Tausch der Bauteile a und b:  Δ = Σ_{k≠a,b} (F[a,k] − F[b,k]) · (D[p[b],p[k]] − D[p[a],p[k]])."""

import numpy as np


def cost(p, F, D) -> int:
    p = np.asarray(p, dtype=np.intp)
    if sorted(p.tolist()) != list(range(len(F))):
        raise ValueError("p muss eine Permutation von 0..n-1 sein")
    return int((F.astype(np.int64) * D[np.ix_(p, p)]).sum() // 2)


def swap_delta(p, a: int, b: int, F, D) -> int:
    """Änderung der Kosten durch den Tausch von a und b nach der Formel (Einzelfall, von Hand prüfbar)."""
    if a == b:
        raise ValueError("zwei verschiedene Bauteile")
    d = 0
    for k in range(len(p)):
        if k in (a, b):
            continue
        d += (int(F[a, k]) - int(F[b, k])) * (int(D[p[b], p[k]]) - int(D[p[a], p[k]]))
    return d


def swapped(p, a: int, b: int) -> list:
    q = list(p)
    q[a], q[b] = q[b], q[a]
    return q


def pair_list(n: int) -> list:
    return [(a, b) for a in range(n) for b in range(a + 1, n)]


def all_deltas(Q, F, D) -> np.ndarray:
    """Δ aller Tausche für alle Anordnungen in Q (Zeilen): Matrix Zeilen × n(n−1)/2 in der Reihenfolge von pair_list."""
    n = Q.shape[1]
    F32 = F.astype(np.int32)
    D32 = D.astype(np.int32)
    pairs = pair_list(n)
    out = np.empty((len(Q), len(pairs)), dtype=np.int32)
    ks = np.arange(n)
    for j, (a, b) in enumerate(pairs):
        pa, pb = Q[:, a], Q[:, b]
        w = (F32[a] - F32[b])[None, :] * (D32[pb[:, None], Q] - D32[pa[:, None], Q])
        mask = (ks != a) & (ks != b)
        out[:, j] = (w * mask[None, :]).sum(axis=1)
    return out
