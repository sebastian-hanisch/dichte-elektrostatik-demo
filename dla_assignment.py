"""Lineare Zuordnung (Ungarische Methode, Variante mit Potenzialen, O(n³)): jede Zeile genau einer Spalte, kleinste Summe.

Dieselbe Aufgabe löst `scipy.optimize.linear_sum_assignment`; hier steht eine eigene Fassung (scipy dient nur als Orakel in den Tests; die App braucht scipy nicht).
Ganzzahlige Matrizen bleiben ganzzahlig (Python-Ganzzahlen), Gleitkommazahlen werden mit einer kleinen Toleranz verglichen."""

import numpy as np


def hungarian(cost) -> tuple:
    """(Spalte je Zeile, Gesamtkosten) für eine quadratische Kostenmatrix. Bei mehreren gleich guten Zuordnungen ist die gelieferte eine davon (nicht festgelegt)."""
    a = np.asarray(cost)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("quadratische Matrix erwartet")
    n = a.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.intp), 0
    if not np.isfinite(a.astype(float)).all():
        raise ValueError("nur endliche Kosten")
    integral = a.dtype.kind in "iu"
    c = [[(int(x) if integral else float(x)) for x in row] for row in a]
    inf = float("inf") if not integral else 1 << 62
    u = [0] * (n + 1) if integral else [0.0] * (n + 1)
    v = [0] * (n + 1) if integral else [0.0] * (n + 1)
    p = [0] * (n + 1)          # p[j]: Zeile (1-basiert), die Spalte j hält
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [inf] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], inf, 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = c[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    cols = np.zeros(n, dtype=np.intp)
    for j in range(1, n + 1):
        cols[p[j] - 1] = j - 1
    total = sum(c[i][int(cols[i])] for i in range(n))
    return cols, total


def hungarian_dual(cost) -> tuple:
    """Lineare Zuordnung mit Dualvariablen (Potenziale): (Spalte je Zeile, u, v, Wert) mit cost[i,j] − u[i] − v[j] ≥ 0 für alle i, j und Gleichheit auf der Zuordnung.

    Es gilt Wert = Σ u + Σ v (starke Dualität). Rechnet mit Gleitkommazahlen (bei den halbzahligen Kosten des Dual Ascent exakt, solange die Zahlen klein bleiben)."""
    a = np.asarray(cost, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("quadratische Matrix erwartet")
    n = a.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.intp), np.zeros(0), np.zeros(0), 0.0
    if not np.isfinite(a).all():
        raise ValueError("nur endliche Kosten")
    inf = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [inf] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], inf, 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = a[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    cols = np.zeros(n, dtype=np.intp)
    for j in range(1, n + 1):
        cols[p[j] - 1] = j - 1
    return cols, np.array(u[1:]), np.array(v[1:]), float(sum(a[i, cols[i]] for i in range(n)))
