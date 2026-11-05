"""Look-ahead-Legalisierung in der Art von SimPL: quadratische Platzierung mit Pseudo-Ankern, Legalisierung auf die Plätze, untere und obere Werte je Runde.

Kern von Stück 9 (`dla_place.py`: Pads, lineares System, Umgewichtung, Kosten mit linearem Term, Zuordnung). Neu: in Runde t ≥ 1 hängt jedes Bauteil mit dem Gewicht λ_t = c · Grundgewicht · t an dem Platz,
auf den die Legalisierung der Vorrunde es gesetzt hat; das System wird (L + diag(W) + λ_t·I) x = b + λ_t · Platz je Achse neu gelöst. Runde 0 ist die Platzierung ohne Pseudo-Anker (Stück 9).
**Untere Länge** = stetige lineare Länge der angekerten Lösung (nur echte Verbindungen und Pads, ohne die Pseudo-Anker); **obere Kosten** = Kosten der legalen Anordnung.
Beides sind **Näherungen**: die echte untere Schranke wäre das Minimum der linearen Länge über alle reellen Orte (in den Tests mit `scipy.optimize.linprog` geprüft). Die Annäherung der beiden Werte kommt vom wachsenden Ankergewicht.
**Legalisierer:** Zuordnung (kleinste Summe der Abstandsquadrate) oder rekursive Zweiteilung (eigene Fassung der Look-ahead-Idee: den Block der Plätze entlang der längeren Seite teilen, die Bauteile nach der Koordinate sortieren,
den ersten Teil dem ersten Block geben). Koordinaten werden vor dem Sortieren auf acht Stellen gerundet, damit Gleichstände auf jedem Rechner gleich entschieden werden."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import dla_constants as C
import dla_place as Pl


def solve_axis_anchored(F, w, a, Fw, ww, lam: float, anchor, center: float) -> np.ndarray:
    """Eine Achse: (diag(Σ Fw + Σ ww + λ) − Fw + δ·I) x = Σ ww·a + λ·anchor + δ·center."""
    F = np.asarray(F, dtype=float)
    n = len(F)
    A = np.diag(Fw.sum(axis=1) + ww.sum(axis=1) + lam) - Fw + C.CENTER_ANCHOR * np.eye(n)
    b = (ww * np.asarray(a, dtype=float)[None, :]).sum(axis=1) + lam * np.asarray(anchor, dtype=float) + C.CENTER_ANCHOR * center
    return np.linalg.solve(A, b)


def positions_anchored(inst, pads, anchors, lam: float, reweights: int = 0) -> np.ndarray:
    """Quadratische Platzierung mit Pseudo-Ankern `anchors` (n, 2) vom Gewicht `lam` (für jedes Bauteil gleich); `reweights` Umgewichtungen der echten Verbindungen und Pads. lam = 0 gibt Stück 9."""
    F = np.asarray(inst.F, dtype=float)
    n = inst.n
    center = inst.xy.astype(float).mean(axis=0)
    pp = pads.pos if pads.count else np.zeros((0, 2))
    ww0 = pads.w if pads.count else np.zeros((n, 0))
    X = np.zeros((n, 2))
    for ax in (0, 1):
        X[:, ax] = solve_axis_anchored(F, ww0, pp[:, ax], F, ww0, lam, np.asarray(anchors)[:, ax], center[ax])
    for _ in range(reweights):
        new = X.copy()
        for ax in (0, 1):
            d = np.abs(X[:, None, ax] - X[None, :, ax])
            dp = np.abs(X[:, None, ax] - pp[None, :, ax])
            new[:, ax] = solve_axis_anchored(F, ww0, pp[:, ax], F / np.maximum(d, C.EPS), ww0 / np.maximum(dp, C.EPS), lam, np.asarray(anchors)[:, ax], center[ax])
        X = new
    return X


# ----------------------------------------------------------------------------- Legalisierer

def bisect_legalize(inst, X) -> list:
    """Rekursive Zweiteilung: Platz je Bauteil. Der Block der Plätze wird entlang der längeren Seite (bei Gleichstand: der Spalten) geteilt, die Bauteile werden nach der Koordinate sortiert (Gleichstände: kleinere Nummer zuerst),
    die ersten so viele, wie der erste Block Plätze hat, gehen in ihn; bis ein Platz bleibt."""
    X = np.asarray(X, dtype=float)
    Q = np.column_stack([Pl.tie_quantized(X[:, 0]), Pl.tie_quantized(X[:, 1])])
    idx = {(int(x), int(y)): i for i, (x, y) in enumerate(inst.xy)}
    perm = [None] * inst.n

    def rec(cells, r0, r1, c0, c1):
        if len(cells) == 1:
            perm[cells[0]] = idx[(c0, r0)]
            return
        if (c1 - c0) >= (r1 - r0):
            mid = (c0 + c1) // 2
            k = (r1 - r0) * (mid - c0)
            cells = sorted(cells, key=lambda i: (Q[i, 0], i))
            rec(cells[:k], r0, r1, c0, mid)
            rec(cells[k:], r0, r1, mid, c1)
        else:
            mid = (r0 + r1) // 2
            k = (mid - r0) * (c1 - c0)
            cells = sorted(cells, key=lambda i: (Q[i, 1], i))
            rec(cells[:k], r0, mid, c0, c1)
            rec(cells[k:], mid, r1, c0, c1)

    rec(list(range(inst.n)), 0, inst.rows, 0, inst.cols)
    return perm


def legalize(inst, X, kind: str) -> list:
    if kind == "assign":
        return Pl.round_to_slots(inst, X)
    if kind == "bisect":
        return bisect_legalize(inst, X)
    raise ValueError(f"unbekannter Legalisierer {kind!r}")


# ----------------------------------------------------------------------------- die Schleife

@dataclass(frozen=True, eq=False)
class Round:
    t: int
    lam: float
    X: np.ndarray             # stetige Orte dieser Runde
    perm: list                # legale Anordnung dieser Runde (Platz je Bauteil)
    lower: float              # stetige lineare Länge (echte Verbindungen und Pads)
    upper: float              # Kosten der legalen Anordnung
    disp: float               # Verschiebung der Rundung: Σ |stetiger Ort − Platz|₁


@dataclass(frozen=True, eq=False)
class Loop:
    legalizer: str
    reweights: int
    anchor: float
    rounds: tuple             # Round 0 … Round T

    @property
    def uppers(self) -> list:
        return [r.upper for r in self.rounds]

    @property
    def lowers(self) -> list:
        return [r.lower for r in self.rounds]

    @property
    def changes(self) -> int:
        """Zahl der Runden, in denen sich die Anordnung gegenüber der Vorrunde ändert."""
        return sum(self.rounds[i].perm != self.rounds[i - 1].perm for i in range(1, len(self.rounds)))

    @property
    def changed(self) -> bool:
        return self.rounds[-1].perm != self.rounds[0].perm

    @property
    def last_change(self) -> int:
        """Runde der letzten Änderung der Anordnung (0: sie ändert sich nie)."""
        ch = [i for i in range(1, len(self.rounds)) if self.rounds[i].perm != self.rounds[i - 1].perm]
        return ch[-1] if ch else 0


def base_weight(inst, pads) -> float:
    """Grundgewicht für das Ankergewicht: das der Pads, sonst der Median der positiven Flüsse (mindestens 1)."""
    if pads.count:
        return float(pads.base)
    pos = np.asarray(inst.F)[np.asarray(inst.F) > 0]
    return max(1.0, float(round(float(np.median(pos))))) if len(pos) else 1.0


def run_loop(inst, pads, c_lin, legalizer: str = C.DEFAULT_LEGALIZER, rounds: int = C.DEFAULT_ROUNDS, reweights: int = C.DEFAULT_REWEIGHTS, anchor: float = C.DEFAULT_ANCHOR) -> Loop:
    """Runde 0: Platzierung ohne Pseudo-Anker (Stück 9) und Legalisierung; Runde t: Pseudo-Anker am legalen Platz der Vorrunde mit λ_t = anchor · Grundgewicht · t, neu platzieren, legalisieren."""
    F, D = np.asarray(inst.F, dtype=float), np.asarray(inst.D, dtype=float)
    base = base_weight(inst, pads)
    out = []
    perm = None
    for t in range(rounds + 1):
        if t == 0:
            lam, X = 0.0, Pl.positions(inst, pads, reweights)
        else:
            lam = anchor * base * t
            X = positions_anchored(inst, pads, inst.xy[perm].astype(float), lam, reweights)
        perm = legalize(inst, X, legalizer)
        out.append(Round(t, lam, X, perm, Pl.continuous_length(inst, X, pads), Pl.cost_of(perm, F, D, c_lin), float(np.abs(X - inst.xy[perm].astype(float)).sum())))
    return Loop(legalizer, reweights, anchor, tuple(out))
