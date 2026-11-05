"""Strafverfahren: glatte lineare Länge plus Dichtestrafe, λ wächst von Stufe zu Stufe, nach jeder Stufe wird gerundet (eigene Fassung, „ePlace-artig“).

Kern von Stück 9 und 10 (`dla_place.py`: Instanz, Pads, Kosten, Platzierung, Rundung). Neu: die Orte X (n × 2) als Variable, f(X) = W(X) + λ · N(X).
**W** ist die glatte lineare Länge Σ F · (√(dx² + γ²) + √(dy² + γ²)) über Paare und Pads, γ = EPS (die Kosten der Anordnung, an den Ecken abgerundet; die Umgewichtung F / max(|Δ|, ε) aus Stück 9 ist ihr Gradient).
**N** kommt aus `dla_density.py` (elektrostatisch, lokale Überlaufstrafe oder keine). Ablauf: Startwert = Platzierung aus Stück 9 (zwei Umgewichtungen) plus eine kleine feste Streuung (SplitMix64, damit gestapelte Bauteile auseinanderlaufen können);
λ beginnt so, dass die Beträge der beiden Gradienten gleich sind, und wächst je Stufe um GROWTH; je Stufe INNER Abstiegsschritte mit Armijo-Rückverfolgung; die Ladungsbreite sinkt von `width` linear auf 1 (Weichzeichnen, dann schärfen);
die Orte bleiben im Gebiet (Block der Plätze plus Randzone). Gezählt werden die **Auswertungen** von f samt Gradient (jeder Versuch der Rückverfolgung zählt mit). Nach jeder Stufe wird mit der Zuordnung aus Stück 9 gerundet."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import dla_constants as C
import dla_density as Dn
import dla_place as Pl
from dla_rng import SplitMix64


def smooth_length(X, F, pads) -> tuple:
    """(Wert, Gradient (n, 2)) der glatten linearen Länge."""
    g2 = C.EPS ** 2
    d = X[:, None, :] - X[None, :, :]
    r = np.sqrt(d * d + g2)
    val = float((F[:, :, None] * r).sum() / 2)
    g = (F[:, :, None] * d / r).sum(axis=1)
    if pads.count:
        dp = X[:, None, :] - pads.pos[None, :, :]
        rp = np.sqrt(dp * dp + g2)
        val += float((pads.w[:, :, None] * rp).sum())
        g += (pads.w[:, :, None] * dp / rp).sum(axis=1)
    return val, g


@dataclass(frozen=True, eq=False)
class Stage:
    t: int                    # Nummer der Stufe, ab 0
    side: float               # Ladungsbreite dieser Stufe
    lam: float                # Gewicht der Dichtestrafe
    X: np.ndarray             # stetige Orte am Ende der Stufe
    evals: int                # Auswertungen bis zum Ende der Stufe (kumuliert)
    length: float             # glatte lineare Länge W
    penalty: float            # Dichtestrafe N (Wert des Modells dieser Stufe)
    overflow: float           # Überlauf mit Einheitsquadraten (für alle Modelle gleich gemessen)
    perm: list                # gerundete legale Anordnung (Platz je Bauteil)
    cost: float               # Kosten der gerundeten Anordnung
    cont: float               # stetige lineare Länge (ohne Glättung) der Orte
    disp: float               # Verschiebung bei der Rundung: Σ |stetiger Ort − Platz|₁
    spread: float             # Streuung der Orte im Verhältnis zu den Plätzen (Stück 9)


@dataclass(frozen=True, eq=False)
class Penalty:
    model: str
    margin: float
    width: float
    start: np.ndarray         # Startwert (Platzierung aus Stück 9 plus Streuung)
    lam0: float
    stages: tuple

    @property
    def costs(self) -> list:
        return [s.cost for s in self.stages]

    @property
    def evals(self) -> int:
        return self.stages[-1].evals

    @property
    def best_index(self) -> int:
        """Stufe mit den kleinsten Kosten der gerundeten Anordnung (bei Gleichstand die erste)."""
        best = 0
        for i, s in enumerate(self.stages):
            if s.cost < self.stages[best].cost - 1e-9:
                best = i
        return best

    def pick(self, rule: str) -> Stage:
        return self.stages[-1] if rule == "last" else self.stages[self.best_index]


def start_positions(inst, pads, seed: int, jitter: float = C.JITTER) -> np.ndarray:
    """Platzierung aus Stück 9 (zwei Umgewichtungen) plus feste Streuung ±jitter/2 aus SplitMix64."""
    X = Pl.positions(inst, pads, C.ROUND_REWEIGHTS).copy()
    if jitter:
        rng = SplitMix64(seed * 6007 + 5)
        X = X + jitter * (np.array([[rng.random() for _ in range(2)] for _ in range(inst.n)]) - 0.5)
    return X


def run_penalty(inst, pads, c, model: str, margin: float, width: float, seed: int, stages: int = C.STAGES, inner: int = C.INNER, growth: float = C.GROWTH, x0=None) -> Penalty:
    """Das Strafverfahren: `stages` Stufen, je `inner` Abstiegsschritte; Rückgabe mit allen Stufen (Orte, Kosten der Rundung, Überlauf, Auswertungen)."""
    if model not in C.MODELS:
        raise ValueError(f"unbekanntes Dichtemodell {model!r}")
    F, D = np.asarray(inst.F, dtype=float), np.asarray(inst.D, dtype=float)
    n = inst.n
    X = start_positions(inst, pads, seed) if x0 is None else np.array(x0, dtype=float)
    start = X.copy()
    bins = Dn.Bins(inst, margin, width)
    lo, hi = bins.center_box()
    X = np.clip(X, lo, hi)
    unit = Dn.Bins(inst, margin, 1.0)
    evals = 0

    def value_grad(Y, lam):
        w, gw = smooth_length(Y, F, pads)
        nv, gn, _ = bins.energy(Y, model)
        return w + lam * nv, gw + lam * gn, w, nv

    _, gw0 = smooth_length(X, F, pads)
    _, gn0, _ = bins.energy(X, model)
    lam = 0.0 if model == "none" else float(np.abs(gw0).sum() / max(np.abs(gn0).sum(), 1e-12))
    lam0 = lam
    step = 0.5
    out = []
    for stage in range(stages):
        side = width + (1.0 - width) * stage / max(stages - 1, 1)
        bins = Dn.Bins(inst, margin, side)
        for _ in range(inner):
            f, g, _, _ = value_grad(X, lam)
            evals += 1
            if float((g * g).sum()) < 1e-12:
                break
            gmax = float(np.abs(g).max())
            step = min(step * 2.0, C.MAX_MOVE / max(gmax, 1e-12) * 4)
            while True:
                Y = np.clip(X - step * g, lo, hi)
                fy = value_grad(Y, lam)[0]
                evals += 1
                if fy <= f - C.ARMIJO * float(((X - Y) * g).sum()) or step < 1e-9:
                    break
                step *= 0.5
            X = Y
        _, _, w, nv = value_grad(X, lam)
        perm = Pl.round_to_slots(inst, X)
        out.append(Stage(stage, side, lam, X.copy(), evals, w, nv, unit.overflow(unit.density(X)[0]), perm, Pl.cost_of(perm, F, D, c), Pl.continuous_length(inst, X, pads),
                         float(np.abs(X - inst.xy[perm].astype(float)).sum()), Pl.spread(inst, X)))
        lam *= growth
    return Penalty(model, margin, width, start, lam0, tuple(out))
