"""Dichte, Poisson-Gleichung und die beiden Dichtestrafen (eigene Fassung, „ePlace-artig“; ePlace, RePlAce und DREAMPlace sind nicht eingesehen).

Über dem Block der Plätze plus einer Randzone liegt ein Bin-Raster (BINS_PER_SLOT Bins je Platzabstand und Achse). Jedes Bauteil ist eine Ladung der Masse 1, gleichmäßig auf ein Quadrat der Kantenlänge `side` verteilt;
die Überdeckung der Bins wird genau gerechnet (Produkt der Überdeckungen je Achse). Dichte ρ_b = Masse im Bin / Bin-Fläche; Sollwert: gleichmäßige Dichte `n / Fläche` (bei Randzone 0 und side 1: überall 1).

**Elektrostatisch:** ρ̃ = ρ − Mittel(ρ), −∇²ψ = ρ̃ mit Neumann-Rand (Kosinus-Transformation als kleine Matrix, ohne FFT, der nullte Modus entfällt), Energie N = ½ · Bin-Fläche · Σ ρ̃ ψ.
**Lokale Überlaufstrafe:** N = Bin-Fläche · Σ max(ρ − Soll, 0)². Der Gradient nach den Orten ist in beiden Fällen exakt (Überdeckungen sind stückweise linear in den Orten; Tests: Differenzenquotient)."""

from __future__ import annotations

import math

import numpy as np

import dla_constants as C


class Bins:
    """Bin-Raster mit Randzone `margin` (Platzabstände) und Ladungsbreite `side`."""

    def __init__(self, inst, margin: float = 0.0, side: float = 1.0, k: int = C.BINS_PER_SLOT):
        self.n = inst.n
        self.side = float(side)
        self.margin = float(margin)
        self.h = 1.0 / k
        self.area = self.h * self.h
        self.x0 = float(inst.xy[:, 0].min()) - 0.5 - margin
        self.y0 = float(inst.xy[:, 1].min()) - 0.5 - margin
        self.nx = int(round((inst.cols + 2 * margin) * k))
        self.ny = int(round((inst.rows + 2 * margin) * k))
        self.edges_x = self.x0 + self.h * np.arange(self.nx + 1)
        self.edges_y = self.y0 + self.h * np.arange(self.ny + 1)
        self.target = inst.n / (self.nx * self.ny * self.area)
        self.Tx = self._cosine_matrix(self.nx)
        self.Ty = self._cosine_matrix(self.ny)
        self.lx = (2 - 2 * np.cos(np.pi * np.arange(self.nx) / self.nx)) / self.h ** 2
        self.ly = (2 - 2 * np.cos(np.pi * np.arange(self.ny) / self.ny)) / self.h ** 2

    @staticmethod
    def _cosine_matrix(m: int) -> np.ndarray:
        """Orthogonale Kosinus-Transformation (DCT-II) als Matrix: Eigenvektoren der Differenzenmatrix mit Neumann-Rand."""
        T = np.cos(np.pi * np.outer(np.arange(m), np.arange(m) + 0.5) / m) * math.sqrt(2.0 / m)
        T[0, :] = 1.0 / math.sqrt(m)
        return T

    @property
    def region(self) -> tuple:
        """(x_min, x_max, y_min, y_max) der Randzone samt Block, in Plätzen."""
        return self.x0, float(self.edges_x[-1]), self.y0, float(self.edges_y[-1])

    def center_box(self) -> tuple:
        """Grenzen für die Mittelpunkte: die Quadrate der Einheitsgröße bleiben im Gebiet."""
        return np.array([self.x0 + 0.5, self.y0 + 0.5]), np.array([self.edges_x[-1] - 0.5, self.edges_y[-1] - 0.5])

    def _overlap(self, x: np.ndarray, edges: np.ndarray) -> tuple:
        """(Überdeckung (n, Bins), Ableitung nach dem Mittelpunkt (n, Bins)) eines Intervalls der Länge `side` um x mit den Bins einer Achse."""
        lo, hi = x[:, None] - self.side / 2, x[:, None] + self.side / 2
        L, H = edges[None, :-1], edges[None, 1:]
        ov = np.clip(np.minimum(hi, H) - np.maximum(lo, L), 0.0, None)
        dv = ((hi > L) & (hi < H)).astype(float) - ((lo > L) & (lo < H)).astype(float)
        return ov, dv

    def density(self, X):
        """(ρ (ny, nx), Überdeckungen und Ableitungen für den Gradienten)."""
        X = np.asarray(X, dtype=float)
        ox, dx = self._overlap(X[:, 0], self.edges_x)
        oy, dy = self._overlap(X[:, 1], self.edges_y)
        rho = oy.T @ ox / (self.area * self.side ** 2)
        return rho, (ox / self.side, dx / self.side, oy / self.side, dy / self.side)

    def poisson(self, rt: np.ndarray) -> np.ndarray:
        """Lösung ψ von −∇²ψ = ρ̃ (Differenzen, Neumann-Rand), nullter Modus entfernt (ρ̃ hat den Mittelwert 0)."""
        c = self.Ty @ rt @ self.Tx.T
        den = self.ly[:, None] + self.lx[None, :]
        den[0, 0] = 1.0
        c = c / den
        c[0, 0] = 0.0
        return self.Ty.T @ c @ self.Tx

    def energy(self, X, kind: str) -> tuple:
        """(Strafwert N, Gradient (n, 2) nach den Orten, Dichte ρ) für `kind` ∈ MODELS."""
        rho, (ox, dx, oy, dy) = self.density(X)
        if kind == "electro":
            rt = rho - rho.mean()
            field = self.poisson(rt)
            value = 0.5 * self.area * float((rt * field).sum())
        elif kind == "overflow":
            ex = np.maximum(rho - self.target, 0.0)
            field = 2.0 * ex
            value = self.area * float((ex ** 2).sum())
        elif kind == "none":
            return 0.0, np.zeros((self.n, 2)), rho
        else:
            raise ValueError(f"unbekanntes Dichtemodell {kind!r}")
        gx = ((oy @ field) * dx).sum(axis=1)
        gy = ((dy @ field) * ox).sum(axis=1)
        return value, np.column_stack([gx, gy]), rho

    def overflow(self, rho: np.ndarray) -> float:
        """Überlauf: Masse über dem Sollwert, geteilt durch die Gesamtmasse n (0 = nirgends zu voll)."""
        return float(np.maximum(rho - self.target, 0.0).sum() * self.area / self.n)

    def potential(self, X) -> np.ndarray:
        """Potential ψ der elektrostatischen Rechnung (für die Abbildung des Feldes)."""
        rho, _ = self.density(X)
        return self.poisson(rho - rho.mean())


def unit_overflow(inst, X, margin: float) -> float:
    """Überlauf der Orte mit Einheitsquadraten (side 1) im Gebiet mit Randzone `margin`: das Maß, mit dem alle Modelle gleich gemessen werden."""
    b = Bins(inst, margin, 1.0)
    return b.overflow(b.density(X)[0])
