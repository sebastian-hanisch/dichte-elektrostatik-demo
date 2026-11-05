"""Kennzahlen, Vergleiche über Instanzen und Verdikte. Jedes Verdikt ist eine reine Funktion (Art, Text); die App ruft st.<Art>(Text).

Alle Aussagen gelten für die gerechneten Instanzen, die sechs erfundenen Familien, die **erfundenen Pads** (Zahl, Ort, Gewicht, Verbindungen) und die **eigene, einfache Fassung** der Dichtestrafe (Randzone, Weichzeichnen, Strafverfahren mit Rückverfolgung);
Zeiten stehen nie in Verdikten. Aufwand wird in **Auswertungen** gezählt (Länge und Dichte samt Gradient, jeder Versuch der Rückverfolgung zählt), die Schleife aus Stück 10 in linearen Systemen: die Einheiten sind nicht vergleichbar.
Die Lücke ist (Kosten − Vergleichswert) / Vergleichswert; der Vergleichswert ist bis n = 9 das aufgezählte Optimum, ab n = 12 die beste von 100 Abstiegen (die Lücke ist dann eine **obere** Grenze der wahren Lücke; 0 heißt „so gut wie der Vergleichswert“)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import dla_constants as C
import dla_density as Dn
import dla_loop as Lp
import dla_penalty as Pn
import dla_place as Pl
from dla_instance import hand_instance, make_instance


@dataclass(frozen=True)
class Settings:
    family: str = C.DEFAULT_FAMILY
    n: int = C.DEFAULT_N
    seed: int = C.DEFAULT_SEED
    pads: int = C.DEFAULT_PADS
    model: str = C.DEFAULT_MODEL
    margin: float = C.DEFAULT_MARGIN
    width: float = C.DEFAULT_WIDTH
    rule: str = C.DEFAULT_RULE


# ----------------------------------------------------------------------------- Formatierung

def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma und typografischem Minuszeichen."""
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


def pct(x: float, digits: int = 0) -> str:
    return de(100.0 * x, digits) + " %"


def fam(name: str) -> str:
    return C.FAMILY_NAMES.get(name, name)


def num(x: float) -> str:
    """Ganze Zahl ohne Nachkommastelle, sonst eine Stelle: 27 / 27,5 / −12,1."""
    return str(int(round(x))) if abs(x - round(x)) < 1e-9 else de(x, 1)


def count(x: float) -> str:
    """Anzahl mit Leerzeichen als Tausendertrenner: 1200 → „1 200“."""
    return C.n_text(int(round(x)))


def rel(a: float, b: float) -> float:
    """(a − b) / b; NaN, wenn b = 0; Rundungsreste unter 1e-12 werden zu 0."""
    if not b:
        return float("nan")
    r = (a - b) / b
    return 0.0 if abs(r) < 1e-12 else r


def median_of(rows: list, key) -> float:
    return float(np.median([r[key] if not callable(key) else key(r) for r in rows]))


def tx(x: float) -> str:
    """Zahl mit Dezimalkomma ohne überflüssige Nullen: 0,25 / 0,5 / 2."""
    return de(x, 2).rstrip("0").rstrip(",")


def stufe(k: int) -> str:
    return f"Stufe {k + 1}"


# ----------------------------------------------------------------------------- eine Instanz

def instance_data(s: Settings) -> dict:
    """Instanz, Pads, Vergleichswert, Stück 9 (einmal runden), Stück 10 (Schleife) und die drei Dichtemodelle mit allen Stufen."""
    inst = make_instance(s.family, s.n, s.seed)
    pads = Pl.make_pads(inst, s.pads, s.seed, C.DEFAULT_FACTOR)
    c = Pl.linear_costs(inst, pads)
    ref, perm_ref, exact, mean = Pl.reference(inst, c, s.seed)
    F, D = inst.F.astype(float), inst.D.astype(float)
    X9 = Pl.positions(inst, pads, C.ROUND_REWEIGHTS)
    p9 = Pl.round_to_slots(inst, X9)
    loop = Lp.run_loop(inst, pads, c)
    runs = {m: Pn.run_penalty(inst, pads, c, m, s.margin, s.width, s.seed) for m in C.MODELS}
    return {"inst": inst, "pads": pads, "c": c, "ref": ref, "ref_perm": perm_ref, "exact": exact, "mean": mean, "gap_mean": rel(mean, ref) if mean is not None else None,
            "X9": X9, "p9": p9, "cost9": Pl.cost_of(p9, F, D, c), "loop": loop, "cost10": loop.rounds[-1].upper, "runs": runs}


def main_result(s: Settings, v: dict):
    """Ergebnis des gewählten Modells mit der gewählten Rundungsregel: (Stufe, Index)."""
    pen = v["runs"][s.model]
    k = pen.best_index if s.rule == "best" else len(pen.stages) - 1
    return pen.stages[k], k


def start_field(s: Settings, v: dict) -> dict:
    """Dichte und Kraft am Startwert (erste Stufe) für das gewählte Modell: ρ, Potential, Kraft je Bauteil (−Gradient der Dichtestrafe), größte Dichte, Überlauf mit Einheitsquadraten."""
    inst = v["inst"]
    pen = v["runs"][s.model]
    bins = Dn.Bins(inst, s.margin, s.width)
    lo, hi = bins.center_box()
    X = np.clip(pen.start, lo, hi)
    rho, _ = bins.density(X)
    _, g, _ = bins.energy(X, s.model)
    unit = Dn.Bins(inst, s.margin, 1.0)
    return {"bins": bins, "X": X, "rho": rho, "psi": bins.potential(X), "force": -g, "rho_max": float(rho.max() / bins.target), "overflow": unit.overflow(unit.density(X)[0])}


# ----------------------------------------------------------------------------- Zeilen für die Vergleiche

KEYS = ("s9", "s10") + tuple(m for m in C.MODELS) + tuple(m + "_best" for m in C.MODELS)


def row_of(family: str, n: int, seed: int, p: int) -> dict:
    """Eine Instanz mit den Standardwerten (Randzone 0,5, Ladungsbreite 3): Kosten von Stück 9, Stück 10 und der drei Modelle (letzte und billigste Stufe), Kennzahlen je Stufe."""
    inst = make_instance(family, n, seed)
    pads = Pl.make_pads(inst, p, seed, C.DEFAULT_FACTOR)
    c = Pl.linear_costs(inst, pads)
    ref, _, exact, mean = Pl.reference(inst, c, seed)
    F, D = inst.F.astype(float), inst.D.astype(float)
    cost = {"s9": Pl.cost_of(Pl.round_to_slots(inst, Pl.positions(inst, pads, C.ROUND_REWEIGHTS)), F, D, c), "s10": Lp.run_loop(inst, pads, c).rounds[-1].upper}
    row = {"family": family, "n": n, "seed": seed, "p": p, "ref": ref, "exact": exact, "gap_mean": rel(mean, ref) if mean is not None else float("nan"), "cost": cost, "stage_gap": {}, "stage_ov": {}, "stage_disp": {}, "ov": {}, "disp": {}, "spread": {}, "evals": {}}
    for m in C.MODELS:
        pen = Pn.run_penalty(inst, pads, c, m, C.DEFAULT_MARGIN, C.DEFAULT_WIDTH, seed)
        cost[m], cost[m + "_best"] = pen.stages[-1].cost, min(pen.costs)
        row["stage_gap"][m] = [rel(x, ref) for x in pen.costs]
        row["stage_ov"][m] = [s_.overflow for s_ in pen.stages]
        row["stage_disp"][m] = [s_.disp for s_ in pen.stages]
        row["ov"][m], row["disp"][m], row["spread"][m], row["evals"][m] = pen.stages[-1].overflow, pen.stages[-1].disp, pen.stages[-1].spread, pen.evals
    row["gap"] = {k: rel(cost[k], ref) for k in KEYS}
    row["hit"] = {k: cost[k] <= ref + 1e-9 for k in KEYS}
    return row


def sweep(family: str, n: int, seeds, pads_list) -> list:
    return [row_of(family, n, s, p) for s in seeds for p in pads_list]


def pooled(by_family: dict) -> list:
    return [r for f in C.FAMILIES if f in by_family for r in by_family[f]]


def select(rows: list, p=None) -> list:
    return [r for r in rows if p is None or r["p"] == p]


def gap_summary(rows: list, p, key: str) -> dict:
    """Median der Lücke und Zahl der Instanzen, in denen der Vergleichswert erreicht wird, für `key` ∈ KEYS (p = None: alle Pad-Zahlen)."""
    rs = select(rows, p)
    return {"count": len(rs), "gap": float(np.median([r["gap"][key] for r in rs])), "mean": float(np.mean([r["gap"][key] for r in rs])), "hit": sum(r["hit"][key] for r in rs)}


def model_summary(rows: list, p, model: str) -> dict:
    """Median von Überlauf, Verschiebung bei der Rundung, Streuung und Auswertungen am Ende (letzte Stufe)."""
    rs = select(rows, p)
    return {"ov": median_of(rs, lambda r: r["ov"][model]), "disp": median_of(rs, lambda r: r["disp"][model]), "spread": median_of(rs, lambda r: r["spread"][model]), "evals": median_of(rs, lambda r: r["evals"][model])}


def stage_medians(rows: list, model: str) -> dict:
    """Median je Stufe von Lücke, Überlauf und Verschiebung (Listen der Länge STAGES)."""
    return {"gap": [float(np.median([r["stage_gap"][model][k] for r in rows])) for k in range(C.STAGES)],
            "ov": [float(np.median([r["stage_ov"][model][k] for r in rows])) for k in range(C.STAGES)],
            "disp": [float(np.median([r["stage_disp"][model][k] for r in rows])) for k in range(C.STAGES)]}


def compare_counts(rows: list, key: str, base: str) -> tuple:
    """(besser, schlechter, gleich): Zahl der Instanzen, in denen `key` weniger, mehr oder gleich viel kostet wie `base`."""
    b = sum(r["cost"][key] < r["cost"][base] - 1e-9 for r in rows)
    w = sum(r["cost"][key] > r["cost"][base] + 1e-9 for r in rows)
    return b, w, len(rows) - b - w


def mean_gap(rows: list, p: int) -> float:
    return median_of([r for r in rows if r["p"] == p], "gap_mean")


# ----------------------------------------------------------------------------- Ablation (Randzone, Ladungsbreite)

ABLATION = {**{f"m{m}": (m, C.DEFAULT_WIDTH) for m in C.MARGIN_CHOICES}, **{f"w{w}": (C.DEFAULT_MARGIN, w) for w in C.WIDTH_CHOICES}}   # Schlüssel → (Randzone, Ladungsbreite)


def ablation_row(family: str, n: int, seed: int, p: int) -> dict:
    """Eine Instanz, elektrostatisch, mit jeder Randzone und jeder Ladungsbreite (je andere Größe auf dem Standardwert): Lücke und Überlauf der letzten Stufe."""
    inst = make_instance(family, n, seed)
    pads = Pl.make_pads(inst, p, seed, C.DEFAULT_FACTOR)
    c = Pl.linear_costs(inst, pads)
    ref = Pl.reference(inst, c, seed)[0]
    F, D = inst.F.astype(float), inst.D.astype(float)
    row = {"family": family, "seed": seed, "p": p, "ref": ref, "gap9": rel(Pl.cost_of(Pl.round_to_slots(inst, Pl.positions(inst, pads, C.ROUND_REWEIGHTS)), F, D, c), ref), "gap": {}, "ov": {}}
    done = {}
    for key, (margin, width) in ABLATION.items():
        if (margin, width) not in done:
            done[(margin, width)] = Pn.run_penalty(inst, pads, c, "electro", margin, width, seed)
        st = done[(margin, width)].stages[-1]
        row["gap"][key], row["ov"][key] = rel(st.cost, ref), st.overflow
    return row


def ablation_sweep(family: str, n: int, seeds, p: int = C.DEFAULT_PADS) -> list:
    return [ablation_row(family, n, s, p) for s in seeds]


def ablation_summary(rows: list, key: str) -> dict:
    return {"gap": float(np.median([r["gap"][key] for r in rows])), "ov": float(np.median([r["ov"][key] for r in rows]))}


# ----------------------------------------------------------------------------- Verdikte

def _head(s: Settings, v: dict) -> str:
    inst = v["inst"]
    return f"Familie **{fam(s.family)}**, {s.n} Bauteile, Raster {inst.rows}×{inst.cols}, Seed {s.seed}, {s.pads} Pads"


def _refname(v: dict) -> str:
    return "das aufgezählte Optimum" if v["exact"] else "die beste von 100 Abstiegen (obere Grenze)"


def _gap(cost: float, v: dict) -> str:
    return f"Kosten {num(cost)} (Lücke {pct(rel(cost, v['ref']), 1)})"


def verdict_core(s: Settings, v: dict):
    st, k = main_result(s, v)
    pen = v["runs"][s.model]
    kind = "success" if st.cost <= v["ref"] + 1e-9 else "info"
    return kind, (f"{_head(s, v)}: {C.MODEL_NAMES[s.model]} (Randzone {tx(s.margin)}, Ladungsbreite {tx(s.width)} → 1), Rundungsregel {C.RULE_NAMES[s.rule].lower()} ({stufe(k)} von {len(pen.stages)}): nach der Rundung {_gap(st.cost, v)} gegen {_refname(v)} {num(v['ref'])}; "
                  f"Überlauf {de(st.overflow, 2)}, {count(st.evals)} Auswertungen. Zum Vergleich: einmal runden (Stück 9) {_gap(v['cost9'], v)}, die Schleife aus Stück 10 {_gap(v['cost10'], v)}. Eine Instanz, kein Befund über die Familie.")


def verdict_field(s: Settings, v: dict):
    f = start_field(s, v)
    if s.model == "none":
        return "info", (f"{_head(s, v)}, am Startwert (Platzierung aus Stück 9, leicht gestreut): größte Dichte das {de(f['rho_max'], 1)}-fache des Sollwerts, Überlauf {de(f['overflow'], 2)} (Einheitsquadrate). Ohne Dichtestrafe wirkt keine Kraft: der Klumpen bleibt, "
                        "bis die Rundung ihn auf die Plätze verteilt.")
    big = float(np.linalg.norm(f["force"], axis=1).max())
    return "info", (f"{_head(s, v)}, am Startwert (Platzierung aus Stück 9, leicht gestreut): größte Dichte das {de(f['rho_max'], 1)}-fache des Sollwerts, Überlauf {de(f['overflow'], 2)} (Einheitsquadrate); "
                    f"die Dichtestrafe ({C.MODEL_NAMES[s.model]}, Ladungsbreite {tx(s.width)}) schiebt die Bauteile vom Klumpen weg, die größte Kraft hat den Betrag {de(big, 2)} (Pfeile, nicht maßstäblich).")


def verdict_path(s: Settings, v: dict):
    pen = v["runs"][s.model]
    gaps = [rel(st.cost, v["ref"]) for st in pen.stages]
    kb = pen.best_index
    return "info", (f"{_head(s, v)}, {C.MODEL_NAMES[s.model]}: Lücke nach der Rundung je Stufe von {pct(gaps[0], 1)} (Stufe 1) über {pct(gaps[kb], 1)} in der besten ({stufe(kb)}) auf {pct(gaps[-1], 1)} am Ende; "
                    f"der Überlauf fällt von {de(pen.stages[0].overflow, 2)} auf {de(pen.stages[-1].overflow, 2)}, die Verschiebung bei der Rundung von {de(pen.stages[0].disp, 1)} auf {de(pen.stages[-1].disp, 1)}. "
                    + ("Die gleichmäßigste (letzte) Stufe ist hier nicht die billigste." if pen.stages[kb].cost < pen.stages[-1].cost - 1e-9 else "Hier ist die letzte Stufe so billig wie die billigste.") + " Eine Instanz.")


def verdict_models(s: Settings, v: dict):
    parts = []
    for m in C.MODELS:
        st, k = main_result(Settings(s.family, s.n, s.seed, s.pads, m, s.margin, s.width, s.rule), v)
        parts.append(f"{C.MODEL_NAMES[m]}: {_gap(st.cost, v)}, Überlauf {de(st.overflow, 2)}, Verschiebung {de(st.disp, 1)}")
    return "info", (f"{_head(s, v)}, Rundungsregel {C.RULE_NAMES[s.rule].lower()}, Vergleichswert {num(v['ref'])} ({_refname(v)}): " + "; ".join(parts) + f". Einmal runden (Stück 9): {_gap(v['cost9'], v)}; Schleife (Stück 10): {_gap(v['cost10'], v)}. Eine Instanz, kein Befund über die Familie.")


def verdict_sweep(n: int, rows: list):
    exact = rows[0]["exact"]
    labels = {"s9": "Stück 9 (einmal runden)", "s10": "Stück 10 (Schleife)", "none": "keine Strafe", "overflow": "lokale Strafe", "electro": "elektrostatisch", "electro_best": "elektrostatisch, billigste Stufe"}
    parts = []
    for k, name in labels.items():
        sm = gap_summary(rows, None, k)
        parts.append(f"{name}: {pct(sm['gap'], 1)} (erreicht in {sm['hit']})")
    b, w, g = compare_counts(rows, "electro", "s10")
    ms = {m: model_summary(rows, None, m) for m in C.MODELS}
    return "info", (f"Bei n = {n}, Median der Lücke über {len(rows)} Instanzen (alle Pad-Zahlen): " + "; ".join(parts) + f". Elektrostatisch gegen Stück 10 Instanz für Instanz: besser in {b}, schlechter in {w}, gleich in {g}. "
                    f"Überlauf am Ende (Median): elektrostatisch {de(ms['electro']['ov'], 2)}, lokal {de(ms['overflow']['ov'], 2)}, keine Strafe {de(ms['none']['ov'], 2)}; Verschiebung bei der Rundung {de(ms['electro']['disp'], 1)} / {de(ms['overflow']['disp'], 1)} / {de(ms['none']['disp'], 1)}; "
                    f"Auswertungen je Instanz (elektrostatisch) {count(ms['electro']['evals'])}." + ("" if exact else " Der Vergleichswert ist die beste von 100 Abstiegen: obere Grenze der Lücke; „erreicht“ heißt gleich oder besser."))


def verdict_ablation(rows: list):
    mg = "; ".join(f"Randzone {tx(m)}: {pct(ablation_summary(rows, f'm{m}')['gap'], 1)}, Überlauf {de(ablation_summary(rows, f'm{m}')['ov'], 2)}" for m in C.MARGIN_CHOICES)
    wd = "; ".join(f"Ladungsbreite {tx(w)}: {pct(ablation_summary(rows, f'w{w}')['gap'], 1)}, Überlauf {de(ablation_summary(rows, f'w{w}')['ov'], 2)}" for w in C.WIDTH_CHOICES)
    return "info", (f"Elektrostatisch, n = {C.SWEEP_N}, {C.DEFAULT_PADS} Pads, Median über {len(rows)} Instanzen, Lücke der letzten Stufe; einmal runden (Stück 9): {pct(float(np.median([r['gap9'] for r in rows])), 1)}. "
                    f"Randzone (Ladungsbreite {tx(C.DEFAULT_WIDTH)}): {mg}. Ladungsbreite (Randzone {tx(C.DEFAULT_MARGIN)}; 1 heißt ohne Weichzeichnen): {wd}.")


# ----------------------------------------------------------------------------- Handbeispiel

def hand_data() -> dict:
    """Zwei Bauteile auf einem Block aus zwei Plätzen (Raster 1×2): die Ladung, das Bin-Raster, die Dichte und die Kraft von Hand (siehe App und Tests)."""
    inst = hand_instance(((0, 1, 1),), 1, 2)
    bins = Dn.Bins(inst, 0.0, 1.0)
    X = np.array([[0.4, 0.0], [0.6, 0.0]])
    rho, _ = bins.density(X)
    value, g, _ = bins.energy(X, "electro")
    return {"inst": inst, "bins": bins, "X": X, "rho": rho, "value": value, "grad": g}
