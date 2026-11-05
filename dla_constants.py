"""Konstanten: Größen, Raster, Familien der Flussmatrix, die Dichtestrafe (Bin-Raster, Stufen, Randzone, Ladungsbreite), feste Reihen der Experimente, Farben und Presets."""

EXACT_SIZES = (6, 8, 9)                               # alle n! Anordnungen werden aufgezählt: das Optimum ist bekannt
LARGE_SIZES = (12, 16)                                # nicht aufgezählt: die beste von 100 Abstiegen aus Zufallsstarts ersetzt das Optimum
SIZES = EXACT_SIZES + LARGE_SIZES
DEFAULT_N = 8
SWEEP_N = 8                                           # Vergleich über feste Instanzen mit aufgezähltem Optimum
LARGE_SWEEP_SIZES = (12, 16)                          # Vergleiche gegen die beste gefundene Lösung
GRID = {6: (2, 3), 8: (2, 4), 9: (3, 3), 12: (3, 4), 16: (4, 4)}    # Plätze als Raster (Zeilen, Spalten), Abstand = Manhattan
SEED_MAX, DEFAULT_SEED = 999999, 48
SWEEP_SEEDS = tuple(range(100000, 100020))            # feste Instanzen: nie ein einzelner Seed als Befund
LARGE_SEEDS = {12: SWEEP_SEEDS[:5], 16: SWEEP_SEEDS[:3]}   # n = 12: fünf, n = 16: drei Instanzen je Familie
STARTS = 100                                          # Zahl der steilsten Abstiege aus Zufallsstarts für die beste gefundene Lösung (ab n = 12)

PAD_CHOICES = (2, 4, 8)                               # Zahl der festen Pads (Anker) auf dem Ring um das Raster
DEFAULT_PADS = 4
PAD_LINKS = 2                                         # jedes Pad hängt an so vielen zufälligen Bauteilen
DEFAULT_FACTOR = 1.0                                  # Pad-Gewicht = Faktor · Grundgewicht (Median der positiven Flüsse, mindestens 1); fest
EPS = 0.3                                             # untere Grenze der Länge in der Umgewichtung F / max(|Δ|, EPS) (in Platzabständen); zugleich die Rundung γ der glatten Länge
CENTER_ANCHOR = 1e-6                                  # schwacher Anker in der Mitte: Bauteile ohne Weg zu einem Pad fallen auf einen Punkt, das System bleibt lösbar
RING_MARGIN = 1.5                                     # Abstand des Pad-Rings vom Raster der Plätze (in Platzabständen)
TIE_DECIMALS = 8                                      # Stellen (relativ zum größten Betrag), auf die Abstandsquadrate und Koordinaten vor Zuordnung und Sortierung gerundet werden
LOOP_ROUNDS, LOOP_REWEIGHTS, LOOP_ANCHOR = 15, 2, 0.2  # die Schleife aus Stück 10 (Zuordnung) als Vergleich
DEFAULT_ROUNDS, DEFAULT_REWEIGHTS, DEFAULT_ANCHOR = LOOP_ROUNDS, LOOP_REWEIGHTS, LOOP_ANCHOR
ROUND_REWEIGHTS = 2                                   # Umgewichtungen der Platzierung für den Startwert und die Rundung aus Stück 9
FACTOR_CHOICES = (0.25, 1.0, 4.0, 16.0)               # Faktoren auf das Pad-Gewicht, nur für Kern-Tests (aus Stück 9)
ANCHOR_CHOICES = (0.02, 0.2, 1.0)                     # nur für die Kern-Tests der Schleife (aus Stück 10)
REWEIGHT_CHOICES = (0, 2)
LEGALIZERS = ("assign", "bisect")
DEFAULT_LEGALIZER = "assign"

MODELS = ("electro", "overflow", "none")              # Dichtestrafe: elektrostatisch (Poisson), lokale Überlaufstrafe, keine
MODEL_NAMES = {"electro": "Elektrostatisch", "overflow": "Lokale Überlaufstrafe", "none": "Keine Dichtestrafe"}
MODEL_HELP = {"electro": "Ladung minus Mittelwert, Poisson-Gleichung mit Neumann-Rand: jedes Bauteil spürt alle anderen (langreichweitig).",
              "overflow": "Quadrat des Überlaufs je Bin (Dichte über dem Sollwert): nur dort, wo es zu voll ist (kurzreichweitig).",
              "none": "Nur die glatte lineare Länge: alles zieht sich zu den Pads und zueinander, es bleibt bei einem Klumpen."}
DEFAULT_MODEL = "electro"
MARGIN_CHOICES = (0.0, 0.25, 0.5, 1.0)                # Randzone um den Block der Plätze, in Platzabständen
DEFAULT_MARGIN = 0.5
WIDTH_CHOICES = (1.0, 2.0, 3.0, 4.0)                  # Kantenlänge der Ladung zu Beginn (Platzabstände); sie sinkt linear auf 1 (1 = ohne Weichzeichnen)
DEFAULT_WIDTH = 3.0
RULES = ("last", "best")                              # Rundungsregel: die letzte Stufe oder die billigste der legalen Anordnungen aller Stufen
RULE_NAMES = {"last": "Letzte Stufe", "best": "Billigste Stufe"}
DEFAULT_RULE = "last"
BINS_PER_SLOT = 2                                     # Bins je Platzabstand und Achse
STAGES = 12                                           # Stufen des Strafverfahrens (feste Zahl)
INNER = 25                                            # Abstiegsschritte je Stufe
GROWTH = 1.8                                          # Wachstum von λ je Stufe
JITTER = 0.3                                          # Breite der festen Streuung des Startwerts (Platzabstände, ±0,15)
MAX_MOVE = 0.25                                       # größte Verschiebung eines Bauteils je Schritt (nur zur Schranke der Schrittweite)
ARMIJO = 1e-4                                         # Armijo-Konstante der Rückverfolgung

FAMILIES = ("netlist", "uniform", "sparse", "cluster", "star", "skewed")
DEFAULT_FAMILY = "cluster"
FAMILY_NAMES = {"netlist": "Netzliste", "uniform": "Gleichverteilt", "sparse": "Dünn besetzt", "cluster": "Zwei Cluster", "star": "Stern", "skewed": "Wenige starke Flüsse"}
FAMILY_HELP = {
    "netlist": "Netzliste aus Dreiergruppen (IC, Kondensator, Widerstand), jedes Paar in einem Netz zählt Fluss 1: die Familie aus Stück 1.",
    "uniform": "Jedes Paar von Bauteilen hat einen zufälligen Fluss von 0 bis 9, gleichverteilt: dichte Matrix ohne Struktur.",
    "sparse": "Nur drei von zehn Paaren haben Fluss (1 bis 9), die übrigen keinen: dünn besetzt ohne weitere Struktur.",
    "cluster": "Zwei Gruppen von Bauteilen: innerhalb einer Gruppe starker Fluss (8 bis 10), zwischen den Gruppen schwacher (0 bis 2).",
    "star": "Ein Bauteil (die Nabe) hat starken Fluss (1 bis 9) zu allen anderen; zwischen den übrigen höchstens 1.",
    "skewed": "Die Flüsse sind schief verteilt: viele Paare ohne oder mit kleinem Fluss, wenige mit sehr großem (bis 60).",
}

# Netzliste (wie in Stück 1 bis 5)
N_NETS_PER_COMPONENT = 1.5
NET_SIZES = (2, 3, 4)
NET_WEIGHTS = (5, 3, 2)
LOCAL_NET_PERCENT = 70
GROUP = 3
# Dünn besetzt, Cluster, Stern, schief (wie in Stück 2)
SPARSE_PERCENT = 30
UNIFORM_MAX = 9
CLUSTER_STRONG, CLUSTER_WEAK_MAX = 8, 2
SKEW_VALUES = (0, 1, 2, 4, 8, 16, 32, 60)
SKEW_WEIGHTS = (45, 22, 13, 8, 6, 3, 2, 1)           # ganzzahlige Gewichte (Summe 100), schweres Ende

HAND_FLOWS = ((0, 1, 5), (1, 2, 1))                   # Handbeispiel 1: drei Bauteile auf einer Zeile, Fluss A–B 5, B–C 1
HAND_GRID = (1, 3)
HAND_GAP_FLOWS = ((0, 2, 1), (0, 3, 1), (1, 2, 1), (1, 3, 1))   # Handbeispiel 3: vier Bauteile auf einer Zeile, A und B gegen C und D
HAND_GAP_GRID = (1, 4)
HAND_GRID_FLOWS = ((0, 1, 3), (0, 2, 1), (1, 3, 2), (2, 3, 4))  # Handbeispiel 2 (aus Stück 1): Raster 2×2
HAND_GRID_GRID = (2, 2)

COLORS = {"best": "#2F6B65", "local": "#c0504d", "pad": "#b08a2e", "cont": "#c0504d", "other": "rgba(150,150,150,0.6)", "line": "#4c78a8", "ok": "#2F6B65", "warn": "#b08a2e", "bad": "#b34a4a",
          "family": {"netlist": "#4c78a8", "uniform": "#9c755f", "sparse": "#b08a2e", "cluster": "#2F6B65", "star": "#8e6bb0", "skewed": "#c0504d"}}

_BASE = {"family": DEFAULT_FAMILY, "n": DEFAULT_N, "seed": DEFAULT_SEED, "pads": DEFAULT_PADS, "model": DEFAULT_MODEL, "margin": DEFAULT_MARGIN, "width": DEFAULT_WIDTH, "rule": DEFAULT_RULE}
PRESETS = {
    "Standardfall: elektrostatisch": dict(_BASE),
    "Lokale Überlaufstrafe": dict(_BASE, model="overflow"),
    "Keine Dichtestrafe": dict(_BASE, model="none"),
    "Ohne Randzone": dict(_BASE, margin=0.0),
    "Ohne Weichzeichnen": dict(_BASE, width=1.0),
    "Billigste Stufe": dict(_BASE, rule="best"),
    "Zwölf Bauteile": dict(_BASE, n=12),
    "Sechzehn Bauteile": dict(_BASE, n=16),
}
PRESET_HELP = {
    "Standardfall: elektrostatisch": "Acht Bauteile, vier Pads: die elektrostatische Dichtestrafe schiebt den Klumpen auseinander; gerundet wird die letzte Stufe.",
    "Lokale Überlaufstrafe": "Dieselbe Aufgabe mit einer einfachen lokalen Strafe auf überfüllte Bins (nur dort, wo es zu voll ist); über viele Instanzen (Experiment 3) bleibt die Anordnung meist schlechter.",
    "Keine Dichtestrafe": "Nur die glatte Länge: die Bauteile bleiben ein Klumpen, die Rundung muss alles auseinanderziehen.",
    "Ohne Randzone": "Die Orte bleiben im Block der Plätze (n Bauteile auf n Plätzen, keine freie Fläche): über viele Instanzen (Experiment 4) wird die Anordnung schlechter.",
    "Ohne Weichzeichnen": "Die Ladung ist von Anfang an ein Einheitsquadrat: der Überlauf bleibt hoch, das Verfahren streut kaum (Experiment 4).",
    "Billigste Stufe": "Gerundet wird nach jeder Stufe; behalten wird die billigste der zwölf legalen Anordnungen (mehr Versuche als die letzte Stufe).",
    "Zwölf Bauteile": "Zwölf Bauteile auf dem 3×4-Raster: nicht aufzählbar; verglichen wird mit der besten von 100 steilsten Abstiegen (obere Grenze der Lücke).",
    "Sechzehn Bauteile": "Sechzehn Bauteile auf dem 4×4-Raster: nicht aufzählbar; die Rechenabhängigkeit des Verfahrens ist hier am größten.",
}


def n_text(n: int) -> str:
    """Anzahl mit Leerzeichen als Tausendertrenner: 362880 → „362 880“."""
    return f"{n:,}".replace(",", " ")


assert set(PRESETS) == set(PRESET_HELP) and set(FAMILY_NAMES) == set(FAMILIES) == set(FAMILY_HELP) and set(MODEL_NAMES) == set(MODELS) == set(MODEL_HELP)
assert all(p["n"] in SIZES and p["family"] in FAMILIES and 0 <= p["seed"] <= SEED_MAX and p["pads"] in PAD_CHOICES and p["model"] in MODELS and p["margin"] in MARGIN_CHOICES and p["width"] in WIDTH_CHOICES and p["rule"] in RULES for p in PRESETS.values())
assert all(r * c == n for n, (r, c) in GRID.items()) and sum(SKEW_WEIGHTS) == 100 and set(GRID) == set(SIZES)
