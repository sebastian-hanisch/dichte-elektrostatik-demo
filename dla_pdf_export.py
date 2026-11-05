"""PDF-Export des aktuellen Ergebnisses (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Sonderzeichen werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

import dla_constants as C
from dla_evaluation import Settings, count, de, main_result, num, pct, rel, tx

_REPLACE = {"→": "->", "–": "-", "—": "-", "−": "-", "≤": "<=", "≥": ">=", "·": "*", "×": "x", "Δ": "Delta", "↔": "<->", "„": '"', "“": '"', "²": "2", "³": "3", "⁻": "-", "¹": "1", "½": "1/2", "⁴": "4", "λ": "lambda", "ρ": "rho", "ψ": "psi"}


def _s(text) -> str:
    out = str(text).replace("**", "").replace("`", "")
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(s: Settings, v: dict) -> bytes:
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Dichte und Drahtlänge: Dichtestrafe statt harter Legalisierung"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    inst = v["inst"]
    how = "alle Anordnungen aufgezählt" if v["exact"] else "nicht aufgezählt, beste von 100 Abstiegen als Vergleich (obere Grenze)"
    pdf.multi_cell(0, 5, _s(f"Familie {C.FAMILY_NAMES[s.family]}, {s.n} Bauteile, {inst.rows}x{inst.cols}-Raster, Seed {s.seed}; {s.pads} Pads; Randzone {tx(s.margin)}; Ladungsbreite {tx(s.width)} -> 1; Modell {C.MODEL_NAMES[s.model]}; {C.RULE_NAMES[s.rule]}; {how}. "
                            "Eine Instanz, kein Befund über die Familie. Erfundene Pads, eigene einfache Fassung der Dichtestrafe. Aufwand in Auswertungen, nicht in Zeit. Alles simuliert."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 9)
    for w, h in ((55, "Verfahren"), (45, "Kosten (Lücke)"), (25, "Überlauf"), (30, "Auswertungen")):
        pdf.cell(w, 6, _s(h), border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    rows = [("Einmal runden (Stück 9)", v["cost9"], "", ""), ("Schleife (Stück 10)", v["cost10"], "", "")]
    for m in C.MODELS:
        st, _ = main_result(Settings(s.family, s.n, s.seed, s.pads, m, s.margin, s.width, s.rule), v)
        rows.append((f"{C.MODEL_NAMES[m]}, {C.RULE_NAMES[s.rule]}", st.cost, de(st.overflow, 2), count(st.evals)))
    for name, cost, ov, ev in rows:
        pdf.cell(55, 5.5, _s(name), border=1)
        pdf.cell(45, 5.5, _s(f"{num(cost)} ({pct(rel(cost, v['ref']), 1)})"), border=1)
        pdf.cell(25, 5.5, _s(ov), border=1)
        pdf.cell(30, 5.5, _s(ev), border=1)
        pdf.ln()
    pdf.ln(3)
    st, k = main_result(s, v)
    ref_name = "Optimum (aufgezählt)" if v["exact"] else "Beste gefundene Lösung (obere Grenze)"
    pdf.multi_cell(0, 5, _s(f"{ref_name}: {num(v['ref'])}. Gewähltes Modell mit gewählter Regel: Stufe {k + 1}, Kosten {num(st.cost)} (Lücke {pct(rel(st.cost, v['ref']), 1)}), {count(st.evals)} Auswertungen."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Quelle: dichte-elektrostatik-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
