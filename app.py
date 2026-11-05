"""Dichte und Drahtlänge - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 11 der Reihe „Anordnen: relaxieren und runden“: statt die Platzierung hart zu legalisieren (Stück 10), geht die Dichte als weiche Strafe in die Zielfunktion ein, elektrostatisch gerechnet (Poisson-Gleichung), und es wird erst am Ende gerundet.
Eigene, einfache „ePlace-artige“ Fassung. Gemessen werden Lücke, Überlauf und Aufwand gegen keine Strafe, eine lokale Überlaufstrafe und die Schleife aus Stück 10. Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import dla_constants as C
from dla_evaluation import (
    Settings,
    ablation_summary,
    ablation_sweep,
    count,
    de,
    gap_summary,
    hand_data,
    instance_data,
    main_result,
    model_summary,
    num,
    pct,
    pooled,
    rel,
    stage_medians,
    start_field,
    sweep,
    tx,
    verdict_ablation,
    verdict_core,
    verdict_field,
    verdict_models,
    verdict_path,
    verdict_sweep,
)
from dla_layout import dunkles_design, kopf_css
from dla_pdf_export import build_pdf
from dla_presets import bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, sidebar_presets, sync_query_params
from dla_visualization import build_ablation_bars, build_arrangement, build_field, build_model_bars, build_path, build_positions, build_stage_curve, build_stage_medians, build_sweep_bars

st.set_page_config(page_title="Dichte und Drahtlänge – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner="Rechne die Stufen …")
def _instance(settings: Settings):
    return instance_data(settings)


@st.cache_data(show_spinner=False)
def _sweep(family, n, seeds, pads_list):
    return sweep(family, n, seeds, pads_list)


@st.cache_data(show_spinner=False)
def _ablation(family, n, seeds):
    return ablation_sweep(family, n, seeds)


def verdict(pair):
    kind, text = pair
    getattr(st, kind)(text)


def sweep_tables(rows, n):
    """Tabellen und Abbildungen für einen Vergleich über feste Instanzen."""
    st.plotly_chart(build_sweep_bars(rows, gap_summary, f"Lücke nach der Rundung je Verfahren, n = {n}"), width="stretch", key=f"sweep_chart_{n}")
    table = []
    for p in C.PAD_CHOICES:
        row = {"Pads": p}
        for k, name in (("s9", "Stück 9"), ("s10", "Stück 10"), ("none", "keine Strafe"), ("overflow", "lokal"), ("electro", "elektrostatisch"), ("electro_best", "elektrostatisch, billigste")):
            row[name] = pct(gap_summary(rows, p, k)["gap"], 1)
        table.append(row)
    st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
    verdict(verdict_sweep(n, rows))
    st.plotly_chart(build_stage_medians(rows, stage_medians, f"Median je Stufe: Lücke und Überlauf, n = {n}"), width="stretch", key=f"stage_chart_{n}")
    ms = {m: model_summary(rows, None, m) for m in C.MODELS}
    st.dataframe(pd.DataFrame([{"Modell": C.MODEL_NAMES[m], "Überlauf am Ende": de(ms[m]["ov"], 2), "Verschiebung bei der Rundung": de(ms[m]["disp"], 1), "Streuung der Orte": de(ms[m]["spread"], 2), "Auswertungen": count(ms[m]["evals"])} for m in C.MODELS]), hide_index=True, width="stretch")


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in dla_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("⚡ Dichte und Drahtlänge")
st.markdown(
    """
**Hilft eine Dichtestrafe statt der harten Legalisierung?** Dieselbe Aufgabe wie in Stück 1 bis 12: `n` Bauteile auf `n` Plätze, Kosten gleich der Summe aus **Fluss mal Abstand**, dazu **feste Pads**. In Stück 10 wurde die Platzierung in jeder Runde **hart** auf die Plätze gelegt und mit Ankern zurückgebunden.
Hier geht die **Dichte** als **weiche Strafe** in die Zielfunktion ein: glatte Länge plus `λ ·` Dichtestrafe, `λ` wächst, und erst am Ende (hier nach jeder Stufe, zum Messen) wird gerundet. Die Dichtestrafe ist **elektrostatisch** gerechnet: die Bauteile sind Ladungen, die Poisson-Gleichung gibt das Feld, das sie auseinanderschiebt;
zum Vergleich eine **lokale Überlaufstrafe** und **keine** Strafe. Die Fassung ist **eigen und einfach** („ePlace-artig“, kein Nachbau); die sechs Familien der Flussmatrix und die **Pads** sind **eigene Erfindungen** (aus Stück 2 und 9). Stück 11 der Reihe „Anordnen: relaxieren und runden“.

Weiter unten: **fünf Experimente** (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert es", expanded=True):
    st.markdown(
        """
1. **Startwert:** die quadratische Platzierung aus Stück 9 (zwei Umgewichtungen) plus eine kleine feste Streuung, damit gestapelte Bauteile auseinanderlaufen können. Die Orte sind **stetig**; sie dürfen bis in eine **Randzone** um den Block der Plätze.
2. **Glatte Länge:** die Länge der Kosten (Summe über Fluss mal Manhattan-Abstand, dazu die Pads), an den Ecken abgerundet, damit sie einen Gradienten hat.
3. **Dichte:** über dem Block liegt ein Bin-Raster; jedes Bauteil ist eine Ladung der Masse 1, gleichmäßig auf ein Quadrat verteilt (**Ladungsbreite**, sie sinkt von dem gewählten Wert auf 1: erst weich zeichnen, dann schärfen). Sollwert ist die gleichmäßige Dichte.
4. **Strafe:** *elektrostatisch* (Ladung minus Mittelwert, Poisson-Gleichung mit Neumann-Rand, Energie `½ Σ ρ̃ ψ`) oder *lokal* (Quadrat des Überlaufs je Bin) oder *keine*. Der Gradient nach den Orten ist exakt.
5. **Stufen:** zwölf Stufen; je Stufe 25 Abstiegsschritte mit Rückverfolgung; `λ` beginnt so, dass beide Gradienten gleich groß sind, und wächst je Stufe um den Faktor 1,8. Gezählt werden die **Auswertungen** (Länge und Dichte samt Gradient).
6. **Rundung:** nach jeder Stufe die lineare Zuordnung der Orte auf die Plätze (wie Stück 9). Gewertet wird die **letzte Stufe** oder die **billigste** der zwölf legalen Anordnungen.
7. **Die Lücke:** (Kosten − Vergleichswert) / Vergleichswert; bis n = 9 ist der Vergleichswert das aufgezählte Optimum, ab n = 12 die beste von 100 steilsten Abstiegen (obere Grenze der Lücke). **Überlauf** und Verschiebung werden für alle Modelle mit Einheitsquadraten gemessen.
8. **Feste Instanzen:** Vergleiche laufen über feste Seeds (n = 8: 20 Instanzen je Familie, n = 12: 5, n = 16: 3), nie über einen einzelnen.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Die Instanz**")
    family = st.selectbox("Familie der Flussmatrix", C.FAMILIES, key="family_select", format_func=lambda f: C.FAMILY_NAMES[f], help="Wie der Fluss zwischen den Bauteilen verteilt ist. Die Beschreibung steht unter dem Regler.")
    st.caption(C.FAMILY_HELP[family])
    n = st.select_slider("Bauteile n", options=C.SIZES, key="n_select", help="Bis n = 9 wird alles aufgezählt (Vergleichswert: Optimum); ab n = 12 steht die beste von 100 steilsten Abstiegen an seiner Stelle.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Instanz", width="stretch", on_click=randomize_seed)
    pads_n = st.select_slider("Zahl der Pads", options=C.PAD_CHOICES, key="pads_select", help="Feste Anker auf dem Ring um das Raster (Gewicht: Grundgewicht, Median der positiven Flüsse).")
    st.markdown("**Die Dichtestrafe**")
    model = st.selectbox("Modell", C.MODELS, key="model_select", format_func=lambda k: C.MODEL_NAMES[k], help="Elektrostatisch (Poisson), lokale Überlaufstrafe oder keine Strafe.")
    st.caption(C.MODEL_HELP[model])
    margin = st.select_slider("Randzone", options=C.MARGIN_CHOICES, key="margin_select", format_func=tx, help="Wie weit (in Platzabständen) die Orte über den Block der Plätze hinauslaufen dürfen. 0: nur im Block.")
    width = st.select_slider("Ladungsbreite zu Beginn", options=C.WIDTH_CHOICES, key="width_select", format_func=tx, help="Kantenlänge der Ladung eines Bauteils in der ersten Stufe (in Platzabständen); sie sinkt linear auf 1. 1 heißt ohne Weichzeichnen.")
    rule = st.selectbox("Rundungsregel", C.RULES, key="rule_select", format_func=lambda k: C.RULE_NAMES[k], help="Die letzte Stufe oder die billigste der zwölf gerundeten Anordnungen (mehr Versuche).")

sync_query_params({"family_select": family, "n_select": int(n), "seed_input": int(seed), "pads_select": int(pads_n), "model_select": model, "margin_select": float(margin), "width_select": float(width), "rule_select": rule})
settings = Settings(family, int(n), int(seed), int(pads_n), model, float(margin), float(width), rule)

v = _instance(settings)
stage_out, k_out = main_result(settings, v)
pen = v["runs"][model]
ref_label = "Optimum" if v["exact"] else "Beste gefundene Lösung"

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
m1, m2, m3 = st.columns(3)
m4, m5 = st.columns(2)
m1.metric("Lücke nach der Rundung", pct(rel(stage_out.cost, v["ref"]), 1), delta=f"Kosten {num(stage_out.cost)}", delta_color="off", help=f"{C.MODEL_NAMES[model]}, {C.RULE_NAMES[rule]} (Stufe {k_out + 1}), gegen den Vergleichswert.")
m2.metric("Überlauf", de(stage_out.overflow, 2), delta=f"Verschiebung {de(stage_out.disp, 1)}", delta_color="off", help="Masse über dem Sollwert, geteilt durch die Gesamtmasse (Einheitsquadrate); darunter die Summe der Beträge, um die die Rundung die Orte verschiebt.")
m3.metric("Auswertungen", count(stage_out.evals), delta=f"Stufe {k_out + 1} von {len(pen.stages)}", delta_color="off", help="Auswertungen von Länge und Dichte samt Gradient bis zum Ende der gewählten Stufe, jeder Versuch der Rückverfolgung gezählt.")
m4.metric(ref_label, num(v["ref"]), help="Die kleinsten Kosten aller n! Anordnungen (aufgezählt)." if v["exact"] else "Die beste von 100 steilsten Abstiegen aus Zufallsstarts: eine obere Schranke des Optimums.")
m5.metric("Pads", f"{settings.pads}", help="Zahl der festen Pads.")
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(build_positions(v["inst"], v["pads"], stage_out.X, start_field(settings, v)["bins"], f"Orte vor der Rundung (Stufe {k_out + 1})"), width="stretch", key="positions_result")
with c2:
    st.plotly_chart(build_arrangement(v["inst"], stage_out.perm, f"Nach der Rundung (Kosten {num(stage_out.cost)})"), width="stretch", key="arrangement_result")

if st.button("📄 PDF des aktuellen Ergebnisses erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(settings, v))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="dichte-elektrostatik.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Hilft eine Dichtestrafe statt der harten Legalisierung?")
verdict(verdict_core(settings, v))

st.markdown("---")

# --- Experiment 1 ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Das Feld")
st.caption("Die Dichte am Startwert (Farbe, in Vielfachen des Sollwerts), die Orte der Bauteile und die Kraft der Dichtestrafe auf jedes Bauteil (Pfeile, auf die größte Kraft normiert). Gestrichelt das Gebiet aus Block der Plätze und Randzone.")
st.plotly_chart(build_field(v["inst"], v["pads"], start_field(settings, v), model, f"Dichte und Kraft am Start ({C.MODEL_NAMES[model]}, Ladungsbreite {tx(width)})"), width="stretch", key="field")
verdict(verdict_field(settings, v))
with st.expander("Handbeispiel zum Nachrechnen"):
    h = hand_data()
    b = h["bins"]
    st.markdown(
        f"**Zwei Bauteile auf zwei Plätzen** (Raster 1×2, Plätze bei x = 0 und x = 1, Randzone 0, Ladungsbreite 1, zwei Bins je Platzabstand: vier Spalten der Breite 0,5 und zwei Zeilen). Die Bauteile stehen bei x = 0,4 und x = 0,6; ihre Quadrate decken [−0,1; 0,9] und [0,1; 1,1] ab. "
        f"Überdeckung je Spalte: Bauteil 1 bedeckt 0,1 / 0,5 / 0,4 / 0 , Bauteil 2 bedeckt 0 / 0,4 / 0,5 / 0,1; in y ist jede Zeile ganz bedeckt (0,5). Dichte = Überdeckung · 0,5 / Bin-Fläche (0,25): **{', '.join(de(x, 1) for x in h['rho'][0])}** in jeder der beiden Zeilen; Sollwert `n / Fläche = 2 / 2 = 1`. "
        f"Überlauf = Masse über dem Sollwert / Gesamtmasse = (2 Spalten · 2 Zeilen · 0,8 · 0,25) / 2 = **0,4**. Die Dichtestrafe schiebt Bauteil 1 nach links und Bauteil 2 nach rechts (Gradient {de(h['grad'][0, 0], 3)} und {de(h['grad'][1, 0], 3)}, in y null), also auseinander."
    )

st.markdown("---")

# --- Experiment 2 ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Der Weg über die Stufen")
st.caption("Oben die Lücke nach der Rundung je Stufe für die drei Modelle (gestrichelt: einmal runden aus Stück 9 und die Schleife aus Stück 10), unten der Überlauf; darunter die Orte nach den Stufen 1, 4, 8 und 12 des gewählten Modells.")
st.plotly_chart(build_stage_curve(v), width="stretch", key="stage_curve")
st.plotly_chart(build_path(v["inst"], v["pads"], pen, start_field(settings, v)["bins"], f"Orte nach den Stufen ({C.MODEL_NAMES[model]})"), width="stretch", key="path")
verdict(verdict_path(settings, v))

st.markdown("---")

# --- Experiment 3 ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Die Modelle im Vergleich")
st.caption("Die Kosten nach der Rundung für dieselbe Instanz: einmal runden (Stück 9), die Schleife aus Stück 10, keine Strafe, lokale Überlaufstrafe, elektrostatisch (letzte und billigste Stufe). Darunter, über feste Instanzen, der Median der Lücke.")
st.plotly_chart(build_model_bars(v, rule), width="stretch", key="model_bars")
verdict(verdict_models(settings, v))
st.markdown(f"**Vergleich über feste Instanzen** (n = {C.SWEEP_N}, je Familie {len(C.SWEEP_SEEDS)} Instanzen, 2, 4 und 8 Pads, Optimum aufgezählt; Randzone {tx(C.DEFAULT_MARGIN)}, Ladungsbreite {tx(C.DEFAULT_WIDTH)}).")
if not st.session_state.get("sweep_ok"):
    st.info("Der Vergleich rechnet 6 Reihen zu je 20 Instanzen, drei Pad-Zahlen, drei Modelle mit zwölf Stufen und die Schleife aus Stück 10 und dauert einige Minuten (auf Streamlit Cloud länger). Mit dem Knopf starten:")
    if st.button("⏱️ Vergleich rechnen (n = 8)", key="sweep_button"):
        bar = st.progress(0.0, text="Rechne die Familien …")
        for i, f in enumerate(C.FAMILIES):
            _sweep(f, C.SWEEP_N, C.SWEEP_SEEDS, C.PAD_CHOICES)
            bar.progress((i + 1) / len(C.FAMILIES), text=f"{C.FAMILY_NAMES[f]} fertig")
        st.session_state["sweep_ok"] = True
        st.rerun()
else:
    sweep_tables(pooled({f: _sweep(f, C.SWEEP_N, C.SWEEP_SEEDS, C.PAD_CHOICES) for f in C.FAMILIES}), C.SWEEP_N)

st.markdown("---")

# --- Experiment 4 ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: Was die Fassung braucht")
st.caption("Elektrostatisch, n = 8, 4 Pads, über feste Instanzen: die Lücke der letzten Stufe für jede Randzone (Ladungsbreite 3) und für jede Ladungsbreite (Randzone 0,5; 1 heißt ohne Weichzeichnen). Gestrichelt einmal runden aus Stück 9.")
if not st.session_state.get("ablation_ok"):
    st.info("Die Rechnung läuft über 120 Instanzen mit je sieben Läufen und dauert einige Minuten (auf Streamlit Cloud länger). Mit dem Knopf starten:")
    if st.button("⏱️ Randzone und Ladungsbreite rechnen", key="ablation_button"):
        bar = st.progress(0.0, text="Rechne die Familien …")
        for i, f in enumerate(C.FAMILIES):
            _ablation(f, C.SWEEP_N, C.SWEEP_SEEDS)
            bar.progress((i + 1) / len(C.FAMILIES), text=f"{C.FAMILY_NAMES[f]} fertig")
        st.session_state["ablation_ok"] = True
        st.rerun()
else:
    arows = [r for f in C.FAMILIES for r in _ablation(f, C.SWEEP_N, C.SWEEP_SEEDS)]
    st.plotly_chart(build_ablation_bars(arows, ablation_summary, C.MARGIN_CHOICES, C.WIDTH_CHOICES, f"Randzone und Ladungsbreite, n = {C.SWEEP_N}, {len(arows)} Instanzen"), width="stretch", key="ablation_chart")
    verdict(verdict_ablation(arows))

st.markdown("---")

# --- Experiment 5: große Instanzen --------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 5: Jenseits der Aufzählung")
st.caption(f"Für n = 12 und 16: dieselben Verfahren gegen die beste von {C.STARTS} steilsten Abstiegen (obere Grenze der Lücke), je Familie 5 bzw. 3 feste Instanzen, 2, 4 und 8 Pads. Die Einzelergebnisse sind hier rechnerabhängig (kleines Rauschen ändert eine Anordnung); belastbar sind nur die Mediane.")
if not st.session_state.get("large_ok"):
    st.info("Die Rechnung dauert einige Minuten (auf Streamlit Cloud länger). Mit dem Knopf starten:")
    if st.button("⏱️ Große Instanzen rechnen (n = 12, 16)", key="large_button"):
        bar = st.progress(0.0, text="Rechne die großen Instanzen …")
        todo = [(f, m) for m in C.LARGE_SWEEP_SIZES for f in C.FAMILIES]
        for i, (f, m) in enumerate(todo):
            _sweep(f, m, C.LARGE_SEEDS[m], C.PAD_CHOICES)
            bar.progress((i + 1) / len(todo), text=f"n = {m}, {C.FAMILY_NAMES[f]} fertig")
        st.session_state["large_ok"] = True
        st.rerun()
else:
    for m in C.LARGE_SWEEP_SIZES:
        sweep_tables(pooled({f: _sweep(f, m, C.LARGE_SEEDS[m], C.PAD_CHOICES) for f in C.FAMILIES}), m)

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Eigene, einfache Fassung** | Glatte Länge, Bin-Raster, Poisson-Gleichung per Kosinus-Matrix, Strafverfahren mit Rückverfolgung, Randzone und Weichzeichnen sind eigene Entscheidungen; ePlace, RePlAce und DREAMPlace (Nesterov mit Schrittschätzung, Füllzellen, gewichtete Durchschnittslänge, Makros, FFT) sind nicht gebaut und nicht nachgelesen. Die Parameter (zwölf Stufen, Faktor 1,8, Randzone 0,5, Ladungsbreite 3) wurden an wenigen Instanzen vorgewählt. | Stück 10 (Platzierung), Literatur zu ePlace |
| **Vergleich Fassung gegen Fassung** | Stück 10 läuft mit seinen Standardwerten (15 Runden, Ankergewicht 0,2), die Elektrostatik mit den hier gewählten; bei anderer Abstimmung kann sich die Rangfolge verschieben. | Stück 10 |
| **Volle Belegung** | n Bauteile auf n Plätzen ist die härteste Dichteaufgabe; in der Praxis ist ein Teil der Fläche frei und mit Füllzellen belegt. Die Randzone ist unsere Antwort darauf, nicht die der Literatur. | – |
| **Die billigste Stufe rundet zwölfmal** | Sie hat mehr Versuche als die letzte Stufe und die Schleife aus Stück 10; deshalb ist die letzte Stufe die Hauptzahl. | – |
| **Aufwand in Auswertungen** | Auswertungen der Dichtestrafe (Strafverfahren) und lineare Systeme (Stück 10) sind verschiedene Einheiten; Zeiten stehen nicht da. | – |
| **Rechenabhängigkeit** | Ein Strafverfahren mit Rückverfolgung und einigen hundert Gradienten ist empfindlich: bei n ≥ 12 kann ein Rundungsrauschen die Anordnung einer Instanz ändern; die Mediane über viele Instanzen sind stabil. | – |
| **Die Pads sind erfunden** | Zahl, Ort, Gewicht und Verbindungen der Pads sind eigene Erfindungen (aus Stück 9). | – |
| **Die sechs Familien sind erfunden** | Echte Platinen haben andere Strukturen; Aussagen über Familien gelten nur für diese. | – |
| **Vergleichswert ab n = 12** | Die beste von 100 Abstiegen ist eine obere Grenze des Optimums und selbst ein Abstiegsergebnis; „0 %“ heißt „so gut wie der Vergleichswert“, nicht „optimal“. | Stück 8 (Beweis bis n ≈ 12) |
| **Wenige Instanzen** | n = 8: 20 Instanzen je Familie, n = 12: 5, n = 16: 3; kleine Unterschiede sind nicht belastbar. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Die Aufgabe (wie in Stück 9).** Flüsse $F$, Plätze $s_1, \dots, s_n$, Pads $a_j$ mit Gewichten $w_{ij}$; Kosten einer Anordnung $\pi$: $\mathrm{cost}(\pi) = \sum_{i<k} F_{ik} \lVert s_{\pi_i} - s_{\pi_k} \rVert_1 + \sum_{i,j} w_{ij} \lVert s_{\pi_i} - a_j \rVert_1$.

**Glatte Länge.** Für stetige Orte $x_i \in \mathbb{R}^2$ mit $\gamma = 0{,}3$: $W(x) = \sum_{i<k} F_{ik} \sum_{d=1}^{2} \sqrt{(x_{id} - x_{kd})^2 + \gamma^2} + \sum_{i,j} w_{ij} \sum_{d=1}^{2} \sqrt{(x_{id} - a_{jd})^2 + \gamma^2}$.
Ihr Gradient in $x_{id}$ ist $\sum_k F_{ik} \frac{x_{id} - x_{kd}}{\sqrt{(x_{id} - x_{kd})^2 + \gamma^2}} + \dots$: die Umgewichtung $F / \max(|\Delta|, \varepsilon)$ aus Stück 9 mit $\varepsilon \approx \gamma$ (dort als lineares System, hier als Gradient).

**Dichte.** Ein Bin-Raster der Seitenlänge $h = 1/2$ über dem Gebiet (Block der Plätze plus Randzone). Bauteil $i$ ist eine Ladung der Masse 1, gleichmäßig auf ein Quadrat der Kantenlänge $\sigma$ um $x_i$ verteilt; mit den Überdeckungen $o^x_{ib}$, $o^y_{ib}$ des Intervalls mit Bin-Spalte und Bin-Zeile ist die Dichte
$$\rho_b = \frac{1}{h^2 \sigma^2} \sum_i o^x_{i b_x}\, o^y_{i b_y}, \qquad \rho^{\mathrm{soll}} = \frac{n}{\text{Fläche}}.$$
Die Ladungsbreite $\sigma$ sinkt in den Stufen $s = 0, \dots, 11$ linear von $\sigma_0$ auf 1.

**Elektrostatisch.** $\tilde\rho = \rho - \bar\rho$, $-\Delta \psi = \tilde\rho$ mit Neumann-Rand (Differenzen), gelöst mit der Kosinus-Transformation $T$ (als Matrix): $\psi = T^{\top} \big( (T \tilde\rho\, T^{\top}) / (\lambda_y + \lambda_x) \big) T$ mit den Eigenwerten $\lambda_m = (2 - 2\cos(\pi m / M)) / h^2$ und dem nullten Modus gleich 0.
Energie $N(x) = \tfrac{1}{2} h^2 \sum_b \tilde\rho_b \psi_b$; der Gradient ist $\partial N / \partial x_{id} = \sum_b \psi_b\, \partial o_{ib} / \partial x_{id} \cdot (\text{die andere Überdeckung}) / \sigma^2$ und stimmt mit Differenzenquotienten überein (Test).
**Lokal.** $N(x) = h^2 \sum_b \max(\rho_b - \rho^{\mathrm{soll}}, 0)^2$.

**Strafverfahren.** $f_s(x) = W(x) + \lambda_s N_s(x)$ mit $\lambda_0 = \lVert \nabla W \rVert_1 / \lVert \nabla N \rVert_1$ am Startwert und $\lambda_{s+1} = 1{,}8\, \lambda_s$. Je Stufe 25 Schritte $x \leftarrow \mathrm{clip}(x - t \nabla f_s)$ mit Armijo-Rückverfolgung ($t$ halbieren, bis $f_s$ um $10^{-4} t \lVert \nabla f_s \rVert^2$ abnimmt, sonst $t$ verdoppeln);
jede Auswertung von $f_s$ samt Gradient zählt. Die Orte bleiben im Gebiet. **Rundung** nach jeder Stufe: lineare Zuordnung der Orte auf die Plätze mit kleinster Summe der Abstandsquadrate (Gleichstände auf acht Stellen gerundet).

**Überlauf und Verschiebung.** $\text{Überlauf} = \sum_b \max(\rho_b - \rho^{\mathrm{soll}}, 0)\, h^2 / n$ mit $\sigma = 1$; Verschiebung $= \sum_i \lVert x_i - s_{\pi_i} \rVert_1$.

**Lücke.** $(\mathrm{cost} - U)/U$ mit dem Vergleichswert $U$ (aufgezähltes Optimum bis $n = 9$, sonst die beste von 100 steilsten Abstiegen: eine **obere** Grenze des Optimums).

**Literatur.** ePlace (Lu u. a.), RePlAce, DREAMPlace und die Poisson-Dichte der analytischen Platzierung: Angaben aus der Fachliteratur übernommen, hier **nicht nachgelesen**; Selbstangaben der Autoren zur Güte werden nicht übernommen. Die glatte Länge, die Randzone, das Weichzeichnen, das Strafverfahren und der Vergleich mit Stück 10 sind **eigene** Herleitungen und werden durch Tests geprüft
(Handrechnung, Differenzenquotient, ausgeschriebene Laplace-Matrix, Doppelschleife für die Dichte, Aufzählung). Die Zahlen dieser Demo sind eigene Rechnungen an eigenen, erfundenen Familien und Pads.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
