"""Pfade der App, die der Rauchtest nicht erreicht: die Experimente im Seiteninhalt, PDF, jede Familie, jede Einstellung, große Instanzen, die Rechenknöpfe (DEMO-PLAYBOOK §1d)."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

import dla_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")
KIND = {"family_select": "selectbox", "n_select": "select_slider", "seed_input": "number_input", "pads_select": "select_slider", "model_select": "selectbox", "margin_select": "select_slider", "width_select": "select_slider", "rule_select": "selectbox"}


def _app(**state) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    for key, value in state.items():
        getattr(at.sidebar, KIND[key])(key=key).set_value(value)
    if state:
        at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texts(at):
    return " ".join(str(e.value) for e in list(at.success) + list(at.info) + list(at.warning) + list(at.error) + list(at.markdown))


def test_default_page_shows_the_metrics_and_the_verdicts():
    at = _app()
    values = {m.label: m.value for m in at.metric}
    assert set(values) == {"Lücke nach der Rundung", "Überlauf", "Auswertungen", "Optimum", "Pads"}
    assert values["Optimum"] == "300" and values["Pads"] == "4" and values["Lücke nach der Rundung"].endswith(" %") and int(values["Auswertungen"].replace(" ", "")) > 500
    text = _texts(at)
    for part in ("Familie **Zwei Cluster**, 8 Bauteile, Raster 2×4, Seed 48, 4 Pads", "Rundungsregel letzte stufe (Stufe 12 von 12)", "am Startwert (Platzierung aus Stück 9, leicht gestreut)", "fünf Experimente"):
        assert part in text, part


def test_the_experiments_the_hand_example_the_limits_table_and_the_formulation_are_on_the_page():
    at = _app()
    headings = [s.value for s in at.subheader]
    assert [h.split(":")[0] for h in headings if h.startswith("🔬 Experiment")] == [f"🔬 Experiment {i}" for i in (1, 2, 3, 4, 5)] and "🚧 Wo die Annahmen enden" in headings
    assert [e.label for e in at.expander] == ["So funktioniert es", "Handbeispiel zum Nachrechnen", "📐 Mathematische Formulierung"]
    assert not at.error and len(at.get("plotly_chart")) == 6
    assert any("0,2, 1,8, 1,8, 0,2" in str(m.value) and "**0,4**" in str(m.value) for m in at.markdown)


def test_every_family_every_setting_and_size_renders():
    for fam in C.FAMILIES:
        at = _app(family_select=fam, n_select=6)
        assert f"Familie **{C.FAMILY_NAMES[fam]}**" in _texts(at) and len(at.get("plotly_chart")) == 6
    for key, values in (("pads_select", C.PAD_CHOICES), ("model_select", C.MODELS), ("margin_select", C.MARGIN_CHOICES), ("width_select", C.WIDTH_CHOICES), ("rule_select", C.RULES)):
        for value in values:
            at = _app(n_select=6, **{key: value})
            assert len(at.get("plotly_chart")) == 6 and not at.exception, (key, value)
    for n in (9, 12, 16):
        at = _app(n_select=n)
        assert f"{n} Bauteile" in _texts(at) and len(at.get("plotly_chart")) == 6


def test_the_model_and_the_rule_change_the_result():
    base = {m.label: m.value for m in _app().metric}
    none = {m.label: m.value for m in _app(model_select="none").metric}
    assert base["Überlauf"] != none["Überlauf"]
    best = {m.label: m.value for m in _app(rule_select="best").metric}
    assert best["Auswertungen"] != base["Auswertungen"] or best["Lücke nach der Rundung"] != base["Lücke nach der Rundung"]
    assert "Rundungsregel billigste stufe" in _texts(_app(rule_select="best"))


def test_the_pdf_button_makes_a_download():
    at = _app(n_select=6)
    at.button(key="pdf_make").click().run()
    assert not at.exception
    assert any(d.proto.label == "⬇️ PDF herunterladen" for d in at.get("download_button"))


def test_the_comparison_buttons_fill_their_sections(monkeypatch):
    monkeypatch.setattr(C, "SWEEP_SEEDS", C.SWEEP_SEEDS[:2])                       # nur für die Geschwindigkeit des Tests: die echten Reihen stehen in test_claims.py
    monkeypatch.setattr(C, "LARGE_SEEDS", {12: C.SWEEP_SEEDS[:1], 16: C.SWEEP_SEEDS[:1]})
    at = _app(n_select=6)
    assert [b.label for b in at.button if "⏱️" in b.label] == ["⏱️ Vergleich rechnen (n = 8)", "⏱️ Randzone und Ladungsbreite rechnen", "⏱️ Große Instanzen rechnen (n = 12, 16)"]
    at.button(key="sweep_button").click().run()
    text = _texts(at)
    assert not at.exception and "Bei n = 8, Median der Lücke über 36 Instanzen (alle Pad-Zahlen)" in text and len(at.get("plotly_chart")) == 6 + 2
    at.button(key="ablation_button").click().run()
    text = _texts(at)
    assert not at.exception and "Elektrostatisch, n = 8, 4 Pads, Median über 12 Instanzen" in text and len(at.get("plotly_chart")) == 6 + 2 + 1
    at.button(key="large_button").click().run()
    text = _texts(at)
    assert not at.exception and "Bei n = 12, Median der Lücke über 18 Instanzen" in text and "Bei n = 16, Median der Lücke über 18 Instanzen" in text and "obere Grenze der Lücke" in text and len(at.get("plotly_chart")) == 6 + 2 + 1 + 4
