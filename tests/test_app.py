"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: laedt app.py mit den Standardeinstellungen, klickt jeden
Preset-Button und faehrt jeden Regler an seine Grenzen und prueft, dass kein Python-Fehler auftritt - insbesondere
`streamlit.errors.StreamlitDuplicateElementId` (mehrere st.plotly_chart-Aufrufe ohne eindeutiges key=) und `StreamlitAPIException`
bei Reglern mit berechneten Grenzen (min == max)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "▶"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen
HEAVY_MARKER = "⏱️"    # Rechenknöpfe (n = 9, Vergleiche): Wirkung in test_app_experiments.py


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label or HEAVY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_choice_control_at_its_first_and_last_value_does_not_raise():
    """Auswahlregler an ihrem ersten und letzten Wert (n = 9 zeigt nur den Rechenknopf), Seed an den Grenzen."""
    import dla_constants as C

    choices = {"family_select": ("selectbox", C.FAMILIES), "n_select": ("select_slider", C.SIZES)}
    for key, (kind, values) in choices.items():
        for value in (values[0], values[-1]):
            at = _fresh_app()
            getattr(at.sidebar, kind)(key=key).set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (key, value, [str(e) for e in at.exception])
    for value in (0, C.SEED_MAX):
        at = _fresh_app()
        at.sidebar.number_input(key="seed_input").set_value(value).run(timeout=TIMEOUT)
        assert not at.exception, (value, [str(e) for e in at.exception])
