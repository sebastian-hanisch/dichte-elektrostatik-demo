"""Verdikte: Art (success/info) und Inhalt an handgesetzten Zeilen und an einer kleinen Instanz; jeder Text nennt, was er misst."""

import numpy as np
import pytest

import dla_constants as C
import dla_evaluation as ev
from test_evaluation import COSTS, fake_row


@pytest.fixture(scope="module")
def inst_default():
    s = ev.Settings(seed=48)
    return s, ev.instance_data(s)


def test_core_verdict_is_success_when_the_reference_is_met_and_names_the_comparison(inst_default):
    s, v = inst_default
    kind, text = ev.verdict_core(ev.Settings(seed=48, rule="best"), v)
    assert kind == "success" and "Rundungsregel billigste stufe" in text and "einmal runden (Stück 9)" in text and "Schleife aus Stück 10" in text and "Eine Instanz, kein Befund über die Familie" in text and "aufgezählte Optimum" in text
    kind, text = ev.verdict_core(s, v)
    assert kind == "info" and "Stufe 12 von 12" in text and "Elektrostatisch" in text


def test_field_verdict_distinguishes_no_penalty(inst_default):
    s, v = inst_default
    assert "Pfeile" in ev.verdict_field(s, v)[1] and "Kraft" in ev.verdict_field(s, v)[1]
    none = ev.verdict_field(ev.Settings(seed=48, model="none"), v)[1]
    assert "Ohne Dichtestrafe wirkt keine Kraft" in none and "Pfeile" not in none


def test_path_verdict_reports_the_best_stage(inst_default):
    s, v = inst_default
    text = ev.verdict_path(s, v)[1]
    pen = v["runs"]["electro"]
    assert "Stufe 1" in text and ("nicht die billigste" in text) == (pen.stages[pen.best_index].cost < pen.stages[-1].cost - 1e-9) and "Eine Instanz" in text


def test_models_verdict_names_all_models_and_both_baselines(inst_default):
    s, v = inst_default
    text = ev.verdict_models(s, v)[1]
    assert all(C.MODEL_NAMES[m] in text for m in C.MODELS) and "Einmal runden (Stück 9)" in text and "Schleife (Stück 10)" in text and "Vergleichswert 300" in text


def test_sweep_verdict_over_handset_rows_exact_and_large():
    rows = [fake_row(2, 100.0, COSTS), fake_row(4, 100.0, {**COSTS, "electro": 112.0}), fake_row(8, 100.0, {**COSTS, "electro": 100.0})]
    kind, text = ev.verdict_sweep(8, rows)
    assert kind == "info" and "Bei n = 8, Median der Lücke über 3 Instanzen" in text and "elektrostatisch: 4,0 % (erreicht in 1)" in text and "Stück 10 (Schleife): 10,0 % (erreicht in 0)" in text
    assert "besser in 2, schlechter in 1, gleich in 0" in text and "Überlauf am Ende (Median): elektrostatisch 0,20" in text and "Auswertungen je Instanz (elektrostatisch) 900" in text and "obere Grenze" not in text
    big = [dict(r, exact=False) for r in rows]
    assert "obere Grenze der Lücke" in ev.verdict_sweep(12, big)[1]


def test_ablation_verdict_lists_every_variant():
    rows = []
    for s in (1, 2, 3):
        rows.append({"family": "cluster", "seed": s, "p": 4, "ref": 100.0, "gap9": 0.10, "gap": {k: 0.02 * (i + 1) for i, k in enumerate(ev.ABLATION)}, "ov": {k: 0.1 * (i + 1) for i, k in enumerate(ev.ABLATION)}})
    kind, text = ev.verdict_ablation(rows)
    assert kind == "info" and "Median über 3 Instanzen" in text and "einmal runden (Stück 9): 10,0 %" in text
    assert all(f"Randzone {ev.tx(m)}:" in text for m in C.MARGIN_CHOICES) and all(f"Ladungsbreite {ev.tx(w)}:" in text for w in C.WIDTH_CHOICES) and "Randzone 0: 2,0 %, Überlauf 0,10" in text
