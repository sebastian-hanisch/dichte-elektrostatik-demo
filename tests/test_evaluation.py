"""Kennzahlen: Formatierung, eine Instanz, Zeilen und Zusammenfassungen der Vergleiche, Ablation, Handbeispiel."""

import numpy as np
import pytest

import dla_constants as C
import dla_evaluation as ev


def fake_row(p, ref, costs, ov=0.2, disp=4.0, spread=1.3, evals=900, family="cluster"):
    """Eine Zeile mit handgesetzten Kosten (Schlüssel wie KEYS)."""
    row = {"family": family, "n": 8, "seed": 1, "p": p, "ref": ref, "exact": True, "gap_mean": 0.3, "cost": dict(costs), "gap": {k: ev.rel(v, ref) for k, v in costs.items()}, "hit": {k: v <= ref + 1e-9 for k, v in costs.items()},
           "ov": {m: ov for m in C.MODELS}, "disp": {m: disp for m in C.MODELS}, "spread": {m: spread for m in C.MODELS}, "evals": {m: evals for m in C.MODELS}}
    row["stage_gap"] = {m: [0.05 - 0.002 * k for k in range(C.STAGES)] for m in C.MODELS}
    row["stage_ov"] = {m: [0.8 - 0.05 * k for k in range(C.STAGES)] for m in C.MODELS}
    row["stage_disp"] = {m: [9.0 - 0.4 * k for k in range(C.STAGES)] for m in C.MODELS}
    return row


COSTS = {"s9": 120.0, "s10": 110.0, "none": 115.0, "overflow": 108.0, "electro": 104.0, "none_best": 112.0, "overflow_best": 104.0, "electro_best": 100.0}


def test_formatting_uses_decimal_commas_and_thin_separators():
    assert ev.de(1.234, 2) == "1,23" and ev.de(-1.5, 1) == "−1,5" and ev.pct(0.125, 1) == "12,5 %" and ev.pct(0.0) == "0 %"
    assert ev.num(27.0) == "27" and ev.num(27.5) == "27,5" and ev.count(1200) == "1 200" and ev.tx(0.5) == "0,5" and ev.tx(2.0) == "2" and ev.tx(0.25) == "0,25"
    assert ev.rel(110.0, 100.0) == pytest.approx(0.1) and ev.rel(5.0, 0.0) != ev.rel(5.0, 0.0) and ev.rel(100.0 + 1e-13, 100.0) == 0.0
    assert ev.stufe(0) == "Stufe 1" and ev.fam("star") == "Stern" and ev.median_of([{"a": 1}, {"a": 3}, {"a": 10}], "a") == 3.0 and ev.median_of([{"a": 1}, {"a": 5}], lambda r: 2 * r["a"]) == 6.0


def test_settings_default_to_the_constants():
    s = ev.Settings()
    assert (s.family, s.n, s.seed, s.pads, s.model, s.margin, s.width, s.rule) == (C.DEFAULT_FAMILY, C.DEFAULT_N, C.DEFAULT_SEED, C.DEFAULT_PADS, C.DEFAULT_MODEL, C.DEFAULT_MARGIN, C.DEFAULT_WIDTH, C.DEFAULT_RULE)


def test_instance_data_has_every_part_and_nothing_beats_the_reference():
    v = ev.instance_data(ev.Settings(n=6, seed=35))
    assert set(v["runs"]) == set(C.MODELS) and v["exact"] and v["mean"] is not None and v["gap_mean"] > 0
    assert v["cost9"] >= v["ref"] - 1e-9 and v["cost10"] >= v["ref"] - 1e-9 and v["cost10"] == v["loop"].rounds[-1].upper
    assert all(c >= v["ref"] - 1e-9 for pen in v["runs"].values() for c in pen.costs)


def test_main_result_follows_the_rule():
    v = ev.instance_data(ev.Settings(seed=48))
    last, k = ev.main_result(ev.Settings(seed=48), v)
    assert k == C.STAGES - 1 and last is v["runs"]["electro"].stages[-1]
    best, kb = ev.main_result(ev.Settings(seed=48, rule="best"), v)
    assert best.cost == min(v["runs"]["electro"].costs) and kb == v["runs"]["electro"].best_index
    other, _ = ev.main_result(ev.Settings(seed=48, model="none"), v)
    assert other is v["runs"]["none"].stages[-1]


def test_start_field_force_is_minus_the_gradient_and_the_density_is_consistent():
    s = ev.Settings(seed=48)
    v = ev.instance_data(s)
    f = ev.start_field(s, v)
    _, g, _ = f["bins"].energy(f["X"], "electro")
    assert f["force"] == pytest.approx(-g) and f["rho"].shape == (f["bins"].ny, f["bins"].nx) and f["rho_max"] == pytest.approx(f["rho"].max() / f["bins"].target) and f["overflow"] > 0.5
    none = ev.start_field(ev.Settings(seed=48, model="none"), v)
    assert not none["force"].any()


def test_row_of_is_consistent():
    r = ev.row_of("cluster", 6, 100001, 4)
    assert r["exact"] and set(r["cost"]) == set(ev.KEYS) and set(r["gap"]) == set(ev.KEYS)
    for k in ev.KEYS:
        assert r["gap"][k] == pytest.approx((r["cost"][k] - r["ref"]) / r["ref"]) and r["hit"][k] == (r["cost"][k] <= r["ref"] + 1e-9) and r["gap"][k] >= -1e-12
    for m in C.MODELS:
        assert len(r["stage_gap"][m]) == len(r["stage_ov"][m]) == len(r["stage_disp"][m]) == C.STAGES
        assert r["gap"][m + "_best"] == pytest.approx(min(r["stage_gap"][m])) and r["gap"][m] == pytest.approx(r["stage_gap"][m][-1]) and r["gap"][m + "_best"] <= r["gap"][m] + 1e-12
        assert r["evals"][m] > 0 and r["ov"][m] == pytest.approx(r["stage_ov"][m][-1]) and r["disp"][m] == pytest.approx(r["stage_disp"][m][-1])


def test_sweep_pooled_and_select():
    rows = ev.sweep("star", 6, (100000, 100001), (2, 8))
    assert len(rows) == 4 and [r["p"] for r in rows] == [2, 8, 2, 8] and len(ev.select(rows, 8)) == 2 and len(ev.select(rows)) == 4
    pooled = ev.pooled({"star": rows, "cluster": ev.sweep("cluster", 6, (100000,), (2,))})
    assert len(pooled) == 5 and pooled[0]["family"] == "cluster" and pooled[1]["family"] == "star"     # Reihenfolge der Familien aus FAMILIES


def test_summaries_over_handset_rows():
    rows = [fake_row(2, 100.0, COSTS), fake_row(4, 100.0, {**COSTS, "electro": 110.0}), fake_row(4, 100.0, {**COSTS, "electro": 100.0})]
    sm = ev.gap_summary(rows, None, "electro")
    assert sm["count"] == 3 and sm["gap"] == pytest.approx(0.04) and sm["mean"] == pytest.approx((0.04 + 0.10 + 0.0) / 3) and sm["hit"] == 1
    assert ev.gap_summary(rows, 4, "electro")["count"] == 2 and ev.gap_summary(rows, 4, "electro")["gap"] == pytest.approx(0.05)
    ms = ev.model_summary(rows, None, "electro")
    assert ms == {"ov": 0.2, "disp": 4.0, "spread": 1.3, "evals": 900.0}
    sm_stage = ev.stage_medians(rows, "electro")
    assert len(sm_stage["gap"]) == C.STAGES and sm_stage["gap"][0] == pytest.approx(0.05) and sm_stage["ov"][-1] == pytest.approx(0.8 - 0.05 * 11) and sm_stage["disp"][1] == pytest.approx(8.6)
    assert ev.compare_counts(rows, "electro", "s10") == (2, 0, 1) and ev.compare_counts(rows, "s10", "s10") == (0, 0, 3)
    assert ev.mean_gap(rows, 4) == pytest.approx(0.3)


def test_ablation_row_has_every_variant_and_shares_the_default_run():
    r = ev.ablation_row("cluster", 6, 100002, 4)
    assert set(r["gap"]) == set(ev.ABLATION) == set(r["ov"]) and r["ref"] > 0
    assert r["gap"]["m0.5"] == pytest.approx(r["gap"]["w3.0"]) and r["ov"]["m0.5"] == pytest.approx(r["ov"]["w3.0"]) and all(g >= -1e-12 for g in r["gap"].values()) and r["gap9"] >= -1e-12
    rows = ev.ablation_sweep("cluster", 6, (100002, 100003))
    sm = ev.ablation_summary(rows, "m0.0")
    assert len(rows) == 2 and sm["gap"] == pytest.approx(float(np.median([x["gap"]["m0.0"] for x in rows]))) and sm["ov"] == pytest.approx(float(np.median([x["ov"]["m0.0"] for x in rows])))


def test_hand_data_matches_the_hand_calculation():
    h = ev.hand_data()
    assert h["rho"] == pytest.approx(np.array([[0.2, 1.8, 1.8, 0.2]] * 2)) and h["value"] == pytest.approx(0.08) and h["grad"][0, 0] == pytest.approx(0.2) and h["grad"][1, 0] == pytest.approx(-0.2)
