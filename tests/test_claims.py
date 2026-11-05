"""Die Zahlen der README (Abschnitte „Befunde“ und „Grenzen“) als Tests: Standardfall, Handbeispiel und die Vergleiche über feste Instanzen.
Prozentwerte auf eine Stelle: n = 8 mit Toleranz 0,06, n = 12 und 16 mit 0,3 (rechnerabhängig, siehe README); Zählungen bei n = 8 exakt, bei n ≥ 12 mit ±3; Zeiten stehen nie in Tests (DEMO-PLAYBOOK §1b, Wiederholbarkeit)."""

import pytest

import dla_constants as C
import dla_evaluation as ev


def pc(x):
    return 100 * x


TOL = {8: 0.06, 12: 0.3, 16: 0.3}
CNT = {8: 0, 12: 3, 16: 3}


@pytest.fixture(scope="module")
def sweeps():
    return {8: ev.pooled({f: ev.sweep(f, 8, C.SWEEP_SEEDS, C.PAD_CHOICES) for f in C.FAMILIES}), 12: ev.pooled({f: ev.sweep(f, 12, C.LARGE_SEEDS[12], C.PAD_CHOICES) for f in C.FAMILIES}),
            16: ev.pooled({f: ev.sweep(f, 16, C.LARGE_SEEDS[16], C.PAD_CHOICES) for f in C.FAMILIES})}


@pytest.fixture(scope="module")
def ablation():
    return [r for f in C.FAMILIES for r in ev.ablation_sweep(f, 8, C.SWEEP_SEEDS)]


def test_standard_instance_numbers():
    s = ev.Settings()
    v = ev.instance_data(s)
    assert (s.family, s.n, s.seed, s.pads, s.model, s.margin, s.width, s.rule) == ("cluster", 8, 48, 4, "electro", 0.5, 3.0, "last") and v["exact"] and v["ref"] == 300.0 and pc(v["gap_mean"]) == pytest.approx(28.7, abs=0.06)
    assert (v["cost9"], v["cost10"]) == (369.0, 319.0)
    e, o, nn = (v["runs"][m] for m in ("electro", "overflow", "none"))
    assert [int(c) for c in e.costs] == [301, 300, 300, 300, 300, 300, 300, 300, 301, 301, 305, 305] and e.stages[-1].cost == 305.0 and e.best_index == 1 and min(e.costs) == 300.0
    assert (o.stages[-1].cost, min(o.costs), o.best_index) == (309.0, 300.0, 1) and (nn.stages[-1].cost, min(nn.costs), nn.best_index) == (359.0, 346.0, 0)
    assert (e.stages[0].evals, e.evals, o.evals, nn.evals) == (76, 906, 904, 901) and e.lam0 == pytest.approx(120.0, abs=0.1)
    assert (round(e.stages[0].overflow, 2), round(e.stages[-1].overflow, 2), round(o.stages[-1].overflow, 2), round(nn.stages[-1].overflow, 2)) == (0.74, 0.20, 0.59, 0.91)
    assert (round(e.stages[0].disp, 1), round(e.stages[-1].disp, 1), round(e.stages[-1].spread, 2)) == (7.7, 4.5, 1.30)
    f = ev.start_field(s, v)
    assert round(f["rho_max"], 2) == 1.67 and round(f["overflow"], 2) == 0.83


def test_standard_instance_without_margin_and_without_softening():
    a = ev.instance_data(ev.Settings(margin=0.0))["runs"]["electro"]
    assert (a.stages[-1].cost, min(a.costs), a.best_index) == (352.0, 300.0, 0) and round(a.stages[-1].overflow, 2) == 0.13
    b = ev.instance_data(ev.Settings(width=1.0))["runs"]["electro"]
    assert b.stages[-1].cost == 300.0 and round(b.stages[-1].overflow, 2) == 0.72


def test_hand_example():
    h = ev.hand_data()
    assert h["rho"][0].tolist() == pytest.approx([0.2, 1.8, 1.8, 0.2]) and h["value"] == pytest.approx(0.08) and h["grad"][0, 0] == pytest.approx(0.2)


GAPS = {  # n -> {Schlüssel: Median der Lücke in Prozent (alle Pad-Zahlen)}
    8: {"s9": 8.1, "s10": 6.2, "none": 7.7, "overflow": 6.8, "electro": 3.9, "electro_best": 1.6},
    12: {"s9": 12.5, "s10": 9.1, "none": 12.0, "overflow": 10.1, "electro": 8.6, "electro_best": 3.1},
    16: {"s9": 13.6, "s10": 10.4, "none": 14.1, "overflow": 8.5, "electro": 8.0, "electro_best": 4.0},
}
PER_PAD = {  # n -> {Schlüssel: (2, 4, 8 Pads)}
    8: {"s9": (14.5, 8.8, 4.4), "s10": (10.8, 6.3, 3.6), "none": (12.4, 9.4, 4.3), "overflow": (10.8, 7.1, 4.8), "electro": (6.3, 4.0, 2.2), "electro_best": (2.4, 1.8, 0.7)},
    12: {"s10": (10.6, 10.6, 5.9), "electro": (8.9, 9.1, 5.3), "overflow": (14.2, 9.9, 6.0), "electro_best": (4.3, 5.4, 1.7)},
    16: {"s10": (14.5, 11.1, 7.3), "electro": (8.3, 10.6, 6.7), "overflow": (10.4, 10.5, 5.4), "electro_best": (3.0, 5.6, 1.7)},
}


@pytest.mark.parametrize("n", [8, 12, 16])
def test_median_gaps_over_all_instances_and_per_pad_count(sweeps, n):
    rows = sweeps[n]
    assert len(rows) == {8: 360, 12: 90, 16: 54}[n] and rows[0]["exact"] == (n == 8)
    for k, v in GAPS[n].items():
        assert pc(ev.gap_summary(rows, None, k)["gap"]) == pytest.approx(v, abs=TOL[n]), (n, k)
    for k, vs in PER_PAD[n].items():
        for p, v in zip(C.PAD_CHOICES, vs):
            assert pc(ev.gap_summary(rows, p, k)["gap"]) == pytest.approx(v, abs=TOL[n]), (n, k, p)


HITS = {8: {"s9": 23, "s10": 34, "none": 18, "overflow": 18, "electro": 54, "electro_best": 112}, 12: {"s9": 1, "s10": 1, "electro": 1, "electro_best": 4}, 16: {"electro_best": 1}}
COUNTS = {8: ((212, 108, 40), (275, 32, 53), (237, 94, 29)), 12: ((50, 40, 0), (79, 9, 2), (54, 35, 1)), 16: ((35, 18, 1), (47, 6, 1), (27, 25, 2))}   # electro gegen Stück 10, billigste Stufe gegen Stück 10, electro gegen lokal


@pytest.mark.parametrize("n", [8, 12, 16])
def test_hits_and_instance_by_instance_counts(sweeps, n):
    rows = sweeps[n]
    for k, h in HITS[n].items():
        assert abs(ev.gap_summary(rows, None, k)["hit"] - h) <= CNT[n], (n, k)
    for got, want in zip((ev.compare_counts(rows, "electro", "s10"), ev.compare_counts(rows, "electro_best", "s10"), ev.compare_counts(rows, "electro", "overflow")), COUNTS[n]):
        assert all(abs(a - b) <= CNT[n] for a, b in zip(got, want)) and sum(got) == len(rows), (n, got, want)


MODEL = {  # n -> {Modell: (Überlauf, Verschiebung, Streuung, Auswertungen)}
    8: {"electro": (0.20, 4.0, 1.32, 908), "overflow": (0.22, 5.7, 1.28, 896), "none": (0.83, 11.3, 0.09, 901)},
    12: {"electro": (0.18, 7.6, 1.27, 907), "overflow": (0.19, 8.7, 1.24, 898), "none": (0.89, 19.6, 0.04, 901)},
    16: {"electro": (0.16, 10.6, 1.24, 907), "overflow": (0.18, 11.4, 1.20, 898), "none": (0.92, 31.5, 0.02, 901)},
}


@pytest.mark.parametrize("n", [8, 12, 16])
def test_overflow_displacement_spread_and_effort_at_the_end(sweeps, n):
    for m, (ov, disp, spread, evals) in MODEL[n].items():
        ms = ev.model_summary(sweeps[n], None, m)
        assert ms["ov"] == pytest.approx(ov, abs=0.02) and ms["disp"] == pytest.approx(disp, abs=0.3 if n == 8 else 1.0) and ms["spread"] == pytest.approx(spread, abs=0.03) and ms["evals"] == pytest.approx(evals, abs=12), (n, m)


STAGES_8 = {"gap": [5.7, 4.5, 3.6, 3.4, 3.2, 3.1, 3.4, 3.8, 3.8, 3.9, 3.8, 3.9], "ov": [0.74, 0.70, 0.70, 0.71, 0.70, 0.47, 0.45, 0.37, 0.33, 0.32, 0.28, 0.20], "disp": [8.9, 7.5, 7.2, 7.1, 7.2, 6.5, 6.9, 6.5, 5.8, 5.5, 5.0, 4.0]}


def test_the_stage_medians_at_n8_have_their_minimum_in_the_middle(sweeps):
    sm = ev.stage_medians(sweeps[8], "electro")
    assert [pc(x) for x in sm["gap"]] == pytest.approx(STAGES_8["gap"], abs=0.1) and sm["ov"] == pytest.approx(STAGES_8["ov"], abs=0.03) and sm["disp"] == pytest.approx(STAGES_8["disp"], abs=0.3)
    assert min(sm["gap"]) == sm["gap"][5] and sm["gap"][5] < sm["gap"][-1] and sm["ov"][-1] == min(sm["ov"]) and sm["disp"][-1] == min(sm["disp"])


FAMILY_8 = {  # Familie -> (Stück 9, Stück 10, keine, lokal, elektrostatisch, billigste Stufe)
    "netlist": (9.0, 7.4, 7.7, 5.9, 3.9, 1.9), "uniform": (5.4, 5.1, 5.3, 4.4, 2.8, 1.4), "sparse": (8.7, 7.3, 8.5, 8.9, 2.7, 0.9),
    "cluster": (6.2, 2.2, 6.0, 5.2, 1.5, 0.0), "star": (8.5, 7.7, 8.8, 7.8, 7.9, 3.8), "skewed": (13.6, 9.3, 14.5, 9.9, 7.1, 3.1),
}


def test_the_families_at_n8(sweeps):
    for f, vals in FAMILY_8.items():
        rs = [r for r in sweeps[8] if r["family"] == f]
        assert len(rs) == 60
        for k, v in zip(("s9", "s10", "none", "overflow", "electro", "electro_best"), vals):
            assert pc(ev.gap_summary(rs, None, k)["gap"]) == pytest.approx(v, abs=0.06), (f, k)


ABL = {"m0.0": (9.6, 0.13), "m0.25": (6.0, 0.17), "m0.5": (4.0, 0.20), "m1.0": (4.6, 0.36), "w1.0": (6.6, 0.53), "w2.0": (4.2, 0.20), "w3.0": (4.0, 0.20), "w4.0": (6.7, 0.21)}


def test_ablation_randzone_and_charge_width(ablation):
    assert len(ablation) == 120 and pc(float(__import__("numpy").median([r["gap9"] for r in ablation]))) == pytest.approx(8.8, abs=0.06)
    for k, (gap, ov) in ABL.items():
        sm = ev.ablation_summary(ablation, k)
        assert pc(sm["gap"]) == pytest.approx(gap, abs=0.06) and sm["ov"] == pytest.approx(ov, abs=0.02), k


FAMILY_12 = {  # Familie -> (Stück 10, elektrostatisch), n = 12, je 15 Instanzen (nur Mediane; rechnerabhängig)
    "netlist": (10.3, 11.5), "uniform": (6.9, 8.0), "sparse": (12.1, 7.5), "cluster": (3.0, 7.2), "star": (9.9, 8.9), "skewed": (19.5, 11.2),
}


def test_the_families_at_n12_where_the_loop_of_piece_10_is_better_and_where_not(sweeps):
    for f, (s10, el) in FAMILY_12.items():
        rs = [r for r in sweeps[12] if r["family"] == f]
        assert len(rs) == 15
        assert pc(ev.gap_summary(rs, None, "s10")["gap"]) == pytest.approx(s10, abs=TOL[12]) and pc(ev.gap_summary(rs, None, "electro")["gap"]) == pytest.approx(el, abs=TOL[12]), f
    better_loop = [f for f, (s10, el) in FAMILY_12.items() if s10 < el]
    assert better_loop == ["netlist", "uniform", "cluster"]
