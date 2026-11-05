"""Abbildungen: sie entstehen für jede Eingabe, Achsen sind gesperrt, Beschriftungen und Daten stimmen mit den Werten überein; PDF-Export."""

import numpy as np

import dla_constants as C
import dla_evaluation as ev
import dla_instance as I
import dla_visualization as V
from dla_pdf_export import build_pdf
from test_evaluation import COSTS, fake_row


def _locked(fig):
    return fig.layout.xaxis.fixedrange and fig.layout.yaxis.fixedrange


def _v(**kw):
    s = ev.Settings(**kw)
    return s, ev.instance_data(s)


def test_arrangement_and_positions_for_every_family_and_size():
    for fam in C.FAMILIES:
        for n in (6, 9, 12, 16):
            inst = I.make_instance(fam, n, 3)
            assert len(V.build_arrangement(inst, list(range(n)), fam).data[-1].x) == n and _locked(V.build_arrangement(inst, list(range(n)), fam))
    s, v = _v(n=6, seed=35)
    st, _ = ev.main_result(s, v)
    f = ev.start_field(s, v)
    fig = V.build_positions(v["inst"], v["pads"], st.X, f["bins"], "t")
    assert list(fig.data[-1].x) == list(st.X[:, 0]) and list(fig.data[-1].y) == list(-st.X[:, 1]) and fig.layout.shapes and _locked(fig)


def test_field_chart_has_density_arrows_and_components():
    s, v = _v(seed=48)
    f = ev.start_field(s, v)
    fig = V.build_field(v["inst"], v["pads"], f, "electro", "t")
    kinds = [type(t).__name__ for t in fig.data]
    assert kinds[0] == "Heatmap" and [t.name for t in fig.data if t.name] == ["Kraft der Dichtestrafe", "Bauteile"] and _locked(fig)
    z = np.asarray(fig.data[0].z)
    assert z.shape == f["rho"].shape and z.max() == f["rho"].max() / f["bins"].target and np.array_equal(z[::-1], f["rho"] / f["bins"].target)      # Zeilen für „y wächst nach oben“ umgedreht
    none = V.build_field(v["inst"], v["pads"], ev.start_field(ev.Settings(seed=48, model="none"), v), "none", "t")
    assert "Kraft der Dichtestrafe" not in [t.name for t in none.data]


def test_stage_curve_carries_all_models_and_both_baselines():
    s, v = _v(seed=48)
    fig = V.build_stage_curve(v)
    names = [t.name for t in fig.data]
    assert names[:6] == [C.MODEL_NAMES[m] for m in C.MODELS for _ in (0, 1)] and names[-2:] == ["einmal runden (Stück 9)", "Schleife (Stück 10)"]
    assert list(fig.data[0].y) == [100 * (st.cost - v["ref"]) / v["ref"] for st in v["runs"]["electro"].stages] and list(fig.data[1].y) == [st.overflow for st in v["runs"]["electro"].stages]
    assert list(fig.data[-1].y)[0] == 100 * (v["cost10"] - v["ref"]) / v["ref"] and list(fig.data[0].x) == list(range(1, C.STAGES + 1)) and _locked(fig)


def test_path_chart_shows_stages_1_4_8_and_12():
    s, v = _v(seed=48)
    f = ev.start_field(s, v)
    fig = V.build_path(v["inst"], v["pads"], v["runs"]["electro"], f["bins"], "t")
    assert [a.text for a in fig.layout.annotations] == ["Stufe 1", "Stufe 4", "Stufe 8", "Stufe 12"]
    marks = [t for t in fig.data if t.mode == "markers" and t.marker.color == C.COLORS["cont"]]
    assert [list(t.x) for t in marks] == [list(v["runs"]["electro"].stages[k].X[:, 0]) for k in (0, 3, 7, 11)]


def test_model_bars_and_sweep_charts():
    s, v = _v(seed=48)
    fig = V.build_model_bars(v, "last")
    assert len(fig.data[0].x) == 6 and set(fig.data[0].y) >= {v["cost9"], v["cost10"], min(v["runs"]["electro"].costs)} and _locked(fig)
    rows = [fake_row(p, 100.0, COSTS) for p in (2, 4, 8)]
    f1 = V.build_sweep_bars(rows, ev.gap_summary, "t")
    assert len(f1.data) == 6 and list(f1.data[0].x) == ["2 Pads", "4 Pads", "8 Pads"] and list(f1.data[4].y) == [4.0, 4.0, 4.0] and _locked(f1)
    f2 = V.build_stage_medians(rows, ev.stage_medians, "t")
    assert len(f2.data) == 6 and list(f2.data[0].y)[0] == 5.0 and _locked(f2)
    arows = [{"family": "cluster", "seed": 1, "p": 4, "ref": 100.0, "gap9": 0.1, "gap": {k: 0.03 for k in ev.ABLATION}, "ov": {k: 0.2 for k in ev.ABLATION}}]
    f3 = V.build_ablation_bars(arows, ev.ablation_summary, C.MARGIN_CHOICES, C.WIDTH_CHOICES, "t")
    assert len(f3.data[0].x) == 8 and list(f3.data[0].y) == [3.0] * 8 and _locked(f3)


def test_the_pdf_export_returns_a_pdf_for_an_exact_and_a_large_instance():
    s, v = _v(family="netlist", n=6, seed=35)
    assert build_pdf(s, v)[:5] == b"%PDF-"
    s2, v2 = _v(family="uniform", n=12, seed=35, pads=2, model="none", margin=0.0, width=1.0, rule="best")
    data = build_pdf(s2, v2)
    assert data[:5] == b"%PDF-" and len(data) > 1500
