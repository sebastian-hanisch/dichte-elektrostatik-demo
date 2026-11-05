"""Plotly-Abbildungen. Alle 2D-Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dla_constants as C

MODEL_COLORS = {"electro": "#2F6B65", "overflow": "#b08a2e", "none": "#9aa0a6"}


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=380, legend_below=True):
    legend = dict(orientation="h", yanchor="top", y=-0.22, x=0) if legend_below else dict(orientation="h", yanchor="bottom", y=1.02, x=0)
    fig.update_layout(height=height + (40 if legend_below else 0), margin=dict(l=10, r=10, t=50, b=10), legend=legend, plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _slots(inst):
    return go.Scatter(x=inst.xy[:, 0].astype(float), y=-inst.xy[:, 1].astype(float), mode="markers", marker=dict(symbol="square", size=34, color="rgba(150,150,150,0.14)", line=dict(width=1, color="rgba(150,150,150,0.5)")), hoverinfo="skip", showlegend=False)


def _pads(pads):
    return go.Scatter(x=pads.pos[:, 0], y=-pads.pos[:, 1], mode="markers", name="Pads", marker=dict(symbol="diamond", size=12, color=C.COLORS["pad"], line=dict(width=1, color="white")), hovertemplate="Pad<extra></extra>")


def _region(fig, bins, row=None, col=None):
    x0, x1, y0, y1 = bins.region
    kw = {} if row is None else dict(row=row, col=col)
    fig.add_shape(type="rect", x0=x0, x1=x1, y0=-y1, y1=-y0, line=dict(color="rgba(76,120,168,0.7)", width=1.5, dash="dot"), **kw)


def build_arrangement(inst, perm, title: str) -> go.Figure:
    """Die gerundete Anordnung: Bauteile auf den Plätzen, Verbindungen mit Fluss als Linien."""
    fig = go.Figure()
    xs, ys = inst.xy[:, 0].astype(float), -inst.xy[:, 1].astype(float)
    pos = [(xs[int(perm[i])], ys[int(perm[i])]) for i in range(inst.n)]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", marker=dict(symbol="square", size=36, color="rgba(150,150,150,0.18)", line=dict(width=1, color="rgba(150,150,150,0.5)")), hoverinfo="skip", showlegend=False))
    lx, ly = [], []
    for a in range(inst.n):
        for b in range(a + 1, inst.n):
            if inst.F[a, b]:
                lx += [pos[a][0], pos[b][0], None]
                ly += [pos[a][1], pos[b][1], None]
    fig.add_trace(go.Scatter(x=lx, y=ly, mode="lines", line=dict(width=1.5, color="rgba(76,120,168,0.35)"), hoverinfo="skip", name="Fluss"))
    fig.add_trace(go.Scatter(x=[q[0] for q in pos], y=[q[1] for q in pos], mode="markers+text", text=list(inst.names), textposition="middle center", showlegend=False, marker=dict(size=26, color=C.COLORS["best"]), textfont=dict(color="white", size=10)))
    fig.update_layout(title=title)
    fig.update_xaxes(visible=False, range=[xs.min() - 0.7, xs.max() + 0.7], scaleanchor="y", scaleratio=1)
    fig.update_yaxes(visible=False, range=[ys.min() - 0.7, ys.max() + 0.7])
    return _base(fig, height=300)


def build_positions(inst, pads, X, bins, title: str) -> go.Figure:
    """Die stetigen Orte (vor der Rundung): Gebiet mit Randzone gestrichelt, Plätze grau, Pads als Rauten."""
    fig = go.Figure()
    fig.add_trace(_slots(inst))
    if pads.count:
        fig.add_trace(_pads(pads))
    fig.add_trace(go.Scatter(x=X[:, 0], y=-X[:, 1], mode="markers+text", text=list(inst.names), textposition="middle center", showlegend=False, marker=dict(size=24, color=C.COLORS["cont"], opacity=0.85), textfont=dict(color="white", size=10), hovertemplate="%{text}<extra></extra>"))
    _region(fig, bins)
    xs = list(inst.xy[:, 0].astype(float)) + list(pads.pos[:, 0]) + [bins.region[0], bins.region[1]]
    ys = list(-inst.xy[:, 1].astype(float)) + list(-pads.pos[:, 1]) + [-bins.region[2], -bins.region[3]]
    fig.update_layout(title=title)
    fig.update_xaxes(visible=False, range=[min(xs) - 0.5, max(xs) + 0.5], scaleanchor="y", scaleratio=1)
    fig.update_yaxes(visible=False, range=[min(ys) - 0.5, max(ys) + 0.5])
    return _base(fig, height=300)


def _heat(bins, z):
    """Heatmap-Spuren in der Darstellung mit nach unten wachsenden Zeilen (y negativ)."""
    xs = bins.x0 + bins.h * (np.arange(bins.nx) + 0.5)
    ys = -(bins.y0 + bins.h * (np.arange(bins.ny) + 0.5))
    return xs, ys[::-1], np.asarray(z)[::-1]


def build_field(inst, pads, f: dict, model: str, title: str) -> go.Figure:
    """Dichte am Startwert (Heatmap, in Vielfachen des Sollwerts), die Orte der Bauteile und die Kraft der Dichtestrafe (Pfeile)."""
    bins = f["bins"]
    xs, ys, z = _heat(bins, f["rho"] / bins.target)
    fig = go.Figure()
    fig.add_trace(go.Heatmap(x=xs, y=ys, z=z, colorscale="YlOrRd", zmin=0, colorbar=dict(title="Dichte / Soll", thickness=12), hovertemplate="Dichte / Soll %{z:.2f}<extra></extra>", showlegend=False))
    fig.add_trace(_slots(inst))
    X, force = f["X"], f["force"]
    norm = float(np.linalg.norm(force, axis=1).max())
    if model != "none" and norm > 0:
        scale = 0.7 / norm
        ax, ay = [], []
        for i in range(len(X)):
            ax += [X[i, 0], X[i, 0] + scale * force[i, 0], None]
            ay += [-X[i, 1], -X[i, 1] - scale * force[i, 1], None]
        fig.add_trace(go.Scatter(x=ax, y=ay, mode="lines", line=dict(color="#222", width=2.5), name="Kraft der Dichtestrafe", hoverinfo="skip"))
        tips = [(X[i, 0] + scale * force[i, 0], -X[i, 1] - scale * force[i, 1]) for i in range(len(X))]
        fig.add_trace(go.Scatter(x=[t[0] for t in tips], y=[t[1] for t in tips], mode="markers", marker=dict(size=7, color="#222"), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=X[:, 0], y=-X[:, 1], mode="markers+text", text=list(inst.names), textposition="middle center", name="Bauteile", marker=dict(size=20, color=C.COLORS["cont"], line=dict(width=1, color="white")), textfont=dict(color="white", size=9)))
    _region(fig, bins)
    x0, x1, y0, y1 = bins.region
    fig.update_layout(title=title)
    fig.update_xaxes(visible=False, range=[x0 - 0.3, x1 + 0.3], scaleanchor="y", scaleratio=1)
    fig.update_yaxes(visible=False, range=[-y1 - 0.3, -y0 + 0.3])
    return _base(fig, height=320)


def build_stage_curve(v: dict, title: str = "Lücke und Überlauf je Stufe") -> go.Figure:
    """Oben die Lücke nach der Rundung je Stufe für die drei Modelle (gestrichelt Stück 9 und 10), unten der Überlauf (Einheitsquadrate)."""
    ref, runs = v["ref"], v["runs"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.6, 0.4])
    xs = list(range(1, C.STAGES + 1))
    for m in C.MODELS:
        pen = runs[m]
        fig.add_trace(go.Scatter(x=xs, y=[100 * (s.cost - ref) / ref for s in pen.stages], mode="lines+markers", name=C.MODEL_NAMES[m], line=dict(color=MODEL_COLORS[m], width=3), marker=dict(size=6), legendgroup=m), row=1, col=1)
        fig.add_trace(go.Scatter(x=xs, y=[s.overflow for s in pen.stages], mode="lines", name=C.MODEL_NAMES[m], line=dict(color=MODEL_COLORS[m], width=2), legendgroup=m, showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=[1, C.STAGES], y=[100 * (v["cost9"] - ref) / ref] * 2, mode="lines", name="einmal runden (Stück 9)", line=dict(color="#4c78a8", dash="dot", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=[1, C.STAGES], y=[100 * (v["cost10"] - ref) / ref] * 2, mode="lines", name="Schleife (Stück 10)", line=dict(color="#c0504d", dash="dash", width=2)), row=1, col=1)
    fig.update_yaxes(title="Lücke (%)", row=1, col=1)
    fig.update_yaxes(title="Überlauf", row=2, col=1)
    fig.update_xaxes(title="Stufe", dtick=1, row=2, col=1)
    fig.update_layout(title=title)
    return _base(fig, height=470)


def build_path(inst, pads, pen, bins, title: str) -> go.Figure:
    """Die Orte nach Stufe 1, 4, 8 und 12 (vier kleine Felder)."""
    picks = [0, 3, 7, C.STAGES - 1]
    fig = make_subplots(rows=1, cols=4, subplot_titles=[f"Stufe {k + 1}" for k in picks], horizontal_spacing=0.03)
    x0, x1, y0, y1 = bins.region
    for j, k in enumerate(picks, start=1):
        sl = _slots(inst)
        fig.add_trace(sl, row=1, col=j)
        X = pen.stages[k].X
        fig.add_trace(go.Scatter(x=X[:, 0], y=-X[:, 1], mode="markers", marker=dict(size=12, color=C.COLORS["cont"], opacity=0.85, line=dict(width=1, color="white")), showlegend=False, hovertemplate="Bauteil %{text}<extra></extra>", text=list(inst.names)), row=1, col=j)
        _region(fig, bins, 1, j)
        fig.update_xaxes(visible=False, range=[x0 - 0.3, x1 + 0.3], scaleanchor=f"y{'' if j == 1 else j}", scaleratio=1, row=1, col=j)
        fig.update_yaxes(visible=False, range=[-y1 - 0.3, -y0 + 0.3], row=1, col=j)
    fig.update_layout(title=title)
    return _base(fig, height=250, legend_below=False)


def build_model_bars(v: dict, rule: str) -> go.Figure:
    """Kosten nach der Rundung je Verfahren derselben Instanz; gestrichelt der Vergleichswert."""
    names = ["Einmal runden (Stück 9)", "Schleife (Stück 10)"] + [C.MODEL_NAMES[m] for m in C.MODELS] + ["Elektrostatisch, billigste Stufe"]
    vals = [v["cost9"], v["cost10"]] + [v["runs"][m].stages[-1].cost for m in C.MODELS] + [min(v["runs"]["electro"].costs)]
    colors = ["#4c78a8", "#c0504d", MODEL_COLORS["electro"], MODEL_COLORS["overflow"], MODEL_COLORS["none"], "#1d4a45"]
    order = [0, 1, 4, 3, 2, 5]
    fig = go.Figure(go.Bar(x=[names[i] for i in order], y=[vals[i] for i in order], marker_color=[colors[i] for i in order], text=[f"{vals[i]:g}" for i in order], textposition="outside"))
    fig.add_hline(y=v["ref"], line_dash="dot", line_color=C.COLORS["best"], annotation_text="Vergleichswert", annotation_position="top left")
    fig.update_yaxes(title="Kosten nach der Rundung", range=[0.8 * min(min(vals), v["ref"]), 1.1 * max(vals)])
    fig.update_layout(title="Die Verfahren an derselben Instanz", showlegend=False)
    return _base(fig, height=340, legend_below=False)


# ----------------------------------------------------------------------------- Vergleiche über feste Instanzen

def build_sweep_bars(rows: list, gap_summary, title: str) -> go.Figure:
    """Median der Lücke je Zahl der Pads für die Verfahren."""
    ps = sorted({r["p"] for r in rows})
    keys = [("s9", "Einmal runden (Stück 9)", "#4c78a8"), ("s10", "Schleife (Stück 10)", "#c0504d"), ("none", C.MODEL_NAMES["none"], MODEL_COLORS["none"]), ("overflow", C.MODEL_NAMES["overflow"], MODEL_COLORS["overflow"]),
            ("electro", C.MODEL_NAMES["electro"], MODEL_COLORS["electro"]), ("electro_best", "Elektrostatisch, billigste Stufe", "#1d4a45")]
    fig = go.Figure()
    for k, name, col in keys:
        fig.add_trace(go.Bar(x=[f"{p} Pads" for p in ps], y=[100 * gap_summary(rows, p, k)["gap"] for p in ps], name=name, marker_color=col))
    fig.update_yaxes(title="Lücke (%, Median)")
    fig.update_layout(title=title, barmode="group")
    return _base(fig, height=400)


def build_stage_medians(rows: list, stage_medians, title: str) -> go.Figure:
    """Median je Stufe über alle Instanzen: Lücke (oben) und Überlauf (unten) für die drei Modelle."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.6, 0.4])
    xs = list(range(1, C.STAGES + 1))
    for m in C.MODELS:
        sm = stage_medians(rows, m)
        fig.add_trace(go.Scatter(x=xs, y=[100 * g for g in sm["gap"]], mode="lines+markers", name=C.MODEL_NAMES[m], line=dict(color=MODEL_COLORS[m], width=3), marker=dict(size=6), legendgroup=m), row=1, col=1)
        fig.add_trace(go.Scatter(x=xs, y=sm["ov"], mode="lines", name=C.MODEL_NAMES[m], line=dict(color=MODEL_COLORS[m], width=2), legendgroup=m, showlegend=False), row=2, col=1)
    fig.update_yaxes(title="Lücke (%, Median)", row=1, col=1)
    fig.update_yaxes(title="Überlauf (Median)", row=2, col=1)
    fig.update_xaxes(title="Stufe", dtick=1, row=2, col=1)
    fig.update_layout(title=title)
    return _base(fig, height=470)


def build_ablation_bars(rows: list, ablation_summary, margins, widths, title: str) -> go.Figure:
    """Median der Lücke je Randzone und je Ladungsbreite (elektrostatisch, letzte Stufe); gestrichelt einmal runden (Stück 9)."""
    keys = [f"m{m}" for m in margins] + [f"w{w}" for w in widths]
    labels = [f"Randzone {m:g}".replace(".", ",") for m in margins] + [f"Ladungsbreite {w:g}".replace(".", ",") for w in widths]
    colors = ["#2F6B65" if k in ("m0.5", "w3.0") else "#7aa5a0" for k in keys]
    fig = go.Figure(go.Bar(x=labels, y=[100 * ablation_summary(rows, k)["gap"] for k in keys], marker_color=colors, text=[f"{100 * ablation_summary(rows, k)['gap']:.1f}".replace(".", ",") + " %" for k in keys], textposition="outside"))
    fig.add_hline(y=100 * float(np.median([r["gap9"] for r in rows])), line_dash="dot", line_color="#4c78a8", annotation_text="einmal runden (Stück 9)", annotation_position="top left")
    fig.update_yaxes(title="Lücke (%, Median)")
    fig.update_layout(title=title, showlegend=False)
    return _base(fig, height=380, legend_below=False)
