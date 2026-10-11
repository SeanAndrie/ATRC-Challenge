"""Plotly figure builders with one shared look.

Colour does one job per chart: grey is the baseline, blue the scenario or the
measured magnitude, and gains/losses use the blue/red diverging pair. Values
and labels stay in ink colours; marks carry the hue. Palette values are the
dataviz reference palette (light mode).
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GAIN = "#2a78d6"
LOSS = "#e34948"
BASE = "#a9a8a1"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _layout(fig: go.Figure, height: int = 360, **kw) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=36 if kw.get("title") else 12, b=8),
        font=dict(family=FONT, color=INK_2, size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(bgcolor="white", bordercolor=GRID, font=dict(family=FONT, color=INK)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
        **kw,
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=MUTED))
    return fig


def compare_lines(df: pd.DataFrame, x: str = "ym_str", base: str = "base",
                  scen: str = "scenario", y_title: str = "Guest-nights",
                  observed_to: str | None = None) -> go.Figure:
    """Baseline (grey) vs scenario (blue), with the gap shaded."""
    fig = go.Figure()
    fig.add_scatter(x=df[x], y=df[base], name="Baseline", mode="lines",
                    line=dict(color=BASE, width=2), hovertemplate="%{y:,.0f}")
    fig.add_scatter(x=df[x], y=df[scen], name="Scenario", mode="lines",
                    line=dict(color=BLUE, width=2.5), fill="tonexty",
                    fillcolor="rgba(42,120,214,0.10)", hovertemplate="%{y:,.0f}")
    if observed_to and df[x].min() <= observed_to <= df[x].max():
        fig.add_vline(x=observed_to, line_width=1, line_color=AXIS)
        fig.add_annotation(x=observed_to, y=1, yref="paper", text="observed ← → held forward",
                           showarrow=False, yanchor="bottom", font=dict(size=11, color=MUTED))
    return _layout(fig, yaxis_title=y_title, hovermode="x unified")


def delta_bars(df: pd.DataFrame, x: str = "ym_str", y: str = "delta",
               y_title: str = "Δ guest-nights", height: int = 300) -> go.Figure:
    """Signed monthly change: blue up, red down."""
    fig = go.Figure(go.Bar(
        x=df[x], y=df[y], marker_color=[GAIN if v >= 0 else LOSS for v in df[y]],
        marker_cornerradius=4, hovertemplate="%{x}<br>%{y:+,.0f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_width=1, line_color=AXIS)
    return _layout(fig, height=height, yaxis_title=y_title, bargap=0.35, showlegend=False)


def stage_funnel(stages: dict[str, float]) -> go.Figure:
    """Where added (or removed) seats end up, stage by stage."""
    labels = ["Seats", "Passengers", "Terminating passengers", "Hotel arrivals"]
    keys = ["seats", "passengers", "terminating_passengers", "hotel_arrivals"]
    vals = [abs(stages[k]) for k in keys]
    fig = go.Figure(go.Funnel(
        y=labels, x=vals, textinfo="value+percent initial", texttemplate="%{value:,.0f}  (%{percentInitial:.0%})",
        marker=dict(color=BLUE_RAMP[2:6]), connector=dict(fillcolor="#eef3fb", line=dict(width=0)),
        textfont=dict(color="white"), hovertemplate="%{y}: %{x:,.0f}<extra></extra>",
    ))
    return _layout(fig, height=280, showlegend=False)


def ranking_bar(df: pd.DataFrame, value: str, label: str = "market", highlight: str | None = None,
                flagged: set[str] | None = None, x_title: str = "", fmt: str = ",.0f",
                height: int | None = None) -> go.Figure:
    """Horizontal ranking, largest on top; one bar optionally highlighted."""
    d = df.iloc[::-1]
    flagged = flagged or set()
    colors = [BLUE if (highlight is None or m == highlight) else "#9ec5f4" for m in d[label]]
    names = [f"{m} ⚠" if m in flagged else m for m in d[label]]
    fig = go.Figure(go.Bar(
        x=d[value], y=names, orientation="h", marker_color=colors, marker_cornerradius=4,
        text=[f"{v:{fmt}}" for v in d[value]], textposition="outside", cliponaxis=False,
        textfont=dict(color=INK_2), hovertemplate="%{y}: %{x:" + fmt + "}<extra></extra>",
    ))
    return _layout(fig, height=height or max(260, 24 * len(d) + 60), xaxis_title=x_title,
                   showlegend=False, bargap=0.3)


def bubble(df: pd.DataFrame) -> go.Figure:
    """Capture vs length of stay, sized by seats; capture > 1 marks the join defect."""
    ok = df[df["capture"] <= 1.0]
    bad = df[df["capture"] > 1.0]
    size = 12 + 38 * (df["seats"] / df["seats"].max()) ** 0.5
    fig = go.Figure()
    for part, color, name in ((ok, BLUE, "Capture ≤ 1"), (bad, ORANGE, "Capture > 1 (join defect)")):
        fig.add_scatter(
            x=part["capture"], y=part["alos"], mode="markers+text", name=name,
            text=part["market"].str.title(), textposition="top center",
            textfont=dict(size=10, color=INK_2),
            marker=dict(size=size[part.index], color=color, opacity=0.75,
                        line=dict(color=SURFACE, width=2)),
            customdata=part[["seats", "arrivals"]],
            hovertemplate="%{text}<br>capture %{x:.2f} · stay %{y:.1f} nights"
                          "<br>seats %{customdata[0]:,.0f}<extra></extra>",
        )
    fig.add_vline(x=1.0, line_width=1, line_dash="dot", line_color=AXIS)
    fig.update_xaxes(type="log")
    return _layout(fig, height=460, xaxis_title="Hotel arrivals per terminating passenger (log)",
                   yaxis_title="Average stay (nights)")


def heatmap(pivot: pd.DataFrame, title: str = "Hotel arrivals per 1,000 seats") -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=[MONTHS[int(m) - 1] for m in pivot.columns], y=pivot.index,
        colorscale=[[i / (len(BLUE_RAMP) - 1), c] for i, c in enumerate(BLUE_RAMP)],
        xgap=2, ygap=2, colorbar=dict(title=None, thickness=10, outlinewidth=0),
        hovertemplate="%{y} · %{x}<br>%{z:,.0f} per 1,000 seats<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    fig.update_xaxes(side="top", gridcolor="rgba(0,0,0,0)")
    return _layout(fig, height=max(360, 20 * len(pivot) + 60))


def tornado(df: pd.DataFrame, labels: dict[str, str]) -> go.Figure:
    """Each lever moved down (red) and up (blue) on its own."""
    names = [labels.get(lv, lv) for lv in df["lever"]]
    fig = go.Figure()
    fig.add_bar(y=names, x=df["low"], orientation="h", name="−10%", marker_color=LOSS,
                marker_cornerradius=4, hovertemplate="%{y} −10%: %{x:+,.0f}<extra></extra>")
    fig.add_bar(y=names, x=df["high"], orientation="h", name="+10%", marker_color=GAIN,
                marker_cornerradius=4, hovertemplate="%{y} +10%: %{x:+,.0f}<extra></extra>")
    fig.add_vline(x=0, line_width=1, line_color=AXIS)
    return _layout(fig, height=320, barmode="overlay", bargap=0.35,
                   xaxis_title="Δ guest-nights over the next 12 months")


def contribution(df: pd.DataFrame) -> go.Figure:
    d = df.sort_values("delta", key=abs)
    fig = go.Figure(go.Bar(
        x=d["delta"], y=d["change"], orientation="h", marker_cornerradius=4,
        marker_color=[GAIN if v >= 0 else LOSS for v in d["delta"]],
        hovertemplate="%{y}<br>%{x:+,.0f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_width=1, line_color=AXIS)
    return _layout(fig, height=max(200, 40 * len(d) + 60), xaxis_title="Δ guest-nights, each change alone",
                   showlegend=False, bargap=0.35)


def chain_sankey(seats: float, pax: float, p2p: float, arrivals: float) -> go.Figure:
    """A year of supply flowing through the chain to hotel arrivals."""
    nodes = ["Seats", "Empty seats", "Passengers", "Transfer & transit",
             "Terminating passengers", "Not staying in hotels", "Hotel arrivals"]
    src = [0, 0, 2, 2, 4, 4]
    tgt = [2, 1, 4, 3, 6, 5]
    val = [pax, seats - pax, p2p, pax - p2p, arrivals, max(p2p - arrivals, 0)]
    keep = "rgba(42,120,214,0.35)"
    leak = "rgba(169,168,161,0.30)"
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(label=nodes, pad=18, thickness=14, line=dict(width=0),
                  color=[BLUE, BASE, BLUE, BASE, BLUE, BASE, "#184f95"],
                  hovertemplate="%{label}: %{value:,.0f}<extra></extra>"),
        link=dict(source=src, target=tgt, value=val, color=[keep, leak, keep, leak, keep, leak],
                  hovertemplate="%{source.label} → %{target.label}: %{value:,.0f}<extra></extra>"),
        textfont=dict(family=FONT, color=INK, size=12),
    ))
    return _layout(fig, height=330)


def backtest_lines(df: pd.DataFrame) -> go.Figure:
    """Monthly actual vs hybrid vs chain on the held-out window."""
    fig = go.Figure()
    fig.add_scatter(x=df["ym_str"], y=df["NewArr"], name="Actual", mode="lines+markers",
                    line=dict(color=INK, width=2), marker=dict(size=8, line=dict(color=SURFACE, width=2)))
    fig.add_scatter(x=df["ym_str"], y=df["NewArrivalHat"], name="Hybrid model", mode="lines+markers",
                    line=dict(color=BLUE, width=2), marker=dict(size=8, line=dict(color=SURFACE, width=2)))
    fig.add_scatter(x=df["ym_str"], y=df["ChainHat"], name="Parametric chain", mode="lines+markers",
                    line=dict(color=ORANGE, width=2, dash="dot"),
                    marker=dict(size=8, line=dict(color=SURFACE, width=2)))
    return _layout(fig, yaxis_title="New hotel arrivals", hovermode="x unified")


def backtest_scatter(df: pd.DataFrame, pred: str, actual: str = "NewArr",
                     title: str = "") -> go.Figure:
    lo = max(1.0, min(df[actual].min(), df[pred].min()))
    hi = max(df[actual].max(), df[pred].max())
    fig = go.Figure()
    fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=AXIS, width=1),
                    hoverinfo="skip", showlegend=False)
    fig.add_scatter(x=df[actual], y=df[pred], mode="markers", text=df["market"],
                    marker=dict(size=8, color=BLUE, opacity=0.6, line=dict(color=SURFACE, width=1)),
                    hovertemplate="%{text}<br>actual %{x:,.0f} · predicted %{y:,.0f}<extra></extra>",
                    showlegend=False)
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")
    return _layout(fig, height=380, xaxis_title="Actual (market-month)", yaxis_title="Predicted",
                   title=dict(text=title, font=dict(size=14, color=INK)) if title else None)
