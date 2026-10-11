"""Pure, testable helpers for the Streamlit front end.

Nothing here imports ``streamlit``: every function takes plain values or
DataFrames and returns plain values, DataFrames or Plotly figures, so the UI's
arithmetic and charts can be smoke-tested headlessly in
``pipeline/tests/test_simulator_app.py``.

The engine itself is ``pipeline.simulator``. These helpers only reshape its
output for display: they never re-implement the chain or the ML layer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.calendar_events import COVERED_FROM, COVERED_TO  # noqa: E402
from pipeline.simulator import (  # noqa: E402
    LEVER_BOUNDS,
    RouteEdit,
    Scenario,
)

# Human labels and one-line help for the six global dials. Keys match
# ``LEVER_BOUNDS`` exactly; the order is the sidebar order.
LEVER_LABELS = {
    "seats_mult": "Seats / frequency",
    "lf_mult": "Load factor",
    "p2p_mult": "P2P share (transfer / transit)",
    "capture_mult": "Capture rate",
    "alos_mult": "Average length of stay",
    "season_mult": "Season",
}

LEVER_HELP = {
    "seats_mult": (
        "Scheduled seat supply. Scales seats, PAX and P2P together, so the "
        "model's supply features and its exposure offset both respond."
    ),
    "lf_mult": "How full the seats are. Acts pre-model, like seats.",
    "p2p_mult": (
        "Share of passengers that terminate (visitors), as opposed to "
        "transfer / transit through AUH."
    ),
    "capture_mult": (
        "Behavioural dial on New Arrivals: the share of terminating "
        "passengers that become hotel guests. A calibration coefficient, not a "
        "conversion probability."
    ),
    "alos_mult": "Average length of stay; scales guest nights only.",
    "season_mult": "Seasonal shape; Scales New Arrivals and Guest Nights.",
}

METRIC_LABELS = {
    "GuestNightsHat": "Guest nights",
    "NewArrivalHat": "New arrivals (hotel guests)",
    "P2P": "Terminating passengers (P2P)",
}


def lever_sliders(window: str = "full") -> dict:
    """The six slider specs in sidebar order, from the engine's ``LEVER_BOUNDS``."""
    return {key: LEVER_BOUNDS[key] for key in LEVER_LABELS}


def month_options() -> list[str]:
    """Every month the pack covers, as ``YYYY-MM`` strings."""
    return [str(p) for p in pd.period_range(COVERED_FROM, COVERED_TO, freq="M")]


def route_edit(
    market: str,
    city: str,
    airline: str,
    flights_per_week: float,
    seats_per_flight: float,
    started_period: str,
) -> RouteEdit:
    """Turn the route form's two supply inputs into one ``RouteEdit``.

    The brief names *weekly frequency* and *aircraft seat capacity* as levers;
    the engine wants seats per week, so they are multiplied here.
    """
    if flights_per_week <= 0:
        raise ValueError("flights_per_week must be > 0")
    if seats_per_flight <= 0:
        raise ValueError("seats_per_flight must be > 0")
    return RouteEdit(
        market=market.strip().upper(),
        city=city.strip(),
        airline=airline.strip(),
        seats_per_week=float(flights_per_week) * float(seats_per_flight),
        started_period=str(started_period),
    )


def build_scenario(
    levers: dict,
    market_mults: tuple = (),
    routes: tuple = (),
) -> Scenario:
    """Assemble a ``Scenario`` from sidebar lever values plus advanced inputs."""
    clean = {k: float(v) for k, v in levers.items() if k in LEVER_LABELS}
    return Scenario(
        market_seats_mult=tuple(market_mults),
        routes=tuple(routes),
        **clean,
    )


# -- views reshaped from the market-month prediction frame --------------------
_PRIMARY = ["P2P", "NewArrivalHat", "GuestNightsHat"]


def monthly_from(market_frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a market-month frame to a monthly series (safe default view)."""
    g = (
        market_frame.groupby("ym", as_index=False)[_PRIMARY]
        .sum()
        .sort_values("ym")
        .reset_index(drop=True)
    )
    g["ym_str"] = g["ym"].astype(str)
    return g


def seasonal_from(market_frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a market-month frame to the seasonal shape (by calendar month)."""
    return (
        market_frame.groupby("month", as_index=False)[_PRIMARY]
        .sum()
        .sort_values("month")
        .reset_index(drop=True)
    )


def market_totals(
    market_frame: pd.DataFrame, metric: str = "GuestNightsHat"
) -> pd.DataFrame:
    """Total ``metric`` per source market, largest first (supply-side-only)."""
    return (
        market_frame.groupby("market", as_index=False)[metric]
        .sum()
        .sort_values(metric, ascending=False)
        .reset_index(drop=True)
    )


# -- headline / deltas -------------------------------------------------------
def headline(scen_frame: pd.DataFrame, base_frame: pd.DataFrame) -> pd.DataFrame:
    """Scenario vs baseline totals with absolute and percentage change."""
    rows = []
    for col in _PRIMARY:
        base = float(base_frame[col].sum())
        scen = float(scen_frame[col].sum())
        rows.append(
            {
                "metric": METRIC_LABELS[col],
                "baseline": base,
                "scenario": scen,
                "change": scen - base,
                "change_pct": 100.0 * (scen / base - 1.0) if base else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def chain_waterfall(sim, scenario: Scenario) -> pd.DataFrame:
    """The parametric chain's five stage totals under ``scenario``.

    Uses ``Simulator.chain`` so the waterfall is the auditable seats -> PAX ->
    P2P -> arrivals -> guest-nights path. The chain applies the six *global*
    dials only; per-market seat multipliers are a hybrid-layer feature (see the
    UI caption).
    """
    f = sim.chain(scenario)
    seats = float(f["Total_Seats"].sum()) * float(scenario.seats_mult)
    return pd.DataFrame(
        {
            "stage": [
                "Scheduled seats",
                "Passengers (PAX)",
                "Terminating (P2P)",
                "New arrivals",
                "Guest nights",
            ],
            "value": [
                seats,
                float(f["PAX"].sum()),
                float(f["P2P"].sum()),
                float(f["NewArrivalHat"].sum()),
                float(f["GuestNightsHat"].sum()),
            ],
        }
    )


# -- chart builders ----------------------------------------------------------
def _import_go():
    import plotly.graph_objects as go

    return go


def monthly_trend(
    scen_monthly: pd.DataFrame,
    base_monthly: pd.DataFrame,
    chain_monthly: pd.DataFrame | None = None,
    metric: str = "GuestNightsHat",
) -> "object":
    """Baseline vs scenario monthly lines, with an optional parametric chain."""
    go = _import_go()
    fig = go.Figure()
    fig.add_scatter(
        x=base_monthly["ym_str"],
        y=base_monthly[metric],
        name="Baseline (observed schedule)",
        mode="lines",
        line=dict(color="#8a8f98", dash="dot"),
    )
    fig.add_scatter(
        x=scen_monthly["ym_str"],
        y=scen_monthly[metric],
        name="Scenario (hybrid)",
        mode="lines",
        line=dict(color="#1f77b4", width=3),
    )
    if chain_monthly is not None:
        fig.add_scatter(
            x=chain_monthly["ym_str"],
            y=chain_monthly[metric],
            name="Scenario (parametric chain)",
            mode="lines",
            line=dict(color="#ff7f0e", dash="dash"),
        )
    fig.update_layout(
        title=f"{METRIC_LABELS[metric]} by month",
        xaxis_title="Month",
        yaxis_title=METRIC_LABELS[metric],
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def seasonal_bars(
    scen_seasonal: pd.DataFrame,
    base_seasonal: pd.DataFrame,
    metric: str = "GuestNightsHat",
) -> "object":
    """Baseline vs scenario seasonal shape (calendar-month bars)."""
    go = _import_go()
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig = go.Figure()
    fig.add_bar(
        x=[months[m - 1] for m in base_seasonal["month"]],
        y=base_seasonal[metric],
        name="Baseline",
        marker_color="#8a8f98",
    )
    fig.add_bar(
        x=[months[m - 1] for m in scen_seasonal["month"]],
        y=scen_seasonal[metric],
        name="Scenario",
        marker_color="#1f77b4",
    )
    fig.update_layout(
        barmode="group",
        title=f"{METRIC_LABELS[metric]} by calendar month",
        xaxis_title="Month",
        yaxis_title=METRIC_LABELS[metric],
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def market_pareto(
    scen_totals: pd.DataFrame,
    base_totals: pd.DataFrame,
    top_n: int = 15,
    metric: str = "GuestNightsHat",
) -> "object":
    """Top markets by scenario guest nights, with the baseline alongside."""
    go = _import_go()
    top = scen_totals.head(top_n)
    base_map = base_totals.set_index("market")[metric]
    fig = go.Figure()
    fig.add_bar(
        x=top["market"],
        y=[float(base_map.get(m, 0.0)) for m in top["market"]],
        name="Baseline",
        marker_color="#8a8f98",
    )
    fig.add_bar(
        x=top["market"],
        y=top[metric],
        name="Scenario",
        marker_color="#1f77b4",
    )
    fig.update_layout(
        barmode="group",
        title=f"Top {top_n} source markets by {METRIC_LABELS[metric].lower()}",
        xaxis_title="Source market",
        yaxis_title=METRIC_LABELS[metric],
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def sensitivity_bars(table: pd.DataFrame, value: str = "ST") -> "object":
    """Lever ranking bar chart from a ``lever`` / metric table."""
    go = _import_go()
    t = table.sort_values(value, ascending=True)
    fig = go.Figure(
        go.Bar(x=t[value], y=t["lever"], orientation="h", marker_color="#1f77b4")
    )
    fig.update_layout(
        title=f"Lever ranking ({value})",
        xaxis_title=value,
        yaxis_title="",
        margin=dict(l=10, r=10, t=60, b=10),
        height=320,
    )
    return fig


def validation_scatter(rep: pd.DataFrame, actual: str, hat: str) -> "object":
    """Actual-vs-predicted scatter with a y=x reference line."""
    go = _import_go()
    lo = float(min(rep[actual].min(), rep[hat].min()))
    hi = float(max(rep[actual].max(), rep[hat].max()))
    fig = go.Figure()
    fig.add_scatter(
        x=rep[actual],
        y=rep[hat],
        mode="markers",
        name="Market-months",
        marker=dict(color="#1f77b4", opacity=0.5, size=6),
    )
    fig.add_scatter(
        x=[lo, hi],
        y=[lo, hi],
        mode="lines",
        name="Perfect prediction",
        line=dict(color="#d62728", dash="dash"),
    )
    fig.update_layout(
        title="Actual vs predicted",
        xaxis_title=actual,
        yaxis_title=hat,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig
