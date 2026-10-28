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

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.calendar_events import COVERED_FROM, COVERED_TO  # noqa: E402
from pipeline.simulator import (  # noqa: E402
    LEVER_BOUNDS,
    RouteChange,
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
    "season_mult": "Season (level)",
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
    "season_mult": (
        "A global level shift on the seasonal profile. Use Advanced: seasonal "
        "shape to change the shape of the year itself."
    ),
}

METRIC_LABELS = {
    "GuestNightsHat": "Guest nights",
    "NewArrivalHat": "New arrivals (hotel guests)",
    "P2P": "Terminating passengers (P2P)",
}

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# The impact window matters: a route that starts in 2025-09 should not be
# diluted by 44 months of pre-route history.
WINDOW_OPTIONS = [
    "Full window (2022-01 → 2026-02)",
    "First 12 months after change",
    "Last 12 months",
    "Calendar 2025",
    "Winter (Dec–Feb)",
    "Summer (Jun–Aug)",
]

SEASON_MONTHS = {
    "Winter": (12, 1, 2),
    "Spring": (3, 4, 5),
    "Summer": (6, 7, 8),
    "Autumn": (9, 10, 11),
}


def filter_window(frame: pd.DataFrame, window: str, route_start: str | None = None):
    """Return the rows that belong to the selected impact window.

    ``First 12 months after change`` needs the earliest route start; if no route
    is configured it falls back to the full frame rather than inventing a date.
    """
    if frame.empty or window == WINDOW_OPTIONS[0]:
        return frame
    f = frame
    if window == "First 12 months after change":
        if not route_start:
            return f
        start = pd.Period(route_start, freq="M")
        end = start + 11
        return f[(f["ym"] >= start) & (f["ym"] <= end)]
    if window == "Last 12 months":
        last = f["ym"].max()
        return f[f["ym"] >= last - 11]
    if window == "Calendar 2025":
        return f[f["ym"].dt.year == 2025]
    if window == "Winter (Dec–Feb)":
        return f[f["month"].isin([12, 1, 2])]
    if window == "Summer (Jun–Aug)":
        return f[f["month"].isin([6, 7, 8])]
    return f


def season_profile_from(winter: float, spring: float, summer: float, autumn: float):
    """Build a Jan..Dec multiplier tuple from four intuitive season dials."""
    return (
        winter, winter, spring, spring, spring, summer,
        summer, summer, autumn, autumn, autumn, winter,
    )


def market_month_pairs(rows, season_months: dict | None = None) -> tuple:
    """Turn a market × season editor into ``((market, month, mult), ...)``."""
    season_months = season_months or SEASON_MONTHS
    pairs = []
    for row in rows:
        market = str(row["market"])
        for season, months in season_months.items():
            value = float(row[season])
            if abs(value - 1.0) > 1e-9:
                pairs.extend((market, month, value) for month in months)
    return tuple(pairs)


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


def route_change(
    market: str,
    seats_mult: float,
    started_period: str,
    ended_period: str | None = None,
    city: str = "",
    airline: str = "",
) -> RouteChange:
    """Build a route-level scale/closure change, rejecting contradictory input."""
    if seats_mult < 0:
        raise ValueError("seats_mult must be >= 0; 0 closes the route")
    if ended_period and str(ended_period) < str(started_period):
        raise ValueError("ended_period must be on or after started_period")
    return RouteChange(
        market=market.strip().upper(),
        seats_mult=float(seats_mult),
        started_period=str(started_period),
        ended_period=str(ended_period) if ended_period else None,
        city=city.strip(),
        airline=airline.strip(),
    )


def build_scenario(
    levers: dict,
    market_mults: tuple = (),
    routes: tuple = (),
    route_changes: tuple = (),
    season_profile: tuple = (),
    market_month_mult: tuple = (),
    closed_markets: tuple = (),
) -> Scenario:
    """Assemble a ``Scenario`` from sidebar lever values plus advanced inputs."""
    clean = {k: float(v) for k, v in levers.items() if k in LEVER_LABELS}
    return Scenario(
        market_seats_mult=tuple(market_mults),
        routes=tuple(routes),
        route_changes=tuple(route_changes),
        season_profile=tuple(season_profile),
        market_month_mult=tuple(market_month_mult),
        closed_markets=tuple(closed_markets),
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
def market_season_matrix(market_frame: pd.DataFrame, metric: str = "GuestNightsHat"):
    """Market × calendar-month matrix, largest markets first.

    This is the visual answer to "by country and time of year". The numbers are
    still supply-side-only; the app must show the attribution warning next to it.
    """
    matrix = market_frame.pivot_table(
        index="market", columns="month", values=metric, aggfunc="sum", fill_value=0.0
    )
    matrix = matrix.reindex(columns=range(1, 13), fill_value=0.0)
    matrix.columns = MONTHS
    order = matrix.sum(axis=1).sort_values(ascending=False).index
    return matrix.loc[order]


def attribution_range(scen_frame: pd.DataFrame, hotel_mix: pd.Series):
    """Bound each market's guest nights under three explicit allocations.

    The model reports the last-departure join. We also allocate the same national
    total by (a) the measured hotel-guest mix and (b) the measured supply mix.
    The spread is a *range across assumptions*, not a confidence interval.
    """
    current = scen_frame.groupby("market")["GuestNightsHat"].sum().astype(float)
    total = float(current.sum())
    markets = current.index

    demand = hotel_mix.reindex(markets).astype(float)
    if float(demand.sum()) <= 0:
        demand = pd.Series(1.0, index=markets)
    demand = demand / float(demand.sum()) * total

    p2p = scen_frame.groupby("market")["P2P"].sum().astype(float)
    if float(p2p.sum()) <= 0:
        p2p = pd.Series(1.0, index=markets)
    supply = p2p / float(p2p.sum()) * total

    out = pd.DataFrame(
        {
            "market": markets,
            "last_departure": current.to_numpy(),
            "demand_mix": demand.to_numpy(),
            "supply_mix": supply.to_numpy(),
        }
    )
    out["low"] = out[["last_departure", "demand_mix", "supply_mix"]].min(axis=1)
    out["high"] = out[["last_departure", "demand_mix", "supply_mix"]].max(axis=1)
    out["spread_pct"] = np.where(
        out["last_departure"] > 0,
        100.0 * (out["high"] - out["low"]) / out["last_departure"],
        np.nan,
    )
    return out.sort_values("last_departure", ascending=False).reset_index(drop=True)


def efficiency_table(scen_frame: pd.DataFrame, capture: pd.Series | None = None):
    """Guest volume per 1,000 scheduled seats, by market.

    ``capture > 1`` marks the known-contaminated markets; their ratio is shown
    but flagged rather than treated as a measured conversion rate.
    """
    g = (
        scen_frame.groupby("market")
        .agg(
            Total_Seats=("Total_Seats", "sum"),
            P2P=("P2P", "sum"),
            NewArrivalHat=("NewArrivalHat", "sum"),
            GuestNightsHat=("GuestNightsHat", "sum"),
        )
        .reset_index()
    )
    g["guests_per_1k_seats"] = np.where(
        g["Total_Seats"] > 0, 1000.0 * g["NewArrivalHat"] / g["Total_Seats"], np.nan
    )
    g["nights_per_1k_seats"] = np.where(
        g["Total_Seats"] > 0, 1000.0 * g["GuestNightsHat"] / g["Total_Seats"], np.nan
    )
    if capture is not None:
        g["capture"] = g["market"].map(capture)
        g["attribution_flag"] = g["capture"] > 1.0
    else:
        g["capture"] = np.nan
        g["attribution_flag"] = False
    return g.sort_values("nights_per_1k_seats", ascending=False).reset_index(drop=True)


def wmape(actual, pred) -> float:
    """Weighted mean absolute percentage error, the brief's metric."""
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(100.0 * np.abs(actual - pred).sum() / np.abs(actual).sum())


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
    # H.simulate now writes the scaled seats into Total_Seats, so the waterfall
    # must not re-apply seats_mult (that would double-count it).
    seats = float(f["Total_Seats"].sum())
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


def market_season_heatmap(
    matrix: pd.DataFrame, metric: str = "GuestNightsHat"
) -> "object":
    """Heatmap of market × month (supply-side-only)."""
    go = _import_go()
    fig = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy(),
            x=list(matrix.columns),
            y=list(matrix.index),
            colorscale="Blues",
            hovertemplate="%{y} · %{x}<br>%{z:,.0f}<extra></extra>",
            colorbar=dict(title=METRIC_LABELS[metric]),
        )
    )
    fig.update_layout(
        title=f"{METRIC_LABELS[metric]} by market and month (supply-side-only)",
        xaxis_title="Month",
        yaxis_title="Market",
        height=max(360, 22 * len(matrix.index)),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def attribution_range_chart(
    ranges: pd.DataFrame, top_n: int = 15
) -> "object":
    """Last-departure point estimate with the demand/supply-mix range."""
    go = _import_go()
    t = ranges.head(top_n).iloc[::-1]
    low_err = t["last_departure"] - t["low"]
    high_err = t["high"] - t["last_departure"]
    fig = go.Figure(
        go.Scatter(
            x=t["last_departure"],
            y=t["market"],
            mode="markers",
            marker=dict(size=9, color="#1f77b4"),
            error_x=dict(
                type="data",
                symmetric=False,
                array=high_err,
                arrayminus=low_err,
                color="#8a8f98",
                thickness=2,
                width=4,
            ),
            name="Last-departure estimate",
            hovertemplate="%{y}<br>%{x:,.0f} guest nights<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Attribution range, top {top_n} markets",
        xaxis_title="Guest nights (last-departure point + assumption range)",
        yaxis_title="",
        height=max(360, 24 * len(t)),
        margin=dict(l=10, r=10, t=50, b=10),
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
