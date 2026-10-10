"""Streamlit front end for the flight-to-hotel scenario simulator.

Run from the repository root::

    .venv/bin/streamlit run app/simulator.py

The app wraps the packaged model in ``pipeline.simulator``: it loads the saved
cache in ``outputs/models/`` when one exists and otherwise fits the competition
pack once (~1 minute). Every scenario is scored by the fitted hybrid and by the
parametric chain, against a no-change baseline over the full horizon -- observed
supply through 2026-02, then a held-forward seasonal projection to 2028-12.

Design notes, in one place:

* **Questions map to levers.** New routes and frequency/seat changes are signed,
  dated ``RouteEdit`` rows; load factor is the ``lf_mult`` dial (global or per
  market); seasonal mix shifts are per-market seat multipliers or winter-only
  route edits. The guided tab shows recipes for each question in the brief.
* **New origins are analog estimates.** The model's market categorical has 33
  fixed levels, so a country that never flew here is modelled on a chosen analog
  market and its delta is read from that market. The UI labels this everywhere.
* **Aggregate first.** The departure-country/nationality join is structurally
  wrong (docs/METHODS.md 4.5): market rows are flagged ``supply-side-only`` and
  the aggregate views are the safe read. Occupancy is not computed -- the pack
  has no room-capacity data -- so winter impact is reported in guest-nights.
* **Baseline is held forward, not forecast.** Months after 2026-02 repeat the
  same calendar month one year earlier; no new airline schedule is invented.
"""

from __future__ import annotations

import dataclasses
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from pipeline import paths  # noqa: E402
from pipeline.calendar_events import COVERED_FROM, COVERED_TO  # noqa: E402
from pipeline.simulator import RouteEdit, Scenario, Simulator  # noqa: E402
from app.shared import baseline_frames, load_simulator  # noqa: E402

OBSERVED_TO = "2026-02"
LEVER_DEFAULTS = dict(
    seats_mult=1.0,
    lf_mult=1.0,
    p2p_mult=1.0,
    capture_mult=1.0,
    alos_mult=1.0,
    season_mult=1.0,
)

st.set_page_config(page_title="Flight → Hotel Demand Simulator", layout="wide")


# ---------------------------------------------------------------------------
# Scenario assembly helpers
# ---------------------------------------------------------------------------
def _text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in ("nan", "none", "<na>") else text


def _number(value):
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _route_weekly_seats(
    supply: pd.DataFrame, market: str, city: str, airline: str, start: str, end: str
) -> float | None:
    sel = supply[
        (supply["market"] == market)
        & (supply["city"] == city)
        & (supply["airline"] == airline)
    ]
    if sel.empty:
        return None
    sel = sel[(sel["ym"].astype(str) >= start) & (sel["ym"].astype(str) <= end)]
    if sel.empty:
        return None
    weekly = sel["Total_Seats"] * 7.0 / sel["ym"].dt.days_in_month
    return float(weekly.min())


def build_scenario(sim, edited_routes, edited_mults, global_levers):
    """Turn the two editor tables into a validated Scenario."""
    errors: list[str] = []
    notes: list[str] = []
    routes: list[RouteEdit] = []
    supply = sim.route_supply()

    for i, row in edited_routes.iterrows():
        market = _text(row.get("market"))
        if not market:
            continue
        delta = _number(row.get("seats_per_week_delta"))
        if delta is None or delta == 0:
            continue
        start = _text(row.get("start"))
        if not start:
            errors.append(f"Route row {i + 1} ({market}): a start month (YYYY-MM) is required.")
            continue
        end = _text(row.get("end")) or None
        city = _text(row.get("city")) or "New route"
        airline = _text(row.get("airline")) or "New carrier"
        label = _text(row.get("new_origin_label")) or None
        if delta < 0:
            limit = _route_weekly_seats(
                supply, market, city, airline, start, end or str(COVERED_TO)
            )
            if limit is None:
                notes.append(
                    f"Route row {i + 1}: {city} / {airline} was not found in the baseline, "
                    "so the reduction is applied at market level."
                )
            elif abs(delta) > limit:
                notes.append(
                    f"Route row {i + 1}: reduction clamped from {delta:+,.0f} to "
                    f"{-limit:+,.0f} seats/week to stay within the route's smallest "
                    "weekly seats in the window."
                )
                delta = -limit
        routes.append(
            RouteEdit(
                market=market,
                city=city,
                airline=airline,
                seats_per_week=delta,
                started_period=start,
                ended_period=end,
                label=label,
            )
        )

    seats_pairs: list[tuple[str, float]] = []
    lf_pairs: list[tuple[str, float]] = []
    for i, row in edited_mults.iterrows():
        market = _text(row.get("market"))
        if not market:
            continue
        seats = _number(row.get("seats_multiplier"))
        lf = _number(row.get("lf_multiplier"))
        if seats is not None and abs(seats - 1.0) > 1e-9:
            seats_pairs.append((market, seats))
        if lf is not None and abs(lf - 1.0) > 1e-9:
            lf_pairs.append((market, lf))

    scenario = Scenario(
        **global_levers,
        market_seats_mult=tuple(seats_pairs),
        market_lf_mult=tuple(lf_pairs),
        routes=tuple(routes),
    )
    try:
        sim.validate_scenario(scenario)
    except ValueError as exc:
        errors.append(str(exc))
    return scenario, errors, notes


def scenario_is_empty(scenario: Scenario) -> bool:
    defaults = all(abs(getattr(scenario, k) - 1.0) < 1e-12 for k in LEVER_DEFAULTS)
    return defaults and not scenario.market_seats_mult and not scenario.market_lf_mult and not scenario.routes


def one_change_scenarios(scenario: Scenario):
    """Yield (label, Scenario) for each single change, scored from baseline.

    One-at-a-time deltas are the brief's sensitivity question ("which factors
    move the number most") made interactive. They do not sum to the combined
    effect -- the model is non-linear by design -- which the UI states.
    """
    out = []
    for key in LEVER_DEFAULTS:
        value = getattr(scenario, key)
        if abs(value - 1.0) > 1e-12:
            out.append((f"{key} = {value:g} (all markets)", Scenario(**{key: value})))
    for market, mult in scenario.market_seats_mult:
        out.append(
            (f"{market}: seats × {mult:g}", Scenario(market_seats_mult=((market, mult),)))
        )
    for market, mult in scenario.market_lf_mult:
        out.append(
            (f"{market}: load factor × {mult:g}", Scenario(market_lf_mult=((market, mult),)))
        )
    for e in scenario.routes:
        origin = f"{e.label} → analog {e.market}" if e.label else e.market
        window = f"{e.started_period}..{e.ended_period or COVERED_TO}"
        out.append(
            (
                f"{origin}: {e.city} / {e.airline} {e.seats_per_week:+,.0f} seats/wk ({window})",
                Scenario(routes=(e,)),
            )
        )
    return out


def _table(df: pd.DataFrame, formats: dict) -> None:
    """Render a formatted, index-less dataframe at full width."""
    st.dataframe(
        df.style.format(formats, na_rep="—"), width="stretch", hide_index=True
    )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.title("Flight → Hotel Demand Simulator")
st.caption(
    "Test how route, frequency, seat, load-factor and market-mix changes move "
    "Abu Dhabi hotel demand — aggregate guest-nights from the fitted hybrid, "
    "with the parametric chain alongside for audit."
)

with st.spinner("Loading the model — the first run fits the pack (~1–2 minutes)…"):
    sim, model_source = load_simulator()

baseline_monthly, baseline_market, baseline_chain = baseline_frames(sim)
base_nights = float(baseline_monthly["GuestNightsHat"].sum())
base_arrivals = float(baseline_monthly["NewArrivalHat"].sum())
markets = sorted(sim.fa["market"].unique())

st.sidebar.header("Global levers")
st.sidebar.caption(
    "These apply to every market. They act before the model, so features and "
    "exposure respond non-linearly — that is the model working, not a bug."
)
seats_mult = st.sidebar.slider(
    "Seats multiplier", 0.8, 1.2, 1.0, 0.01,
    help="Scales scheduled seats in every market (frequency or gauge changes at scale).",
)
lf_mult = st.sidebar.slider(
    "Load-factor multiplier", 0.9, 1.1, 1.0, 0.01,
    help="Scales the share of seats actually filled.",
)
p2p_mult = st.sidebar.slider(
    "P2P-share multiplier", 0.9, 1.1, 1.0, 0.01,
    help="Scales point-to-point share, i.e. moves transfer/transit traffic.",
)
with st.sidebar.expander("Advanced dials"):
    capture_mult = st.slider(
        "Capture multiplier", 0.7, 1.3, 1.0, 0.01,
        help="Post-hoc scale on conversion from inbound visitors to hotel guests.",
    )
    alos_mult = st.slider(
        "ALOS multiplier", 0.9, 1.1, 1.0, 0.01,
        help="Post-hoc scale on average length of stay (guest-nights per arrival).",
    )
    season_mult = st.slider(
        "Seasonality multiplier", 0.9, 1.1, 1.0, 0.01,
        help="Post-hoc demand-level override on top of the learned seasonality.",
    )
st.sidebar.divider()
st.sidebar.caption(
    f"Horizon **{COVERED_FROM} → {COVERED_TO}**. Observed supply ends "
    f"{OBSERVED_TO}; later months hold forward the same calendar month one year "
    "earlier (a baseline, not a schedule forecast). Model: " + model_source + "."
)

global_levers = dict(
    seats_mult=seats_mult,
    lf_mult=lf_mult,
    p2p_mult=p2p_mult,
    capture_mult=capture_mult,
    alos_mult=alos_mult,
    season_mult=season_mult,
)

builder_tab, results_tab, insights_tab, validation_tab, library_tab = st.tabs(
    ["Scenario builder", "Results", "Route insights", "Validation & caveats", "Scenario library"]
)

# --- builder ----------------------------------------------------------------
with builder_tab:
    st.subheader("Route, frequency, seat and season changes")
    st.caption(
        "One row per change. **Seats/week Δ** is signed: positive opens or grows "
        "a route, negative trims one. Use the start/end months for winter-only "
        "or seasonal edits. For a country that has never flown here, pick the "
        "closest **model market** as its analog and name the origin in the last "
        "column — the estimate is then read from the analog's behaviour and "
        "labelled as such."
    )
    route_edits = st.data_editor(
        pd.DataFrame(
            {
                "market": pd.Series(dtype="object"),
                "city": pd.Series(dtype="object"),
                "airline": pd.Series(dtype="object"),
                "seats_per_week_delta": pd.Series(dtype="float64"),
                "start": pd.Series(dtype="object"),
                "end": pd.Series(dtype="object"),
                "new_origin_label": pd.Series(dtype="object"),
            }
        ),
        num_rows="dynamic",
        hide_index=True,
        key="route_edits",
        column_config={
            "market": st.column_config.SelectboxColumn(
                "Model market", options=markets, required=True,
                help="The market whose behaviour prices this change. For a new origin, the analog.",
            ),
            "city": st.column_config.TextColumn(
                "City / route", help="e.g. Mumbai — or a new city for an existing market.",
            ),
            "airline": st.column_config.TextColumn(
                "Airline", help="e.g. Etihad, or 'New carrier'.",
            ),
            "seats_per_week_delta": st.column_config.NumberColumn(
                "Seats/week Δ", format="%+d", step=100,
                help="≈ Δ flights/week × seats per flight. Negative trims supply.",
            ),
            "start": st.column_config.TextColumn(
                "Start (YYYY-MM)", help=f"Within {COVERED_FROM}..{COVERED_TO}.",
            ),
            "end": st.column_config.TextColumn(
                "End (YYYY-MM)", help="Inclusive. Blank runs to the horizon.",
            ),
            "new_origin_label": st.column_config.TextColumn(
                "New origin label",
                help="Only for a country with no flights yet: display name, e.g. KENYA.",
            ),
        },
    )

    with st.expander("Aviation calculators — turn flights and aircraft into seats/week"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Frequency change** — Δ flights/week × seats per flight")
            flights_delta = st.number_input(
                "Flights per week Δ", value=2, step=1, key="calc_flights"
            )
            seats_per_flight = st.number_input(
                "Seats per flight", value=320, step=10, key="calc_seats"
            )
            st.metric("Seats per week Δ", f"{flights_delta * seats_per_flight:+,.0f}")
        with c2:
            st.markdown("**Aircraft upsize** — flights/week × Δ seats per flight")
            weekly_flights = st.number_input(
                "Existing flights per week", value=14, step=1, key="calc_freq"
            )
            seat_delta = st.number_input(
                "Seats per flight Δ", value=40, step=5, key="calc_seatdelta"
            )
            st.metric("Seats per week Δ", f"{weekly_flights * seat_delta:+,.0f}")

    st.subheader("Market multipliers — season and mix")
    st.caption(
        "Per-market scaling on top of the global levers: raise one market's "
        "seats to shift the seasonal mix, or its load factor to test who fills "
        "better. A multiplier of 1.0 is a no-op."
    )
    market_mults = st.data_editor(
        pd.DataFrame(
            {
                "market": pd.Series(dtype="object"),
                "seats_multiplier": pd.Series(dtype="float64"),
                "lf_multiplier": pd.Series(dtype="float64"),
            }
        ),
        num_rows="dynamic",
        hide_index=True,
        key="market_mults",
        column_config={
            "market": st.column_config.SelectboxColumn(
                "Market", options=markets, required=True,
            ),
            "seats_multiplier": st.column_config.NumberColumn(
                "Seats ×", min_value=0.1, max_value=3.0, step=0.05, format="%.2f",
                help="e.g. 1.15 = +15% seats in this market across the horizon.",
            ),
            "lf_multiplier": st.column_config.NumberColumn(
                "Load factor ×", min_value=0.5, max_value=1.5, step=0.02, format="%.2f",
                help="e.g. 1.05 = this market's seats fill 5% better.",
            ),
        },
    )

    with st.expander("Recipes — the brief's five questions as rows"):
        st.markdown(
            """
| Question | How to express it |
| --- | --- |
| New route from a country that didn't fly here | Model market = closest analog; city = the new city; airline = carrier; **Seats/week Δ** = weekly flights × seats per flight; start/end = launch window; **New origin label** = the country name |
| Airline runs more (or fewer) flights per week | Market + existing city + airline; **Seats/week Δ** = ±Δ flights × seats per flight (a cut is negative) |
| Airlines add seats via larger aircraft | Same route row; **Seats/week Δ** = flights/week × Δ seats per flight |
| A higher/lower share of seats gets filled | Global **Load-factor multiplier**, or a per-market row in the multipliers table |
| The mix of countries shifts with the seasons | Per-market **Seats ×** row, or a route edit with start/end set to the season (e.g. 2026-11 → 2027-02) |
"""
        )

    scenario, build_errors, build_notes = build_scenario(
        sim, route_edits, market_mults, global_levers
    )
    for error in build_errors:
        st.error(error)
    for note in build_notes:
        st.warning(note)

    if not build_errors:
        if scenario_is_empty(scenario):
            st.info("No changes yet — add a row above and the Results tab updates instantly.")
        else:
            parts = []
            for key, value in LEVER_DEFAULTS.items():
                current = getattr(scenario, key)
                if abs(current - 1.0) > 1e-12:
                    parts.append(f"{key} = {current:g}")
            for market, mult in scenario.market_seats_mult:
                parts.append(f"{market} seats × {mult:g}")
            for market, mult in scenario.market_lf_mult:
                parts.append(f"{market} LF × {mult:g}")
            for e in scenario.routes:
                origin = f"{e.label} (analog {e.market})" if e.label else e.market
                parts.append(
                    f"{origin}: {e.city}/{e.airline} {e.seats_per_week:+,.0f} seats/wk "
                    f"{e.started_period}..{e.ended_period or COVERED_TO}"
                )
            st.success("Scenario: " + " · ".join(parts))

# --- results ----------------------------------------------------------------
with results_tab:
    if build_errors:
        st.error("Fix the scenario errors in the builder tab first.")
    else:
        with st.spinner("Scoring the scenario…"):
            scenario_monthly = sim.monthly(scenario)
            scenario_market = sim.market(scenario)
            scenario_chain = sim.chain(scenario)

        delta_nights = float(scenario_monthly["GuestNightsHat"].sum()) - base_nights
        delta_arrivals = float(scenario_monthly["NewArrivalHat"].sum()) - base_arrivals
        pct_nights = 100.0 * delta_nights / base_nights if base_nights else 0.0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Δ Guest-nights (horizon)", f"{delta_nights:+,.0f}")
        m2.metric("Δ New arrivals (horizon)", f"{delta_arrivals:+,.0f}")
        m3.metric("Δ Guest-nights vs baseline", f"{pct_nights:+.2f}%")
        m4.metric("Scenario guest-nights", f"{scenario_monthly['GuestNightsHat'].sum():,.0f}")

        if scenario_is_empty(scenario):
            st.info("This is the baseline itself — add changes in the builder to see a delta.")

        st.subheader("Monthly guest-nights: baseline vs scenario")
        monthly_cmp = (
            baseline_monthly[["ym_str", "GuestNightsHat"]]
            .rename(columns={"GuestNightsHat": "Baseline"})
            .merge(
                scenario_monthly[["ym_str", "GuestNightsHat"]].rename(
                    columns={"GuestNightsHat": "Scenario"}
                ),
                on="ym_str",
                how="outer",
            )
            .sort_values("ym_str")
        )
        fig = go.Figure()
        fig.add_scatter(
            x=monthly_cmp["ym_str"], y=monthly_cmp["Baseline"],
            name="Baseline", mode="lines", line=dict(color="#8a8a8a"),
        )
        fig.add_scatter(
            x=monthly_cmp["ym_str"], y=monthly_cmp["Scenario"],
            name="Scenario", mode="lines", line=dict(color="#1f77b4", width=3),
        )
        fig.add_vline(x=OBSERVED_TO, line_dash="dot", line_color="#bbbbbb")
        fig.add_annotation(
            x=OBSERVED_TO, y=1.02, yref="paper", showarrow=False,
            text="observed | held-forward baseline", font=dict(size=11, color="#777777"),
        )
        fig.update_layout(
            height=380, margin=dict(l=10, r=10, t=30, b=10),
            yaxis_title="Guest-nights", xaxis_title=None,
            legend=dict(orientation="h", yanchor="bottom", y=1.0),
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Guest-nights per month. Months after 2026-02 use the held-forward "
            "baseline; the model has no future schedule, so read those as "
            "scenario differences, not forecasts."
        )

        left, right = st.columns(2)
        with left:
            st.subheader("Seasonal shape (calendar month)")
            seasonal = (
                baseline_market.groupby("month")["GuestNightsHat"].sum().rename("Baseline")
                .to_frame()
                .join(scenario_market.groupby("month")["GuestNightsHat"].sum().rename("Scenario"))
                .reset_index()
            )
            month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            seasonal["label"] = seasonal["month"].map(lambda m: month_names[int(m) - 1])
            sfig = go.Figure()
            sfig.add_bar(x=seasonal["label"], y=seasonal["Baseline"], name="Baseline", marker_color="#8a8a8a")
            sfig.add_bar(x=seasonal["label"], y=seasonal["Scenario"], name="Scenario", marker_color="#1f77b4")
            sfig.update_layout(
                barmode="group", height=340, margin=dict(l=10, r=10, t=20, b=10),
                yaxis_title="Guest-nights",
            )
            st.plotly_chart(sfig, width="stretch")

        with right:
            st.subheader("Chain stages: baseline vs scenario")
            stage_cols = [
                ("Seats", "Total_Seats"),
                ("PAX", "PAX"),
                ("P2P", "P2P"),
                ("New arrivals", "NewArrivalHat"),
                ("Guest-nights", "GuestNightsHat"),
            ]
            stage_rows = []
            for name, col in stage_cols:
                b = float(baseline_chain[col].sum())
                s = float(scenario_chain[col].sum())
                stage_rows.append(
                    {
                        "Stage": name,
                        "Baseline": b,
                        "Scenario": s,
                        "Δ": s - b,
                        "Δ %": 100.0 * (s - b) / b if b else 0.0,
                    }
                )
            _table(
                pd.DataFrame(stage_rows),
                {"Baseline": "{:,.0f}", "Scenario": "{:,.0f}", "Δ": "{:+,.0f}", "Δ %": "{:+.2f}%"},
            )
            st.caption(
                "The parametric chain under the same scenario — every ratio "
                "auditable. The hybrid may diverge from it; that divergence is "
                "the learned market/season behaviour."
            )

        st.subheader("Contribution by change (one-at-a-time vs baseline)")
        changes = one_change_scenarios(scenario)
        if not changes:
            st.caption("No changes to decompose.")
        elif len(changes) > 12:
            st.caption(f"{len(changes)} changes — showing the largest 12.")
            changes = changes[:12]
        if changes:
            rows = []
            for label, single in changes:
                m = sim.monthly(single)
                rows.append(
                    {
                        "Change": label,
                        "Δ Guest-nights": float(m["GuestNightsHat"].sum()) - base_nights,
                        "Δ New arrivals": float(m["NewArrivalHat"].sum()) - base_arrivals,
                    }
                )
            contrib = pd.DataFrame(rows).sort_values(
                "Δ Guest-nights", key=lambda s: s.abs(), ascending=False
            )
            cfig = go.Figure(
                go.Bar(
                    x=contrib["Δ Guest-nights"],
                    y=contrib["Change"],
                    orientation="h",
                    marker_color=["#2ca02c" if v >= 0 else "#d62728" for v in contrib["Δ Guest-nights"]],
                )
            )
            cfig.update_layout(
                height=max(240, 34 * len(contrib) + 80),
                margin=dict(l=10, r=10, t=20, b=10),
                xaxis_title="Δ Guest-nights vs baseline",
            )
            st.plotly_chart(cfig, width="stretch")
            _table(contrib, {"Δ Guest-nights": "{:+,.0f}", "Δ New arrivals": "{:+,.0f}"})
            st.caption(
                "Each change scored alone against the baseline. Contributions "
                "are non-linear and need not sum to the combined result — that "
                "is the model's interactions showing."
            )

        st.subheader("Market breakdown (supply-side-only)")
        st.warning(
            "Market rows attribute demand by departure country, not guest "
            "nationality — the pack's join is structurally wrong (METHODS §4.5). "
            "Use them to compare scenarios, not as measured origin levels. "
            "A new-origin label is an analog estimate, not an observation."
        )
        base_m = (
            baseline_market.groupby("market")
            .agg(BaseNights=("GuestNightsHat", "sum"), BaseArrivals=("NewArrivalHat", "sum"), BaseSeats=("Total_Seats", "sum"))
        )
        scen_m = (
            scenario_market.groupby("market")
            .agg(ScenNights=("GuestNightsHat", "sum"), ScenArrivals=("NewArrivalHat", "sum"), ScenSeats=("Total_Seats", "sum"))
        )
        market_delta = base_m.join(scen_m, how="outer").fillna(0.0)
        market_delta["ΔNights"] = market_delta["ScenNights"] - market_delta["BaseNights"]
        market_delta["ΔArrivals"] = market_delta["ScenArrivals"] - market_delta["BaseArrivals"]
        market_delta["ΔSeats"] = market_delta["ScenSeats"] - market_delta["BaseSeats"]
        market_delta = market_delta.sort_values("ΔNights", key=lambda s: s.abs(), ascending=False)
        new_origin_map = {e.market: e.label for e in scenario.routes if e.label}
        market_delta = market_delta.reset_index()
        market_delta["market"] = market_delta["market"].map(
            lambda m: f"{m} ← {new_origin_map[m]} (analog)" if m in new_origin_map else m
        )
        _table(
            market_delta[
                ["market", "BaseNights", "ScenNights", "ΔNights", "ΔArrivals", "ΔSeats"]
            ],
            {
                "BaseNights": "{:,.0f}", "ScenNights": "{:,.0f}",
                "ΔNights": "{:+,.0f}", "ΔArrivals": "{:+,.0f}", "ΔSeats": "{:+,.0f}",
            },
        )

# --- insights ----------------------------------------------------------------
with insights_tab:
    st.subheader("Which markets bring the most hotel guests per available seat?")
    st.caption(
        "Measured on the matched training panel (2022-01..2025-07). The market "
        "key is departure country, so this ranks supply-side attribution — the "
        "right way to compare markets in this pack, not a claim about guest "
        "nationality (METHODS §4.5)."
    )
    panel = sim.panel
    by_market = (
        panel.groupby("market")
        .agg(
            Guests=("Guests", "sum"),
            NewArrivals=("NewArr", "sum"),
            P2P=("Total_P2P", "sum"),
            Seats=("Total_Seats", "sum"),
            Months=("ym_str", "nunique"),
        )
        .reset_index()
    )
    by_market["Guests per 1,000 seats"] = 1000.0 * by_market["Guests"] / by_market["Seats"]
    by_market["Arrivals per 1,000 seats"] = 1000.0 * by_market["NewArrivals"] / by_market["Seats"]
    by_market["Capture (arrivals/P2P)"] = by_market["NewArrivals"] / by_market["P2P"]
    by_market["ALOS (nights/arrival)"] = by_market["Guests"] / by_market["NewArrivals"]
    ranked = by_market.sort_values("Guests per 1,000 seats", ascending=False)

    top = ranked.head(15).iloc[::-1]
    ifig = go.Figure(
        go.Bar(
            x=top["Guests per 1,000 seats"],
            y=top["market"],
            orientation="h",
            marker_color="#1f77b4",
        )
    )
    ifig.update_layout(
        height=440, margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title="Hotel guests per 1,000 available seats",
    )
    st.plotly_chart(ifig, width="stretch")
    _table(
        ranked[
            [
                "market", "Guests", "NewArrivals", "Seats",
                "Guests per 1,000 seats", "Arrivals per 1,000 seats",
                "Capture (arrivals/P2P)", "ALOS (nights/arrival)",
            ]
        ],
        {
            "Guests": "{:,.0f}", "NewArrivals": "{:,.0f}", "Seats": "{:,.0f}",
            "Guests per 1,000 seats": "{:,.1f}",
            "Arrivals per 1,000 seats": "{:,.1f}",
            "Capture (arrivals/P2P)": "{:,.2f}",
            "ALOS (nights/arrival)": "{:,.2f}",
        },
    )
    st.caption(
        "Capture above 1.0 is the join defect in plain sight (five markets "
        "exceed it) — another reason to read market levels as supply-side."
    )

    st.subheader("Route explorer — current weekly supply")
    st.caption(
        "Mean weekly seats and flights per route over the last three observed "
        "months (2025-12..2026-02), for sizing scenario edits. Copy the market, "
        "city and airline into the builder."
    )
    routes = sim.route_supply()
    observed = routes[routes["basis"] == "observed"]
    recent_months = sorted(observed["ym"].unique())[-3:]
    recent = observed[observed["ym"].isin(recent_months)].copy()
    recent["weekly_seats"] = recent["Total_Seats"] * 7.0 / recent["ym"].dt.days_in_month
    route_stats = (
        recent.groupby(["market", "city", "airline"])
        .agg(
            weekly_seats=("weekly_seats", "mean"),
            weekly_flights=("flights_per_week", "mean"),
            monthly_seats=("Total_Seats", "mean"),
        )
        .reset_index()
        .sort_values("weekly_seats", ascending=False)
    )
    market_pick = st.selectbox("Market", options=markets, key="route_market")
    _table(
        route_stats[route_stats["market"] == market_pick],
        {
            "weekly_seats": "{:,.0f}",
            "weekly_flights": "{:,.1f}",
            "monthly_seats": "{:,.0f}",
        },
    )

# --- validation ---------------------------------------------------------------
with validation_tab:
    st.subheader("Held-out validation")
    st.markdown(
        "The hybrid is scored on the competition test window (2025-08..2026-02) "
        "against actual New Arrivals — a window the model never trained on. The "
        "replay below re-fits that window on demand and surfaces the pinned "
        "numbers live."
    )
    if st.button("Replay the T5 competition window (trains on demand, ≈40 s)"):
        with st.spinner("Training the competition-window ensemble…"):
            st.session_state["t5"] = sim.reproduce_competition_window()
    if "t5" in st.session_state:
        t5 = st.session_state["t5"]
        actual = float(t5["NewArr"].sum())
        predicted = float(t5["NewArrivalHat"].sum())
        v1, v2, v3 = st.columns(3)
        v1.metric("WMAPE (New Arrivals)", f"{t5.attrs['wm_na']:.2f}%")
        v2.metric("Aggregate bias", f"{t5.attrs['bias_pct']:+.2f}%")
        v3.metric("Predicted / actual", f"{predicted:,.0f} / {actual:,.0f}")
        st.caption(
            "The parametric chain scores 32.77% on the same window; the hybrid "
            "cuts that by ~12 points at near-zero aggregate bias."
        )
    else:
        st.caption("Run the replay to surface the pinned numbers live.")

    st.subheader("What this model can and cannot say")
    st.markdown(
        """
**Can say** — aggregate guest-nights and arrivals under a scenario, broken
down by month and season; how each lever moves the total (one-at-a-time
contributions); relative market comparisons on the supply side.

**Cannot say** — measured guest levels by origin country. The flight market is
the last departure country before AUH; the hotel market is guest nationality.
The join between them is structurally wrong (capture exceeds 1.0 in five
markets), so every market-level row is flagged `supply-side-only` and a
new-origin result is an analog estimate.

**Occupancy** is not computed: the pack carries no room-capacity or room-nights
data, so winter impact is reported as lost/gained guest-nights. Add hotel
capacity data to turn that into occupancy points.

**Baseline beyond 2026-02** holds forward the same calendar month one year
earlier — seasonality carries, no new schedule is invented. Read those months
as scenario differences, not forecasts.
"""
    )

# --- library ------------------------------------------------------------------
with library_tab:
    st.subheader("Save and compare scenarios")
    st.caption(
        "Scenarios are saved to `outputs/sessions/` as a parquet of monthly "
        "results plus a JSON spec. On a hosted instance the local filesystem may "
        "reset; download anything you want to keep."
    )
    default_name = "scenario-" + pd.Timestamp.now().strftime("%Y%m%d-%H%M")
    scenario_name = st.text_input("Scenario name", value=default_name, key="scenario_name")

    if build_errors:
        st.error("Fix the scenario errors in the builder tab before saving.")
    elif st.button("Save current scenario"):
        slug = re.sub(r"[^a-z0-9_-]+", "-", scenario_name.lower()).strip("-") or "scenario"
        paths.SESSIONS.mkdir(parents=True, exist_ok=True)
        monthly_save = (
            baseline_monthly[["ym_str", "GuestNightsHat"]]
            .rename(columns={"GuestNightsHat": "baseline_guest_nights"})
            .merge(
                scenario_monthly[["ym_str", "GuestNightsHat"]].rename(
                    columns={"GuestNightsHat": "scenario_guest_nights"}
                ),
                on="ym_str",
                how="outer",
            )
            .sort_values("ym_str")
        )
        monthly_save["delta_guest_nights"] = (
            monthly_save["scenario_guest_nights"] - monthly_save["baseline_guest_nights"]
        )
        monthly_save.to_parquet(paths.SESSIONS / f"{slug}.parquet", index=False)
        spec = dataclasses.asdict(scenario)
        summary = {
            "name": scenario_name,
            "saved_at": pd.Timestamp.now().isoformat(timespec="seconds"),
            "delta_guest_nights": delta_nights,
            "delta_new_arrivals": delta_arrivals,
            "pct_guest_nights": pct_nights,
        }
        (paths.SESSIONS / f"{slug}.json").write_text(
            json.dumps({"spec": spec, "summary": summary}, indent=2, default=str)
        )
        st.success(f"Saved `{slug}` to {paths.SESSIONS}.")

    saved = sorted(paths.SESSIONS.glob("*.parquet")) if paths.SESSIONS.exists() else []
    if not saved:
        st.caption("No saved scenarios yet.")
    else:
        rows = []
        for path in saved:
            spec_path = path.with_suffix(".json")
            try:
                summary = json.loads(spec_path.read_text())["summary"]
            except Exception:
                summary = {"name": path.stem, "saved_at": "", "delta_guest_nights": np.nan,
                           "delta_new_arrivals": np.nan, "pct_guest_nights": np.nan}
            rows.append(
                {
                    "scenario": summary.get("name", path.stem),
                    "saved_at": summary.get("saved_at", ""),
                    "Δ Guest-nights": summary.get("delta_guest_nights", np.nan),
                    "Δ New arrivals": summary.get("delta_new_arrivals", np.nan),
                    "Δ %": summary.get("pct_guest_nights", np.nan),
                    "file_path": path,
                }
            )
        listing = pd.DataFrame(rows).sort_values("saved_at", ascending=False)
        _table(
            listing[["scenario", "saved_at", "Δ Guest-nights", "Δ New arrivals", "Δ %"]],
            {"Δ Guest-nights": "{:+,.0f}", "Δ New arrivals": "{:+,.0f}", "Δ %": "{:+.2f}%"},
        )

        with st.expander("Download saved scenarios"):
            for row in listing.itertuples():
                cols = st.columns([3, 1])
                cols[0].write(f"**{row.scenario}** — {row.file_path.name}")
                cols[1].download_button(
                    "Download",
                    data=row.file_path.read_bytes(),
                    file_name=row.file_path.name,
                    mime="application/octet-stream",
                    key=f"dl_{row.file_path.stem}",
                )

        if len(saved) >= 2:
            st.subheader("Compare two saved scenarios")
            names = {path.stem: path for path in saved}
            pick1 = st.selectbox("Scenario A", options=list(names), key="cmp_a")
            pick2 = st.selectbox("Scenario B", options=list(names), index=min(1, len(names) - 1), key="cmp_b")
            if pick1 and pick2:
                a = pd.read_parquet(names[pick1])
                b = pd.read_parquet(names[pick2])
                cmp = a.merge(b, on="ym_str", suffixes=("_a", "_b")).sort_values("ym_str")
                cfig = go.Figure()
                cfig.add_scatter(x=cmp["ym_str"], y=cmp["delta_guest_nights_a"], name=pick1, mode="lines")
                cfig.add_scatter(x=cmp["ym_str"], y=cmp["delta_guest_nights_b"], name=pick2, mode="lines")
                cfig.add_hline(y=0, line_dash="dot", line_color="#bbbbbb")
                cfig.update_layout(
                    height=340, margin=dict(l=10, r=10, t=20, b=10),
                    yaxis_title="Δ Guest-nights vs baseline",
                    legend=dict(orientation="h", yanchor="bottom", y=1.0),
                )
                st.plotly_chart(cfig, width="stretch")
