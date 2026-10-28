"""Interactive scenario simulator for DCT Abu Dhabi hotel demand.

Run with::

    .venv/bin/streamlit run app/simulator.py

The engine lives in ``pipeline.simulator`` (``Simulator`` / ``Scenario``); the
pure display logic in ``app.logic``; the prose in ``app.copy``. This module only
wires Streamlit widgets to those helpers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import copy as C  # noqa: E402
from app import logic as L  # noqa: E402
from pipeline import paths  # noqa: E402
from pipeline import sensitivity as S  # noqa: E402
from pipeline.simulator import LEVER_BOUNDS, Scenario, Simulator  # noqa: E402

st.set_page_config(page_title="Flight → Hotel Simulator", layout="wide")

# Lever name -> LEVER_BOUNDS key, for the sensitivity problem on a-priori bounds.
_BOUNDS_KEY = {
    "seats / frequency": "seats_mult",
    "load factor": "lf_mult",
    "P2P share (transfer/transit)": "p2p_mult",
    "capture rate": "capture_mult",
    "ALOS": "alos_mult",
    "season": "season_mult",
}


@st.cache_resource(show_spinner=False)
def get_simulator(window: str = "full") -> Simulator:
    """Load the saved pickle if present, else fit once and cache the artifact.

    A pickle is a convenience, not a guarantee: a different LightGBM/pandas
    version can make an old artifact unpicklable. Fall back to a clean fit rather
    than crashing on a reviewer's machine.
    """
    path = paths.MODELS / f"simulator_{window}.pkl"
    if path.exists():
        try:
            return Simulator.load(path, window=window)
        except Exception as exc:  # pragma: no cover - environment-dependent
            st.warning(
                f"Saved model at `{path}` could not be loaded ({exc}). "
                "Fitting from the competition pack instead."
            )
    sim = Simulator(window=window).fit()
    sim.save(path)
    return sim


def _market_mix_editor() -> tuple:
    """Advanced per-market seat multipliers; returns the non-1.0 pairs."""
    sim = _SIM
    markets = sorted(sim.panel["market"].unique())
    base = pd.DataFrame({"market": markets, "seat_multiplier": 1.0})
    with st.sidebar.expander("Advanced: per-market seat mix"):
        st.caption(
            "Scale a single market's supply on top of the global seats dial. "
            "Supply-side-only — see the warning in the Market tab."
        )
        edited = st.data_editor(
            base,
            key="market_mix_editor",
            hide_index=True,
            width="stretch",
            disabled=["market"],
            column_config={
                "seat_multiplier": st.column_config.NumberColumn(
                    "× seats", min_value=0.05, max_value=3.0, step=0.05, format="%.2f"
                )
            },
        )
        if st.button("Reset mix"):
            st.session_state.pop("market_mix_editor", None)
            st.rerun()
    pairs = tuple(
        (row.market, float(row.seat_multiplier))
        for row in edited.itertuples()
        if abs(float(row.seat_multiplier) - 1.0) > 1e-9
    )
    return pairs


def _seasonal_shape_editor() -> tuple:
    """Four season dials that build a Jan..Dec multiplier profile."""
    with st.sidebar.expander("Advanced: seasonal shape"):
        st.caption(
            "Change the shape of the year, not just its level. 1.0 = the "
            "observed seasonal profile. Applied to New Arrivals and Guest Nights."
        )
        winter = st.slider("Winter (Dec–Feb)", 0.7, 1.3, 1.0, 0.01, key="season_winter")
        spring = st.slider("Spring (Mar–May)", 0.7, 1.3, 1.0, 0.01, key="season_spring")
        summer = st.slider("Summer (Jun–Aug)", 0.7, 1.3, 1.0, 0.01, key="season_summer")
        autumn = st.slider("Autumn (Sep–Nov)", 0.7, 1.3, 1.0, 0.01, key="season_autumn")
    if all(abs(v - 1.0) < 1e-9 for v in (winter, spring, summer, autumn)):
        return ()
    return L.season_profile_from(winter, spring, summer, autumn)


def _seasonal_mix_editor() -> tuple:
    """A compact market × season mix editor for the top markets."""
    sim = _SIM
    top = (
        sim.panel.groupby("market")["Guests"].sum().nlargest(8).index.tolist()
    )
    base = pd.DataFrame(
        {
            "market": top,
            "Winter": 1.0,
            "Spring": 1.0,
            "Summer": 1.0,
            "Autumn": 1.0,
        }
    )
    with st.sidebar.expander("Advanced: seasonal market mix"):
        st.caption(
            "Scale one market's supply in a season. This is how the origin mix "
            "can shift with the year. Supply-side-only; see the market warning."
        )
        edited = st.data_editor(
            base,
            key="season_mix_editor",
            hide_index=True,
            width="stretch",
            disabled=["market"],
            column_config={
                "Winter": st.column_config.NumberColumn("Winter", min_value=0.0, max_value=2.0, step=0.05, format="%.2f"),
                "Spring": st.column_config.NumberColumn("Spring", min_value=0.0, max_value=2.0, step=0.05, format="%.2f"),
                "Summer": st.column_config.NumberColumn("Summer", min_value=0.0, max_value=2.0, step=0.05, format="%.2f"),
                "Autumn": st.column_config.NumberColumn("Autumn", min_value=0.0, max_value=2.0, step=0.05, format="%.2f"),
            },
        )
        if st.button("Reset seasonal mix"):
            st.session_state.pop("season_mix_editor", None)
            st.rerun()
    return L.market_month_pairs(edited.to_dict("records"))


def _suspend_markets() -> tuple:
    """Switch a market's supply off entirely (a structural zero)."""
    sim = _SIM
    markets = sorted(sim.panel["market"].unique())
    with st.sidebar.expander("Advanced: suspend a market"):
        st.caption(
            "A suspension is a structural zero: no seats means no arrivals. "
            "Use it for a market exit, not for a small frequency cut."
        )
        closed = st.multiselect(
            "Markets to suspend",
            markets,
            default=(),
            key="closed_markets",
            help="Only markets already in the model; new source countries are out of scope.",
        )
    return tuple(closed)


def _route_editor() -> tuple:
    """Add/remove route edits held in session state; returns the tuple."""
    sim = _SIM
    routes = st.session_state.setdefault("routes", [])
    with st.sidebar.expander("Add a new route / extra frequency"):
        markets = sorted(sim.panel["market"].unique())
        st.caption(
            "Only markets already flying to AUH can be edited — the model cannot "
            "add a brand-new source country."
        )
        market = st.selectbox("Market (departure country)", markets, key="route_market")
        city = st.text_input("Departure city", value="", key="route_city")
        airline = st.text_input("Airline", value="", key="route_airline")
        freq = st.number_input(
            "Flights per week", min_value=1.0, value=3.0, step=1.0, key="route_freq"
        )
        seats = st.number_input(
            "Seats per flight", min_value=1.0, value=180.0, step=10.0, key="route_seats"
        )
        start = st.selectbox("From month", L.month_options(), key="route_start")
        if st.button("Add route"):
            try:
                routes.append(
                    L.route_edit(market, city, airline, freq, seats, start)
                )
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
        for i, r in enumerate(routes):
            cols = st.columns([4, 1])
            cols[0].caption(
                f"{r.market} → {r.city or '—'} ({r.airline or '—'}): "
                f"{r.seats_per_week:,.0f} seats/wk from {r.started_period}"
            )
            if cols[1].button("✕", key=f"rm_route_{i}"):
                routes.pop(i)
                st.rerun()
    return tuple(routes)


def _route_change_editor() -> tuple:
    """Scale or close an existing market-city-airline route."""
    sim = _SIM
    changes = st.session_state.setdefault("route_changes", [])
    routes = sim.base_routes()
    with st.sidebar.expander("Change or close an existing route"):
        st.caption(
            "Scale frequency/seats on one city-airline route, or set 0 to close "
            "it. This is the brief's 'more or fewer flights per week' lever."
        )
        markets = sorted(routes["market"].dropna().unique())
        market = st.selectbox("Market", markets, key="rc_market")
        sub = routes[routes["market"] == market]
        cities = sorted(
            str(c) for c in sub["city"].dropna().unique() if str(c).strip()
        )
        city = st.selectbox("City (optional)", ["All cities", *cities], key="rc_city")
        if city == "All cities":
            airline_pool = sub
        else:
            airline_pool = sub[sub["city"].astype(str) == city]
        airlines = sorted(
            str(a) for a in airline_pool["airline"].dropna().unique() if str(a).strip()
        )
        airline = st.selectbox(
            "Airline (optional)", ["All airlines", *airlines], key="rc_airline"
        )
        mult = st.number_input(
            "Frequency / seat multiplier",
            min_value=0.0,
            max_value=2.0,
            value=1.0,
            step=0.05,
            key="rc_mult",
            help="0.0 closes the route; 1.0 leaves it unchanged.",
        )
        start = st.selectbox("From month", L.month_options(), key="rc_start")
        end = st.selectbox(
            "To month",
            ["No end (to 2026-02)", *L.month_options()],
            key="rc_end",
        )
        if st.button("Apply route change"):
            try:
                changes.append(
                    L.route_change(
                        market,
                        mult,
                        start,
                        None if end.startswith("No end") else end,
                        "" if city == "All cities" else city,
                        "" if airline == "All airlines" else airline,
                    )
                )
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
        for i, c in enumerate(changes):
            cols = st.columns([4, 1])
            scope = f"{c.city or 'all cities'} / {c.airline or 'all airlines'}"
            cols[0].caption(
                f"{c.market} · {scope}: ×{c.seats_mult:.2f} "
                f"from {c.started_period} to {c.ended_period or '2026-02'}"
            )
            if cols[1].button("✕", key=f"rm_route_change_{i}"):
                changes.pop(i)
                st.rerun()
    return tuple(changes)


def _headline_block(
    scen_frame: pd.DataFrame,
    base_frame: pd.DataFrame,
    scen_chain: pd.DataFrame | None = None,
    base_chain: pd.DataFrame | None = None,
) -> None:
    table = L.headline(scen_frame, base_frame)
    chain_row = None
    if scen_chain is not None and base_chain is not None:
        chain_base = float(base_chain["GuestNightsHat"].sum())
        chain_scen = float(scen_chain["GuestNightsHat"].sum())
        chain_pct = (
            100.0 * (chain_scen / chain_base - 1.0) if chain_base else float("nan")
        )
        chain_row = (chain_scen, chain_scen - chain_base, chain_pct)

    cols = st.columns(len(table) + (1 if chain_row is not None else 0))
    for col, row in zip(cols, table.itertuples()):
        col.metric(
            row.metric,
            f"{row.scenario:,.0f}",
            f"{row.change:+,.0f} ({row.change_pct:+.1f}%)",
            delta_color="off" if row.change_pct == 0 else "normal",
        )
    if chain_row is not None:
        chain_scen, chain_change, chain_pct = chain_row
        cols[-1].metric(
            "Guest nights — parametric chain",
            f"{chain_scen:,.0f}",
            f"{chain_change:+,.0f} ({chain_pct:+.1f}%)",
            delta_color="off" if chain_pct == 0 else "normal",
        )
    st.caption("Totals cover Jan 2022 to Feb 2026. Read the percent change.")
    with st.expander("Totals table"):
        st.dataframe(
            table.style.format(
                {
                    "baseline": "{:,.0f}",
                    "scenario": "{:,.0f}",
                    "change": "{:+,.0f}",
                    "change_pct": "{:+.1f}%",
                }
            ),
            width="stretch",
        )


def _decision_summary(
    scen_frame: pd.DataFrame,
    base_frame: pd.DataFrame,
    window: str,
    route_start: str | None,
) -> str:
    """A plain-language, scenario-specific answer for a DCT planner."""
    if scen_frame.empty or base_frame.empty:
        return (
            "**Decision summary.** The selected impact window contains no rows "
            "for this scenario. Choose a wider window or a route start inside "
            "the covered 2022-01 → 2026-02 period."
        )
    table = L.headline(scen_frame, base_frame)
    gn = table.set_index("metric").loc[L.METRIC_LABELS["GuestNightsHat"]]
    direction = "rise" if gn["change"] >= 0 else "fall"
    route_note = (
        f" The earliest route change starts in **{route_start}**, so the "
        "impact window is measured from there."
        if route_start and window == L.WINDOW_OPTIONS[1]
        else ""
    )
    return (
        f"**Decision summary — {window.lower()}.** Guest nights would "
        f"{direction} by **{gn['change']:+,.0f} ({gn['change_pct']:+.1f}%)** "
        f"against the observed schedule.{route_note} The national total is the "
        "validated number; the market split is a range, not a measurement. "
        "Treat a change smaller than the ~20.6% model error as noise, and use "
        "the market tab to see how wide the attribution range is before "
        "committing a budget."
    )


def _scenario_tab(scen_frame, base_frame, scen_chain, base_chain, impact_window: str) -> None:
    metric = st.selectbox(
        "Metric",
        list(L.METRIC_LABELS),
        format_func=lambda k: L.METRIC_LABELS[k],
        key="scen_metric",
    )
    _headline_block(scen_frame, base_frame, scen_chain, base_chain)
    st.caption(
        "Scenarios replay the 2022-01 → 2026-02 window; the model cannot project "
        "beyond Feb 2026. The headline above uses the selected impact window."
    )
    st.caption(C.IMPACT_CAPTION)
    st.caption(C.MODEL_ERROR_CAPTION)

    scen_monthly = L.monthly_from(scen_frame)
    base_monthly = L.monthly_from(base_frame)
    chain_monthly = L.monthly_from(scen_chain) if scen_chain is not None else None
    show_chain = st.checkbox(
        "Show the parametric chain alongside the hybrid", value=True
    )
    st.plotly_chart(
        L.monthly_trend(
            scen_monthly, base_monthly, chain_monthly if show_chain else None, metric
        ),
        width="stretch",
    )
    st.plotly_chart(
        L.seasonal_bars(L.seasonal_from(scen_frame), L.seasonal_from(base_frame), metric),
        width="stretch",
    )

    st.subheader("The conversion chain, step by step")
    st.caption(C.CHAIN_CAPTION)
    st.dataframe(
        L.chain_waterfall(_SIM, st.session_state["scenario"]).style.format(
            {"value": "{:,.0f}"}
        ),
        width="stretch",
    )


def _market_tab(scen_frame, base_frame, window: str) -> None:
    st.warning(C.MARKET_WARNING)
    st.caption(f"Impact window: **{window}**")
    metric = st.selectbox(
        "Metric",
        list(L.METRIC_LABELS),
        format_func=lambda k: L.METRIC_LABELS[k],
        key="mkt_metric",
    )
    scen_t = L.market_totals(scen_frame, metric)
    base_t = L.market_totals(base_frame, metric)
    st.plotly_chart(L.market_pareto(scen_t, base_t, metric=metric), width="stretch")

    st.subheader("Market × season (supply-side-only)")
    matrix = L.market_season_matrix(scen_frame, metric)
    st.plotly_chart(L.market_season_heatmap(matrix, metric), width="stretch")
    with st.expander("Market × month table"):
        st.dataframe(
            matrix.style.format("{:,.0f}"), width="stretch"
        )

    st.subheader("How wide is the attribution range?")
    st.caption(C.ATTRIBUTION_RANGE_CAPTION)
    hotel_mix = _SIM.panel.groupby("market")["Guests"].sum()
    ranges = L.attribution_range(scen_frame, hotel_mix)
    st.plotly_chart(L.attribution_range_chart(ranges), width="stretch")
    with st.expander("Attribution range table"):
        st.dataframe(
            ranges.style.format(
                {
                    "last_departure": "{:,.0f}",
                    "demand_mix": "{:,.0f}",
                    "supply_mix": "{:,.0f}",
                    "low": "{:,.0f}",
                    "high": "{:,.0f}",
                    "spread_pct": "{:+.1f}%",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    st.subheader("Guests per 1,000 scheduled seats")
    st.caption(C.EFFICIENCY_CAPTION)
    eff = L.efficiency_table(scen_frame, _SIM.params["capture"])
    st.dataframe(
        eff.style.format(
            {
                "Total_Seats": "{:,.0f}",
                "P2P": "{:,.0f}",
                "NewArrivalHat": "{:,.0f}",
                "GuestNightsHat": "{:,.0f}",
                "guests_per_1k_seats": "{:,.1f}",
                "nights_per_1k_seats": "{:,.1f}",
                "capture": "{:.3f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    seasonal = L.seasonal_from(scen_frame)
    metric_cols = ["P2P", "NewArrivalHat", "GuestNightsHat"]
    table = (
        seasonal.assign(month_name=[L.MONTHS[m - 1] for m in seasonal["month"]])[
            ["month_name", *metric_cols]
        ]
        .rename(columns=L.METRIC_LABELS)
    )
    st.subheader("Seasonal shape (aggregate — safe)")
    st.dataframe(
        table.style.format("{:,.0f}", subset=[L.METRIC_LABELS[c] for c in metric_cols]),
        width="stretch",
        hide_index=True,
    )

    with st.expander("Full market-month table (supply-side-only)"):
        cols = [
            "ym_str", "market", "month", "Total_Seats", "PAX", "P2P",
            "NewArrivalHat", "GuestNightsHat", "data_status",
        ]
        st.dataframe(scen_frame[cols], width="stretch", hide_index=True)


def _sensitivity_tab() -> None:
    st.subheader("Which factors move hotel guest numbers most")
    st.caption(
        "Two labelled views. **A-priori** = the room the sliders allow (feeds the "
        "simulator). **Empirical** = how much each factor actually moved over the "
        "fit window (descriptive). They answer different questions and are never "
        "merged."
    )
    if st.button("Compute sensitivity") or "sensitivity" in st.session_state:
        if "sensitivity" not in st.session_state:
            with st.spinner("Running the Sobol decomposition and lever spreads…"):
                panel, params = _SIM.panel, _SIM.params
                spread = S.lever_spread_table(panel, params)
                emp = S.empirical_lever_bounds(panel, params)
                apri = {
                    name: (LEVER_BOUNDS[key][0], LEVER_BOUNDS[key][1])
                    for name, key in _BOUNDS_KEY.items()
                }
                sobol_apri = S.sobol_sensitivity(panel, params, apri)
                sobol_emp = S.sobol_sensitivity(panel, params, emp)
            st.session_state["sensitivity"] = {
                "spread": spread,
                "apri": sobol_apri,
                "emp": sobol_emp,
            }
        res = st.session_state["sensitivity"]
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                L.sensitivity_bars(res["apri"], "ST"), width="stretch"
            )
            st.caption("A-priori Sobol total-order (ST) over the slider ranges.")
        with col2:
            st.plotly_chart(
                L.sensitivity_bars(res["emp"], "ST"), width="stretch"
            )
            st.caption("Empirical Sobol total-order (ST) over observed spreads.")
        st.markdown("**Empirical lever spreads over the fit window**")
        st.dataframe(res["spread"], width="stretch", hide_index=True)
        st.caption(
            "Capture dominates both views. The views disagree on the runner-up "
            "(a-priori: seats/frequency; empirical: P2P share), because seats "
            "only wobble ~±9% historically while capture swings far wider."
        )
    else:
        st.info("Sensitivity is computed on demand — press the button above.")


def _validation_tab() -> None:
    st.subheader("Validation with an honest error metric")
    st.caption(
        "The figures below are recomputed live from the fitted model. The two "
        "baselines (simple formula 32.8%, seasonal benchmark 36.0%) are provided "
        "for comparison and are backed by the test suite, not re-computed here. "
        "The first run trains and scores the final test window (~1 minute)."
    )
    if st.button("Run validation (final test window + last-year holdout)"):
        with st.spinner("Training and scoring the final test window…"):
            t5 = _SIM.reproduce_competition_window()
            t4 = _SIM.reproduce_holdout()
        st.session_state["validation"] = {"t5": t5, "t4": t4}
    if "validation" in st.session_state:
        res = st.session_state["validation"]
        t5, t4 = res["t5"], res["t4"]
        monthly = t5.groupby("ym_str").agg(
            NewArr=("NewArr", "sum"), NewArrivalHat=("NewArrivalHat", "sum")
        )
        seasonal = t5.groupby("month").agg(
            NewArr=("NewArr", "sum"), NewArrivalHat=("NewArrivalHat", "sum")
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("New Arrivals — weighted error (by market and month)", f"{t5.attrs['wm_na']:.1f}%")
        c1.metric("Overall bias", f"{t5.attrs['bias_pct']:+.1f}%")
        c2.metric(
            "New Arrivals — weighted error (monthly total)",
            f"{L.wmape(monthly['NewArr'], monthly['NewArrivalHat']):.1f}%",
        )
        c2.metric(
            "New Arrivals — weighted error (seasonal total)",
            f"{L.wmape(seasonal['NewArr'], seasonal['NewArrivalHat']):.1f}%",
        )
        c3.metric("Guest nights — weighted error (last-year holdout)", f"{t4.attrs['wm_guest']:.1f}%")
        c3.metric("Simple formula baseline", "32.8%")
        c3.metric("Seasonal benchmark", "36.0%")
        st.caption(
            "The by-market-and-month figure is the most conservative: it scores a "
            "composition that the pack misfiles. The monthly and seasonal totals "
            "are the decision-relevant metrics and are closer."
        )
        st.plotly_chart(
            L.validation_scatter(t5, "NewArr", "NewArrivalHat"),
            width="stretch",
        )
    else:
        st.info("Press the button to run the validation.")


def _assumptions_tab() -> None:
    st.markdown(C.ASSUMPTIONS)


# -- sidebar scenario controls ------------------------------------------------
st.title(C.TITLE)
st.caption(C.SUBTITLE)
st.info(C.DECISION_CALLOUT)

with st.spinner("Loading the fitted simulator (trains once, then cached)…"):
    _SIM = get_simulator("full")

if "baseline" not in st.session_state:
    st.session_state["baseline"] = _SIM.market(Scenario())
    st.session_state["baseline_chain"] = _SIM.chain(Scenario())

with st.sidebar:
    st.header("Scenario levers")
    st.caption("1.0 = the observed schedule (2022-01 to 2026-02). Move any of them.")
    impact_window = st.selectbox(
        "Measure impact over",
        L.WINDOW_OPTIONS,
        index=0,
        help="A new route is diluted by pre-route history; use the first 12 months.",
    )
    preset = st.selectbox(
        "Quick preset",
        ["Custom", "Seats +10%", "Load factor +5%", "Capture +10%", "Winter demand −20%"],
        index=0,
    )
    if st.button("Apply preset"):
        if preset == "Seats +10%":
            st.session_state["lev_seats_mult"] = 1.1
        elif preset == "Load factor +5%":
            st.session_state["lev_lf_mult"] = 1.05
        elif preset == "Capture +10%":
            st.session_state["lev_capture_mult"] = 1.1
        elif preset == "Winter demand −20%":
            st.session_state["season_winter"] = 0.8
        st.rerun()
    levers = {}
    for key, (lo, hi, step) in L.lever_sliders().items():
        levers[key] = st.slider(
            L.LEVER_LABELS[key], lo, hi, 1.0, step, help=L.LEVER_HELP[key], key=f"lev_{key}"
        )
    market_mults = _market_mix_editor()
    seasonal_shape = _seasonal_shape_editor()
    seasonal_mix = _seasonal_mix_editor()
    closed_markets = _suspend_markets()
    routes = _route_editor()
    route_changes = _route_change_editor()
    if st.button("Reset all levers"):
        for key in L.lever_sliders():
            st.session_state.pop(f"lev_{key}", None)
        for key in ("season_winter", "season_spring", "season_summer", "season_autumn"):
            st.session_state.pop(key, None)
        for key in (
            "market_mix_editor",
            "season_mix_editor",
            "closed_markets",
            "routes",
            "route_changes",
        ):
            st.session_state.pop(key, None)
        st.rerun()

scenario = L.build_scenario(
    levers,
    market_mults=market_mults,
    routes=routes,
    route_changes=route_changes,
    season_profile=seasonal_shape,
    market_month_mult=seasonal_mix,
    closed_markets=closed_markets,
)
st.session_state["scenario"] = scenario

with st.spinner("Scoring the scenario…"):
    try:
        scen_frame = _SIM.market(scenario)
        scen_chain = _SIM.chain(scenario)
    except ValueError as exc:
        st.error(f"This scenario is outside what the model supports. {exc}")
        st.stop()

base_frame = st.session_state["baseline"]
base_chain = st.session_state["baseline_chain"]
route_start = min((r.started_period for r in scenario.routes), default=None)

# The impact window is the unit of decision: a route that starts in 2025-09 is
# diluted by 44 months of pre-route history if we always report the full window.
scen_window = L.filter_window(scen_frame, impact_window, route_start)
base_window = L.filter_window(base_frame, impact_window, route_start)
scen_chain_window = L.filter_window(scen_chain, impact_window, route_start)
base_chain_window = L.filter_window(base_chain, impact_window, route_start)

st.info(_decision_summary(scen_window, base_window, impact_window, route_start))

tab_scen, tab_mkt, tab_sens, tab_val, tab_about = st.tabs(
    ["Scenario", "By market & season", "Which levers matter", "Validation", "Assumptions & limits"]
)
with tab_scen:
    _scenario_tab(
        scen_window, base_window, scen_chain_window, base_chain_window, impact_window
    )
with tab_mkt:
    _market_tab(scen_window, base_window, impact_window)
with tab_sens:
    _sensitivity_tab()
with tab_val:
    _validation_tab()
with tab_about:
    _assumptions_tab()
