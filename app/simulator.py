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
    """Load the saved pickle if present, else fit once and cache the artifact."""
    path = paths.MODELS / f"simulator_{window}.pkl"
    if path.exists():
        return Simulator.load(path, window=window)
    sim = Simulator(window=window).fit()
    sim.save(path)
    return sim


def _market_mix_editor() -> tuple:
    """Advanced per-market seat multipliers; returns the non-1.0 pairs."""
    sim = _SIM
    markets = sorted(sim.fa["market"].unique())
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
                    "× seats", min_value=0.0, max_value=3.0, step=0.05, format="%.2f"
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


def _route_editor() -> tuple:
    """Add/remove route edits held in session state; returns the tuple."""
    sim = _SIM
    routes = st.session_state.setdefault("routes", [])
    with st.sidebar.expander("Add a new route / extra frequency"):
        markets = sorted(sim.fa["market"].unique())
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


def _headline_block(scen_frame: pd.DataFrame, base_frame: pd.DataFrame) -> None:
    table = L.headline(scen_frame, base_frame)
    cols = st.columns(3)
    for col, row in zip(cols, table.itertuples()):
        col.metric(
            row.metric,
            f"{row.scenario:,.0f}",
            f"{row.change:+,.0f} ({row.change_pct:+.1f}%)",
        )
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


def _scenario_tab(scen_frame, base_frame, scen_chain) -> None:
    metric = st.selectbox(
        "Metric",
        list(L.METRIC_LABELS),
        format_func=lambda k: L.METRIC_LABELS[k],
        key="scen_metric",
    )
    _headline_block(scen_frame, base_frame)

    scen_monthly = L.monthly_from(scen_frame)
    base_monthly = L.monthly_from(base_frame)
    chain_monthly = L.monthly_from(scen_chain) if scen_chain is not None else None
    show_chain = st.checkbox(
        "Show the parametric chain alongside the hybrid", value=False
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


def _market_tab(scen_frame, base_frame) -> None:
    st.warning(C.MARKET_WARNING)
    metric = st.selectbox(
        "Metric",
        list(L.METRIC_LABELS),
        format_func=lambda k: L.METRIC_LABELS[k],
        key="mkt_metric",
    )
    scen_t = L.market_totals(scen_frame, metric)
    base_t = L.market_totals(base_frame, metric)
    st.plotly_chart(L.market_pareto(scen_t, base_t, metric=metric), width="stretch")

    seasonal = L.seasonal_from(scen_frame)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    metric_cols = ["P2P", "NewArrivalHat", "GuestNightsHat"]
    table = (
        seasonal.assign(month_name=[months[m - 1] for m in seasonal["month"]])[
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
        "Replayed from the fitted model, not asserted. The first run trains the "
        "competition window (~1 minute)."
    )
    if st.button("Run validation (T5 competition window + T4 holdout)"):
        with st.spinner("Training the competition window and scoring…"):
            t5 = _SIM.reproduce_competition_window()
            t4 = _SIM.reproduce_holdout()
        st.session_state["validation"] = {"t5": t5, "t4": t4}
    if "validation" in st.session_state:
        res = st.session_state["validation"]
        t5, t4 = res["t5"], res["t4"]
        c1, c2 = st.columns(2)
        c1.metric("New Arrivals WMAPE (T5)", f"{t5.attrs['wm_na']:.1f}%")
        c1.metric("Aggregate bias (T5)", f"{t5.attrs['bias_pct']:+.1f}%")
        c2.metric("Guest-nights WMAPE (T4 holdout)", f"{t4.attrs['wm_guest']:.1f}%")
        c2.metric("Seasonal-naive baseline (T5)", "32.8%")
        st.plotly_chart(
            L.validation_scatter(t5, "NewArr", "NewArrivalHat"),
            width="stretch",
        )
    else:
        st.info("Press the button to replay the pinned validation numbers.")


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

with st.sidebar:
    st.header("Scenario levers")
    st.caption("1.0 = the observed 2025 schedule. Move any of them.")
    levers = {}
    for key, (lo, hi, step) in L.lever_sliders().items():
        levers[key] = st.slider(
            L.LEVER_LABELS[key], lo, hi, 1.0, step, help=L.LEVER_HELP[key], key=f"lev_{key}"
        )
    market_mults = _market_mix_editor()
    routes = _route_editor()
    if st.button("Reset all levers"):
        for key in L.lever_sliders():
            st.session_state.pop(f"lev_{key}", None)
        st.session_state.pop("market_mix_editor", None)
        st.session_state.pop("routes", None)
        st.rerun()

scenario = L.build_scenario(levers, market_mults=market_mults, routes=routes)
st.session_state["scenario"] = scenario

with st.spinner("Scoring the scenario…"):
    scen_frame = _SIM.market(scenario)
    scen_chain = _SIM.chain(scenario)

base_frame = st.session_state["baseline"]

tab_scen, tab_mkt, tab_sens, tab_val, tab_about = st.tabs(
    ["Scenario", "By market & season", "Which levers matter", "Validation", "Assumptions & limits"]
)
with tab_scen:
    _scenario_tab(scen_frame, base_frame, scen_chain)
with tab_mkt:
    _market_tab(scen_frame, base_frame)
with tab_sens:
    _sensitivity_tab()
with tab_val:
    _validation_tab()
with tab_about:
    _assumptions_tab()
