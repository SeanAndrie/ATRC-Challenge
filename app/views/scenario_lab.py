"""Free-form scenarios: global levers, route changes and seasonal mix shifts."""

import json
from dataclasses import asdict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import charts, core, ui

ui.header(
    "Scenario lab",
    caption="Combine any changes, see the national effect, which change drives it, and "
    "which levers matter most. Every change is scored against the no-change baseline.",
)
markets = core.markets()
opts = ui.month_options()

# --- inputs ------------------------------------------------------------------
with st.container(border=True):
    st.markdown("**Network-wide levers** · applied to every market and month")
    cols = st.columns(6)
    levers = {}
    for col, (key, (lo, hi, step)) in zip(cols, core.LEVER_BOUNDS.items()):
        levers[key] = col.slider(core.LEVER_LABELS[key], lo, hi, 1.0, step, format="%.2f×",
                                 key=f"lever_{key}")

    t_routes, t_mix = st.tabs(["Route changes", "Seasonal market mix"])
    with t_routes:
        st.caption("New service, more flights or cuts. Negative flights cut the market's busiest "
                   "route. Add a country name to price a country with no flights on an analog.")
        routes = st.data_editor(
            pd.DataFrame([{"market": "INDIA", "flights": 2, "seats": 290, "from": "2026-11",
                           "to": "2027-10", "new_country": None}]),
            num_rows="dynamic", hide_index=True, width="stretch", key="lab_routes",
            column_config={
                "market": st.column_config.SelectboxColumn("Market (or analog)", options=markets, required=True),
                "flights": st.column_config.NumberColumn("Δ flights / week", step=1, format="%+d"),
                "seats": st.column_config.NumberColumn("Seats / flight", min_value=50, max_value=600, step=10),
                "from": st.column_config.SelectboxColumn("From", options=opts, required=True),
                "to": st.column_config.SelectboxColumn("To", options=opts, required=True),
                "new_country": st.column_config.TextColumn("New country (optional)"),
            },
        )
    with t_mix:
        st.caption("Scale a market's seats over a season, e.g. +20% Russia and −10% India in winter.")
        mix = st.data_editor(
            pd.DataFrame({"market": pd.Series(dtype="object"), "mult": pd.Series(dtype="float"),
                          "from": pd.Series(dtype="object"), "to": pd.Series(dtype="object")}),
            num_rows="dynamic", hide_index=True, width="stretch", key="lab_mix",
            column_config={
                "market": st.column_config.SelectboxColumn("Market", options=markets, required=True),
                "mult": st.column_config.NumberColumn("Seats ×", min_value=0.1, max_value=3.0, step=0.05,
                                                      format="%.2f"),
                "from": st.column_config.SelectboxColumn("From", options=opts),
                "to": st.column_config.SelectboxColumn("To", options=opts),
            },
        )


# --- build -------------------------------------------------------------------
def build() -> tuple[core.Scenario, list[tuple[str, core.Scenario]], list[str]]:
    """The combined scenario, each change on its own, and any input problems."""
    parts: list[tuple[str, core.Scenario]] = []
    errors: list[str] = []
    for key, v in levers.items():
        if abs(v - 1.0) > 1e-9:
            parts.append((f"{core.LEVER_LABELS[key]} {v:.2f}×", core.Scenario(**{key: v})))
    for r in routes.dropna(subset=["market", "flights", "from", "to"]).to_dict("records"):
        start, end, flights = r["from"], r["to"], float(r["flights"])
        if flights == 0:
            continue
        if end < start:
            errors.append(f"{r['market']}: 'To' is before 'From'.")
            continue
        seats = r["seats"] if pd.notna(r["seats"]) else 300
        label = r["new_country"].strip().upper() if isinstance(r["new_country"], str) else ""
        try:
            if flights > 0:
                sc, _ = core.new_route(r["market"], flights, seats, start,
                                       len(core.months(start, end)), label=label or None)
            else:
                # Cuts use the busiest route's own aircraft size, not the typed one.
                sc, _ = core.frequency_change(r["market"], flights, start, end)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        name = f"{label.title()} (as {r['market'].title()})" if label else r["market"].title()
        parts.append((f"{name} {flights:+.0f} flights/wk, {start}–{end}", sc))
    for r in mix.dropna(subset=["market", "mult"]).to_dict("records"):
        start = r["from"] if isinstance(r["from"], str) else core.WINTER[0]
        end = r["to"] if isinstance(r["to"], str) else core.WINTER[1]
        if end < start or abs(r["mult"] - 1) < 1e-9:
            continue
        sc, _ = core.seasonal_mix({r["market"]: r["mult"]}, start, end)
        parts.append((f"{r['market'].title()} seats {r['mult']:.2f}×, {start}–{end}", sc))

    combined = core.Scenario(
        **levers, routes=tuple(e for _, sc in parts for e in sc.routes),
    )
    try:
        core.get_sim().validate_scenario(combined)
    except ValueError as exc:
        errors.append(str(exc))
    return combined, parts, errors


scenario, parts, errors = build()
for e in errors:
    st.error(e)
if errors:
    st.stop()
if not parts:
    st.info("Move a lever or add a change to see its effect.")
    st.stop()

window_mode = st.radio("Measure over", ["Months that change", "Next 12 months (Mar 2026 – Feb 2027)"],
                       horizontal=True)
window = core.FORWARD_YEAR if window_mode.startswith("Next") else None
s = core.summarize(scenario, window)

# --- results -----------------------------------------------------------------
st.subheader(f"Result · {ui.window_label(s.window)}")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Δ Guest-nights", ui.signed(s.delta_nights), f"{s.pct_of_window:+.2f}%", border=True)
k2.metric("Δ Hotel arrivals", ui.signed(s.delta_arrivals), border=True)
k3.metric("Per year", ui.signed(s.per_year), border=True)
k4.metric("Fixed-ratio chain says", ui.signed(s.chain_delta_nights), border=True,
          help="The transparent parametric chain under the same scenario, for audit.")

left, right = st.columns([3, 2])
with left:
    st.markdown("**National guest-nights**")
    st.plotly_chart(charts.compare_lines(s.monthly, observed_to=core.OBSERVED_TO), width="stretch")
with right:
    st.markdown("**What drives the result** · each change scored alone")
    contrib = pd.DataFrame(
        [{"change": name, "delta": core.summarize(sc, s.window).delta_nights} for name, sc in parts]
    )
    st.plotly_chart(charts.contribution(contrib), width="stretch")
    if len(parts) > 1:
        st.caption(f"Sum of parts {ui.signed(contrib['delta'].sum())} vs combined "
                   f"{ui.signed(s.delta_nights)}: the model is non-linear, so changes interact.")

with st.expander("Supply chain and market detail"):
    a, b = st.columns(2)
    a.dataframe(pd.DataFrame({"stage": [k.replace("_", " ") for k in s.stages],
                              "change": list(s.stages.values())}),
                hide_index=True, width="stretch",
                column_config={"change": st.column_config.NumberColumn(format="%+,.0f")})
    movers = s.movers.assign(market=s.movers["market"].map(lambda m: s.labels.get(m, m)))
    b.dataframe(movers, hide_index=True, width="stretch",
                column_config={"delta_nights": st.column_config.NumberColumn("Δ guest-nights",
                                                                             format="%+,.0f")})
    ui.market_caveat()

# --- sensitivity ---------------------------------------------------------------
st.divider()
st.subheader("Which levers move hotel demand most?")
sens = core.lever_sensitivity()
st.plotly_chart(charts.tornado(sens, core.LEVER_LABELS), width="stretch")
st.caption(
    "Each lever moved ±10% on its own over the next 12 months. Capture, stay and seasonality scale "
    "demand one-for-one. Seats, load factor and transfer share pass through the model, which "
    "responds less than proportionally: an extra seat does not bring a proportional extra guest."
)

# --- library -----------------------------------------------------------------
st.divider()
st.subheader("Scenario library")
lib = st.session_state.setdefault("library", {})
c1, c2 = st.columns([3, 1])
name = c1.text_input("Name this scenario", f"Scenario {len(lib) + 1}", label_visibility="collapsed")
if c2.button("Save scenario", width="stretch", type="primary"):
    full = core.summarize(scenario, (core.FIRST_FORWARD, core.HORIZON_END))
    lib[name] = {
        "summary": {"window": ui.window_label(s.window), "delta_guest_nights": s.delta_nights,
                    "delta_arrivals": s.delta_arrivals, "changes": [p[0] for p in parts]},
        "monthly": full.monthly[["ym_str", "delta"]],
        "spec": asdict(scenario),
    }
if lib:
    st.dataframe(
        pd.DataFrame([{"scenario": k, **{x: v["summary"][x] for x in ("window", "delta_guest_nights",
                                                                         "delta_arrivals")},
                       "changes": " · ".join(v["summary"]["changes"])} for k, v in lib.items()]),
        hide_index=True, width="stretch",
        column_config={"delta_guest_nights": st.column_config.NumberColumn("Δ guest-nights", format="%+,.0f"),
                       "delta_arrivals": st.column_config.NumberColumn("Δ arrivals", format="%+,.0f")},
    )
    fig = go.Figure()
    for i, (k, v) in enumerate(lib.items()):
        fig.add_scatter(x=v["monthly"]["ym_str"], y=v["monthly"]["delta"], name=k, mode="lines",
                        line=dict(width=2, color=[charts.BLUE, charts.ORANGE, "#1baf7a"][i % 3]))
    fig.add_hline(y=0, line_width=1, line_color=charts.AXIS)
    st.plotly_chart(charts._layout(fig, height=300, yaxis_title="Δ guest-nights vs baseline"),
                    width="stretch")
    st.download_button("Download library (JSON)", json.dumps(
        {k: {"summary": v["summary"], "spec": v["spec"]} for k, v in lib.items()}, indent=2, default=str),
        file_name="scenarios.json", mime="application/json")
    st.caption("Saved for this browser session; download to keep.")
