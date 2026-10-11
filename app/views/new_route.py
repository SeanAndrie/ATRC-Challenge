"""Q1: how many extra hotel nights would a new twice-weekly flight generate?"""

import streamlit as st

from app import charts, core, ui

ui.header(
    "New route",
    "How many extra hotel nights would a new twice-weekly flight from a given country generate?",
)
markets = core.markets()

with st.container(border=True):
    c1, c2, c3, c4 = st.columns([1.4, 1, 1.2, 1.2])
    with c1:
        kind = st.radio("Origin", ["Already flies here", "New country"], horizontal=True)
        if kind == "Already flies here":
            market = st.selectbox("Country", markets, index=markets.index("INDIA"))
            label = None
        else:
            label = st.text_input("New country", "Kenya").strip().upper() or "NEW ORIGIN"
            market = st.selectbox(
                "Priced like", markets, index=markets.index("SOUTH AFRICA") if "SOUTH AFRICA" in markets
                else markets.index("EGYPT"),
                help="The model only knows the 33 markets in the data. Pick the one whose "
                     "travellers most resemble the new country's.",
            )
    with c2:
        flights = st.number_input("Flights per week", 1, 28, 2)
    with c3:
        aircraft = st.selectbox("Aircraft", [*core.AIRCRAFT, "Custom"], index=2)
        seats = (st.number_input("Seats per flight", 50, 600, 300, step=10)
                 if aircraft == "Custom" else core.AIRCRAFT[aircraft])
    with c4:
        opts = ui.month_options()
        start = st.selectbox("First month", opts, index=opts.index("2026-11"))
        n_months = st.slider("Months operated", 3, 24, 12)
    with st.expander("Modelling option"):
        as_new = st.toggle(
            "Also count it as an extra route and airline for the market",
            help="The model's route and airline counts mostly reflect market size, so "
                 "bumping them for one service inflates the estimate several-fold. Off by "
                 "default: the estimate is the effect of the added seats.",
        )

sc, notes = core.new_route(market, flights, seats, start, n_months, label=label,
                           count_as_new_route=as_new)
s = core.summarize(sc)
origin = label.title() if label else market.title()
for n in notes:
    st.info(n, icon=":material/info:")

st.subheader(f"{flights}× weekly from {origin}, {ui.window_label(s.window)}")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Extra guest-nights", ui.signed(s.delta_nights), border=True)
k2.metric("Extra hotel arrivals", ui.signed(s.delta_arrivals), border=True)
k3.metric("Guest-nights per year", ui.signed(s.per_year), border=True)
k4.metric("Guest-nights per seat added",
          f"{s.delta_nights / s.stages['seats']:.2f}" if s.stages["seats"] else "—", border=True,
          help="Hotel nights generated per extra seat flown in.")

left, right = st.columns([3, 2])
with left:
    st.markdown("**Extra guest-nights by month**")
    st.plotly_chart(charts.delta_bars(s.monthly), width="stretch")
with right:
    st.markdown("**Where the added seats go**")
    st.plotly_chart(charts.stage_funnel(s.stages), width="stretch")
st.caption(
    f"The parametric chain (fixed ratios) gives {ui.signed(s.chain_delta_nights)} guest-nights for "
    f"the same change; the hybrid adjusts the conversion for {market.title()}'s seasonality and mix. "
    "Passengers and terminating share follow the market's observed load factor and transfer share."
)
ui.market_caveat()

# --- which origin is worth most ----------------------------------------------
st.divider()
st.subheader("Where should the next route come from?")
st.caption(
    f"The same service — {flights}× weekly, {seats} seats, {ui.window_label(s.window)} — "
    "scored from every market."
)
if st.toggle("Compare all origins (about 20 seconds the first time)", key="sweep"):
    with st.spinner("Scoring 33 origins…"):
        sweep = core.origin_sweep(flights, seats, start, n_months, as_new)
    flagged = set(core.market_yield().query("capture > 1")["market"])
    st.plotly_chart(
        charts.ranking_bar(sweep, "delta_nights", highlight=market, flagged=flagged,
                           x_title="Extra guest-nights", fmt="+,.0f"),
        width="stretch",
    )
    st.caption(
        "⚠ marks markets whose measured conversion exceeds one hotel arrival per terminating "
        "passenger — a sign that connecting traffic of other nationalities is filed under them, "
        "so their estimates run high."
    )
