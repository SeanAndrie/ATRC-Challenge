"""Landing page: what the model does, how good it is, and the three answers."""

import streamlit as st

from app import charts, core, ui

st.title("Flight → Hotel Demand Simulator")
st.markdown(
    "Most visitors reach Abu Dhabi by air, but only some of them stay in hotels. "
    "This tool converts **flight decisions into hotel guest-nights**: a transparent "
    "chain (seats → passengers → terminating passengers → hotel arrivals → "
    "guest-nights) with a LightGBM model that learns how each market converts, "
    "month by month."
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Held-out error (WMAPE)", f"{core.PINNED['t5_wmape']:.2f}%",
          f"{core.PINNED['t5_wmape'] - core.PINNED['chain_wmape']:.1f} pts vs fixed ratios",
          delta_color="inverse", border=True,
          help="New hotel arrivals, Aug 2025 – Feb 2026, months the model never saw.")
k2.metric("National total bias", f"{core.PINNED['t5_bias']:+.2f}%", border=True,
          help="Predicted vs actual national new arrivals on the held-out window.")
k3.metric("Source markets", f"{core.get_sim().panel['market'].nunique()}", border=True)
k4.metric("Scenario horizon", f"to {core.HORIZON_END}", border=True,
          help=f"Observed flight supply ends {core.OBSERVED_TO}; later months are held forward.")

# --- the chain, one year -----------------------------------------------------
st.subheader("A year of flights, followed to the hotel")
base = core.baseline()
year = core.months("2025-03", core.OBSERVED_TO)
m = base["monthly"][base["monthly"]["ym_str"].isin(year)]
seats, pax, p2p, arr, nights = (float(m[c].sum()) for c in
                                ("Total_Seats", "PAX", "P2P", "NewArrivalHat", "GuestNightsHat"))
left, right = st.columns([3, 1])
with left:
    st.plotly_chart(charts.chain_sankey(seats, pax, p2p, arr), width="stretch")
with right:
    st.metric("Seats flown in", ui.compact(seats))
    st.metric("Hotel arrivals", ui.compact(arr), f"{1000 * arr / seats:.0f} per 1,000 seats",
              delta_color="off")
    st.metric("Guest-nights", ui.compact(nights), f"{nights / arr:.1f} nights per arrival",
              delta_color="off")
st.caption(
    "Mar 2025 – Feb 2026, the 33 markets with direct flights, model estimates. "
    "Transfer and transit passengers never leave the airport; many terminating "
    "passengers are residents or visit friends and family."
)

# --- the three answers -------------------------------------------------------
st.subheader("The brief's three questions, answered live")
c1, c2, c3 = st.columns(3)

with c1, st.container(border=True):
    st.markdown("**A new twice-weekly flight**")
    sc, _ = core.new_route("INDIA", 2, 290, "2026-11", 12)
    s = core.summarize(sc)
    st.metric("Extra guest-nights in its first year", ui.signed(s.delta_nights),
              f"{ui.signed(s.delta_arrivals)} hotel arrivals", delta_color="off")
    st.caption("2 × 787-9 a week from India, Nov 2026 – Oct 2027.")
    st.page_link("views/new_route.py", label="Try any country →", icon=":material/add_road:")

with c2, st.container(border=True):
    st.markdown("**An airline cuts winter flights**")
    sc, _ = core.frequency_change("UNITED KINGDOM", -7, *core.WINTER, city="London", airline="Etihad")
    s = core.summarize(sc)
    rooms, _src = core.hotel_rooms()
    occ = core.occupancy_points(s.delta_nights, s.window, rooms, 1.7)
    st.metric("Guest-nights lost, Nov – Feb", ui.signed(s.delta_nights),
              f"{s.pct_of_window:+.1f}% of winter demand · ≈{occ:+.1f} occupancy pts",
              delta_color="off")
    st.caption("7 fewer London flights a week, winter 2026/27.")
    st.page_link("views/winter_cut.py", label="Cut any route →", icon=":material/ac_unit:")

with c3, st.container(border=True):
    st.markdown("**Most hotel guests per seat**")
    y = core.market_yield()
    valid = y[y["capture"] <= 1.0].head(3)
    for r in valid.itertuples():
        st.markdown(f"{r.market.title()} — **{r.arrivals_per_1000_seats:,.0f}** per 1,000 seats")
    st.caption("Markets with a consistent flight–hotel link, 2022–2025.")
    st.page_link("views/market_yield.py", label="See the full ranking →", icon=":material/leaderboard:")

with st.expander("How to read these numbers"):
    st.markdown(
        f"""
- **National totals are the reliable figure.** The model's national total was within
  {abs(core.PINNED['t5_bias']):.2f}% of actual on months it never saw.
- **Market splits are supply-side.** Flight data files a passenger under the last
  departure country, while hotel data uses nationality. Someone connecting through Doha
  counts as Qatar supply and, say, Indian demand.
- **A country with no flights yet** is priced on a similar market you choose (an analog),
  and labelled as such.
- **After {core.OBSERVED_TO}** the baseline repeats last year's schedule month by month,
  so results are differences between scenarios, not forecasts.
- **Occupancy** needs room data the pack doesn't have; the app converts guest-nights to
  occupancy points using the published room stock and a stated guests-per-room assumption.
"""
    )
