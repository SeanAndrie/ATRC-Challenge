"""Landing page: what the model does and how good it is."""

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

st.markdown(
    "**Test a change:** open the **Scenario lab** to add or cut flights, change aircraft size, "
    "load factor or the seasonal market mix, or **Ask the model** in plain language."
)
c1, c2 = st.columns(2)
c1.page_link("views/scenario_lab.py", label="Scenario lab →", icon=":material/tune:")
c2.page_link("views/assistant.py", label="Ask the model →", icon=":material/forum:")

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
