"""Back-test evidence and the model's limits."""

import streamlit as st

from app import charts, core, ui

ui.header(
    "Validation",
    caption="Scored on months the model never trained on. Every number here is recomputed from "
    "the saved predictions, not typed in.",
)

frame = core.validation()
if frame is None:
    st.info("Back-test results were not shipped with this deployment.")
    if st.button("Replay the back-tests now (about a minute)", type="primary"):
        from app.build_model import validation_frame
        with st.spinner("Replaying…"):
            frame = validation_frame(core.get_sim())
            frame.to_parquet(core.VALIDATION_PATH, index=False)
            core.validation.clear()
    if frame is None:
        st.stop()

t5 = frame[frame["window"] == "T5"]
t4 = frame[frame["window"] == "T4"]


def wmape(actual, pred) -> float:
    return 100.0 * (actual - pred).abs().sum() / actual.sum()


hyb, chain = wmape(t5["NewArr"], t5["NewArrivalHat"]), wmape(t5["NewArr"], t5["ChainHat"])
bias = 100.0 * (t5["NewArrivalHat"].sum() / t5["NewArr"].sum() - 1.0)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Hybrid error (WMAPE)", f"{hyb:.2f}%", f"{hyb - chain:.1f} pts vs chain",
          delta_color="inverse", border=True)
k2.metric("Fixed-ratio chain error", f"{chain:.2f}%", border=True)
k3.metric("National total bias", f"{bias:+.2f}%", border=True,
          help=f"Predicted {t5['NewArrivalHat'].sum():,.0f} vs actual {t5['NewArr'].sum():,.0f} arrivals.")
k4.metric("Guest-nights error (earlier holdout)", f"{wmape(t4['Guests'], t4['GuestNightsHat']):.1f}%",
          border=True, help="Feb – Jul 2025 holdout, with the learned length of stay.")

st.subheader("New hotel arrivals, Aug 2025 – Feb 2026 (held out)")
monthly = t5.groupby("ym_str")[["NewArr", "NewArrivalHat", "ChainHat"]].sum().reset_index()
st.plotly_chart(charts.backtest_lines(monthly), width="stretch")

a, b = st.columns(2)
with a:
    st.plotly_chart(charts.backtest_scatter(t5, "NewArrivalHat", title="Hybrid · each market-month"),
                    width="stretch")
with b:
    st.plotly_chart(charts.backtest_scatter(t5, "ChainHat", title="Fixed-ratio chain · each market-month"),
                    width="stretch")
st.caption(
    "WMAPE is the absolute error summed over all market-months, divided by actual arrivals. "
    "The hybrid's points hug the diagonal more tightly, most visibly for small markets."
)

st.subheader("What this model can and cannot say")
c1, c2 = st.columns(2)
with c1, st.container(border=True):
    st.markdown(
        """
**Can say**
- National guest-nights and hotel arrivals under a scenario, by month and season
- How much each change contributes, and which levers matter most
- Relative comparisons between markets on the supply side
- An indicative occupancy change, under stated room assumptions
"""
    )
with c2, st.container(border=True):
    st.markdown(
        f"""
**Cannot say**
- Measured demand by guest nationality: flight data files passengers under the last departure
  country, so connecting traffic lands on hubs (five markets show more hotel arrivals than
  terminating passengers)
- Demand from a country never flown: it is an analog estimate
- Forecasts beyond {core.OBSERVED_TO}: the baseline repeats last year's schedule
- Occupancy from data: the pack has no room counts
"""
    )
