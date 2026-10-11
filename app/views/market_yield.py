"""Q3: which countries bring in the most hotel guests per available seat?"""

import streamlit as st

from app import charts, core, ui

ui.header(
    "Guests per seat",
    "Which countries bring in the most hotel guests per available seat?",
    "Measured, not modelled: actual hotel arrivals and guest-nights against seats flown, "
    "Jan 2022 – Jul 2025.",
)
y = core.market_yield()
flagged = set(y.loc[y["capture"] > 1.0, "market"])

c1, c2 = st.columns([1, 1])
metric = c1.radio(
    "Rank by", ["Hotel arrivals per 1,000 seats", "Guest-nights per 1,000 seats"], horizontal=True
)
hide = c2.toggle("Hide markets with an inconsistent flight–hotel link", value=False,
                 help="Markets with more hotel arrivals than terminating passengers — impossible "
                      "unless travellers of that nationality connect via other countries.")
col = "arrivals_per_1000_seats" if metric.startswith("Hotel") else "nights_per_1000_seats"
d = y[~y["market"].isin(flagged)] if hide else y
d = d.sort_values(col, ascending=False)

valid = y[~y["market"].isin(flagged)].sort_values(col, ascending=False)
best, worst = valid.iloc[0], valid.iloc[-1]
k1, k2, k3 = st.columns(3)
k1.metric("Highest yield (consistent link)", best["market"].title(),
          f"{best[col]:,.0f} per 1,000 seats", delta_color="off", border=True)
k2.metric("Lowest yield", worst["market"].title(), f"{worst[col]:,.0f} per 1,000 seats",
          delta_color="off", border=True)
total = y["arrivals" if col.startswith("arrivals") else "guest_nights"].sum()
k3.metric("All markets", f"{1000 * total / y['seats'].sum():,.0f}",
          "per 1,000 seats", delta_color="off", border=True)

tab_rank, tab_mix, tab_season, tab_table = st.tabs(
    ["Ranking", "Conversion × length of stay", "Seasonality", "Table"]
)
with tab_rank:
    st.plotly_chart(charts.ranking_bar(d, col, flagged=flagged, x_title=metric), width="stretch")
    st.caption(
        "⚠ More hotel arrivals than terminating passengers from that country: travellers of that "
        "nationality largely connect through hubs, so per-seat figures overstate the route itself. "
        "Hubs show the mirror image (low yield)."
    )
with tab_mix:
    st.markdown(
        "**Two ways a market earns its place:** more of its passengers check into hotels "
        "(right), or they stay longer (up). Bubble size is seats flown."
    )
    st.plotly_chart(charts.bubble(y), width="stretch")
with tab_season:
    st.markdown("**Hotel arrivals per 1,000 seats, by calendar month.** Darker is better.")
    pivot = core.yield_by_month().loc[d["market"]]
    st.plotly_chart(charts.heatmap(pivot), width="stretch")
    st.caption("A market whose yield peaks in winter is a stronger case for winter capacity.")
with tab_table:
    st.dataframe(
        d[["market", "seats", "arrivals", "guest_nights", "arrivals_per_1000_seats",
           "nights_per_1000_seats", "capture", "alos"]],
        hide_index=True, width="stretch",
        column_config={
            "market": "Market",
            "seats": st.column_config.NumberColumn("Seats", format="%,.0f"),
            "arrivals": st.column_config.NumberColumn("Hotel arrivals", format="%,.0f"),
            "guest_nights": st.column_config.NumberColumn("Guest-nights", format="%,.0f"),
            "arrivals_per_1000_seats": st.column_config.NumberColumn("Arrivals / 1k seats", format="%.0f"),
            "nights_per_1000_seats": st.column_config.NumberColumn("Nights / 1k seats", format="%.0f"),
            "capture": st.column_config.NumberColumn("Arrivals per terminating pax", format="%.2f"),
            "alos": st.column_config.NumberColumn("Avg stay (nights)", format="%.1f"),
        },
    )
ui.market_caveat()
